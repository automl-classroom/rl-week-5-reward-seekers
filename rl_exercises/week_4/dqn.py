"""
Deep Q-Learning implementation.
"""

from typing import Any, Dict, List, Tuple

import itertools
import os

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from rl_exercises.agent import AbstractAgent
from rl_exercises.week_4.buffers import ReplayBuffer

# from rl_exercises.week_4.networks import QNetwork
from rl_exercises.week_4.mod_networks import QNetwork


def set_seed(env: gym.Env, seed: int = 0) -> None:
    """
    Seed Python, NumPy, PyTorch and the Gym environment for reproducibility.

    Parameters
    ----------
    env : gym.Env
        The Gym environment to seed.
    seed : int
        Random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    # some spaces also support .seed()
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    if hasattr(env.observation_space, "seed"):
        env.observation_space.seed(seed)


class DQNAgent(AbstractAgent):
    """
    Deep Q-Learning agent with ε-greedy policy and target network.

    Derives from AbstractAgent by implementing:
      - predict_action
      - save / load
      - update_agent
    """

    def __init__(
        self,
        env: gym.Env,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_final: float = 0.01,
        epsilon_decay: int = 500,
        target_update_freq: int = 1000,
        seed: int = 0,
        hidden_dim: int = 64,
        depth: int = 2,
    ) -> None:
        """
        Initialize replay buffer, Q-networks, optimizer, and hyperparameters.

        Parameters
        ----------
        env : gym.Env
            The Gym environment.
        buffer_capacity : int
            Max experiences stored.
        batch_size : int
            Mini-batch size for updates.
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon_start : float
            Initial ε for exploration.
        epsilon_final : float
            Final ε.
        epsilon_decay : int
            Exponential decay parameter.
        target_update_freq : int
            How many updates between target-network syncs.
        seed : int
            RNG seed.
        """
        super().__init__(
            env,
            buffer_capacity,
            batch_size,
            lr,
            gamma,
            epsilon_start,
            epsilon_final,
            epsilon_decay,
            target_update_freq,
            seed,
        )
        self.env = env
        set_seed(env, seed)

        obs_dim = env.observation_space.shape[0]
        n_actions = env.action_space.n

        # main Q-network and frozen target
        self.q = QNetwork(obs_dim, n_actions, hidden_dim, depth)
        self.target_q = QNetwork(obs_dim, n_actions, hidden_dim, depth)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_capacity)

        # hyperparams
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_final = epsilon_final
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq

        self.total_steps = 0  # for ε decay and target sync

    def epsilon(self) -> float:
        """
        Compute current ε by exponential decay.

        Returns
        -------
        float
            Exploration rate.
        """
        # TODO: implement exponential‐decayin
        # ε = ε_final + (ε_start - ε_final) * exp(-total_steps / ε_decay)
        epsilon_ = self.epsilon_final + (
            self.epsilon_start - self.epsilon_final
        ) * np.exp(-self.total_steps / self.epsilon_decay)
        return epsilon_
        # Currently, it is constant and returns the starting value ε

    def predict_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[int, Dict]:
        """
        Choose action via ε-greedy (or purely greedy in eval mode).

        Parameters
        ----------
        state : np.ndarray
            Current observation.
        info : dict
            Gym info dict (unused here).
        evaluate : bool
            If True, always pick argmax(Q).

        Returns
        -------
        action : int
        info_out : dict
            Empty dict (compatible with interface).
        """
        if evaluate:
            # TODO: select purely greedy action from Q(s)
            with torch.no_grad():
                # pass state through Q-network
                qvals = self.q(torch.tensor(state, dtype=torch.float32).unsqueeze(0))  # noqa: F841

            action = torch.argmax(qvals, dim=1).item()  # get index of max Q-value
        else:
            if np.random.rand() < self.epsilon():
                # TODO: sample random action
                action = self.env.action_space.sample()
            else:
                # TODO: select purely greedy action from Q(s)
                action = torch.argmax(
                    self.q(torch.tensor(state, dtype=torch.float32).unsqueeze(0)),
                    dim=1,
                ).item()

        return action

    def save(self, path: str) -> None:
        """
        Save model & optimizer state to disk.

        Parameters
        ----------
        path : str
            File path.
        """
        torch.save(
            {
                "parameters": self.q.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        """
        Load model & optimizer state from disk.

        Parameters
        ----------
        path : str
            File path.
        """
        checkpoint = torch.load(path)
        self.q.load_state_dict(checkpoint["parameters"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

    def update_agent(
        self, training_batch: List[Tuple[Any, Any, float, Any, bool, Dict]]
    ) -> float:
        """
        Perform one gradient update on a batch of transitions.

        Parameters
        ----------
        training_batch : list of transitions
            Each is (state, action, reward, next_state, done, info).

        Returns
        -------
        loss_val : float
            MSE loss value.
        """
        # unpack
        states, actions, rewards, next_states, dones, _ = zip(*training_batch)  # noqa: F841
        s = torch.tensor(np.array(states), dtype=torch.float32)  # noqa: F841
        a = torch.tensor(np.array(actions), dtype=torch.int64).unsqueeze(1)  # noqa: F841
        r = torch.tensor(np.array(rewards), dtype=torch.float32)  # noqa: F841
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)  # noqa: F841
        mask = torch.tensor(np.array(dones), dtype=torch.float32)  # noqa: F841

        # # TODO: pass batched states through self.q and gather Q(s,a)
        pred = self.q(s).gather(1, a)  # .squeeze(1)  # shape (batch,)

        # TODO: compute TD target with frozen network
        with torch.no_grad():
            # TD target: r + γ * max Q(s', a') * (1 - done)
            # pass next states through target network
            target = r + self.gamma * self.target_q(s_next).max(dim=1)[0] * (1 - mask)
            target = target.unsqueeze(1)

        loss = nn.MSELoss()(pred, target)

        # gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # occasionally sync target network
        if self.total_steps % self.target_update_freq == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        self.total_steps += 1
        return float(loss.item())

    def train(self, num_frames: int, eval_interval: int = 1000) -> None:
        """
        Run a training loop for a fixed number of frames.

        Parameters
        ----------
        num_frames : int
            Total environment steps.
        eval_interval : int
            Every this many episodes, print average reward.
        """
        state, _ = self.env.reset()
        ep_reward = 0.0
        recent_rewards: List[float] = []
        x_frames: List[int] = []
        y_mean_rewards: List[float] = []

        for frame in range(1, num_frames + 1):
            action = self.predict_action(state)
            next_state, reward, done, truncated, _ = self.env.step(action)

            # store and ste
            self.buffer.add(state, action, reward, next_state, done or truncated, {})
            state = next_state
            ep_reward += reward

            # update if ready
            if len(self.buffer) >= self.batch_size:
                # TODO: sample a batch from replay buffer
                batch = self.buffer.sample(self.batch_size)
                _ = self.update_agent(batch)

            if done or truncated:
                state, _ = self.env.reset()
                recent_rewards.append(ep_reward)
                ep_reward = 0.0
                # logging
                # print(len(recent_rewards) % eval_interval)
                # print(frame)
                if len(recent_rewards) % 10 == 0:
                    # TODO: compute avg over last eval_interval episodes and print
                    avg = np.mean(recent_rewards[-eval_interval:])
                    x_frames.append(frame)
                    y_mean_rewards.append(avg)
                    print(
                        f"Frame {frame}, AvgReward(10): {avg:.2f}, ε={self.epsilon():.3f}"
                    )

        print("Training complete.")
        return x_frames, y_mean_rewards


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # 1) build env
    print("start")
    env = gym.make(cfg.env.name)
    set_seed(env, cfg.seed)
    print(f"test confi: {cfg.env.name}, {cfg.seed}")
    # 3) TODO: instantiate & train the agent
    depths = [2, 3]
    widths = [64, 128]
    buffer_sizes = [10000, 15000]
    batch_sizes = [32, 64]

    original_cwd = get_original_cwd()
    plots_dir = os.path.join(original_cwd, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    combinations = list(itertools.product(depths, widths, buffer_sizes, batch_sizes))

    for depth, width, buffer_capacity, batch_size in combinations:
        print(
            f"\nRunning config: depth={depth}, width={width}, buffer={buffer_capacity}, batch={batch_size}"
        )
        agent = DQNAgent(
            env=env,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            lr=cfg.agent.learning_rate,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            seed=cfg.seed,
            hidden_dim=width,
            depth=depth,
        )

        x_frames, y_mean_rewards = agent.train(
            num_frames=cfg.train.num_frames,
            eval_interval=cfg.train.eval_interval,
        )

        plot_title = f"D{depth}_W{width}_B{batch_size}_Buf{buffer_capacity}"
        plot_path = os.path.join(plots_dir, f"training_curve_{plot_title}.png")

        plt.figure()
        plt.plot(x_frames, y_mean_rewards, label="Mean Reward")
        plt.xlabel("Number of Frames")
        plt.ylabel("Mean Reward")
        plt.title(f"Training Curve\n{plot_title}")
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()


if __name__ == "__main__":
    main()
