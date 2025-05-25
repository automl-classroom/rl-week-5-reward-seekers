import os

import gymnasium as gym
import hydra
import matplotlib.pyplot as plt
import numpy as np
import rliable.library as rlib
import rliable.metrics as rmetrics
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from omegaconf import DictConfig
from rl_exercises.week_4.dqn import DQNAgent, set_seed

# Configuration
SEEDS = [0]  # Use at least 5 seeds for robust evaluation
NUM_FRAMES = 20000
EVAL_INTERVAL = 1000
ENV_NAME = "CartPole-v1"


@hydra.main(config_path="../configs/agent/", config_name="dqn", version_base="1.1")
def main(cfg: DictConfig):
    # Initialize HydraConfig
    HydraConfig.instance()
    RESULTS_DIR = os.path.join(get_original_cwd(), "rliable_results")
    os.makedirs(RESULTS_DIR, exist_ok=True)

    def run_experiment(seed: int) -> list:
        env = gym.make(ENV_NAME)
        set_seed(env, seed)
        agent = DQNAgent(
            env=env,
            buffer_capacity=cfg.agent.buffer_capacity,
            batch_size=cfg.agent.batch_size,
            lr=cfg.agent.learning_rate,
            gamma=cfg.agent.gamma,
            epsilon_start=cfg.agent.epsilon_start,
            epsilon_final=cfg.agent.epsilon_final,
            epsilon_decay=cfg.agent.epsilon_decay,
            target_update_freq=cfg.agent.target_update_freq,
            seed=seed,  # Pass the seed to the agent
            hidden_dim=cfg.agent.hidden_dim,
            depth=cfg.agent.depth,
        )
        episode_rewards = agent.train(
            num_frames=NUM_FRAMES, eval_interval=EVAL_INTERVAL
        )
        return episode_rewards

    def aggregate_runs() -> dict:
        all_runs = {}
        for seed in SEEDS:
            print(f"Running seed {seed}")
            rewards = run_experiment(seed)
            all_runs[f"seed_{seed}"] = np.array(rewards)
        return all_runs

    def plot_rliable_metrics(scores: dict):
        # Align data by padding with NaNs
        max_len = max(len(v) for v in scores.values())
        padded_scores = []
        for k, v in scores.items():
            padded_scores.append(
                np.pad(v, (0, max_len - len(v)), constant_values=np.nan)
            )

        score_matrix = np.array(padded_scores)

        # RLiable expects a 2D array [num_runs, num_episodes]
        # We need to ensure that score_matrix has this shape.
        if score_matrix.ndim != 2:
            raise ValueError(
                "Input to rmetrics.compute_interval_estimates must be a 2D array [num_runs, num_episodes]"
            )

        # Metrics: IQM, Mean, Median
        aggregated_data, ci = rlib.compute_interval_estimates(score_matrix, reps=1000)

        # Plotting
        rmetrics.plot_score_distribution(scores, list(scores.keys()))
        plt.title("Score Distribution Across Seeds")
        plt.savefig(os.path.join(RESULTS_DIR, "score_distribution.png"))

        rmetrics.plot_performance_profiles(scores, list(scores.keys()))
        plt.title("Performance Profile")
        plt.savefig(os.path.join(RESULTS_DIR, "performance_profile.png"))

        print("Aggregate Scores (DQN):")
        print(f"IQM: {aggregated_data[0]:.2f} ± {ci[0]:.2f}")
        print(f"Mean: {aggregated_data[1]:.2f} ± {ci[1]:.2f}")
        print(f"Median: {aggregated_data[2]:.2f} ± {ci[2]:.2f}")

    scores = aggregate_runs()
    plot_rliable_metrics(scores)


if __name__ == "__main__":
    main()
