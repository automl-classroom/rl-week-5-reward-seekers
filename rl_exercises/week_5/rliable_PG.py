'''import numpy as np
import matplotlib.pyplot as plt
from rliable import library as rly
from rliable import metrics
import gymnasium as gym
import itertools
from policy_gradient import REINFORCEAgent

def run(config, env_name="CartPole-v1", seeds=3):
    returns_all = []
    for seed in range(seeds):
        env = gym.make(env_name)
        agent = REINFORCEAgent(
                env=env,
                lr=config['lr'],
                gamma=config['gamma'],
                seed=seed,
                hidden_size=config['hidden_size']
            )
        eval_env = gym.make(env_name)
        episode_returns = []

        for _ in range(0, config['episodes'], 50):
            agent.train(50, eval_interval=float('inf'))
            ret, _ = agent.evaluate(eval_env, num_episodes=5)
            episode_returns.append(ret)

        returns_all.append(episode_returns)
        env.close()
        eval_env.close()

    return np.array(returns_all)

def hyper_sweep():
    sweep_space = {
        'lr': [1e-3],
        'gamma': [0.99],
        'hidden_size': [64],
        'episodes': [500]
    }

    best, best_score = None, -np.inf
    results = {}
    for combo in itertools.product(*sweep_space.values()):
        config = dict(zip(sweep_space, combo))
        print("Testing:", config)
        rets = run(config)
        results[str(config)] = rets
        score = np.mean(rets[:, -1])
        if score > best_score:
            best, best_score = config, score

    return best, results

def plot_results(results, env_name="CartPole-v1"):
    algorithms = {cfg[:30]: vals[:, -1] for cfg, vals in results.items()}
    agg_func = lambda x: np.array([
        metrics.aggregate_median(x),
        metrics.aggregate_iqm(x),
        metrics.aggregate_mean(x)
    ])
    agg_scores, agg_cis = rly.get_interval_estimates(algorithms, agg_func, reps=2000)

    fig, ax = plt.subplots()
    rly.plot_interval_estimates(
        agg_scores, agg_cis,
        metric_names=["Median", "IQM", "Mean"],
        xlabel="Performance", ax=ax)
    plt.title(f"Rliable Summary - {env_name}")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    for cfg, vals in results.items():
        mean = np.mean(vals, axis=0)
        std = np.std(vals, axis=0)
        x = np.arange(0, len(mean) * 50, 50)
        ax.plot(x, mean, label=cfg[:20])
        ax.fill_between(x, mean - std, mean + std, alpha=0.2)
    ax.set_xlabel("Episodes")
    ax.set_ylabel("Return")
    ax.set_title("Learning Curves")
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    best_config, results = hyper_sweep()
    plot_results(results)
    print("Best config:", best_config)
'''