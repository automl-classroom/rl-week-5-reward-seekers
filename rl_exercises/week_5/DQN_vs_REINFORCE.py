'''
Trajectory Length: Longer episodes -> more stable REINFORCE updates
Architecture: Larger networks help both algorithms, but diminishing returns
Learning Rate: REINFORCE more sensitive to lr than DQN
Sample Complexity: DQN more sample efficient than REINFORCE
'''
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
from collections import defaultdict

from rl_exercises.week_4.dqn import DQNAgent
from policy_gradient import REINFORCEAgent

class EmpiricalAnalyzer:
    def __init__(self, env_name: str = "CartPole-v1", seed: int = 42):
        self.env_name = env_name
        self.seed = seed
        self.results = defaultdict(list)
    
    def run_experiment(self, agent_type: str, config: Dict, episodes: int = 500) -> List[float]:
        env = gym.make(self.env_name)
        
        if agent_type == "REINFORCE":
            agent = REINFORCEAgent(
                env=env,
                lr=config.get('lr', 1e-2),
                gamma=config.get('gamma', 0.99),
                seed=self.seed,
                hidden_size=config.get('hidden_size', 128)
            )
        else:  
            agent = DQNAgent(
                env=env,
                lr=config.get('lr', 1e-3),
                gamma=config.get('gamma', 0.99),
                seed=self.seed,
                hidden_dim=config.get('hidden_size', 64),
                epsilon_decay=config.get('epsilon_decay', 500)
            )
        
        returns = []
        for ep in range(episodes):
            state, _ = env.reset()
            episode_return = 0
            done = False
            
            if agent_type == "REINFORCE":
                batch = []
                while not done:
                    action, info = agent.predict_action(state)
                    next_state, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    batch.append((state, action, reward, next_state, done, info))
                    episode_return += reward
                    state = next_state
                
                if batch:
                    agent.update_agent(batch)
            
            # DQN
            else:
                while not done:
                    action = agent.predict_action(state)
                    next_state, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    
                    agent.buffer.add(state, action, reward, next_state, done, {})
                    episode_return += reward
                    state = next_state
                    
                    if len(agent.buffer) >= agent.batch_size:
                        batch = agent.buffer.sample(agent.batch_size)
                        agent.update_agent(batch)
            
            returns.append(episode_return)
        
        return returns

    def analyze_trajectory_length(self):
        print("trajectory length")
        
        max_steps_configs = [200, 500, 1000]
        
        for max_steps in max_steps_configs:
            env = gym.make(self.env_name)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
            
            agent = REINFORCEAgent(env=env, seed=self.seed)
            returns = []
            
            for ep in range(100):
                state, _ = env.reset()
                episode_return = 0
                done = False
                batch = []
                
                while not done:
                    action, info = agent.predict_action(state)
                    next_state, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    batch.append((state, action, reward, next_state, done, info))
                    episode_return += reward
                    state = next_state
                
                if batch:
                    agent.update_agent(batch)
                returns.append(episode_return)
            
            self.results[f'trajectory_length_{max_steps}'] = returns

    def analyze_architecture_impact(self):
        print("architecture impact")
        configs = [
            {'hidden_size': 64, 'lr': 1e-3, 'name': 'Small_LowLR'},
            {'hidden_size': 128, 'lr': 1e-2, 'name': 'Medium_HighLR'},
            {'hidden_size': 256, 'lr': 1e-2, 'name': 'Large_HighLR'},
        ]
        
        for config in configs:
            returns = self.run_experiment("REINFORCE", config, episodes=200)
            self.results[f"REINFORCE_{config['name']}"] = returns
            
            returns = self.run_experiment("DQN", config, episodes=200)
            self.results[f"DQN_{config['name']}"] = returns

    def compare_sample_complexity(self):
        print("sample complexity.")
        
        base_config = {'hidden_size': 128, 'lr': 1e-2}
        
        reinforce_returns = self.run_experiment("REINFORCE", base_config, episodes=300)
        dqn_returns = self.run_experiment("DQN", base_config, episodes=300)
        
        self.results['REINFORCE_sample_complexity'] = reinforce_returns
        self.results['DQN_sample_complexity'] = dqn_returns

    def plot_results(self):
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # trajectory len
        ax = axes[0, 0]
        for key in self.results:
            if 'trajectory_length' in key:
                returns = self.results[key]
                window_size = 10
                smoothed = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
                ax.plot(smoothed, label=key.replace('trajectory_length_', 'Max Steps: '))
        ax.set_title('Trajectory Length')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True)
        
        # architectur
        ax = axes[0, 1]
        for key in self.results:
            if 'REINFORCE_' in key and 'complexity' not in key:
                returns = self.results[key]
                window_size = 10
                smoothed = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
                ax.plot(smoothed, label=key.replace('REINFORCE_', ''))
        ax.set_title('REINFORCE: Architecture, Learning Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True)
        
        # DQN 
        ax = axes[1, 0]
        for key in self.results:
            if 'DQN_' in key and 'complexity' not in key:
                returns = self.results[key]
                window_size = 10
                smoothed = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
                ax.plot(smoothed, label=key.replace('DQN_', ''))
        ax.set_title('DQN: Architecture & Learning Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True)
        
        # Sample complexity 
        ax = axes[1, 1]
        if 'REINFORCE_sample_complexity' in self.results:
            returns = self.results['REINFORCE_sample_complexity']
            window_size = 20
            smoothed = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
            ax.plot(smoothed, label='REINFORCE', linewidth=2)
        
        if 'DQN_sample_complexity' in self.results:
            returns = self.results['DQN_sample_complexity']
            window_size = 20
            smoothed = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
            ax.plot(smoothed, label='DQN', linewidth=2)
        
        ax.set_title('Sample Complexity: REINFORCE vs DQN')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Return')
        ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('empirical_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    analyzer = EmpiricalAnalyzer(env_name="CartPole-v1")
    
    analyzer.analyze_trajectory_length()
    analyzer.analyze_architecture_impact()
    analyzer.compare_sample_complexity()
    
    analyzer.plot_results()

if __name__ == "__main__":
    main()
