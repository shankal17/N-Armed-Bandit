import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class NArmedBanditMachine():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.true_values = np.random.uniform(0, 1, num_actions)
        self.optimal_action_index = np.argmax(self.true_values)
    
    def take_action(self, action_idx):
        return np.random.normal(self.true_values[action_idx])
    
class Agent():
    def __init__(self, policy, inital_value_estimator):
        self.policy = policy
        self.inital_value_estimator = inital_value_estimator

class Policy():
    def __init__(self) -> None:
        pass

class EpsilonGreedyPolicy(Policy):
    def __init__(self) -> None:
        super(EpsilonGreedyPolicy, self).__init__()

class AgentComparision():
    def __init__(self, agents, bandit) -> None:
        self.agents = agents
        self.bandit = bandit

    def run_comparison(num_actions=2000):
        pass
