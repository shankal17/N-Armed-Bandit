import numpy as np

class NArmedBandit():
    def __init__(self, n):
        self.initialize_actions(n)

    def initialize_actions(self, n):
        self.true_values = np.random.uniform(0, 1, n)
        self.optimal_action_index = np.argmax(self.true_values)
    
    def take_action(self, action_idx):
        return np.random.normal(self.true_values[action_idx])
    
class Agent():
    def __init__(self, policy, bandit, inital_value_estimator):
        pass

class AgentComparision():
    def __init__(self) -> None:
        pass
