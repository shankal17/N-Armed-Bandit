import numpy as np

class Agent():
    def __init__(self, description='General Agent'):
        self.action_value_estimates = None
        self.action_counts = None
    
    def init_action_values(self, bandit):
        self.action_value_estimates = np.zeros(bandit.num_actions)
        self.action_counts = np.zeros(bandit.num_actions)

    def choose_action(self):
        raise NotImplementedError

    def update_value_estimates(self):
        raise NotImplementedError

class EpsilonGreedyAgent(Agent):
    def __init__(self, epsilon, description='Epsilon Greedy'):
        super().__init__(description)
        self.epsilon = epsilon

    def choose_action(self):
        optimal_action_idx = np.argmax(self.action_value_estimates)
        if np.random.uniform() > self.epsilon:
            # Exploit current knowledge
            return optimal_action_idx
        else:
            # Explore other options
            non_optimal_choice_idxs = np.arange(len(self.action_value_estimates))
            mask = np.ones(len(non_optimal_choice_idxs), dtype=bool)
            mask[[optimal_action_idx]] = False
            non_optimal_choice_idxs = non_optimal_choice_idxs[mask]
            action_idx = np.random.choice(non_optimal_choice_idxs)
            return action_idx

    def update_value_estimates(self, action_result):
        action_idx = action_result["action_idx"]
        action_sample_val = action_result["action_sample_val"]
        self.action_counts[action_idx] += 1
        self.action_value_estimates[action_idx] += (action_sample_val - self.action_value_estimates[action_idx]) / self.action_counts[action_idx]
