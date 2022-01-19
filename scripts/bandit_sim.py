import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class NArmedBanditMachine():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.true_values = np.random.normal(0, 1, num_actions)
        self.optimal_action_index = np.argmax(self.true_values)
    
    def take_action(self, action_idx):

        return {
            "action_idx" : action_idx,
            "action_sample_val" : np.random.normal(self.true_values[action_idx])
        }
    
class Agent():
    def __init__(self):
        self.action_value_estimates = None
    
    def init_action_values(self, bandit):
        self.action_value_estimates = 0*np.ones(bandit.num_actions)

    def choose_action(self):
        raise NotImplementedError

    def update_value_estimates(self):
        raise NotImplementedError

class EpsilonGreedyAgent(Agent):
    def __init__(self, epsilon):
        super().__init__()
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
        pass

class AgentComparision():
    def __init__(self, agents, bandit) -> None:
        self.agents = agents
        self.bandit = bandit

    def run_comparison(self, num_actions=1000):

        # Initialize the agents' value estimates
        for agent in self.agents:
            agent.init_action_values(self.bandit)

        # Run through specified number of agent decisions
        for run in tqdm(range(num_actions)):
            for agent in self.agents:
                chosen_action_idx = agent.choose_action()
                action_result = self.bandit.take_action(chosen_action_idx)
                agent.update_value_estimates(action_result)

if __name__ == '__main__':
    AC = AgentComparision([EpsilonGreedyAgent(0.1)], NArmedBanditMachine(15))
    AC.run_comparison()
