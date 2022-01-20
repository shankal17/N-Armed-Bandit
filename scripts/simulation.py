import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import EpsilonGreedyAgent
from bandit_machine import NArmedBanditMachine

class AgentComparision():
    def __init__(self, agents, bandit) -> None:
        self.agents = agents
        self.bandit = bandit

    def run_comparison(self, num_trials=100, num_decisions=2000):
        
        # Run through specified number of trials
        for trial in tqdm(range(num_trials)):
            # Initialize the bandit's true action values
            self.bandit.init_action_values()
            # Initialize the agents' value estimates
            for agent in self.agents:
                agent.init_action_values(self.bandit)

            # Have agents make their decisions
            for agent in self.agents:
                for run in range(num_decisions):
                    chosen_action_idx = agent.choose_action()
                    action_result = self.bandit.take_action(chosen_action_idx)
                    agent.update_value_estimates(action_result)

if __name__ == '__main__':
    agents = [EpsilonGreedyAgent(0.1)]
    bandit = NArmedBanditMachine(5)
    comparison = AgentComparision(agents, bandit)
    comparison.run_comparison()