"""Module containing the class to compare different agents' performance
...

Classes
-------
AgentComparision(agents, bandit)
    Class that facilitates agent bandit interaction
"""

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from agents import EpsilonGreedyAgent
from bandit_machine import NArmedBanditMachine

class AgentComparision():
    """Class that facilitates agent bandit interaction
    ...
    
    Attributes
    ----------
    agents : iter <Agent>
        Iterable of agents that are being compared
    bandit : NArmedBanditMachine
        Bandit machine
    
    Methods
    -------
    run_comparison(num_trials, num_decisions)
        Compares agent performance
    """

    def __init__(self, agents, bandit) -> None:
        """
        Parameters
        ----------
        agents : iter <Agent>
            Iterable of agents that are being compared
        bandit : NArmedBanditMachine
            Bandit machine
        """

        self.agents = agents
        self.bandit = bandit

    def run_comparison(self, num_trials=1000, num_decisions=1000):
        """Compares agent performance

        Parameters
        ----------
        num_trials : int
            Number of trials/experiments to run
        num_decisions : int
            Number of decisions that each agent should make per trial
        """

        average_rewards = np.zeros((len(self.agents), num_decisions))
        average_optimal_action_count = np.zeros_like(average_rewards)

        # Run through specified number of trials
        for trial in tqdm(range(num_trials)):
            # Initialize stats
            agent_rewards = np.zeros((len(self.agents), num_decisions))
            agent_optimal_decision_count = np.zeros_like(agent_rewards)

            # Initialize the bandit's true action values
            self.bandit.init_action_values()
            optimal_action_idx = self.bandit.optimal_action_idx

            # Initialize the agents' value estimates
            for agent in self.agents:
                agent.init_action_value_estimates(self.bandit)

            # Have agents make their decisions and record performace stats
            for i, agent in enumerate(self.agents):
                for run in range(num_decisions):
                    chosen_action_idx = agent.choose_action()
                    action_result = self.bandit.take_action(chosen_action_idx)
                    agent.update_value_estimates(action_result)
                    agent_rewards[i][run] = action_result["action_sample_val"]
                    agent_optimal_decision_count[i][run] += int(chosen_action_idx == optimal_action_idx)
            
            # Update iterative average
            average_rewards += (agent_rewards - average_rewards) / (trial+1)
            average_optimal_action_count += (agent_optimal_decision_count - average_optimal_action_count) / (trial+1)

        # Plot performances
        fig, axs = plt.subplots(2, 1)
        for i, agent in enumerate(self.agents):
            axs[0].plot(average_rewards[i], label=agent.description)
            axs[1].plot(average_optimal_action_count[i], label=agent.description)

        axs[1].set_xlabel('Steps')
        axs[1].set_ylabel('% Optimal Action')
        axs[0].set_ylabel('Average Reward')
        axs[0].legend(loc='lower right')
        axs[1].legend(loc='lower right')

        # plt.show()
        return fig, axs

if __name__ == '__main__':
    agents = [EpsilonGreedyAgent(0.1, "0.1 Epsilon Greedy"),\
              EpsilonGreedyAgent(0.01, "0.01 Epsilon Greedy"),
              EpsilonGreedyAgent(0.0, "Greedy")]
    bandit = NArmedBanditMachine(15)
    comparison = AgentComparision(agents, bandit)
    comparison.run_comparison(num_trials=2000, num_decisions=1000)