"""Module containing n-armed bandit machine class.
...

Classes
-------
NArmedBanditMachine(num_actions)
    Class representing a n-armed gaussian bandit machine
"""

import numpy as np

class NArmedBanditMachine():
    """Class representing a n-armed gaussian bandit machine
    ...
    
    Attributes
    ----------
    num_actions : int
        Number of actions
    true_values : ndarray <Float>
        True means of action distributions
    optimal_action_idx : int
        Index of the optimal action (highest true mean)
    
    Methods
    -------
    init_action_values()
        Initializes the action distributions and finds the optimal action
    take_action(action_idx)
        Pulls a sample from the specified action distribution
    """

    def __init__(self, num_actions):
        """
        Parameters
        ----------
        num_actions : int
            Number of actions
        """

        self.num_actions = num_actions
        self.init_action_values()

    def init_action_values(self):
        """Initializes the action distributions and finds the optimal action"""

        self.true_values = np.random.normal(0, 2, self.num_actions)
        self.optimal_action_idx = np.argmax(self.true_values)
    
    def take_action(self, action_idx):
        """Pulls a sample from the specified action distribution

        Parameters
        ----------
        action_idx : int
            Index of the action to take

        Returns
        -------
        dict
            Action sample value and echoed action index
        """

        return {
            "action_idx" : action_idx,
            "action_sample_val" : np.random.normal(self.true_values[action_idx])
        }