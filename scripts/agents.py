"""Module containing agent classes.
...

Classes
-------
Agent(description)
    Base class for all agents
EpsilonGreedyAgent(epsilon, description)
    Agent with epsilon-greedy action selection policy
"""

import numpy as np

class Agent():
    """Base class to represent a general agent
    ...
    
    Attributes
    ----------
    action_value_estimates : ndarray <Float>
        Number of actions
    description : string
        Description of agent function
    
    Methods
    -------
    init_action_value_estimates(bandit)
        Initializes the action value estimates and action counts
    choose_action()
        Chooses an action to execute
    update_value_estimates()
        Updates the action value estimates from experience
    """

    def __init__(self, description='General Agent'):
        """
        Parameters
        ----------
        description : string
            Description of agent function
        """

        self.action_value_estimates = None
        self.description = description
    
    def init_action_value_estimates(self, bandit):
        """Initializes the action value estimates and action counts

        Parameters
        ----------
        bandit : NArmedBanditMachine
            Bandit mechanism
        """
        
        self.action_value_estimates = np.zeros(bandit.num_actions)
        self.action_counts = np.zeros_like(self.action_value_estimates)

    def choose_action(self):
        """Chooses an action to execute"""

        raise NotImplementedError

    def update_value_estimates(self):
        """Updates the action value estimates from experience"""

        raise NotImplementedError

class EpsilonGreedyAgent(Agent):
    """Agent with epsilon greedy policy selection
    ...
    
    Attributes
    ----------
    epsilon : float
        Probability that a non-greedy action will be taken
    description : string
        Description of agent function
    
    Methods
    -------
    choose_action()
        Chooses an action to execute by epsilon-greedy policy
    update_value_estimates(action_result)
        Updates the action value estimates from experience by averaging
    """

    def __init__(self, epsilon, description='Epsilon Greedy'):
        """
        Parameters
        ----------
        epsilon : float
            Probability that a non-greedy action will be taken
        description : string
            Description of agent function
        """

        super().__init__(description)
        self.epsilon = epsilon

    def choose_action(self):
        """Chooses an action to execute by epsilon-greedy policy

        Returns
        -------
        int
            Index of the chosen action
        """

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
        """Updates the action value estimates from experience by averaging
        
        Parameters
        ----------
        action_result : dict
            Result of bandit action
        """

        action_idx = action_result["action_idx"]
        action_sample_val = action_result["action_sample_val"]
        self.action_counts[action_idx] += 1
        self.action_value_estimates[action_idx] += (action_sample_val - self.action_value_estimates[action_idx]) / self.action_counts[action_idx]

class SoftmaxAgent(Agent):
    """Agent with softmax policy selection
    ...
    
    Attributes
    ----------
    temp : float
        Temperature of softmax function
    description : string
        Description of agent function
    
    Methods
    -------
    choose_action()
        Chooses an action to execute by softmax policy
    update_value_estimates(action_result)
        Updates the action value estimates from experience by averaging
    """

    def __init__(self, temp, description='Softmax Agent'):
        """
        Parameters
        ----------
        temp : float
            Temperature of softmax function
        description : string
            Description of agent function
        """

        super().__init__(description)
        self.temp = temp

    def choose_action(self):
        """Chooses an action to execute by softmax policy

        Returns
        -------
        int
            Index of the chosen action
        """

        softmax_numerator = np.exp(self.action_value_estimates/self.temp)
        softmax_denominator = np.sum(softmax_numerator)
        softmax = softmax_numerator / softmax_denominator

        return np.random.choice(len(softmax_numerator), p=softmax)

    def update_value_estimates(self, action_result):
        """Updates the action value estimates from experience by averaging
        
        Parameters
        ----------
        action_result : dict
            Result of bandit action
        """

        action_idx = action_result["action_idx"]
        action_sample_val = action_result["action_sample_val"]
        self.action_counts[action_idx] += 1
        self.action_value_estimates[action_idx] += (action_sample_val - self.action_value_estimates[action_idx]) / self.action_counts[action_idx]
