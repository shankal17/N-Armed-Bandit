"""Module containing agent classes.
...

Classes
-------
Agent(description)
    Base class for all agents
EpsilonGreedyAgent(epsilon, initial_estimate, description)
    Agent with epsilon-greedy action selection policy
SoftmaxAgent(temp, initial_estimate, description):
    Agent with softmax policy selection
UCBAgent(c, initial_estimate, description)
    Agent with upper-confidence-bound policy selection
"""

import numpy as np

class Agent():
    """Base class to represent a general agent
    ...
    
    Attributes
    ----------
    initial_estimate : float
        Initial action-value estimate
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

    def __init__(self, initial_estimate=0, description='General Agent'):
        """
        Parameters
        ----------
        initial_estimate : float
            Initial action-value estimate
        description : string
            Description of agent function
        """
        self.initial_estimate = initial_estimate
        self.action_value_estimates = None
        self.description = description
    
    def init_action_value_estimates(self, bandit):
        """Initializes the action value estimates and action counts

        Parameters
        ----------
        bandit : NArmedBanditMachine
            Bandit mechanism
        """
        
        self.action_value_estimates = self.initial_estimate*np.ones(bandit.num_actions)
        self.action_counts = np.zeros_like(self.action_value_estimates, dtype=int)

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
    initial_estimate : float
        Initial action-value estimate
    description : string
        Description of agent function
    
    Methods
    -------
    choose_action()
        Chooses an action to execute by epsilon-greedy policy
    update_value_estimates(action_result)
        Updates the action value estimates from experience by averaging
    """

    def __init__(self, epsilon, initial_estimate=0, description='Epsilon Greedy'):
        """
        Parameters
        ----------
        epsilon : float
            Probability that a non-greedy action will be taken
        initial_estimate : float
            Initial action-value estimate
        description : string
            Description of agent function
        """

        super().__init__(initial_estimate, description)
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
    initial_estimate : float
        Initial action-value estimate
    description : string
        Description of agent function
    
    Methods
    -------
    choose_action()
        Chooses an action to execute by softmax policy
    update_value_estimates(action_result)
        Updates the action value estimates from experience by averaging
    """

    def __init__(self, temp, initial_estimate=0, description='Softmax Agent'):
        """
        Parameters
        ----------
        temp : float
            Temperature of softmax function
        initial_estimate : float
            Initial action-value estimate
        description : string
            Description of agent function
        """

        super().__init__(initial_estimate, description)
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

class UCBAgent(Agent):
    """Agent with upper-confidence-bound policy selection
    ...
    
    Attributes
    ----------
    c : float
        Exploration constant
    initial_estimate : float
        Initial action-value estimate
    description : string
        Description of agent function
    
    Methods
    -------
    choose_action()
        Chooses an action to execute by epsilon-greedy policy
    update_value_estimates(action_result)
        Updates the action value estimates from experience by averaging
    """

    def __init__(self, c, initial_estimate=0, description='UCB Agent'):
        """
        Parameters
        ----------
        c : float
            Exploration coefficient
        initial_estimate : float
            Initial action-value estimate
        description : string
            Description of agent function
        """

        super().__init__(initial_estimate, description)
        self.c = c

    def choose_action(self):
        """Chooses an action to execute by epsilon-greedy policy

        Returns
        -------
        int
            Index of the chosen action
        """

        if int(0) in self.action_counts:
            return np.argmin(np.abs(self.action_value_estimates))
        else:
            t = np.sum(self.action_counts)
            argmax_arg = self.c*np.sqrt(np.log(t) / self.action_counts)
            argmax_arg += self.action_value_estimates
            return np.argmax(argmax_arg)

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
