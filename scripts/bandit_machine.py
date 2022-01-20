import numpy as np

class NArmedBanditMachine():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.init_action_values()

    def init_action_values(self):
        self.true_values = np.random.normal(0, 1, self.num_actions)
        self.optimal_action_idx = np.argmax(self.true_values)
    
    def take_action(self, action_idx):

        return {
            "action_idx" : action_idx,
            "action_sample_val" : np.random.normal(self.true_values[action_idx])
        }