from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.optim as optim

class DQN(nn.Module):
    """the NN for DQN
    """
    def __init__(self, state_shape, hidden_shape, num_actions):
        super(DQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=state_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=num_actions, bias=True), # final
        )
        self.optimizer = optim.Adam(self.parameters(), lr=0.0002)
    
    def forward(self, state):
        return self.main(state.float())

    # def train(self):
    #     q_curr = self.forward(state)
    #     q_next = self.forward(next_state)
    #     delta = q_curr - (r + q_next)