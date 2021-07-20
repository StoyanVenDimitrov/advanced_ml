from numpy.lib.function_base import select
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class FlatDQN(nn.Module):
    """the NN for DQN
    """
    def __init__(self, state_shape, hidden_shape, num_actions):
        super(FlatDQN, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(in_features=state_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=hidden_shape, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=hidden_shape, out_features=num_actions, bias=True), # final
        )
    
    def forward(self, state):
        return self.main(state)

class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 2)
        self.fc1 = nn.Linear(16 * 1 * 1, 84)
        self.fc2 = nn.Linear(84, n_actions)

    def forward(self, x):
        x = x.permute(0,3, 1, 2)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)                     # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

