import torch
from torch import nn
import torch.nn.functional as F
from MoveEncoder import MoveEmbedder
from custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):

    def __init__(self, action_dim=64):
        super().__init__()
        # Hidden layer dim
        self.hl1_dim = 256
        self.hl2_dim = 128

        self.fc1 = nn.Linear(ENCODING_CONSTANTS.STATE, self.hl1_dim)
        self.fc2 = nn.Linear(self.hl1_dim, self.hl2_dim)

        # Value stream
        self.fc_value = nn.Linear(self.hl2_dim, self.hl2_dim)
        self.value = nn.Linear(self.hl2_dim, 1)

        # Advantage stream
        self.fc_advantages = nn.Linear(self.hl2_dim, self.hl2_dim)
        self.advantages = nn.Linear(self.hl2_dim, action_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))

        # Value calc
        v = F.relu(self.fc_value(h))
        V = self.value(v)

        # Advantage calc
        a = F.relu(self.fc_advantages(h))
        A = self.advantages(a)

        # Calc Q
        Q = V + A - torch.mean(A, dim=1, keepdim=True)

        return Q