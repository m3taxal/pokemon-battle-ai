import torch
from torch import nn
import torch.nn.functional as F
from move_encoder import MoveEmbedder
from custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):

    def __init__(self, action_dim=64):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(ENCODING_CONSTANTS.STATE, 256) 
        self.output_layer = nn.Linear(256, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.output_layer(x)
        return x