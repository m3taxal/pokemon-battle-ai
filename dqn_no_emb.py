from torch import nn
import torch.nn.functional as F
from .move_encoder import MoveEmbedder
from .custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):

    def __init__(self, action_dim=64):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(ENCODING_CONSTANTS.STATE, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.output_layer = nn.Linear(128, 64)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.output_layer(x)
        return x