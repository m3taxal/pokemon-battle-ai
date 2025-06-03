import torch
import torch.nn as nn
import torch.nn.functional as F
from custom_enviroment.custom_encodings import ENCODING_CONSTANTS

class MoveEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(ENCODING_CONSTANTS.MOVE, ENCODING_CONSTANTS.EMBEDDED_MOVE)

    def forward(self, moves):
        """
        moves: tensor of shape (batch, N_moves, len(raw move featurs))
        returns: tensor of shape (batch, N_moves, embed_dim)
        """
        b, n, d = moves.shape
        h = moves.reshape(b * n, d)
        h = self.fc1(h)
        return h.reshape(b, n, -1)