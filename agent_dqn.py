import torch
from torch import nn
import torch.nn.functional as F
from MoveEncoder import MoveEmbedder
from custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):
    def __init__(self, num_quantiles: int = 51, action_dim: int = 64):
        super().__init__()
        self.n_moves = ENCODING_CONSTANTS.MAX_MOVES * ENCODING_CONSTANTS.MAX_PKM_PER_TEAM + ENCODING_CONSTANTS.MAX_PKM_PER_TEAM
        self.embed_dim = ENCODING_CONSTANTS.EMBEDDED_MOVE
        self.field_dim = ENCODING_CONSTANTS.STATE - ENCODING_CONSTANTS.MOVE * self.n_moves

        self.num_quantiles = num_quantiles
        self.action_dim = action_dim

        # Move embedder
        self.move_embedder = MoveEmbedder()

        # Hidden layers
        self.fc1 = nn.Linear(self.field_dim + self.n_moves * self.embed_dim, 256)
        self.fc2 = nn.Linear(256, 192)
        self.fc3 = nn.Linear(192, 128)

        # Value stream (outputs quantiles)
        self.fc_value = nn.Linear(128, 128)
        self.value = nn.Linear(128, num_quantiles)  # Output: (batch, num_quantiles)

        # Advantage stream (outputs quantiles for each action)
        self.fc_advantages = nn.Linear(128, 128)
        self.advantages = nn.Linear(128, action_dim * num_quantiles)  # Output: (batch, action_dim * num_quantiles)

    def forward(self, x):
        batch_size = x.size(0)
        field_state = x[:, :self.field_dim]

        moves_raw = x[:, self.field_dim:]
        moves_raw = moves_raw.reshape(-1, self.n_moves, ENCODING_CONSTANTS.MOVE)
        moves_emb = self.move_embedder(moves_raw)
        moves_emb = moves_emb.reshape(batch_size, -1)

        fused = torch.cat([field_state, moves_emb], dim=1)
        h = F.relu(self.fc1(fused))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))

        # Value stream
        v = F.relu(self.fc_value(h))
        V = self.value(v).unsqueeze(1)  # shape: (batch, 1, num_quantiles)

        # Advantage stream
        a = F.relu(self.fc_advantages(h))
        A = self.advantages(a).view(batch_size, self.action_dim, self.num_quantiles)  # shape: (batch, action_dim, num_quantiles)

        # Combine streams: V + (A - mean_A)
        mean_A = A.mean(dim=1, keepdim=True)  # shape: (batch, 1, num_quantiles)
        Q = V + (A - mean_A)  # shape: (batch, action_dim, num_quantiles)

        return Q  # shape: (batch, action_dim, num_quantiles)
