import torch
from torch import nn
import torch.nn.functional as F
from move_encoder import MoveEmbedder
from custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):

    def __init__(self, action_dim=64):
        super().__init__()
        self.n_moves    = ENCODING_CONSTANTS.MAX_MOVES*ENCODING_CONSTANTS.MAX_PKM_PER_TEAM + ENCODING_CONSTANTS.MAX_PKM_PER_TEAM
        self.embed_dim  = ENCODING_CONSTANTS.EMBEDDED_MOVE
        self.field_dim  = ENCODING_CONSTANTS.STATE-ENCODING_CONSTANTS.MOVE*self.n_moves

        # Initialize move embedder
        self.move_embedder = MoveEmbedder()

        # Hidden layer dim
        self.hl1_dim = 256
        self.hl2_dim = 128

        # DQN now takes aggregated field and embedded move features
        self.fc1 = nn.Linear(self.field_dim + self.n_moves * ENCODING_CONSTANTS.EMBEDDED_MOVE, self.hl1_dim)
        self.fc2 = nn.Linear(self.hl1_dim, self.hl2_dim)

        # Value stream
        self.fc_value = nn.Linear(self.hl2_dim, self.hl2_dim)
        self.value = nn.Linear(self.hl2_dim, 1)

        # Advantage stream
        self.fc_advantages = nn.Linear(self.hl2_dim, self.hl2_dim)
        self.advantages = nn.Linear(self.hl2_dim, action_dim)

    def forward(self, x):
        # x is (batch, state features + move features)
        field_state = x[:, :self.field_dim]
        
        # Handle move encoding
        moves_raw   = x[:, self.field_dim:]
        moves_raw = moves_raw.reshape(-1, self.n_moves, ENCODING_CONSTANTS.MOVE)
        moves_emb = self.move_embedder(moves_raw)
        moves_emb = moves_emb.reshape(-1, self.n_moves * self.embed_dim)

        # concat with field_state -> (batch, len(field features) + len(embedded move features))
        fused = torch.cat([field_state, moves_emb], dim=1)

        h = F.relu(self.fc1(fused))
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