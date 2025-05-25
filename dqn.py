import torch
from torch import nn
import torch.nn.functional as F
from move_encoder import MoveEmbedder
from custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):

    def __init__(self, hidden_dim=256, action_dim=64):
        super().__init__()
        self.n_moves    = ENCODING_CONSTANTS.MAX_MOVES*ENCODING_CONSTANTS.MAX_PKM_PER_TEAM
        self.embed_dim  = ENCODING_CONSTANTS.EMBEDDED_MOVE
        self.field_dim  = ENCODING_CONSTANTS.STATE-ENCODING_CONSTANTS.MOVE*ENCODING_CONSTANTS.MAX_MOVES*ENCODING_CONSTANTS.MAX_PKM_PER_TEAM

        # Initialize move embedder
        self.move_embedder = MoveEmbedder()

        # DQN head now takes aggregated field and embedded move features
        self.fc1 = nn.Linear(self.field_dim + self.n_moves * ENCODING_CONSTANTS.EMBEDDED_MOVE, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x is (batch, state features)
        field_state = x[:, :self.field_dim]
        
        # Handle move encoding
        # Convert each raw move into an embedded move
        # Maximum of 8 moves (2 of our own pokemon can be on the field)
        moves_raw   = x[:, self.field_dim:]
        moves_raw = moves_raw.reshape(-1, self.n_moves, ENCODING_CONSTANTS.MOVE)
        moves_emb = self.move_embedder(moves_raw)
        moves_emb = moves_emb.reshape(-1, self.n_moves * self.embed_dim)

        # concat with field_state -> (batch, len(field features) + len(embedded move features))
        fused = torch.cat([field_state, moves_emb], dim=1)
        
        # pass through DQN head
        h = F.relu(self.fc1(fused))
        return self.fc2(h)