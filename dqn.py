import torch
from torch import nn
import torch.nn.functional as F
from custom_enviroment.move_encoder import MoveEmbedder
from custom_enviroment.custom_encodings import ENCODING_CONSTANTS

class DQN(nn.Module):

    def __init__(self, action_dim=64):
        super().__init__()
        
        # We have 2 sides (own and opponent side), each side can have a maximum of 2 active pkm, each pkm can have a maximum
        # of 4 moves -> 2*2*4 = 16 moves
        self.n_moves    = ENCODING_CONSTANTS.MAX_MOVES*ENCODING_CONSTANTS.MAX_PKM_PER_TEAM*ENCODING_CONSTANTS.MAX_TEAMS
        
        self.embed_dim  = ENCODING_CONSTANTS.EMBEDDED_MOVE

        # Our field features include things such as weather, trickroom and all active pokemon on the battlefield and
        # their attributes
        self.field_dim  = ENCODING_CONSTANTS.STATE-ENCODING_CONSTANTS.MOVE*self.n_moves

        # Initialize move embedder
        self.move_embedder = MoveEmbedder()

        # DQN head now takes aggregated field and embedded move features
        self.fc1 = nn.Linear(self.field_dim + self.n_moves * ENCODING_CONSTANTS.EMBEDDED_MOVE, 256)
        self.norm1 = nn.LayerNorm(256)
        self.fc2 = nn.Linear(256, 128)
        self.norm2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, action_dim)

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

        # pass through DQN head
        h = F.relu(self.norm1(self.fc1(fused)))
        h = F.relu(self.norm2(self.fc2(h)))
        return self.fc3(h)