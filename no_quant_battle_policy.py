import numpy as np
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import State, BattleCommand, TeamView
from typing import Optional
import gymnasium as gym
import torch
from custom_encodings import ENCODING_CONSTANTS, encode_state
from dqn_no_qr import DQN

class NoQuantBattlePolicy(BattlePolicy):
    def __init__(self, version):
        self.e = np.zeros(ENCODING_CONSTANTS.STATE)
        
        self.action_space = gym.spaces.MultiDiscrete([4, 2] * 2, start=[0, 0] * 2)
        self.agent = DQN()

        # Load agent
        self.agent.load_state_dict(torch.load("/home/aytur/Projects/pokemon-vgc-engine/tutorial/DeepWolfeSubmission/models/pokemon_battle"+str(version)+".pt"))
        self.agent.eval()

    def decision(self,
                 state: State,
                 opp_view: Optional[TeamView] = None) -> list[BattleCommand]:
        cmds: list[BattleCommand] = []

        # Reset observation
        self.e = np.zeros(ENCODING_CONSTANTS.STATE)
        
        # Encode state
        encode_state(self.e, state)

        # Feed state observation as tensor to DQN and get action
        self.e = torch.tensor(self.e, dtype=torch.float)
        action = self.agent(self.e.unsqueeze(dim=0)).squeeze().argmax()
        action = self.index_to_action(action.item(), self.action_space)

        cmds += action

        return cmds
    
    def index_to_action(self, index: int, action_space: gym.spaces.MultiDiscrete):
            """
            Convert flat index to grouped MultiDiscrete action as tuples.

            Returns List[Tuple[int, int]]: Grouped action, e.g. [(0, 1), (2, 3)]
            """
            nvec = action_space.nvec

            # Decode index into individual MultiDiscrete values
            action = []
            for size in reversed(nvec):
                action.append(index % size)
                index //= size
            action = list(reversed(action))  # Now it's [0, 1, 2, 3], etc.

            # Group into tuples
            grouped = [tuple(action[i:i+2]) for i in range(0, len(action), 2)]
            return grouped