import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType, RenderFrame
from gymnasium.envs.registration import register
from typing import SupportsFloat, Any
from vgc2.agent import BattlePolicy
from vgc2.agent.battle import RandomBattlePolicy, GreedyBattlePolicy
from vgc2.battle_engine import BattleEngine, TeamView, State, StateView, BattleRuleParam, BattleCommand
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.competition.match import label_teams
from vgc2.util.generator import gen_team
from custom_encodings import encode_state, ENCODING_CONSTANTS, EncodeContext
from vgc2.util.forward import *

# Register this module as a gym environment. Once registered, the id is usable in gym.make().
register(
    id='pokemon_battle_env',
    entry_point='pokemon_battle_env:PokemonBattleEnv',
)

class PokemonBattleEnv(gym.Env):
    def __init__(self,
                 randomize_enemy: bool = False,
                 ctx: EncodeContext = EncodeContext(),
                 n_active: int = 2,
                 max_team_size: int = 4,
                 max_pkm_moves: int = 4,
                 params: BattleRuleParam = BattleRuleParam(),
                 opponent: BattlePolicy = GreedyBattlePolicy()):
        self.randomize_enemy = randomize_enemy
        self.ctx = ctx
        self.n_active = n_active
        self.max_team_size = max_team_size
        self.max_pkm_moves = max_pkm_moves
        self.params = params
        self.opponents = [GreedyBattlePolicy(), RandomBattlePolicy()]
        self.opponent = opponent
        self.encode_state = encode_state
        self.gen_team = gen_team
        self.encode_len = ENCODING_CONSTANTS.STATE
        self.int_action_space = gym.spaces.MultiDiscrete([max_pkm_moves, max(max_team_size - n_active, n_active)] * n_active,
                                          start=[0, 0] * n_active)
        self.action_space = gym.spaces.Discrete(64)
        self.emb_obs_space_len = ENCODING_CONSTANTS.EMBEDDED_MOVE
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.encode_len,), dtype=np.float32)
        self.engine, self.state_view = self._get_engine_view()
        self.encode_buffer = np.zeros(self.encode_len)
        self.matches = 0
        self.wins = 0
        self.winrate = 0
        self.winrate_history = np.array([])

    def _get_engine_view(self) -> tuple[BattleEngine, tuple[StateView, StateView]]:
        team = (self.gen_team(self.max_team_size, self.max_pkm_moves),
                self.gen_team(self.max_team_size, self.max_pkm_moves))
        label_teams(team)
        team_view = TeamView(team[0]), TeamView(team[1])
        state = State(get_battle_teams(team, self.n_active))
        return BattleEngine(state, self.params), (StateView(state, 0, team_view), StateView(state, 1, team_view))
    
    def set_opponent(self, opponent: BattlePolicy = RandomBattlePolicy()):
        self.opponent = np.random.choice(self.opponents)

    def set_random_teams(self):
        self.engine, self.state_view = self._get_engine_view()

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.engine, self.state_view = self._get_engine_view()
        self.engine.reset()
        observation = self._get_obs()
        info = self._get_info()
        self.matches += 1
        self.winrate = self.wins / self.matches
        self.winrate_history = np.append(self.winrate_history, self.winrate)
        # Randomizes opponent (GreedyBot or RandomBot) for next battle
        if self.randomize_enemy:
            self.set_opponent()
        return observation, info

    def step(self,
             action) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # Setup for reward calculation
        state = copy_state(self.state_view[0])
        own_team = state.sides[0].team.active+state.sides[0].team.reserve
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve
        reward = 0
        
        # Handle index to action conversion and add to battle commands
        action = self.index_to_action(action, self.int_action_space)
        cmds: list[BattleCommand] = []
        cmds += action

        # Punish if agent chose disabled or out-of-pp move
        for i, battle_pkm in enumerate(state.sides[0].team.active):
            if battle_pkm.battling_moves[action[i][0]].disabled or battle_pkm.battling_moves[action[i][0]].pp <= 0:
                reward -= 5

        # Amount of dead opp pokemon
        dead_opp_pkm_before_turn = sum([1 if pkm.fainted() else 0 for pkm in opp_team])

        # Amount of dead own pokemon
        dead_own_pkm_before_turn = sum([1 if pkm.fainted() else 0 for pkm in own_team])

        # Run turn
        opp_action = self.opponent.decision(self.state_view[1])
        self.engine.run_turn((cmds, opp_action))

        # Reassign setup to calculate reward
        state = copy_state(self.state_view[0])
        own_team = state.sides[0].team.active+state.sides[0].team.reserve
        opp_team = state.sides[1].team.active+state.sides[1].team.reserve

        # Reward if enemy was killed
        dead_opp_pkm_after_turn = sum([1 if pkm.fainted() else 0 for pkm in opp_team])
        reward += (dead_opp_pkm_after_turn-dead_opp_pkm_before_turn) # A maximum of 2 pkm can be killed in a single turn

        # Punish if own pkm was killed
        dead_own_pkm_after_turn = sum([1 if pkm.fainted() else 0 for pkm in own_team])
        reward += dead_own_pkm_before_turn-dead_own_pkm_after_turn

        terminated = self.engine.state.terminal()

        # Reward agent if they won the battle, punish if lost
        if terminated:
            if self.engine.winning_side == 0:
                reward += 4
                self.wins += 1
            else:
                reward += -4

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def index_to_action(self, index: int, action_space: gym.spaces.MultiDiscrete):
            """
            Convert flat index to grouped MultiDiscrete action as tuples.

            Returns List[Tuple[int, int]]: Grouped action, e.g. [(0, 1), (2, 3)]
            """
            nvec = action_space.nvec
            dims = len(nvec)

            # Decode index into individual MultiDiscrete values
            action = []
            for size in reversed(nvec):
                action.append(index % size)
                index //= size
            action = list(reversed(action))  # Now it's [0, 1, 2, 3], etc.

            # Group into tuples
            grouped = [tuple(action[i:i+2]) for i in range(0, len(action), 2)]
            return grouped

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass

    def close(self):
        pass

    def _get_obs(self):
        self.encode_buffer = np.zeros(ENCODING_CONSTANTS.STATE)
        self.encode_state(self.encode_buffer, self.state_view[0], self.ctx)
        return self.encode_buffer

    def _get_info(self):
        return {}
    