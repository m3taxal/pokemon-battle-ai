import numpy as np
from vgc2.battle_engine.game_state import Side, State
from vgc2.battle_engine.modifiers import Weather, Terrain, Hazard, Stat, Status
from vgc2.battle_engine.move import Move, BattlingMove
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.util.encoding import EncodeContext, one_hot
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.util.forward import copy_state

class ENCODING_CONSTANTS:
    MAX_TEAMS = 2
    MAX_PKM_PER_TEAM = 2
    MAX_MOVES = 4
    MOVE = 36 # Meaning that each pokemon move can be encoded as a 36 dimensional vector (this is with calculated dmg)
    EMBEDDED_MOVE = 16 # We encode 36-dimensional move vectors into 16-dimensional embedded move vectors
    STATE = 421 # In total, each battle state has a total 423 observations

def encode_move(e: np.array,
                move: Move,
                ctx = EncodeContext()) -> int:
    i = 0
    e[i] = move.accuracy
    i += 1
    e[i] = move.priority / ctx.max_priority
    i += 1
    e[i] = move.effect_prob
    i += 1
    e[i] = float(move.force_switch)
    i += 1
    e[i] = float(move.self_switch)
    i += 1
    e[i] = float(move.ignore_evasion)
    i += 1
    e[i] = float(move.protect)
    i += 1
    e[i] = move.heal / ctx.max_ratio
    i += 1
    e[i] = move.recoil / ctx.max_ratio
    i += 1
    e[i] = float(move.toggle_trickroom)
    i += 1
    e[i] = float(move.toggle_reflect)
    i += 1
    e[i] = float(move.toggle_lightscreen)
    i += 1
    e[i] = float(move.toggle_tailwind)
    i += 1
    e[i] = float(move.change_type)
    i += 1
    e[i] = float(move.disable)
    i += 1
    e[i] = move.boosts[Stat.SPEED] / ctx.max_boost
    i += 1
    e[i] = move.boosts[Stat.EVASION] / ctx.max_boost
    i += 1
    e[i] = move.boosts[Stat.ACCURACY] / ctx.max_boost
    i += 1
    if move.weather_start != Weather.CLEAR:
        one_hot(e[i:], move.weather_start - 1, ctx.n_weather)
    i += ctx.n_weather
    if move.field_start != Terrain.NONE:
        one_hot(e[i:], move.field_start - 1, ctx.n_terrain)
    i += ctx.n_terrain
    if move.hazard != Hazard.NONE:
        one_hot(e[i:], move.hazard - 1, ctx.n_hazard)
    i += ctx.n_hazard
    if move.status != Status.NONE:
        one_hot(e[i:], move.status - 1, ctx.n_status)
    i += ctx.n_status
    return i

def encode_battling_move(e: np.array,
                         move: BattlingMove,
                         ctx = EncodeContext()) -> int:
    i = encode_move(e, move.constants)
    e[i] = move.pp / ctx.max_pp
    i += 1
    return i

def encode_battling_pokemon(e: np.array,
                            pokemon: BattlingPokemon,
                            ctx = EncodeContext()) -> int:
    i = 0

    # Remaining HP
    e[i] = pokemon.hp / pokemon.constants.stats[Stat.MAX_HP]
    i += 1

    # Important stats
    e[i] = pokemon.constants.stats[Stat.SPEED] / ctx.max_hp
    i += 1

    # Important boosts
    e[i] = pokemon.boosts[Stat.SPEED]
    i += 1
    e[i] = pokemon.boosts[Stat.EVASION]
    i += 1
    e[i] = pokemon.boosts[Stat.ACCURACY]
    i += 1

    # Protect
    e[i] = float(pokemon.protect)
    i += 1

    if pokemon.status != Status.NONE:
        one_hot(e[i:], pokemon.status, ctx.n_status)
    i += ctx.n_status
    return i

def encode_battling_team(e: np.array,
                         team: list[BattlingPokemon],
                         state: State) -> int:
    i = 0
    pkm_enc_len = 0 
    for m in team:
        pkm_enc_len = encode_battling_pokemon(e[i:], m)
        i += pkm_enc_len
    if len(team)==1:
        i += pkm_enc_len # We want to keep a consistent encoding length, so if there is only one pokemon left on the team, 
                         # features of the second non-existing pokemon will all be encoded as 0
    return i

def encode_side(e: np.array,
                side: Side,
                state: State) -> int:
    i = 0
    i += encode_battling_team(e[i:], side.team.active, state)
    e[i] = float(side.conditions.reflect)
    i += 1
    e[i] = float(side.conditions.lightscreen)
    i += 1
    e[i] = float(side.conditions.tailwind)
    i += 1
    e[i] = float(side.conditions.stealth_rock)
    i += 1
    e[i] = float(side.conditions.poison_spikes)
    i += 1
    return i

def encode_state(e: np.array,
                 state: State,
                 ctx = EncodeContext()) -> int:
    i = 0
    # Encode buffer never actually gets zerod out...
    # Sides
    for s in state.sides:
        i += encode_side(e[i:], s, state) # This doesn't encode pokemon moves

    if state.weather != Weather.CLEAR:
        one_hot(e[i:], state.weather - 1, ctx.n_weather)
    i += ctx.n_weather
    if state.field != Terrain.NONE:
        one_hot(e[i:], state.field - 1, ctx.n_terrain)
    i += ctx.n_terrain
    e[i] = float(state.trickroom)
    i += 1

    params = BattleRuleParam()

    # Encode my pkm's moves
    for battle_pkm in state.sides[0].team.active:
        for m in battle_pkm.battling_moves:
            for defender in state.sides[1].team.active:
                e[i] = calculate_damage(params, 0, m.constants, state, battle_pkm, defender) / ctx.max_hp
                i += 1
            if len(state.sides[1].team.active)==1:
                i += 1 # Keep state observation consistent length
            i += encode_battling_move(e[i:], m)
        diff = ENCODING_CONSTANTS.MAX_MOVES - len(battle_pkm.battling_moves)
        i += diff*ENCODING_CONSTANTS.MOVE # Possible that a pokemon has less than 4 moves
    diff = ENCODING_CONSTANTS.MAX_PKM_PER_TEAM - len(state.sides[0].team.active)
    i += ENCODING_CONSTANTS.MOVE*ENCODING_CONSTANTS.MAX_MOVES*diff # Keep state observation consistent length

    # TODO: Encode predicted move of opponent
    opp = GreedyBattlePolicy()
    opp_state = State(tuple((state.sides[1], state.sides[0])))
    opp_decision = opp.decision(opp_state)
    for j, pkm in enumerate(opp_state.sides[0].team.active):
        if len(pkm.battling_moves) == 0:
            i += ENCODING_CONSTANTS.MOVE-1
            continue
        move = pkm.battling_moves[opp_decision[j][0]]
        target = opp_state.sides[1].team.active[opp_decision[j][1]]
        e[i] = calculate_damage(params, 0, move.constants, opp_state, pkm, target) / ctx.max_hp
        i += 1
        i += encode_battling_move(e[i:], move)
    diff = ENCODING_CONSTANTS.MAX_PKM_PER_TEAM - len(opp_state.sides[0].team.active)
    i += diff*(ENCODING_CONSTANTS.MOVE-1)

    return i