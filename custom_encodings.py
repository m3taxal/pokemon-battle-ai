import numpy as np
from vgc2.battle_engine.game_state import Side, State
from vgc2.battle_engine.modifiers import Weather, Terrain, Hazard, Stat, Status
from vgc2.battle_engine.move import Move, BattlingMove
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.team import BattlingTeam
from vgc2.util.encoding import EncodeContext, one_hot, multi_hot
from vgc2.battle_engine.damage_calculator import calculate_damage
from vgc2.battle_engine.constants import BattleRuleParam

class ENCODING_CONSTANTS:
    MAX_TEAMS = 2
    MAX_PKM_PER_TEAM = 2
    MAX_MOVES = 4
    EMBEDDED_MOVE = 16
    MOVE = 36
    STATE = 335

def encode_move(e: np.array,
                move: Move,
                attacker: BattlingPokemon,
                state: State,
                params = BattleRuleParam(),
                ctx = EncodeContext()) -> int:
    i = 0
    for defender in state.sides[1].team.active:
        e[i] = calculate_damage(params, 0, move, state, attacker, defender) / ctx.max_hp
        i += 1
    if len(state.sides[1].team.active)==1:
        i += 1 # Keep state observation consistent length
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
                         attacker: BattlingPokemon,
                         state: State,
                         ctx = EncodeContext()) -> int:
    i = encode_move(e, move.constants, attacker, state)
    e[i] = move.pp / ctx.max_pp
    i += 1
    return i

def encode_battling_pokemon(e: np.array,
                            pokemon: BattlingPokemon,
                            state: State,
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

    # Simple status check (1 if pokemon is affected by any status effect, 0 otherwise)
    # May change later...
    if pokemon.status != Status.NONE:
        e[i] = 1
    i += 1

    #if pokemon.status != Status.NONE:
    #    one_hot(e[i:], pokemon.status, ctx.n_status)
    #i += ctx.n_status
    return i

def encode_battling_team(e: np.array,
                         team: list[BattlingPokemon],
                         state: State,
                         ctx = EncodeContext()) -> int:
    i = 0
    pkm_enc_len = 0 # Encoding length of pokemon may vary during development
    for m in team:
        pkm_enc_len = encode_battling_pokemon(e[i:], m, state)
        i += pkm_enc_len
    if len(team)==1:
        i += pkm_enc_len
    return i

def encode_side(e: np.array,
                side: Side,
                state: State,
                ctx = EncodeContext()) -> int:
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

    # Sides
    for s in state.sides:
        i += encode_side(e[i:], s, state) # This doesn't encode pokemon moves

    # Only encode moves of my pokemon, not enemy's
    for battle_pkm in state.sides[0].team.active:
        for m in battle_pkm.battling_moves:
            i += encode_battling_move(e[i:], m, battle_pkm, state)
    
    if len(state.sides[0].team.active)==1:
        i += ENCODING_CONSTANTS.MOVE*4 # Keep state observation consistent length

    if state.weather != Weather.CLEAR:
        one_hot(e[i:], state.weather - 1, ctx.n_weather)
    i += ctx.n_weather
    if state.field != Terrain.NONE:
        one_hot(e[i:], state.field - 1, ctx.n_terrain)
    i += ctx.n_terrain
    e[i] = float(state.trickroom)
    i += 1
    return i