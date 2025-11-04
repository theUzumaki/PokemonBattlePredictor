import variables as v
import numpy as np

def extract_battle_features(battleline: v.battleline, max_moves: int = 4) -> np.ndarray:
    """
    Extract battle-level features that combine information from both teams.
    Each row represents one complete battle.
    
    Args:
        battleline: The battleline struct containing battles
        max_moves: Maximum number of moves per Pokemon (default: 4)
    
    Returns:
        numpy array where each row represents a battle's features
        
    Team1 individual Pokemon features (for each of 6 Pokemon):
        - HP percentage
        - Base stats (6 values)
        - Boosts (6 values: atk, def, spa, spd, spe, hp)
        - Status (one-hot, 8 values)
        - Effects (one-hot, 8 values)
        - Types (one-hot, 19 values for type1 and 19 for type2)
        - Move features (max_moves * (4 base features + 19 type features))
          - base_pwr, accuracy, priority, cat
          - type (one-hot, 19 values)
        
    Team2 (adversary) features:
        - pkmn_alive, pkmn_dscvrd_alive, hp_leader
        - Type presence (one-hot encoding)
        - Status presence (one-hot encoding)
    """
    
    battle_features_list = []
    
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    type_list = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy', 'notype']
    
    for battle_id, battle in battleline.battles.items():
        battle_features = []
        
        # Extract features for each Pokemon individually (up to 6 Pokemon)
        team1_pokemon = battle.team1.pkmns
        
        # Pad or truncate to exactly 6 Pokemon
        max_pokemon = 6
        for i in range(max_pokemon):
            if i < len(team1_pokemon):
                pkmn = team1_pokemon[i]
                
                # HP percentage
                battle_features.append(pkmn.hps)
                
                # Base stats
                battle_features.extend([
                    pkmn.base_stats.atk,
                    pkmn.base_stats.def_,
                    pkmn.base_stats.spa,
                    pkmn.base_stats.spd,
                    pkmn.base_stats.spe,
                    pkmn.base_stats.hp
                ])
                
                # Boosts
                battle_features.extend([
                    pkmn.boosts.atk,
                    pkmn.boosts.def_,
                    pkmn.boosts.spa,
                    pkmn.boosts.spd,
                    pkmn.boosts.spe,
                    pkmn.boosts.hp
                ])
                
                # Status (one-hot)
                for status in status_list:
                    battle_features.append(1 if pkmn.status == status else 0)
                
                # Effects (one-hot)
                for effect in effects_list:
                    battle_features.append(1 if effect in pkmn.effects else 0)

                # Types (one-hot) - type1 first, then type2
                for type_name in type_list:
                    battle_features.append(1 if type_name in pkmn.type1 else 0)
                for type_name in type_list:
                    battle_features.append(1 if type_name in pkmn.type2 else 0)
                
                # Move features
                for j in range(max_moves):
                    if j < len(pkmn.moves):
                        move = pkmn.moves[j]
                        battle_features.extend([
                            move.base_pwr,
                            move.accuracy,
                            move.priority,
                            move.cat
                        ])
                        # Move type (one-hot encoding)
                        for type_name in type_list:
                            battle_features.append(1 if move.type == type_name else 0)
                    else:
                        # Padding for missing moves: 4 base features + 19 type features
                        battle_features.extend([0] * (4 + len(type_list)))
            else:
                # Padding for missing Pokemon (all zeros)
                # HP + 6 base stats + 6 boosts + 8 status + 8 effects + 38 types (19*2) + (max_moves * (4 + 19)) moves
                padding_size = 1 + 6 + 6 + 8 + 8 + 38 + (max_moves * (4 + 19))
                battle_features.extend([0] * padding_size)
        
        # ===== TEAM2 (ADVERSARY) FEATURES =====
        battle_features.append(battle.team2.pkmn_alive)
        battle_features.append(battle.team2.pkmn_dscvrd_alive)
        battle_features.append(battle.team2.hp_leader)
        
        # Type presence (one-hot)
        for type_name in type_list:
            battle_features.append(1 if type_name in battle.team2.types else 0)
        
        # Status one-hot encoding for alive Pokemon
        # Check if each status is present in the alive Pokemon
        for status in status_list:
            battle_features.append(1 if status in battle.team2.statuses else 0)
        
        battle_features_list.append(battle_features)
    
    return np.array(battle_features_list)