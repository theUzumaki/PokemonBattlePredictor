import variables as v
import battleline_extractor as be
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
import chronicle.logger as logger

from test_battleline_struct import example_battleline as example

def extract_battleline_features(battleline: v.battleline, max_moves: int = 4) -> np.ndarray:
    """
    Extract numerical features from battleline struct defined in variables.py.
    
    For each battle:
    - Team1 (full team): For each Pokemon, extracts:
      - Pokemon HP percentage (hps)
      - Base stats: atk, def, spa, spd, spe, hp
      - Boosts: atk, def, spa, spd, spe
      - Status (one-hot encoding for 8 status types)
      - Effects (one-hot encoding for 8 effect types)
      - Move features: base_pwr, accuracy, priority, cat (for each move separately, up to max_moves)
    
    - Team2 (adv_team): Extracts aggregated features:
      - pkmn_alive
      - pkmn_dscvrd_alive
      - hp_leader
      - type counts (one-hot encoding for each type)
      - status counts (one-hot encoding for each status)
    
    Args:
        battleline: The battleline struct containing battles
        max_moves: Maximum number of moves to extract per Pokemon (default: 4)
    """
    
    features_list = []
    
    # Status list for one-hot encoding
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    
    # Effects list for one-hot encoding
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    
    # Type list for adversary team encoding
    type_list = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy', 'notype']
    
    # Iterate through all battles
    for battle_id, battle in battleline.battles.items():
        # Process team1 (full team with individual Pokemon)
        for pkmn in battle.team1.pkmns:
            pkmn_features = []
            
            # HP percentage
            pkmn_features.append(pkmn.hps)
            
            # Base stats
            pkmn_features.extend([
                pkmn.base_stats.atk,
                pkmn.base_stats.def_,
                pkmn.base_stats.spa,
                pkmn.base_stats.spd,
                pkmn.base_stats.spe,
                pkmn.base_stats.hp
            ])
            
            # Boosts (stat changes during battle)
            pkmn_features.extend([
                pkmn.boosts.atk,
                pkmn.boosts.def_,
                pkmn.boosts.spa,
                pkmn.boosts.spd,
                pkmn.boosts.spe
            ])
            
            # Status (one-hot encoding)
            for status in status_list:
                pkmn_features.append(1 if pkmn.status == status else 0)
            
            # Effects (one-hot encoding)
            for effect in effects_list:
                # Set to 1 if the effect is present in the Pokemon's effects list
                pkmn_features.append(1 if effect in pkmn.effects else 0)
            
            # Move features (kept separate, padded to max_moves)
            # Each move has: base_pwr, accuracy, priority, cat
            for i in range(max_moves):
                if i < len(pkmn.moves):
                    move = pkmn.moves[i]
                    pkmn_features.extend([
                        move.base_pwr,
                        move.accuracy,
                        move.priority,
                        move.cat
                    ])
                else:
                    # Padding for missing moves
                    pkmn_features.extend([0, 0, 0, 0])
            
            features_list.append(pkmn_features)
    
    return np.array(features_list)


def extract_battle_features(battleline: v.battleline, max_moves: int = 4, use_individual_pokemon: bool = False) -> np.ndarray:
    """
    Extract battle-level features that combine information from both teams.
    Each row represents one complete battle.
    
    Args:
        battleline: The battleline struct containing battles
        max_moves: Maximum number of moves per Pokemon (default: 4)
        use_individual_pokemon: If True, keeps individual Pokemon features (results in larger feature vectors).
                               If False (default), aggregates Pokemon into team averages/counts.
    
    Returns:
        numpy array where each row represents a battle's features
        
    Feature extraction modes:
    
    If use_individual_pokemon=False (AGGREGATED):
        Team1 aggregated features:
        - Average HP across all Pokemon
        - Average base stats
        - Average boosts
        - Count of each status type
        - Count of each effect type
        - Average move power, accuracy
        - Number of Pokemon alive (HP > 0)
        
        Team2 (adversary) features:
        - pkmn_alive, pkmn_dscvrd_alive, hp_leader
        - Type presence (one-hot encoding)
        - Status presence (one-hot encoding - whether each status exists among alive Pokemon)
    
    If use_individual_pokemon=True (INDIVIDUAL):
        Team1 individual Pokemon features (for each of 6 Pokemon):
        - HP percentage
        - Base stats (6 values)
        - Boosts (5 values)
        - Status (one-hot, 8 values)
        - Effects (one-hot, 8 values)
        - Move features (max_moves * 4 values)
        
        Team2 (adversary) features (same as aggregated mode):
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
        
        if use_individual_pokemon:
            # ===== INDIVIDUAL POKEMON MODE =====
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
                        pkmn.boosts.spe
                    ])
                    
                    # Status (one-hot)
                    for status in status_list:
                        battle_features.append(1 if pkmn.status == status else 0)
                    
                    # Effects (one-hot)
                    for effect in effects_list:
                        battle_features.append(1 if effect in pkmn.effects else 0)
                    
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
                        else:
                            battle_features.extend([0, 0, 0, 0])
                else:
                    # Padding for missing Pokemon (all zeros)
                    # HP + 6 base stats + 5 boosts + 8 status + 8 effects + (max_moves * 4) moves
                    padding_size = 1 + 6 + 5 + 8 + 8 + (max_moves * 4)
                    battle_features.extend([0] * padding_size)
        
        else:
            # ===== AGGREGATED MODE (original) =====
            team1_pokemon = battle.team1.pkmns
            n_team1 = len(team1_pokemon)
            
            # Average HP
            avg_hp = np.mean([p.hps for p in team1_pokemon])
            battle_features.append(avg_hp)
            
            # Count of Pokemon alive (HP > 0)
            alive_count = sum(1 for p in team1_pokemon if p.hps > 0)
            battle_features.append(alive_count)
            
            # Average base stats
            avg_base_atk = np.mean([p.base_stats.atk for p in team1_pokemon])
            avg_base_def = np.mean([p.base_stats.def_ for p in team1_pokemon])
            avg_base_spa = np.mean([p.base_stats.spa for p in team1_pokemon])
            avg_base_spd = np.mean([p.base_stats.spd for p in team1_pokemon])
            avg_base_spe = np.mean([p.base_stats.spe for p in team1_pokemon])
            avg_base_hp = np.mean([p.base_stats.hp for p in team1_pokemon])
            battle_features.extend([avg_base_atk, avg_base_def, avg_base_spa, avg_base_spd, avg_base_spe, avg_base_hp])
            
            # Average boosts
            avg_boost_atk = np.mean([p.boosts.atk for p in team1_pokemon])
            avg_boost_def = np.mean([p.boosts.def_ for p in team1_pokemon])
            avg_boost_spa = np.mean([p.boosts.spa for p in team1_pokemon])
            avg_boost_spd = np.mean([p.boosts.spd for p in team1_pokemon])
            avg_boost_spe = np.mean([p.boosts.spe for p in team1_pokemon])
            battle_features.extend([avg_boost_atk, avg_boost_def, avg_boost_spa, avg_boost_spd, avg_boost_spe])
            
            # Status counts for team1
            for status in status_list:
                count = sum(1 for p in team1_pokemon if p.status == status)
                battle_features.append(count)
            
            # Effect counts for team1
            for effect in effects_list:
                count = sum(1 for p in team1_pokemon if effect in p.effects)
                battle_features.append(count)
            
            # Average move statistics
            all_moves = [move for p in team1_pokemon for move in p.moves]
            if all_moves:
                avg_move_power = np.mean([m.base_pwr for m in all_moves])
                avg_move_accuracy = np.mean([m.accuracy for m in all_moves])
                avg_move_priority = np.mean([m.priority for m in all_moves])
                pct_special_moves = np.mean([m.cat for m in all_moves])  # cat=1 for special
            else:
                avg_move_power = avg_move_accuracy = avg_move_priority = pct_special_moves = 0
            battle_features.extend([avg_move_power, avg_move_accuracy, avg_move_priority, pct_special_moves])
            
            # Number of moves known
            total_moves = len(all_moves)
            battle_features.append(total_moves)        # ===== TEAM2 (ADVERSARY) FEATURES =====
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


def extract_adversary_features(battleline: v.battleline) -> np.ndarray:
    """
    Extract features from adversary teams (adv_team) in the battleline.
    
    For each battle's team2 (adv_team), extracts:
    - pkmn_alive: Number of Pokemon still alive
    - pkmn_dscvrd_alive: Number of discovered Pokemon that are alive
    - hp_leader: HP percentage of the leading Pokemon
    - Type counts: One-hot encoding for each type present
    - Status counts: Count of each status type
    
    Args:
        battleline: The battleline struct containing battles
        
    Returns:
        numpy array of adversary team features (one row per battle)
    """
    features_list = []
    
    # Type list for one-hot encoding
    type_list = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy', 'notype']
    
    # Status list for counting
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    
    for battle_id, battle in battleline.battles.items():
        adv_features = []
        
        # Basic counts
        adv_features.append(battle.team2.pkmn_alive)
        adv_features.append(battle.team2.pkmn_dscvrd_alive)
        adv_features.append(battle.team2.hp_leader)
        
        # Type presence (one-hot encoding)
        for type_name in type_list:
            adv_features.append(1 if type_name in battle.team2.types else 0)
        
        # Status counts
        for status in status_list:
            count = battle.team2.statuses.count(status)
            adv_features.append(count)
        
        features_list.append(adv_features)
    
    return np.array(features_list)


def perform_pca_on_battles(battleline: v.battleline, n_components: int = 10, max_moves: int = 4, 
                           use_individual_pokemon: bool = False):
    """
    Perform PCA on battle-level features that combine both teams.
    
    Args:
        battleline: A battleline struct containing battles with teams and Pokemon
        n_components: Number of principal components to extract
        max_moves: Maximum number of moves per Pokemon (default: 4)
        use_individual_pokemon: If True, keeps individual Pokemon features instead of aggregating
        
    Returns:
        Tuple of (pca_model, transformed_data, scaler, feature_matrix, feature_names)
    """
    mode_str = "INDIVIDUAL POKEMON" if use_individual_pokemon else "AGGREGATED TEAM"
    logger.log_info(f"Extracting battle-level features ({mode_str} mode)")
    features = extract_battle_features(battleline, max_moves=max_moves, use_individual_pokemon=use_individual_pokemon)
    logger.log(1, 0, 0, logger.Colors.INFO, f"Extracted features shape: {features.shape}")
    logger.log(1, 0, 0, logger.Colors.INFO, f"Total battles: {features.shape[0]}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Features per battle: {features.shape[1]}")
    
    # Build feature names
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    type_list = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy', 'notype']
    
    feature_names = []
    
    if use_individual_pokemon:
        # Individual Pokemon mode feature names
        max_pokemon = 6
        for pkmn_idx in range(max_pokemon):
            prefix = f'p{pkmn_idx+1}_'
            feature_names.append(prefix + 'hps')
            feature_names.extend([prefix + f'base_{s}' for s in ['atk', 'def', 'spa', 'spd', 'spe', 'hp']])
            feature_names.extend([prefix + f'boost_{s}' for s in ['atk', 'def', 'spa', 'spd', 'spe']])
            feature_names.extend([prefix + f'status_{s}' for s in status_list])
            feature_names.extend([prefix + f'effect_{e}' for e in effects_list])
            for move_idx in range(1, max_moves + 1):
                feature_names.extend([
                    prefix + f'move{move_idx}_power',
                    prefix + f'move{move_idx}_accuracy',
                    prefix + f'move{move_idx}_priority',
                    prefix + f'move{move_idx}_cat'
                ])
    else:
        # Aggregated mode feature names (original)
        feature_names.extend(['team1_avg_hp', 'team1_alive_count'])
        feature_names.extend(['team1_avg_base_atk', 'team1_avg_base_def', 'team1_avg_base_spa', 
                             'team1_avg_base_spd', 'team1_avg_base_spe', 'team1_avg_base_hp'])
        feature_names.extend(['team1_avg_boost_atk', 'team1_avg_boost_def', 'team1_avg_boost_spa',
                             'team1_avg_boost_spd', 'team1_avg_boost_spe'])
        feature_names.extend([f'team1_status_count_{s}' for s in status_list])
        feature_names.extend([f'team1_effect_count_{e}' for e in effects_list])
        feature_names.extend(['team1_avg_move_power', 'team1_avg_move_accuracy', 
                             'team1_avg_move_priority', 'team1_pct_special_moves', 'team1_total_moves'])
    
    # Team2 features (same for both modes)
    feature_names.extend(['team2_pkmn_alive', 'team2_pkmn_dscvrd_alive', 'team2_hp_leader'])
    feature_names.extend([f'team2_has_type_{t}' for t in type_list])
    feature_names.extend([f'team2_has_status_{s}' for s in status_list])
    
    if use_individual_pokemon:
        logger.log(0, 1, 0, logger.Colors.CYAN, f"Feature categories ({len(feature_names)} total):")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Individual Pokemon features: {max_pokemon} Pokemon × {(len(feature_names) - 30) // max_pokemon} features each")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team2 basic: 3 features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team2 types: {len(type_list)} features")
        logger.log(1, 0, 1, logger.Colors.INFO, f"Team2 status presence: {len(status_list)} features")
    else:
        logger.log(0, 1, 0, logger.Colors.CYAN, f"Feature categories ({len(feature_names)} total):")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team1 basic: 2 features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team1 base stats: 6 features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team1 boosts: 5 features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team1 status counts: {len(status_list)} features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team1 effect counts: {len(effects_list)} features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team1 move stats: 5 features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team2 basic: 3 features")
        logger.log(1, 0, 0, logger.Colors.INFO, f"Team2 types: {len(type_list)} features")
        logger.log(1, 0, 1, logger.Colors.INFO, f"Team2 status presence: {len(status_list)} features")
    
    logger.log_info("Standardizing features...", newline_before=1)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    n_components = min(n_components, features.shape[0], features.shape[1])
    logger.log_info(f"Performing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    logger.log_subsection("PCA Results")
    logger.log(0, 0, 0, logger.Colors.INFO, f"Transformed data shape: {features_pca.shape}")
    logger.log(0, 0, 0, logger.Colors.INFO, f"Explained variance ratio: {pca.explained_variance_ratio_}")
    logger.log(0, 0, 1, logger.Colors.INFO, f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Print top contributing features for each component
    logger.log(0, 1, 0, logger.Colors.CYAN, "Top 10 contributing features for each component:")
    components = pca.components_
    for i, component in enumerate(components):
        if pca.explained_variance_ratio_[i] > 0.01:  # Only show components explaining >1% variance
            top_indices = np.argsort(np.abs(component))[-10:][::-1]
            logger.log(0, 1, 0, logger.Colors.BRIGHT_BLUE, f"PC{i+1} (explains {pca.explained_variance_ratio_[i]*100:.2f}% of variance):")
            for idx in top_indices:
                logger.log(1, 0, 0, logger.Colors.DIM, f"{feature_names[idx]}: {component[idx]:.4f}")
    
    return pca, features_pca, scaler, features, feature_names


def main():
    # Use the example battleline struct from test_battleline_struct.py
    battleline_struct = example
    
    logger.log_info(f"Battleline contains {len(battleline_struct.battles)} battles")
    total_pokemon = sum(len(battle.team1.pkmns) for battle in battleline_struct.battles.values())
    logger.log_info(f"Total Pokemon in team1 across all battles: {total_pokemon}")
    logger.log_info(f"Total adversary teams (team2): {len(battleline_struct.battles)}", newline_after=1)
    
    # Choose mode: False for aggregated, True for individual
    use_individual = True  # Change this to True to use individual Pokemon features
    
    # Perform PCA on battle-level features (combining both teams)
    mode_name = "INDIVIDUAL POKEMON MODE" if use_individual else "AGGREGATED TEAM MODE"
    logger.log_section_header(f"BATTLE-LEVEL PCA ANALYSIS ({mode_name})")
    pca_model, transformed_data, scaler, original_features, feature_names = perform_pca_on_battles(
        battleline=battleline_struct,
        n_components=min(10, len(battleline_struct.battles)),  # Can't have more components than samples
        use_individual_pokemon=use_individual
    )
    
    # Display the actual battle feature values
    logger.log_section_header("BATTLE FEATURE VALUES")
    for i, battle_id in enumerate(battleline_struct.battles.keys()):
        logger.log(0, 1, 0, logger.Colors.BRIGHT_CYAN, f"Battle {battle_id} feature values (first 20 features):")
        for j in range(min(20, len(feature_names))):
            logger.log(1, 0, 0, logger.Colors.DIM, f"{feature_names[j]}: {original_features[i, j]:.3f}")
        logger.log(1, 0, 0, logger.Colors.DIM, "...")
    
    # Show PCA-transformed coordinates for each battle
    logger.log_section_header("BATTLE COORDINATES IN PCA SPACE")
    for i, battle_id in enumerate(battleline_struct.battles.keys()):
        logger.log(0, 1, 0, logger.Colors.BRIGHT_CYAN, f"Battle {battle_id} in PCA space:")
        coords = ", ".join([f"PC{j+1}={transformed_data[i, j]:.3f}" for j in range(min(5, transformed_data.shape[1]))])
        logger.log(1, 0, 0, logger.Colors.INFO, f"[{coords}]")
    
    return {
        'battleline_struct': battleline_struct,
        'pca_model': pca_model,
        'transformed_data': transformed_data,
        'scaler': scaler,
        'original_features': original_features,
        'feature_names': feature_names
    }


if __name__ == "__main__":
    results = main()