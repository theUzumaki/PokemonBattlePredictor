import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import variables as v
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import chronicle.logger as logger
from extractor import extract_battle_features


def perform_pca_on_battles(battleline: v.battleline, n_components: int = 10, max_moves: int = 4):
    """
    Perform PCA on battle-level features that combine both teams.
    
    Args:
        battleline: A battleline struct containing battles with teams and Pokemon
        n_components: Number of principal components to extract
        max_moves: Maximum number of moves per Pokemon (default: 4)
        
    Returns:
        Tuple of (pca_model, transformed_data, scaler, feature_matrix, feature_names)
    """
    logger.log_info("Extracting battle-level features (INDIVIDUAL POKEMON mode)")
    features = extract_battle_features(battleline, max_moves=max_moves)
    logger.log(1, 0, 0, logger.Colors.INFO, f"Extracted features shape: {features.shape}")
    logger.log(1, 0, 0, logger.Colors.INFO, f"Total battles: {features.shape[0]}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Features per battle: {features.shape[1]}")
    
    # Debug: Show first Pokemon's features from first battle
    logger.log(0, 1, 0, logger.Colors.YELLOW, "DEBUG: First Pokemon of first battle - Feature extraction check:")
    first_battle = list(battleline.battles.values())[0]
    first_pokemon = first_battle.team1.pkmns[0]
    logger.log(1, 0, 0, logger.Colors.DIM, f"Pokemon ID: {first_pokemon.id}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"HP: {first_pokemon.hps}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"Type1: {first_pokemon.type1}, Type2: {first_pokemon.type2}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"Status: {first_pokemon.status}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"Effects: {first_pokemon.effects}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"Base stats: atk={first_pokemon.base_stats.atk}, def={first_pokemon.base_stats.def_}, spa={first_pokemon.base_stats.spa}, spd={first_pokemon.base_stats.spd}, spe={first_pokemon.base_stats.spe}, hp={first_pokemon.base_stats.hp}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"Boosts: atk={first_pokemon.boosts.atk}, def={first_pokemon.boosts.def_}, spa={first_pokemon.boosts.spa}, spd={first_pokemon.boosts.spd}, spe={first_pokemon.boosts.spe}, hp={first_pokemon.boosts.hp}")
    logger.log(1, 0, 0, logger.Colors.DIM, f"Number of moves: {len(first_pokemon.moves)}")
    for i, move in enumerate(first_pokemon.moves):
        logger.log(1, 0, 0, logger.Colors.DIM, f"  Move {i+1}: {move.name} (type={move.type}, power={move.base_pwr}, acc={move.accuracy}, pri={move.priority}, cat={move.cat})")
    logger.log(1, 0, 1, logger.Colors.DIM, f"Extracted feature values (first Pokemon only, first 80 features): {features[0, :80]}")
    
    # Build feature names
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    type_list = ['normal', 'fire', 'water', 'electric', 'grass', 'ice', 'fighting', 'poison',
                 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy', 'notype']
    
    feature_names = []
    
    # Individual Pokemon mode feature names
    max_pokemon = 6
    for pkmn_idx in range(max_pokemon):
        prefix = f'p{pkmn_idx+1}_'
        feature_names.append(prefix + 'hps')
        feature_names.extend([prefix + f'base_{s}' for s in ['atk', 'def', 'spa', 'spd', 'spe', 'hp']])
        feature_names.extend([prefix + f'boost_{s}' for s in ['atk', 'def', 'spa', 'spd', 'spe', 'hp']])
        feature_names.extend([prefix + f'status_{s}' for s in status_list])
        feature_names.extend([prefix + f'effect_{e}' for e in effects_list])
        feature_names.extend([prefix + f'type1_{t}' for t in type_list])
        feature_names.extend([prefix + f'type2_{t}' for t in type_list])
        for move_idx in range(1, max_moves + 1):
            feature_names.extend([
                prefix + f'move{move_idx}_power',
                prefix + f'move{move_idx}_accuracy',
                prefix + f'move{move_idx}_priority',
                prefix + f'move{move_idx}_cat'
            ])
            feature_names.extend([prefix + f'move{move_idx}_type_{t}' for t in type_list])
    
    # Team2 features
    feature_names.extend(['team2_pkmn_alive', 'team2_pkmn_dscvrd_alive', 'team2_hp_leader'])
    feature_names.extend([f'team2_has_type_{t}' for t in type_list])
    feature_names.extend([f'team2_has_status_{s}' for s in status_list])
    
    logger.log(0, 1, 0, logger.Colors.CYAN, f"Feature categories ({len(feature_names)} total):")
    logger.log(1, 0, 0, logger.Colors.INFO, f"Individual Pokemon features: {max_pokemon} Pokemon × {(len(feature_names) - 30) // max_pokemon} features each")
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
    
    # Perform PCA on battle-level features (combining both teams)
    logger.log_section_header("BATTLE-LEVEL PCA ANALYSIS (INDIVIDUAL POKEMON MODE)")
    pca_model, transformed_data, scaler, original_features, feature_names = perform_pca_on_battles(
        battleline=battleline_struct,
        n_components=min(10, len(battleline_struct.battles))  # Can't have more components than samples
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
