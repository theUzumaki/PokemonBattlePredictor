import variables as v
import battleline_extractor as be
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any

from test_battleline_struct import example_battleline as example

def extract_battleline_features(battleline: v.battleline, max_moves: int = 4) -> np.ndarray:
    """
    Extract numerical features from battleline struct defined in variables.py.
    
    For each battle and each Pokemon in both teams, extracts:
    - Pokemon HP percentage (hps)
    - Base stats: atk, def, spa, spd, spe, hp
    - Boosts: atk, def, spa, spd, spe
    - Status (one-hot encoding for 8 status types)
    - Effects (one-hot encoding for 8 effect types)
    - Move features: base_pwr, accuracy, priority, cat (for each move separately, up to max_moves)
    
    Args:
        battleline: The battleline struct containing battles
        max_moves: Maximum number of moves to extract per Pokemon (default: 4)
    """
    
    features_list = []
    
    # Status list for one-hot encoding
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    
    # Effects list for one-hot encoding
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    
    # Iterate through all battles
    for battle_id, battle in battleline.battles.items():
        # Process both teams
        for team in [battle.team1, battle.team2]:
            # Process each Pokemon in the team
            for pkmn in team.pkmns:
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


def perform_pca_on_battleline(battleline: v.battleline, n_components: int = 10, max_moves: int = 4):
    """
    Perform PCA on battleline struct from variables.py.
    
    Args:
        battleline: A battleline struct containing battles with teams and Pokemon
        n_components: Number of principal components to extract
        max_moves: Maximum number of moves per Pokemon (default: 4)
        
    Returns:
        Tuple of (pca_model, transformed_data, scaler, feature_matrix)
    """
    print("Extracting battleline features...")
    features = extract_battleline_features(battleline, max_moves=max_moves)
    print(f"Extracted features shape: {features.shape}")
    print(f"Total Pokemon: {features.shape[0]}")
    print(f"Features per Pokemon: {features.shape[1]}")
    
    # Feature names for reference
    feature_names = [
        'hps',  # HP percentage
        'base_atk', 'base_def', 'base_spa', 'base_spd', 'base_spe', 'base_hp',  # Base stats
        'boost_atk', 'boost_def', 'boost_spa', 'boost_spd', 'boost_spe',  # Boosts
    ]
    
    # Add status one-hot encoding
    status_list = ['nostatus', 'slp', 'frz', 'tox', 'brn', 'par', 'fnt', 'psn']
    feature_names.extend([f'status_{status}' for status in status_list])
    
    # Add effects one-hot encoding
    effects_list = ['firespin', 'wrap', 'substitute', 'clamp', 'confusion', 'typechange', 'noeffect', 'reflect']
    feature_names.extend([f'effect_{effect}' for effect in effects_list])
    
    # Add move features for each move slot
    for i in range(1, max_moves + 1):
        feature_names.extend([
            f'move{i}_power',
            f'move{i}_accuracy',
            f'move{i}_priority',
            f'move{i}_cat'
        ])
    
    len_feature_names = len(feature_names)
    len_base = len(feature_names) - (len(status_list) + len(effects_list) + 4 * max_moves)
    len_status = len(status_list)
    len_effects = len(effects_list)
    print(f"Feature names ({len_feature_names} total):")
    print(f"  Base features ({len_base}): {feature_names[:len_base]}")
    print(f"  Status features ({len_status}): {feature_names[len_base:len_base+len_status]}")
    print(f"  Effect features ({len_effects}): {feature_names[len_base+len_status:len_base+len_status+len_effects]}")
    print(f"  Move features ({4*max_moves}): {feature_names[len_base+len_status+len_effects:]}")
    
    print("\nStandardizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    print(f"Performing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features_scaled)
    
    print(f"\nPCA Results:")
    print(f"Transformed data shape: {features_pca.shape}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Cumulative explained variance: {np.cumsum(pca.explained_variance_ratio_)}")
    
    # Print top contributing features for each component
    print("\nTop 5 contributing features for each component:")
    components = pca.components_
    for i, component in enumerate(components):
        top_indices = np.argsort(np.abs(component))[-5:][::-1]
        print(f"\nPC{i+1}:")
        for idx in top_indices:
            print(f"  {feature_names[idx]}: {component[idx]:.4f}")
    
    return pca, features_pca, scaler, features


def main():
    # Use the example battleline struct from test_battleline_struct.py
    battleline_struct = example
    
    print(f"Battleline contains {len(battleline_struct.battles)} battles")
    total_pokemon = sum(len(battle.team1.pkmns) + len(battle.team2.pkmns) 
                       for battle in battleline_struct.battles.values())
    print(f"Total Pokemon across all battles: {total_pokemon}\n")
    
    # Perform PCA on battleline data
    pca_model, transformed_data, scaler, original_features = perform_pca_on_battleline(
        battleline=battleline_struct,
        n_components=10
    )
    
    return {
        'battleline_struct': battleline_struct,
        'pca_model': pca_model,
        'transformed_data': transformed_data,
        'scaler': scaler,
        'original_features': original_features
    }


if __name__ == "__main__":
    results = main()