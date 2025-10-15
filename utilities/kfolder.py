# K-Fold Target Encoding for Pokemon Names
from typing import List, Dict, Any
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def kfold_target_encode_pokemon(data: List[Dict[str, Any]], n_splits: int = 5, random_state: int = 42) -> Dict[str, float]:
    """
    Perform k-fold target encoding for Pokemon names based on win rate.
    
    Args:
        data: List of battle records from JSONL
        n_splits: Number of folds for cross-validation
        random_state: Random seed for reproducibility
    
    Returns:
        Dictionary mapping Pokemon name to encoded value (mean win rate)
    """

    pokemon_records = []
    for record in data:
        target = 1 if record.get("player_won", False) else 0
        for pokemon in record.get("p1_team_details", []):
            pokemon_name = pokemon.get("name", "")
            if pokemon_name:
                pokemon_records.append({
                    'name': pokemon_name,
                    'target': target
                })
    
    if not pokemon_records:
        print("Warning: No Pokemon records found for target encoding")
        return {}
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(pokemon_records)
    
    # Initialize encoding dictionary with global mean as fallback
    global_mean = df['target'].mean()
    encoding_dict = defaultdict(lambda: global_mean)
    
    # Create indices array
    indices = np.arange(len(df))
    
    # Perform k-fold target encoding
    print(f"Performing K-Fold Target Encoding with {n_splits} splits...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # All encodings collected from each fold
    fold_encodings = defaultdict(list)
    
    print("Percentage of folds completed:")
    for i, (train_idx, val_idx) in enumerate(kf.split(indices)):
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Calculate mean target for each Pokemon in training set
        train_means = train_df.groupby('name')['target'].mean().to_dict()
        
        # Apply encoding to validation set
        for pokemon_name in val_df['name'].unique():
            if pokemon_name in train_means:
                fold_encodings[pokemon_name].append(train_means[pokemon_name])
            else:
                # Use global mean if Pokemon not seen in training fold
                fold_encodings[pokemon_name].append(global_mean)
        
        print(f"{(i + 1) / n_splits * 100:.1f}%", end="\r", flush=True)
    print("\nK-Fold Target Encoding completed.")
    
    # Average the encodings across all folds
    for pokemon_name, encodings in fold_encodings.items():
        encoding_dict[pokemon_name] = np.mean(encodings)
    
    # For Pokemon that weren't in validation sets, use overall mean from full dataset
    all_pokemon = df['name'].unique()
    for pokemon_name in all_pokemon:
        if pokemon_name not in encoding_dict:
            encoding_dict[pokemon_name] = df[df['name'] == pokemon_name]['target'].mean()
    
    return dict(encoding_dict)