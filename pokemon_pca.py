from pathlib import Path
import json
from typing import List, Dict, Any, Iterator, Union
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
import numpy as np
from collections import defaultdict
import pandas as pd

pokemon_species = []
pokemon_species_count = defaultdict(int)  # Counter for Pokémon species occurrences
feature_names = ["name", "atk", "def", "hp", "spa", "spd", "spe"]

def iter_test_data(path: Union[str, Path] = None, entry_to_print = 0) -> Iterator[Dict[str, Any]]:
    """
    Yield JSON objects from a JSONL file one by one (memory efficient).
    """
    if path is None:
        path = Path(__file__).resolve().parent / "data" / "train.jsonl"
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
                if lineno <= entry_to_print:
                    entry = json.loads(line)
                    entry_str = json.dumps(entry, indent=2)
                    if len(entry_str) > 500:  # Truncate if exceeds 500 characters
                        entry_str = entry_str[:500] + "...\n[Truncated]"
                    print(f"Entry {lineno}:\n{entry_str}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} in {path}: {e.msg}") from e
            
def name_to_int(name: str) -> int:
    if name in pokemon_species:
        pokemon_species_count[name] += 1  # Increment the counter for this Pokémon
        return pokemon_species.index(name)
    else:
        pokemon_species.append(name)
        pokemon_species_count[name] += 1  # Initialize the counter for this Pokémon
        return len(pokemon_species) - 1

# K-Fold Target Encoding for Pokemon Names
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
    # Prepare data: extract all Pokemon from p1_team and their corresponding target
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
        print("Warning: No Pokemon records found for target encoding!")
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


# Extract numerical data with optional target encoding
def extract_features_with_encoding(
    data: List[Dict[str, Any]], 
    max_elements: int = float("inf"),
    use_target_encoding: bool = False,
    target_encoding_dict: Dict[str, float] = None
) -> np.ndarray:
    """
    Extract features from Pokemon data with optional target encoding for names.
    
    Args:
        data: List of battle records
        max_elements: Maximum number of records to process
        use_target_encoding: Whether to use target encoding instead of label encoding
        target_encoding_dict: Dictionary mapping Pokemon names to encoded values
    
    Returns:
        NumPy array of features
    """
    features = {}
    for feat in feature_names:
        features[feat] = []
        
    i = 0
    for record in data:
        if i >= max_elements:
            break
        i += 1

        for pokemon in record["p1_team_details"]:
            for key in feature_names:
                format_key = f"base_{key}" if key != "name" else "name"
                if format_key not in pokemon:
                    raise KeyError(f"Missing key '{key}' in Pokémon data: {pokemon}")

                if key == "name":
                    if use_target_encoding and target_encoding_dict:
                        # Use target encoding
                        pokemon_name = pokemon[format_key]
                        encoded_value = target_encoding_dict.get(pokemon_name, 0.5)  # 0.5 as fallback
                        features["name"].append(encoded_value)
                    else:
                        # Use original label encoding
                        features["name"].append(name_to_int(pokemon[format_key]))
                else:
                    features[key].append(pokemon[f"{format_key}"])
    return np.column_stack([features[key] for key in features])



if __name__ == "__main__":
    # quick check when running the module directly
    data = list(iter_test_data())
    components = 2

    print("\n\n---- Data Summary ----\n")
    print(f"Loaded {len(data)} records from data/test.jsonl")

    # Perform K-Fold Target Encoding
    print("\n---- K-Fold Target Encoding ----\n")
    target_encoding_dict = kfold_target_encode_pokemon(data, n_splits=10000, random_state=42)
    
    print(f"Generated target encodings for {len(target_encoding_dict)} unique Pokemon species: ")
    sorted_pokemon = sorted(target_encoding_dict.items(), key=lambda x: x[1], reverse=True)
    for i, (pokemon_name, encoded_value) in enumerate(sorted_pokemon[:10]):
        print(f"{i + 1:2}. {pokemon_name:12} - Win rate: {encoded_value:.4f}")

    # Perform PCA with target encoding
    print("\n\n---- PCA with Target Encoding ----\n")
    features_encoded = extract_features_with_encoding(
        data, 
        max_elements=10000, 
        use_target_encoding=True,
        target_encoding_dict=target_encoding_dict
    )

    # Show all Pokemon with their encoded values
    print(f"Pokemon species with target encodings:\n")
    for i, (pokemon_name, encoded_value) in enumerate(sorted_pokemon):
        print(f"{i + 1:3}: {pokemon_name:12}\t(Encoded: {encoded_value:.4f})")
        if i >= 19:
            break
    print()

    if features_encoded.size > 0:
        pca = PCA(n_components=components)
        transformed = pca.fit_transform(features_encoded)

        # Retrieve the names of the best features
        best_features = [feature_names[i] for i in pca.components_[0].argsort()[-components:][::-1]]
        print("Best features and contribution to first principal component:")
        for i in range(len(best_features)):
            percentage = pca.explained_variance_ratio_[i] * 100
            print(f"{best_features[i]}\t-\t{percentage:.2f}%")
    else:
        print("No valid features found for PCA.")

    # Also show comparison with label encoding
    print("\n\n---- PCA with Label Encoding (Original) ----\n")
    features_label = extract_features_with_encoding(
        data, 
        max_elements=10000, 
        use_target_encoding=False
    )
    
    if features_label.size > 0:
        pca_label = PCA(n_components=components)
        transformed_label = pca_label.fit_transform(features_label)
        
        best_features_label = [feature_names[i] for i in pca_label.components_[0].argsort()[-components:][::-1]]
        print("Best features and contribution to first principal component:")
        for i in range(len(best_features_label)):
            percentage = pca_label.explained_variance_ratio_[i] * 100
            print(f"{best_features_label[i]}\t-\t{percentage:.2f}%")


    print("\n---- End of Summary ----\n\n")