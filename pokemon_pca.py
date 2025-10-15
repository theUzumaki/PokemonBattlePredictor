import utilities.data_parser as data_parser
from utilities.logger import log
from typing import List, Dict, Any
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict

DEBUG = True
pokemon_species = []
pokemon_species_count = defaultdict(int)  # Counter for Pokémon species occurrences
feature_names = ["atk", "hp", "def", "types", "spa", "spd", "spe"]
types_list = ["grass", "psychic", "fire", "water", "electric", "ice", "fighting",
                "poison", "ground", "flying", "bug", "rock", "ghost", "dragon", "dark",
                "steel", "fairy", "normal"]
partial_types_list = ["grass", "psychic", "fire", "water", "electric", "ice",
                "poison", "ground", "flying", "rock", "ghost", "dragon",
                "normal"]

def pokemon_info():
    local_pokemon_set = set()

    local_features = sorted(feature_names, key=lambda x: (x.startswith("type_"), x))

    log("NAME\t\t\t", color='yellow', end=" ")
    for key in local_features:
        if key.startswith("type_"):
            continue
        else:
            log(f"{key.upper():<3}\t", color='yellow', end=" ")
    log("\n")

    for record in data:
        for pokemon in record["p1_team_details"]:
            if pokemon["name"] in local_pokemon_set:
                continue
            local_pokemon_set.add(pokemon["name"])

            log(f"{pokemon['name']:12} ({pokemon_species_count.get(pokemon['name'], 0)})\t", color='cyan', end=" ")
            for key in local_features:
                if key == "name":
                    continue
                if key.startswith("type_"):
                    type_ = key.split("_", 1)[1]
                    if type_ in pokemon["types"]:
                        log(f"{type_:3}", color='cyan', end=" ")
                    else:
                        log(f"{type_:3}", color='gray', end=" ")
                else:
                    log(f"{pokemon[f'base_{key}']:<3}\t", color='cyan', end=" ")
            log()


# Extract numerical data with optional target encoding
def extract_features_with_encoding(
    data: List[Dict[str, Any]], 
    max_elements: int = float("inf"),
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
                format_key = f"base_{key}"

                if key.startswith("type_"):
                    type_ = key.split("_", 1)[1]
                    if type_ in pokemon["types"]:
                        features[f"type_{type_}"].append(1)
                    else:
                        features[f"type_{type_}"].append(0)
                else:
                    features[key].append(pokemon[f"{format_key}"])
    return np.column_stack([features[key] for key in features])

if __name__ == "__main__":

    if "types" in feature_names:
        for type_ in partial_types_list:
            feature_names.append(f"type_{type_}")
        feature_names.remove("types")
    np.random.shuffle(feature_names)

    # quick check when running the module directly
    data, pokemon_species_count = list(data_parser.iter_test_data())
    components = 4 if len(feature_names) > 7 else len(feature_names) - 1

    log("\n\n---- Data Summary ----\n", color='magenta')
    log(f"Loaded {len(data)} records from data/train.jsonl", color='magenta')

    if DEBUG:
        log("\n---- Debug: Pokemon Records ----\n", color='cyan')
        pokemon_info()
        log("\n---- End of Debug ----\n", color='cyan')

    # Perform PCA
    log("\n\n---- PCA Analysis ----\n", color='blue')
    features_encoded = extract_features_with_encoding( data, max_elements=10000 )

    features_encoded = features_encoded.astype(float)
    if features_encoded.size > 0:
        # Normalize features except for name and type encodings
        for col in range(features_encoded.shape[1]):

            if feature_names[col] == "name" or feature_names[col].startswith("type_"):
                continue  # Skip normalization for name and type encodings

            min_val = 1
            max_val = 255
            for row in range(features_encoded.shape[0]):
                features_encoded[row, col] = (features_encoded[row, col] - min_val) / (max_val - min_val)

        pca = PCA(n_components=components)
        transformed = pca.fit_transform(features_encoded)

        for i in range(components):
            # Retrieve the names of the best features
            best_features = [feature_names[h] for h in pca.components_[i].argsort()[-components:][::-1]]
            log(f"Best features and contribution to principal component {i + 1}:", color='blue')
            for j in range(len(best_features)):
                percentage = pca.explained_variance_ratio_[j] * 100
                feature_color = "green" if percentage >= 2 else "red"

                log(f"{best_features[j]:<15}\t-", color='blue', tabs=1, end="\t")
                log(f"{percentage:.2f}%", color=feature_color)
    else:
        log("No valid features found for PCA.", color='red')

    log("\n\n---- End of Summary ----\n\n", color='magenta')