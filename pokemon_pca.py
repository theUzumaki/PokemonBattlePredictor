import utilities.data_parser as data_parser
from utilities.logger import log
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict

DEBUG = True
SECOND_TEAM = False  # Whether to include the second team's Pokémon in feature extraction
USE_SPECIES_WEIGHTING = True  # Enable/disable species-based weighting for class balancing
WEIGHTING_METHOD = "sqrt_inverse"  # Method: "inverse_frequency", "sqrt_inverse", "log_inverse", "min_max", "z_score"

pokemon_species = []
pokemon_species_count = defaultdict(int)  # Counter for Pokémon species occurrences
feature_names = ["atk", "hp", "def", "types", "spa", "spd", "spe"]
types_list = ["grass", "psychic", "fire", "water", "electric", "ice", "fighting",
                "poison", "ground", "flying", "bug", "rock", "ghost", "dragon", "dark",
                "steel", "fairy", "normal"]
partial_types_list = ["grass", "psychic", "fire", "water", "electric", "ice",
                "poison", "ground", "flying", "rock", "ghost", "dragon",
                "normal"]


def normalize_species_counts(
    species_count: Dict[str, int],
    method: str = "inverse_frequency"
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Normalize Pokemon species counts to apply class balancing.
    
    This function helps address class imbalance by computing weights for each Pokemon species.
    Rare species receive higher weights, while common species receive lower weights.
    
    Example usage:
        >>> weights, freqs = normalize_species_counts(pokemon_species_count)
        >>> # Use weights when sampling or training:
        >>> sample_weight = weights[pokemon_name]
        >>> # Or use frequencies for statistical analysis:
        >>> probability = freqs[pokemon_name]
    
    Args:
        species_count: Dictionary mapping Pokemon names to their occurrence counts
        method: Normalization method to use:
            - "inverse_frequency": Weight inversely proportional to frequency (1/count)
              Best for: Strong class balancing, giving rare classes high importance
            - "sqrt_inverse": Square root of inverse frequency (1/sqrt(count))
              Best for: Moderate balancing, less aggressive than inverse
            - "log_inverse": Logarithmic inverse (1/log(count+1))
              Best for: Mild balancing, preserves some of the original distribution
            - "min_max": Min-max normalization to [0, 1] range
              Best for: Scaling counts to a fixed range for visualization
            - "z_score": Z-score normalization (standardization)
              Best for: Statistical analysis, centering around mean
    
    Returns:
        Tuple of (normalized_weights, normalized_frequencies)
        - normalized_weights: Dictionary mapping Pokemon names to balancing weights
          (Higher weight = rarer species, should be given more importance)
        - normalized_frequencies: Dictionary mapping Pokemon names to normalized frequencies
          (Frequency = proportion of total, sums to 1.0)
    """
    
    if not species_count:
        return {}, {}
    
    # Calculate total count and statistics
    counts = np.array(list(species_count.values()))
    total_count = counts.sum()
    mean_count = counts.mean()
    std_count = counts.std()
    min_count = counts.min()
    max_count = counts.max()
    
    normalized_weights = {}
    normalized_frequencies = {}
    
    for species_name, count in species_count.items():
        # Calculate normalized frequency (probability)
        normalized_frequencies[species_name] = count / total_count
        
        # Calculate weight based on chosen method
        if method == "inverse_frequency":
            # Inverse frequency: rare species get higher weights
            normalized_weights[species_name] = 1.0 / count
            
        elif method == "sqrt_inverse":
            # Square root of inverse frequency: less aggressive than inverse
            normalized_weights[species_name] = 1.0 / np.sqrt(count)
            
        elif method == "log_inverse":
            # Logarithmic inverse: even less aggressive
            normalized_weights[species_name] = 1.0 / np.log(count + 1)
            
        elif method == "min_max":
            # Min-max normalization: scales to [0, 1]
            if max_count == min_count:
                normalized_weights[species_name] = 1.0
            else:
                normalized_weights[species_name] = (count - min_count) / (max_count - min_count)
                
        elif method == "z_score":
            # Z-score normalization: mean=0, std=1
            if std_count == 0:
                normalized_weights[species_name] = 0.0
            else:
                normalized_weights[species_name] = (count - mean_count) / std_count
                
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    # Normalize weights to sum to the number of unique species (optional, for balance)
    if method in ["inverse_frequency", "sqrt_inverse", "log_inverse"]:
        weight_sum = sum(normalized_weights.values())
        num_species = len(species_count)
        for species_name in normalized_weights:
            normalized_weights[species_name] = (normalized_weights[species_name] / weight_sum) * num_species
    
    return normalized_weights, normalized_frequencies


def apply_species_weights_to_data(
    data: List[Dict[str, Any]],
    normalized_weights: Dict[str, float],
    weight_key: str = "sample_weight"
) -> List[Dict[str, Any]]:
    
    for record in data:
        if "p1_team_details" in record:
            for pokemon in record["p1_team_details"]:
                pokemon_name = pokemon.get("name", "")
                pokemon[weight_key] = normalized_weights.get(pokemon_name, 1.0)
    
    return data


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

def extract_features_with_encoding(
    data: List[Dict[str, Any]], 
    max_elements: int = float("inf"),
    target_encoding_dict: Dict[str, float] = None,
    use_sample_weights: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from Pokemon data with optional target encoding for names.
    
    Args:
        data: List of battle records
        max_elements: Maximum number of records to process
        target_encoding_dict: Dictionary mapping Pokemon names to encoded values
        use_sample_weights: Whether to extract and return sample weights
    
    Returns:
        If use_sample_weights is False:
            NumPy array of features
        If use_sample_weights is True:
            Tuple of (features array, sample weights array)
    """

    features = {}
    for feat in feature_names:
        features[feat] = []
    
    sample_weights = [] if use_sample_weights else None
        
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
            
            # Extract sample weight if requested
            if use_sample_weights:
                sample_weights.append(pokemon.get("sample_weight", 1.0))

        if SECOND_TEAM:
            pokemon = record["p2_lead_details"]
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
            
            # Extract sample weight if requested
            if use_sample_weights:
                sample_weights.append(pokemon.get("sample_weight", 1.0))
    
    features_array = np.column_stack([features[key] for key in features])
    
    if use_sample_weights:
        return features_array, np.array(sample_weights)
    else:
        return features_array

if __name__ == "__main__":

    if "types" in feature_names:
        for type_ in partial_types_list:
            feature_names.append(f"type_{type_}")
        feature_names.remove("types")
    np.random.shuffle(feature_names)

    # quick check when running the module directly
    data, pokemon_species_count = list(data_parser.iter_test_data(SECOND_TEAM=SECOND_TEAM))
    components = 7 if len(feature_names) > 7 else len(feature_names) - 1


    log("\n\n---- Data Summary ----\n", color='magenta')
    total_pokemon_count = 0
    for name, count in pokemon_species_count.items():
        total_pokemon_count += count
    log(f"Total Pokemon Species Count: {total_pokemon_count}", color='magenta')
    log(f"Unique Pokemon Species: {len(pokemon_species_count)}", color='magenta')
    log(f"Loaded {len(data)} records from data/train.jsonl", color='magenta')
    log(f"Species Weighting: {'ENABLED' if USE_SPECIES_WEIGHTING else 'DISABLED'}", 
        color='green' if USE_SPECIES_WEIGHTING else 'red')
    
    # Initialize weights
    normalized_weights = None
    normalized_frequencies = None
    
    # Apply species normalization if enabled
    if USE_SPECIES_WEIGHTING:
        log("\n\n---- Species Normalization ----\n", color='yellow')
        log(f"Using weighting method: {WEIGHTING_METHOD}\n", color='yellow')
        
        # Compare different normalization methods
        normalization_methods = ["inverse_frequency", "sqrt_inverse", "log_inverse"]
        log("Comparing normalization methods for most/least common Pokemon:\n", color='yellow')
        
        # Sort species by frequency (most common first)
        sorted_species = sorted(pokemon_species_count.items(), key=lambda x: x[1], reverse=True)
        most_common = sorted_species[0]
        least_common = sorted_species[-1]
        
        log(f"Most common:  {most_common[0]:<15} (count: {most_common[1]})", color='cyan')
        log(f"Least common: {least_common[0]:<15} (count: {least_common[1]})", color='cyan')
        log(f"\n{'Method':<20} {'Most Common Weight':<20} {'Least Common Weight':<20} {'Ratio':<10}", color='yellow')
        
        for method in normalization_methods:
            weights, _ = normalize_species_counts(pokemon_species_count, method=method)
            most_weight = weights[most_common[0]]
            least_weight = weights[least_common[0]]
            ratio = least_weight / most_weight if most_weight > 0 else 0
            log(f"{method:<20} {most_weight:<20.4f} {least_weight:<20.4f} {ratio:<10.2f}x", color='cyan')
        
        # Use the configured method
        normalized_weights, normalized_frequencies = normalize_species_counts(
            pokemon_species_count, 
            method=WEIGHTING_METHOD
        )
        
        # Apply weights to the data
        log("\n\nApplying weights to data samples...", color='yellow')
        data = apply_species_weights_to_data(data, normalized_weights)
        
    else:
        log("\n\nSpecies weighting is DISABLED. Using uniform weights.", color='yellow')

    if DEBUG:
        log("\n---- Debug: Pokemon Records ----\n", color='cyan')
        pokemon_info()
        log("\n---- End of Debug ----\n", color='cyan')

    # Perform PCA
    log("\n\n---- PCA Analysis ----\n", color='blue')
    
    # Extract features with optional sample weights
    if USE_SPECIES_WEIGHTING:
        features_encoded, sample_weights = extract_features_with_encoding(
            data, max_elements=10000, use_sample_weights=True
        )
        log(f"Using weighted PCA with {len(sample_weights)} samples", color='blue')
        log(f"Weight statistics: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}\n", color='blue')
    else:
        features_encoded = extract_features_with_encoding(
            data, max_elements=10000, use_sample_weights=False
        )
        sample_weights = None
        log(f"Using unweighted PCA with {features_encoded.shape[0]} samples\n", color='blue')

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

        if USE_SPECIES_WEIGHTING and sample_weights is not None:
            log("Applying sample weights to PCA...\n", color='blue')
            weighted_features = features_encoded * np.sqrt(sample_weights[:, np.newaxis])
            transformed = pca.fit_transform(weighted_features)
        else:
            transformed = pca.fit_transform(features_encoded)

        for i in range(components):
            feature_contributions = zip(feature_names, pca.components_[i])
            sorted_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)

            log(f"PCA Component {i + 1}: Explained Variance = {pca.explained_variance_ratio_[i] * 100:.2f}%", color='blue')
            log(f"Feature relevance for principal component {i + 1}:", color='blue')

            max_features = 4
            for feature, contribution in sorted_features:
                if max_features <= 0:
                    pass
                max_features -= 1

                contribution_percentage = abs(contribution)
                feature_color = "green" if contribution_percentage >= 0.1 else "red"

                log(f"{feature:<15}\t-", color='blue', tabs=1, end="\t")
                log(f"{contribution_percentage:.2f}", color=feature_color)
        
            log("\n")
    else:
        log("No valid features found for PCA.", color='red')

    log("\n\n---- End of Summary ----\n\n", color='magenta')