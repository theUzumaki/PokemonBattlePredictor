from sklearn.discriminant_analysis import StandardScaler
import utilities.data_parser as data_parser
from utilities.logger import log
from utilities.normalizer import normalize_species_counts, apply_species_weights_to_data, weight_handling
from utilities.feature_extractor import extract_features_with_encoding
from utilities.debug_utils import pokemon_info
from typing import List, Dict, Any, Tuple
from sklearn.decomposition import PCA
import numpy as np
from collections import defaultdict
import json
import os

# Load configuration from config file
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "pkmn.json")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config

# Load configuration
config = load_config(CONFIG_PATH)

# Extract configuration values
DEBUG = config.get("debug", False)
SECOND_TEAM = config.get("second_team", False)  # Whether to include the second team's Pokémon in feature extraction
USE_SPECIES_WEIGHTING = config.get("use_species_weighting", True)  # Enable/disable species-based weighting for class balancing
WEIGHTING_METHOD = config.get("weighting_method", "sqrt_inverse")  # Method: "inverse_frequency", "sqrt_inverse", "log_inverse", "min_max", "z_score"
COVERAGE = config.get("coverage", 0.9)  # Target cumulative explained variance coverage for PCA
ENTRY_TO_PRINT = config.get("entry_to_print", 0)  # Number of data entries to print for debugging
FIELDS_TO_LOOK_INTO = config.get("fields_to_look_into", ["p1_team_details"])  # Nested field path to extract Pokémon data from

entry = []
entry_count = defaultdict(int)  # Counter for Pokémon species occurrences
feature_names = config.get("feature_names", ["atk", "hp", "def", "types", "spa", "spd", "spe"])
types_list = config.get("types_list", ["grass", "psychic", "fire", "water", "electric", "ice", "fighting",
                "poison", "ground", "flying", "bug", "rock", "ghost", "dragon", "dark",
                "steel", "fairy", "normal"])
partial_types_list = config.get("partial_types_list", ["grass", "psychic", "fire", "water", "electric", "ice",
                "poison", "ground", "flying", "rock", "ghost", "dragon",
                "normal"])


if __name__ == "__main__":

    log(f"\n\nLoading configuration from: {CONFIG_PATH}", color='cyan')
    log(f"Features to analyze: {feature_names}", color='cyan')
    log(f"Field path for data extraction: {' -> '.join(FIELDS_TO_LOOK_INTO)}", color='cyan')
    log("")

    if "types" in feature_names:
        for type_ in partial_types_list:
            feature_names.append(f"type_{type_}")
        feature_names.remove("types")
    np.random.shuffle(feature_names)

    # quick check when running the module directly
    data, entry_count = list(data_parser.iter_test_data(
        entry_to_print=ENTRY_TO_PRINT, 
        SECOND_TEAM=SECOND_TEAM,
        fields_to_look_into=FIELDS_TO_LOOK_INTO
    ))
    components = config.get("pca_components", 10) if len(feature_names) > 7 else len(feature_names) - 1


    log("\n\n---- Data Summary ----\n", color='magenta')
    total_entry_count = 0
    for name, count in entry_count.items():
        total_entry_count += count
    log(f"Total Entry Count: {total_entry_count}", color='magenta')
    log(f"Unique Entry Count: {len(entry_count)}", color='magenta')
    log(f"Loaded {len(data)} records from data/train.jsonl", color='magenta')
    log(f"Species Weighting: {'ENABLED' if USE_SPECIES_WEIGHTING else 'DISABLED'}", 
        color='green' if USE_SPECIES_WEIGHTING else 'red')
    
    # Initialize weights
    normalized_weights = None
    normalized_frequencies = None
    
    # Apply species normalization if enabled
    if USE_SPECIES_WEIGHTING:
        data = weight_handling(data, entry_count, WEIGHTING_METHOD)
    else:
        log("\n\nSpecies weighting is DISABLED. Using uniform weights.", color='yellow')

    if DEBUG:
        log("\n---- Debug: Pokemon Records ----\n", color='cyan')
        pokemon_info(data, feature_names, entry_count, fields_to_look_into=FIELDS_TO_LOOK_INTO)
        log("\n---- End of Debug ----\n", color='cyan')

    # Perform PCA
    log("\n\n---- PCA Analysis ----\n", color='blue')
    
    # Extract features with optional sample weights
    if USE_SPECIES_WEIGHTING:
        features_encoded, sample_weights = extract_features_with_encoding(
            data, feature_names, max_elements=10000, use_sample_weights=True, second_team=SECOND_TEAM
        )
        log(f"Using weighted PCA with {len(sample_weights)} samples", color='blue')
        log(f"Weight statistics: min={sample_weights.min():.4f}, max={sample_weights.max():.4f}, mean={sample_weights.mean():.4f}\n", color='blue')
    else:
        features_encoded = extract_features_with_encoding(
            data, feature_names, max_elements=10000, use_sample_weights=False, second_team=SECOND_TEAM
        )
        sample_weights = None
        log(f"Using unweighted PCA with {features_encoded.shape[0]} samples\n", color='blue')

    features_encoded = features_encoded.astype(float)
    if features_encoded.size > 0:

        # Identify which features to standardize (exclude type encodings)
        continuous_mask = [not feat.startswith("type_") for feat in feature_names]
        type_mask = [feat.startswith("type_") for feat in feature_names]

        # Standardize continuous features (HP, Attack, Defense, etc.)
        log("Standardizing continuous features (mean centering + scaling)...", color='blue')
        scaler = StandardScaler()
        if (USE_SPECIES_WEIGHTING):
            weighted_features = features_encoded * np.sqrt(sample_weights[:, np.newaxis])
            features_encoded[:, continuous_mask] = scaler.fit_transform(
                features_encoded[:, continuous_mask],
                sample_weight=sample_weights
            )
        else:
            features_encoded[:, continuous_mask] = scaler.fit_transform(
                features_encoded[:, continuous_mask]
            )
        
        # Mean center the binary type features (optional but recommended for PCA)
        if np.any(type_mask):
            log("Standardizing binary type features (mean centering)...", color='blue')
            type_features = features_encoded[:, type_mask]
            type_means = np.mean(type_features, axis=0)
            features_encoded[:, type_mask] = type_features - type_means

        pca = PCA(n_components=components)

        if USE_SPECIES_WEIGHTING and sample_weights is not None:
            log("Applying sample weights to PCA...\n", color='blue')
            weighted_features = features_encoded * np.sqrt(sample_weights[:, np.newaxis])
            transformed = pca.fit_transform(weighted_features)
        else:
            transformed = pca.fit_transform(features_encoded)

        local_coverage = 0
        for i in range(components):
            if local_coverage >= COVERAGE:
                break

            feature_contributions = zip(feature_names, pca.components_[i])
            sorted_features = sorted(feature_contributions, key=lambda x: abs(x[1]), reverse=True)
            local_coverage += pca.explained_variance_ratio_[i]

            log(f"PCA Component {i + 1}: Explained Variance = {pca.explained_variance_ratio_[i] * 100:.2f}%", color='blue')
            log(f"Feature relevance for principal component {i + 1}:", color='blue')

            max_features = len(sorted_features)
            for feature, contribution in sorted_features:
                if max_features <= 0:
                    continue
                max_features -= 1

                contribution_percentage = abs(contribution)
                feature_color = "green" if contribution_percentage >= 0.1 else "red"

                if feature_color == "red":
                    break
                log(f"{feature:<15}\t-", color='blue', tabs=1, end="\t")
                log(f"{contribution_percentage:.2f}", color=feature_color)
        
            log("\n")
        log(f"\nTotal coverage: {local_coverage * 100:.2f}%", color='blue')
    else:
        log("No valid features found for PCA.", color='red')

    log("\n\n---- End of Summary ----\n\n", color='magenta')