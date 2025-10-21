"""
Normalization and weighting utilities for Pokemon species data.

This module provides functions for normalizing Pokemon species counts
and applying weights to data for class balancing.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from utilities.logger import log


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
    """
    Apply species weights to Pokemon data records.
    
    Args:
        data: List of battle records
        normalized_weights: Dictionary mapping Pokemon names to weights
        weight_key: Key to store the weight in the Pokemon dictionary
    
    Returns:
        Modified data with weights applied
    """
    for record in data:
        if "p1_team_details" in record:
            for pokemon in record["p1_team_details"]:
                pokemon_name = pokemon.get("name", "")
                pokemon[weight_key] = normalized_weights.get(pokemon_name, 1.0)
    
    return data


def weight_handling(
    data: List[Dict[str, Any]], 
    pokemon_species_count: Dict[str, int],
    weighting_method: str = "sqrt_inverse"
) -> List[Dict[str, Any]]:
    """
    Handle species weighting with logging and comparison of methods.
    
    Args:
        data: List of battle records
        pokemon_species_count: Dictionary mapping Pokemon names to occurrence counts
        weighting_method: Normalization method to use
    
    Returns:
        Data with weights applied
    """
    log("\n\n---- Species Normalization ----\n", color='yellow')
    log(f"Using weighting method: {weighting_method}\n", color='yellow')
    
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
        method=weighting_method
    )
    
    # Apply weights to the data
    log("\n\nApplying weights to data samples...", color='yellow')
    return apply_species_weights_to_data(data, normalized_weights)
