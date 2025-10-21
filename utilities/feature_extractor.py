"""
Feature extraction utilities for Pokemon data.

This module provides functions for extracting and encoding features
from Pokemon battle data.
"""

from typing import List, Dict, Any, Tuple, Union
import numpy as np


def extract_features_with_encoding(
    data: List[Dict[str, Any]],
    feature_names: List[str],
    max_elements: int = float("inf"),
    target_encoding_dict: Dict[str, float] = None,
    use_sample_weights: bool = False,
    second_team: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract features from Pokemon data with optional target encoding for names.
    
    Args:
        data: List of battle records
        feature_names: List of feature names to extract
        max_elements: Maximum number of records to process
        target_encoding_dict: Dictionary mapping Pokemon names to encoded values
        use_sample_weights: Whether to extract and return sample weights
        second_team: Whether to include the second team's Pokemon in feature extraction
    
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

        if second_team:
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
