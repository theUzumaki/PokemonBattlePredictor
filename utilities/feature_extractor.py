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
    use_sample_weights: bool = False,
    fields_to_look_into: List[str] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extract features from Pokemon data with optional target encoding for names.
    
    Args:
        data: List of battle records
        feature_names: List of feature names to extract
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
    for entry in data:
        i += 1

        # Navigate through the nested fields
        current_data = entry
        field_found = True
        
        for field in fields_to_look_into:
            if isinstance(current_data, dict) and field in current_data:
                current_data = current_data[field]
            else:
                field_found = False
                break

        if not field_found:
            continue

        for sub_entry in current_data:
            for key in feature_names:

                if key.startswith("type_"):
                    type_ = key.split("_", 1)[1]
                    if type_ in sub_entry["types"]:
                        features[f"type_{type_}"].append(1)
                    else:
                        features[f"type_{type_}"].append(0)
                else:
                    features[key].append(sub_entry[f"{key}"])
            
            # Extract sample weight if requested
            if use_sample_weights:
                sample_weights.append(sub_entry.get("sample_weight", 1.0))

    features_array = np.column_stack([features[key] for key in features])
    
    if use_sample_weights:
        return features_array, np.array(sample_weights)
    else:
        return features_array
