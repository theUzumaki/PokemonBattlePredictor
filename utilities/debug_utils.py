"""
Debug and information display utilities for Pokemon data.

This module provides functions for displaying Pokemon information
in a formatted manner for debugging and analysis.
"""

from typing import List, Dict, Any
from utilities.logger import log


def dataset_info(
    data: Dict[str, Dict[str, Any]],
    feature_names: List[str],
    entry_count: Dict[str, int],
) -> None:
    """
    Display dataset information in a formatted table.
    
    Args:
        data: Dictionary of battle records
        feature_names: List of feature names to display
        entry_count: Dictionary mapping entry identifiers to occurrence counts
    """

    local_features = sorted(feature_names, key=lambda x: (x.startswith("type_"), x))

    log("ID\t\t\t", color='yellow', end=" ")
    for key in local_features:
        if key.startswith("type_"):
            continue
        else:
            log(f"{key.upper():<3}\t", color='yellow', end=" ")
    log("\n")

    for entry_key, entry_value in data.items():

        # If no name field, just display the entry
        count = entry_count.get(entry_key, 0)
        log(f"{entry_key[:12]} ({count})\t", color='cyan', end=" ")
        
        for key in local_features:
            
            if key.startswith("type_"):
                type_ = key.split("_", 1)[1]
                if "types" in entry_value and type_ in entry_value["types"]:
                    log(f"{type_:3}", color='cyan', end=" ")
                else:
                    log(f"{type_:3}", color='gray', end=" ")
            else:
                # Try to get the value with base_ prefix first, then without
                value = entry_value.get(f'base_{key}', entry_value.get(key, 'N/A'))
                log(f"{value:<3}\t", color='cyan', end=" ")
        log()
