"""
Debug and information display utilities for Pokemon data.

This module provides functions for displaying Pokemon information
in a formatted manner for debugging and analysis.
"""

from typing import List, Dict, Any
from utilities.logger import log


def pokemon_info(
    data: List[Dict[str, Any]],
    feature_names: List[str],
    pokemon_species_count: Dict[str, int],
    fields_to_look_into: List[str] = None
) -> None:
    """
    Display Pokemon information in a formatted table.
    
    Args:
        data: List of battle records
        feature_names: List of feature names to display
        pokemon_species_count: Dictionary mapping Pokemon names to occurrence counts
        fields_to_look_into: List of nested field names to traverse (e.g., ["p1_team_details"])
    """
    if fields_to_look_into is None:
        fields_to_look_into = ["p1_team_details"]
    
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
        # Navigate through the nested fields
        current_data = record
        field_found = True
        
        for field in fields_to_look_into:
            if isinstance(current_data, dict) and field in current_data:
                current_data = current_data[field]
            else:
                field_found = False
                break
        
        # Process the data at the target field
        if not field_found or not isinstance(current_data, list):
            continue
            
        for pokemon in current_data:
            if not isinstance(pokemon, dict):
                continue
                
            # Check if pokemon has a name field for deduplication
            if "name" in pokemon:
                if pokemon["name"] in local_pokemon_set:
                    continue
                local_pokemon_set.add(pokemon["name"])
                
                # Create a key for looking up count (same as in data_parser)
                count_key = ''.join(str(value) for value in pokemon.values())
                count = pokemon_species_count.get(count_key, 0)

                log(f"{pokemon['name']:12} ({count})\t", color='cyan', end=" ")
            else:
                # If no name field, just display the entry
                count_key = ''.join(str(value) for value in pokemon.values())
                count = pokemon_species_count.get(count_key, 0)
                log(f"{'Entry':12} ({count})\t", color='cyan', end=" ")
            
            for key in local_features:
                if key == "name":
                    continue
                if key.startswith("type_"):
                    type_ = key.split("_", 1)[1]
                    if "types" in pokemon and type_ in pokemon["types"]:
                        log(f"{type_:3}", color='cyan', end=" ")
                    else:
                        log(f"{type_:3}", color='gray', end=" ")
                else:
                    # Try to get the value with base_ prefix first, then without
                    value = pokemon.get(f'base_{key}', pokemon.get(key, 'N/A'))
                    log(f"{value:<3}\t", color='cyan', end=" ")
            log()
