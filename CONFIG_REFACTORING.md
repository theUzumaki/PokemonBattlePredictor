# Configuration Refactoring Summary

## Overview
Refactored the Pokemon PCA analysis code to use a centralized configuration file (`configs/pkmn.json`) for all configurable parameters, including dynamic field path navigation.

## Changes Made

### 1. Configuration File (`configs/pkmn.json`)
Created a new configuration file with the following parameters:

```json
{
  "feature_names": ["atk", "hp", "def", "types", "spa", "spd", "spe"],
  "types_list": [...],
  "partial_types_list": [...],
  "debug": false,
  "second_team": false,
  "use_species_weighting": true,
  "weighting_method": "sqrt_inverse",
  "coverage": 0.9,
  "pca_components": 10,
  "entry_to_print": 0,
  "fields_to_look_into": ["p1_team_details"]
}
```

### 2. Data Parser (`utilities/data_parser.py`)
**Key Changes:**
- Added `fields_to_look_into: List[str]` parameter to `iter_test_data()` function
- Implemented dynamic field path navigation through nested JSON structures
- Supports traversing multiple levels (e.g., `["Year", "Foods", "fruits"]`)
- Maintains backward compatibility with default value `["p1_team_details"]`

**Example Usage:**
```python
# To access: Year -> Foods -> fruits
fields_to_look_into = ["Year", "Foods", "fruits"]
data, counts = iter_test_data(fields_to_look_into=fields_to_look_into)
```

### 3. Debug Utils (`utilities/debug_utils.py`)
**Key Changes:**
- Added `fields_to_look_into: List[str]` parameter to `pokemon_info()` function
- Implemented same dynamic field path navigation as data_parser
- Enhanced robustness:
  - Checks if dict entries exist before accessing
  - Handles cases with or without "name" field
  - Falls back to generic "Entry" label when name is not available
  - Tries both `base_{key}` and `{key}` for field access

### 4. Main Script (`pokemon_pca.py`)
**Key Changes:**
- Added config loading system with `load_config()` function
- All configuration parameters now read from `configs/pkmn.json`
- Added `FIELDS_TO_LOOK_INTO` variable from config
- Updated function calls to pass `fields_to_look_into` parameter:
  - `data_parser.iter_test_data()`
  - `pokemon_info()`
- Added logging to show field path being used

## Benefits

1. **Flexibility**: Can now analyze different nested field structures without code changes
2. **Maintainability**: All configuration in one place
3. **Reusability**: Easy to create different config files for different analysis scenarios
4. **Documentation**: Config file serves as clear documentation of analysis parameters

## Usage Examples

### Example 1: Default Pokemon Analysis
```json
{
  "fields_to_look_into": ["p1_team_details"]
}
```

### Example 2: Nested Structure
```json
{
  "fields_to_look_into": ["game_data", "teams", "player1"]
}
```

### Example 3: Deep Nesting
```json
{
  "fields_to_look_into": ["Year", "Foods", "fruits"]
}
```

## Backward Compatibility
All parameters have sensible defaults, so existing code will continue to work without modification.
