# K-Fold Target Encoding Implementation

## Overview
This implementation adds k-fold target encoding for Pokemon names in the `pokemon_pca.py` script. Target encoding is a technique that replaces categorical variables (Pokemon names) with the mean of the target variable (win rate) for that category.

## What is K-Fold Target Encoding?

K-fold target encoding is a method to encode categorical features using the target variable while avoiding data leakage. It works as follows:

1. **Split the data** into K folds (default: 5 folds)
2. **For each fold**:
   - Use the other K-1 folds as training data
   - Calculate the mean target value for each category in the training folds
   - Apply this encoding to the validation fold
3. **Average** the encodings across all folds to get the final encoding for each category

## Key Features

### Function: `kfold_target_encode_pokemon()`
- **Purpose**: Generates target encodings for Pokemon names based on win rates
- **Parameters**:
  - `data`: List of battle records from JSONL files
  - `n_splits`: Number of folds for cross-validation (default: 5)
  - `random_state`: Random seed for reproducibility (default: 42)
- **Returns**: Dictionary mapping Pokemon names to their encoded values (mean win rate)

### Function: `extract_features_with_encoding()`
- **Purpose**: Extract features with optional target encoding
- **Parameters**:
  - `data`: List of battle records
  - `max_elements`: Maximum number of records to process
  - `use_target_encoding`: Boolean to enable/disable target encoding
  - `target_encoding_dict`: Dictionary of pre-computed encodings
- **Returns**: NumPy array of features

## Results

The implementation shows:

1. **Top performing Pokemon** (highest win rates):
   - Cloyster: 57.10% win rate
   - Jynx: 54.63% win rate
   - Articuno: 54.47% win rate

2. **Bottom performing Pokemon** (lowest win rates):
   - Dragonite: 35.65% win rate
   - Charizard: 36.46% win rate
   - Slowbro: 38.90% win rate

3. **PCA Analysis**: 
   - With target encoding: Defense contributes 90.66%, Attack 9.34%
   - With label encoding: Defense contributes 90.15%, Attack 9.30%

## Advantages of K-Fold Target Encoding

1. **Prevents overfitting**: By using out-of-fold predictions
2. **Captures information**: Encodes the relationship between category and target
3. **Numerical representation**: Converts categorical variables to meaningful numbers
4. **Smooth encodings**: Averages across folds reduce variance

## Usage Example

```python
# Load data
data = list(iter_test_data())

# Perform k-fold target encoding
target_encoding_dict = kfold_target_encode_pokemon(data, n_splits=5, random_state=42)

# Extract features with target encoding
features = extract_features_with_encoding(
    data, 
    max_elements=10000, 
    use_target_encoding=True,
    target_encoding_dict=target_encoding_dict
)

# Use features for machine learning models
# ...
```

## Dependencies

- `pandas`: For data manipulation during encoding
- `scikit-learn`: For KFold cross-validation and PCA
- `numpy`: For numerical operations

## Notes

- The global mean is used as a fallback for Pokemon not seen in training folds
- The encoding preserves the order of magnitude of the target variable (0 to 1 for binary classification)
- This approach is particularly useful for high-cardinality categorical features
