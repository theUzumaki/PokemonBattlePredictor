# Random Forest & Ensemble Learning Approach

This directory contains the Random Forest and ensemble learning implementation for the Pokemon Battle Predictor project.

## Overview

Unlike the PCA + Logistic Regression approach, this method leverages tree-based ensemble models that can:
- Handle high-dimensional features without PCA
- Capture non-linear relationships automatically
- Provide feature importance rankings
- Resist overfitting through ensemble techniques

## Implemented Models

### 1. Random Forest
- **Description**: Ensemble of decision trees with bootstrap aggregating
- **Advantages**: 
  - Handles non-linear patterns naturally
  - Built-in feature importance
  - Robust to outliers and overfitting
  - No feature scaling required

### 2. Gradient Boosting (XGBoost)
- **Description**: Sequential ensemble that builds trees to correct previous errors
- **Advantages**:
  - Often achieves higher accuracy than Random Forest
  - Efficient handling of missing values
  - Built-in regularization

### 3. Ensemble Voting Classifier
- **Description**: Combines multiple models (Random Forest + XGBoost + Extra Trees)
- **Advantages**:
  - Leverages strengths of different algorithms
  - More robust predictions through voting
  - Often better generalization

## Pipeline

1. **Feature Extraction**: Reuses the same feature extraction from `load_into_pca.py`
   - Individual Pokemon features (HP, stats, boosts, types, moves, status, effects)
   - Adversary team features (alive count, types, HP)
   
2. **Normalization**: StandardScaler for consistency (optional for tree-based models)

3. **Model Training**: Multiple ensemble models with hyperparameter tuning

4. **Prediction**: Generate predictions using trained models

## Usage

### Training

```bash
python train.py
```

The script will:
- Load training data from `../data/train.jsonl`
- Extract features using shared utilities
- Train multiple ensemble models
- Save models to `models/model_N/`
- Display accuracy metrics

### Prediction

```bash
python predict.py
```

The script will:
- Load the latest trained model
- Process test data from `../data/test.jsonl`
- Generate predictions in CSV format
- Save to `predictions/prediction_N/prediction.csv`

## Model Parameters

Default hyperparameters can be adjusted in `train.py`:

- **Random Forest**:
  - `n_estimators`: 200 (number of trees)
  - `max_depth`: 20
  - `min_samples_split`: 5
  - `min_samples_leaf`: 2

- **XGBoost**:
  - `n_estimators`: 200
  - `max_depth`: 6
  - `learning_rate`: 0.1
  - `subsample`: 0.8

- **Ensemble**:
  - Soft voting (probability-based)
  - Equal weights for all models

## Directory Structure

```
RandomForests+Ensemble/
├── train.py              # Training script
├── predict.py            # Prediction script
├── README.md             # This file
├── models/               # Saved models
│   ├── model_1/
│   │   ├── trained_model.pkl
│   │   └── params.txt
│   └── model_2/
│       └── ...
└── predictions/          # Prediction outputs
    ├── prediction_1/
    │   ├── prediction.csv
    │   └── params.txt
    └── prediction_2/
        └── ...
```

## Feature Importance

After training, the models automatically compute feature importance scores, helping identify which Pokemon attributes and battle features most influence the outcome.

## Comparison with PCA + Logistic Regression

| Aspect | PCA + LogReg | Random Forest + Ensemble |
|--------|--------------|--------------------------|
| Feature Engineering | PCA dimensionality reduction | Direct feature usage |
| Linearity | Linear decision boundary | Non-linear patterns |
| Interpretability | Less interpretable | Feature importance available |
| Training Speed | Fast | Moderate to slow |
| Accuracy Potential | Good for linear patterns | Better for complex patterns |
| Overfitting Risk | Lower | Mitigated by ensemble |

## Expected Performance

Tree-based ensemble methods typically achieve **2-5% higher accuracy** than linear models on complex datasets with non-linear relationships, making them well-suited for Pokemon battle prediction where type matchups, stat interactions, and move combinations create complex decision boundaries.
