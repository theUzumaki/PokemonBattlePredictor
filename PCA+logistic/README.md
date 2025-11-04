# PCA + Logistic Regression Approach

This directory contains the original implementation of the Pokemon Battle Predictor using PCA (Principal Component Analysis) for dimensionality reduction and Logistic Regression for classification.

## Files in this directory:

### Core Training & Prediction
- **`train.py`**: Main training script using Logistic Regression on PCA-transformed features
- **`predict.py`**: Generate predictions from a trained PCA+LogReg model
- **`load_into_pca.py`**: Feature extraction and PCA transformation utilities
- **`pca.py`**: PCA helper functions and visualization tools

### Examples & Legacy
- **`example_usage.py`**: Example usage of the training pipeline
- **`old_pca/`**: Previous PCA implementations (legacy code)

## Pipeline Overview:

1. **Feature Extraction** (`load_into_pca.py`):
   - Extracts detailed Pokemon features (stats, moves, types, HP, boosts, status)
   - Creates high-dimensional feature vectors for each battle

2. **Preprocessing**:
   - StandardScaler normalization
   - PCA dimensionality reduction (default: 200 components)

3. **Classification** (`train.py`):
   - Logistic Regression with configurable regularization (C parameter)
   - Binary classification: predict battle winner (0/1)

4. **Prediction** (`predict.py`):
   - Load trained model, scaler, and PCA transformer
   - Apply same preprocessing pipeline to test data
   - Generate predictions with configurable threshold (default: 0.5)

## Usage:

### Training:
```bash
cd "PCA+logistic"
python train.py
```

### Prediction:
```bash
cd "PCA+logistic"
python predict.py
```

## Default Hyperparameters:
- **PCA Components**: 200
- **Logistic Regression C**: 1.0
- **Max Iterations**: 1000
- **Test Size**: 0.0 (use full dataset)
- **Threshold**: 0.5

## Model Outputs:
- Trained models saved to: `../models/model_N/trained_model.pkl`
- Predictions saved to: `../predictions/prediction_N/prediction.csv`

---

**Note**: This approach has been superseded by ensemble learning methods (Random Forests, Gradient Boosting, etc.) which may provide better performance without requiring PCA dimensionality reduction.
