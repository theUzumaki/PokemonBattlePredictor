"""
Simplified training module for Pokemon Battle Predictor.
Uses Logistic Regression on PCA-transformed features.
"""

import sys
from pathlib import Path

# Add parent directory to path for shared modules
sys.path.append(str(Path(__file__).parent.parent))
# Add current directory to path for local modules
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import json

import variables as v
import battleline_extractor as be
from load_into_pca import perform_pca_on_battles
import chronicle.logger as logger
from utilities.time_utils import utc_iso_now

# Global config: default test set size for train/test split.
# Set to 0.0 to use the full dataset for training (no test split).
TEST_SIZE = 0.0
# Global default number of PCA components to use during training
N_COMPONENTS = 200
# Global default threshold used to convert predicted probabilities to binary labels
THRESHOLD = 0.5

def get_labels_from_battleline(battleline: v.battleline):
    """Extract win/loss labels from battleline."""
    labels = []
    for battle_id, battle in battleline.battles.items():
        labels.append(battle.win)
    return np.array(labels)


def train_model(
    battleline: v.battleline,
    n_components: int = 10,
    C: float = 1.0,
    max_iter: int = 1000,
    test_size: float = 0.2,
    threshold: float = 0.5,
):
    """
    Train a logistic regression model on PCA features.
    
    Args:
        battleline: The battleline struct with battle data
        n_components: Number of PCA components (default: 10)
        C: Inverse of regularization strength (default: 1.0)
        max_iter: Maximum number of iterations for solver (default: 1000)
        test_size: Fraction of data for testing (default: 0.2)
        threshold: Threshold for converting probabilities to binary predictions (default: 0.5)
        
    Returns:
        Dictionary with model, metrics, and fitted transformers
    """
    logger.log_application_title("TRAINING POKEMON BATTLE PREDICTOR")
    
    # Step 1: Get labels
    logger.log_step(1, 4, "Extracting labels")
    labels = get_labels_from_battleline(battleline)
    logger.log(1, 0, 0, logger.Colors.INFO, f"Total battles: {len(labels)}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Wins: {np.sum(labels)}, Losses: {len(labels) - np.sum(labels)}")
    
    # Step 2: Apply PCA
    logger.log_step(2, 4, f"Applying PCA ({n_components} components)")
    pca_model, pca_features, scaler, _, feature_names = perform_pca_on_battles(
        battleline=battleline,
        n_components=n_components,
    )
    logger.log(1, 0, 1, logger.Colors.INFO, f"Feature shape: {pca_features.shape}")
    
    # Step 3: Split data (optionally skip test split by setting test_size <= 0)
    logger.log_step(3, 4, f"Splitting data (test={test_size*100:.0f}%)")
    if test_size is None:
        test_size = 0.0

    if test_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(
            pca_features, labels, test_size=test_size, random_state=42
        )
        logger.log(1, 0, 1, logger.Colors.INFO, f"Train: {len(X_train)}, Test: {len(X_test)}")
    else:
        # Use full dataset for training when test_size <= 0
        X_train = pca_features
        y_train = labels
        X_test = np.empty((0, pca_features.shape[1]))
        y_test = np.empty((0,))
        logger.log(1, 0, 1, logger.Colors.INFO, f"Train: {len(X_train)}, Test: {len(X_test)} (no test split)")
    
    # Step 4: Train model
    logger.log_step(4, 4, "Training model")
    model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
    logger.log(1, 0, 1, logger.Colors.INFO, f"Model: Logistic Regression (C={C}, max_iter={max_iter})")
    
    model.fit(X_train, y_train)
    
    # Evaluate using custom threshold
    train_proba = model.predict_proba(X_train)[:, 1]
    train_pred = (train_proba >= threshold).astype(int)
    train_acc = accuracy_score(y_train, train_pred)

    if X_test.shape[0] > 0:
        test_proba = model.predict_proba(X_test)[:, 1]
        test_pred = (test_proba >= threshold).astype(int)
        test_acc = accuracy_score(y_test, test_pred)
    else:
        test_acc = None
    
    logger.log_section_header("RESULTS")
    logger.log(0, 0, 0, logger.Colors.BRIGHT_GREEN + logger.Colors.BOLD, f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    if test_acc is not None:
        logger.log(0, 0, 1, logger.Colors.BRIGHT_GREEN + logger.Colors.BOLD, f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    else:
        logger.log(0, 0, 1, logger.Colors.BRIGHT_GREEN + logger.Colors.BOLD, "Test Accuracy:  N/A (no test split)")
    
    return {
        'model': model,
        'pca_model': pca_model,
        'scaler': scaler,
        'n_components': n_components,
        'C': C,
        'max_iter': max_iter,
        'test_size': test_size,
        'threshold': threshold,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }


def save_model(result, filepath='models/trained_model.pkl'):
    """Save the trained model and transformers."""
    import os
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    model_package = {
        'model': result['model'],
        'pca_model': result['pca_model'],
        'scaler': result['scaler']
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(model_package, f)
    
    logger.log_success(f"Model saved to: {filepath}", newline_before=1, newline_after=1)


def save_model_run(result, models_root: str = 'models', prefix: str = 'model'):
    """Save the model and a params.txt inside a sequential run folder under models_root.

    The function will create models_root/ if missing, then create models_root/{prefix}_{N}
    where N is the next integer sequence (starting at 1). Inside the run folder it
    writes 'trained_model.pkl' and 'params.txt' with run metadata.
    """
    from pathlib import Path

    # Get repository root (parent of PCA+logistic)
    repo_root = Path(__file__).parent.parent
    root = repo_root / "PCA+logistic" / models_root
    root.mkdir(parents=True, exist_ok=True)

    existing = [p.name for p in root.iterdir() if p.is_dir() and p.name.startswith(f"{prefix}_")]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[-1]))
        except Exception:
            continue
    next_n = max(nums) + 1 if nums else 1
    run_dir = root / f"{prefix}_{next_n}"
    run_dir.mkdir()

    # Save model package
    model_file = run_dir / 'trained_model.pkl'
    model_package = {
        'model': result['model'],
        'pca_model': result['pca_model'],
        'scaler': result['scaler']
    }
    with model_file.open('wb') as fh:
        pickle.dump(model_package, fh)

    # Prepare params
    total_battles = 0
    if 'y_train' in result and 'y_test' in result:
        total_battles = len(result['y_train']) + len(result['y_test'])

    params = [
        ('model_file', str(model_file)),
        ('n_components', str(result.get('n_components', ''))),
        ('C', str(result.get('C', ''))),
        ('max_iter', str(result.get('max_iter', ''))),
        ('test_size', str(result.get('test_size', ''))),
        ('prediction_threshold', str(result.get('threshold', ''))),
        ('train_accuracy', str(result.get('train_accuracy', ''))),
        ('test_accuracy', str(result.get('test_accuracy', ''))),
        ('num_battles', str(total_battles)),
        ('run_timestamp', utc_iso_now())
    ]

    params_file = run_dir / 'params.txt'
    with params_file.open('w', encoding='utf-8') as fh:
        for k, v in params:
            fh.write(f"{k}: {v}\n")

    logger.log_success(f"Model run saved to: {run_dir}", newline_before=1, newline_after=1)


def load_model(filepath='models/trained_model.pkl'):
    """Load a saved model and transformers."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def predict(battleline: v.battleline, model_package, threshold=0.5):
    """
    Predict battle outcomes using a trained model.
    
    Args:
        battleline: Battleline struct with battles to predict
        model_package: Loaded model package (from load_model)
        threshold: Prediction threshold (default: 0.5)
        
    Returns:
        Dictionary with predictions for each battle
    """
    from load_into_pca import extract_battle_features
    
    # Extract and transform features
    features = extract_battle_features(battleline, use_individual_pokemon=False)
    features_scaled = model_package['scaler'].transform(features)
    features_pca = model_package['pca_model'].transform(features_scaled)
    
    # Predict using predict_proba for logistic regression
    probabilities = model_package['model'].predict_proba(features_pca)[:, 1]
    binary_preds = (probabilities >= threshold).astype(int)
    
    # Map to battle IDs
    results = {}
    for i, battle_id in enumerate(battleline.battles.keys()):
        results[battle_id] = {
            'probability': probabilities[i],
            'prediction': binary_preds[i]
        }
    
    return results


if __name__ == "__main__":
    
    logger.log_info("Training on example data...", newline_before=1)

    # Get repository root (parent of PCA+logistic)
    from pathlib import Path
    repo_root = Path(__file__).parent.parent
    data_path = repo_root / "data" / "train.jsonl"

    train_data = []
    with open(data_path, 'r') as f:
        for line in f:
            # json.loads() parses one line (one JSON object) into a Python dictionary
            line = line.strip()
            if line:  # Skip empty lines
                train_data.append(json.loads(line))

    # Create battleline struct
    battleline_struct = be.create_final_turn_feature(train_data)
    
    # Train model
    result = train_model(
        battleline=battleline_struct,
        n_components=N_COMPONENTS,
        C=1.0,
        max_iter=1000,
        test_size=TEST_SIZE,
        threshold=THRESHOLD
    )
    
    # Save model into a sequential run folder under models/
    save_model_run(result, models_root='models', prefix='model')

    logger.log_final_result(True, "Training complete!")
