"""
Simplified training module for Pokemon Battle Predictor.
Uses Linear Regression on PCA-transformed features.
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import pickle

import variables as v
import battleline_extractor as be
from load_into_pca import perform_pca_on_battles
import chronicle.logger as logger


def get_labels_from_battleline(battleline: v.battleline):
    """Extract win/loss labels from battleline."""
    labels = []
    for battle_id, battle in battleline.battles.items():
        labels.append(battle.win)
    return np.array(labels)


def train_model(
    battleline: v.battleline,
    n_components: int = 10,
    use_ridge: bool = True,
    alpha: float = 1.0,
    test_size: float = 0.2
):
    """
    Train a linear regression model on PCA features.
    
    Args:
        battleline: The battleline struct with battle data
        n_components: Number of PCA components (default: 10)
        use_ridge: Use Ridge regression instead of standard (default: True)
        alpha: Regularization strength for Ridge (default: 1.0)
        test_size: Fraction of data for testing (default: 0.2)
        
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
        use_individual_pokemon=False
    )
    logger.log(1, 0, 1, logger.Colors.INFO, f"Feature shape: {pca_features.shape}")
    
    # Step 3: Split data
    logger.log_step(3, 4, f"Splitting data (test={test_size*100:.0f}%)")
    X_train, X_test, y_train, y_test = train_test_split(
        pca_features, labels, test_size=test_size, random_state=42
    )
    logger.log(1, 0, 1, logger.Colors.INFO, f"Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Step 4: Train model
    logger.log_step(4, 4, "Training model")
    if use_ridge:
        model = Ridge(alpha=alpha)
        logger.log(1, 0, 1, logger.Colors.INFO, f"Model: Ridge Regression (alpha={alpha})")
    else:
        model = LinearRegression()
        logger.log(1, 0, 1, logger.Colors.INFO, "Model: Linear Regression")
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = (model.predict(X_train) >= 0.5).astype(int)
    test_pred = (model.predict(X_test) >= 0.5).astype(int)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    logger.log_section_header("RESULTS")
    logger.log(0, 0, 0, logger.Colors.BRIGHT_GREEN + logger.Colors.BOLD, f"Train Accuracy: {train_acc:.4f} ({train_acc*100:.2f}%)")
    logger.log(0, 0, 1, logger.Colors.BRIGHT_GREEN + logger.Colors.BOLD, f"Test Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    return {
        'model': model,
        'pca_model': pca_model,
        'scaler': scaler,
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
    
    # Predict
    predictions = model_package['model'].predict(features_pca)
    binary_preds = (predictions >= threshold).astype(int)
    
    # Map to battle IDs
    results = {}
    for i, battle_id in enumerate(battleline.battles.keys()):
        results[battle_id] = {
            'probability': predictions[i],
            'prediction': binary_preds[i]
        }
    
    return results


if __name__ == "__main__":
    from test_battleline_struct import example_battleline
    
    logger.log_info("Training on example data...", newline_before=1)
    
    # Train model
    result = train_model(
        battleline=example_battleline,
        n_components=10,
        use_ridge=True,
        alpha=1.0,
        test_size=0.2
    )
    
    # Save model
    save_model(result, 'models/trained_model.pkl')

    logger.log_final_result(True, "Training complete!")
