"""
Random Forest and Ensemble training module for Pokemon Battle Predictor.
Uses tree-based ensemble methods without PCA dimensionality reduction.
"""

import sys
from pathlib import Path

# Add parent directory to path for shared modules
sys.path.append(str(Path(__file__).parent.parent))
# Add current directory to path for local modules
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import pickle
import json

import variables as v
import battleline_extractor as be
from utilities.time_utils import utc_iso_now

# Import feature extraction from PCA module (reusing the same features)
sys.path.append(str(Path(__file__).parent.parent / "PCA+logistic"))
from extractor import extract_battle_features
from sklearn.preprocessing import StandardScaler

# Try importing XGBoost (optional but recommended)
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

# Try importing logger
try:
    import Chronicle.logger as logger
except ImportError:
    try:
        import chronicle.logger as logger
    except ImportError:
        # Fallback simple logger
        class SimpleLogger:
            @staticmethod
            def log_application_title(title):
                print(f"\n{'='*60}\n{title.center(60)}\n{'='*60}\n")
            
            @staticmethod
            def log_step(step, total, description):
                print(f"\n[Step {step}/{total}] {description}")
            
            @staticmethod
            def log(indent1, indent2, indent3, color, message):
                print("  " * (indent1 + indent2 + indent3) + message)
            
            @staticmethod
            def log_section_header(title):
                print(f"\n{'-'*60}\n{title}\n{'-'*60}")
            
            @staticmethod
            def log_success(message, newline_before=0, newline_after=0):
                print("\n" * newline_before + f"✓ {message}" + "\n" * newline_after)
            
            @staticmethod
            def log_info(message, newline_before=0):
                print("\n" * newline_before + f"ℹ {message}")
            
            @staticmethod
            def log_final_result(success, message):
                print(f"\n{'='*60}\n{'✓' if success else '✗'} {message}\n{'='*60}\n")
            
            class Colors:
                INFO = ""
                BRIGHT_GREEN = ""
                BOLD = ""
                DIM = ""
                YELLOW = ""
        
        logger = SimpleLogger()


# Global config
TEST_SIZE = 0.15  # Validation split (reduced from 0.2 to have more training data - Improvement #3)
THRESHOLD = 0.5  # Classification threshold
USE_FEATURE_SELECTION = True  # Enable feature selection based on importance (Improvement #2)
TOP_FEATURES = 300  # Number of top features to keep (reduced from 984 - Improvement #2)


def get_labels_from_battleline(battleline: v.battleline):
    """Extract win/loss labels from battleline."""
    labels = []
    for battle_id, battle in battleline.battles.items():
        labels.append(battle.win)
    return np.array(labels)


def select_top_features(X_train, y_train, X_test, n_features=300):
    """
    Select top N most important features using Random Forest feature importance.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        n_features: Number of top features to select
        
    Returns:
        X_train_selected, X_test_selected, feature_selector
    """
    logger.log(1, 0, 0, logger.Colors.INFO, f"Performing feature selection (top {n_features} features)...")
    
    # Train a quick Random Forest to get feature importance
    rf_selector = RandomForestClassifier(
        n_estimators=50,  # Quick training just for feature selection
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_selector.fit(X_train, y_train)
    
    # Get feature importances
    importances = rf_selector.feature_importances_
    
    # Select top N features
    indices = np.argsort(importances)[::-1][:n_features]
    
    # Transform data
    X_train_selected = X_train[:, indices]
    X_test_selected = X_test[:, indices] if X_test.shape[0] > 0 else X_test
    
    logger.log(1, 0, 1, logger.Colors.INFO, f"Feature reduction: {X_train.shape[1]} → {n_features}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Top feature importance: {importances[indices[0]]:.4f}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Lowest selected feature importance: {importances[indices[-1]]:.4f}")
    
    return X_train_selected, X_test_selected, indices


def train_ensemble(
    battleline: v.battleline,
    test_size: float = 0.15,
    threshold: float = 0.5,
    use_scaling: bool = True,
    use_feature_selection: bool = True,
    n_top_features: int = 300,
):
    """
    Train an ensemble of multiple classifiers (Random Forest, Extra Trees, XGBoost, Gradient Boosting).
    
    Args:
        battleline: The battleline struct with battle data
        test_size: Fraction of data for testing (default: 0.15)
        threshold: Threshold for converting probabilities to binary predictions (default: 0.5)
        use_scaling: Whether to apply StandardScaler (default: True)
        use_feature_selection: Whether to select top features (default: True)
        n_top_features: Number of top features to keep (default: 300)
        
    Returns:
        Dictionary with model, metrics, and transformers
    """
    logger.log_application_title("ENSEMBLE LEARNING - POKEMON BATTLE PREDICTOR")
    
    # Step 1: Get labels
    logger.log_step(1, 4, "Extracting labels")
    labels = get_labels_from_battleline(battleline)
    logger.log(1, 0, 0, logger.Colors.INFO, f"Total battles: {len(labels)}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Wins: {np.sum(labels)}, Losses: {len(labels) - np.sum(labels)}")
    
    # Step 2: Extract features
    logger.log_step(2, 4, "Extracting features")
    features = extract_battle_features(battleline, max_moves=4)
    logger.log(1, 0, 0, logger.Colors.INFO, f"Feature shape: {features.shape}")
    logger.log(1, 0, 1, logger.Colors.INFO, f"Features per battle: {features.shape[1]}")
    
    # Optional: Apply scaling
    scaler = None
    if use_scaling:
        logger.log(1, 0, 1, logger.Colors.INFO, "Applying StandardScaler")
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    
    # Step 3: Split data
    logger.log_step(3, 4, f"Splitting data (test={test_size*100:.0f}%)")
    if test_size > 0.0:
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=42
        )
        logger.log(1, 0, 1, logger.Colors.INFO, f"Train: {len(X_train)}, Test: {len(X_test)}")
    else:
        X_train = features
        y_train = labels
        X_test = np.empty((0, features.shape[1]))
        y_test = np.empty((0,))
        logger.log(1, 0, 1, logger.Colors.INFO, f"Train: {len(X_train)}, Test: {len(X_test)} (no test split)")
    
    # Step 3.5: Feature selection (Improvement #2)
    feature_indices = None
    if use_feature_selection:
        logger.log(1, 0, 1, logger.Colors.YELLOW, "=== IMPROVEMENT #2: Feature Selection ===")
        X_train, X_test, feature_indices = select_top_features(
            X_train, y_train, X_test, n_features=n_top_features
        )
    
    # Step 4: Train ensemble
    logger.log_step(4, 4, "Training Ensemble Models")
    logger.log(1, 0, 0, logger.Colors.YELLOW, "=== IMPROVEMENT #1: Increased Regularization ===")
    
    # Create individual estimators
    estimators = []
    
    # Random Forest - Strong regularization to prevent overfitting (Improvement #1)
    logger.log(1, 0, 0, logger.Colors.INFO, "Adding Random Forest to ensemble")
    rf = RandomForestClassifier(
        n_estimators=200,        # Increased for better averaging
        max_depth=10,            # Reduced from 12 to prevent deep memorization
        min_samples_split=15,    # Increased from 10 to require more samples
        min_samples_leaf=8,      # Increased from 5 to create larger leaves
        max_features='sqrt',     # Use sqrt of features for more randomness
        random_state=42,
        n_jobs=-1
    )
    estimators.append(('rf', rf))
    
    # Extra Trees - Strong regularization (Improvement #1)
    logger.log(1, 0, 0, logger.Colors.INFO, "Adding Extra Trees to ensemble")
    et = ExtraTreesClassifier(
        n_estimators=200,        # Increased for better averaging
        max_depth=10,            # Reduced from 12
        min_samples_split=15,    # Increased from 10
        min_samples_leaf=8,      # Increased from 5
        max_features='sqrt',     # Use sqrt of features
        random_state=42,
        n_jobs=-1
    )
    estimators.append(('et', et))
    
    # Gradient Boosting - More conservative learning (Improvement #1)
    logger.log(1, 0, 0, logger.Colors.INFO, "Adding Gradient Boosting to ensemble")
    gb = GradientBoostingClassifier(
        n_estimators=100,        # Keep at 100
        max_depth=4,             # Reduced from 5 for shallower trees
        learning_rate=0.03,      # Reduced from 0.05 for slower, more careful learning
        subsample=0.65,          # Reduced from 0.7 for more regularization
        min_samples_split=15,    # Increased for more constraint
        min_samples_leaf=8,      # Increased for more constraint
        random_state=42
    )
    estimators.append(('gb', gb))
    
    # XGBoost (if available) - Strong regularization (Improvement #1)
    if XGBOOST_AVAILABLE:
        logger.log(1, 0, 0, logger.Colors.INFO, "Adding XGBoost to ensemble")
        xgb_model = xgb.XGBClassifier(
            n_estimators=150,        # Keep at 150
            max_depth=4,             # Reduced from 5 for shallower trees
            learning_rate=0.03,      # Reduced from 0.05
            subsample=0.65,          # Reduced from 0.7
            colsample_bytree=0.7,    # Reduced from 0.8: use 70% of features per tree
            min_child_weight=8,      # Increased from 5: minimum samples in leaf
            gamma=0.2,               # Increased from 0.1: minimum loss reduction for split
            reg_alpha=0.3,           # Increased from 0.1: L1 regularization
            reg_lambda=2.0,          # Increased from 1.0: L2 regularization
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )
        estimators.append(('xgb', xgb_model))
    
    # Create voting classifier
    logger.log(1, 0, 1, logger.Colors.INFO, f"Creating VotingClassifier with {len(estimators)} models")
    model = VotingClassifier(
        estimators=estimators,
        voting='soft',  # Use predicted probabilities
        n_jobs=-1
    )
    
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
    
    # Summary of improvements
    if use_feature_selection:
        logger.log(0, 0, 1, logger.Colors.YELLOW, f"✓ Feature selection: {n_top_features} features used")
    logger.log(0, 0, 1, logger.Colors.YELLOW, f"✓ Increased regularization applied to all models")
    logger.log(0, 0, 1, logger.Colors.YELLOW, f"✓ More training data: {len(X_train)} samples ({test_size*100:.0f}% validation split)")
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_indices': feature_indices,
        'test_size': test_size,
        'threshold': threshold,
        'use_scaling': use_scaling,
        'use_feature_selection': use_feature_selection,
        'n_top_features': n_top_features if use_feature_selection else None,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'model_type': 'Ensemble',
        'n_estimators_ensemble': len(estimators)
    }


def save_model_run(result, models_root: str = 'models', prefix: str = 'model'):
    """Save the model and a params.txt inside a sequential run folder under models_root.

    The function will create models_root/ if missing, then create models_root/{prefix}_{N}
    where N is the next integer sequence (starting at 1). Inside the run folder it
    writes 'trained_model.pkl' and 'params.txt' with run metadata.
    """
    from pathlib import Path

    # Get current directory (RandomForests+Ensemble)
    current_dir = Path(__file__).parent
    root = current_dir / models_root
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
        'scaler': result.get('scaler'),
        'feature_indices': result.get('feature_indices'),
        'model_type': result.get('model_type', 'Unknown')
    }
    with model_file.open('wb') as fh:
        pickle.dump(model_package, fh)

    # Prepare params
    total_battles = 0
    if 'y_train' in result and 'y_test' in result:
        total_battles = len(result['y_train']) + len(result['y_test'])

    params = [
        ('model_file', str(model_file)),
        ('model_type', str(result.get('model_type', 'Unknown'))),
        ('test_size', str(result.get('test_size', ''))),
        ('prediction_threshold', str(result.get('threshold', ''))),
        ('use_scaling', str(result.get('use_scaling', ''))),
        ('use_feature_selection', str(result.get('use_feature_selection', ''))),
        ('n_top_features', str(result.get('n_top_features', ''))),
        ('train_accuracy', str(result.get('train_accuracy', ''))),
        ('test_accuracy', str(result.get('test_accuracy', ''))),
        ('num_battles', str(total_battles)),
        ('run_timestamp', utc_iso_now())
    ]
    
    # Add model-specific parameters
    if result.get('model_type') == 'Ensemble':
        params.append(('n_estimators_in_ensemble', str(result.get('n_estimators_ensemble', ''))))

    params_file = run_dir / 'params.txt'
    with params_file.open('w', encoding='utf-8') as fh:
        for k, v in params:
            fh.write(f"{k}: {v}\n")

    logger.log_success(f"Model run saved to: {run_dir}", newline_before=1, newline_after=1)
    return run_dir


if __name__ == "__main__":
    
    logger.log_info("Training Ensemble model...", newline_before=1)

    # Get repository root
    from pathlib import Path
    repo_root = Path(__file__).parent.parent
    data_path = repo_root / "data" / "train.jsonl"

    # Load training data
    train_data = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                train_data.append(json.loads(line))

    # Create battleline struct
    battleline_struct = be.create_final_turn_feature(train_data)
    
    # Train Ensemble (includes Random Forest + Extra Trees + Gradient Boosting + XGBoost)
    ensemble_result = train_ensemble(
        battleline=battleline_struct,
        test_size=TEST_SIZE,
        threshold=THRESHOLD,
        use_scaling=True,
        use_feature_selection=USE_FEATURE_SELECTION,
        n_top_features=TOP_FEATURES
    )
    save_model_run(ensemble_result, models_root='models', prefix='model')

    logger.log_final_result(True, "Ensemble training complete!")
