"""Generate predictions CSV from a trained Random Forest/Ensemble model and a JSONL test file.

Usage:
    python predict.py

The script expects the saved model to be a pickle containing a dict with keys:
    - 'model': the fitted estimator with a .predict_proba() method
    - 'scaler': fitted scaler with .transform() (optional, can be None)
    - 'model_type': type of model ('RandomForest', 'Ensemble', etc.)

It reuses the repository's `battleline_extractor.create_final_turn_feature` and
feature extraction to build features compatible with training.
"""

from __future__ import annotations

import sys
import csv
import json
import pickle
from pathlib import Path
from typing import Any

# Add parent directory to path to import shared modules
sys.path.append(str(Path(__file__).parent.parent))
# Add current directory to path for local modules
sys.path.insert(0, str(Path(__file__).parent))

# Try importing logger
try:
    import utilities.logger as logger
except ImportError:
    try:
        import chronicle.logger as logger
    except ImportError:
        # Fallback simple logger
        class SimpleLogger:
            @staticmethod
            def log_info(message, newline_before=0):
                print("\n" * newline_before + f"ℹ {message}")
            
            @staticmethod
            def log_success(message, newline_before=0, newline_after=0):
                print("\n" * newline_before + f"✓ {message}" + "\n" * newline_after)
            
            @staticmethod
            def log_application_title(title):
                print(f"\n{'='*60}\n{title.center(60)}\n{'='*60}\n")
            
            @staticmethod
            def log_step(step, total, description):
                print(f"\n[Step {step}/{total}] {description}")
            
            class Colors:
                INFO = ""
                BRIGHT_GREEN = ""
        
        logger = SimpleLogger()

from utilities.time_utils import utc_iso_now


def load_model_package(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return pickle.load(fh)


def read_jsonl(path: Path) -> list[dict]:
    data: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {lineno} of {path}: {e.msg}") from e
            data.append(obj)
    return data


def main():
    """Run prediction using repository defaults.

    Defaults:
      - model: latest model from models/ directory
      - input: ../data/test.jsonl
      - threshold: 0.5
    """

    # Get the current directory (RandomForests+Ensemble)
    current_dir = Path(__file__).parent
    repo_root = current_dir.parent
    
    # Config - paths relative to repo root
    input_path = repo_root / "data" / "test.jsonl"
    threshold = 0.5

    # Lazy imports that depend on the project
    try:
        from battleline_extractor import create_final_turn_feature
    except Exception as e:
        raise RuntimeError("Failed to import project feature helpers. Run this script from the repository root and ensure PYTHONPATH includes the project.") from e

    # Import feature extraction from PCA module (reusing the same features)
    sys.path.append(str(repo_root / "PCA+logistic"))
    from extractor import extract_battle_features

    # Find the latest model in the models directory
    models_root = current_dir / "models"
    model_path = None
    
    if models_root.exists():
        candidates = [p for p in models_root.iterdir() if p.is_dir() and p.name.startswith("model_")]
        nums = []
        for p in candidates:
            try:
                nums.append((int(p.name.split("_")[-1]), p))
            except Exception:
                continue
        if nums:
            latest = sorted(nums)[-1][1]
            candidate = latest / "trained_model.pkl"
            if candidate.exists():
                logger.log_info(f"Using latest model from: {candidate}")
                model_path = candidate
    
    if model_path is None or not model_path.exists():
        raise FileNotFoundError(f"No trained model found in {models_root}. Please run train.py first.")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

    logger.log_application_title("PREDICTION - RANDOM FOREST/ENSEMBLE")
    
    # Step 1: Load model
    logger.log_step(1, 5, "Loading trained model")
    model_package = load_model_package(model_path)
    model = model_package['model']
    scaler = model_package.get('scaler')
    feature_indices = model_package.get('feature_indices')
    model_type = model_package.get('model_type', 'Unknown')
    logger.log_info(f"Model type: {model_type}")
    if scaler is not None:
        logger.log_info("Scaler loaded")
    else:
        logger.log_info("No scaler (features not scaled)")
    if feature_indices is not None:
        logger.log_info(f"Feature selection enabled: {len(feature_indices)} features")
    else:
        logger.log_info("No feature selection")

    # Step 2: Read test data
    logger.log_step(2, 5, f"Reading test data from {input_path.name}")
    test_records = read_jsonl(input_path)
    logger.log_info(f"Loaded {len(test_records)} test battles")

    # Step 3: Create battleline struct
    logger.log_step(3, 5, "Creating battleline structure")
    test_battleline = create_final_turn_feature(test_records, is_train=False)
    logger.log_info(f"Battleline created with {len(test_battleline.battles)} battles")

    # Step 4: Extract features
    logger.log_step(4, 5, "Extracting features")
    test_features = extract_battle_features(test_battleline, max_moves=4)
    logger.log_info(f"Feature shape: {test_features.shape}")
    
    # Apply scaling if scaler exists
    if scaler is not None:
        logger.log_info("Applying scaler transformation")
        test_features = scaler.transform(test_features)
    
    # Apply feature selection if indices exist
    if feature_indices is not None:
        logger.log_info(f"Applying feature selection: {test_features.shape[1]} → {len(feature_indices)}")
        test_features = test_features[:, feature_indices]

    # Step 5: Generate predictions
    logger.log_step(5, 5, "Generating predictions")
    
    # Get probabilities
    probabilities = model.predict_proba(test_features)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    logger.log_info(f"Generated {len(predictions)} predictions")
    logger.log_info(f"Predicted wins: {predictions.sum()}, losses: {len(predictions) - predictions.sum()}")

    # Create predictions directory structure
    predictions_root = current_dir / "predictions"
    predictions_root.mkdir(exist_ok=True)
    
    # Find next prediction number
    existing = [p.name for p in predictions_root.iterdir() if p.is_dir() and p.name.startswith("prediction_")]
    nums = []
    for name in existing:
        try:
            nums.append(int(name.split("_")[-1]))
        except Exception:
            continue
    next_n = max(nums) + 1 if nums else 1
    pred_dir = predictions_root / f"prediction_{next_n}"
    pred_dir.mkdir()
    
    # Save predictions CSV
    output_path = pred_dir / "prediction.csv"
    
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "player_won"])
        
        for i, (battle_id, pred) in enumerate(zip(test_battleline.battles.keys(), predictions)):
            writer.writerow([i, pred])
    
    logger.log_success(f"Predictions saved to: {output_path}", newline_before=1)
    
    # Save params file
    params_file = pred_dir / "params.txt"
    with params_file.open('w', encoding='utf-8') as fh:
        fh.write(f"model_path: {model_path}\n")
        fh.write(f"model_type: {model_type}\n")
        fh.write(f"input_file: {input_path}\n")
        fh.write(f"output_file: {output_path}\n")
        fh.write(f"num_predictions: {len(predictions)}\n")
        fh.write(f"threshold: {threshold}\n")
        fh.write(f"prediction_timestamp: {utc_iso_now()}\n")
    
    logger.log_success(f"Parameters saved to: {params_file}", newline_after=1)
    
    return output_path


if __name__ == "__main__":
    try:
        output = main()
        print(f"\n{'='*60}")
        print(f"✓ SUCCESS - Predictions generated!")
        print(f"{'='*60}\n")
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ ERROR: {e}")
        print(f"{'='*60}\n")
        raise
