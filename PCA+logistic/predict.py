"""Generate predictions CSV from a trained model and a JSONL test file.

Usage:
	python predict.py --model models/trained_model.pkl --input data/test.jsonl --output predictions.csv

The script expects the saved model to be a pickle containing a dict with keys:
	- 'model': the fitted estimator with a .predict() method
	- 'pca_model': fitted PCA transformer with .transform()
	- 'scaler': fitted scaler with .transform()

It reuses the repository's `battleline_extractor.create_final_turn_feature` and
`load_into_pca.extract_battle_features` to build features compatible with training.
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

import chronicle.logger as logger
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
	"""Run prediction using repository defaults (no CLI parser).

	Defaults:
	  - model: models/trained_model.pkl
	  - input: data/test.jsonl
	  - output: predictions.csv
	  - threshold: 0.5
	"""

	# Get the repository root (parent of PCA+logistic directory)
	repo_root = Path(__file__).parent.parent
	
	# Config (hardcoded defaults) - paths relative to repo root
	model_path = repo_root / "models" / "trained_model.pkl"
	input_path = repo_root / "data" / "test.jsonl"
	threshold = 0.5

	# Lazy imports that depend on the project
	try:
		from battleline_extractor import create_final_turn_feature
		from load_into_pca import extract_battle_features
	except Exception as e:  # pragma: no cover - helpful error for users
		raise RuntimeError("Failed to import project feature helpers. Run this script from the repository root and ensure PYTHONPATH includes the project.") from e

	# If a fixed model path doesn't exist, try to locate the latest run under models/*
	if not model_path.exists():
		models_root = repo_root / "PCA+logistic" / "models"
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
		# If still not found, raise
	if not model_path.exists():
		raise FileNotFoundError(f"Model file not found: {model_path}")
	if not input_path.exists():
		raise FileNotFoundError(f"Input JSONL file not found: {input_path}")

	logger.log_info(f"Loading model from: {model_path}")
	model_pkg = load_model_package(model_path)

	# Read input data
	logger.log_info(f"Reading test data from: {input_path}")
	test_data = read_jsonl(input_path)

	logger.log_info(f"Creating battleline struct for {len(test_data)} battles...")
	# test data does not contain labels, so signal is_train=False
	battleline = create_final_turn_feature(test_data, is_train=False)

	logger.log_info("Extracting features from battleline...")
	features = extract_battle_features(battleline)

	# Apply scaler and PCA, then predict
	scaler = model_pkg.get("scaler")
	pca = model_pkg.get("pca_model")
	model = model_pkg.get("model")

	if scaler is None or pca is None or model is None:
		raise KeyError("Loaded model package must contain 'model', 'pca_model', and 'scaler' keys")

	logger.log_info("Scaling and projecting features into PCA space...")
	features_scaled = scaler.transform(features)
	features_pca = pca.transform(features_scaled)

	logger.log_info("Running predictions...")
	probs = model.predict_proba(features_pca)[:, 1]
	wins = (probs >= threshold)

	# Map predictions to battle ids (the same ordering used by feature extractor)
	battle_ids = list(battleline.battles.keys())
	if len(battle_ids) != len(wins):
		raise RuntimeError("Number of predictions does not match number of battles")

	# Write CSV
	# Prepare predictions directory and sequential run folder
	pred_root = repo_root / "PCA+logistic" / "predictions"
	pred_root.mkdir(parents=True, exist_ok=True)

	# Find next sequential run number
	existing = [p.name for p in pred_root.iterdir() if p.is_dir() and p.name.startswith("prediction_")]
	nums = []
	for name in existing:
		try:
			nums.append(int(name.split("_")[-1]))
		except Exception:
			continue
	next_n = max(nums) + 1 if nums else 1
	run_dir = pred_root / f"prediction_{next_n}"
	run_dir.mkdir()

	# Write prediction.csv inside the run folder
	pred_file = run_dir / "prediction.csv"
	logger.log_info(f"Writing {len(battle_ids)} predictions to: {pred_file}")
	with pred_file.open("w", newline="", encoding="utf-8") as fh:
		writer = csv.writer(fh)
		writer.writerow(["battle_id", "player_won"])
		for bid, win in zip(battle_ids, wins):
			writer.writerow([bid, int(win)])

	# Compare with previous prediction if it exists
	diff_percentage = None
	previous_pred_num = next_n - 1
	if previous_pred_num >= 1:
		prev_run_dir = pred_root / f"prediction_{previous_pred_num}"
		prev_pred_file = prev_run_dir / "prediction.csv"
		
		if prev_pred_file.exists():
			logger.log_info(f"Comparing with previous prediction: {prev_pred_file}")
			
			# Read previous predictions
			prev_predictions = {}
			with prev_pred_file.open("r", encoding="utf-8") as fh:
				reader = csv.DictReader(fh)
				for row in reader:
					prev_predictions[str(row["battle_id"])] = int(row["player_won"])
			
			# Count differences
			num_differences = 0
			num_compared = 0
			for bid, win in zip(battle_ids, wins):
				bid_str = str(bid)
				if bid_str in prev_predictions:
					num_compared += 1
					if int(win) != prev_predictions[bid_str]:
						num_differences += 1
			
			if num_compared > 0:
				diff_percentage = (num_differences / num_compared) * 100
				logger.log_info(f"Difference from previous prediction: {num_differences}/{num_compared} battles ({diff_percentage:.2f}%)")
			else:
				logger.log_info("No overlapping battle IDs with previous prediction")

	# Write params.txt with run parameters (human-readable)
	params_file = run_dir / "params.txt"
	params = [
		("model_path", str(model_path)),
		("input_path", str(input_path)),
		("threshold", str(threshold)),
		("num_battles", str(len(battle_ids))),
		("run_timestamp", utc_iso_now()),
	]
	
	if diff_percentage is not None:
		params.append(("diff_from_previous_pct", f"{diff_percentage:.2f}"))
		params.append(("previous_prediction", str(previous_pred_num)))
	
	with params_file.open("w", encoding="utf-8") as fh:
		for k, v in params:
			fh.write(f"{k}: {v}\n")

	logger.log_info(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
	main()

