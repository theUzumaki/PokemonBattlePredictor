#!/usr/bin/env bash
set -euo pipefail

# defaults
RUN_TRAIN=1
RUN_PREDICT=1
VENV_DIR=".venv"
PYTHON="$VENV_DIR/bin/python3"
DIR="PCA+logistic"

usage() {
    cat <<EOF
Usage: $0 [options]

Options:
    --pca         Use PCA + Logistic Regression model (default)
    --rf          Use Random Forest model
    --ensemble    Use Ensemble model (Random Forest + XGBoost + Gradient Boosting)
    --train       Run only training (disable prediction)
    --predict     Run only prediction (disable training)
    --rm          Remove previous model and prediction files before running
    -h, --help    Show this help
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pca)
            DIR="PCA+logistic"
            shift
            ;;
        --rf|--random-forest)
            DIR="RandomForests+Ensemble"
            shift
            ;;
        --ensemble)
            DIR="RandomForests+Ensemble"
            shift
            ;;
        --train)
            RUN_TRAIN=1
            RUN_PREDICT=0
            shift
            ;;
        --predict)
            RUN_TRAIN=0
            RUN_PREDICT=1
            shift
            ;;
        --rm)
            rm -rf "$DIR/models/"*
            rm -rf "$DIR/predictions/"*
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ $RUN_TRAIN -eq 0 && $RUN_PREDICT -eq 0 ]]; then
    echo "Both training and prediction are disabled. Nothing to do." >&2
    exit 1
fi

# verify python in virtualenv exists
if [[ ! -x "$PYTHON" ]]; then
    echo "Python not found in virtualenv ($PYTHON). Please create the virtual environment in $VENV_DIR." >&2
    exit 2
fi

# run requested steps
if [[ $RUN_TRAIN -eq 1 ]]; then
    "$PYTHON" "$DIR/train.py"
fi

if [[ $RUN_PREDICT -eq 1 ]]; then
    "$PYTHON" "$DIR/predict.py"
fi