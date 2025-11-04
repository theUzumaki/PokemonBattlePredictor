# Pokemon Battle Predictor

A machine learning project to predict the outcome of Pokemon battles using battle logs and Pokemon statistics.

## Project Structure

```
PokemonBattlePredictor/
├── PCA+logistic/          # PCA + Logistic Regression approach
│   ├── train.py           # Training script with PCA + LogReg
│   ├── predict.py         # Prediction script for PCA + LogReg models
│   ├── load_into_pca.py   # Feature extraction and PCA utilities
│   ├── pca.py             # PCA helper functions
│   └── models/            # Saved PCA+LogReg models
│
├── RandomForests+Ensemble/ # Random Forest & Ensemble approaches (NEW)
│   ├── train.py           # Training script for RF and Ensemble models
│   ├── predict.py         # Prediction script
│   ├── models/            # Saved RF/Ensemble models
│   ├── predictions/       # Prediction outputs
│   └── README.md          # Approach-specific documentation
│
├── utilities/             # Shared utility modules
│   ├── data_parser.py     # Data parsing utilities
│   ├── feature_extractor.py
│   ├── normalizer.py
│   ├── logger.py
│   ├── debug_utils.py
│   ├── time_utils.py
│   └── kfolder.py
│
├── data/                  # Dataset files
│   ├── train.jsonl        # Training battle logs
│   ├── test.jsonl         # Test battle logs
│   └── sample_submission.csv
│
├── tools/                 # Development tools
│   └── scraper.py
│
├── battleline_extractor.py  # Core battle data extraction
├── variables.py             # Data structure definitions
├── test_battleline_struct.py
├── requirements.txt
├── run.sh                   # Main execution script
└── approaches.txt          # Documentation of different ML approaches
```

## Core Components

### Shared Modules (Used by all approaches)

These modules are **approach-agnostic** and can be used with any ML method:

- **`battleline_extractor.py`**: Parses raw battle JSON data into structured battleline objects
- **`variables.py`**: Defines core data structures (Pokemon, Team, Battle, etc.)
- **`configs/`**: Pokemon and move statistics (static data)
- **`utilities/`**: Helper functions for data processing, normalization, logging, etc.

### Approach-Specific Modules

- **`PCA+logistic/`**: Complete implementation using PCA dimensionality reduction + Logistic Regression
- **`RandomForests+Ensemble/`**: Random Forest and Ensemble methods (Random Forest + XGBoost + Gradient Boosting + Extra Trees)

## Machine Learning Approaches

### 1. PCA + Logistic Regression (Current - in `PCA+logistic/`)

**Pipeline:**
1. Extract high-dimensional features from battle data
2. Normalize with StandardScaler
3. Reduce dimensions with PCA (200 components)
4. Train Logistic Regression classifier

**Pros:**
- Simple, interpretable
- Fast training
- Works well for linearly separable data

**Cons:**
- May miss non-linear relationships
- PCA loses some information
- Limited capacity for complex patterns

### 2. Random Forest & Ensemble Learning (Recommended Next Step)

**Potential Approaches:**

#### Random Forest
- No PCA needed (handles high dimensions natively)
- Captures non-linear relationships
- Built-in feature importance
- Resistant to overfitting

#### Gradient Boosting (XGBoost/LightGBM)
- Often best performance on tabular data
- Handles imbalanced datasets well
- Industry-standard for competitions

#### Voting Ensemble
- Combine predictions from multiple models
- Average probabilities for more robust predictions
- Best of all worlds

#### Stacking
- Use predictions from base models as features
- Meta-learner makes final prediction
- Maximum flexibility

## Getting Started

### Installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Using PCA + Logistic Regression

```bash
# Using the run script
./run.sh --pca

# Or directly
cd "PCA+logistic"
python train.py
python predict.py
```

### Using Random Forest & Ensemble Methods

```bash
# Using the run script
./run.sh --rf          # For Random Forest
./run.sh --ensemble    # For Ensemble (RF + XGBoost + GradBoost)

# Or directly
cd "RandomForests+Ensemble"
python train.py   # Trains both Random Forest and Ensemble models
python predict.py # Uses latest trained model for prediction
```

### Run Script Options

```bash
./run.sh --pca         # Use PCA + Logistic Regression (default)
./run.sh --rf          # Use Random Forest
./run.sh --ensemble    # Use Ensemble methods
./run.sh --train       # Run only training
./run.sh --predict     # Run only prediction
```

## Data Structures

See `variables.py` for complete definitions:

- **`pkmn`**: Individual Pokemon (stats, moves, HP, status, boosts)
- **`team`**: Player's team (6 Pokemon)
- **`adv_team`**: Adversary team (partially observed)
- **`battle`**: Complete battle state
- **`battleline`**: Collection of battles

## Model Outputs

- **Models**: `models/model_N/trained_model.pkl`
- **Predictions**: `predictions/prediction_N/prediction.csv`
- **Parameters**: Each run includes `params.txt` with hyperparameters and metrics

## Contributing

When implementing new approaches:

1. Create a new directory for the approach (e.g., `RandomForest/`, `GradientBoosting/`)
2. Reuse shared modules (`battleline_extractor.py`, `variables.py`, `utilities/`)
3. Document approach-specific code in the new directory
4. Update this README with new approach details

## Dependencies

See `requirements.txt` for core dependencies:
- numpy
- scikit-learn
- chronicle (custom logging)
- xgboost (for gradient boosting)
- lightgbm (for LightGBM)

---

**Current Date**: November 4, 2025
**Status**: ✓ PCA+LogReg implemented | ✓ Random Forest & Ensemble implemented | Both approaches ready for use!
