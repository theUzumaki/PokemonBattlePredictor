# Model Comparison: PCA+Logistic vs Random Forest & Ensemble

## Quick Reference

| Feature | PCA + Logistic Regression | Random Forest | Ensemble |
|---------|---------------------------|---------------|----------|
| **Training Time** | Fast (~seconds) | Moderate (~minutes) | Slower (~minutes) |
| **Prediction Speed** | Very Fast | Fast | Moderate |
| **Feature Engineering** | PCA (200 components) | Direct features | Direct features |
| **Handles Non-linearity** | ❌ No | ✅ Yes | ✅ Yes |
| **Feature Importance** | ❌ Limited | ✅ Built-in | ✅ Built-in |
| **Overfitting Risk** | Low | Low-Medium | Very Low |
| **Interpretability** | Moderate | Good (feature importance) | Good (feature importance) |
| **Best For** | Linear patterns | Complex patterns | Maximum accuracy |

## Detailed Comparison

### 1. PCA + Logistic Regression

**Location:** `PCA+logistic/`

**Pipeline:**
1. Extract ~2000+ raw features per battle
2. Standardize with StandardScaler
3. Reduce to 200 dimensions with PCA
4. Train Logistic Regression (linear classifier)

**Advantages:**
- ✅ Very fast training and prediction
- ✅ Simple to understand and debug
- ✅ Low memory footprint
- ✅ Works well if data is linearly separable
- ✅ No hyperparameter tuning needed

**Disadvantages:**
- ❌ Can't capture non-linear patterns
- ❌ PCA loses some information
- ❌ May underperform on complex interactions
- ❌ Limited feature interpretation after PCA

**When to Use:**
- Quick baseline model
- Data appears linearly separable
- Need fast predictions
- Limited computational resources

### 2. Random Forest

**Location:** `RandomForests+Ensemble/`

**Pipeline:**
1. Extract ~2000+ raw features per battle
2. Optionally standardize (not required for trees)
3. Train 200 decision trees with bootstrap sampling
4. Average predictions from all trees

**Advantages:**
- ✅ Handles non-linear relationships naturally
- ✅ No feature scaling required
- ✅ Built-in feature importance ranking
- ✅ Robust to outliers and noise
- ✅ Parallel training (uses all CPU cores)
- ✅ Less prone to overfitting

**Disadvantages:**
- ❌ Slower training than logistic regression
- ❌ Larger model size
- ❌ Can be memory intensive

**Hyperparameters:**
- `n_estimators`: 200 (number of trees)
- `max_depth`: 20 (maximum tree depth)
- `min_samples_split`: 5 (minimum samples to split)
- `min_samples_leaf`: 2 (minimum samples per leaf)

**When to Use:**
- Data has non-linear patterns
- Want feature importance analysis
- Care about accuracy over speed
- Have sufficient computational resources

### 3. Ensemble (Voting Classifier)

**Location:** `RandomForests+Ensemble/`

**Pipeline:**
1. Extract ~2000+ raw features per battle
2. Optionally standardize
3. Train multiple models in parallel:
   - Random Forest (200 trees)
   - Extra Trees (200 trees)
   - Gradient Boosting (100 trees)
   - XGBoost (200 trees) [if available]
4. Combine predictions using soft voting (probability averaging)

**Advantages:**
- ✅ Best accuracy potential
- ✅ Leverages strengths of different algorithms
- ✅ More robust to different data patterns
- ✅ Better generalization
- ✅ Feature importance from multiple perspectives

**Disadvantages:**
- ❌ Slowest training time
- ❌ Largest model size
- ❌ Most complex to debug

**Component Models:**
- **Random Forest**: Parallel ensemble, bootstrap sampling
- **Extra Trees**: Even more randomization, faster
- **Gradient Boosting**: Sequential learning, corrects errors
- **XGBoost**: Optimized gradient boosting with regularization

**When to Use:**
- Maximum accuracy is priority
- Have computational resources
- Production deployment (worth the training time)
- Competing in Kaggle-style competitions

## Feature Extraction (Shared)

Both approaches use the same feature extraction from `load_into_pca.extract_battle_features()`:

**Per Pokemon (6 Pokemon per team):**
- HP percentage (1)
- Base stats (6): atk, def, spa, spd, spe, hp
- Boosts (6): stat modifications
- Status (8): one-hot encoding
- Effects (8): one-hot encoding  
- Types (38): type1 + type2 one-hot
- Moves (92): 4 moves × (4 base features + 19 type features)

**Adversary Team:**
- Pokemon alive counts (2)
- Leader HP (1)
- Type presence (19)
- Status presence (8)

**Total: ~1000+ features per battle**

## Performance Expectations

Based on typical Pokemon battle data:

| Metric | PCA+LogReg | Random Forest | Ensemble |
|--------|------------|---------------|----------|
| Expected Accuracy | 65-75% | 70-80% | 75-85% |
| Training Time (10k battles) | ~10s | ~2min | ~5min |
| Prediction Time (1k battles) | <1s | ~2s | ~5s |
| Model Size | ~5MB | ~50MB | ~200MB |

## Usage Examples

### Train and Predict with PCA + Logistic Regression
```bash
./run.sh --pca
```

### Train and Predict with Random Forest
```bash
./run.sh --rf
```

### Train and Predict with Ensemble
```bash
./run.sh --ensemble
```

### Train Only (Any Method)
```bash
./run.sh --pca --train
./run.sh --rf --train
./run.sh --ensemble --train
```

### Predict Only (Any Method)
```bash
./run.sh --pca --predict
./run.sh --rf --predict
./run.sh --ensemble --predict
```

## Choosing the Right Approach

**Use PCA + Logistic Regression if:**
- You need a quick baseline
- Speed is more important than accuracy
- You have limited computational resources
- The data appears linearly separable

**Use Random Forest if:**
- You want better accuracy than logistic regression
- You need feature importance analysis
- You have moderate computational resources
- You want a good balance of speed and accuracy

**Use Ensemble if:**
- Maximum accuracy is the goal
- You have sufficient computational resources
- You're deploying to production (one-time training cost)
- You're competing or need the best possible model

## Feature Importance Analysis

Only available with Random Forest and Ensemble methods. After training, check the console output for:

**Top 20 Most Important Features** - showing which Pokemon attributes, moves, and battle states most influence the outcome.

This can reveal insights like:
- Which Pokemon stats matter most
- Which types provide advantages
- Whether status effects are significant
- How move power vs accuracy trade-off works

## Recommendations

**For Development/Experimentation:**
- Start with PCA + LogReg for quick iteration
- Move to Random Forest for better accuracy
- Use Ensemble for final model

**For Production:**
- Use Ensemble for maximum accuracy
- Consider Random Forest if prediction speed matters
- Cache predictions when possible

**For Research/Analysis:**
- Use Random Forest or Ensemble for feature importance
- Compare all three to understand data characteristics
- Use insights to engineer better features
