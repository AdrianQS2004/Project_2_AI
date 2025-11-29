# Best Model Parameters - Score: 0.92100

> **📚 New to Ensemble Learning?** See `ENSEMBLE_QUICK_GUIDE.md` for a beginner-friendly explanation, or `HOW_ENSEMBLE_WORKS.md` for detailed step-by-step examples with code!

## Final Ensemble Method

**Score**: 0.92100  
**Method**: Weighted Ensemble  
**File**: `submissions/my_submission_weighted_80.csv` or `submissions/my_submission_weighted_90.csv`

> **📚 New to Ensemble Learning?** See `HOW_ENSEMBLE_WORKS.md` for a detailed beginner-friendly explanation of how we combine multiple models!

### Ensemble Weights

- **Advanced Ensemble**: 80% or 90% (both scored 0.92100)
- **CatBoost Ensemble**: 20% or 10%
- **Combination**: Simple weighted average
  ```
  final_prediction = 0.8 * advanced_predictions + 0.2 * catboost_predictions
  # OR
  final_prediction = 0.9 * advanced_predictions + 0.1 * catboost_predictions
  ```

---

## Component 1: Advanced Ensemble (0.92088)

### Ensemble Configuration

- **Models**: 5 models combined
- **Combining Method**: Optimized weighted average (weights optimized on validation set)
- **Models Used**:
  1. Random Forest
  2. Gradient Boosting
  3. XGBoost
  4. LightGBM
  5. Extra Trees

### Feature Engineering

- **Original Features**: One-hot encoded categorical variables (all categorical columns)
- **Additional Features Created**:
  1. `income_to_loan`: `annual_income / (loan_amount + 1)`
  2. `debt_score_interaction`: `debt_to_income_ratio * credit_score`
  3. `debt_score_ratio`: `debt_to_income_ratio / (credit_score + 1)`
  4. `interest_loan_interaction`: `interest_rate * loan_amount`

**Total Features**: Original + 4 new interaction features

### Data Preprocessing

- One-hot encoding with `drop_first=True`
- Standardization: `(x - mean) / std` (using training set statistics)
- Train/Validation split: 80/20, `random_state=2025`
- Zero std replacement: Replaced with 1 to avoid division by zero

### Model 1: Random Forest

```python
RandomForestClassifier(
    n_estimators=400,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # Feature subsampling
    random_state=2025,
    n_jobs=-1,
    verbose=0
)
```

### Model 2: Gradient Boosting

```python
GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.85,
    max_features='sqrt',  # Feature subsampling
    random_state=2025,
    verbose=0
)
```

### Model 3: XGBoost

```python
XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    colsample_bylevel=0.85,  # Additional feature subsampling
    gamma=0.2,
    reg_alpha=0.2,  # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    random_state=2025,
    eval_metric='auc',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0
)
```

### Model 4: LightGBM

```python
LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=40,
    min_child_samples=15,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,  # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    random_state=2025,
    n_jobs=-1,
    verbosity=-1
)
```

### Model 5: Extra Trees

```python
ExtraTreesClassifier(
    n_estimators=400,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',  # Feature subsampling
    random_state=2025,
    n_jobs=-1,
    verbose=0
)
```

### Advanced Ensemble Combining Details

The ensemble uses the following methods and selects the best based on validation ROC AUC:

1. **Simple Average**: Mean of all 5 model predictions
2. **Weighted Average**: Weighted by individual validation ROC AUC scores
3. **Optimized Weights**: Weights optimized using scipy.optimize.minimize to maximize validation ROC AUC
4. **Stacking**: Logistic Regression meta-learner trained on validation predictions

**Best method selected**: The one with highest validation ROC AUC

### Advanced Ensemble Combining Method

The Advanced Ensemble uses **4 different combination methods** and picks the best one:

1. **Simple Average**: Just average all 5 model predictions equally

   ```python
   final = (RF + GB + XGB + LGB + ET) / 5
   ```

2. **Weighted Average**: Weight each model by its validation performance

   ```python
   weights = [rf_score, gb_score, xgb_score, lgb_score, et_score]
   weights = weights / sum(weights)  # Normalize
   final = RF*w1 + GB*w2 + XGB*w3 + LGB*w4 + ET*w5
   ```

3. **Optimized Weights**: Use math (scipy.optimize) to find the BEST weights

   - Finds weights that maximize validation ROC AUC
   - Automatically balances the models

4. **Stacking**: Train another model (Logistic Regression) to learn HOW to combine
   - Uses predictions from 5 models as "features"
   - Meta-learner learns optimal combination

**Selected Method**: The one with highest validation ROC AUC

> **💡 How does this work?** See `ENSEMBLE_QUICK_GUIDE.md` for simple examples with numbers!

---

## Component 2: CatBoost Ensemble (0.92016)

### Ensemble Configuration

- **Models**: 4 models combined
- **Combining Method**: Optimized weighted average

### Data Preprocessing

- One-hot encoding with `drop_first=True`
- Standardization: `(x - mean) / std` (using training set statistics)
- Train/Validation split: 80/20, `random_state=2025`
- **Note**: No additional feature engineering (CatBoost handles categoricals natively)

### Model 1: CatBoost

```python
CatBoostClassifier(
    iterations=600,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=3,
    border_count=32,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=2025,
    verbose=False,
    thread_count=-1
)
# Early stopping: 50 rounds without improvement
```

### Model 2: XGBoost

```python
XGBClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=2,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.2,
    reg_alpha=0.2,  # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    random_state=2025,
    eval_metric='auc',
    use_label_encoder=False,
    n_jobs=-1,
    verbosity=0
)
```

### Model 3: LightGBM

```python
LGBMClassifier(
    n_estimators=600,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=40,
    min_child_samples=15,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.2,  # L1 regularization
    reg_lambda=1.5,  # L2 regularization
    random_state=2025,
    n_jobs=-1,
    verbosity=-1
)
# Early stopping: 50 rounds without improvement
```

### Model 4: Gradient Boosting

```python
GradientBoostingClassifier(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.85,
    max_features='sqrt',
    random_state=2025,
    verbose=0
)
```

### CatBoost Ensemble Combining Method

- Simple average
- Weighted average (by individual validation ROC AUC)
- Optimized weights (using scipy.optimize.minimize on validation set)
- **Selected Method**: Best one based on validation ROC AUC

---

## Key Hyperparameters Summary

### Common Patterns Across All Models

1. **Learning Rate**: 0.03 (lower than default 0.1)
2. **Regularization**:
   - L1 (reg_alpha): 0.2
   - L2 (reg_lambda): 1.5
3. **Subsampling**: 0.85 (row sampling)
4. **Feature Subsampling**: 0.85 or 'sqrt'
5. **More Trees**: 400-600 (higher than default)
6. **Deeper Trees**: max_depth 7-8 (moderately deep)

### Training Configuration

- **Random State**: 2025 (for reproducibility)
- **Train/Validation Split**: 80/20
- **Validation Metric**: ROC AUC
- **Early Stopping**: 50 rounds (for CatBoost and LightGBM)
- **Standardization**: Applied after feature engineering

---

## File Locations

### Input Files

- Training data: `Datasets/train.csv`
- Test data: `Datasets/test.csv`

### Output Files

- Best submission: `submissions/my_submission_weighted_80.csv` (score: 0.92100)
- Alternative: `submissions/my_submission_weighted_90.csv` (score: 0.92100)

### Script Files

- Advanced ensemble: `ensemble_advanced.py`
- CatBoost ensemble: `ensemble_catboost.py`
- Weighted combination: `ensemble_weighted_advanced.py`

---

## Reproducibility

To reproduce the 0.92100 score:

1. **Run Advanced Ensemble**:

   ```bash
   python ensemble_advanced.py
   # Produces: submissions/my_submission_advanced.csv (0.92088)
   ```

2. **Run CatBoost Ensemble**:

   ```bash
   python ensemble_catboost.py
   # Produces: submissions/my_submission_catboost.csv (0.92016)
   ```

3. **Create Weighted Ensemble**:
   ```bash
   python ensemble_weighted_advanced.py
   # Produces: submissions/my_submission_weighted_80.csv (0.92100)
   ```

Or use the existing submission files directly.

---

## Performance Breakdown

| Component          | Score       | Weight | Contribution |
| ------------------ | ----------- | ------ | ------------ |
| Advanced Ensemble  | 0.92088     | 0.8    | 0.73670      |
| CatBoost Ensemble  | 0.92016     | 0.2    | 0.18403      |
| **Weighted Final** | **0.92100** | 1.0    | **0.92073**  |

_Note: Weighted average calculation: 0.8 × 0.92088 + 0.2 × 0.92016 = 0.920736_

---

## Key Success Factors

1. **Feature Engineering**: Interaction features were crucial (Advanced ensemble)
2. **Ensemble Diversity**: 5 different models in Advanced ensemble
3. **Weighted Combination**: Weighting better model more heavily (80-90%)
4. **Lower Learning Rates**: 0.03 allowed models to learn more gradually
5. **More Trees**: 400-600 trees provided better generalization
6. **Regularization**: L1/L2 regularization prevented overfitting
7. **Optimized Weights**: Weights optimized on validation set

---

## Notes

- Both 80% and 90% weights achieved the same score (0.92100)
- 95% weight performed slightly worse (0.92096) - too much weight reduces diversity
- Simple average (50/50) would have scored lower (~0.92077)
- Feature engineering in Advanced ensemble was key differentiator
