# Experiment Log - Loan Default Prediction Competition

## Score History

| Date    | Model               | ROC AUC     | Improvement | Notes                                            |
| ------- | ------------------- | ----------- | ----------- | ------------------------------------------------ |
| Initial | Baseline            | 0.91298     | Baseline    | Initial submission                               |
| -       | Ensemble (4 models) | 0.92045     | +0.00747    | Combined RF, GB, XGB, LGB with optimized weights |
| -       | Advanced Ensemble   | 0.92088     | +0.00043    | Feature engineering + 5 models + stacking        |
| -       | CatBoost Ensemble   | 0.92016     | -0.00029    | CatBoost + XGB + LGB + GB (weaker than advanced) |
| -       | Combined Ensemble   | 0.92077     | -0.00011    | Simple average of Advanced + CatBoost            |
| -       | Weighted 80%        | **0.92100** | +0.00012    | Advanced 80% + CatBoost 20%                      |
| -       | Weighted 90%        | **0.92100** | +0.00012    | Advanced 90% + CatBoost 10%                      |
| -       | Weighted 95%        | 0.92096     | +0.00008    | Advanced 95% + CatBoost 5%                       |

## Current Best: 0.92100 (Weighted Ensemble 80% or 90%) 🏆

---

## Experiment 1: Initial Baseline

- **Model**: Single model (unknown which one)
- **Score**: 0.91298
- **Date**: Initial submission
- **Notes**: Baseline performance

---

## Experiment 2: Ensemble Method

- **Date**: Current session
- **Model**: Ensemble of 4 models
  - Random Forest (300 trees, max_depth=25)
  - Gradient Boosting (300 trees, lr=0.05, max_depth=6)
  - XGBoost (500 trees, lr=0.05, max_depth=7, regularization)
  - LightGBM (500 trees, lr=0.05, max_depth=7, regularization)
- **Method**: Optimized weighted average (weights optimized on validation set)
- **Score**: 0.92045
- **Improvement**: +0.00747 (+0.82% relative improvement)
- **Key Changes**:
  - Combined multiple models instead of single model
  - Used optimized weights based on validation ROC AUC
  - Increased number of trees (300-500 vs 200)
  - Lower learning rates (0.05 vs 0.1)
  - Added regularization (L1/L2)
- **Files**: `ensemble_method.py`

---

## Experiment 3: Advanced Ensemble ✅ COMPLETED

- **Date**: Current session
- **File**: `ensemble_advanced.py`
- **Score**: 0.92088
- **Improvement**: +0.00043 over 0.92045
- **Changes from Experiment 2**:
  - Feature engineering (interaction features)
  - More models (added Extra Trees) - 5 models total
  - Higher tree counts (400-600 vs 300-500)
  - Lower learning rates (0.03 vs 0.05)
  - Deeper trees (max_depth 7-8 vs 6-7)
  - More regularization
  - Stacking with logistic regression meta-learner
- **Best Method**: [Check output - which ensemble method won?]
- **Notes**: Slight improvement, feature engineering helped
- **Status**: ✅ Completed

---

## Experiment 4: CatBoost Ensemble ✅ COMPLETED

- **Date**: Current session
- **File**: `ensemble_catboost.py`
- **Score**: 0.92016
- **Improvement**: -0.00029 vs Advanced (0.92088)
- **Changes**:
  - Added CatBoost (often outperforms XGBoost/LightGBM)
  - CatBoost handles categorical features natively
  - Optimized hyperparameters
  - Ensemble with XGBoost, LightGBM, Gradient Boosting
- **Notes**:
  - Performed worse than Advanced ensemble
  - CatBoost didn't help as much as expected for this dataset
  - Feature engineering in Advanced was more effective
- **Status**: ✅ Completed

---

## Experiment 5: Combined Both Ensembles ✅ COMPLETED

- **Date**: Current session
- **File**: `run_both_ensembles.py`
- **Score**: 0.92077
- **Improvement**: -0.00011 vs Advanced (0.92088)
- **What it does**:
  - Runs both `ensemble_advanced.py` and `ensemble_catboost.py`
  - Saves individual results
  - Combines predictions from both (simple average)
  - Creates `my_submission_combined.csv`
- **Results Analysis**:
  - Combined (0.92077) < Advanced (0.92088)
  - Simple average diluted the strong Advanced predictions
  - CatBoost (0.92016) was weaker, bringing down the average
- **Key Learning**:
  - Not all ensembles benefit from simple averaging
  - When one method is clearly better, averaging can hurt
  - Should have used weighted average favoring Advanced
- **Status**: ✅ Completed

---

## Experiment 6: Weighted Ensemble ✅ COMPLETED

- **Date**: Current session
- **File**: `ensemble_weighted_advanced.py`
- **Scores**:
  - Weighted 80% (Advanced 80%, CatBoost 20%): **0.92100** 🏆
  - Weighted 90% (Advanced 90%, CatBoost 10%): **0.92100** 🏆
  - Weighted 95% (Advanced 95%, CatBoost 5%): 0.92096
- **Improvement**: +0.00012 over Advanced (0.92088)
- **Key Finding**:
  - Weighting Advanced more heavily (80-90%) improved performance
  - 80% and 90% performed equally well
  - 95% was slightly worse (too much weight on one model)
- **Status**: ✅ Completed

---

## Experiment 7: Advanced Feature Engineering (In Progress)

- **File**: `feature_engineering_advanced.py`
- **Status**: ⏳ Running - waiting for results
- **Expected**: +0.001 to +0.003 improvement
- **Changes**:
  - More interaction features (10+ new features)
  - Polynomial features (squared terms)
  - Log transformations
  - More trees (500-700 vs 400-600)
  - Lower learning rates (0.02 vs 0.03)
  - Deeper trees (max_depth 8-9 vs 7-8)

---

## Next Experiments (To Try)

### Experiment 8: Hyperparameter Optimization

- Grid search or random search
- Bayesian optimization (Optuna)
- Cross-validation for robust evaluation

### Experiment 9: Advanced Neural Network

- Better architecture
- More layers
- Attention mechanisms

### Experiment 10: Blending

- Different train/validation splits
- Multiple folds
- Out-of-fold predictions

---

## Technical Details

### Data Preprocessing

- One-hot encoding for categorical variables
- Standardization (mean=0, std=1)
- Train/validation split: 80/20, random_state=2025
- Column alignment between train/test sets

### Model Configurations

#### Random Forest

- n_estimators: 300
- max_depth: 25
- min_samples_split: 5
- min_samples_leaf: 2

#### Gradient Boosting

- n_estimators: 300
- learning_rate: 0.05
- max_depth: 5
- subsample: 0.8

#### XGBoost

- n_estimators: 500
- learning_rate: 0.05
- max_depth: 6
- min_child_weight: 3
- subsample: 0.8
- colsample_bytree: 0.8
- gamma: 0.1
- reg_alpha: 0.1 (L1)
- reg_lambda: 1.0 (L2)

#### LightGBM

- n_estimators: 500
- learning_rate: 0.05
- max_depth: 6
- num_leaves: 31
- min_child_samples: 20
- subsample: 0.8
- colsample_bytree: 0.8
- reg_alpha: 0.1 (L1)
- reg_lambda: 1.0 (L2)

### Ensemble Method

- Simple average
- Weighted average (by individual ROC AUC)
- Optimized weights (scipy.optimize.minimize on validation set)
- Best method selected based on validation ROC AUC

---

## Lessons Learned

1. **Ensemble > Single Model**: Combining multiple models significantly improves performance
2. **Optimized Weights**: Optimizing ensemble weights on validation set beats simple averaging
3. **More Trees + Lower LR**: Better than fewer trees with higher learning rate
4. **Regularization Helps**: L1/L2 regularization improves generalization
5. **Feature Engineering Works**: Interaction features were key to Advanced ensemble's success
6. **Weighted > Simple Average**: When one model is better, weight it more heavily
7. **80-90% Weight Optimal**: Too much weight (95%) can reduce diversity and hurt performance

---

## Next Steps

1. Wait for Advanced Feature Engineering results
2. Try hyperparameter optimization
3. Experiment with cross-validation
4. Try CatBoost
5. Cross-validation for more robust evaluation
