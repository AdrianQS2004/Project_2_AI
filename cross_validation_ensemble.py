# Cross-Validation Ensemble
# Uses K-fold CV for more robust evaluation and better generalization
# Expected: +0.0003 to +0.001 improvement

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import time

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False

print("="*80)
print("CROSS-VALIDATION ENSEMBLE (WITH OPTIMIZED HYPERPARAMETERS)")
print("="*80)
print("Using K-fold CV for robust evaluation and better generalization")
print("Using optimized hyperparameters from 50-trial Optuna optimization")

# Configuration
N_FOLDS = 5  # 5-fold CV (can increase to 10 for more robustness)
print(f"\nConfiguration: {N_FOLDS}-fold cross-validation")

# Load data
print("\n[1/4] Loading and preprocessing data...")
training_db = pd.read_csv("Datasets/train.csv", header=0)
test_db = pd.read_csv("Datasets/test.csv", header=0)

# One-hot encode
training_db = pd.get_dummies(training_db, prefix_sep="_", drop_first=True, dtype=int)
labels = training_db["loan_paid_back"]
ids = test_db['id']
training_db = training_db.drop(columns=["loan_paid_back", "id"])

test_db = test_db.drop(columns=["id"])
test_db = pd.get_dummies(test_db, prefix_sep="_", drop_first=True, dtype=int)
test_db = test_db.reindex(columns=training_db.columns, fill_value=0)

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()

# Advanced feature engineering (same as feature_engineering_advanced.py)
print("\n[2/4] Creating advanced features...")
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()

def find_col(pattern, cols):
    matches = [c for c in cols if pattern.lower() in c.lower()]
    return matches[0] if matches else None

features_created = 0

# Income-related features
if find_col('annual_income', train_data.columns):
    income_col = find_col('annual_income', train_data.columns)
    if find_col('loan_amount', train_data.columns):
        loan_col = find_col('loan_amount', train_data.columns)
        train_data['income_to_loan'] = train_data[income_col] / (train_data[loan_col] + 1)
        test_data['income_to_loan'] = test_data[income_col] / (test_data[loan_col] + 1)
        features_created += 1
        
        train_data['loan_to_income'] = train_data[loan_col] / (train_data[income_col] + 1)
        test_data['loan_to_income'] = test_data[loan_col] / (test_data[income_col] + 1)
        features_created += 1

# Debt and credit interactions
if find_col('debt_to_income', train_data.columns):
    debt_col = find_col('debt_to_income', train_data.columns)
    if find_col('credit_score', train_data.columns):
        score_col = find_col('credit_score', train_data.columns)
        train_data['debt_score_product'] = train_data[debt_col] * train_data[score_col]
        test_data['debt_score_product'] = test_data[debt_col] * test_data[score_col]
        features_created += 1
        
        train_data['debt_score_ratio'] = train_data[debt_col] / (train_data[score_col] + 1)
        test_data['debt_score_ratio'] = test_data[debt_col] / (test_data[score_col] + 1)
        features_created += 1
        
        train_data['score_debt_interaction'] = train_data[score_col] * (1 - train_data[debt_col])
        test_data['score_debt_interaction'] = test_data[score_col] * (1 - test_data[debt_col])
        features_created += 1

# Interest rate interactions
if find_col('interest_rate', train_data.columns):
    rate_col = find_col('interest_rate', train_data.columns)
    if find_col('loan_amount', train_data.columns):
        loan_col = find_col('loan_amount', train_data.columns)
        train_data['interest_loan_product'] = train_data[rate_col] * train_data[loan_col]
        test_data['interest_loan_product'] = test_data[rate_col] * test_data[loan_col]
        features_created += 1
        
        train_data['total_interest'] = train_data[rate_col] * train_data[loan_col] / 100
        test_data['total_interest'] = test_data[rate_col] * test_data[loan_col] / 100
        features_created += 1

# Polynomial features
key_features = []
for pattern in ['annual_income', 'credit_score', 'loan_amount', 'debt_to_income', 'interest_rate']:
    col = find_col(pattern, train_data.columns)
    if col:
        key_features.append(col)

for col in key_features[:3]:
    train_data[f'{col}_squared'] = train_data[col] ** 2
    test_data[f'{col}_squared'] = test_data[col] ** 2
    features_created += 1

# Log transformations
for col in key_features[:2]:
    train_data[f'{col}_log'] = np.log1p(np.abs(train_data[col]))
    test_data[f'{col}_log'] = np.log1p(np.abs(test_data[col]))
    features_created += 1

print(f"  Created {features_created} new features")
print(f"  Total features: {len(train_data.columns)}")

# Standardize test data (will standardize train per fold)
test_means = train_data.mean()
test_stds = train_data.std().replace(0, 1)
test_data = (test_data - test_means) / test_stds

# Cross-validation setup
print(f"\n[3/4] Setting up {N_FOLDS}-fold cross-validation...")
kf = sklearn.model_selection.KFold(n_splits=N_FOLDS, shuffle=True, random_state=2025)

# Store out-of-fold predictions and test predictions
oof_predictions = np.zeros(len(train_data))
test_predictions_list = []

fold_scores = []

print(f"\n[4/4] Training models with {N_FOLDS}-fold CV...")
print("="*80)

for fold, (train_idx, val_idx) in enumerate(kf.split(train_data), 1):
    print(f"\nFold {fold}/{N_FOLDS}")
    print("-"*80)
    
    # Split data
    X_train_fold = train_data.iloc[train_idx].copy()
    X_val_fold = train_data.iloc[val_idx].copy()
    y_train_fold = train_labels.iloc[train_idx]
    y_val_fold = train_labels.iloc[val_idx]
    
    # Standardize per fold
    fold_means = X_train_fold.mean()
    fold_stds = X_train_fold.std().replace(0, 1)
    X_train_fold = (X_train_fold - fold_means) / fold_stds
    X_val_fold = (X_val_fold - fold_means) / fold_stds
    
    # Store fold predictions and scores
    fold_val_preds = []
    fold_test_preds = []
    fold_roc_scores = []  # Store ROC scores for weighting
    
    # Model 1: XGBoost (using optimized hyperparameters from 50-trial run)
    if XGBOOST_AVAILABLE:
        print("  Training XGBoost (optimized params)...")
        start_time = time.time()
        # Using optimized parameters from hyperparameter_optimization.py (50 trials)
        # These are approximate - actual best params may vary slightly per fold
        model_xgb = xgb.XGBClassifier(
            n_estimators=1000, learning_rate=0.025, max_depth=9,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            gamma=0.3, reg_alpha=0.2, reg_lambda=2.0, random_state=2025,
            eval_metric='auc', use_label_encoder=False, verbosity=0,
            tree_method='hist', n_jobs=8
        )
        model_xgb.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)], verbose=False)
        val_pred_xgb = model_xgb.predict_proba(X_val_fold)[:, 1]
        test_pred_xgb = model_xgb.predict_proba(test_data)[:, 1]
        fold_val_preds.append(val_pred_xgb)
        fold_test_preds.append(test_pred_xgb)
        roc_xgb = sklearn.metrics.roc_auc_score(y_val_fold, val_pred_xgb)
        fold_roc_scores.append(roc_xgb)
        print(f"    ROC AUC: {roc_xgb:.6f} | Time: {time.time() - start_time:.1f}s")
        fold_scores.append(('XGBoost', roc_xgb))
    
    # Model 2: LightGBM (using optimized hyperparameters from 50-trial run)
    if LIGHTGBM_AVAILABLE:
        print("  Training LightGBM (optimized params)...")
        start_time = time.time()
        # Using best parameters from 50-trial optimization: Trial 31, ROC AUC 0.920327
        model_lgb = lgb.LGBMClassifier(
            n_estimators=1200, learning_rate=0.028624, max_depth=8,
            num_leaves=50, min_child_samples=10, subsample=0.75,
            colsample_bytree=0.85, reg_alpha=0.5, reg_lambda=1.0,
            random_state=2025, verbosity=-1, n_jobs=8
        )
        model_lgb.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)],
                      eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        val_pred_lgb = model_lgb.predict_proba(X_val_fold)[:, 1]
        test_pred_lgb = model_lgb.predict_proba(test_data)[:, 1]
        fold_val_preds.append(val_pred_lgb)
        fold_test_preds.append(test_pred_lgb)
        roc_lgb = sklearn.metrics.roc_auc_score(y_val_fold, val_pred_lgb)
        fold_roc_scores.append(roc_lgb)
        print(f"    ROC AUC: {roc_lgb:.6f} | Time: {time.time() - start_time:.1f}s")
        fold_scores.append(('LightGBM', roc_lgb))
    
    # Model 3: Gradient Boosting
    print("  Training Gradient Boosting...")
    start_time = time.time()
    model_gb = sklearn.ensemble.GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=8,
        min_samples_split=5, subsample=0.85, max_features='sqrt',
        random_state=2025, verbose=0
    )
    model_gb.fit(X_train_fold, y_train_fold)
    val_pred_gb = model_gb.predict_proba(X_val_fold)[:, 1]
    test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
    fold_val_preds.append(val_pred_gb)
    fold_test_preds.append(test_pred_gb)
    roc_gb = sklearn.metrics.roc_auc_score(y_val_fold, val_pred_gb)
    fold_roc_scores.append(roc_gb)
    print(f"    ROC AUC: {roc_gb:.6f} | Time: {time.time() - start_time:.1f}s")
    fold_scores.append(('GradientBoosting', roc_gb))
    
    # Ensemble fold predictions (weighted by individual ROC AUC scores)
    # Weight by performance: better models get more weight
    if len(fold_val_preds) > 1:
        # Calculate weights based on individual ROC AUC scores
        weights = np.array(fold_roc_scores)
        weights = weights / weights.sum()  # Normalize to sum to 1
        
        fold_val_ensemble = np.average(np.array(fold_val_preds), axis=0, weights=weights)
        fold_test_ensemble = np.average(np.array(fold_test_preds), axis=0, weights=weights)
        print(f"    Ensemble weights: {dict(zip(['XGB', 'LGB', 'GB'][:len(weights)], weights.round(3)))}")
    else:
        # Single model - no weighting needed
        fold_val_ensemble = np.array(fold_val_preds).mean(axis=0)
        fold_test_ensemble = np.array(fold_test_preds).mean(axis=0)
    
    # Store out-of-fold predictions
    oof_predictions[val_idx] = fold_val_ensemble
    test_predictions_list.append(fold_test_ensemble)
    
    fold_roc = sklearn.metrics.roc_auc_score(y_val_fold, fold_val_ensemble)
    print(f"  Fold {fold} Ensemble ROC AUC: {fold_roc:.6f}")

# Average test predictions across folds
test_predictions = np.array(test_predictions_list).mean(axis=0)

# Calculate overall CV score
cv_score = sklearn.metrics.roc_auc_score(train_labels, oof_predictions)

print("\n" + "="*80)
print("CROSS-VALIDATION RESULTS")
print("="*80)
print(f"Overall CV ROC AUC: {cv_score:.6f}")
print(f"  (Average across {N_FOLDS} folds)")

# Save submission
submission = pd.DataFrame({'id': ids, 'loan_paid_back': test_predictions})
submission.to_csv('submissions/my_submission_cv_ensemble_optimized.csv', index=False)
print(f"\n✓ Generated submission file: submissions/my_submission_cv_ensemble_optimized.csv")
print(f"  Using optimized hyperparameters + weighted averaging")
print(f"  Expected improvement: Should match or beat 0.92175 (optimized ensemble)")
print(f"  Target: 0.9218 - 0.9222")

