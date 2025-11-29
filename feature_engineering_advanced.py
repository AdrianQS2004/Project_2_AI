# Advanced Feature Engineering - Create More Interaction Features
# Building on successful Advanced Ensemble (0.92088)

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

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

print("="*80)
print("ADVANCED FEATURE ENGINEERING ENSEMBLE")
print("="*80)

# Load data
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

# ============================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================
print("\n[Advanced Feature Engineering] Creating interaction features...")

# Find numeric columns (before one-hot encoding, we need original names)
# We'll work with the encoded data and try to identify original features
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()

# Try to identify original feature columns by common patterns
def find_col(pattern, cols):
    matches = [c for c in cols if pattern.lower() in c.lower()]
    return matches[0] if matches else None

# Create many more interaction features
features_created = 0

# 1. Income-related features
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

# 2. Debt and credit interactions
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

# 3. Interest rate interactions
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

# 4. Polynomial features (squared terms for key features)
key_features = []
for pattern in ['annual_income', 'credit_score', 'loan_amount', 'debt_to_income', 'interest_rate']:
    col = find_col(pattern, train_data.columns)
    if col:
        key_features.append(col)

for col in key_features[:3]:  # Limit to avoid too many features
    train_data[f'{col}_squared'] = train_data[col] ** 2
    test_data[f'{col}_squared'] = test_data[col] ** 2
    features_created += 1

# 5. Log transformations (for skewed features)
for col in key_features[:2]:
    train_data[f'{col}_log'] = np.log1p(np.abs(train_data[col]))
    test_data[f'{col}_log'] = np.log1p(np.abs(test_data[col]))
    features_created += 1

print(f"  Original features: {len(training_db.columns)}")
print(f"  Features created: {features_created}")
print(f"  Total features: {len(train_data.columns)}")

# Split for validation
train_data_split, val_data, train_labels_split, val_labels = \
    sklearn.model_selection.train_test_split(
        train_data, train_labels,
        test_size=0.2, shuffle=True, random_state=2025
    )

# Standardize
train_means = train_data_split.mean()
train_stds = train_data_split.std().replace(0, 1)
train_data_split = (train_data_split - train_means) / train_stds
val_data = (val_data - train_means) / train_stds
test_data = (test_data - train_means) / train_stds

val_predictions = []
test_predictions = []
model_names = []

# ============================================================
# Train Models with Enhanced Features
# ============================================================

# Model 1: Random Forest
print("\n[1/5] Training Random Forest...")
start_time = time.time()
model_rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=500, max_depth=30, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=2025, n_jobs=-1, verbose=0
)
model_rf.fit(train_data_split, train_labels_split)
val_pred_rf = model_rf.predict_proba(val_data)[:, 1]
test_pred_rf = model_rf.predict_proba(test_data)[:, 1]
val_roc_rf = sklearn.metrics.roc_auc_score(val_labels, val_pred_rf)
val_predictions.append(val_pred_rf)
test_predictions.append(test_pred_rf)
model_names.append("RandomForest")
print(f"  Val ROC AUC: {val_roc_rf:.6f} | Time: {time.time() - start_time:.1f}s")

# Model 2: Gradient Boosting
print("\n[2/5] Training Gradient Boosting...")
start_time = time.time()
model_gb = sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.02, max_depth=8,
    min_samples_split=5, subsample=0.85, max_features='sqrt', random_state=2025, verbose=0
)
model_gb.fit(train_data_split, train_labels_split)
val_pred_gb = model_gb.predict_proba(val_data)[:, 1]
test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
val_roc_gb = sklearn.metrics.roc_auc_score(val_labels, val_pred_gb)
val_predictions.append(val_pred_gb)
test_predictions.append(test_pred_gb)
model_names.append("GradientBoosting")
print(f"  Val ROC AUC: {val_roc_gb:.6f} | Time: {time.time() - start_time:.1f}s")

# Model 3: XGBoost
if XGBOOST_AVAILABLE:
    print("\n[3/5] Training XGBoost...")
    start_time = time.time()
    model_xgb = xgb.XGBClassifier(
        n_estimators=700, learning_rate=0.02, max_depth=9,
        min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
        gamma=0.3, reg_alpha=0.2, reg_lambda=2.0, random_state=2025,
        eval_metric='auc', use_label_encoder=False, n_jobs=-1, verbosity=0
    )
    model_xgb.fit(train_data_split, train_labels_split, eval_set=[(val_data, val_labels)], verbose=False)
    val_pred_xgb = model_xgb.predict_proba(val_data)[:, 1]
    test_pred_xgb = model_xgb.predict_proba(test_data)[:, 1]
    val_roc_xgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_xgb)
    val_predictions.append(val_pred_xgb)
    test_predictions.append(test_pred_xgb)
    model_names.append("XGBoost")
    print(f"  Val ROC AUC: {val_roc_xgb:.6f} | Time: {time.time() - start_time:.1f}s")

# Model 4: LightGBM
if LIGHTGBM_AVAILABLE:
    print("\n[4/5] Training LightGBM...")
    start_time = time.time()
    model_lgb = lgb.LGBMClassifier(
        n_estimators=700, learning_rate=0.02, max_depth=9,
        num_leaves=50, min_child_samples=15, subsample=0.85,
        colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0,
        random_state=2025, n_jobs=-1, verbosity=-1
    )
    model_lgb.fit(train_data_split, train_labels_split, eval_set=[(val_data, val_labels)],
                  eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
    val_pred_lgb = model_lgb.predict_proba(val_data)[:, 1]
    test_pred_lgb = model_lgb.predict_proba(test_data)[:, 1]
    val_roc_lgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_lgb)
    val_predictions.append(val_pred_lgb)
    test_predictions.append(test_pred_lgb)
    model_names.append("LightGBM")
    print(f"  Val ROC AUC: {val_roc_lgb:.6f} | Time: {time.time() - start_time:.1f}s")

# Model 5: Extra Trees
print("\n[5/5] Training Extra Trees...")
start_time = time.time()
model_et = sklearn.ensemble.ExtraTreesClassifier(
    n_estimators=500, max_depth=30, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=2025, n_jobs=-1, verbose=0
)
model_et.fit(train_data_split, train_labels_split)
val_pred_et = model_et.predict_proba(val_data)[:, 1]
test_pred_et = model_et.predict_proba(test_data)[:, 1]
val_roc_et = sklearn.metrics.roc_auc_score(val_labels, val_pred_et)
val_predictions.append(val_pred_et)
test_predictions.append(test_pred_et)
model_names.append("ExtraTrees")
print(f"  Val ROC AUC: {val_roc_et:.6f} | Time: {time.time() - start_time:.1f}s")

# ============================================================
# Ensemble
# ============================================================
print("\n" + "="*80)
print("ENSEMBLING PREDICTIONS")
print("="*80)

val_predictions = np.array(val_predictions).T
test_predictions = np.array(test_predictions).T

# Simple average
val_ensemble_avg = val_predictions.mean(axis=1)
test_ensemble_avg = test_predictions.mean(axis=1)
roc_avg = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_avg)
print(f"\nSimple Average: {roc_avg:.6f}")

# Weighted average
weights = np.array([val_roc_rf, val_roc_gb])
if XGBOOST_AVAILABLE:
    weights = np.append(weights, val_roc_xgb)
if LIGHTGBM_AVAILABLE:
    weights = np.append(weights, val_roc_lgb)
weights = np.append(weights, val_roc_et)
weights = weights / weights.sum()

val_ensemble_weighted = np.average(val_predictions, axis=1, weights=weights)
test_ensemble_weighted = np.average(test_predictions, axis=1, weights=weights)
roc_weighted = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_weighted)
print(f"Weighted Average: {roc_weighted:.6f}")
print(f"  Weights: {dict(zip(model_names, weights))}")

# Optimized weights
ensemble_methods = {
    'Simple Average': (roc_avg, test_ensemble_avg),
    'Weighted Average': (roc_weighted, test_ensemble_weighted)
}

if SCIPY_AVAILABLE:
    try:
        def objective(weights):
            ensemble_pred = np.average(val_predictions, axis=1, weights=weights)
            roc = sklearn.metrics.roc_auc_score(val_labels, ensemble_pred)
            return -roc
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
        bounds = [(0, 1)] * len(model_names)
        initial_weights = np.ones(len(model_names)) / len(model_names)
        
        result = minimize(objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_weights = result.x
        val_ensemble_opt = np.average(val_predictions, axis=1, weights=opt_weights)
        test_ensemble_opt = np.average(test_predictions, axis=1, weights=opt_weights)
        roc_opt = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_opt)
        
        print(f"\nOptimized Weights: {roc_opt:.6f}")
        for name, weight in zip(model_names, opt_weights):
            print(f"  {name}: {weight:.4f}")
        
        ensemble_methods['Optimized Weights'] = (roc_opt, test_ensemble_opt)
    except Exception as e:
        print(f"  Optimization failed: {e}")

# Stacking
try:
    meta_model = sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=2025, C=1.0)
    meta_model.fit(val_predictions, val_labels)
    val_ensemble_stack = meta_model.predict_proba(val_predictions)[:, 1]
    test_ensemble_stack = meta_model.predict_proba(test_predictions)[:, 1]
    roc_stack = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_stack)
    print(f"\nStacking (Logistic): {roc_stack:.6f}")
    ensemble_methods['Stacking'] = (roc_stack, test_ensemble_stack)
except Exception as e:
    print(f"  Stacking failed: {e}")

# Best method
best_method = max(ensemble_methods.items(), key=lambda x: x[1][0])
best_name, (best_roc, best_test_pred) = best_method

print("\n" + "="*80)
print(f"BEST METHOD: {best_name}")
print(f"  Validation ROC AUC: {best_roc:.6f}")
print("="*80)

submission = pd.DataFrame({'id': ids, 'loan_paid_back': best_test_pred})
submission.to_csv('submissions/my_submission.csv', index=False)
submission.to_csv('submissions/my_submission_advanced_features.csv', index=False)  # Also save with descriptive name
print(f"\n✓ Generated submission files:")
print(f"  - submissions/my_submission.csv (main file)")
print(f"  - submissions/my_submission_advanced_features.csv (backup with descriptive name)")
print(f"  Expected improvement: +0.001 to +0.003 over 0.92088")

