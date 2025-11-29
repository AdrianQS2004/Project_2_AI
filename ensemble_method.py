# Ensemble Method: Combine Multiple Models
# This script trains multiple models and combines their predictions
# Often improves ROC AUC by 0.001-0.005 over single best model

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import time

try:
    import xgboost as xgb
    # Test if XGBoost actually works by creating a simple classifier
    _ = xgb.XGBClassifier()
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Note: XGBoost not available ({type(e).__name__}: {str(e)[:100]}). Continuing without it...")

try:
    import lightgbm as lgb
    # Test if LightGBM actually works
    _ = lgb.LGBMClassifier()
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Note: LightGBM not available ({type(e).__name__}: {str(e)[:100]}). Continuing without it...")

# Load and prepare the data
training_db = pd.read_csv("Datasets/train.csv", header=0)
test_db = pd.read_csv("Datasets/test.csv", header=0)

# One-hot encode categorical variables
training_db = pd.get_dummies(training_db, prefix_sep="_", drop_first=True, dtype=int)
labels = training_db["loan_paid_back"]
ids = test_db['id']
training_db = training_db.drop(columns=["loan_paid_back", "id"])

test_db = test_db.drop(columns=["id"])
test_db = pd.get_dummies(test_db, prefix_sep="_", drop_first=True, dtype=int)

# Align test columns to match training columns
test_db = test_db.reindex(columns=training_db.columns, fill_value=0)

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()

# Split training data for validation
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

print("="*80)
print("ENSEMBLE METHOD: Combining Multiple Models")
print("="*80)

# Store predictions from each model
val_predictions = []
test_predictions = []
model_names = []

# ============================================================
# Model 1: Random Forest (tuned)
# ============================================================
print("\n[1/4] Training Random Forest...")
start_time = time.time()

model_rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    min_samples_split=3,
    min_samples_leaf=1,
    random_state=2025,
    n_jobs=-1,
    verbose=0
)
model_rf.fit(train_data_split, train_labels_split)

val_pred_rf = model_rf.predict_proba(val_data)[:, 1]
test_pred_rf = model_rf.predict_proba(test_data)[:, 1]
val_roc_rf = sklearn.metrics.roc_auc_score(val_labels, val_pred_rf)

val_predictions.append(val_pred_rf)
test_predictions.append(test_pred_rf)
model_names.append("RandomForest")

print(f"  Val ROC AUC: {val_roc_rf:.6f} | Time: {time.time() - start_time:.1f}s")

# ============================================================
# Model 2: Gradient Boosting (tuned)
# ============================================================
print("\n[2/4] Training Gradient Boosting...")
start_time = time.time()

model_gb = sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    random_state=2025,
    verbose=0
)
model_gb.fit(train_data_split, train_labels_split)

val_pred_gb = model_gb.predict_proba(val_data)[:, 1]
test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
val_roc_gb = sklearn.metrics.roc_auc_score(val_labels, val_pred_gb)

val_predictions.append(val_pred_gb)
test_predictions.append(test_pred_gb)
model_names.append("GradientBoosting")

print(f"  Val ROC AUC: {val_roc_gb:.6f} | Time: {time.time() - start_time:.1f}s")

# ============================================================
# Model 3: XGBoost (tuned)
# ============================================================
if XGBOOST_AVAILABLE:
    print("\n[3/4] Training XGBoost...")
    start_time = time.time()
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=2025,
        eval_metric='auc',
        use_label_encoder=False,
        n_jobs=-1,
        verbosity=0
    )
    
    model_xgb.fit(
        train_data_split, train_labels_split,
        eval_set=[(val_data, val_labels)],
        verbose=False
    )
    
    val_pred_xgb = model_xgb.predict_proba(val_data)[:, 1]
    test_pred_xgb = model_xgb.predict_proba(test_data)[:, 1]
    val_roc_xgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_xgb)
    
    val_predictions.append(val_pred_xgb)
    test_predictions.append(test_pred_xgb)
    model_names.append("XGBoost")
    
    print(f"  Val ROC AUC: {val_roc_xgb:.6f} | Time: {time.time() - start_time:.1f}s")
else:
    print("\n[3/4] XGBoost skipped (not installed)")

# ============================================================
# Model 4: LightGBM (tuned)
# ============================================================
if LIGHTGBM_AVAILABLE:
    print("\n[4/4] Training LightGBM...")
    start_time = time.time()
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=7,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=2025,
        n_jobs=-1,
        verbosity=-1
    )
    
    model_lgb.fit(
        train_data_split, train_labels_split,
        eval_set=[(val_data, val_labels)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    val_pred_lgb = model_lgb.predict_proba(val_data)[:, 1]
    test_pred_lgb = model_lgb.predict_proba(test_data)[:, 1]
    val_roc_lgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_lgb)
    
    val_predictions.append(val_pred_lgb)
    test_predictions.append(test_pred_lgb)
    model_names.append("LightGBM")
    
    print(f"  Val ROC AUC: {val_roc_lgb:.6f} | Time: {time.time() - start_time:.1f}s")
else:
    print("\n[4/4] LightGBM skipped (not installed)")

# ============================================================
# Combine Predictions
# ============================================================
print("\n" + "="*80)
print("ENSEMBLING PREDICTIONS")
print("="*80)

# Convert to numpy arrays
val_predictions = np.array(val_predictions).T  # Shape: (n_samples, n_models)
test_predictions = np.array(test_predictions).T

# Method 1: Simple Average
val_ensemble_avg = val_predictions.mean(axis=1)
test_ensemble_avg = test_predictions.mean(axis=1)
roc_avg = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_avg)

print(f"\nSimple Average:")
print(f"  Val ROC AUC: {roc_avg:.6f}")

# Method 2: Weighted Average (weight by individual performance)
weights = np.array([val_roc_rf, val_roc_gb])
if XGBOOST_AVAILABLE:
    weights = np.append(weights, val_roc_xgb)
if LIGHTGBM_AVAILABLE:
    weights = np.append(weights, val_roc_lgb)

# Normalize weights
weights = weights / weights.sum()

val_ensemble_weighted = np.average(val_predictions, axis=1, weights=weights)
test_ensemble_weighted = np.average(test_predictions, axis=1, weights=weights)
roc_weighted = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_weighted)

print(f"\nWeighted Average (by individual ROC AUC):")
print(f"  Weights: {dict(zip(model_names, weights))}")
print(f"  Val ROC AUC: {roc_weighted:.6f}")

# Initialize ensemble methods dictionary
ensemble_methods = {
    'Simple Average': (roc_avg, test_ensemble_avg),
    'Weighted Average': (roc_weighted, test_ensemble_weighted)
}

# Method 3: Optimize weights using validation set
print(f"\nOptimizing weights on validation set...")
roc_opt = None
test_ensemble_opt = None
try:
    from scipy.optimize import minimize
    
    def objective(weights):
        """Minimize negative ROC AUC (maximize ROC AUC)"""
        ensemble_pred = np.average(val_predictions, axis=1, weights=weights)
        roc = sklearn.metrics.roc_auc_score(val_labels, ensemble_pred)
        return -roc  # Minimize negative = maximize
    
    # Constraint: weights sum to 1
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}
    bounds = [(0, 1)] * len(model_names)
    initial_weights = np.ones(len(model_names)) / len(model_names)
    
    result = minimize(objective, initial_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    
    opt_weights = result.x
    val_ensemble_opt = np.average(val_predictions, axis=1, weights=opt_weights)
    test_ensemble_opt = np.average(test_predictions, axis=1, weights=opt_weights)
    roc_opt = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_opt)
    
    print(f"\nOptimized Weights:")
    for name, weight in zip(model_names, opt_weights):
        print(f"  {name}: {weight:.4f}")
    print(f"  Val ROC AUC: {roc_opt:.6f}")
    
    ensemble_methods['Optimized Weights'] = (roc_opt, test_ensemble_opt)
except ImportError:
    print("  (scipy not available, skipping optimized weights)")
    print("  Install with: pip install scipy")
except Exception as e:
    print(f"  (Optimization failed: {e}, using weighted average instead)")

# Choose best ensemble method

best_method = max(ensemble_methods.items(), key=lambda x: x[1][0])
best_name, (best_roc, best_test_pred) = best_method

print("\n" + "="*80)
print(f"BEST ENSEMBLE METHOD: {best_name}")
print(f"  Validation ROC AUC: {best_roc:.6f}")
print("="*80)

# Generate submission
submission = pd.DataFrame({'id': ids, 'loan_paid_back': best_test_pred})
submission.to_csv('submissions/my_submission.csv', index=False)
print(f"\n✓ Generated submission 'submissions/my_submission.csv' using {best_name}")
print(f"  Expected improvement: +0.001 to +0.005 over single best model")

