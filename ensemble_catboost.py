# Ensemble with CatBoost (often performs better than XGBoost/LightGBM)
# CatBoost handles categorical features natively and often gives better results

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import time

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("CatBoost not available. Install with: pip install catboost")

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

if not CATBOOST_AVAILABLE:
    print("Please install CatBoost: pip install catboost")
    exit(1)

print("="*80)
print("ENSEMBLE WITH CATBOOST")
print("="*80)

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

# Align test columns
test_db = test_db.reindex(columns=training_db.columns, fill_value=0)

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()

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
# Model 1: CatBoost (often best single model)
# ============================================================
print("\n[1/4] Training CatBoost...")
start_time = time.time()

model_cat = CatBoostClassifier(
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

model_cat.fit(
    train_data_split, train_labels_split,
    eval_set=(val_data, val_labels),
    early_stopping_rounds=50,
    verbose=False
)

val_pred_cat = model_cat.predict_proba(val_data)[:, 1]
test_pred_cat = model_cat.predict_proba(test_data)[:, 1]
val_roc_cat = sklearn.metrics.roc_auc_score(val_labels, val_pred_cat)

val_predictions.append(val_pred_cat)
test_predictions.append(test_pred_cat)
model_names.append("CatBoost")

print(f"  Val ROC AUC: {val_roc_cat:.6f} | Time: {time.time() - start_time:.1f}s")

# ============================================================
# Model 2: XGBoost
# ============================================================
if XGBOOST_AVAILABLE:
    print("\n[2/4] Training XGBoost...")
    start_time = time.time()
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.5,
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

# ============================================================
# Model 3: LightGBM
# ============================================================
if LIGHTGBM_AVAILABLE:
    print("\n[3/4] Training LightGBM...")
    start_time = time.time()
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=40,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,
        reg_lambda=1.5,
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

# ============================================================
# Model 4: Gradient Boosting
# ============================================================
print("\n[4/4] Training Gradient Boosting...")
start_time = time.time()

model_gb = sklearn.ensemble.GradientBoostingClassifier(
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
model_gb.fit(train_data_split, train_labels_split)

val_pred_gb = model_gb.predict_proba(val_data)[:, 1]
test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
val_roc_gb = sklearn.metrics.roc_auc_score(val_labels, val_pred_gb)

val_predictions.append(val_pred_gb)
test_predictions.append(test_pred_gb)
model_names.append("GradientBoosting")

print(f"  Val ROC AUC: {val_roc_gb:.6f} | Time: {time.time() - start_time:.1f}s")

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
weights = np.array([val_roc_cat])
if XGBOOST_AVAILABLE:
    weights = np.append(weights, val_roc_xgb)
if LIGHTGBM_AVAILABLE:
    weights = np.append(weights, val_roc_lgb)
weights = np.append(weights, val_roc_gb)
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
        
        result = minimize(objective, initial_weights, method='SLSQP', 
                         bounds=bounds, constraints=constraints)
        
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

# Best method
best_method = max(ensemble_methods.items(), key=lambda x: x[1][0])
best_name, (best_roc, best_test_pred) = best_method

print("\n" + "="*80)
print(f"BEST ENSEMBLE METHOD: {best_name}")
print(f"  Validation ROC AUC: {best_roc:.6f}")
print("="*80)

submission = pd.DataFrame({'id': ids, 'loan_paid_back': best_test_pred})
submission.to_csv('submissions/my_submission.csv', index=False)
submission.to_csv('submissions/my_submission_catboost.csv', index=False)
print(f"\n✓ Generated submission files:")
print(f"  - submissions/my_submission.csv (main file)")
print(f"  - submissions/my_submission_catboost.csv (descriptive name)")
print(f"  Expected improvement: +0.001 to +0.003 over 0.92045")

