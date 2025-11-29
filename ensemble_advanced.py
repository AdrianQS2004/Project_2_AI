# Advanced Ensemble with Feature Engineering and Stacking
# Building on the successful ensemble method (0.92045)

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import sklearn.linear_model
import time

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception) as e:
    XGBOOST_AVAILABLE = False
    print(f"Note: XGBoost not available. Continuing without it...")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception) as e:
    LIGHTGBM_AVAILABLE = False
    print(f"Note: LightGBM not available. Continuing without it...")

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Note: scipy not available for weight optimization")

print("="*80)
print("ADVANCED ENSEMBLE WITH FEATURE ENGINEERING")
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

# Align test columns to match training columns
test_db = test_db.reindex(columns=training_db.columns, fill_value=0)

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()

# ============================================================
# FEATURE ENGINEERING
# ============================================================
print("\n[Feature Engineering] Creating interaction features...")

# Try to find numeric columns for feature engineering
# Look for common financial features
numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()

# Create interaction features if we can identify the right columns
# Note: After one-hot encoding, we need to be careful
# We'll create features based on common patterns

# Feature: income to loan ratio (if columns exist)
if 'annual_income' in train_data.columns and 'loan_amount' in train_data.columns:
    train_data['income_to_loan'] = train_data['annual_income'] / (train_data['loan_amount'] + 1)
    test_data['income_to_loan'] = test_data['annual_income'] / (test_data['loan_amount'] + 1)

# Feature: debt ratio interactions (if columns exist)
if 'debt_to_income_ratio' in train_data.columns and 'credit_score' in train_data.columns:
    train_data['debt_score_interaction'] = train_data['debt_to_income_ratio'] * train_data['credit_score']
    test_data['debt_score_interaction'] = test_data['debt_to_income_ratio'] * test_data['credit_score']
    
    train_data['debt_score_ratio'] = train_data['debt_to_income_ratio'] / (train_data['credit_score'] + 1)
    test_data['debt_score_ratio'] = test_data['debt_to_income_ratio'] / (test_data['credit_score'] + 1)

# Feature: interest rate interactions
if 'interest_rate' in train_data.columns and 'loan_amount' in train_data.columns:
    train_data['interest_loan_interaction'] = train_data['interest_rate'] * train_data['loan_amount']
    test_data['interest_loan_interaction'] = test_data['interest_rate'] * test_data['loan_amount']

print(f"  Original features: {len(training_db.columns)}")
print(f"  After feature engineering: {len(train_data.columns)}")

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

# Store predictions from each model
val_predictions = []
test_predictions = []
model_names = []

# ============================================================
# Model 1: Random Forest (tuned further)
# ============================================================
print("\n[1/5] Training Random Forest (enhanced)...")
start_time = time.time()

model_rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=400,  # Increased from 300
    max_depth=30,  # Increased from 25
    min_samples_split=2,  # More flexible
    min_samples_leaf=1,
    max_features='sqrt',  # Feature subsampling
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
# Model 2: Gradient Boosting (tuned further)
# ============================================================
print("\n[2/5] Training Gradient Boosting (enhanced)...")
start_time = time.time()

model_gb = sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=400,  # Increased from 300
    learning_rate=0.03,  # Lower learning rate
    max_depth=7,  # Slightly deeper
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.85,  # Slightly more data
    max_features='sqrt',  # Feature subsampling
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
# Model 3: XGBoost (tuned further)
# ============================================================
if XGBOOST_AVAILABLE:
    print("\n[3/5] Training XGBoost (enhanced)...")
    start_time = time.time()
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=600,  # Increased from 500
        learning_rate=0.03,  # Lower learning rate
        max_depth=8,  # Slightly deeper
        min_child_weight=2,  # More flexible
        subsample=0.85,
        colsample_bytree=0.85,
        colsample_bylevel=0.85,  # Additional feature subsampling
        gamma=0.2,  # More regularization
        reg_alpha=0.2,  # More L1
        reg_lambda=1.5,  # More L2
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
    print("\n[3/5] XGBoost skipped (not available)")

# ============================================================
# Model 4: LightGBM (tuned further)
# ============================================================
if LIGHTGBM_AVAILABLE:
    print("\n[4/5] Training LightGBM (enhanced)...")
    start_time = time.time()
    
    model_lgb = lgb.LGBMClassifier(
        n_estimators=600,  # Increased from 500
        learning_rate=0.03,  # Lower learning rate
        max_depth=8,  # Slightly deeper
        num_leaves=40,  # More leaves
        min_child_samples=15,  # More flexible
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.2,  # More L1
        reg_lambda=1.5,  # More L2
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
    print("\n[4/5] LightGBM skipped (not available)")

# ============================================================
# Model 5: Extra Trees (for diversity)
# ============================================================
print("\n[5/5] Training Extra Trees (for diversity)...")
start_time = time.time()

model_et = sklearn.ensemble.ExtraTreesClassifier(
    n_estimators=400,
    max_depth=30,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=2025,
    n_jobs=-1,
    verbose=0
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
# Combine Predictions
# ============================================================
print("\n" + "="*80)
print("ENSEMBLING PREDICTIONS")
print("="*80)

# Convert to numpy arrays
val_predictions = np.array(val_predictions).T
test_predictions = np.array(test_predictions).T

# Method 1: Simple Average
val_ensemble_avg = val_predictions.mean(axis=1)
test_ensemble_avg = test_predictions.mean(axis=1)
roc_avg = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_avg)

print(f"\nSimple Average:")
print(f"  Val ROC AUC: {roc_avg:.6f}")

# Method 2: Weighted Average
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

print(f"\nWeighted Average:")
print(f"  Weights: {dict(zip(model_names, weights))}")
print(f"  Val ROC AUC: {roc_weighted:.6f}")

# Method 3: Optimized weights
ensemble_methods = {
    'Simple Average': (roc_avg, test_ensemble_avg),
    'Weighted Average': (roc_weighted, test_ensemble_weighted)
}

if SCIPY_AVAILABLE:
    print(f"\nOptimizing weights on validation set...")
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
        
        print(f"\nOptimized Weights:")
        for name, weight in zip(model_names, opt_weights):
            print(f"  {name}: {weight:.4f}")
        print(f"  Val ROC AUC: {roc_opt:.6f}")
        
        ensemble_methods['Optimized Weights'] = (roc_opt, test_ensemble_opt)
    except Exception as e:
        print(f"  Optimization failed: {e}")

# Method 4: Stacking with Logistic Regression meta-learner
print(f"\nTraining stacking meta-learner...")
try:
    # Use base model predictions as features for meta-learner
    meta_train = val_predictions.copy()
    meta_test = test_predictions.copy()
    
    meta_model = sklearn.linear_model.LogisticRegression(
        max_iter=1000,
        random_state=2025,
        C=1.0
    )
    meta_model.fit(meta_train, val_labels)
    
    val_ensemble_stack = meta_model.predict_proba(meta_train)[:, 1]
    test_ensemble_stack = meta_model.predict_proba(meta_test)[:, 1]
    roc_stack = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_stack)
    
    print(f"  Val ROC AUC: {roc_stack:.6f}")
    print(f"  Meta-learner coefficients: {meta_model.coef_[0]}")
    
    ensemble_methods['Stacking (Logistic)'] = (roc_stack, test_ensemble_stack)
except Exception as e:
    print(f"  Stacking failed: {e}")

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
submission.to_csv('submissions/my_submission_advanced.csv', index=False)
print(f"\n✓ Generated submission files:")
print(f"  - submissions/my_submission.csv (main file)")
print(f"  - submissions/my_submission_advanced.csv (descriptive name)")
print(f"  Using {best_name}")
print(f"  Expected improvement over 0.92045: +0.001 to +0.003")

