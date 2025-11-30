# Hyperparameter Optimization for Best Models
# Uses Optuna (Bayesian optimization) to tune XGBoost, LightGBM, and GradientBoosting
# Target: Improve from 0.92100 to 0.922-0.924

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import time

# CPU Configuration: For background operation without overheating
# AMD 9800X3D has 12 cores - using 8 cores leaves 4 for system/background tasks
# This prevents overheating while still being efficient
num_jobs = 8  # Use 8 cores (leaves 4 cores free for system)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except (ImportError, Exception):
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, Exception):
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available. Install with: pip install lightgbm")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except (ImportError, Exception):
    OPTUNA_AVAILABLE = False
    print("Optuna not available. Install with: pip install optuna")
    print("Will use GridSearchCV instead (slower but works)")

print("="*80)
print("HYPERPARAMETER OPTIMIZATION")
print("="*80)
if OPTUNA_AVAILABLE:
    print("Using Optuna (Bayesian optimization) - Fast and efficient")
else:
    print("Using GridSearchCV - Slower but works without Optuna")

# Configuration: Adjust trials for speed vs thoroughness
# - 20 trials: ~1-1.5 hours (XGB/LGB), but GB is much slower (sequential)
# - 50 trials: ~2-4 hours (XGB/LGB), GB will take 10-15+ hours
N_TRIALS = 50  # Set to 50 for thorough search (best results), or 20 for faster
# Gradient Boosting is sequential and MUCH slower - use fewer trials
# RECOMMENDATION: Set to False to skip GB optimization (saves 10-15+ hours)
# GB is the weakest model (0.915 vs 0.918-0.919 for XGB/LGB), so optimizing it has less impact
OPTIMIZE_GB = False  # Set to True if you want to optimize GB (will take 10-15+ hours)
GB_TRIALS = 10  # Only used if OPTIMIZE_GB = True

if OPTIMIZE_GB:
    print(f"\nConfiguration: {N_TRIALS} trials per model (XGBoost, LightGBM)")
    print(f"              {GB_TRIALS} trials for Gradient Boosting (slower, sequential)")
    print(f"  Total trials: {N_TRIALS * 2 + GB_TRIALS} (XGB + LGB + GB)")
    print(f"  Estimated time: XGB/LGB ~2-3 hours, GB ~{GB_TRIALS * 0.5:.1f}-{GB_TRIALS * 0.7:.1f} hours")
    print(f"  WARNING: GB is sequential - each trial takes 20-40 minutes!")
else:
    print(f"\nConfiguration: {N_TRIALS} trials per model (XGBoost, LightGBM)")
    print(f"              Gradient Boosting: Using default parameters (skipping optimization)")
    print(f"  Total trials: {N_TRIALS * 2} (XGB + LGB only)")
    print(f"  Estimated time: ~2-3 hours (GB optimization skipped)")
    print(f"  Reason: GB is weakest model (0.915) and too slow to optimize (20-40 min/trial)")
print(f"  CPU usage: {num_jobs} cores (XGB/LGB), 1 core (GB if optimizing) - safe for background operation")

# Submission filename - automatically includes trial count
SUBMISSION_FILENAME = f'submissions/my_submission_optimized{N_TRIALS}trials.csv'

# Load data
print("\n[1/5] Loading and preprocessing data...")
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
print("\n[2/5] Creating advanced features...")
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

print(f"\n[3/5] Training data: {len(train_data_split)} samples")
print(f"      Validation data: {len(val_data)} samples")

# Store best models
best_models = {}
best_scores = {}

# ============================================================
# Model 1: XGBoost Optimization
# ============================================================
if XGBOOST_AVAILABLE:
    print("\n" + "="*80)
    print("OPTIMIZING XGBOOST")
    print("="*80)
    
    if OPTUNA_AVAILABLE:
        def objective_xgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 600, 1200, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, log=True),
                'max_depth': trial.suggest_int('max_depth', 7, 11),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 5),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95, step=0.05),
                'gamma': trial.suggest_float('gamma', 0.1, 0.5, step=0.1),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.5, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0, step=0.5),
                'random_state': 2025,
                'eval_metric': 'auc',
                'use_label_encoder': False,
                'verbosity': 0,
                'tree_method': 'hist',
                'n_jobs': num_jobs
            }
            
            model = xgb.XGBClassifier(**params)
            model.fit(
                train_data_split, train_labels_split,
                eval_set=[(val_data, val_labels)],
                verbose=False
            )
            
            val_pred = model.predict_proba(val_data)[:, 1]
            roc_auc = sklearn.metrics.roc_auc_score(val_labels, val_pred)
            return roc_auc
        
        study_xgb = optuna.create_study(direction='maximize', study_name='xgb_optimization')
        print(f"  Running Optuna optimization ({N_TRIALS} trials)...")
        print("  This may take a while - each trial trains a full model...")
        start_time = time.time()
        study_xgb.optimize(objective_xgb, n_trials=N_TRIALS, show_progress_bar=True)
        elapsed = time.time() - start_time
        
        print(f"\n  Best XGBoost ROC AUC: {study_xgb.best_value:.6f}")
        print(f"  Optimization time: {elapsed:.1f}s")
        print(f"  Best parameters:")
        for key, value in study_xgb.best_params.items():
            print(f"    {key}: {value}")
        
        # Train best model
        best_params_xgb = study_xgb.best_params.copy()
        best_params_xgb.update({
            'random_state': 2025,
            'eval_metric': 'auc',
            'use_label_encoder': False,
            'verbosity': 0,
            'tree_method': 'hist',
            'n_jobs': num_jobs
        })
        
        model_xgb = xgb.XGBClassifier(**best_params_xgb)
        model_xgb.fit(
            train_data_split, train_labels_split,
            eval_set=[(val_data, val_labels)],
            verbose=False
        )
        
        val_pred_xgb = model_xgb.predict_proba(val_data)[:, 1]
        test_pred_xgb = model_xgb.predict_proba(test_data)[:, 1]
        val_roc_xgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_xgb)
        
        best_models['XGBoost'] = model_xgb
        best_scores['XGBoost'] = val_roc_xgb
        best_models['XGBoost_test'] = test_pred_xgb
        
        print(f"  Final validation ROC AUC: {val_roc_xgb:.6f}")
    else:
        print("  Optuna not available, using default XGBoost parameters")
        model_xgb = xgb.XGBClassifier(
            n_estimators=700, learning_rate=0.02, max_depth=9,
            min_child_weight=2, subsample=0.85, colsample_bytree=0.85,
            gamma=0.3, reg_alpha=0.2, reg_lambda=2.0, random_state=2025,
            eval_metric='auc', use_label_encoder=False, verbosity=0,
            tree_method='hist', n_jobs=num_jobs
        )
        model_xgb.fit(train_data_split, train_labels_split, eval_set=[(val_data, val_labels)], verbose=False)
        val_pred_xgb = model_xgb.predict_proba(val_data)[:, 1]
        test_pred_xgb = model_xgb.predict_proba(test_data)[:, 1]
        val_roc_xgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_xgb)
        best_models['XGBoost'] = model_xgb
        best_scores['XGBoost'] = val_roc_xgb
        best_models['XGBoost_test'] = test_pred_xgb
        print(f"  Validation ROC AUC: {val_roc_xgb:.6f}")

# ============================================================
# Model 2: LightGBM Optimization
# ============================================================
if LIGHTGBM_AVAILABLE:
    print("\n" + "="*80)
    print("OPTIMIZING LIGHTGBM")
    print("="*80)
    
    if OPTUNA_AVAILABLE:
        def objective_lgb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 600, 1200, step=100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, log=True),
                'max_depth': trial.suggest_int('max_depth', 7, 11),
                'num_leaves': trial.suggest_int('num_leaves', 30, 70, step=10),
                'min_child_samples': trial.suggest_int('min_child_samples', 10, 25, step=5),
                'subsample': trial.suggest_float('subsample', 0.75, 0.95, step=0.05),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.75, 0.95, step=0.05),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 0.5, step=0.1),
                'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0, step=0.5),
                'random_state': 2025,
                'verbosity': -1,
                'n_jobs': num_jobs
            }
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                train_data_split, train_labels_split,
                eval_set=[(val_data, val_labels)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
            )
            
            val_pred = model.predict_proba(val_data)[:, 1]
            roc_auc = sklearn.metrics.roc_auc_score(val_labels, val_pred)
            return roc_auc
        
        study_lgb = optuna.create_study(direction='maximize', study_name='lgb_optimization')
        print(f"  Running Optuna optimization ({N_TRIALS} trials)...")
        print("  This may take a while - each trial trains a full model...")
        start_time = time.time()
        study_lgb.optimize(objective_lgb, n_trials=N_TRIALS, show_progress_bar=True)
        elapsed = time.time() - start_time
        
        print(f"\n  Best LightGBM ROC AUC: {study_lgb.best_value:.6f}")
        print(f"  Optimization time: {elapsed:.1f}s")
        print(f"  Best parameters:")
        for key, value in study_lgb.best_params.items():
            print(f"    {key}: {value}")
        
        # Train best model
        best_params_lgb = study_lgb.best_params.copy()
        best_params_lgb.update({
            'random_state': 2025,
            'verbosity': -1,
            'n_jobs': num_jobs
        })
        
        model_lgb = lgb.LGBMClassifier(**best_params_lgb)
        model_lgb.fit(
            train_data_split, train_labels_split,
            eval_set=[(val_data, val_labels)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        val_pred_lgb = model_lgb.predict_proba(val_data)[:, 1]
        test_pred_lgb = model_lgb.predict_proba(test_data)[:, 1]
        val_roc_lgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_lgb)
        
        best_models['LightGBM'] = model_lgb
        best_scores['LightGBM'] = val_roc_lgb
        best_models['LightGBM_test'] = test_pred_lgb
        
        print(f"  Final validation ROC AUC: {val_roc_lgb:.6f}")
    else:
        print("  Optuna not available, using default LightGBM parameters")
        model_lgb = lgb.LGBMClassifier(
            n_estimators=700, learning_rate=0.02, max_depth=9,
            num_leaves=50, min_child_samples=15, subsample=0.85,
            colsample_bytree=0.85, reg_alpha=0.2, reg_lambda=2.0,
            random_state=2025, verbosity=-1, n_jobs=num_jobs
        )
        model_lgb.fit(train_data_split, train_labels_split, eval_set=[(val_data, val_labels)],
                      eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)])
        val_pred_lgb = model_lgb.predict_proba(val_data)[:, 1]
        test_pred_lgb = model_lgb.predict_proba(test_data)[:, 1]
        val_roc_lgb = sklearn.metrics.roc_auc_score(val_labels, val_pred_lgb)
        best_models['LightGBM'] = model_lgb
        best_scores['LightGBM'] = val_roc_lgb
        best_models['LightGBM_test'] = test_pred_lgb
        print(f"  Validation ROC AUC: {val_roc_lgb:.6f}")

# ============================================================
# Model 3: Gradient Boosting Optimization (Optional)
# ============================================================
if OPTIMIZE_GB:
    print("\n" + "="*80)
    print("OPTIMIZING GRADIENT BOOSTING")
    print("="*80)
    
    if OPTUNA_AVAILABLE:
        def objective_gb(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 200, 500, step=100),  # Reduced from 400-800
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.03, log=True),
                'max_depth': trial.suggest_int('max_depth', 6, 9),  # Reduced from 6-10
                'min_samples_split': trial.suggest_int('min_samples_split', 5, 8, step=3),  # Reduced range
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 3),  # Reduced from 1-4
                'subsample': trial.suggest_float('subsample', 0.8, 0.95, step=0.05),  # Reduced range
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),  # Reduced options
                'random_state': 2025,
                'verbose': 0
            }
            
            model = sklearn.ensemble.GradientBoostingClassifier(**params)
            model.fit(train_data_split, train_labels_split)
            
            val_pred = model.predict_proba(val_data)[:, 1]
            roc_auc = sklearn.metrics.roc_auc_score(val_labels, val_pred)
            return roc_auc
        
        study_gb = optuna.create_study(direction='maximize', study_name='gb_optimization')
        print(f"  Running Optuna optimization ({GB_TRIALS} trials)...")
        print("  WARNING: Gradient Boosting is sequential (single-core) and much slower!")
        print(f"  Each trial may take 20-40 minutes. Total time: ~{GB_TRIALS * 0.5:.1f}-{GB_TRIALS * 0.7:.1f} hours")
        print("  Consider reducing GB_TRIALS or skipping GB optimization if too slow")
        start_time = time.time()
        study_gb.optimize(objective_gb, n_trials=GB_TRIALS, show_progress_bar=True)
        elapsed = time.time() - start_time
        
        print(f"\n  Best Gradient Boosting ROC AUC: {study_gb.best_value:.6f}")
        print(f"  Optimization time: {elapsed:.1f}s")
        print(f"  Best parameters:")
        for key, value in study_gb.best_params.items():
            print(f"    {key}: {value}")
        
        # Train best model
        best_params_gb = study_gb.best_params.copy()
        best_params_gb.update({
            'random_state': 2025,
            'verbose': 0
        })
        
        model_gb = sklearn.ensemble.GradientBoostingClassifier(**best_params_gb)
        model_gb.fit(train_data_split, train_labels_split)
        
        val_pred_gb = model_gb.predict_proba(val_data)[:, 1]
        test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
        val_roc_gb = sklearn.metrics.roc_auc_score(val_labels, val_pred_gb)
        
        best_models['GradientBoosting'] = model_gb
        best_scores['GradientBoosting'] = val_roc_gb
        best_models['GradientBoosting_test'] = test_pred_gb
        
        print(f"  Final validation ROC AUC: {val_roc_gb:.6f}")
    else:
        print("  Optuna not available, using default Gradient Boosting parameters")
        model_gb = sklearn.ensemble.GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.02, max_depth=8,
            min_samples_split=5, subsample=0.85, max_features='sqrt',
            random_state=2025, verbose=0
        )
        model_gb.fit(train_data_split, train_labels_split)
        val_pred_gb = model_gb.predict_proba(val_data)[:, 1]
        test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
        val_roc_gb = sklearn.metrics.roc_auc_score(val_labels, val_pred_gb)
        best_models['GradientBoosting'] = model_gb
        best_scores['GradientBoosting'] = val_roc_gb
        best_models['GradientBoosting_test'] = test_pred_gb
        print(f"  Validation ROC AUC: {val_roc_gb:.6f}")
else:
    # Skip GB optimization - use default parameters
    print("\n" + "="*80)
    print("GRADIENT BOOSTING (Using Default Parameters)")
    print("="*80)
    print("  Skipping optimization - GB is sequential and too slow (20-40 min/trial)")
    print("  GB is also the weakest model (0.915 vs 0.918-0.919 for XGB/LGB)")
    print("  Using default parameters from feature_engineering_advanced.py")
    
    model_gb = sklearn.ensemble.GradientBoostingClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=8,
        min_samples_split=5, subsample=0.85, max_features='sqrt',
        random_state=2025, verbose=0
    )
    print("  Training with default parameters...")
    start_time = time.time()
    model_gb.fit(train_data_split, train_labels_split)
    elapsed = time.time() - start_time
    
    val_pred_gb = model_gb.predict_proba(val_data)[:, 1]
    test_pred_gb = model_gb.predict_proba(test_data)[:, 1]
    val_roc_gb = sklearn.metrics.roc_auc_score(val_labels, val_pred_gb)
    
    best_models['GradientBoosting'] = model_gb
    best_scores['GradientBoosting'] = val_roc_gb
    best_models['GradientBoosting_test'] = test_pred_gb
    
    print(f"  Validation ROC AUC: {val_roc_gb:.6f} | Time: {elapsed:.1f}s")
    print(f"  (Skipped optimization - saved ~10-15 hours)")

# ============================================================
# Ensemble Optimized Models
# ============================================================
print("\n" + "="*80)
print("ENSEMBLING OPTIMIZED MODELS")
print("="*80)

# Collect predictions
val_predictions = []
test_predictions = []
model_names = []

if 'XGBoost' in best_models:
    val_pred = best_models['XGBoost'].predict_proba(val_data)[:, 1]
    val_predictions.append(val_pred)
    test_predictions.append(best_models['XGBoost_test'])
    model_names.append("XGBoost")
    print(f"XGBoost: {best_scores['XGBoost']:.6f}")

if 'LightGBM' in best_models:
    val_pred = best_models['LightGBM'].predict_proba(val_data)[:, 1]
    val_predictions.append(val_pred)
    test_predictions.append(best_models['LightGBM_test'])
    model_names.append("LightGBM")
    print(f"LightGBM: {best_scores['LightGBM']:.6f}")

if 'GradientBoosting' in best_models:
    val_pred = best_models['GradientBoosting'].predict_proba(val_data)[:, 1]
    val_predictions.append(val_pred)
    test_predictions.append(best_models['GradientBoosting_test'])
    model_names.append("GradientBoosting")
    print(f"GradientBoosting: {best_scores['GradientBoosting']:.6f}")

val_predictions = np.array(val_predictions).T
test_predictions = np.array(test_predictions).T

# Simple average
val_ensemble_avg = val_predictions.mean(axis=1)
test_ensemble_avg = test_predictions.mean(axis=1)
roc_avg = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_avg)
print(f"\nSimple Average: {roc_avg:.6f}")

# Weighted average
weights = np.array([best_scores[name] for name in model_names])
weights = weights / weights.sum()
val_ensemble_weighted = np.average(val_predictions, axis=1, weights=weights)
test_ensemble_weighted = np.average(test_predictions, axis=1, weights=weights)
roc_weighted = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_weighted)
print(f"Weighted Average: {roc_weighted:.6f}")
print(f"  Weights: {dict(zip(model_names, weights))}")

# Optimized weights
try:
    from scipy.optimize import minimize
    
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
    
    # Select best method
    ensemble_methods = {
        'Simple Average': (roc_avg, test_ensemble_avg),
        'Weighted Average': (roc_weighted, test_ensemble_weighted),
        'Optimized Weights': (roc_opt, test_ensemble_opt)
    }
    
    best_method = max(ensemble_methods.items(), key=lambda x: x[1][0])
    best_name, (best_roc, best_test_pred) = best_method
    
except Exception as e:
    print(f"\nOptimization failed: {e}")
    if roc_weighted > roc_avg:
        best_name = 'Weighted Average'
        best_roc = roc_weighted
        best_test_pred = test_ensemble_weighted
    else:
        best_name = 'Simple Average'
        best_roc = roc_avg
        best_test_pred = test_ensemble_avg

print("\n" + "="*80)
print(f"BEST METHOD: {best_name}")
print(f"  Validation ROC AUC: {best_roc:.6f}")
print("="*80)

# Save submission
submission = pd.DataFrame({'id': ids, 'loan_paid_back': best_test_pred})
submission.to_csv(SUBMISSION_FILENAME, index=False)
print(f"\n✓ Generated submission file: {SUBMISSION_FILENAME}")
print(f"  Expected improvement: +0.0005 to +0.002 over 0.92100")
print(f"  Target: 0.9215 - 0.923")


