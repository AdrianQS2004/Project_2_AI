# Advanced Feature Engineering - Create More Interaction Features
# Building on successful Advanced Ensemble (0.92088)

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.ensemble
import sklearn.metrics
import time
import torch
import torch.nn as nn

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

# Check for CUDA availability
CUDA_AVAILABLE = False
XGBOOST_GPU_AVAILABLE = False
LIGHTGBM_GPU_AVAILABLE = False

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    if CUDA_AVAILABLE:
        print(f"✓ CUDA detected: {torch.cuda.get_device_name(0)}")
except ImportError:
    pass

# Check if XGBoost actually supports GPU (not just if CUDA exists)
if CUDA_AVAILABLE and XGBOOST_AVAILABLE:
    try:
        import xgboost as xgb
        # Try to check if XGBoost has GPU support by checking available tree methods
        # We'll test this by trying to create a simple config
        test_params = {'tree_method': 'gpu_hist'}
        # Check if gpu_hist is in valid tree methods by trying to get valid options
        # Actually, we'll just try to use it and catch the error, or check XGBoost version
        # For now, we'll be conservative and only use GPU if we can verify it works
        # The safest approach is to try CPU first and let user configure GPU if needed
        XGBOOST_GPU_AVAILABLE = False  # Default to False, will be set if verified
    except:
        XGBOOST_GPU_AVAILABLE = False

# Check if LightGBM supports GPU
if CUDA_AVAILABLE and LIGHTGBM_AVAILABLE:
    try:
        import lightgbm as lgb
        # LightGBM GPU support check - we'll be conservative
        LIGHTGBM_GPU_AVAILABLE = False  # Default to False
    except:
        LIGHTGBM_GPU_AVAILABLE = False

print("="*80)
print("ADVANCED FEATURE ENGINEERING ENSEMBLE")
if CUDA_AVAILABLE:
    print("CUDA: Available")
    if XGBOOST_GPU_AVAILABLE:
        print("XGBoost GPU: Available")
    else:
        print("XGBoost GPU: Not available (will use CPU)")
    if LIGHTGBM_GPU_AVAILABLE:
        print("LightGBM GPU: Available")
    else:
        print("LightGBM GPU: Not available (will use CPU)")
else:
    print("GPU Support: CPU only")
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
print("\n[1/7] Training Random Forest...")
start_time = time.time()
model_rf = sklearn.ensemble.RandomForestClassifier(
    n_estimators=500, max_depth=30, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=2025, n_jobs=12, verbose=0
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
print("\n[2/7] Training Gradient Boosting...")
print(f"  Training on {len(train_data_split)} samples with {len(train_data_split.columns)} features...")
print("  Note: Gradient Boosting is sequential (each tree depends on previous), so lower CPU usage is normal.")
print("  However, it should still use 1 core actively. If CPU is very low, training may be stuck.")
start_time = time.time()
model_gb = sklearn.ensemble.GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.02, max_depth=8,
    min_samples_split=5, subsample=0.85, max_features='sqrt', random_state=2025, verbose=1
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
    print("\n[3/7] Training XGBoost...")
    start_time = time.time()
    xgb_params = {
        'n_estimators': 700, 'learning_rate': 0.02, 'max_depth': 9,
        'min_child_weight': 2, 'subsample': 0.85, 'colsample_bytree': 0.85,
        'gamma': 0.3, 'reg_alpha': 0.2, 'reg_lambda': 2.0, 'random_state': 2025,
        'eval_metric': 'auc', 'use_label_encoder': False, 'verbosity': 0
    }
    # Only use GPU if XGBoost actually supports it
    # For now, use CPU with multiple threads (safer and works with standard XGBoost)
    xgb_params['tree_method'] = 'hist'  # Use hist for better performance than exact
    xgb_params['n_jobs'] = 12  # Use multiple CPU cores
    print("  Using CPU with 12 threads")
    model_xgb = xgb.XGBClassifier(**xgb_params)
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
    print("\n[4/7] Training LightGBM...")
    start_time = time.time()
    lgb_params = {
        'n_estimators': 700, 'learning_rate': 0.02, 'max_depth': 9,
        'num_leaves': 50, 'min_child_samples': 15, 'subsample': 0.85,
        'colsample_bytree': 0.85, 'reg_alpha': 0.2, 'reg_lambda': 2.0,
        'random_state': 2025, 'verbosity': -1
    }
    # Use CPU with multiple threads (safer - GPU requires special LightGBM build)
        lgb_params['n_jobs'] = 12
    print("  Using CPU with 12 threads")
    model_lgb = lgb.LGBMClassifier(**lgb_params)
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
print("\n[5/7] Training Extra Trees...")
start_time = time.time()
model_et = sklearn.ensemble.ExtraTreesClassifier(
    n_estimators=500, max_depth=30, min_samples_split=2,
    min_samples_leaf=1, max_features='sqrt', random_state=2025, n_jobs=12, verbose=0
)
model_et.fit(train_data_split, train_labels_split)
val_pred_et = model_et.predict_proba(val_data)[:, 1]
test_pred_et = model_et.predict_proba(test_data)[:, 1]
val_roc_et = sklearn.metrics.roc_auc_score(val_labels, val_pred_et)
val_predictions.append(val_pred_et)
test_predictions.append(test_pred_et)
model_names.append("ExtraTrees")
print(f"  Val ROC AUC: {val_roc_et:.6f} | Time: {time.time() - start_time:.1f}s")

# Model 6: Shallow Neural Network
print("\n[6/7] Training Shallow Neural Network...")
start_time = time.time()
device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
if device.type == "cuda":
    print(f"  Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("  Using CPU (CUDA not available)")

# Convert data to tensors
X_train_nn = torch.tensor(train_data_split.values, dtype=torch.float32, device=device)
Y_train_nn = torch.tensor(train_labels_split.values.reshape(-1, 1), dtype=torch.float32, device=device)
X_val_nn = torch.tensor(val_data.values, dtype=torch.float32, device=device)
X_test_nn = torch.tensor(test_data.values, dtype=torch.float32, device=device)

# Create shallow NN: input -> 100 -> 1
n_inputs_nn = train_data_split.shape[1]
shallow_nn = nn.Sequential(
    nn.Linear(n_inputs_nn, 100),
    nn.PReLU(),  # Parametric ReLU - learns negative slope, often better for tabular data
    nn.Linear(100, 1)
).to(device)

# Initialize weights (PReLU works well with He initialization)
nn.init.kaiming_normal_(shallow_nn[0].weight, nonlinearity="relu")
nn.init.xavier_normal_(shallow_nn[2].weight)

# Training setup
optimizer_shallow = torch.optim.Adam(shallow_nn.parameters(), lr=1e-4)
loss_fn = nn.BCEWithLogitsLoss()
batch_size_nn = 2048
n_epochs_shallow = 200
early_stop_patience = 10

best_val_roc_shallow = 0
best_state_shallow = None
epochs_no_improve = 0

num_batches = int(np.ceil(len(train_data_split) / batch_size_nn))

for epoch in range(n_epochs_shallow):
    shallow_nn.train()
    epoch_loss = 0.0
    
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size_nn
        end_i = min(start_i + batch_size_nn, len(train_data_split))
        X_batch = X_train_nn[start_i:end_i]
        Y_batch = Y_train_nn[start_i:end_i]
        
        optimizer_shallow.zero_grad()
        logits = shallow_nn(X_batch)
        loss = loss_fn(logits, Y_batch)
        loss.backward()
        optimizer_shallow.step()
        epoch_loss += loss.item()
    
    # Validation
    if (epoch + 1) % 10 == 0 or epoch == 0:
        shallow_nn.eval()
        with torch.no_grad():
            val_logits = shallow_nn(X_val_nn)
            val_pred_proba = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_roc_shallow = sklearn.metrics.roc_auc_score(val_labels.values, val_pred_proba)
        
        if val_roc_shallow > best_val_roc_shallow:
            best_val_roc_shallow = val_roc_shallow
            best_state_shallow = {k: v.clone() for k, v in shallow_nn.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stop_patience:
            break

# Restore best model
if best_state_shallow:
    shallow_nn.load_state_dict(best_state_shallow)

# Get predictions
shallow_nn.eval()
with torch.no_grad():
    val_logits = shallow_nn(X_val_nn)
    val_pred_shallow = torch.sigmoid(val_logits).cpu().numpy().flatten()
    test_logits = shallow_nn(X_test_nn)
    test_pred_shallow = torch.sigmoid(test_logits).cpu().numpy().flatten()

val_roc_shallow = sklearn.metrics.roc_auc_score(val_labels, val_pred_shallow)
val_predictions.append(val_pred_shallow)
test_predictions.append(test_pred_shallow)
model_names.append("ShallowNN")
print(f"  Val ROC AUC: {val_roc_shallow:.6f} | Time: {time.time() - start_time:.1f}s")

# Model 7: Deep Neural Network (Improved NN)
print("\n[7/7] Training Deep Neural Network...")
start_time = time.time()

# Create deep NN: input -> 128 -> 64 -> 32 -> 1
def create_deep_nn(n_inputs, hidden_layers=[128, 64, 32], dropout_rate=0.3):
    layers = []
    input_size = n_inputs
    for hidden_size in hidden_layers:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.PReLU())  # Parametric ReLU - learns negative slope, often better for tabular data
        layers.append(nn.Dropout(dropout_rate))
        input_size = hidden_size
    layers.append(nn.Linear(input_size, 1))
    return nn.Sequential(*layers)

deep_nn = create_deep_nn(n_inputs_nn).to(device)

# Initialize weights (PReLU works well with He initialization)
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

deep_nn.apply(init_weights)

# Training setup
optimizer_deep = torch.optim.Adam(deep_nn.parameters(), lr=1e-4, weight_decay=1e-5)
n_epochs_deep = 300
early_stop_patience_deep = 20

best_val_roc_deep = 0
best_state_deep = None
epochs_no_improve_deep = 0

for epoch in range(n_epochs_deep):
    deep_nn.train()
    epoch_loss = 0.0
    
    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size_nn
        end_i = min(start_i + batch_size_nn, len(train_data_split))
        X_batch = X_train_nn[start_i:end_i]
        Y_batch = Y_train_nn[start_i:end_i]
        
        optimizer_deep.zero_grad()
        logits = deep_nn(X_batch)
        loss = loss_fn(logits, Y_batch)
        loss.backward()
        optimizer_deep.step()
        epoch_loss += loss.item()
    
    # Validation
    if (epoch + 1) % 10 == 0 or epoch == 0:
        deep_nn.eval()
        with torch.no_grad():
            val_logits = deep_nn(X_val_nn)
            val_pred_proba = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_roc_deep = sklearn.metrics.roc_auc_score(val_labels.values, val_pred_proba)
        
        if val_roc_deep > best_val_roc_deep:
            best_val_roc_deep = val_roc_deep
            best_state_deep = {k: v.clone() for k, v in deep_nn.state_dict().items()}
            epochs_no_improve_deep = 0
        else:
            epochs_no_improve_deep += 1
        
        if epochs_no_improve_deep >= early_stop_patience_deep:
            break

# Restore best model
if best_state_deep:
    deep_nn.load_state_dict(best_state_deep)

# Get predictions
deep_nn.eval()
with torch.no_grad():
    val_logits = deep_nn(X_val_nn)
    val_pred_deep = torch.sigmoid(val_logits).cpu().numpy().flatten()
    test_logits = deep_nn(X_test_nn)
    test_pred_deep = torch.sigmoid(test_logits).cpu().numpy().flatten()

val_roc_deep = sklearn.metrics.roc_auc_score(val_labels, val_pred_deep)
val_predictions.append(val_pred_deep)
test_predictions.append(test_pred_deep)
model_names.append("DeepNN")
print(f"  Val ROC AUC: {val_roc_deep:.6f} | Time: {time.time() - start_time:.1f}s")

# Collect model scores in the same order as model_names / predictions
model_scores = [
    val_roc_rf,          # RandomForest
    val_roc_gb           # GradientBoosting
]
if XGBOOST_AVAILABLE:
    model_scores.append(val_roc_xgb)
if LIGHTGBM_AVAILABLE:
    model_scores.append(val_roc_lgb)
model_scores.extend([
    val_roc_et,          # ExtraTrees
    val_roc_shallow,     # ShallowNN
    val_roc_deep         # DeepNN
])

model_scores = np.array(model_scores)

print("\nIndividual model ROC AUC scores:")
for name, score in zip(model_names, model_scores):
    print(f"  {name:16s}: {score:.6f}")

# ============================================================
# Manually select a strong subset of models to avoid overfitting
# ============================================================
# Based on experiments and prior project results, tree boosting models
# tend to be the strongest and most stable:
#   - GradientBoosting
#   - XGBoost
#   - LightGBM
#
# We will ensemble ONLY these models and drop:
#   - ExtraTrees (weaker here)
#   - RandomForest (weaker here)
#   - ShallowNN / DeepNN (currently lower ROC, risk of overfitting)
keep_model_names = {"GradientBoosting", "XGBoost", "LightGBM"}

selected_idx = [i for i, name in enumerate(model_names) if name in keep_model_names]

# Fallback: if for some reason none are available (e.g. missing XGBoost/LightGBM),
# keep all models.
if not selected_idx:
    selected_idx = list(range(len(model_names)))

selected_idx = np.array(sorted(selected_idx))

print("\nUsing the following models in the ensemble:")
for i in selected_idx:
    print(f"  - {model_names[i]:16s}: {model_scores[i]:.6f}")

# Filter predictions and names down to selected models
val_predictions = np.array(val_predictions).T[:, selected_idx]
test_predictions = np.array(test_predictions).T[:, selected_idx]
model_names = [model_names[i] for i in selected_idx]
model_scores = model_scores[selected_idx]

# ============================================================
# Ensemble
# ============================================================
print("\n" + "="*80)
print("ENSEMBLING PREDICTIONS (Best models only)")
print("="*80)

# Simple average
val_ensemble_avg = val_predictions.mean(axis=1)
test_ensemble_avg = test_predictions.mean(axis=1)
roc_avg = sklearn.metrics.roc_auc_score(val_labels, val_ensemble_avg)
print(f"\nSimple Average: {roc_avg:.6f}")

# Weighted average (weights proportional to validation ROC AUC)
weights = model_scores / model_scores.sum()

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

# Stacking with multiple meta-learners
print("\n" + "-"*80)
print("STACKING (Meta-Learning)")
print("-"*80)

# Try different meta-learners - sometimes one works better than others
stacking_methods = {}

# 1. Logistic Regression (simple, often works well)
try:
    meta_lr = sklearn.linear_model.LogisticRegression(max_iter=2000, random_state=2025, C=1.0, solver='lbfgs')
    meta_lr.fit(val_predictions, val_labels)
    val_stack_lr = meta_lr.predict_proba(val_predictions)[:, 1]
    test_stack_lr = meta_lr.predict_proba(test_predictions)[:, 1]
    roc_stack_lr = sklearn.metrics.roc_auc_score(val_labels, val_stack_lr)
    stacking_methods['Stacking (Logistic)'] = (roc_stack_lr, test_stack_lr)
    print(f"  Logistic Regression: {roc_stack_lr:.6f}")
except Exception as e:
    print(f"  Logistic Regression failed: {e}")

# 2. Ridge Regression (more regularized, can help prevent overfitting)
try:
    from sklearn.linear_model import Ridge
    meta_ridge = Ridge(alpha=1.0, random_state=2025)
    meta_ridge.fit(val_predictions, val_labels)
    val_stack_ridge = np.clip(meta_ridge.predict(val_predictions), 0, 1)
    test_stack_ridge = np.clip(meta_ridge.predict(test_predictions), 0, 1)
    roc_stack_ridge = sklearn.metrics.roc_auc_score(val_labels, val_stack_ridge)
    stacking_methods['Stacking (Ridge)'] = (roc_stack_ridge, test_stack_ridge)
    print(f"  Ridge Regression: {roc_stack_ridge:.6f}")
except Exception as e:
    print(f"  Ridge Regression failed: {e}")

# 3. Random Forest meta-learner (can capture non-linear combinations)
try:
    meta_rf = sklearn.ensemble.RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=2025, n_jobs=4, verbose=0
    )
    meta_rf.fit(val_predictions, val_labels)
    val_stack_rf = meta_rf.predict_proba(val_predictions)[:, 1]
    test_stack_rf = meta_rf.predict_proba(test_predictions)[:, 1]
    roc_stack_rf = sklearn.metrics.roc_auc_score(val_labels, val_stack_rf)
    stacking_methods['Stacking (RF)'] = (roc_stack_rf, test_stack_rf)
    print(f"  Random Forest: {roc_stack_rf:.6f}")
except Exception as e:
    print(f"  Random Forest failed: {e}")

# Add best stacking method to ensemble_methods
if stacking_methods:
    best_stacking = max(stacking_methods.items(), key=lambda x: x[1][0])
    best_stacking_name, (best_stacking_roc, best_stacking_pred) = best_stacking
    ensemble_methods[best_stacking_name] = (best_stacking_roc, best_stacking_pred)
    print(f"\n  Best Stacking: {best_stacking_name} ({best_stacking_roc:.6f})")
else:
    print("  All stacking methods failed")

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

