# Classic Model 4: LightGBM Classifier
# Introduction to Artificial Intelligence
# Loan Default Prediction Competition
# By Team
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.metrics
import time

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    print("LightGBM not available. Install with: pip install lightgbm")
    LIGHTGBM_AVAILABLE = False

if LIGHTGBM_AVAILABLE:
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

    # Align test columns to match training columns (fill missing with 0, drop extra)
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

    # Standardize scale for all columns
    train_means = train_data_split.mean()
    train_stds = train_data_split.std().replace(0, 1)  # Replace zero std with 1
    train_data_split = (train_data_split - train_means) / train_stds
    val_data = (val_data - train_means) / train_stds
    test_data = (test_data - train_means) / train_stds

    # Hyperparameters for LightGBM
    n_estimators = 200
    learning_rate = 0.1
    max_depth = 6
    num_leaves = 31
    min_child_samples = 20
    subsample = 0.8
    colsample_bytree = 0.8
    random_state = 2025

    print(f"LightGBM Classifier")
    print(f"Hyperparameters:")
    print(f"  n_estimators: {n_estimators}")
    print(f"  learning_rate: {learning_rate}")
    print(f"  max_depth: {max_depth}")
    print(f"  num_leaves: {num_leaves}")
    print(f"  min_child_samples: {min_child_samples}")
    print(f"  subsample: {subsample}")
    print(f"  colsample_bytree: {colsample_bytree}")

    # Create and train LightGBM model
    start_time = time.time()

    model = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        min_child_samples=min_child_samples,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1
    )

    model.fit(
        train_data_split, train_labels_split,
        eval_set=[(val_data, val_labels)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
    )

    elapsed = time.time() - start_time
    print(f"Training completed in {elapsed:.2f} seconds")

    # Evaluate on validation set
    val_pred_proba = model.predict_proba(val_data)[:, 1]
    val_roc_auc = sklearn.metrics.roc_auc_score(val_labels, val_pred_proba)

    # Evaluate on full training set
    train_pred_proba = model.predict_proba(train_data_split)[:, 1]
    train_roc_auc = sklearn.metrics.roc_auc_score(train_labels_split, train_pred_proba)

    print(f"\nTrain ROC AUC: {train_roc_auc:.6f}")
    print(f"Val ROC AUC: {val_roc_auc:.6f}")

    # Predict on test set
    test_pred_proba = model.predict_proba(test_data)[:, 1]

    # Build submission DataFrame
    submission = pd.DataFrame({'id': ids, 'loan_paid_back': test_pred_proba})
    submission.to_csv('my_submission.csv', index=False)
    print(f"\nWrote submission 'my_submission.csv' with {len(submission)} rows")

    # Save hyperparameters for tracking
    hyperparameters = {
        'n_estimators': n_estimators,
        'learning_rate': learning_rate,
        'max_depth': max_depth,
        'num_leaves': num_leaves,
        'min_child_samples': min_child_samples,
        'subsample': subsample,
        'colsample_bytree': colsample_bytree,
        'random_state': random_state
    }

    print(f"\nHyperparameters used: {hyperparameters}")
