# Baseline Model 2: Logistic Regression (sklearn)
# Introduction to Artificial Intelligence
# Loan Default Prediction Competition
# By Team
# Copyright 2025, Texas Tech University - Costa Rica

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import torch
import time 

# Load and prepare the data
training_db = pd.read_csv("Datasets/train.csv", header=0)
test_db = pd.read_csv("Datasets/test.csv", header=0)


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
train_stds = train_data_split.std().replace(0, 1)  # Replace zero std with 1 to avoid division by zero
train_data_split = (train_data_split - train_means) / train_stds
val_data = (val_data - train_means) / train_stds
test_data = (test_data - train_means) / train_stds

# Select columns of interest (all columns)
cols = train_data_split.columns

# Start timer for training + prediction
start_time = time.time()

# Create and train a Logistic Regression model
model = sklearn.linear_model.LogisticRegression(
    max_iter=1000,
    random_state=2025,
    n_jobs=-1
)

model.fit(train_data_split[cols], train_labels_split)

# End timer and report elapsed time for fit + predict steps
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f} seconds")

# Evaluate on validation set
val_pred_proba = model.predict_proba(val_data[cols])[:, 1]
val_roc_auc = sklearn.metrics.roc_auc_score(val_labels, val_pred_proba)

# Evaluate on training set
train_pred_proba = model.predict_proba(train_data_split[cols])[:, 1]
train_roc_auc = sklearn.metrics.roc_auc_score(train_labels_split, train_pred_proba)

print(f"\nTrain ROC AUC: {train_roc_auc:.6f}")
print(f"Val ROC AUC: {val_roc_auc:.6f}")

# Get predictions on test set
test_pred_proba = model.predict_proba(test_data[cols])[:, 1]

print(f"\nModel produced {len(test_pred_proba)} predictions. Sum of probabilities: {test_pred_proba.sum():.2f}")

# Build submission DataFrame using the original ids column
submission = pd.DataFrame({'id': ids, 'loan_paid_back': test_pred_proba})
submission.to_csv('my_submission.csv', index=False)
print(f"Wrote submission 'my_submission.csv' with {len(submission)} rows")
