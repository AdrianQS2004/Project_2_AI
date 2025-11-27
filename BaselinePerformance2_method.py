# Introduction to Artificial Intelligence
# Vehicle Price dataset
# Linear Regression in PyTorch
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sklearn.model_selection
import torch
import time 

# Load and prepare the data
training_db = pd.read_csv("train.csv", header=0)
test_db = pd.read_csv("test.csv", header=0)


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


# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std().replace(0, 1)  # Replace zero std with 1 to avoid division by zero
train_data = (train_data - train_means) / train_stds
test_data  = (test_data  - train_means) / train_stds


# Select columns of interest (all columns)
cols = train_data.columns

# Start timer for training + prediction
start_time = time.time()

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()

model.fit(train_data[cols], train_labels)


# Get model outputs on the test set and convert to binary predictions.
# For LinearRegression this will produce continuous scores; threshold at 0.8
# (change threshold as needed). We also clip scores to [0,1] for safety.
preds_raw = model.predict(test_data[cols])
preds = np.clip(preds_raw, 0.0, 1.0)
# Get the prediction probabilities


print(f"Model produced {len(preds)} predictions. Sum of positives: {preds.sum()}")

# End timer and report elapsed time for fit + predict steps
elapsed = time.time() - start_time
print(f"Elapsed time (train + predict): {elapsed:.2f} seconds")

# Build submission DataFrame using the original ids column
submission = pd.DataFrame({'id': ids, 'loan_paid_back': preds})
submission.to_csv('my_submission.csv', index=False)
print(f"Wrote submission 'my_submission.csv' with {len(submission)} rows")
