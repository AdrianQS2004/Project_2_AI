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

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# PyTorch constants

# Input vectors (convert to float32 tensors)
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1,1), dtype=torch.float32)

# Compute predictions from test data
X_test = torch.tensor(test_data.values, dtype=torch.float32)

# ============================================================
# Create and initialize weights and bias
# ============================================================

# Create a vector of coefficients with random values between -1 and 1
W = torch.rand((ncoeffs, 1)) * 2 - 1

# Create a bias variable initialized to zero
B = torch.zeros(1, dtype=torch.float32)

# Start tracking gradients on W & B
W.requires_grad_(True)
B.requires_grad_(True)

# ============================================================
# Training constants
# ============================================================
n_iterations = 5000
learning_rate = 0.1
eval_step = 100

# ============================================================
# Training loop
# ============================================================

# Record start time for the training loop
start_time = time.time()

for iteration in range(n_iterations):

    # Forward pass: compute logits (raw predictions)
    logits = X @ W + B
    
    # Compute loss (binary cross-entropy)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)

    # Backward pass: compute gradients
    cost.backward()

    # Update weights and bias (gradient descent)
    with torch.no_grad():
        W -= learning_rate * W.grad
        B -= learning_rate * B.grad
        W.grad.zero_()
        B.grad.zero_()

    # Print progress every eval_step iterations
    if iteration % eval_step == 0:
        print(f"Iteration {iteration:4d}: Train Cost = {cost.item():.6f}")

# Training finished — compute elapsed time
elapsed = time.time() - start_time
print(f"Training completed in {elapsed:.2f} seconds")


# Write the submission file

# After training, compute final probabilities on the test set and write submission
with torch.no_grad():
    final_test_logits = X_test @ W + B
    final_test_proba = torch.sigmoid(final_test_logits).cpu().numpy().reshape(-1)

# Threshold to binary predictions (0/1). Use 0.8 by default.
preds = (final_test_proba >= 0.8).astype(int)

# Build submission DataFrame using the original ids column
submission = pd.DataFrame({'id': ids, 'loan_paid_back': preds})
submission.to_csv('my_submission.csv', index=False)
print(f"Wrote submission 'my_submission.csv' with {len(submission)} rows")
