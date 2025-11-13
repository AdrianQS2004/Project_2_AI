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
training_db = pd.read_csv("train.csv")
test_db = pd.read_csv("test.csv")
labels = training_db["loan_paid_back"]
ids = test_db['id']

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()
test_labels = test_db.copy()

# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std()
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
n_iterations = 10000
learning_rate = 0.1
eval_step = 100

# ============================================================
# Training loop
# ============================================================

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
        print(f"Iteration {iteration:4d}: Loss = {cost.item():.6f}")


# Write the submission file

# After training, compute final probabilities on the test set and write submission
with torch.no_grad():
    final_test_logits = X_test @ W + B
    final_test_proba = torch.sigmoid(final_test_logits).cpu().numpy().reshape(-1)

# Threshold to binary predictions (0/1). Use 0.5 by default.
preds = (final_test_proba >= 0.5).astype(int)

# Build submission DataFrame using the original ids column
submission = pd.DataFrame({'id': ids, 'loan_paid_back': preds})
submission.to_csv('my_submission.csv', index=False)
print(f"Wrote submission 'my_submission.csv' with {len(submission)} rows")
