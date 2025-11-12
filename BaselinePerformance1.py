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
submission_db = pd.read_csv("sample_submission.csv")
# Doubt about this label
labels = submission_db["loan_paid_back"]
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

#Is this needed?
# Insert a column of ones to serve as x0
#train_data["ones"] = 1
#test_data["ones"] = 1

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# PyTorch constants

# Input vectors (convert to float32 tensors)
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1,1), dtype=torch.float32)

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
#XT = X.T
#W = torch.inverse(XT @ X) @ XT @ Y

# Print the coefficients
#print("W=")
#print(W)

# Compute predictions from test data
X_test = torch.tensor(test_data.values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1,1), dtype=torch.float32)

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

# Training loop
train_cost_hist = []
test_metric_hist = []

for iteration in range(n_iterations):

    # Forward pass: predictions
    logits = X @ W + B
    #Y_pred = torch.sigmoid(logits)

    # Loss computation (logistic_loss)

    # Option 1: Explicit binary cross-entropy formula
    #  The 1e-7 is to avoid log(0) in case of exact 0 or 1 predictions
    #cost = -torch.mean(Y * torch.log(Y_pred + 1e-7) + (1 - Y) * torch.log(1 - Y_pred + 1e-7))
    
    # Option 2: Built-in function (more efficient)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(logits, Y)

    # Compute gradients of MSE with respect to W & B
    # Will be stored in W.grad & B.grad
    cost.backward()

    # Parameter update (gradient descent)
    with torch.no_grad():
        W -= learning_rate * W.grad
        B -= learning_rate * B.grad

    # Zero gradients for next iteration
    W.grad.zero_()
    B.grad.zero_()

    # Evaluate and record cost every eval_step iterations
    if iteration % eval_step == 0:

        with torch.no_grad():
            train_cost = cost.item()

            # Predictions on test data
            test_logits = X_test @ W + B
            test_pred_proba = torch.sigmoid(test_logits).numpy()
        
            # Compute test ROC AUC
            test_roc_auc = sklearn.metrics.roc_auc_score(test_labels, test_pred_proba)

            train_cost_hist.append(train_cost)
            test_metric_hist.append(test_roc_auc)

            print(f"Iteration {iteration:4d}: Train cost: {train_cost:.4f}  Test ROC AUC: {test_roc_auc:.4f}")




# Plot results
iterations_hist = [i for i in range(0, n_iterations, eval_step)]
plt.plot(iterations_hist, train_cost_hist, "b")
plt.xlabel("Iteration")
plt.ylabel("Train Cost")
plt.title("Train Cost Evolution")
plt.figure()
plt.plot(iterations_hist, test_metric_hist, "r")
plt.xlabel("Iteration")
plt.ylabel("Test ROC AUC")
plt.title("Test ROC AUC Evolution")
plt.show()
