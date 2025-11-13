# Introduction to Machine Learning
# Credit Default Dataset
# Logistic regression solved through gradient descent 
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import torch

# ============================================================
# Load and prepare data
# ============================================================

df = pd.read_csv("train.csvs", header=0)
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)
labels = df["loan_paid_back"]
df = df.drop(columns="loan_paid_back")
train_data, test_data, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(df, labels,
            test_size=0.2, shuffle=True, random_state=2025)

# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data  = (test_data  - train_means) / train_stds

# Get some lengths
ncoeffs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants
# ============================================================
n_iterations = 10000
learning_rate = 0.1
eval_step = 100

# Print the configuration
print(f"Num iterations: {n_iterations}  Learning rate: {learning_rate}")

# ============================================================
# Convert data to PyTorch tensors
# ============================================================
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(test_data.values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32)

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
