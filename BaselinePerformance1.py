# Introduction to Artificial Intelligence
# Vehicle Price dataset
# Linear Regression in PyTorch
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import sklearn.model_selection
import torch

def collapse_small_categories(df, col, min_count=10, other_label="others"):
    counts = df[col].value_counts()
    rare = counts[counts < min_count].index
    df[col] = df[col].where(~df[col].isin(rare), other_label)
    return df

# Load and prepare the data
df = pd.read_csv("vehicles_clean2.csv", header=0)
df = collapse_small_categories(df, "manufacturer", min_count=100)
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)
labels = df["price"]
df = df.drop(columns="price")
train_data, test_data, train_labels, test_labels = \
            sklearn.model_selection.train_test_split(df, labels,
            test_size=0.2, shuffle=True, random_state=2025)

# Standardize scale for all columns
train_means = train_data.mean()
train_stds = train_data.std()
train_data = (train_data - train_means) / train_stds
test_data  = (test_data  - train_means) / train_stds

# Insert a column of ones to serve as x0
train_data["ones"] = 1
test_data["ones"] = 1

# PyTorch constants

# Input vectors (convert to float32 tensors)
X = torch.tensor(train_data.values, dtype=torch.float32)
Y = torch.tensor(train_labels.values.reshape(-1,1), dtype=torch.float32)

# Solve the Normal equations: W = (X' * X)^-1 * X' * Y
XT = X.T
W = torch.inverse(XT @ X) @ XT @ Y

# Print the coefficients
print("W=")
print(W)

# Compute predictions from test data
X_test = torch.tensor(test_data.values, dtype=torch.float32)
Y_test = torch.tensor(test_labels.values.reshape(-1,1), dtype=torch.float32)

Y_pred = X_test @ W

# Print the first 10 results
for idx in range(10):
    print("Predicted: {:8.0f}  Correct: {:8d}".format(
        Y_pred[idx][0].item(),
        int(test_labels.values[idx])
    ))

# Compute RMSE
mse = torch.mean((Y_test - Y_pred) ** 2)
rmse = torch.sqrt(mse)

print("RMSE:", rmse.item())