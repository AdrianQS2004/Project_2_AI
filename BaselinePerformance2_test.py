# Introduction to Artificial Intelligence
# Credit Delinquency Dataset
# Linear regression, precision-recall graph
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import sklearn.model_selection
import sklearn.linear_model
import matplotlib.pyplot as plt

# Load and prepare the data
# Load the dataset
df = pd.read_csv("Datasets/train.csv", header=0)
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

# Select columns of interest (all columns)
cols = train_data.columns

# Create and train a new LinearRegression model
model = sklearn.linear_model.LinearRegression()
model.fit(train_data[cols], train_labels)

# Predict new labels for test data
pred_proba = model.predict(test_data[cols])

# Draw a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center right")
plt.xlabel("Threshold")
plt.axis([0.0, 1.0, 0.0, 1.0])
plt.show()