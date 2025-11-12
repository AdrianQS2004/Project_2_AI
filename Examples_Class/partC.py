# Introduction to AI Homework 4
# Part C
# By Keisy Nunez and Adrian Quiros

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics

# Load and prepare the data
df = pd.read_csv("credit_delinquency_v2.csv", header=0)
df = pd.get_dummies(df, prefix_sep="_", drop_first=True, dtype=int)
labels = df["Delinquent"]
df = df.drop(columns="Delinquent")
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

# Create and train a new logistic regression classifier
model = sklearn.linear_model.LogisticRegression(\
        solver='newton-cg', tol=1e-6)

# Train it with the training data and labels
model.fit(train_data[cols], train_labels)

# Print some results
print("Iterations used: ", model.n_iter_)
print("Intercept: ", model.intercept_)
print("Coeffs: \n", model.coef_)

# Map coefficients to column names (sorted by abs coefficient value)
# This is used por Part D
print("\nTop Ten Input Columns (sorted by coefficient absolute value):")
coef_dict = dict(zip(cols, model.coef_[0]))
for feature, coef in sorted(coef_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"{feature}: {coef:.6f}")

# Get the prediction probabilities
pred_proba = model.predict_proba(test_data[cols])[:,1]

# Compute a precision & recall graph
precisions, recalls, thresholds = \
    sklearn.metrics.precision_recall_curve(test_labels, pred_proba)
plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
plt.legend(loc="center right")
plt.xlabel("Threshold")
plt.axis([0.0, 1.0, 0.0, 1.0])
# plt.show()

# Plot a ROC curve (Receiver Operating Characteristic)
fpr, tpr, _ = sklearn.metrics.roc_curve(test_labels, pred_proba)
plt.plot(fpr,tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
# plt.show()

# Compute the area under the ROC curve (ROC AUC)
auc_score = sklearn.metrics.roc_auc_score(test_labels, pred_proba)
# print("Test AUC score: {:.4f}".format(auc_score))

# Compute ROC AUC against training data
pred_proba_training = model.predict_proba(train_data[cols])[:,1]

auc_score_training = sklearn.metrics.roc_auc_score(\
    train_labels, pred_proba_training)
# print("Train AUC score: {:.4f}".format(auc_score_training))

