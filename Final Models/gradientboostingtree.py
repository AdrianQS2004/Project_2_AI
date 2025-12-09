# Intro to AI: Project 2
# Luis Baeza, Adrian Quiros, Adrian de Souza
# Classic model: Gradient boosting tree

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt

# config
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

TARGET_COL = "loan_paid_back"   
ID_COL     = "id"               

TEST_SIZE    = 0.8
RANDOM_STATE = 42

N_ESTIMATORS   = 200        
MAX_DEPTH      = 7          
LEARNING_RATE  = 0.03 
SUBSAMPLE      = 0.7           


def load_and_prepare_data():
    """
    Load train/test CSVs, drop ID from features,
    one-hot encode categoricals, and return encoded train/test.
    """
    train = pd.read_csv(TRAIN_PATH)
    test  = pd.read_csv(TEST_PATH)

    y = train[TARGET_COL].astype(int)

    X = train.drop(columns=[TARGET_COL])

    X_features    = X.drop(columns=[ID_COL])
    test_features = test.drop(columns=[ID_COL])
    test_ids      = test[ID_COL].copy()

    combined = pd.concat([X_features, test_features], axis=0, ignore_index=True)

    combined_encoded = pd.get_dummies(combined, drop_first=True)

    X_encoded  = combined_encoded.iloc[:len(train), :].copy()
    X_test_enc = combined_encoded.iloc[len(train):, :].copy()

    return X_encoded, y, X_test_enc, test_ids


def train_valid_split(X, y):
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )


def build_model():
    model = GradientBoostingClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        random_state=RANDOM_STATE
    )
    return model


def evaluate_model(model, X_train, X_valid, y_train, y_valid,
                   model_name="Gradient Boosting Trees"):

    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_valid_proba = model.predict_proba(X_valid)[:, 1]

    train_auc = roc_auc_score(y_train, y_train_proba)
    valid_auc = roc_auc_score(y_valid, y_valid_proba)

    print(f"{model_name}")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Valid AUC: {valid_auc:.4f}")
    print(f"  Training time: {train_time:.2f} seconds")

    # Precision–Recall vs Threshold 
    precisions, recalls, thresholds = precision_recall_curve(y_valid, y_valid_proba)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title(f"{model_name} – Precision & Recall vs Threshold")
    plt.legend()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_valid, y_valid_proba)
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} – ROC Curve (AUC = {valid_auc:.4f})")

    plt.tight_layout()
    plt.show()

    return model, valid_auc


def train_full_and_save_submission(model, X_full, y_full, X_test, test_ids):
    """
    Retrain GB model on ALL training data and create Kaggle submission CSV.
    """
    model.fit(X_full, y_full)

    test_proba = model.predict_proba(X_test)[:, 1]

    submission = pd.DataFrame({
        ID_COL: test_ids,
        TARGET_COL: test_proba
    })

    submission.to_csv("submission_gradient_boosting.csv", index=False)
    print("Saved Kaggle file: submission_gradient_boosting.csv")


def main():
    X, y, X_test, test_ids = load_and_prepare_data()

    X_train, X_valid, y_train, y_valid = train_valid_split(X, y)

    model = build_model()

    model, valid_auc = evaluate_model(model, X_train, X_valid, y_train, y_valid)

    train_full_and_save_submission(model, X, y, X_test, test_ids)


if __name__ == "__main__":
    main()
