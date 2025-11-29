# Example: How to Track Model Results
# This script shows how to use the ResultsTracker to record experiment results

from results_tracker import ResultsTracker

# Initialize the tracker
tracker = ResultsTracker("model_results.csv")

# Example: Add results from Baseline Model 1
tracker.add_result(
    model_name="Baseline_1_LogisticRegression_PyTorch",
    model_type="Baseline",
    roc_auc_train=0.8234,  # Replace with actual value from model output
    roc_auc_val=0.8156,    # Replace with actual value from model output
    hyperparameters={
        'n_iterations': 5000,
        'learning_rate': 0.1
    },
    notes="PyTorch implementation with manual gradient descent",
    submission_file="my_submission.csv"
)

# Example: Add results from Random Forest
tracker.add_result(
    model_name="Classic_1_RandomForest",
    model_type="Classic",
    roc_auc_train=0.8623,
    roc_auc_val=0.8541,
    hyperparameters={
        'n_estimators': 200,
        'max_depth': 20,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    },
    notes="Random Forest with default hyperparameters",
    submission_file="my_submission.csv"
)

# Example: Add results from Improved Neural Network
tracker.add_result(
    model_name="NeuralNetwork_Improved_v1",
    model_type="Neural Network",
    roc_auc_train=0.8756,
    roc_auc_val=0.8612,
    hyperparameters={
        'hidden_layers': [128, 64, 32],
        'dropout_rate': 0.3,
        'activation': 'ELU',
        'batch_size': 2048,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5
    },
    notes="Improved NN with 3 hidden layers, dropout, and L2 regularization",
    submission_file="my_submission.csv"
)

# Print summary of all results
tracker.print_summary()

# Get the best model
best = tracker.get_best_model()
if best:
    print(f"\n🏆 Best Model: {best['model_name']}")
    print(f"   Validation ROC AUC: {best['roc_auc_val']:.4f}")
    print(f"   Submission file: {best['submission_file']}")

