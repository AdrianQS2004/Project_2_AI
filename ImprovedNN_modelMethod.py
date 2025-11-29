# Improved Neural Network with Hyperparameter Tuning
# Introduction to Artificial Intelligence
# Loan Default Prediction Competition
# By Team
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import torch
import torch.nn as nn
import time

# ============================================================
# Load and prepare data
# ============================================================

training_db = pd.read_csv("Datasets/train.csv", header=0)
test_db = pd.read_csv("Datasets/test.csv", header=0)

training_db = pd.get_dummies(training_db, prefix_sep="_", drop_first=True, dtype=int)
labels = training_db["loan_paid_back"]
ids = test_db['id']
training_db = training_db.drop(columns=["loan_paid_back", "id"])

test_db = test_db.drop(columns=["id"])
test_db = pd.get_dummies(test_db, prefix_sep="_", drop_first=True, dtype=int)

# Align test columns to match training columns
test_db = test_db.reindex(columns=training_db.columns, fill_value=0)

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()

# Split for validation
train_data_split, val_data, train_labels_split, val_labels = \
    sklearn.model_selection.train_test_split(
        train_data, train_labels,
        test_size=0.2, shuffle=True, random_state=2025
    )

# Standardize
train_means = train_data_split.mean()
train_stds = train_data_split.std().replace(0, 1)
train_data_split = (train_data_split - train_means) / train_stds
val_data = (val_data - train_means) / train_stds
test_data = (test_data - train_means) / train_stds

n_inputs = train_data_split.shape[1]
nsamples = train_data_split.shape[0]

# ============================================================
# Hyperparameters - Can be tuned
# ============================================================

# Architecture
hidden_layers = [128, 64, 32]  # Number of nodes in each hidden layer
dropout_rate = 0.3  # Dropout for regularization
activation = 'ELU'  # Options: 'ReLU', 'ELU', 'Tanh'

# Training
batch_size = 2048
learning_rate = 1e-4
n_epochs = 500
eval_step = 10
early_stop_patience = 20

# Regularization
weight_decay = 1e-5  # L2 regularization

print("="*80)
print("IMPROVED NEURAL NETWORK")
print("="*80)
print(f"Architecture: {n_inputs} -> {' -> '.join(map(str, hidden_layers))} -> 1")
print(f"Activation: {activation}")
print(f"Dropout: {dropout_rate}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Max epochs: {n_epochs}")
print(f"Weight decay (L2): {weight_decay}")
print(f"Early stop patience: {early_stop_patience}")
print("="*80)

num_batches = int(np.ceil(nsamples / batch_size))
print(f"Number of batches per epoch: {num_batches}\n")

# ============================================================
# Select PyTorch device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB\n")

# ============================================================
# Convert data to PyTorch tensors
# ============================================================
X = torch.tensor(train_data_split.values, dtype=torch.float32, device=device)
Y = torch.tensor(train_labels_split.values.reshape(-1, 1), dtype=torch.float32, device=device)

X_val = torch.tensor(val_data.values, dtype=torch.float32, device=device)
Y_val = torch.tensor(val_labels.values.reshape(-1, 1), dtype=torch.float32, device=device)

X_test = torch.tensor(test_data.values, dtype=torch.float32, device=device)

# ============================================================
# Create neural network model
# ============================================================

def create_model(n_inputs, hidden_layers, dropout_rate, activation):
    """Create a neural network with specified architecture"""
    layers = []
    input_size = n_inputs
    
    # Activation function
    if activation == 'ReLU':
        act_fn = nn.ReLU()
    elif activation == 'ELU':
        act_fn = nn.ELU()
    elif activation == 'Tanh':
        act_fn = nn.Tanh()
    else:
        act_fn = nn.ELU()
    
    # Build hidden layers
    for hidden_size in hidden_layers:
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(act_fn)
        layers.append(nn.Dropout(dropout_rate))
        input_size = hidden_size
    
    # Output layer (no activation, will use sigmoid in loss)
    layers.append(nn.Linear(input_size, 1))
    
    return nn.Sequential(*layers)

model = create_model(n_inputs, hidden_layers, dropout_rate, activation)
model.to(device)
print(model)
print()

# Initialize weights
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

model.apply(init_weights)

# ============================================================
# Optimizer and Loss
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
loss_fn = nn.BCEWithLogitsLoss()

print(f"Optimizer: Adam (lr={learning_rate}, weight_decay={weight_decay})")
print(f"Loss: Binary Cross-Entropy with Logits\n")

# ============================================================
# Training loop
# ============================================================

train_loss_hist = []
val_roc_auc_hist = []

# Best model tracking
best_val_roc_auc = float('-inf')
best_epoch = -1
best_state_dict = None
epochs_without_improvement = 0

start_time = time.time()

print("Starting training...")
print("-" * 80)

for epoch in range(n_epochs):
    
    # Training mode
    model.train()
    epoch_loss = 0.0

    for batch_idx in range(num_batches):
        start_i = batch_idx * batch_size
        end_i = min(start_i + batch_size, nsamples)

        X_batch = X[start_i:end_i]
        Y_batch = Y[start_i:end_i]

        # Forward pass
        logits = model(X_batch)
        loss = loss_fn(logits, Y_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Evaluation
    if (epoch + 1) % eval_step == 0:
        avg_train_loss = epoch_loss / num_batches
        train_loss_hist.append(avg_train_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_pred_proba = torch.sigmoid(val_logits).cpu().numpy().flatten()
            val_roc_auc = sklearn.metrics.roc_auc_score(val_labels.values, val_pred_proba)

        val_roc_auc_hist.append(val_roc_auc)

        print_line = f"Epoch {epoch+1:4d} | Train Loss: {avg_train_loss:.4f} | Val ROC AUC: {val_roc_auc:.6f}"

        # Check for improvement
        if val_roc_auc > best_val_roc_auc:
            best_val_roc_auc = val_roc_auc
            best_epoch = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print_line += "  [NEW BEST]"
        else:
            epochs_without_improvement += 1

        print(print_line)

        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            print(f"Best validation ROC AUC: {best_val_roc_auc:.6f} at epoch {best_epoch}")
            break

elapsed_time = time.time() - start_time
print(f"\nTraining completed in {elapsed_time:.1f} seconds")

# ============================================================
# Restore best model and final evaluation
# ============================================================

print("\n" + "="*80)
print(f"Restoring best model (epoch {best_epoch}, Val ROC AUC: {best_val_roc_auc:.6f})")
print("="*80)

model.load_state_dict(best_state_dict)
model.eval()

with torch.no_grad():
    # Final evaluation
    train_logits = model(X)
    train_pred_proba = torch.sigmoid(train_logits).cpu().numpy().flatten()
    train_roc_auc = sklearn.metrics.roc_auc_score(train_labels_split.values, train_pred_proba)

    val_logits = model(X_val)
    val_pred_proba = torch.sigmoid(val_logits).cpu().numpy().flatten()
    final_val_roc_auc = sklearn.metrics.roc_auc_score(val_labels.values, val_pred_proba)

    # Test predictions
    test_logits = model(X_test)
    test_pred_proba = torch.sigmoid(test_logits).cpu().numpy().flatten()

print(f"\nFinal Evaluation:")
print(f"  Train ROC AUC: {train_roc_auc:.6f}")
print(f"  Val ROC AUC: {final_val_roc_auc:.6f}")

# ============================================================
# Generate submission
# ============================================================

submission = pd.DataFrame({'id': ids, 'loan_paid_back': test_pred_proba})
submission.to_csv('my_submission.csv', index=False)
print(f"\nWrote submission 'my_submission.csv' with {len(submission)} rows")

# ============================================================
# Plot training history
# ============================================================

epochs_hist = np.arange(eval_step, len(train_loss_hist) * eval_step + 1, eval_step)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_hist, train_loss_hist, 'b-', label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs_hist, val_roc_auc_hist, 'r-', label='Val ROC AUC')
plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best (epoch {best_epoch})')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('Validation ROC AUC Evolution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nn_training_history.png', dpi=150, bbox_inches='tight')
print("Saved training history plot to 'nn_training_history.png'")
# plt.show()

# Save hyperparameters
hyperparameters = {
    'hidden_layers': hidden_layers,
    'dropout_rate': dropout_rate,
    'activation': activation,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'n_epochs': n_epochs,
    'weight_decay': weight_decay,
    'early_stop_patience': early_stop_patience
}

print(f"\nHyperparameters used: {hyperparameters}")

