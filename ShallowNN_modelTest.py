# Shallow? or deep neural network?
# Introduction to Machine Learning
# Credit Default Dataset
# Shallow neural network
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.metrics
import torch
import time

# ============================================================
# Load and prepare data
# ============================================================

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

# Get some lengths
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants
# ============================================================
n_nodes_l1 = 8
batch_size = 2048       # Mini-batch gradient descent
learning_rate = 1e-4
L2_regularization_rate = 0
n_epochs = 1500
eval_step = 1
early_stop_patience = 10   # number of eval steps allowed without improvement

# Print the configuration
print(f"Num epochs: {n_epochs}  Batch size: {batch_size}  Learning rate: {learning_rate}")

num_batches = int(np.ceil(nsamples / batch_size))
print(f"Number of batches per epoch: {num_batches}")

# ============================================================
# Select PyTorch device (GPU if available)
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# ============================================================
# Convert data to PyTorch tensors and move to device
# ============================================================
X = torch.tensor(train_data.values, dtype=torch.float32, device=device)
Y = torch.tensor(train_labels.values.reshape(-1, 1), dtype=torch.float32, device=device)

X_test = torch.tensor(test_data.values, dtype=torch.float32, device=device)
Y_test = torch.tensor(test_labels.values.reshape(-1, 1), dtype=torch.float32, device=device)

# ============================================================
# Create and initialize neural network
# ============================================================

# Note that the final layer is skipping the sigmoid activation
# because we will use BCEWithLogitsLoss which combines a sigmoid layer
model = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, n_nodes_l1),
    #torch.nn.ELU(),
    #torch.nn.ReLU(),
    torch.nn.Tanh(),
    #torch.nn.LeakyReLU(),
    torch.nn.Linear(n_nodes_l1, 1)
)
model.to(device)
print(model)

# Initialize weights explicitly
torch.nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
torch.nn.init.xavier_normal_(model[2].weight)

# Use defaults for biases


# ============================================================
# Optimizer and Loss
# ============================================================
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=L2_regularization_rate)
loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"Optimizer: Adam with binary cross-entropy loss.  Learning rate: {learning_rate}")

# ============================================================
# Training loop
# ============================================================

train_loss_hist = []
test_roc_auc_hist = []

# Backup for best model
best_test_roc_auc = float('-inf')
best_epoch = -1
best_state_dict = None

epochs_without_improvement = 0

start_time = time.time()

for epoch in range(n_epochs):
    
    # Switch model to training mode
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

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Evaluation
    if (epoch + 1) % eval_step == 0:

        avg_train_loss = epoch_loss / num_batches
        train_loss_hist.append(avg_train_loss)

        # Switch model to evaluation mode 
        model.eval()

        with torch.no_grad():
            test_logits = model(X_test)
            test_pred_proba = torch.sigmoid(test_logits).cpu().numpy()
            test_roc_auc = sklearn.metrics.roc_auc_score(test_labels, test_pred_proba)

        test_roc_auc_hist.append(test_roc_auc)

        print_line = f"Epoch {epoch+1:3d}, Train loss: {avg_train_loss:.4f}, Test ROC AUC: {test_roc_auc:.4f}"

        if test_roc_auc > best_test_roc_auc:
            best_test_roc_auc = test_roc_auc
            best_epoch = epoch + 1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}

            epochs_without_improvement = 0   # reset patience counter
            print_line += "  New best"
        else:
            epochs_without_improvement += 1  # increment patience counter

        # Early stopping condition
        if epochs_without_improvement > early_stop_patience:
            print_line += "  **Early stop triggered**"
            print(print_line)
            break
        
        print(print_line)

elapsed_time = time.time() - start_time
print("Execution time: {:.1f}".format(elapsed_time))

# ============================================================
# Restore best model and final evaluation
# ============================================================

print("\nRestoring best model parameters (Test ROC AUC = {:.4f} Epoch = {})".format(best_test_roc_auc, best_epoch))
model.load_state_dict(best_state_dict)
model.eval()

with torch.no_grad():
    final_train_roc_auc = sklearn.metrics.roc_auc_score(train_labels, torch.sigmoid(model(X)).cpu().numpy())
    final_test_roc_auc = sklearn.metrics.roc_auc_score(test_labels, torch.sigmoid(model(X_test)).cpu().numpy())

print("\nFinal Evaluation (best model from epoch {}):".format(best_epoch))
print("Best Test  ROC AUC: {:.4f}".format(final_test_roc_auc))
print("     Train ROC AUC: {:.4f}".format(final_train_roc_auc))

# ============================================================
# Plot cost and accuracy evolution
# ============================================================
epochs_hist = np.arange(1, epoch + 1, eval_step)

# Plot train loss evolution
plt.plot(epochs_hist, train_loss_hist, "b")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss Evolution")
plt.figure()

# Plot test ROC AUC evolution
plt.plot(epochs_hist, test_roc_auc_hist, "r", label="Test ROC AUC")
plt.xlabel("Epoch")
plt.ylabel("ROC AUC")
plt.title("ROC AUC Evolution")
plt.legend()
plt.show()