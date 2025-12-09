# Intro to AI: Project 2
# Luis Baeza, Adrian Quiros, Adrian de Souza
# Shallow Neural Network Model - Method 
# This script uses the train data to train the model and then predicts on the test data for submission.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

# ============================================================
# Load and prepare data
# ============================================================

# Load and prepare the data
training_db = pd.read_csv("Datasets/train.csv", header=0)
test_db = pd.read_csv("Datasets/test.csv", header=0)


training_db = pd.get_dummies(training_db, prefix_sep="_", drop_first=True, dtype=int)
labels = training_db["loan_paid_back"]
ids = test_db['id']
training_db = training_db.drop(columns=["loan_paid_back", "id"])

test_db = test_db.drop(columns=["id"])
test_db = pd.get_dummies(test_db, prefix_sep="_", drop_first=True, dtype=int)

# Align test columns to match training columns (fill missing with 0, drop extra)
test_db = test_db.reindex(columns=training_db.columns, fill_value=0)

train_data = training_db.copy()
train_labels = labels.copy()
test_data = test_db.copy()


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
batch_size = 2048    
learning_rate = 1e-3
n_epochs = 900
L2_regularization_rate = 1e-5
eval_step = 1

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

# ============================================================
# Create and initialize neural network
# ============================================================

model = torch.nn.Sequential(
    torch.nn.Linear(n_inputs, n_nodes_l1),
    torch.nn.ELU(),
    torch.nn.Linear(n_nodes_l1, 1)
)
model.to(device)
print(model)

# Initialize weights explicitly
torch.nn.init.kaiming_normal_(model[0].weight, nonlinearity="relu")
torch.nn.init.xavier_normal_(model[2].weight)

# ============================================================
# Optimizer and Loss
# ============================================================
# Normal Optimizer without weight decay
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Weight decay optimizer / Regularization
#optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=L2_regularization_rate)

loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"Optimizer: Adam with binary cross-entropy loss.  Learning rate: {learning_rate}")

# ============================================================
# Training loop
# ============================================================

train_loss_hist = []

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

    avg_train_loss = epoch_loss / num_batches
    train_loss_hist.append(avg_train_loss)

    if (epoch + 1) % eval_step == 0:
        print(f"Epoch {epoch+1:3d}, Train loss: {avg_train_loss:.4f}")


elapsed_time = time.time() - start_time
print("Execution time: {:.1f}".format(elapsed_time))

# ============================================================
# Uses the model for prediction
# ============================================================

model.eval()

with torch.no_grad():
    new_logits = model(X_test)
    preds = torch.sigmoid(new_logits).cpu().numpy().flatten()


# Build submission DataFrame using the original ids column
submission = pd.DataFrame({'id': ids, 'loan_paid_back': preds})
submission.to_csv('my_submission.csv', index=False)
print(f"Wrote submission 'my_submission.csv' with {len(submission)} rows")

# ============================================================
# Plots the training loss
# ============================================================
epochs_hist = np.arange(1, n_epochs + 1)
plt.plot(epochs_hist, train_loss_hist, "b")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss Evolution")
plt.show()