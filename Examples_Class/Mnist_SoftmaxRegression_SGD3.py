# Introduction to Machine Learning
# MNIST Dataset
# Softmax regression with stochastic gradient descent
# By Juan Carlos Rojas
# Copyright 2025, Texas Tech University - Costa Rica

import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import torch

# ============================================================
# Load and prepare data
# ============================================================

# Load the training and test data from the Pickle file
with open("mnist_dataset.pickle", "rb") as f:
      train_data, train_labels, test_data, test_labels = pickle.load(f)

# Scale pixels to range 0-1.0
maxval = train_data.max()
train_data = train_data / maxval
test_data = test_data / maxval

# Get some lengths
num_classes = len(np.unique(train_labels))
n_inputs = train_data.shape[1]
nsamples = train_data.shape[0]

# ============================================================
# Training constants
# ============================================================

learning_rate = 1.0
batch_size = 6000
num_epochs = 10
eval_step = 1

# Print the configuration
print(f"Num epochs: {num_epochs}  Batch size: {batch_size}  Learning rate: {learning_rate}")

num_batches = int(np.ceil(nsamples / batch_size))
total_iterations = num_epochs * num_batches
print(f"Number of batches per epoch: {num_batches}")
print(f"Total training iterations: {total_iterations}")

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
X = torch.tensor(train_data, dtype=torch.float32, device=device)
Y = torch.tensor(train_labels, dtype=torch.long, device=device)
X_test = torch.tensor(test_data, dtype=torch.float32, device=device)
Y_test = torch.tensor(test_labels, dtype=torch.long, device=device)

# ============================================================
# Create and initialize weights and bias
# ============================================================

# Xavier (Glorot) normal initialization 
W = torch.empty((n_inputs, num_classes), device=device)
torch.nn.init.xavier_normal_(W)

B = torch.zeros(num_classes, device=device)

# Enable gradient tracking
W.requires_grad_(True)
B.requires_grad_(True)

# ============================================================
# Training loop
# ============================================================
train_cost_hist = []
test_acc_hist = []

start_time = time.time()

iteration_count = 0
for epoch in range(num_epochs):
    for batch_idx in range(num_batches):

        # Get the batch data
        start = batch_idx * batch_size
        end = min(start + batch_size, nsamples)
        X_batch = X[start:end]
        Y_batch = Y[start:end]

        # Forward pass: logits
        logits = X_batch @ W + B

        # Compute cross-entropy loss
        cost = torch.nn.functional.cross_entropy(logits, Y_batch)

        # Backpropagate
        cost.backward()

        # Parameter update (gradient descent)
        with torch.no_grad():
            W -= learning_rate * W.grad
            B -= learning_rate * B.grad

        # Zero gradients for next iteration
        W.grad.zero_()
        B.grad.zero_()

        # Evaluate after a certain number of iterations
        iteration_count += 1
        if iteration_count % eval_step == 0:
            with torch.no_grad():
                train_cost = cost.item()

                test_logits = X_test @ W + B
                test_pred = torch.argmax(test_logits, dim=1)
                test_acc = (test_pred == Y_test).float().mean().item()

                train_cost_hist.append(train_cost)
                test_acc_hist.append(test_acc)

                print(f"Epoch {epoch+1:3d}, Batch {batch_idx+1:3d}: Train cost: {train_cost:.4f}  Test Acc: {test_acc:.4f}")

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")    

# ============================================================
# Plot cost and accuracy evolution
# ============================================================
iterations_hist = [i for i in range(0, total_iterations, eval_step)]
plt.plot(iterations_hist, train_cost_hist, "b")
plt.xlabel("Iteration")
plt.ylabel("Train Cost")
plt.title("Train Cost Evolution")

plt.figure()
plt.plot(iterations_hist, test_acc_hist, "r")
plt.xlabel("Iteration")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy Evolution")
plt.show()