import numpy as np

# Define the structure of the neural network
input_size = 2    # 2 input neurons
hidden_size = 3   # 3 neurons in the hidden layer
output_size = 1   # 1 output neuron

# Initialize weights randomly
np.random.seed(42)  # Ensures reproducibility
W1 = np.random.randn(input_size, hidden_size)  # Weights for Input → Hidden
b1 = np.zeros((1, hidden_size))  # Bias for hidden layer
W2 = np.random.randn(hidden_size, output_size)  # Weights for Hidden → Output
b2 = np.zeros((1, output_size))  # Bias for output layer

print("W1:", W1)
print("b1:", b1)
print("W2:", W2)
print("b2:", b2)

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Forward propagation function
def forward_propagation(X):
    global W1, b1, W2, b2  # Use global weights and biases
    
    # Compute Hidden Layer Activations
    Z1 = np.dot(X, W1) + b1  # Weighted sum
    A1 = sigmoid(Z1)         # Activation function
    
    # Compute Output Layer Activations
    Z2 = np.dot(A1, W2) + b2  # Weighted sum
    A2 = sigmoid(Z2)          # Activation function
    
    return A1, A2  # Final prediction

# Sample Input (2 features per example)
X_sample = np.array([[0.5, 0.8]])  # Example input
A1, output = forward_propagation(X_sample)

print("Network Output:", output)

# Binary Cross-Entropy Loss function
def compute_loss(y_true, y_pred):
    m = y_true.shape[0]  # Number of examples
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Sample Target Value (True Output)
y_sample = np.array([[1]])  # Expected output

# Compute loss for the sample input
loss = compute_loss(y_sample, output)
print("Loss:", loss)

# Learning rate (α)
learning_rate = 0.1

# Backpropagation: Adjust weights based on gradients
def backpropagation(X, y_true, A1, y_pred):
    global W1, b1, W2, b2  # Use global weights

    # Compute Gradients
    dA2 = y_pred - y_true  # Gradient of output layer
    dW2 = np.dot(A1.T, dA2) / X.shape[0]  # Gradient of W2
    db2 = np.sum(dA2, axis=0, keepdims=True) / X.shape[0]  # Gradient of b2

    dA1 = np.dot(dA2, W2.T) * A1 * (1 - A1)  # Gradient of hidden layer (Sigmoid Derivative)
    dW1 = np.dot(X.T, dA1) / X.shape[0]  # Gradient of W1
    db1 = np.sum(dA1, axis=0, keepdims=True) / X.shape[0]  # Gradient of b1

    # Update Weights
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# Training Loop
epochs = 100

for epoch in range(epochs):
    A1, y_pred = forward_propagation(X_sample)  # Forward pass
    loss = compute_loss(y_sample, y_pred)  # Compute loss
    backpropagation(X_sample, y_sample, A1, y_pred)  # Update weights
    
    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")

print("Training Complete!")
