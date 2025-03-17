!pip install tabulate
import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from tabulate import tabulate

# -------------------------------
# Utility functions: activations
# -------------------------------

def sigmoid(z):
    """Logistic (sigmoid) activation function."""
    return 1 / (1 + np.exp(-z))


def softmax(z):
    """Softmax activation for output layer."""
    z = z - np.max(z, keepdims=True)  # Ensure correct dimension handling
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, keepdims=True)


# -------------------------------
# Forward propagation function
# -------------------------------
def compute_probability(x, weights, biases):
    a_values = []    # List of pre-activation values of each layer
    h_values = [x]   # List of activation values including input layer
    L = len(weights)  # Total number of layers (hidden + output)

    for i in range(L):
        a = np.dot(weights[i], h_values[-1]) + biases[i]
        a_values.append(a)

        if i == L - 1:
            h = softmax(a)   # Output layer uses softmax
        else:
            h = sigmoid(a)   # Hidden layers use sigmoid

        h_values.append(h)

    return h_values[-1]

# Main code:

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Select only 10 training examples for demonstration
x_train = x_train[:10]
y_train = y_train[:10]

# Preprocess: Flatten images and normalize pixel values to [0, 1]
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0

# One-hot encode labels (10 classes)
y_train = to_categorical(y_train, 10)

# Set hyperparameters
total_hidden_layers = 3
hidden_layer_size = 64
num_layers = total_hidden_layers + 1

# Define network architecture
input_size = 784
output_size = 10
layer_sizes = [input_size] + [hidden_layer_size] * total_hidden_layers + [output_size]
print(layer_sizes)
print('\n')
m = x_train.shape[0]

# Initialize weights and biases for each layer
weights = []
biases = []


for i in range(num_layers):
        fan_in = layer_sizes[i]
        W = np.random.randn(layer_sizes[i+1], fan_in) * np.sqrt(1.0 / fan_in)
        b = np.zeros((layer_sizes[i+1], 1))
        weights.append(W)
        biases.append(b)


# Prepare headers for the table
headers = ["Sample"] + [f"Class {i}" for i in range(10)]
table_rows = []

for i in range(m):
    x = x_train[i].reshape(-1, 1)
    y_hat = compute_probability(x, weights, biases)
    # Create a row with sample index and probabilities formatted to 4 decimals
    row = [i] + [f"{prob[0]:.4f}" for prob in y_hat]
    table_rows.append(row)

# Print the table with borders around each cell using "grid" format
print(tabulate(table_rows, headers=headers, tablefmt="fancy_grid"))