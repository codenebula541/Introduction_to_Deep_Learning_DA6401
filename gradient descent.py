import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

# -------------------------------
# Activation functions and derivatives
# -------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    # a is the output of sigmoid(z)
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    # derivative of tanh is 1 - a^2, where a = tanh(z)
    return 1 - a**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
    # For ReLU, if a is the output (which equals z for positive inputs), then derivative is 1 if a > 0 else 0.
    return (a > 0).astype(float)

def activation(z, act_type):
    if act_type == 'sigmoid':
        return sigmoid(z)
    elif act_type == 'tanh':
        return tanh(z)
    elif act_type == 'relu':
        return relu(z)
    else:
        raise ValueError("Unsupported activation type")

def activation_derivative(a, act_type):
    if act_type == 'sigmoid':
        return sigmoid_derivative(a)
    elif act_type == 'tanh':
        return tanh_derivative(a)
    elif act_type == 'relu':
        return relu_derivative(a)
    else:
        raise ValueError("Unsupported activation type")

def softmax(z):
    z = z - np.max(z)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)

# -------------------------------
# Weight initialization function
# -------------------------------
def initialize_parameters(layer_sizes, weight_init='random'):
    weights = []
    biases = []
    for i in range(len(layer_sizes)-1):
        fan_in = layer_sizes[i]
        if weight_init == 'random':
            W = np.random.randn(layer_sizes[i+1], fan_in) * 0.01
        elif weight_init == 'xavier':
            W = np.random.randn(layer_sizes[i+1], fan_in) * np.sqrt(1.0 / fan_in)
        else:
            raise ValueError("Unsupported weight initialization method")
        b = np.zeros((layer_sizes[i+1], 1))
        weights.append(W)
        biases.append(b)
    return weights, biases

# -------------------------------
# Forward propagation function
# -------------------------------
def forward_propagation(x, weights, biases, activation_type='sigmoid'):
    """
    x: input column vector (shape: (input_size, 1))
    weights, biases: lists of parameters for each layer.
    activation_type: activation for hidden layers.

    Returns:
       a_values: list of preactivation outputs.
       h_values: list of activation outputs, with h_values[0] being the activation of layer 1 (input is not included).
    """
    a_values = []
    h_values = [x]  # temporary: index 0 is input; we will remove it later.
    L = len(weights)
    for i in range(L):
        a = np.dot(weights[i], h_values[-1]) + biases[i]
        a_values.append(a)
        if i == L - 1:
            # Output layer uses softmax
            h = softmax(a)
        else:
            h = activation(a, activation_type)
        h_values.append(h)
    # Remove the input so that h_values[0] is now output of first layer.
    h_values = h_values[1:]
    return a_values, h_values

# -------------------------------
# Backpropagation function
# -------------------------------
def backpropagation(a_values, h_values, weights, biases, x, y_true, activation_type='sigmoid'):
    """
    Computes gradients for one training example.

    a_values: list of preactivation outputs for each layer.
    h_values: list of activation outputs for each layer (first element corresponds to first layer output).
    weights, biases: current parameters.
    x: input column vector (shape: (input_size, 1)).
    y_true: true label in one-hot format (shape: (output_size, 1)).
    activation_type: activation function used in hidden layers.

    Returns:
       grad_weights, grad_biases: gradients for each parameter.
    """
    L = len(weights)
    # Initialize gradient lists
    grad_weights = [np.zeros_like(W) for W in weights]
    grad_biases  = [np.zeros_like(b) for b in biases]
    grad_a_values = [np.zeros_like(a) for a in a_values]
    grad_h_values = [np.zeros_like(h) for h in h_values]

    # ---------- Output layer gradient ----------
    grad_a_values[L-1] = h_values[-1] - y_true  # δ_L = output - true
    # ---------- Backpropagate through hidden layers ----------
    for i in range(L-1, -1, -1):
        if i == 0:
            grad_weights[i] = np.dot(grad_a_values[i], x.T)
        else:
            grad_weights[i] = np.dot(grad_a_values[i], h_values[i-1].T)
        grad_biases[i] = grad_a_values[i]
        if i > 0:
            grad_h_values[i-1] = np.dot(weights[i].T, grad_a_values[i])
            grad_a_values[i-1] = grad_h_values[i-1] * activation_derivative(h_values[i-1], activation_type)
    return grad_weights, grad_biases

# -------------------------------
# Training function using mini-batch gradient descent and L2 regularization (weight decay)
# -------------------------------
def train_network(X, Y, weights, biases, epochs, learning_rate, batch_size=32, activation_type='sigmoid', weight_decay=0.0):
    """
    X: training data, shape (m, input_size)
    Y: training labels, one-hot encoded, shape (m, output_size)
    weights, biases: initial parameters.
    epochs: number of training epochs.
    learning_rate: learning rate.
    batch_size: mini-batch size.
    activation_type: activation function for hidden layers.
    weight_decay: L2 regularization coefficient.

    Returns:
       weights, biases: trained parameters.
    """
    m = X.shape[0]
    num_batches = int(np.ceil(m / batch_size))
    for epoch in range(epochs):
        total_loss = 0
        # Shuffle the data for each epoch
        indices = np.arange(m)
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]

        for b in range(num_batches):
            start = b * batch_size
            end = min((b+1)*batch_size, m)
            batch_size_actual = end - start

            # Initialize accumulators for gradients for this batch
            batch_grad_weights = [np.zeros_like(W) for W in weights]
            batch_grad_biases = [np.zeros_like(bias) for bias in biases]
            batch_loss = 0

            for i in range(start, end):
                x = X[i].reshape(-1, 1)   # (input_size, 1)
                y_true = Y[i].reshape(-1, 1)  # (output_size, 1)

                # Forward propagation
                a_vals, h_vals = forward_propagation(x, weights, biases, activation_type)
                y_hat = h_vals[-1]  # output layer activation

                # Compute cross-entropy loss (with epsilon for stability)
                loss = -np.sum(y_true * np.log(y_hat + 1e-8))
                batch_loss += loss

                # Backpropagation
                grad_w, grad_b = backpropagation(a_vals, h_vals, weights, biases, x, y_true, activation_type)

                # Accumulate gradients for the batch
                for j in range(len(weights)):
                    batch_grad_weights[j] += grad_w[j]
                    batch_grad_biases[j] += grad_b[j]

            # Average gradients over the batch and add L2 regularization term
            for j in range(len(weights)):
                batch_grad_weights[j] = batch_grad_weights[j] / batch_size_actual + weight_decay * weights[j]
                weights[j] -= learning_rate * batch_grad_weights[j]
                biases[j]  -= learning_rate * (batch_grad_biases[j] / batch_size_actual)

            total_loss += batch_loss

        avg_loss = total_loss / m

        # Compute training accuracy on the entire training set
        correct = 0
        for i in range(m):
            x = X[i].reshape(-1, 1)
            _, h_vals = forward_propagation(x, weights, biases, activation_type)
            y_hat = h_vals[-1]
            if np.argmax(y_hat) == np.argmax(Y[i]):
                correct += 1
        accuracy = correct / m * 100
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    return weights, biases

# -------------------------------
# Main code
# -------------------------------

# Load Fashion-MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# For quick experimentation, you can select a subset (or use full data)
# Here, using all data; you can also slice to 1000 examples if desired.
x_train = x_train[:10000]
y_train = y_train[:10000]

# Preprocess: Flatten images and normalize to [0,1]
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # shape: (m, 784)
x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0

# One-hot encode labels (10 classes)
y_train = to_categorical(y_train, 10)
y_test  = to_categorical(y_test, 10)

# Set hyperparameters
epochs = 3
learning_rate = 0.001
batch_size = 32
total_hidden_layers = 4
hidden_layer_size = 128
input_size = 784
output_size = 10
layer_sizes = [input_size] + [hidden_layer_size] * total_hidden_layers + [output_size]

# Choose activation function: 'sigmoid', 'tanh', or 'relu'
activation_type = 'tanh'   # For example, try 'relu'

# Set weight decay (L2 regularization): e.g., 0, 0.0005, or 0.5
weight_decay = 0.0005

# Set weight initialization method: 'random' or 'xavier'
weight_init = 'xavier'

# Initialize weights and biases
weights, biases = initialize_parameters(layer_sizes, weight_init)


# Train the network
weights, biases = train_network(x_train, y_train, weights, biases, epochs, learning_rate, batch_size, activation_type, weight_decay)


# -------------------------------
# Evaluate on test set
# -------------------------------
correct = 0
m_test = x_test.shape[0]
for i in range(m_test):
    x = x_test[i].reshape(-1, 1)
    _, h_vals = forward_propagation(x, weights, biases, activation_type)
    y_hat = h_vals[-1]
    if np.argmax(y_hat) == np.argmax(y_test[i]):
        correct += 1
test_accuracy = correct / m_test * 100
print("Test Accuracy: {:.2f}%".format(test_accuracy))
