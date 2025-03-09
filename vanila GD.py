import numpy as np
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
# Select only 1000 training examples
x_train = x_train[:1000]
y_train = y_train[:1000]
def sigmoid_derivative_from_activation(a):
    """
    Derivative of sigmoid given its output 'a' (since a = sigmoid(z)).
    That is: a * (1 - a)
    """
    return a * (1 - a)
def softmax(z):
    """Softmax activation for output layer."""
    # subtract max for numerical stability
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)
# -------------------------------
# Forward propagation function
# -------------------------------
def forward_propagation(x, weights, biases):

    a_values = []    #list of preactivation values of each layer
    h_values = [x]  # list of activation values of each layer including hidden layer
    L = len(weights)  # total number of layers (hidden + output)
    for i in range(L):
        a = np.dot(weights[i], h_values[-1]) + biases[i]     # result is a column vector. # h_values[-1] is activation output (vector) of the previous layer (most recently calculated).h_values[-1] dynamically points to the last element in the list
        a_values.append(a)
        if i == L - 1:
            # For output layer, use softmax to get probability distribution.
            h = softmax(a)
        else:
            # For hidden layers, use sigmoid activation.
            h = sigmoid(a)
        h_values.append(h)
    return a_values, h_values
# -------------------------------
# Backpropagation function
# -------------------------------
# def backpropagation(a_values, h_values, weights, y_true):

#     L = len(weights)  # number of layers
#     grad_weights = [np.zeros_like(W) for W in weights]
#     grad_biases  = [np.zeros_like(b) for b in biases]

#     # ---------- Output layer gradient -----------
#     # Using cross-entropy with softmax, the derivative for the preactivation is:
#     # delta_L = y_hat - e (i.e., softmax output minus one-hot true vector)
#     delta = h_values[-1] - y_true  # shape (output_size, 1)

#     # Gradients for output layer parameters:
#     grad_weights[L-1] = np.dot(delta, h_values[L-1].T)
#     grad_biases[L-1]  = delta

#     # ---------- Hidden layers gradients -----------
#     # Propagate error backwards from layer L-1 to layer 1.
#     for i in range(L-2, -1, -1):
#         # For hidden layer i+1, h_values[i+1] = sigmoid(a_values[i])
#         # The derivative of sigmoid is h*(1-h)
#         # Propagate delta back: delta = (W_(i+1)^T . delta_next) * sigmoid_deriv(a_i)
#         delta = np.dot(weights[i+1].T, delta) * sigmoid_derivative_from_activation(h_values[i+1])
#         grad_weights[i] = np.dot(delta, h_values[i].T)
#         grad_biases[i]  = delta

#     return grad_weights, grad_biases

# -------------------------------
# # Backpropagation function
# # -------------------------------
# def backpropagation(a_values, h_values, weights,biases,x, y_true):

#     L = len(weights)  # number of layers
#     grad_weights = [np.zeros_like(W) for W in weights]
#     grad_biases  = [np.zeros_like(b) for b in biases]
#     grad_a_values = [np.zeros_like(a) for a in a_values]
#     grad_h_values = [np.zeros_like(h) for h in h_values]

#     # ---------- Output layer gradient -----------
#     grad_a_values[L-1] = h_values[-1] - y_true
#     #grad_h_values[L-1] = h_values[-1]
#     #now calculate grad_W_k and grad_b_k from Lth layer to 1st layer
#     for i in range(L-1, -1, -1):
#       if i > 0:
#         #compute gradient w.r.t parameters
#         grad_weights[i] = np.dot(grad_a_values[i], h_values[i-1].T)
#         grad_biases[i]  = grad_a_values[i]
#         #compute gradinet w.r.t layer below
#         grad_h_values[i-1] = np.dot(weights[i].T, grad_a_values[i])
#         grad_a_values[i-1] = grad_h_values[i-1] * sigmoid_derivative_from_activation(h_values[i-1])
#       else:
#         grad_weights[i] = np.dot(grad_a_values[i], x.T)
#         grad_biases[i]  = grad_a_values[i]
#     return grad_weights, grad_biases
def backpropagation(a_values, h_values, weights, biases, x, y_true):

    L = len(weights)  # number of layers (hidden + output)
    h_values = h_values[1:]
    # Initialize gradient lists with same shapes as a_values and h_values
    grad_weights = [np.zeros_like(W) for W in weights]
    grad_biases  = [np.zeros_like(b) for b in biases]
    grad_a_values = [np.zeros_like(a) for a in a_values]  # gradient w.r.t preactivation (a)
    grad_h_values = [np.zeros_like(h) for h in h_values]   # gradient w.r.t activation (h)

    # ---------- Output Layer Gradient ----------
    # For output layer, using cross-entropy with softmax, the gradient w.r.t preactivation is:
    # δ_L = h_values[-1] - y_true
    grad_a_values[L-1] = h_values[-1] - y_true
    #grad_h_values[L-1] = h_values[-1]  # (this value is not used further)

    # ---------- Backpropagate for Hidden Layers ----------
    # Loop from the output layer (index L-1) backward to the first layer (index 0)
    for i in range(L-1, -1, -1):
        if i == 0:
            grad_weights[i] = np.dot(grad_a_values[i], x.T)

        else:
            grad_weights[i] = np.dot(grad_a_values[i], h_values[i-1].T)
        grad_biases[i]  = grad_a_values[i]
        if i > 0:
            grad_h_values[i-1] = np.dot(weights[i].T, grad_a_values[i])
            # For the hidden layers (i-1 >= 1), apply the derivative of the sigmoid.
            # For the input layer (i-1 == 0), no activation derivative is applied.
            grad_a_values[i-1] = grad_h_values[i-1] * sigmoid_derivative_from_activation(h_values[i-1])

    return grad_weights, grad_biases

#-----------------------------
# Training function using full batch gradient descent
# -------------------------------
def train_network(X, Y, weights, biases, epochs, learning_rate):

    m = X.shape[0]  # number of training examples

    for epoch in range(epochs):
        # Initialize the gradients and loss
        sum_grad_weights = [np.zeros_like(W) for W in weights]   #creates a list with each element is a zero matrix with the same size as the corresponding weight matrix in weights. No. of layers X size of layer(weight matrix of each layer)
        sum_grad_biases  = [np.zeros_like(b) for b in biases]    #creates a list of zero vectors that matches the corresponding bias vector # size is No. of layers X No. of neuron in a layer
        total_loss = 0
        # Process each training example
        for i in range(m):
            # Get the i-th training example and reshape as a column vector.
            x = X[i].reshape(-1, 1)  # Reshapes the selected input sample from (784,) to (784, 1). its a 2d column vector of 784X1
            y_true = Y[i].reshape(-1, 1)  # Reshapes the selected input sample from (10,) to (10, 1). its a 2d column vector of 10X1

            # Forward propagation
            a_vals, h_vals = forward_propagation(x, weights, biases)

            # Compute loss for this example (cross-entropy)
            y_hat = h_vals[-1]    # fetches activation value at output layer
            # Add a small epsilon to avoid log(0)
            loss = -np.sum(y_true * np.log(y_hat + 1e-8))
            total_loss += loss

            # Backpropagation to compute gradients for this example
            grad_w, grad_b = backpropagation(a_vals, h_vals, weights,biases,x, y_true)

            # Accumulate gradients
            for j in range(len(weights)):
                sum_grad_weights[j] += grad_w[j]
                sum_grad_biases[j]  += grad_b[j]

        # Average gradients over all examples
        avg_grad_weights = [g / m for g in sum_grad_weights]
        avg_grad_biases  = [g / m for g in sum_grad_biases]

        # Update parameters using gradient descent rule
        for j in range(len(weights)):
            weights[j] -= learning_rate * avg_grad_weights[j]
            biases[j]  -= learning_rate * avg_grad_biases[j]

        # Compute average loss for the epoch
        avg_loss = total_loss / m

        # Calculate training accuracy for monitoring
        correct = 0
        for i in range(m):
            x = X[i].reshape(-1, 1)
            _, h_vals = forward_propagation(x, weights, biases)
            y_hat = h_vals[-1]
            if np.argmax(y_hat) == np.argmax(Y[i]):
                correct += 1
        accuracy = correct / m * 100
        print(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%")

    return weights, biases
import numpy as np
from keras.utils import to_categorical
# Main code:

# Preprocess: Flatten images and normalize pixel values to [0,1]
x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # reshape: (m, 784)  #x_train.shape[0] gets no. of training example
x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0     # earlier shape mx28x28 after reshape mx784

# One-hot encode labels (10 classes)
y_train = to_categorical(y_train, 10)      #eg: any train ex class is 9 then y_train_oh is 0 0 0 0 0 0 0 0 0 1, y_train is mX10 size
y_test  = to_categorical(y_test, 10)

# Set hyperparameters
epochs = 20
learning_rate = 0.001
total_hidden_layers= 5
hidden_layer_size = 128
num_layers= total_hidden_layers+1

# Define network architecture. # Example: one hidden layer with 128 nodes. You can add more layers.
input_size = 784      #input layer feature vector size
output_size = 10       #output layer feature vector size
layer_sizes = [input_size] + [hidden_layer_size] * total_hidden_layers + [output_size]



# Initialize weights and biases for each layer
weights = []
biases = []

for i in range(num_layers):
    W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01  #  Weights are initialized with small random values for stable training. W[0] size is 32X784
    b = np.zeros((layer_sizes[i+1], 1))                            # Biases initialized to zero, b[0] size is 32X1
    weights.append(W)
    biases.append(b)
# Train the network using full batch gradient descent
weights, biases = train_network(x_train, y_train, weights, biases, epochs, learning_rate)

# -------------------------------
# Evaluate on test set
# -------------------------------
correct = 0
m_test = x_test.shape[0]
for i in range(m_test):
    x = x_test[i].reshape(-1, 1)
    _, h_vals = forward_propagation(x, weights, biases)
    y_hat = h_vals[-1]
    if np.argmax(y_hat) == np.argmax(y_test[i]):
        correct += 1
test_accuracy = correct / m_test * 100
print("Test Accuracy: {:.2f}%".format(test_accuracy))
