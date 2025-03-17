!pip install wandb
# Log in to your W&B account
import wandb
import random
import math
wandb.login()

import numpy as np
import wandb
from keras.datasets import fashion_mnist
from keras.utils import to_categorical

sweep_config = {
    'name': "fixed_config",
    'method': 'grid',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'hiddenlayers': {'values': [5]},
        'num_epochs': {'values': [10]},
        'hiddennodes': {'values': [128]},
        'learning_rate': {'values': [0.0001]},
        'initializer': {'values': ["xavier"]},
        'batch_size': {'values': [16]},
        'opt': {'values': ["adam"]},
        'activation_func': {'values': ["relu"]},
        'weight_decay': {'values': [0.5]}
    }
}

sweep_id = wandb.sweep(sweep_config, project="DA6401-Assignment1")


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix

# -------------------------------
# Activation functions and derivatives
# -------------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    return a * (1 - a)

def tanh(z):
    return np.tanh(z)

def tanh_derivative(a):
    return 1 - a**2

def relu(z):
    return np.maximum(0, z)

def relu_derivative(a):
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
    z = z - np.max(z, keepdims=True)  # numerical stability
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, keepdims=True)

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
    a_values = []
    h_values = [x]  # h_values[0] is input; will remove later.
    L = len(weights)
    for i in range(L):
        a = np.dot(weights[i], h_values[-1]) + biases[i]
        a_values.append(a)
        if i == L - 1:
            h = softmax(a)
        else:
            h = activation(a, activation_type)
        h_values.append(h)
    h_values = h_values[1:]  # remove the input so that h_values[0] is the output of the first layer.
    return a_values, h_values

# -------------------------------
# Backpropagation function
# -------------------------------
def backpropagation(a_values, h_values, weights, biases, x, y_true, activation_type='sigmoid'):
    L = len(weights)
    grad_weights = [np.zeros_like(W) for W in weights]
    grad_biases  = [np.zeros_like(b) for b in biases]
    grad_a_values = [np.zeros_like(a) for a in a_values]
    grad_h_values = [np.zeros_like(h) for h in h_values]

    # Output layer gradient: δ_L = h_output - y_true
    grad_a_values[L-1] = h_values[-1] - y_true

    # Backpropagation for hidden layers
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
# Optimizer update functions
# -------------------------------
def initialize_optimizer_state(weights, biases, optimizer, **kwargs):
    state = {}
    num_layers = len(weights)
    if optimizer in ['sgd']:
        state = None
    elif optimizer in ['momentum', 'nesterov']:
        momentum = kwargs.get('momentum', 0.9)
        state['momentum'] = momentum
        state['v_w'] = [np.zeros_like(W) for W in weights]
        state['v_b'] = [np.zeros_like(b) for b in biases]
    elif optimizer == 'rmsprop':
        decay_rate = kwargs.get('decay_rate', 0.9)
        epsilon = kwargs.get('epsilon', 1e-8)
        state['decay_rate'] = decay_rate
        state['epsilon'] = epsilon
        state['s_w'] = [np.zeros_like(W) for W in weights]
        state['s_b'] = [np.zeros_like(b) for b in biases]
    elif optimizer in ['adam', 'nadam']:
        beta1 = kwargs.get('beta1', 0.9)
        beta2 = kwargs.get('beta2', 0.999)
        epsilon = kwargs.get('epsilon', 1e-8)
        state['beta1'] = beta1
        state['beta2'] = beta2
        state['epsilon'] = epsilon
        state['m_w'] = [np.zeros_like(W) for W in weights]
        state['v_w'] = [np.zeros_like(W) for W in weights]
        state['m_b'] = [np.zeros_like(b) for b in biases]
        state['v_b'] = [np.zeros_like(b) for b in biases]
        state['t'] = 0
    else:
        raise ValueError("Unsupported optimizer")
    return state

def update_parameters(weights, biases, grad_weights, grad_biases, optimizer_state, optimizer, learning_rate, t=1):
    if optimizer == 'sgd':
        for i in range(len(weights)):
            weights[i] -= learning_rate * grad_weights[i]
            biases[i]  -= learning_rate * grad_biases[i]
    elif optimizer == 'momentum':
        for i in range(len(weights)):
            optimizer_state['v_w'][i] = optimizer_state['momentum'] * optimizer_state['v_w'][i] + grad_weights[i]
            optimizer_state['v_b'][i] = optimizer_state['momentum'] * optimizer_state['v_b'][i] + grad_biases[i]
            weights[i] -= learning_rate * optimizer_state['v_w'][i]
            biases[i]  -= learning_rate * optimizer_state['v_b'][i]
    elif optimizer == 'nesterov':
        for i in range(len(weights)):
            v_prev_w = optimizer_state['v_w'][i].copy()
            v_prev_b = optimizer_state['v_b'][i].copy()
            optimizer_state['v_w'][i] = optimizer_state['momentum'] * optimizer_state['v_w'][i] + grad_weights[i]
            optimizer_state['v_b'][i] = optimizer_state['momentum'] * optimizer_state['v_b'][i] + grad_biases[i]
            weights[i] -= learning_rate * (optimizer_state['momentum'] * v_prev_w + (1 + optimizer_state['momentum']) * optimizer_state['v_w'][i])
            biases[i]  -= learning_rate * (optimizer_state['momentum'] * v_prev_b + (1 + optimizer_state['momentum']) * optimizer_state['v_b'][i])
    elif optimizer == 'rmsprop':
        for i in range(len(weights)):
            state = optimizer_state
            state['s_w'][i] = state['decay_rate'] * state['s_w'][i] + (1 - state['decay_rate']) * (grad_weights[i]**2)
            state['s_b'][i] = state['decay_rate'] * state['s_b'][i] + (1 - state['decay_rate']) * (grad_biases[i]**2)
            weights[i] -= learning_rate * grad_weights[i] / (np.sqrt(state['s_w'][i]) + state['epsilon'])
            biases[i]  -= learning_rate * grad_biases[i] / (np.sqrt(state['s_b'][i]) + state['epsilon'])
    elif optimizer == 'adam':
        for i in range(len(weights)):
            state = optimizer_state
            state['t'] += 1
            state['m_w'][i] = state['beta1'] * state['m_w'][i] + (1 - state['beta1']) * grad_weights[i]
            state['m_b'][i] = state['beta1'] * state['m_b'][i] + (1 - state['beta1']) * grad_biases[i]
            state['v_w'][i] = state['beta2'] * state['v_w'][i] + (1 - state['beta2']) * (grad_weights[i]**2)
            state['v_b'][i] = state['beta2'] * state['v_b'][i] + (1 - state['beta2']) * (grad_biases[i]**2)
            m_hat_w = state['m_w'][i] / (1 - state['beta1']**state['t'])
            m_hat_b = state['m_b'][i] / (1 - state['beta1']**state['t'])
            v_hat_w = state['v_w'][i] / (1 - state['beta2']**state['t'])
            v_hat_b = state['v_b'][i] / (1 - state['beta2']**state['t'])
            weights[i] -= learning_rate * m_hat_w / (np.sqrt(v_hat_w) + state['epsilon'])
            biases[i]  -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + state['epsilon'])
    elif optimizer == 'nadam':
        for i in range(len(weights)):
            state = optimizer_state
            state['t'] += 1
            state['m_w'][i] = state['beta1'] * state['m_w'][i] + (1 - state['beta1']) * grad_weights[i]
            state['m_b'][i] = state['beta1'] * state['m_b'][i] + (1 - state['beta1']) * grad_biases[i]
            state['v_w'][i] = state['beta2'] * state['v_w'][i] + (1 - state['beta2']) * (grad_weights[i]**2)
            state['v_b'][i] = state['beta2'] * state['v_b'][i] + (1 - state['beta2']) * (grad_biases[i]**2)
            m_hat_w = state['m_w'][i] / (1 - state['beta1']**state['t'])
            m_hat_b = state['m_b'][i] / (1 - state['beta1']**state['t'])
            v_hat_w = state['v_w'][i] / (1 - state['beta2']**state['t'])
            v_hat_b = state['v_b'][i] / (1 - state['beta2']**state['t'])
            weights[i] -= learning_rate * ((state['beta1'] * m_hat_w + (1 - state['beta1']) * grad_weights[i] / (1 - state['beta1']**state['t'])) / (np.sqrt(v_hat_w) + state['epsilon']))
            biases[i]  -= learning_rate * ((state['beta1'] * m_hat_b + (1 - state['beta1']) * grad_biases[i] / (1 - state['beta1']**state['t'])) / (np.sqrt(v_hat_b) + state['epsilon']))
    else:
        raise ValueError("Unsupported optimizer")
    return weights, biases, optimizer_state

# -------------------------------
# Training function using mini-batch gradient descent with optimizer flexibility and validation split
# -------------------------------
def train_network(X, Y, weights, biases, epochs, learning_rate, batch_size=32,
                  activation_type='sigmoid', weight_decay=0.0, optimizer='sgd', val_split=0.1, **optimizer_kwargs):

    m = X.shape[0]
    # Split data into training and validation sets
    split_index = int(m * (1 - val_split))
    X_train, X_val = X[:split_index], X[split_index:]
    Y_train, Y_val = Y[:split_index], Y[split_index:]

    m_train = X_train.shape[0]
    num_batches = int(np.ceil(m_train / batch_size))
    optimizer_state = initialize_optimizer_state(weights, biases, optimizer, **optimizer_kwargs)

    for epoch in range(epochs):
        total_loss = 0
        indices = np.arange(m_train)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        Y_train = Y_train[indices]

        for b in range(num_batches):
            start = b * batch_size
            end = min((b+1)*batch_size, m_train)
            batch_size_actual = end - start

            batch_grad_weights = [np.zeros_like(W) for W in weights]
            batch_grad_biases = [np.zeros_like(bias) for bias in biases]
            batch_loss = 0

            for i in range(start, end):
                x = X_train[i].reshape(-1, 1)   # (input_size, 1)
                y_true = Y_train[i].reshape(-1, 1)  # (output_size, 1)

                a_vals, h_vals = forward_propagation(x, weights, biases, activation_type)
                y_hat = h_vals[-1]
                loss = -np.sum(y_true * np.log(y_hat + 1e-8))
                batch_loss += loss

                grad_w, grad_b = backpropagation(a_vals, h_vals, weights, biases, x, y_true, activation_type)
                for j in range(len(weights)):
                    batch_grad_weights[j] += grad_w[j]
                    batch_grad_biases[j] += grad_b[j]

            # Compute averaged gradients over the mini-batch and add weight decay
            grad_w_avg_list = []
            grad_b_avg_list = []
            for j in range(len(weights)):
                grad_w_avg = batch_grad_weights[j] / batch_size_actual + weight_decay * weights[j]
                grad_b_avg = batch_grad_biases[j] / batch_size_actual
                grad_w_avg_list.append(grad_w_avg)
                grad_b_avg_list.append(grad_b_avg)
            weights, biases, optimizer_state = update_parameters(weights, biases, batch_grad_weights, batch_grad_biases, optimizer_state, optimizer, learning_rate)
            total_loss += batch_loss

        avg_train_loss = total_loss / m_train

        # Compute training accuracy
        train_correct = 0
        for i in range(m_train):
            x = X_train[i].reshape(-1, 1)
            _, h_vals = forward_propagation(x, weights, biases, activation_type)
            y_hat = h_vals[-1]
            if np.argmax(y_hat) == np.argmax(Y_train[i]):
                train_correct += 1
        train_accuracy = (train_correct / m_train) * 100

        # Evaluate on validation set
        val_loss = 0
        val_correct = 0
        m_val = X_val.shape[0]
        for i in range(m_val):
            x = X_val[i].reshape(-1, 1)
            _, h_vals = forward_propagation(x, weights, biases, activation_type)
            y_hat = h_vals[-1]
            loss = -np.sum(Y_val[i].reshape(-1, 1) * np.log(y_hat + 1e-8))
            val_loss += loss
            if np.argmax(y_hat) == np.argmax(Y_val[i]):
                val_correct += 1
        avg_val_loss = val_loss / m_val
        val_accuracy = (val_correct / m_val) * 100

        wandb.log({
          "train_loss": avg_train_loss,
          "train_accuracy": train_accuracy,
          "val_loss": avg_val_loss,
          "val_accuracy": val_accuracy
        })

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {avg_train_loss:.4f}, Train Acc = {train_accuracy:.2f}%, Val Loss = {avg_val_loss:.4f}, Val Acc = {val_accuracy:.2f}%")

    return weights, biases

# -------------------------------
# Run Experiment Function for W&B Sweep (Best Performing Configuration)
# -------------------------------
def run_experiment():
    # Instead of using cfg, we use our best performing configuration values directly.
    # Best performing values:
    best_num_epochs = 10
    best_learning_rate = 0.0001
    best_batch_size = 16
    best_hiddenlayers = 5
    best_hiddennodes = 128
    best_initializer = 'xavier'
    best_activation_func = 'relu'
    best_weight_decay = 0.5
    best_optimizer = 'adam'

    run = wandb.init()
    run.name = f"BestConfig: epochs {best_num_epochs} hidden_layers {best_hiddenlayers} hidden_size {best_hiddennodes} learning_rate {best_learning_rate} opt {best_optimizer} batch_size {best_batch_size} init {best_initializer} activation {best_activation_func} weight_decay {best_weight_decay}"

    # Load Fashion-MNIST data
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Preprocess: Flatten images and normalize pixel values to [0,1]
    x_train = x_train.reshape(x_train.shape[0], -1) / 255.0
    x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0
    y_train = to_categorical(y_train, 10)
    y_test  = to_categorical(y_test, 10)

    # Set network architecture
    input_size = 784
    num_classes = 10
    layer_sizes = [input_size] + [best_hiddennodes] * best_hiddenlayers + [num_classes]

    # Initialize parameters
    weights, biases = initialize_parameters(layer_sizes, weight_init=best_initializer.lower())

    # Train model (using 10% of training data as validation)
    weights, biases = train_network(x_train, y_train, weights, biases,
                                    epochs=best_num_epochs,
                                    learning_rate=best_learning_rate,
                                    batch_size=best_batch_size,
                                    activation_type=best_activation_func,
                                    weight_decay=best_weight_decay,
                                    optimizer=best_optimizer.lower(),
                                    **{'momentum': 0.9, 'decay_rate': 0.9, 'epsilon': 1e-8, 'beta1': 0.9, 'beta2': 0.999},
                                    val_split=0.1)

    # -------------------------------
    # Test Set Evaluation and Confusion Matrix
    # -------------------------------
    test_predictions = []
    test_labels = []
    m_test = x_test.shape[0]
    for i in range(m_test):
        x = x_test[i].reshape(-1, 1)
        _, h_vals = forward_propagation(x, weights, biases, activation_type=best_activation_func)
        y_hat = h_vals[-1]
        pred = np.argmax(y_hat)
        test_predictions.append(pred)
        true_label = np.argmax(y_test[i])
        test_labels.append(true_label)

    test_accuracy = np.mean(np.array(test_predictions) == np.array(test_labels)) * 100
    wandb.log({"final_test_accuracy": test_accuracy})
    print("Test Accuracy: {:.2f}%".format(test_accuracy))

    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[str(i) for i in range(num_classes)],
                yticklabels=[str(i) for i in range(num_classes)])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix on Fashion-MNIST Test Set")
    plt.show()

    run.finish()

# -------------------------------
# Run W&B Sweep Agent for the Best Configuration
# -------------------------------
wandb.agent("fmm03ezl", run_experiment, project="DA6401-Assignment1", count=1)


