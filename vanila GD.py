{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AdAhchu9GC6L"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from keras.datasets import fashion_mnist\n",
        "from keras.utils import to_categorical\n",
        "(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()\n",
        "# Select only 1000 training examples\n",
        "#x_train = x_train[:1000]\n",
        "#y_train = y_train[:1000]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 81,
      "metadata": {
        "id": "ZLvzc1ECSqBV"
      },
      "outputs": [],
      "source": [
        "# -------------------------------\n",
        "# Utility functions: activations\n",
        "# -------------------------------\n",
        "\n",
        "def sigmoid(z):\n",
        "    \"\"\"Logistic (sigmoid) activation function.\"\"\"\n",
        "    return 1 / (1 + np.exp(-z))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 82,
      "metadata": {
        "id": "q8J8vJwMSsii"
      },
      "outputs": [],
      "source": [
        "def sigmoid_derivative_from_activation(a):\n",
        "    \"\"\"\n",
        "    Derivative of sigmoid given its output 'a' (since a = sigmoid(z)).\n",
        "    That is: a * (1 - a)\n",
        "    \"\"\"\n",
        "    return a * (1 - a)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 83,
      "metadata": {
        "id": "kF31iZmeSwOd"
      },
      "outputs": [],
      "source": [
        "def softmax(z):\n",
        "    \"\"\"Softmax activation for output layer.\"\"\"\n",
        "    # subtract max for numerical stability\n",
        "    z = z - np.max(z)\n",
        "    exp_z = np.exp(z)\n",
        "    return exp_z / np.sum(exp_z)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 84,
      "metadata": {
        "collapsed": true,
        "id": "f41mkSq2S7xr"
      },
      "outputs": [],
      "source": [
        "# -------------------------------\n",
        "# Forward propagation function\n",
        "# -------------------------------\n",
        "def forward_propagation(x, weights, biases):\n",
        "\n",
        "    a_values = []    #list of preactivation values of each layer\n",
        "    h_values = [x]  # list of activation values of each layer including hidden layer\n",
        "    L = len(weights)  # total number of layers (hidden + output)\n",
        "    for i in range(L):\n",
        "        a = np.dot(weights[i], h_values[-1]) + biases[i]     # result is a column vector. # h_values[-1] is activation output (vector) of the previous layer (most recently calculated).h_values[-1] dynamically points to the last element in the list\n",
        "        a_values.append(a)\n",
        "        if i == L - 1:\n",
        "            # For output layer, use softmax to get probability distribution.\n",
        "            h = softmax(a)\n",
        "        else:\n",
        "            # For hidden layers, use sigmoid activation.\n",
        "            h = sigmoid(a)\n",
        "        h_values.append(h)\n",
        "    return a_values, h_values"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "wZC6Mdx3TdMD"
      },
      "source": [
        "my own backpropagation code"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 85,
      "metadata": {
        "id": "ykwl59rFoDOH"
      },
      "outputs": [],
      "source": [
        "def backpropagation(a_values, h_values, weights, biases, x, y_true):\n",
        "\n",
        "    L = len(weights)  # number of layers (hidden + output)\n",
        "    h_values = h_values[1:]\n",
        "    # Initialize gradient lists with same shapes as a_values and h_values\n",
        "    grad_weights = [np.zeros_like(W) for W in weights]\n",
        "    grad_biases  = [np.zeros_like(b) for b in biases]\n",
        "    grad_a_values = [np.zeros_like(a) for a in a_values]  # gradient w.r.t preactivation (a)\n",
        "    grad_h_values = [np.zeros_like(h) for h in h_values]   # gradient w.r.t activation (h)\n",
        "\n",
        "    # ---------- Output Layer Gradient ----------\n",
        "    # For output layer, using cross-entropy with softmax, the gradient w.r.t preactivation is:\n",
        "    # δ_L = h_values[-1] - y_true\n",
        "    grad_a_values[L-1] = h_values[-1] - y_true\n",
        "    #grad_h_values[L-1] = h_values[-1]  # (this value is not used further)\n",
        "\n",
        "    # ---------- Backpropagate for Hidden Layers ----------\n",
        "    # Loop from the output layer (index L-1) backward to the first layer (index 0)\n",
        "    for i in range(L-1, -1, -1):\n",
        "        if i == 0:\n",
        "            grad_weights[i] = np.dot(grad_a_values[i], x.T)\n",
        "\n",
        "        else:\n",
        "            grad_weights[i] = np.dot(grad_a_values[i], h_values[i-1].T)\n",
        "        grad_biases[i]  = grad_a_values[i]\n",
        "        if i > 0:\n",
        "            grad_h_values[i-1] = np.dot(weights[i].T, grad_a_values[i])\n",
        "            # For the hidden layers (i-1 >= 1), apply the derivative of the sigmoid.\n",
        "            # For the input layer (i-1 == 0), no activation derivative is applied.\n",
        "            grad_a_values[i-1] = grad_h_values[i-1] * sigmoid_derivative_from_activation(h_values[i-1])\n",
        "\n",
        "    return grad_weights, grad_biases\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EZr01hA3TGVC"
      },
      "outputs": [],
      "source": [
        "#-----------------------------\n",
        "# Training function using full batch gradient descent\n",
        "# -------------------------------\n",
        "def vanila(X, Y, weights, biases, epochs, learning_rate):\n",
        "\n",
        "    m = X.shape[0]  # number of training examples\n",
        "\n",
        "    for epoch in range(epochs):\n",
        "        # Initialize the gradients and loss\n",
        "        sum_grad_weights = [np.zeros_like(W) for W in weights]   #creates a list with each element is a zero matrix with the same size as the corresponding weight matrix in weights. No. of layers X size of layer(weight matrix of each layer)\n",
        "        sum_grad_biases  = [np.zeros_like(b) for b in biases]    #creates a list of zero vectors that matches the corresponding bias vector # size is No. of layers X No. of neuron in a layer\n",
        "        total_loss = 0\n",
        "        # Process each training example\n",
        "        for i in range(m):\n",
        "            # Get the i-th training example and reshape as a column vector.\n",
        "            x = X[i].reshape(-1, 1)  # Reshapes the selected input sample from (784,) to (784, 1). its a 2d column vector of 784X1\n",
        "            y_true = Y[i].reshape(-1, 1)  # Reshapes the selected input sample from (10,) to (10, 1). its a 2d column vector of 10X1\n",
        "\n",
        "            # Forward propagation\n",
        "            a_vals, h_vals = forward_propagation(x, weights, biases)\n",
        "\n",
        "            # Compute loss for this example (cross-entropy)\n",
        "            y_hat = h_vals[-1]    # fetches activation value at output layer\n",
        "            # Add a small epsilon to avoid log(0)\n",
        "            loss = -np.sum(y_true * np.log(y_hat + 1e-8))\n",
        "            total_loss += loss\n",
        "\n",
        "            # Backpropagation to compute gradients for this example\n",
        "            grad_w, grad_b = backpropagation(a_vals, h_vals, weights,biases,x, y_true)\n",
        "\n",
        "            # Accumulate gradients\n",
        "            for j in range(len(weights)):\n",
        "                sum_grad_weights[j] += grad_w[j]\n",
        "                sum_grad_biases[j]  += grad_b[j]\n",
        "\n",
        "        # Average gradients over all examples\n",
        "        avg_grad_weights = [g / m for g in sum_grad_weights]\n",
        "        avg_grad_biases  = [g / m for g in sum_grad_biases]\n",
        "\n",
        "        # Update parameters using gradient descent rule\n",
        "        for j in range(len(weights)):\n",
        "            weights[j] -= learning_rate * avg_grad_weights[j]\n",
        "            biases[j]  -= learning_rate * avg_grad_biases[j]\n",
        "\n",
        "        # Compute average loss for the epoch\n",
        "        avg_loss = total_loss / m\n",
        "\n",
        "        # Calculate training accuracy for monitoring\n",
        "        correct = 0\n",
        "        for i in range(m):\n",
        "            x = X[i].reshape(-1, 1)\n",
        "            _, h_vals = forward_propagation(x, weights, biases)\n",
        "            y_hat = h_vals[-1]\n",
        "            if np.argmax(y_hat) == np.argmax(Y[i]):\n",
        "                correct += 1\n",
        "        accuracy = correct / m * 100\n",
        "        print(f\"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.2f}%\")\n",
        "\n",
        "    return weights, biases"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 87,
      "metadata": {
        "collapsed": true,
        "id": "9a8D4VyprLRq"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from keras.utils import to_categorical\n",
        "# Main code:\n",
        "\n",
        "# Preprocess: Flatten images and normalize pixel values to [0,1]\n",
        "x_train = x_train.reshape(x_train.shape[0], -1) / 255.0  # reshape: (m, 784)  #x_train.shape[0] gets no. of training example\n",
        "x_test  = x_test.reshape(x_test.shape[0], -1) / 255.0     # earlier shape mx28x28 after reshape mx784\n",
        "\n",
        "# One-hot encode labels (10 classes)\n",
        "y_train = to_categorical(y_train, 10)      #eg: any train ex class is 9 then y_train_oh is 0 0 0 0 0 0 0 0 0 1, y_train is mX10 size\n",
        "y_test  = to_categorical(y_test, 10)\n",
        "\n",
        "# Set hyperparameters\n",
        "epochs = 50\n",
        "learning_rate = 0.001\n",
        "total_hidden_layers= 5\n",
        "hidden_layer_size = 128\n",
        "num_layers= total_hidden_layers+1\n",
        "\n",
        "# Define network architecture. # Example: one hidden layer with 128 nodes. You can add more layers.\n",
        "input_size = 784      #input layer feature vector size\n",
        "output_size = 10       #output layer feature vector size\n",
        "layer_sizes = [input_size] + [hidden_layer_size] * total_hidden_layers + [output_size]\n",
        "\n",
        "\n",
        "\n",
        "# Initialize weights and biases for each layer\n",
        "weights = []\n",
        "biases = []\n",
        "\n",
        "for i in range(num_layers):\n",
        "    W = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * 0.01  #  Weights are initialized with small random values for stable training. W[0] size is 32X784\n",
        "    b = np.zeros((layer_sizes[i+1], 1))                            # Biases initialized to zero, b[0] size is 32X1\n",
        "    weights.append(W)\n",
        "    biases.append(b)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "collapsed": true,
        "id": "7AU_WPb5to-X",
        "outputId": "0b74a148-bcbf-494c-9dc0-8d45eb0d3c58"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/50: Loss = 2.3047, Accuracy = 10.70%\n",
            "Epoch 2/50: Loss = 2.3046, Accuracy = 10.70%\n",
            "Epoch 3/50: Loss = 2.3046, Accuracy = 10.70%\n",
            "Epoch 4/50: Loss = 2.3046, Accuracy = 10.70%\n",
            "Epoch 5/50: Loss = 2.3045, Accuracy = 10.70%\n",
            "Epoch 6/50: Loss = 2.3045, Accuracy = 10.70%\n",
            "Epoch 7/50: Loss = 2.3045, Accuracy = 10.70%\n",
            "Epoch 8/50: Loss = 2.3044, Accuracy = 10.70%\n",
            "Epoch 9/50: Loss = 2.3044, Accuracy = 10.70%\n",
            "Epoch 10/50: Loss = 2.3044, Accuracy = 10.70%\n",
            "Epoch 11/50: Loss = 2.3043, Accuracy = 10.70%\n",
            "Epoch 12/50: Loss = 2.3043, Accuracy = 10.70%\n",
            "Epoch 13/50: Loss = 2.3043, Accuracy = 10.70%\n",
            "Epoch 14/50: Loss = 2.3042, Accuracy = 10.70%\n",
            "Epoch 15/50: Loss = 2.3042, Accuracy = 10.70%\n",
            "Epoch 16/50: Loss = 2.3042, Accuracy = 10.70%\n",
            "Epoch 17/50: Loss = 2.3042, Accuracy = 10.70%\n",
            "Epoch 18/50: Loss = 2.3041, Accuracy = 10.70%\n",
            "Epoch 19/50: Loss = 2.3041, Accuracy = 10.70%\n",
            "Epoch 20/50: Loss = 2.3041, Accuracy = 10.70%\n",
            "Epoch 21/50: Loss = 2.3040, Accuracy = 10.70%\n",
            "Epoch 22/50: Loss = 2.3040, Accuracy = 10.70%\n",
            "Epoch 23/50: Loss = 2.3040, Accuracy = 10.70%\n",
            "Epoch 24/50: Loss = 2.3039, Accuracy = 10.70%\n",
            "Epoch 25/50: Loss = 2.3039, Accuracy = 10.70%\n",
            "Epoch 26/50: Loss = 2.3039, Accuracy = 10.70%\n",
            "Epoch 27/50: Loss = 2.3039, Accuracy = 10.70%\n",
            "Epoch 28/50: Loss = 2.3038, Accuracy = 10.70%\n",
            "Epoch 29/50: Loss = 2.3038, Accuracy = 10.70%\n",
            "Epoch 30/50: Loss = 2.3038, Accuracy = 10.70%\n",
            "Epoch 31/50: Loss = 2.3038, Accuracy = 10.70%\n",
            "Epoch 32/50: Loss = 2.3037, Accuracy = 10.70%\n",
            "Epoch 33/50: Loss = 2.3037, Accuracy = 10.70%\n",
            "Epoch 34/50: Loss = 2.3037, Accuracy = 10.70%\n",
            "Epoch 35/50: Loss = 2.3036, Accuracy = 10.70%\n",
            "Epoch 36/50: Loss = 2.3036, Accuracy = 10.70%\n",
            "Epoch 37/50: Loss = 2.3036, Accuracy = 10.70%\n",
            "Epoch 38/50: Loss = 2.3036, Accuracy = 10.70%\n",
            "Epoch 39/50: Loss = 2.3035, Accuracy = 10.70%\n",
            "Epoch 40/50: Loss = 2.3035, Accuracy = 10.70%\n",
            "Epoch 41/50: Loss = 2.3035, Accuracy = 10.70%\n",
            "Epoch 42/50: Loss = 2.3035, Accuracy = 10.70%\n",
            "Epoch 43/50: Loss = 2.3034, Accuracy = 10.70%\n",
            "Epoch 44/50: Loss = 2.3034, Accuracy = 10.70%\n",
            "Epoch 45/50: Loss = 2.3034, Accuracy = 10.70%\n",
            "Epoch 46/50: Loss = 2.3034, Accuracy = 10.70%\n",
            "Epoch 47/50: Loss = 2.3033, Accuracy = 10.70%\n",
            "Epoch 48/50: Loss = 2.3033, Accuracy = 10.70%\n",
            "Epoch 49/50: Loss = 2.3033, Accuracy = 10.70%\n",
            "Epoch 50/50: Loss = 2.3033, Accuracy = 10.70%\n",
            "Test Accuracy: 10.00%\n"
          ]
        }
      ],
      "source": [
        "# Train the network using full batch gradient descent\n",
        "weights, biases = vanila(x_train, y_train, weights, biases, epochs, learning_rate)\n",
        "\n",
        "# -------------------------------\n",
        "# Evaluate on test set\n",
        "# -------------------------------\n",
        "correct = 0\n",
        "m_test = x_test.shape[0]\n",
        "for i in range(m_test):\n",
        "    x = x_test[i].reshape(-1, 1)\n",
        "    _, h_vals = forward_propagation(x, weights, biases)\n",
        "    y_hat = h_vals[-1]\n",
        "    if np.argmax(y_hat) == np.argmax(y_test[i]):\n",
        "        correct += 1\n",
        "test_accuracy = correct / m_test * 100\n",
        "print(\"Test Accuracy: {:.2f}%\".format(test_accuracy))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 88,
      "metadata": {
        "id": "7Ad22WfF9Hea"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
