# Backpropagation and Experiment Tracking with Wandb.ai  

## Overview  
This project involves implementing a **feedforward neural network** from scratch using **backpropagation** and **gradient descent** (along with its variants) for a classification task. The model will be trained on the **Fashion-MNIST dataset**, which consists of 28x28 grayscale images categorized into 10 different classes.  

Additionally, we will use **Weights & Biases (Wandb.ai)** to log, visualize, and track experiments effectively.  

## Objectives  
- **Implement Backpropagation:** Develop a neural network and implement the backpropagation algorithm using **Python (NumPy, Pandas)**.  
- **Optimize with Gradient Descent:** Train the network using **gradient descent** and its **variants** (SGD, Momentum, Adam).  
- **Track Experiments:** Use **Wandb.ai** to log results, visualize training performance, and generate a report.      

## Dataset  
- The **Fashion-MNIST dataset** consists of **28x28 pixel grayscale images** belonging to **10 different classes** (e.g., shirts, shoes, bags).  
- The goal is to train a neural network to classify images into these categories.  

## Getting Started  
### 1. Clone the Repository  
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
pip install numpy pandas wandb
import wandb
wandb.init(project="backpropagation-fashion-mnist")
python train.py
```
## Results
