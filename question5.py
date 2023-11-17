import numpy as np
from urllib import request
import gzip
import pickle
import os

import math
import matplotlib.pyplot as plt
from numpy import random
from tqdm import tqdm
import random
random.seed(0)

from data import load_mnist


def get_one_hot_encoding(y):
    """
    This method is used to encode class labels into one-hot vectors
    """

    # initial matrix with only zeros
    one_hot_Y = np.zeros((y.size, y.max() + 1))

    # we set the necessary elements to 1 based on the indices from the y
    one_hot_Y[np.arange(y.size), y] = 1

    return one_hot_Y


def forward_pass(x_input, y_input, w, bias1, v, bias2):
    """
    This method performs a forward pass in the neural network, calculates
    the loss and gives a prediction based on the input that we give to the method
    """
    # for the first layer we multiply input x to weights w to get k
    k = np.matmul(x_input, w)

    # for the second layer we need to apply the Sigmoid Activation to k to get h
    h = 1 / (1 + np.exp(-k))
        
    # we multiply h to weights v to get m
    m = np.matmul(h, v.T)
    
    probs = (np.exp(m) / np.sum(np.exp(m)))
    
    # calculate loss in relation to 'True' class
    loss = - np.sum(y_input * np.log(probs))

    return loss, probs, np.matrix(h), bias1, bias2, w, v


def backward_pass(x_input, y_input, probs, h, bias1, bias2, w, v, learning_rate):
    """
    This method performs a backward pass through the neural network to compute the gradients and
    returns them.
    """

    # der. of Loss w.r.t. to 'True' class
    d_loss = -np.sum(y_input * (1 / probs))

    # calculate m
    m_derivative = []
    for i in range(len(probs)):
        if y_input[i] != 0:
            m_derivative.append(probs[i] - 1)
        else:
            m_derivative.append(-d_loss * probs[i] * probs[np.where(y_input == 1)[0][0]])

    m_derivative = np.matrix(m_derivative)
    bias2_derivative = m_derivative
    
    # derivatives for v and h
    v_derivative = np.matmul(m_derivative.T, h)
    h_derivative = np.matmul(m_derivative, v)
    
    k_derivative = np.matmul(np.matmul(h_derivative, h.T), 1 - h)

    bias1_derivative = k_derivative

    # derivatives of w
    x_input_matrix = np.matrix(x_input)
    w_derivative = np.matmul(x_input_matrix.T, k_derivative)
    
    return np.array(w_derivative), bias1_derivative, np.array(v_derivative), bias2_derivative



def visual(loss_list, l):
    # Create count of the number of epochs
    epoch_count = range(len(loss_list))

    # Create a figure and axis object
    plt.figure(figsize=(8, 6))  # Adjust the figure size as needed

    # Visualize loss history with a smoother line, and add marker points
    plt.plot(epoch_count, loss_list, color='blue', linestyle='-', marker='o', markersize=4, label=l + ' Loss')

    # Set plot title and labels
    plt.title('Loss Over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)

    # Add legend and grid
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True)

    # Save the plot to a file (create the 'Results' folder if it doesn't exist)
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    plt.tight_layout()  # Adjust layout for better spacing
    plt.savefig("./Results/" + l + "_Loss_Q5.png")
    plt.show()  # Display the plot (optional - remove if saving only)


if __name__ == "__main__":

    (x_train, t_train), (x_test, t_test), num_cls = load_mnist()
    
    # network params
    hidden_layer_nodes = 300
    output_layer_nodes = 10

    # the weights for layer 1 and layer 2
    w = np.random.randn(int(len(x_train[0])), hidden_layer_nodes)
    v = np.random.randn(output_layer_nodes, hidden_layer_nodes)
    
    bias1 = np.zeros(hidden_layer_nodes)
    bias2 = np.zeros(output_layer_nodes)

    learning_rate = 0.01
    
    # get the one hot encoding for target classes
    y_train = get_one_hot_encoding(t_train)

    train_loss_list = []
    train_loss_temp = []

    # samples per batch
    batch_size = 100

    for epoch in tqdm(range(15), position=0, leave=True):
        for i in tqdm(range(0, len(x_train), batch_size), position=0, leave=True):
            
            # get data in batches
            x_batch = x_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]

            # initialize gradient values
            w_grad = 0
            v_grad = 0 
            bias1_grad = 0 
            bias2_grad = 0 

            for j in range(len(x_batch)):

                # normalize the data
                x = np.array(x_batch[j] / 255)

                # perform forward and backward pass
                loss, probs, h, bias_1, bias_2, w, v = forward_pass(x, y_batch[j], w, bias1, v, bias2)
                w_derivative, bias1_derivative, v_derivative, bias2_derivative = backward_pass(x, y_batch[j], probs, h, bias1, bias2, w, v, learning_rate)

                # add the gradient results from backward pass
                w_grad += w_derivative
                v_grad += v_derivative
                bias1_grad += bias1_derivative
                bias2_grad += bias2_derivative 

                train_loss_temp.append(loss)

            # we get the average of the gradients and update the weights
            w = w - learning_rate * (w_grad / batch_size)
            v = v - learning_rate * (v_grad / batch_size)
            bias1 = bias1 - learning_rate * (bias1_grad / batch_size)
            bias2 = bias2 - learning_rate * (bias2_grad / batch_size)
        
            # Compute and store average loss for the epoch
            train_loss_list.append(np.mean(train_loss_temp))

    # Loss Visualization
    visual(train_loss_list, "Training")