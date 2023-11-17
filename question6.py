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


def get_derivative_of_m(y_input, probs, d_loss):
    """
    Method that calculates the derivative of 'm' based on the given inputs
    """
    m_derivative = []
    indices = np.nonzero(y_input == 1)[1]

    for i in range(len(probs)):
        if i == indices[1]:
            m_derivative.append(probs[i]-1)
        else:
            m_derivative.append(- d_loss * probs[i] * probs[indices][i])

    m_derivative = np.array(m_derivative)
    return m_derivative


def forward_pass(x_input, y_input, w, bias1, v, bias2, batch_size):
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
    # perform reshaping in order to transforms m into a new shape of (300, 10, 28)
    m = m.reshape(300,10,28)
    
    probs = (np.exp(m) / np.sum(np.exp(m)))
    
    # calculate loss in relation to 'True' class
    loss = (-np.sum(y_input * np.log(np.mean(probs.reshape(300,28,10),axis=1))))/batch_size

    return loss, probs, h, bias1, bias2, w, v


def backward_pass(x_input, y_input, probs, h, bias1, bias2, w, v, learning_rate, batch_size):
    """
    This method performs a backward pass through the neural network to compute the gradients and
    returns them.
    """

    # der. of Loss w.r.t. to 'True' class
    d_loss = (-np.sum(y_input*(1/(np.mean(probs.reshape(300,28,10),axis=1)+(1e-6)))))/batch_size

    # calculate m
    m_derivative = get_derivative_of_m(y_input, probs, d_loss)
    bias2_derivative = np.mean(m_derivative, axis=0)
    
    # derivatives for v , m and h
    v_derivative = np.mean(np.matmul(m_derivative, h), axis=0)
    m_derivative = m_derivative.reshape(300,28,10)
    h_derivative = np.matmul(m_derivative, v)

    # Derivative of Loss based on k (applying sigmoid derivative)
    k_derivative = h_derivative * h * (1-h)

    bias1_derivative = np.mean(k_derivative, axis=0)

    # derivative of w
    w_derivative = np.mean(np.matmul(x_input, k_derivative), axis=0)
    
    return w_derivative, bias1_derivative, v_derivative, bias2_derivative



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

    # data normalization on each data point in x_train.
    # dividing by 255 to scale the pixel values between 0 and 1
    normalized_training_data = []
    for i in range(len(x_train)):
        normalized_training_data.append(np.array(x_train[i]/255))
    normalized_training_data = np.array(normalized_training_data)


    # the weights for layer 1 and layer 2
    w = np.random.randn(28, hidden_layer_nodes)
    v = np.random.randn(output_layer_nodes, hidden_layer_nodes)
    
    bias1 = np.zeros(hidden_layer_nodes)
    bias2 = np.zeros(output_layer_nodes)

    learning_rate = 0.01
    
    # get the one hot encoding for target classes
    y_train = get_one_hot_encoding(t_train)

    train_loss_list = []
    train_loss_temp = []

    # samples per batch
    batch_size = 300


    for i in tqdm(range(0,len(normalized_training_data), batch_size),position=0, leave=True):

        # # Split data in batches
        # x_train_batches = []
        # y_train_batches = []

        # for i in range(0, len(normalized_training_data), batch_size):
        #     x_batch = normalized_training_data[i: i + batch_size]
        #     y_batch = y_train[i: i + batch_size]
            
        #     x_train_batches.append(x_batch)
        #     y_train_batches.append(y_batch)
        try:
            # Split data in batches
            x_train_batches = normalized_training_data[i: i+ batch_size]
            x_train_batches = np.array(x_train_batches).reshape(batch_size,28,28)
            y_train_batches = y_train[i: i+ batch_size]
        except Exception as e:
            print(f"Error occurred: {e}")
            break  # End the loop if an exception occurs

        # perform forward and backward pass
        loss, probs, h, bias_1, bias_2, w, v = forward_pass(x_train_batches, y_train_batches, w, bias1, v, bias2, batch_size)
        w_derivative, bias1_derivative, v_derivative, bias2_derivative = backward_pass(x_train_batches, y_train_batches, probs, h, bias1, bias2, w, v, learning_rate, batch_size)


        # we update the weights
        w = w - learning_rate * (w_derivative)
        v = v - learning_rate * (v_derivative)
        bias2 = bias2 - (learning_rate * (bias2_derivative.T))
        bias1 = bias1 - (learning_rate * bias1_derivative)
    
        train_loss_list.append(probs)

    # Loss Visualization
    visual(train_loss_list, "Training")