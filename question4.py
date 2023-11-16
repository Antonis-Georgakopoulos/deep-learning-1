import math, random
import numpy as np
from data import load_mnist, load_synth


#extras
from urllib import request
import gzip
import pickle
import os

import matplotlib.pyplot as plt
random.seed(0)


def apply_sigmoid_function(k):
    """
    Given a list of values, apply a sigmoid function to all the values and return the new list
    """
    
    # the updated list that we will return
    h = []

    for i in range(len(k)):
        h.append(1/(1+math.exp(-k[i])))
    
    return h


def apply_softmax_function(m):
    """
    Calculate the softmax activation function based on the list of values given as parameters.
    Return the list of probabilities 
    """
    probabilities = []

    denominator = 0

    # find denominator by iterating through all values of m
    for i in range(len(m)):
        denominator += math.exp(m[i])

    for i in range(len(m)):
        numerator = math.exp(m[i])
        probabilities.append(numerator/denominator)
    
    return probabilities


def forward_pass(x, bias1, bias2, w, v, target):
    """
    This function performs a full forward pass on the given neural network
    and returns the loss value based on the prediction of the network and the actual target class,
    the values of the sigmoid layer and the probabilities after we apply the softmax function. 
    """

    # initialization of k layer values, initially we start with everything assigned to 0
    k = [0] * len(w[0])

    for j in range(len(w[0])):
        for i in range(len(x)):
            k[j] += w[i][j]*x[i]
        k[j] += bias1[j]


    # applying sigmoid to the k layer
    h = apply_sigmoid_function(k)

    # the next layer after applying the sigmoid function
    m = [0,0]

    for j in range(len(v[0])):
        for i in range(len(h)):
            m[j] += v[i][j]*h[i]
        m[j] += bias2[j] 

    # apply softmax function
    probabilities = apply_softmax_function(m)
    
    # calculate the loss based on the actual target class
    loss = -math.log(probabilities[target])

    return loss, h, probabilities


def update_values(v, w, bias1_derivative, bias2_derivative, w_derivative, v_derivative, learning_rate, bias1, bias2):
    """
    Extend the functionality of backpropagation in order to update the weights and the values according to the learning rate that we have.
    """

    # updating weights of second layer 
    v_updated_weights = []
    for i in range(len(v)):
        values = []
        for j in range(len(v[0])):
            values.append(v[i][j] - (learning_rate*v_derivative[i][j]))
        v_updated_weights.append(values)
    

    # updating weights of first layer
    w_updated_weights = []
    for i in range(len(w)):
        values = []
        for j in range(len(w[0])):
            values.append(w[i][j] - (learning_rate*w_derivative[i][j]))
        w_updated_weights.append(values)


    # updating values of the bias2                  
    bias2_updated_values = []
    for i in range(len(bias2)):
        bias2_updated_values.append(bias2[i] - (learning_rate*bias2_derivative[i]))


    # updating values of the bias1  
    bias1_updated_values = []
    for i in range(len(bias1)):
        bias1_updated_values.append(bias1[i] - (learning_rate*bias1_derivative[i]))


    return v_updated_weights, w_updated_weights, bias2_updated_values, bias1_updated_values


def backward_pass(x, v, w, h, target, probabilities, learning_rate, bias1, bias2):
    """
    This function performs a full backward pass on our given neural network.
    It returns the values for the derivatives of w, v, bias1 and bias2.
    """

    #Dervative of Loss with regards to our target class
    derivative_of_loss = -1/(probabilities[target])

    # Multiplying the derivative of loss with the output to get the derivatives of the m nodes
    m_derivative = []
    for i in range(len(probabilities)):
        if(i == target):
            m_derivative.append(derivative_of_loss * probabilities[i]*(1-probabilities[i]))
        else:
            m_derivative.append(- derivative_of_loss * probabilities[i]*probabilities[target])

    # As we've written in the report, the derivative of loss with regards to bias2 is equal to
    # the derivative of loss with regards to m.
    bias2_derivative = m_derivative

    v_derivative = []
    h_derivative = []

    for i in range(len(h)):
        derivatives = []
        for j in range(len(m_derivative)):
            derivatives.append(m_derivative[j] * h[i])
        v_derivative.append(derivatives)

        h_derivative.append(m_derivative[0] * v[i][0] + m_derivative[1] * v[i][1])

    # for the first hidden layer
    k_derivative = []

    for i in range(len(h)):
        k_derivative.append(h_derivative[i]*h[i]*(1-h[i]))

    # For the first layer's weights
    w_derivative =[]
    for i in range(2):
        derivatives = []
        for j in range(3):
            derivatives.append(k_derivative[j] * x[i])

        w_derivative.append(derivatives)

    # Give the bias derivative the same value as the k derivative as seen in the report
    bias1_derivative = k_derivative

    v_updated_weights, w_updated_weights, bias2_updated_values, bias1_updated_values = update_values(v, w, bias1_derivative, bias2_derivative, w_derivative, v_derivative, learning_rate, bias1, bias2)
    
    return bias1_updated_values, bias2_updated_values, w_updated_weights, v_updated_weights


def visual(loss_list,l):
    
    # Folder to save results
    if not os.path.exists("./Results"):
        os.makedirs("./Results")
    
    # Create count of the number of epochs
    epoch_count = range(len(loss_list))

    # Visualize loss history
    plt.plot(epoch_count, loss_list, 'b')
    
    plt.legend([ l + 'Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig("./Results/"+l+"_Loss_Q4.png")
    plt.close()



if __name__ == "__main__":

    # target class
    target = 0

    learning_rate = 0.01

    (xtrain, ytrain), (xval, yval), num_cls = load_synth()

    # Generate a list of lists with normally distributed random values
    # for the w weights and the v weights respectively
    w = [[random.uniform(-1, 1) for _ in range(3)] for _ in range(2)]
    v = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(3)]

    # biases
    bias1 = [0.0, 0.0, 0.0]
    bias2 = [0.0, 0.0]

    train_loss_list = []
    val_loss_list = []
    y_pred_list = []

    for epoch in range(3):
        for i in range(len(xtrain)):

            loss, h, probabilities  = forward_pass(xtrain[i], bias1, bias2, w, v, ytrain[i])
            bias1, bias2, w, v = backward_pass(xtrain[i], v, w, h, ytrain[i] ,probabilities, learning_rate, bias1, bias2)

            train_loss_list.append(loss)
            y_pred_list.append(probabilities)

        for i in range(len(xval)):
            loss, temp1, temp2  = forward_pass(xval[i], bias1, bias2, w, v, yval[i])
            val_loss_list.append(loss)

    # Loss Visualization
    visual(train_loss_list, "Training")
    # Loss Visualization
    visual(val_loss_list, 'Val')