import math, random
import numpy as np
from data import load_mnist, load_synth

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


def backward_pass(x, v, h, target, probabilities):
    """
    This function performs a full backward pass on our given neural network.
    It returns the values for the derivatives of w, v, bias1 and bias2.
    """

    #Dervative of Loss with regards to our target class
    derivative_of_loss = -1/(probabilities[target])

    # Multiplying the derivative of loss with the output to get the derivatives of the m nodes
    m_derivative = []
    for i in range(len(probabilities)):
        if(i is not target):
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
    
    return bias1_derivative, bias2_derivative, w_derivative, v_derivative



if __name__ == "__main__":

    # inputs
    x = [1, -1]

    # weights & biases
    bias1 = [0.0, 0.0, 0.0]
    bias2 = [0.0, 0.0]
    w = [[1., 1., 1.], [-1., -1., -1.]]
    v = [[1., 1.], [-1., -1.], [-1., -1.]]

    # target class
    target = 0

    loss, h, probabilities = forward_pass(x, bias1, bias2, w, v, target)

    bias1_derivative, bias2_derivative, w_derivative, v_derivative = backward_pass(x, v, h, target, probabilities)


    print(f'Derivatives of bias1: {bias1_derivative}')
    print(f'Derivatives of bias2: {bias2_derivative}')
    print(f'Derivatives of the w weights: {w_derivative}')
    print(f'Derivatives of the v weights: {v_derivative}')

    (xtrain, ytrain), (xval, yval), num_cls = load_synth()
