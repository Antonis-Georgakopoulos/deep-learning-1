import math, random
import numpy as np

def apply_sigmoid_function(k):
    """
    Given a list of values, apply a sigmoid function to all the values and return the new list
    """
    
    # the updated list that we will return
    h = []

    for i in range(len(k)):
        h.append(1/(1+math.exp(-k[i])))
    
    return h


def calculate_softmax(m):
    """
    Calculate the softmax activation function based on the list of values given as parameters.
    Return the list of probabilities 
    """
    probabilities = []

    denominator = np.sum(np.exp(m))
    for i in range(len(m)):
        numerator = math.exp(m[i])
        probabilities.append(numerator/denominator)
    
    return probabilities


def forward_pass(x, bias1, bias2, w, v):
    """
    docstring, define what it does and give param names
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


    probabilities = calculate_softmax(m)

    return probabilities



if __name__ == "__main__":

    # inputs
    x = [1, -1]

    # weights & biases
    bias1 = [0.0, 0.0, 0.0]
    bias2 = [0.0, 0.0]
    w = [[1., 1., 1.], [-1., -1., -1.]]
    v = [[1., 1.], [-1., -1.], [-1., -1.]]

    probabilities = forward_pass(x, bias1, bias2, w, v)
    print(probabilities)