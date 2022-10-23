import numpy as np


def sigmoid(arg):
    return 1/(1+np.exp(-arg))


def sigmoidDerivative(arg):
    temp = sigmoid(arg)
    return temp*(1-temp)
