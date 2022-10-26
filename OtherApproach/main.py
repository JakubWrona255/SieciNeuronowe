import sys
import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

        self.inputs = inputs

    def backward(self,dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T,dvalues)
        self.dbiases = np.sum(dvalues,axis=0,keepdims=True)
        #Gradients on values
        self.dinputs = np.dot(dvalues,self.weights.T)


class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)

    def backward(self,dvalues):
        # Since we need to modify the original variable,
        # let's make a copy of the values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0




class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilities

class Loss:
    def calculate(self,output,y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped  = np.clip(y_pred,1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axix=1)
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

    def backward(self,dvalues,y_true):
        #number of samples
        samples = len(dvalues)
        #Number of labels in every sample
        #We'll use the first sample to count them
        labels = len(dvalues[0])

        #If labels are sparse, turn them into one-hot vector
        if(len(y_true.shape) == 1):
            y_true = np.eye(labels)[y_true]

        #Calculate gradient
        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs/samples

#Strona 215
if __name__ == '__main__':
    nnfs.init()
    X, y = spiral_data(samples=100,classes=3)
    dense1 = Layer_Dense(2,3)
    activation1 = Activation_ReLU()
    dense2 = Layer_Dense(3, 3)
    activation2 = Activation_Softmax()

    dense1.forward(X)
    activation1.forward(dense1.output)

    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    print(activation2.output[:5])

    loss_function = Loss_CategoricalCrossentropy()
    loss = loss_function.calculate(activation2.output,y)
    print("Loss:",loss)


    '''X = [[1, 2, 3, 2.5],
         [2.0,5.0,-1.0,2.0],
         [-1.5,2.7,3.3,-0.8]]

    inputs = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
    output = []
    for i in inputs:
        output.append(max(0,i))
    print(output)
    '''
'''
    layer1 = Layer_Dense(4,5)
    layer2 = Layer_Dense(5,2)
    layer1.forward(X)
    #print(layer1.output)
    layer2.forward(layer1.output)
    print(layer2.output)

'''


