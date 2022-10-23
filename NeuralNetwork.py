import AcrivationFunctions as af
import numpy as np
import handlingFiles as handleFiles
import csv as csv


class Layer:

    def __init__(self, numOfNeurons, numOfNeuronsInPreviousLayer,num):
        self.layerNumber = num
        self.numOfNeurons = numOfNeurons
        self.numOfNeuronsInPreviousLayer = numOfNeuronsInPreviousLayer

        self.biases = np.zeros(numOfNeurons)
        self.activationValues = np.zeros_like(self.biases)
        self.nodeSums = np.zeros_like(self.biases)

        self.weights = np.random.uniform(low=-0.1, high=0.1, size=(numOfNeurons, numOfNeuronsInPreviousLayer))

        self.weightsGradient = np.zeros_like(self.weights)
        self.biasesGradient = np.zeros_like(self.biases)

    def printLayer(self):
        print('layer',self.layerNumber,' activation val: ',self.activationValues)
        print('layer',self.layerNumber,'biases: ', self.biases)
        print('layer',self.layerNumber,'weights: ', self.weights, '\n')


class Network:

    learningRate = 0.05
    costC = 0

    def __init__(self,numOfHiddenLayers,numOfNeuronsInLayers):

        self.numOfLayers = numOfHiddenLayers+2
        self.numOfHiddenLayers = numOfHiddenLayers
        self.Layers = []
        self.Layers.append(Layer(numOfNeuronsInLayers[0], 0,0))  # append input layer without connections

        for i in range(1,self.numOfLayers):
            self.Layers.append(Layer(numOfNeuronsInLayers[i],numOfNeuronsInLayers[i-1],i))

        self.desiredOutput = np.zeros(shape=self.Layers[self.numOfLayers-1].numOfNeurons)

    def feedForward(self):
        for i in range(1,self.numOfLayers):  # iterate through layers, start from 1 and not 0 - 0 is the input layer, no calculations take place here
            self.Layers[i].nodeSums = np.matmul(self.Layers[i-1].activationValues, self.Layers[i].weights.T) + self.Layers[i].biases
            self.Layers[i].activationValues = af.sigmoid(self.Layers[i].nodeSums)


    def propagateBackwards(self):
        outputLayer = self.numOfLayers-1
        propagatedCostC = 0
        #calculate error
        self.costC = np.array(self.Layers[outputLayer].activationValues - self.desiredOutput)

        #adjust weights of outputLayer
        for i in range(self.Layers[outputLayer].numOfNeurons):

            self.Layers[outputLayer].biasesGradient[i] = 2 * self.costC[i] * af.sigmoidDerivative(self.Layers[outputLayer].nodeSums[i])

            for j in range(self.Layers[outputLayer].numOfNeuronsInPreviousLayer):
                self.Layers[outputLayer].weightsGradient[i,j] = 2 * self.costC[i] * af.sigmoidDerivative(self.Layers[outputLayer].nodeSums[i]) * self.Layers[outputLayer - 1].activationValues[j]

        self.Layers[outputLayer].weights = self.Layers[outputLayer].weights - self.learningRate * self.Layers[outputLayer].weightsGradient
        self.Layers[outputLayer].biases = self.Layers[outputLayer].biases - self.learningRate * self.Layers[outputLayer].biasesGradient

        for k in range(outputLayer-1,1):  # iterate through layers - start from max layer number -2 and go backwards
            propagatedCostC = 0
            for i in range(self.Layers[k].numOfNeurons):
                self.Layers[k].biasesGradient[i] = 2 * self.costC[i] * af.sigmoidDerivative(self.Layers[k].nodeSums[i])
                for j in range(self.Layers[k].numOfNeuronsInPreviousLayer):
                    self.Layers[k].weightsGradient[i, j] = 2 * self.costC[i] * af.sigmoidDerivative(self.Layers[k].nodeSums[i]) * self.Layers[k - 1].activationValues[j]
            self.Layers[k].weights = self.Layers[k].weights - self.learningRate * self.Layers[k].weightsGradient
            self.Layers[k].biases = self.Layers[k].biases - self.learningRate * self.Layers[k].biasesGradient

    def writeInputData(self, data):
        for i in range(0, len(self.Layers[0].activationValues)):
            self.Layers[0].activationValues[i] = data[i]

    def loadTrainingExample(self,inputData, desiredOutput):
        self.writeInputData(inputData)
        self.desiredOutput = desiredOutput

    def saveNetwork(self,filename):
        accumulatedNetworkData = [[self.numOfLayers]]
        for layer in self.Layers:
            accumulatedNetworkData.append([layer.numOfNeurons])
            accumulatedNetworkData.append([layer.numOfNeuronsInPreviousLayer])
            accumulatedNetworkData.append(layer.biases)
            accumulatedNetworkData.append(layer.weights.flatten())
        handleFiles.writeCSV(filename, accumulatedNetworkData)

    def loadNetwork(self,fileName):
        file = open(fileName, 'r')
        csvReader = csv.reader(file)
        self.numOfLayers = int(next(csvReader)[0])

        for i in range(0,self.numOfLayers):
            self.Layers[i].numOfNeurons = int(next(csvReader)[0])
            self.Layers[i].numOfNeuronsInPreviousLayer = int(next(csvReader)[0])
            self.Layers[i].biases = np.array(next(csvReader))
            self.Layers[i].biases = self.Layers[i].biases.astype(np.float)
            tempArray = np.array(next(csvReader))
            tempArray = tempArray.astype(np.float)  #convert to float
            self.Layers[i].weights = np.reshape(tempArray, newshape=(self.Layers[i].numOfNeurons, self.Layers[i].numOfNeuronsInPreviousLayer))
        file.close()

    def printNetwork(self):
        for layer in self.Layers:
            layer.printLayer()


