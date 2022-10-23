import handlingFiles as handleFiles
import numpy as np
import NeuralNetwork as nn

for i in range (1,3):
    print(i)
fileName = 'test.csv'    #to read network
newFileName = 'saved_copy.csv'  # to save network
#
network1 = nn.Network(1,[2,3,2])
#network1.loadNetwork('newTest.csv')
#
#trainingInput = [0.1,0.2]
#desiredResult = [1,0]
#network1.loadTrainingExample(trainingInput,desiredResult)
network1.feedForward()
network1.propagateBackwards()
network1.printNetwork()
#network1.saveNetwork('newTest.csv')






















