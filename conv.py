import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_mnist_from_csv(path):
    data = pd.read_csv(path)

    data = np.array(data)
    m, n = data.shape

    dataTest = data[0:1000].T
    dataTrain = data[1000:m].T
    print(dataTrain[0])
    Y_test = dataTest[0]
    X_test = dataTest[1:n]
    print(X_test.shape)
    for i in range(0,1):
        pass
    #X_test.reshape((28, 28))
    #plt.imshow(X_test[0], cmap=plt.get_cmap('gray'))
    #plt.show()
    #X_test = X_test/255


    #Y_train = dataTrain[0]
    #X_train = dataTrain[1:n]
    #X_train.reshape((28, 28))
    #X_train = X_train / 255

    #Pierwsze 1000 do testow
    #Reszta do Treningu

    #return X_train,Y_train, X_test, Y_test


def init_params():
    Filters1 = np.random.random((32,3,3)) * 5 - 2.5
    print(Filters1[0])

read_mnist_from_csv('train.csv')
