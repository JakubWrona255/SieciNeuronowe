import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def read_mnist_from_csv(path):
    data = pd.read_csv(path)
    data = np.array(data)
    m, n = data.shape

    testSplit = 10

    dataTest = data[0:testSplit]
    dataTrain = data[testSplit:m]

    Y_test = np.zeros(shape=testSplit)
    X_test = np.zeros(shape=(testSplit,28,28))
    for i in range(0,testSplit):
        Y_test[i] = dataTest[i,0]
        tempArr = np.array(dataTest[i,1:n])
        X_test[i] = tempArr.reshape(28,28) / 255

    Y_train = np.zeros(shape=m-testSplit)
    X_train = np.zeros(shape=(m-testSplit, 28, 28))
    for i in range(0,m-testSplit):
        Y_train[i] = dataTrain[i, 0]
        tempArr = np.array(dataTrain[i, 1:n])
        X_train[i] = tempArr.reshape(28, 28) / 255

    #AKTUALNIE
    #wymiar zbioru X to (batchsize, 28, 28) a Y to (batchsize, 1)
    return X_train,Y_train, X_test, Y_test


def init_params():

    edge_1 = np.array([
            [-1,-1,-1],
            [-1, 4, -1],
            [-1,-1,-1]])
    edge_2 = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]])
    sharpen = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, -0]])
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]])
    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]])
    edge_3 = np.array([
        [1, -1, -1],
        [0, 1, -1],
        [0, 0, 1]])
    edge_4 = np.array([
        [-1, -1, 1],
        [-1, 1, 0],
        [1, 0, 0]])
    corner_1 = np.array([
        [1, 2, 2],
        [-1, -1, 2],
        [-1, -1, 1]])
    corner_2 = np.array([
        [2, 2, 1],
        [2, -1, -1],
        [1, -1, -1]])
    corner_3 = np.array([
        [1, -1, -1],
        [2, -1, -1],
        [2, 2, 1]])
    corner_4 = np.array([
        [-1, -1, 1],
        [-1, -1, 2],
        [1, 2, 2]])

    Filters_1 = np.zeros(shape=(11,3,3))
    Filters_1[0] = edge_1
    Filters_1[1] = edge_2
    Filters_1[2] = edge_3
    Filters_1[3] = edge_4
    Filters_1[4] = sharpen
    Filters_1[5] = sobel_x
    Filters_1[6] = sobel_y
    Filters_1[7] = corner_1
    Filters_1[8] = corner_2
    Filters_1[9] = corner_3
    Filters_1[10] = corner_4

    return Filters_1


def conv_forward(dataX,filters):

    le,x,y = dataX.shape
    filtersNum, xf, yf = filters.shape
    outputData = np.zeros(shape=(le,filtersNum,x,y))

    for i in range(0,le):   #batch size

        tempPicture = np.zeros(shape=(30,30))   #padding
        tempPicture[1:29,1:29] = dataX[i]        #paste image to center
        outPictures = np.zeros(shape=(filtersNum, 28, 28))  # empty output image

        for f in range(0,filtersNum):

            for j in range(0,x):                    #picture size

                for k in range(0,y):
                    tempSum = 0

                    for h in range(0,xf):            #filter size
                        for g in range(0,yf):
                            tempSum =  tempPicture[j+h,k+g] * filters[f,h,g] + tempSum
                    outPictures[f,j,k] = tempSum
        outputData[i] = outPictures
    return outputData


def flatten(data):

    le,fil,x,y = data.shape
    dataOut = np.zeros(shape=(le,fil*x*y))
    for i in range(0,le):
        tempArr = data[i].reshape(fil*x*y)
        dataOut[i] = tempArr
    return dataOut


def main():

    X_train, Y_train, X_test, Y_test = read_mnist_from_csv('train.csv')
    filters = init_params()
    X_test_data = conv_forward(X_test,filters)

    for i in range(0,10):
        for j in range(0,9):
            current_image = X_test_data[i,j]
            plt.subplot(330 + 1 + j)
            X_show = current_image
            title = "Mask: " + str(j)
            plt.title(title)
            single_img = X_show.reshape((28, 28)) * 255
            plt.imshow(single_img)#, cmap=plt.get_cmap('gray'))

        plt.tight_layout()
        plt.show()


main()
