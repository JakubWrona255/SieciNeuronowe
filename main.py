import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt


def flatten(data):
    le,fil,x,y = data.shape
    dataOut = np.zeros(shape=(le,fil*x*y))
    for i in range(0,le):
        tempArr = data[i].reshape(fil*x*y)
        dataOut[i] = tempArr
    return dataOut


def init_filters():
    edge_1 = np.array([
            [-1,-1,-1],
            [ -1,4, -1],
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


def read_mnist_from_csv(path):
    data = pd.read_csv(path)
    data = np.array(data)
    m, n = data.shape
    m = 4100              #obcinamy set do x przykładów
    testSplit = 500

    dataTest = data[0:testSplit]
    dataTrain = data[testSplit:m]

    Y_test = np.zeros(shape=testSplit,dtype=int)
    X_test = np.zeros(shape=(testSplit,28,28))
    for i in range(0,testSplit):
        Y_test[i] = dataTest[i,0]
        tempArr = np.array(dataTest[i,1:n])
        X_test[i] = tempArr.reshape(28,28) / 255

    Y_train = np.zeros(shape=m-testSplit,dtype=int)
    X_train = np.zeros(shape=(m-testSplit, 28, 28))
    for i in range(0,m-testSplit):
        Y_train[i] = dataTrain[i, 0]
        tempArr = np.array(dataTrain[i, 1:n])
        X_train[i] = tempArr.reshape(28, 28) / 255

    #AKTUALNIE
    #wymiar zbioru X to (batchsize, 28, 28) a Y to (batchsize, 1)
    return X_train,Y_train, X_test, Y_test


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


def init_params():
    W1 = np.random.rand(100, 8624) - 0.5
    b1 = np.random.rand(100, 1) - 0.5
    W2 = np.random.rand(10, 100) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2
    #Dwie warstwy


def ReLU(Z):
    return np.maximum(Z, 0)


def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def ReLU_deriv(Z):
    return Z > 0


def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y  #loss
    dW2 = 1 / Y.shape[0] * dZ2.dot(A1.T)
    db2 = 1 / Y.shape[0] * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / Y.shape[0] * dZ1.dot(X.T)
    db1 = 1 / Y.shape[0] * np.sum(dZ1)
    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def train(X, Y,X_test, Y_test, alpha, iterations):

    W1, b1, W2, b2 = init_params()
    filters = init_filters()
    dev_prediction = 0

    for j in range(iterations):

        st = time.time()

        convData = conv_forward(X,filters)
        Xdata = flatten(convData)
        Xdata = Xdata.T
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, Xdata)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, Xdata, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        end = time.time()
        time0 = end - st

        if j % 1 == 0:
            print("Epoch: ", j)
            print("Epoch time: ",time0)
            predictions = get_predictions(A2)
            print("Train accuracy: ", get_accuracy(predictions, Y))
            dev_prediction = run_test_prediction(X_test,Y_test, filters, W1, b1, W2, b2)

    return W1, b1, W2, b2, filters,dev_prediction


def run_test_prediction(X_test,Y_test, filters, W1, b1, W2, b2):
    testData = conv_forward(X_test, filters)
    testData = flatten(testData)
    testData = testData.T
    dev_prediction = make_predictions(testData, W1, b1, W2, b2)
    acc = get_accuracy(dev_prediction, Y_test)
    print('Test accuracy:',acc)
    return dev_prediction


def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions


def test_prediction(index, X_test, W1, b1, W2, b2):
    current_image = X_test[:, index, None]
    prediction = make_predictions(X_test[:, index, None], W1, b1, W2, b2)
    label = Y_test[index]
    print("Prediction: ", prediction)
    print("Label: ", label)

    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()


def show_subplots(start,stop,dev_prediction):
    for i in range(start,stop):
       plt.subplot(330 + 1 + i-start)
       X_show = X_test[i]
       Y_tru = Y_test[i]
       Y_pred = dev_prediction[i]
       title = "Ground: " + str(Y_tru) + " Pred: " + str(Y_pred)
       plt.title(title)
       single_img = X_show.reshape((28, 28)) * 255
       plt.imshow(single_img, cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    st1 = time.time()
    X_train, Y_train, X_test, Y_test = read_mnist_from_csv('train.csv')
    W1, b1, W2,b2,filters, dev_prediction = train(X_train, Y_train,X_test, Y_test, 0.05,20)
    np.save('W_1_2',W1)
    np.save('W_2_2',W2)
    np.save('b_1_2',b1)
    np.save('b_2_2',b2)
    np.save('filters_2',filters)

    #show_subplots(9, 18, dev_prediction)
    #show_subplots(18, 27, dev_prediction)
    #show_subplots(27, 36, dev_prediction)

    #W1 = np.load('W_1.npy')
    #b1 = np.load('b_1.npy')
    #W2 = np.load('W_2.npy')
    #b2 = np.load('b_2.npy')
    #filters = np.load('filters.npy')

    end1 = time.time()
    time1 = end1 - st1
    print(time1)
    np.save('time',time1)
