import numpy as np
import pandas as pd

from matplotlib import pyplot as plt


def read_mnist_from_csv(path):
    data = pd.read_csv(path)
    data = np.array(data)
    m, n = data.shape
    dataTest = data[0:1000].T
    dataTrain = data[1000:m].T

    Y_test = dataTest[0]
    X_test = dataTest[1:n]
    X_test = X_test/255

    Y_train = dataTrain[0]
    X_train = dataTrain[1:n]
    X_train = X_train / 255

    #Pierwsze 1000 do testow
    #Reszta do Treningu

    return X_train,Y_train, X_test, Y_test

#zdefiniowanie wag, dla dwóch warstw sieci neuronowej


def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
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
    #one_hot_Y.shape - (42000,10)
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
    print(Y,predictions)
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()

    for j in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if j % 10 == 0:
            print("Iteration: ", j)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


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


def show_subplots():
    for i in range(27,36):
       plt.subplot(330 + 1 + i-27)
       X_show = X_test[:, i, None]
       Y_tru = Y_test[i]
       Y_pred = dev_prediction[i]
       title = "Ground: " + str(Y_tru) + " Pred: " + str(Y_pred)
       plt.title(title)
       single_img = X_show.reshape((28, 28)) * 255
       plt.imshow(single_img, cmap=plt.get_cmap('gray'))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':

    X_train, Y_train, X_test, Y_test = read_mnist_from_csv('train.csv')
    #W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.05,1000)
    #np.save('W_1',W1)
    #np.save('W_2',W2)
    #np.save('b_1',b1)
    #np.save('b_2',b2)

    W1 = np.load('W1.npy')
    b1 = np.load('b1.npy')
    W2 = np.load('W2.npy')
    b2 = np.load('b2.npy')

    dev_prediction = make_predictions(X_test,W1,b1,W2,b2)
    acc = get_accuracy(dev_prediction,Y_test)
    print('Accuracy na zbiorze testowym:')
    print(acc)

    show_subplots()

    #X_dev = data_dev[1:,1,None]
    #Y_dev = data_dev[0][0]

    #for i in range(100,200):
    #    test_prediction(i,X_test,W1,b1,W2,b2)
