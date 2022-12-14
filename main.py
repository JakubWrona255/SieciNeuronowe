import numpy as np
import pandas as pd

from matplotlib import pyplot as plt







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data = pd.read_csv('train.csv')
    data = np.array(data)


    data_dev = data.T
    X_dev = data_dev[1:,1,None]
    Y_dev = data_dev[0][0]

    single_img = X_dev.reshape((28,28)) * 255
    plt.gray()
    plt.imshow(single_img)
    plt.show()

    print(Y_dev)


