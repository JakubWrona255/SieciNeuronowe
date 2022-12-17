import numpy as np
import os
import time
import matplotlib.pyplot as plt

start = 4
stop = 1400
diff = stop - start

plt1 = np.zeros(diff)
plt2 = np.zeros(diff)

for h in range(0,stop-start):

    sizeParam = h + start
    arr1 = np.random.uniform(1,1000,size=(sizeParam,sizeParam))
    arr2 = np.random.uniform(1,1000,size=(sizeParam,sizeParam))

    st1 = time.time()
    res1 = np.matmul(arr1,arr2)
    end1 = time.time()
    time1 = end1-st1
    plt1[h] = time1

    st2 = time.time()
    res2 = arr1.dot(arr2)
    end2 = time.time()
    time2 = end2-st2
    plt2[h] = time2


plt3 = np.zeros(70)

for h in range(0,70):

    sizeParam = h
    arr1 = np.random.uniform(1, 1000, size=(sizeParam, sizeParam))
    arr2 = np.random.uniform(1, 1000, size=(sizeParam, sizeParam))
    res3 = np.zeros(shape=(sizeParam,sizeParam))

    st3 = time.time()
    for i in range(0,sizeParam ):
        for j in range(0, sizeParam):
            for k in range(0, sizeParam):
                res3[i][j] += arr1[i][k] * arr2[k][j]
    end3 = time.time()
    time3 = end3-st3
    plt3[h] = time3


fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2,figsize=(12, 6))
ax0.set_title('Operacje na macierzach')
ax0.plot(plt1,label="np.matmul(A,B)")
ax0.plot(plt2,label="A.dot(B)")
ax0.set(xlabel='rozmiar macierzy kwadratowej []', ylabel='czas [s]')

ax1.set_title('Operacje w pętli')
ax1.plot(plt3,label="for i in range()...")
ax1.set(xlabel='rozmiar macierzy kwadratowej []', ylabel='czas [s]')
ax1.legend()
ax0.legend()
plt.show()


