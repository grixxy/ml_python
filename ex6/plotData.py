import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
    plotDataNoShow(X, y)
    plt.show()


def plotDataNoShow(X, y):

    pos = y == 1
    pos = pos.flatten()
    neg = np.where(y == 0)[0]

    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'ko')


