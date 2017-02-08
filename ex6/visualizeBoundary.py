import numpy as np
import matplotlib.pyplot as plt

from plotData import plotDataNoShow

def visualizeBoundaryLinear(X, y, model):

    xp = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
    b = model.intercept_[0]
    w = model.coef_[0]
    yp = - (w[0]*xp + b)/w[1]
    plotDataNoShow(X, y)
    plt.plot(xp, yp, '-b')
    plt.show()

def visualizeBoundary(X, y, model):
    plotDataNoShow(X, y)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(np.min(X[:,0]), np.max(X[:,0]), 100).T
    x2plot = np.linspace(np.min(X[:,1]), max(X[:,1]), 100).T
    [X1, X2] = np.meshgrid(x1plot, x2plot)
    vals = np.zeros((np.shape(X1)))
    for i in range(0,np.size(X1,1)):
        t_x1 = X1[:, i].reshape(-1,1)
        t_x2 = X2[:, i].reshape(-1,1)
        temp_X = np.hstack((t_x1, t_x2 ))
        vals[:, i] = model.predict(temp_X)
    plt.contour(X1, X2, vals)
    plt.show()
    np.savetxt('/Users/gregory/Desktop/me/coursera/machine_learning/ml_007/ex6/python_vals.txt', vals)
