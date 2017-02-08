import matplotlib.pyplot as plt
import numpy as np
from multivariateGaussian import multivariateGaussian

def visualizeFit(X, mu, sigma2):
#VISUALIZEFIT Visualize the dataset and its estimated distribution.
#   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the
#   probability density function of the Gaussian distribution. Each example
#   has a location (x1, x2) that depends on its feature values.
    x = np.arange(0, 35, 0.5)
    X1,X2 = np.meshgrid(x,x)
    X1_s = X1.flatten()
    X1_s = X1.reshape(-1,1)
    X2_s = X2.flatten()
    X2_s = X2_s.reshape(-1,1)
    X1X2 = np.hstack((X1_s,X2_s))
    Z = multivariateGaussian(X1X2,mu,sigma2)
    Z = Z.reshape(X1.shape)
    #ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex8/ex8/'
    #np.savetxt(ml_dir+'Z.csv', Z)
    plt.plot(X[:, 0], X[:, 1],'bx')
    l = 10.**np.arange(-20, 0, 3)
    #if (sum(isinf(Z)) == 0)
    plt.contour(X1, X2, Z, l)#, 10.^(-20:3:0)')
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()