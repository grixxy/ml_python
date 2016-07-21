import matplotlib.pyplot as plt
import numpy as np

def plotData(X, y):
#PLOTDATA Plots the data points X and y into a new figure
#   PLOTDATA(x,y) plots the data points with + for the positive examples
#   and o for the negative examples. X is assumed to be a Mx2 matrix.

# Create New Figure
#figure; hold on;


# Find Indices of Positive and Negative Examples
    pos = np.where(y==1)
    neg = np.where(y==0)
# Plot Examples
    plt.plot(X[pos, 0], X[pos, 1], 'k+')
    plt.plot(X[neg, 0], X[neg, 1], 'ko')
    plt.hold(True)

