import matplotlib.pyplot as plt
from polyFeatures import polyFeatures
import numpy as np

def plotFit(min_x, max_x, mu, sigma, theta, p):
    '''
    %PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    %Also works with linear regression.
    %   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    %   fit with power p and feature normalization (mu, sigma).

    % Hold on to the current figure
    hold on;

    % We plot a range slightly bigger than the min and max values to get
    % an idea of how the fit will vary outside the range of the data points
    '''
    x = np.arange(min_x - 15.,max_x + 25., 0.05)

    #% Map the X values

    x = x.reshape(-1,1)

    X_poly = polyFeatures(x, p)
    X_poly = X_poly- mu
    X_poly = X_poly/sigma

    #% Add ones

    ones = np.ones((X_poly.shape[0], 1))
    X_poly = np.hstack((ones, X_poly))
    theta = theta.reshape(-1,1)
    y = X_poly.dot(theta) #2512x9 9x1

    #% Plot
    plt.plot(x,y , '-')
