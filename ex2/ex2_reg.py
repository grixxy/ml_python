import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from costFunction import costFunction, costFunctionAndGradient, costFunctionGrad, sigmoid, costFunctionAndGradientReg, costFunctionReg, costFunctionGradientReg
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict
from mapFeature import mapFeature
import scipy.optimize as opt


# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# Initialization
#clear ; close all; clc

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex2/ex2/'

data = np.loadtxt(ml_dir+'ex2data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]


y = y.reshape((-1,1))
X = X.reshape(-1,2)

plotData(X, y)

# Labels and Legend
plt.xlabel('Microchip Test 1')
plt.ylabel('Microchip Test 2')
#plt.show()

'''
%% =========== Part 1: Regularized Logistic Regression ============
%  In this part, you are given a dataset with data points that are not
%  linearly separable. However, you would still like to use logistic
%  regression to classify the data points.
%
%  To do so, you introduce more features to use -- in particular, you add
%  polynomial features to our data matrix (similar to polynomial
%  regression).
%

% Add Polynomial Features

% Note that mapFeature also adds a column of ones for us, so the intercept
% term is handled
'''

X = mapFeature(X[:,0], X[:,1])

n = X.shape[1]

# Initialize fitting parameters
initial_theta = np.zeros((n, 1))

# Set regularization parameter lambda to 1
lambda_ = 1;

# Compute and display initial cost and gradient for regularized logistic
# regression
cost = costFunctionReg(initial_theta, X, y, lambda_)


'''
%% ============= Part 2: Regularization and Accuracies =============
%  Optional Exercise:
%  In this part, you will get to try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%
'''
l = [0,1,100]

for lambda_ in l:
    # Initialize fitting parameters
    initial_theta = np.zeros((n, 1))

    # Set regularization parameter lambda to 1 (you should vary this)


    # Optimize

    options={'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': 400000}
    theta = opt.minimize(costFunctionReg,initial_theta, args=(X,y,lambda_),method='BFGS', options=options)

    theta = theta.x


    # Plot Boundary
    plotDecisionBoundary(theta, X, y)


    # Labels and Legend
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.show()


    # Compute accuracy on our training set
    p = predict(theta, X)

    print('Train Accuracy: \n', np.mean((p == np.ndarray.flatten(y))) * 100.)



