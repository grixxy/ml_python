import numpy as np
from featureNormalize import featureNormalize
from gradientDescent import gradientDescent
import matplotlib.pyplot as plt
from sklearn import linear_model


# Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:

#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#

# Initialization

# ================ Part 1: Feature Normalization ================

# Clear and Close Figures
#clear ; close all; clc
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex1/ex1/'

print('Loading data ...\n')

# Load Data
data = np.loadtxt(ml_dir + 'ex1data2.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]


y = y.reshape((-1,1))
X = X.reshape(-1,2)
print(y.shape)
print(X.shape)
m = np.size(y)



# Print out some data points
print('First 10 examples from the dataset: \n')
print(' x = , y =  \n', X[:10,:], y[:10,:])

# Scale features and set them to zero mean
print('Normalizing Features ...\n')
X_nonnorm = X
X = featureNormalize(X)


print('First 10 examples after normilize: \n')
print(' x = , y =  \n', X[:10,:], y[:10,:])

# Add intercept term to X
ones = np.ones((m,1))
X = np.hstack((ones,X))


'''
%% ================ Part 2: Gradient Descent ================

% ====================== YOUR CODE HERE ======================
% Instructions: We have provided you with the following starter
%               code that runs gradient descent with a particular
%               learning rate (alpha).
%
%               Your task is to first make sure that your functions -
%               computeCost and gradientDescent already work with
%               this starter code and support multiple variables.
%
%               After that, try running gradient descent with
%               different values of alpha and see which one gives
%               you the best result.
%
%               Finally, you should complete the code at the end
%               to predict the price of a 1650 sq-ft, 3 br house.
%
% Hint: By using the 'hold on' command, you can plot multiple
%       graphs on the same figure.
%
% Hint: At prediction, make sure you do the same feature normalization.
%
'''
print('Running gradient descent ...\n')

# Choose some alpha value
alpha = 0.01
num_iters = 400;

# Init Theta and Run Gradient Descent
theta = np.zeros((3, 1))
theta_J_history = gradientDescent(X, y, theta, alpha, num_iters)
theta = theta_J_history[0]
J_history= theta_J_history[1]

alpha = 0.001
theta = np.zeros((3, 1))
J_history1 = gradientDescent(X, y, theta, alpha, num_iters)[1]

alpha = 1
theta = np.zeros((3, 1))
J_history2 = gradientDescent(X, y, theta, alpha, num_iters)[1]



# Plot the convergence graph
x = np.arange(0,np.size(J_history))
plt.figure()
plt.plot(x,J_history, x,J_history1,  x,J_history2)

plt.show()



theta = theta_J_history[0]
# Display gradient descent's result
print('Theta computed from gradient descent: \n');
print(theta);
print('\n');

clf = linear_model.Ridge (alpha = .01)
clf.fit(X_nonnorm,y)
print('Coefficients from scikit learn LinearRegression: \n',clf.intercept_,' ',clf.coef_)


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
# Recall that the first column of X is all-ones. Thus, it does
# not need to be normalized.
price = np.array([1,1650,3]).dot(theta)


print('Predicted price of a 1650 sq-ft, 3 br house ', price)

price = np.array([1650,3]).dot(clf.coef_[0])+clf.intercept_

print('Predicted price of a 1650 sq-ft, 3 br house from Ridge Linear regressor ', price)


# ================ Part 3: Normal Equations ================

print('Solving with normal equations...\n');
'''
% ====================== YOUR CODE HERE ======================
% Instructions: The following code computes the closed form
%               solution for linear regression using the normal
%               equations. You should complete the code in
%               normalEqn.m
%
%               After doing so, you should complete this code
%               to predict the price of a 1650 sq-ft, 3 br house.
%
'''



# Add intercept term to X
ones = np.ones((m,1))
X = np.hstack((ones,X_nonnorm))

print(X.shape)
print(y.shape)
# Calculate the parameters from the normal equation
theta = np.linalg.lstsq(X, y)[0]

# Display normal equation's result
print('Theta computed from lstsq: \n')
print(theta)
print('\n')


# Estimate the price of a 1650 sq-ft, 3 br house
# ====================== YOUR CODE HERE ======================
price = np.array([1,1650,3]).dot(theta)


print('Predicted price of a 1650 sq-ft, 3 br house (using normal lstsq): ', price)


theta = np.linalg.pinv((X.T.dot(X))).dot(X.T).dot(y)
#X_prod = X.T.dot(X)

#pinv((X'*X))*X'*y;


# Display normal equation's result
print('Theta computed from the normal equations: \n')
print(theta)
print('\n')

price = np.array([1,1650,3]).dot(theta)


print('Predicted price of a 1650 sq-ft, 3 br house (using normal equations---manually): ', price)