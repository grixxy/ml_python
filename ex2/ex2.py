import numpy as np
import matplotlib.pyplot as plt
from plotData import plotData
from costFunction import costFunction, costFunctionAndGradient, costFunctionGrad, sigmoid
from plotDecisionBoundary import plotDecisionBoundary
from predict import predict
import scipy.optimize as opt


ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex2/ex2/'

'''
%% Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the logistic
%  regression exercise. You will need to complete the following functions
%  in this exericse:
%
%     sigmoid.m
%     costFunction.m
%     predict.m
%     costFunctionReg.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% Load Data
%  The first two columns contains the exam scores and the third column
%  contains the label.
'''


data = np.loadtxt(ml_dir+'ex2data1.txt', delimiter=',')

X = data[:, 0:2]
y = data[:, 2]


y = y.reshape((-1,1))
X = X.reshape(-1,2)
print(y.shape)
print(X.shape)
m = np.size(y)


# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

#print('Plotting data with + indicating (y = 1) examples and o ' 'indicating (y = 0) examples.\n');

plotData(X, y)
plt.show()


# ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

# Setup the data matrix appropriately, and add ones for the intercept term
n = X.shape[1]

# Add intercept term to x and X_test
X = np.hstack((np.ones((m,1)), X))


# Initialize fitting parameters
initial_theta = np.zeros((n + 1, 1))

# Compute and display initial cost and gradient
(cost, grad) = costFunctionAndGradient(initial_theta, X, y);

print('Cost at initial theta (zeros): \n', cost);
print('Gradient at initial theta (zeros): \n');
print( grad);


#============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.



#Nelder-Mead Simplex algorithm (fmin)
theta = opt.fmin(costFunction,initial_theta, args=(X,y),  xtol=0.0001)
print('Cost at theta found by fmin: ', cost)
print('theta: \n')
print(theta)


# Plot Boundary
plotDecisionBoundary(theta, X, y)
plt.show()




'''
%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1
%  and score 85 on exam 2
'''

prob = sigmoid(np.array([1,45,85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of ', prob)

# Compute accuracy on our training set
p = predict(theta, X)
p = p.reshape((-1,1))

#p_y = np.hstack((p,y))
#print(p_y)
print('Train Accuracy: ', np.mean((p == y) * 100.))

