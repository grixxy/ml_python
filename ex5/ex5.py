import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from costFunctionLinear import *
from trainLinearReg import trainLinearReg
from learningCurve import learningCurve
from ex1.featureNormalize import featureNormalize
from polyFeatures import polyFeatures
from plotFit import plotFit

'''
%% Machine Learning Online Class
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment and plot
%  the data.
%
% Load from ex5data1:
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');


'''
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex5/ex5/'

data = sio.loadmat(ml_dir+'ex5data1.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']
Xval = data['Xval']
yval = data['yval']
Xtest = data['Xtest']
ytest = data['ytest']

# m = Number of examples
m = X.shape[0]
ones = np.ones((m,1))
ones_X = np.hstack((ones,X))
ones = np.ones((Xval.shape[0],1))
ones_Xval = np.hstack((ones,Xval))


# Plot training data
plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

'''
%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear
%  regression.
%
'''
theta = np.array([1 , 1])

J = linearCostFunctionReg(theta, ones_X, y, 1)

print('Cost at theta = [1 ; 1]: \n(this value should be about 303.993192)\n', J)


'''
%% =========== Part 3: Regularized Linear Regression Gradient =============
%  You should now implement the gradient for regularized linear
%  regression.
%
'''

J, grad = linearCostFunctionAndGradientReg(theta, ones_X, y, 1)

print('Gradient at theta = [1 ; 1] \n(this value should be about [-15.303016; 598.250744])\n',grad[0], grad[1])



'''
%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train
%  regularized linear regression.
%
%  Write Up Note: The data is non-linear, so this will not give a great
%                 fit.
%

%  Train linear regression with lambda = 0
'''
lambda_ = 0
theta = trainLinearReg(ones_X, y, lambda_)

#  Plot fit over the data
plt.plot(X, y, 'rx')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
y_new = ones_X.dot(theta)
plt.plot(X, y_new, '--')
plt.show()

'''
%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function.
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
%
'''
lambda_ = 0
error_train, error_val = learningCurve(ones_X, y, ones_Xval, yval, lambda_)

plt.plot(range(0,m), error_train, range(0,m), error_val)
plt.title('Learning curve for linear regression')
#plt.legend('Train', 'Cross Validation')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([1,13,1,150])


print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(0,m):
    print(error_train[i], error_val[i])
plt.show()
'''
%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%
'''
p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p)
print('Poly Training Example 1:\n', X_poly[0, :])
X_poly, mu, sigma = featureNormalize(X_poly)  # Normalize



print('mu:\n', mu)
print('Sigma:\n', sigma)

ones = np.ones((X_poly.shape[0],1))
X_poly = np.hstack((ones,X_poly))

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test/sigma

ones = np.ones((X_poly_test.shape[0],1))
X_poly_test = np.hstack((ones,X_poly_test))


# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val/sigma
ones = np.ones((X_poly_val.shape[0],1))
X_poly_val = np.hstack((ones,X_poly_val))

print('Normalized Training Example 1:\n', X_poly[0, :])






'''
%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%
'''

_lambda = 0

theta = trainLinearReg(X_poly, y, _lambda)

# Plot training data and fit

plt.figure(1)
plt.plot(X, y, 'rx')
plotFit(np.min(X), np.max(X), mu, sigma, theta, p)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title ('Polynomial Regression Fit')

plt.figure(2)
[error_train, error_val] = learningCurve(X_poly, y, X_poly_val, yval, _lambda)
_x = np.arange(0,m)
plt.plot(_x, error_train, _x, error_val)

plt.title('Polynomial Regression Learning Curve ')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.axis([0,13,0,100])
plt.legend('Train', 'Cross Validation')


print('Polynomial Regression (lambda = \n\n ',_lambda)
print('# Training Examples\tTrain Error\tCross Validation Error\n')
for i in range(0,m):
    print(i,'\t',error_train[i],'\t', error_val[i],'\n')
plt.show()


'''
%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

close all;
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

fprintf('Program paused. Press enter to continue.\n');
pause;
'''