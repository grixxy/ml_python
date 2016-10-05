import numpy as np
import math as m
import scipy.io as sio

def linearCostFunctionAndGradient(theta, X, y):

    m = X.shape[0]

    h = X.dot(theta)  # m by 1
    h = h.reshape(m, -1)
    diff = h - y # m by 1  - result for each training set
    J = 1 / (2 * m) * (diff.T.dot(diff))
    summ=X.T.dot(diff) #  % X 97*2   2 X 97   * 97 X 1
    grad = summ/m
    return (J,grad)

def linearCostFunction(theta, X, y):
    (J,grad) = linearCostFunctionAndGradient(theta, X, y)
    return J

def linearCostFunctionGrad(theta, X, y):
    (J,grad) = linearCostFunctionAndGradient(theta, X, y)
    grad = np.ndarray.flatten(grad)
    return grad


def linearCostFunctionAndGradientReg(theta, X, y, lambda_):
    m = X.shape[0]
    n = X.shape[1]
    (J,grad) = linearCostFunctionAndGradient(theta, X, y)
    J = J+lambda_*np.sum(theta[1:]**2)/(2.*m)
    p1 = lambda_*theta[1:]/m
   # p1 = p1.
    grad[1:] = grad[1:]+p1.reshape(-1,1)
    grad = np.ndarray.flatten(grad)
    return J,grad


def linearCostFunctionReg(theta, X, y, lambda_):
    (J,grad) = linearCostFunctionAndGradientReg(theta, X, y, lambda_)

    return np.sum(J)

def linearCostFunctionGradientReg(theta, X, y, lambda_):
    (J,grad) = linearCostFunctionAndGradientReg(theta, X, y, lambda_)
    return grad


