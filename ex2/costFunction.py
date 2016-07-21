import numpy as np
import math as m

def costFunctionAndGradient(theta, X, y):

    theta = theta.reshape(-1,1)

    m = X.shape[0]

    J = 0
    grad = np.zeros(np.size(theta))

    h = (sigmoid(X.dot(theta)))


    first = (-y).T.dot(np.log(h))

    second = (1-y).T.dot((np.log(1-h)))

    J = (first-second)/m

    diff=h-y # % m by 1  - result for each training set
    summ=X.T.dot(diff) #  % X 97*2   2 X 97   * 97 X 1

    grad = summ/m

    return (J,grad)

def costFunction(theta, X, y):
    (J,grad) = costFunctionAndGradient(theta, X, y)
    return J

def costFunctionGrad(theta, X, y):
    (J,grad) = costFunctionAndGradient(theta, X, y)
    grad = np.ndarray.flatten(grad)
    return grad

def sigmoid(X):
    return 1/(1+np.exp(X))

def costFunctionAndGradientReg(theta, X, y, lambda_):
    theta = theta.reshape(-1,1)
    m = X.shape[0]
    n = X.shape[1]
    (J,grad) = costFunctionAndGradient(theta, X, y)
    J = J+lambda_*np.sum(theta[1:]**2)/(2.*m)
    p1 = lambda_*theta[1:]/m
   # p1 = p1.
    grad[1:] = grad[1:]+p1
    return (J,grad)


def costFunctionReg(theta, X, y, lambda_):
    (J,grad) = costFunctionAndGradientReg(theta, X, y, lambda_)
    #print(J)
    return np.sum(J)

def costFunctionGradientReg(theta, X, y, lambda_):
    (J,grad) = costFunctionAndGradientReg(theta, X, y, lambda_)
    grad = np.ndarray.flatten(grad)
    return grad


