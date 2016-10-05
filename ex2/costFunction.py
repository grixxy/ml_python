import numpy as np
import math as m
import scipy.io as sio

def costFunctionAndGradient(theta, X, y):
    #print(y[:10,:])


    m = X.shape[0]

    J = 0
    grad = np.zeros(np.size(theta))

    h = (sigmoid(X.dot(theta)))
    #print(sum(h))
    h = h.reshape(m,-1)


   # print(yT[:10,:])
    first = -(y.T.dot(np.log(h)))
#    print('first', first)
    #print('first2', (-y[:500,:]).T.dot(np.log(h[:500,:])))
  #  print('first2', np.sum(-y[:100,:]))
 #   print(yT)
#    print(np.log(h))
    #print('res = ', yT[:,:500].dot(np.log(h[:500,:])))

    #print('first ', first)


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
    return 1./(1.+np.exp(-X))

def costFunctionAndGradientReg(theta, X, y, lambda_):
    m = X.shape[0]
    n = X.shape[1]
    (J,grad) = costFunctionAndGradient(theta, X, y)
    J = J+lambda_*np.sum(theta[1:]**2)/(2.*m)
    p1 = lambda_*theta[1:]/m
   # p1 = p1.
    grad[1:] = grad[1:]+p1.reshape(-1,1)
    grad = np.ndarray.flatten(grad)
    return J,grad


def costFunctionReg(theta, X, y, lambda_):
    (J,grad) = costFunctionAndGradientReg(theta, X, y, lambda_)
    print(J)
    return np.sum(J)

def costFunctionGradientReg(theta, X, y, lambda_):
    (J,grad) = costFunctionAndGradientReg(theta, X, y, lambda_)
    return grad


