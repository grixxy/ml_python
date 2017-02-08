import numpy as np
from trainLinearReg import trainLinearReg
from costFunctionLinear import *


def validationCurve(X, y, Xval, yval):

    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    l_len = np.size(lambda_vec)
    error_train = np.zeros((l_len, 1))
    error_val = np.zeros((l_len, 1))


    for i in range(0,l_len):
        _lambda = lambda_vec[i]
        theta = trainLinearReg(X, y, _lambda)
        error_train[i] = linearCostFunctionReg(theta, X, y, 0)
        error_val[i] = linearCostFunctionReg(theta, Xval, yval, 0)

    return lambda_vec, error_train, error_val

