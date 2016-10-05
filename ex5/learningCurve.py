
import numpy as np
from trainLinearReg import *
from costFunctionLinear import *


def learningCurve(X, y, Xval, yval, lambda_):
    m = np.size(X, 0)
    error_train = np.zeros((m,))
    error_val = np.zeros((m,))

    for i in range(0,m):
        X_curr = X[:(i+1), :]
        y_curr = y[:(i+1)]
        theta = trainLinearReg(X_curr,y_curr, lambda_ )
        error_train[i] = linearCostFunctionReg(theta, X_curr,y_curr,  lambda_)
        error_val[i] = linearCostFunctionReg(theta, Xval, yval,  lambda_)

    return error_train, error_val