import numpy as np
from costFunctionLinear import linearCostFunctionAndGradientReg
import scipy.optimize as opt
import functools

def trainLinearReg(X, y, lambda_):
    #%TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
    #%regularization parameter lambda
    #%   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
    #%   the dataset (X, y) and regularization parameter lambda. Returns the
    #%   trained parameters theta.

    #% Initialize Thetanp.ones((n, 1))
    n = X.shape[1]
    initial_theta = np.zeros((n, 1))

    options = {'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': 450}
    theta = opt.minimize(linearCostFunctionAndGradientReg, initial_theta, args=(X, y , lambda_), method='CG', jac=True,
                         options=options)

    return theta.x
