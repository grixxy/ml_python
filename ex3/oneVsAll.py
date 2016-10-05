import numpy as np
import scipy.optimize as opt
from ex2.costFunction import *
import time
import datetime

def oneVsAll(X, y, num_labels, lambda_):


    m = X.shape[0]
    # Add ones to the X data matrix
    X = np.hstack((np.ones((m, 1)), X))
    n = X.shape[1]

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n))



    print('X\n',X[:3, :])
    print('y\n', y[:3,:])

    options={'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': 450}
    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(st)


    for c in range(0,10):
        initial_theta = np.ones((n, 1))
        #% Set options for fminunc
        # Run fmincg to obtain the optimal theta
        # This function will return theta and the cost
        theta = opt.minimize(costFunctionAndGradientReg,initial_theta, args=(X,y==c,lambda_),method='CG',jac = True, options=options)
        #print(theta)
        all_theta [c, :] = theta.x

    ts = time.time()
    st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
    print(st)
    return all_theta

