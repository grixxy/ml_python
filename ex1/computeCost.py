import numpy as np

def computeCost(X, y, theta):
    m = np.size(y) # number of training examples

    h = X.dot(theta) # m by 1
    diff=h-y;  # m by 1  - result for each training set
    J=1/(2*m)*(diff.T.dot(diff))
    #X_THETA = X.dot(theta)
    #J = 1./(2.*m)*np.sum((X_THETA-y)**2)
    #print(J)
    return J[0][0]