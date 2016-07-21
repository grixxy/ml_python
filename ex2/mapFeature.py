import numpy as np

def mapFeature(X1, X2):
    '''
    % MAPFEATURE Feature mapping function to polynomial features
    %
    %   MAPFEATURE(X1, X2) maps the two input features
    %   to quadratic features used in the regularization exercise.
    %
    %   Returns a new feature array with more features, comprising of
    %   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
    %
    %   Inputs X1, X2 must be the same size
    %
    '''

    degree = 6
    m = np.size(X1)
    X1 = X1.reshape(-1,1)
    X2 = X2.reshape(-1,1)
    out = np.ones((m,1))
    for i in range(1,degree+1):
        for j in range(0,i+1):
            temp = X1**(i-j)*(X2**j)
            out = np.hstack((out,temp))
    return out
