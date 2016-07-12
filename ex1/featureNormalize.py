'''
% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
%

'''

import numpy as np

def featureNormalize(X):

    n = X.shape[1]
    X_norm = X
    mu = np.zeros((1, n))
    sigma = np.zeros((1, n))
    num_features = X.shape[1]


    for i in range(0,num_features):
        mu[0,i]= np.mean(X[:,i])
        X_norm[:,i] = X_norm[:,i]-mu[0,i]
        sigma[0,i] = np.std(X[:,i])
        X_norm[:,i] = X_norm[:,i]/sigma[0,i];

    return X_norm



