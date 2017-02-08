import numpy as np

def estimateGaussian(X):
    m,n = X.shape
    mu = (np.sum(X,axis=0)) / m
    sigma2 = np.var(X,0)
    return mu, sigma2