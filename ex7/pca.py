import numpy as np
def pca(X):
    (m, n) = X.shape
    Sigma = X.T.dot(X)/m
    return np.linalg.svd(Sigma)



