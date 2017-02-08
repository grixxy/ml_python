import numpy as np

def computeCentroids(X, idx, K):

    n = np.size(X,1)

    # You need to return the following variables correctly.
    centroids = np.zeros((K, n))



    for i in range(K):
        cent_temp = X[idx[:, 0]==i, :]
        centroids[i, :] = np.sum(cent_temp, axis=0)/np.size(cent_temp, 0)
    return centroids


