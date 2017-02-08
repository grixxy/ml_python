import numpy as np
def findClosestCentroids(X, centroids):
# FINDCLOSESTCENTROIDS computes the centroid memberships
#for every example idx = FINDCLOSESTCENTROIDS(X, centroids)
#returns the closest centroids in idx
#for a dataset X where each row is a single example.idx = m x 1
# vector of centroid assignments(i.e.each entry in range[1..K])
# Set K
    K = np.size(centroids, 0)
    m = np.size(X, 0)

#% You need to return the following variables correctly.
    idx = np.zeros((m, 1), dtype='int')

# == == == == == == == == == == == YOUR CODE HERE == == == == == == == == == == ==
# Instructions: Go over every example, find its closest centroid, and store
# the index inside idx at the appropriate location.
# Concretely, idx(i) should contain the index of the centroid closest to example
#i.Hence, it should be a value in the range 1..K Note: You can use
#for -loop over the examples to compute this.


    vec_diff = 0
    minim = 0

    for i in range(m):
        for j in range(K):
            vec_diff = X[i,:] - centroids[j,:]
            vec_diff = vec_diff.dot(vec_diff.T)
            if (j == 0):
                minim = vec_diff
            elif(minim > vec_diff):
                minim = vec_diff
                idx[i] = j
    return idx
