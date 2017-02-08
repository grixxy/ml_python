import numpy as np
def gaussianKernel(x1, x2, sigma = 0.1):
    diff = x1 - x2
    summ_sq = diff.T.dot(diff)
    sim = np.exp(-summ_sq / (2 * sigma * sigma))
    return sim