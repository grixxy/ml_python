import numpy as np

def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval),np.max(pval),stepsize):
    #for epsilon = min(pval):stepsize:max(pval):
        cvPredictions = pval < epsilon
        cvPredictions = cvPredictions.reshape(-1,1)
        fp = np.sum(np.logical_and((cvPredictions == 1), (yval == 0)))
        tp = np.sum(np.logical_and((cvPredictions == 1), (yval == 1)))
        fn = np.sum(np.logical_and((cvPredictions == 0), (yval == 1)))
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = 2 * prec * rec / (prec + rec)
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1