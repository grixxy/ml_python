import numpy as np

def polyFeatures(Xtest, p):
    X_poly = Xtest
    for i in range(2,p+1):
        X_poly = np.hstack((X_poly,Xtest**i))
    return X_poly