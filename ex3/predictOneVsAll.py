import numpy as np
from ex2.costFunction import sigmoid

def predictOneVsAll(theta, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X)) #[5000,401]
    num_labels = theta.shape[0] #[10,401]

    h = sigmoid(X.dot(theta.T))
    res = np.argmax(h,axis = 1)
    return res



