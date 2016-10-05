import numpy as np
from ex2.costFunction import sigmoid

def predict(Thetha1, Thetha2, X):
    m = X.shape[0] #5000 x 401
    # Add ones to the X data matrix
    a1 = np.hstack((np.ones((m, 1)), X))
    n = X.shape[1]

    z2 = a1.dot(Thetha1.T)
    a2 = sigmoid(z2) #5000 x 25
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2)) #5000 x 26
    z3 = a2.dot(Thetha2.T)
    h =  sigmoid(z3)
    h = np.roll(h,1,1)
    res = np.argmax(h,axis = 1)
    res = res.reshape(-1,1)
    return res

def predictRight(Thetha1, Thetha2, X):
    m = X.shape[0] #5000 x 401
    # Add ones to the X data matrix
    a1 = np.hstack((np.ones((m, 1)), X))
    n = X.shape[1]

    z2 = a1.dot(Thetha1.T)
    a2 = sigmoid(z2) #5000 x 25
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2)) #5000 x 26
    z3 = a2.dot(Thetha2.T)
    h =  sigmoid(z3)

    res = np.argmax(h,axis = 1)
    res = res.reshape(-1,1)
    return res