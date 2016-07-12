import numpy as np
import matplotlib.pyplot as plt
from computeCost import computeCost

def gradientDescent(X, y, theta, alpha, num_iters):
    #GRADIENTDESCENT Performs gradient descent to learn theta
    #  theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
    #   taking num_iters gradient steps with learning rate alpha

    # Initialize some useful values
    m = np.size(y); # number of training examples
    J_history = np.zeros((num_iters, 1))

    for iter in range(1,num_iters):
        h = X.dot(theta) # m by 1
        diff=h-y  # m by 1  - result for each training set
        summ=(X.T).dot(diff)  # X 97*2   2 X 97   * 97 X 1
        gradient = summ/m
        theta -= alpha * gradient
        J_history[iter] = computeCost(X, y, theta)

    return (theta, J_history)