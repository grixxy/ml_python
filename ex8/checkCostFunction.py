import numpy as np
from ex4.computeNumericalGradient import computeNumericalGradient
from cofiCostFunc import cofiCostFunc
import functools


def checkCostFunction(lambda_ = 0):
#CHECKCOSTFUNCTION Creates a collaborative filering problem
#to check your cost function and gradients
#   CHECKCOSTFUNCTION(lambda) Creates a collaborative filering problem
#   to check your cost function and gradients, it will output the
#   analytical gradients produced by your code and the numerical gradients
#   (computed using computeNumericalGradient). These two gradient
#   computations should result in very similar values.


#Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

# Zap out most entries
    Y = X_t.dot(Theta_t.T)
    y_shape = Y.shape
    rand_mask = np.random.rand(y_shape[0],y_shape[1])
    mask = rand_mask>0.5

    Y[mask] = 0
    R = np.zeros(Y.shape)
    R[Y != 0] = 1

# Run Gradient Checking
    X = np.random.rand(X_t.shape[0],X_t.shape[1])
    Theta = np.random.rand(Theta_t.shape[0],Theta_t.shape[1])
    num_users = Y.shape[1]
    num_movies = Y.shape[0]
    num_features = Theta_t.shape[1]

#Cost func reference

    # Unroll parameters
    params = np.hstack((np.hstack((X.flatten(),Theta.flatten()))))

    # Short hand for cost function
    #cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_)
    costFunc = functools.partial(cofiCostFunc, Y = Y, R = R, num_users = num_users, num_movies = num_movies, num_features = num_features, lambda_ = 0)


    numgrad = computeNumericalGradient(costFunc, params)


#numgrad = computeNumericalGradient( ...
#                @(t) cofiCostFunc(t, Y, R, num_users, num_movies, ...
#                                num_features, lambda), [X(:); Theta(:)]);


    cost, grad = cofiCostFunc(params,  Y, R, num_users, num_movies, num_features, 0)

    print (np.vstack((numgrad, grad)).flatten('F').reshape(-1,2))
    print('The above two columns you get should be very similar.\n (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \nthe relative difference will be small (less than 1e-9). \n\nRelative Difference: n', diff)
