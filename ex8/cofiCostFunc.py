import numpy as np
def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, lambda_):
#COFICOSTFUNC Collaborative filtering cost function
#   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
#   num_features, lambda) returns the cost and gradient for the
#   collaborative filtering problem.


# Unfold the U and W matrices from params
	X = np.reshape(params[0:num_movies*num_features], (num_movies, num_features))
	Theta = np.reshape(params[num_movies*num_features:], (num_users, num_features))




# You need to return the following values correctly
	J = 0
	X_grad = 0
	Theta_grad = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost function and gradient for collaborative
#               filtering. Concretely, you should first implement the cost
#               function (without regularization) and make sure it is
#               matches our costs. After that, you should implement the
#               gradient and use the checkCostFunction routine to check
#               that the gradient is correct. Finally, you should implement
#               regularization.
#
# Notes: X - num_movies  x num_features matrix of movie features
#        Theta - num_users  x num_features matrix of user features
#        Y - num_movies x num_users matrix of user ratings of movies
#        R - num_movies x num_users matrix, where R(i, j) = 1 if the
#            i-th movie was rated by the j-th user
#
# You should set the following variables correctly:

#        X_grad - num_movies x num_features matrix, containing the
#                 partial derivatives w.r.t. to each element of X
#        Theta_grad - num_users x num_features matrix, containing the
#                     partial derivatives w.r.t. to each element of Theta

	h = X.dot(Theta.T)
	diff=h-Y  # m by 1  - result for each training set
	diff = np.multiply(diff, R)
	diff = diff**2
	J=np.sum(np.sum(diff))/2  + np.sum(np.sum(Theta**2))*lambda_/2 + np.sum(np.sum(X**2))*lambda_/2

	X_grad = (np.multiply((X.dot(Theta.T) - Y), R)).dot(Theta) + lambda_*X
	Theta_grad = (np.multiply((X.dot(Theta.T) - Y), R)).T.dot(X) + lambda_*Theta

	grad = np.hstack((X_grad.flatten(), Theta_grad.flatten()))
	return J,grad
