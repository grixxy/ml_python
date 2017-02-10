import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from cofiCostFunc import cofiCostFunc
from checkCostFunction import checkCostFunction
from normalizeRatings import normalizeRatings
import pandas as pd
import scipy.optimize as opt
import functools

# Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:

#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m

#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# =============== Part 1: Loading movie ratings dataset ================
#  You will start by loading the movie ratings dataset to understand the
#  structure of the data.

print('Loading movie ratings dataset.\n\n')

#  Load data
#load ('ex8_movies.mat');
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex8/ex8/'

data = sio.loadmat(ml_dir+'ex8_movies.mat') # % training data stored in arrays X, y
Y = data['Y']
R = data['R']


#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on
#  943 users

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  From the matrix, we can compute statistics like average rating.
print('Average rating for movie 1 (Toy Story): / 5\n\n', np.mean(Y[0, R[0, :]]))

#  We can "visualize" the ratings matrix by plotting it with imagesc
plt.imshow(Y)
plt.ylabel('Movies')
plt.xlabel('Users')
plt.show()


# ============ Part 2: Collaborative Filtering Cost Function ===========
#  You will now implement the cost function for collaborative filtering.
#  To help you debug your cost function, we have included set of weights
#  that we trained on that. Specifically, you should complete the code in
#  cofiCostFunc.m to return J.

#  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
#load ('ex8_movieParams.mat');
data = sio.loadmat(ml_dir+'ex8_movieParams.mat')
Theta = data['Theta']
X = data['X']


#  Reduce the data set size so that this runs faster
num_users = 4
num_movies = 5
num_features = 3
X = X[0:num_movies, 0:num_features]
Theta = Theta[0:num_users, 0:num_features]
Y = Y[0:num_movies, 0:num_users]
R = R[0:num_movies, 0:num_users]




#  Evaluate cost function

J,grad = cofiCostFunc(np.hstack((X.flatten(),Theta.flatten())), Y, R, num_users, num_movies, num_features, 0)

print('Cost at loaded parameters: \n(this value should be about 22.22)\n', J)


# ============== Part 3: Collaborative Filtering Gradient ==============
#  Once your cost function matches up with ours, you should now implement
#  the collaborative filtering gradient function. Specifically, you should
#  complete the code in cofiCostFunc.m to return the grad argument.

print('\nChecking Gradients (without regularization) ... \n')

#  Check gradients by running checkNNGradients
checkCostFunction(0)


# ========= Part 4: Collaborative Filtering Cost Regularization ========
#  Now, you should implement regularization for the cost function for
#  collaborative filtering. You can implement it by adding the cost of
#  regularization to the original cost computation.


#  Evaluate cost function
J = cofiCostFunc(np.hstack((X.flatten(),Theta.flatten())), Y, R, num_users, num_movies, num_features, 1.5)

print('Cost at loaded parameters (lambda = 1.5): \n(this value should be about 31.34)\n', J[0])


# ======= Part 5: Collaborative Filtering Gradient Regularization ======
#  Once your cost matches up with ours, you should proceed to implement
#  regularization for the gradient.



print('\nChecking Gradients (with regularization) ... \n')

#  Check gradients by running checkNNGradients
checkCostFunction(1.5)


# ============== Part 6: Entering ratings for a new user ===============
#  Before we will train the collaborative filtering model, we will first
#  add ratings that correspond to a new user that we just observed. This
#  part of the code will also allow you to put in your own ratings for the
#  movies in our dataset!
#

movieList = pd.read_csv(ml_dir+'movie_ids.txt', delimiter=r"^[^\s]*\s" ,header = None)
movieList = movieList.values[:,1]

#  Initialize my ratings
my_ratings = np.zeros((1682,))

# Check the file movie_idx.txt for id of each movie in our dataset
# For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings[0] = 1

# Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings[97] = 2

# We have selected a few movies we liked / did not like and the ratings we
# gave are as follows:
my_ratings[96] = 5
my_ratings[6] = 1
my_ratings[11]= 1
my_ratings[53] = 1
my_ratings[63]= 1
my_ratings[65]= 1
my_ratings[68] = 4
my_ratings[182] = 1
my_ratings[225] = 1
my_ratings[354]= 1

my_ratings[466]= 5
my_ratings[520]= 5
my_ratings[522]= 5
my_ratings[549]= 5
my_ratings[567]= 5
my_ratings[575]= 5
my_ratings[688]= 5
my_ratings[55]= 5
my_ratings[1088]= 5
my_ratings[54]= 5



print('\n\nNew user ratings:\n')
for i in np.arange(my_ratings.shape[0]):
    if my_ratings[i] > 0:
        print('Rated ', my_ratings[i], 'for ',movieList[i])



# ================== Part 7: Learning Movie Ratings ====================
#  Now, you will train the collaborative filtering model on a movie rating
#  dataset of 1682 movies and 943 users


print('\nTraining collaborative filtering...\n')
#load('ex8_movies.mat');
data = sio.loadmat(ml_dir+'ex8_movies.mat') # % training data stored in arrays X, y
Y = data['Y']
R = data['R']

#  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
#  943 users

#  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
#  rating to movie i

#  Add our own ratings to the data matrix
Y = np.hstack((my_ratings.reshape(-1,1),Y))
mask  = my_ratings!=0
R = np.hstack((mask.reshape(-1,1),R))
#R = [(my_ratings ~= 0) R];


#  Normalize Ratings
Ynorm, Ymean = normalizeRatings(Y, R)


#  Useful Values
num_users = Y.shape[1]
num_movies = Y.shape[0]
num_features = 10

# Set Initial Parameters (Theta, X)
X = np.random.rand(num_movies, num_features)
Theta = np.random.rand(num_users, num_features)

initial_parameters = np.hstack((np.hstack((X.flatten(),Theta.flatten()))))


# Set options for fmincg
#options = optimset('GradObj', 'on', 'MaxIter', 100);

# Set Regularization
lambda_ = 10
#theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
#                                num_features, lambda)), ...
#                initial_parameters, options);

#costFunc = functools.partial(nnCostFunction, input_layer_size = input_layer_size, hidden_layer_size = hidden_layer_size, num_labels = num_labels, X = X, y = y, _lambda = _lambda)
costFunc = functools.partial(cofiCostFunc, Y=Y, R=R, num_users=num_users, num_movies=num_movies,
                             num_features=num_features, lambda_=0)

options = {'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': 500}

theta = opt.minimize(costFunc, initial_parameters, method='CG',jac = True, options=options)
theta = theta.x
# Unfold the returned theta back into U and W

X = np.reshape(theta[0:num_movies * num_features], (num_movies, num_features))
Theta = np.reshape(theta[num_movies * num_features:], (num_users, num_features))

print('Recommender system learning completed.\n')


# ================== Part 8: Recommendation for you ====================
#  After training the model, you can now make recommendations by computing
#  the predictions matrix.


p = X.dot(Theta.T)
my_predictions = p[:,0].reshape(-1,1) + Ymean


idx = np.argsort(my_predictions.flatten())[::-1]
print('\nTop recommendations for you:\n')
for i in range(10):
    j = idx[i]
    print('Predicting rating ', my_predictions[j,0],' for movie ', movieList[j])


print('\n\nOriginal ratings provided:\n')
for i in range(my_ratings.shape[0]):
    if my_ratings[i] > 0:
        print('Rated', my_ratings[i],' for ', movieList[i])