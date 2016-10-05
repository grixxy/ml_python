import scipy.io as sio
import scipy.optimize as opt
import numpy as np
import functools
from ex3.predict import predictRight
from nnCostFunction import nnCostFunction
from sigmoidGradient import sigmoidGradient
from randInitializeWeights import randInitializeWeights
from checkNNGradients import checkNNGradients

#%% Machine Learning Online Class - Exercise 4 Neural Network Learning

#%% Setup the parameters you will use for this exercise
input_layer_size  = 400  #% 20x20 Input Images of Digits
hidden_layer_size = 25   #% 25 hidden units
num_labels = 10          #% 10 labels, from 1 to 10

'''
%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset.
%  You will be working with a dataset that contains handwritten digits.
%
'''


# Load Training Data
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex4/ex4/'

data = sio.loadmat(ml_dir+'ex4data1.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']

#y[y==10] = 0
y = y-1
#print(X[0:10,200:210])
#print(y[0:10, :])

print(X.shape)
print(y.shape)


m = X.shape[0]




#%% ================ Part 2: Loading Parameters ================
#% In this part of the exercise, we load some pre-initialized
#% neural network parameters.

# Load the weights into variables Theta1 and Theta2
data = sio.loadmat(ml_dir+'ex4weights.mat')
Theta1 = data['Theta1']
Theta2 = data['Theta2']
print(Theta1.shape)
print(Theta2.shape)

#% Unroll parameters
nn_params = np.hstack((Theta1.flatten(),Theta2.flatten()))




'''
%% ================ Part 3: Compute Cost (Feedforward) ================
%  To the neural network, you should first start by implementing the
%  feedforward part of the neural network that returns the cost only. You
%  should complete the code in nnCostFunction.m to return cost. After
%  implementing the feedforward to compute the cost, you can verify that
%  your implementation is correct by verifying that you get the same cost
%  as us for the fixed debugging parameters.
%
%  We suggest implementing the feedforward cost *without* regularization
%  first so that it will be easier for you to debug. Later, in part 4, you
%  will get to implement the regularized cost.
%
fprintf('\nFeedforward Using Neural Network ...\n')
'''


#% Weight regularization parameter (we set this to 0 here).
_lambda = 0

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights):(this value should be about 0.287629)\n', J)




# =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
# continue to implement the regularization with the cost.
#

#fprintf('\nChecking Cost Function (w/ Regularization) ... \n')

# Weight regularization parameter (we set this to 1 here).
_lambda = 1

J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

print('Cost at parameters (loaded from ex4weights): (this value should be about 0.383770)', J)


#% ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]\n ', g)



#% ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack((initial_Theta1.flatten(),initial_Theta2.flatten()))


# =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#

#fprintf('\nChecking Backpropagation... \n');

#  Check gradients by running checkNNGradients
checkNNGradients(0)


'''
%% =============== Part 8: Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%
'''

print('\nChecking Backpropagation (w/ Regularization) ... \n')

#  Check gradients by running checkNNGradients
_lambda = 3
checkNNGradients(_lambda)

# Also output the costFunction debugging values
debug_J  = nnCostFunction(nn_params, input_layer_size,hidden_layer_size, num_labels, X, y, _lambda)

print('\n\nCost at (fixed) debugging parameters (w/ lambda = 10): \n(this value should be about 0.576051)\n\n', debug_J[0])

'''
%% =================== Part 8: Training NN ===================
%  You have now implemented all the code necessary to train a neural
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
'''


options = {'disp': False, 'gtol': 1e-05, 'eps': 1.4901161193847656e-08, 'return_all': False, 'maxiter': 400}
#  You should also try different values of lambda
_lambda = 0.5

# Create "short hand" for the cost function to be minimized
costFunc = functools.partial(nnCostFunction, input_layer_size = input_layer_size, hidden_layer_size = hidden_layer_size, num_labels = num_labels, X = X, y = y, _lambda = _lambda)
theta = opt.minimize(costFunc,initial_nn_params,method='CG',jac = True, options=options)
Theta1 = theta.x[:(hidden_layer_size * (input_layer_size + 1))].reshape(hidden_layer_size, (input_layer_size + 1))
Theta2 = theta.x[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, (hidden_layer_size + 1))


#print(theta)

'''
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

#% Now, costFunction is a function that takes in only one argument (the
#% neural network parameters)




# Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Part 9: Visualize Weights =================
%  You can now "visualize" what the neural network is learning by
%  displaying the hidden units to see what features they are capturing in
%  the data.

fprintf('\nVisualizing Neural Network... \n')

displayData(Theta1(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================= Part 10: Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.
'''

pred = predictRight(Theta1, Theta2, X)

pred = pred.reshape(-1,1)
print('Training Set Accuracy is : ', np.mean(pred == y) * 100.)


