import numpy as np
from nnCostFunction import nnCostFunction
from computeNumericalGradient import computeNumericalGradient
import functools

def checkNNGradients(_lambda):
#CHECKNNGRADIENTS Creates a small neural network to check the
#backpropagation gradients
#   CHECKNNGRADIENTS(lambda) Creates a small neural network to check the
#   backpropagation gradients, it will output the analytical gradients
#   produced by your backprop code and the numerical gradients (computed
#   using computeNumericalGradient). These two gradient computations should
#   result in very similar values.
#

    if '_lambda' not in locals():
        _lambda = 0;


    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # We generate some 'random' test data
    Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
    Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    X  = debugInitializeWeights(m, input_layer_size - 1)
    y  = np.mod(np.arange(0,m), num_labels).reshape(-1,1)


    # Unroll parameters
    nn_params = np.hstack((Theta1.flatten(), Theta2.flatten()))

    # Short hand for cost function
    costFunc = functools.partial(nnCostFunction, input_layer_size = input_layer_size, hidden_layer_size = hidden_layer_size, num_labels = num_labels, X = X, y = y, _lambda = _lambda)
    #cost, grad = nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
    #(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)
    cost, grad  = costFunc(nn_params)
    numgrad = computeNumericalGradient(costFunc, nn_params)


# Visually examine the two gradient computations.  The two columns
# you get should be very similar.

    print (np.vstack((numgrad, grad)))
    print('The above two columns you get should be very similar.\n (Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n')

# Evaluate the norm of the difference between two solutions.
# If you have a correct implementation, and assuming you used EPSILON = 0.0001
# in computeNumericalGradient.m, then diff below should be less than 1e-9
    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)

    print('If your backpropagation implementation is correct, then \nthe relative difference will be small (less than 1e-9). \nRelative Difference:', diff)





###########################
def debugInitializeWeights(fan_out, fan_in):
#DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
#incoming connections and fan_out outgoing connections using a fixed
#strategy, this will help you later in debugging
#   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
#   of a layer with fan_in incoming connections and fan_out outgoing
#   connections using a fix set of values
#
#   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
#   the first row of W handles the "bias" terms
#

    _shape = (fan_out, 1 + fan_in)
    size = _shape[0]*_shape[1]

# Initialize W using "sin", this ensures that W is always of the same
# values and will be useful for debugging
    W = np.sin(np.arange(0,size))/10.
    W = W.reshape(_shape)
    return W
