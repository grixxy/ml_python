import numpy as np


from ex2.costFunction import *
from sigmoidGradient import sigmoidGradient

def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):

    theta1 = nn_params[:(hidden_layer_size * (input_layer_size +1))].reshape(hidden_layer_size, (input_layer_size+1))
    theta2 = nn_params[(hidden_layer_size * (input_layer_size +1)):].reshape(num_labels, (hidden_layer_size+1))
    #print(theta1.shape)
    #print(theta2.shape)



    # Setup some useful variables
    m = X.shape[0]

    # You need to return the following variables correctly
    J = 0
    theta1_grad = np.zeros(np.size(theta1))
    theta2_grad = np.zeros(np.size(theta2))
    '''
    % ====================== YOUR CODE HERE ======================
    % Instructions: You should complete the code by working through the
    %               following parts.
    %
%  Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
    '''

    X = np.hstack((np.ones((m, 1)), X))


    #Calculating z2 5000 by 25
    #X 5000 401 Theta1 25 x 401 Theta2 10 x 26
    z2 = X.dot(theta1.T) # %5000 x 25
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((a2.shape[0], 1)), a2))
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3) #5000 x 10

    labels_transform = np.eye(a3.shape[1])
    y_new = labels_transform[y[:,0],:] #5000x x 10

    s = 0

    #costFuntions
    y_new_flat = y_new.reshape(1,-1)
    h_flat = a3.reshape(-1,1)
    s = (-y_new_flat).dot(np.log(h_flat))
    s = s-(1-y_new_flat).dot(np.log(1-h_flat))
    J = (s/m)[0][0]


    unbias_Theta1 = theta1[:, 1:theta1.shape[1]]
    unbias_Theta2 = theta2[:, 1:theta2.shape[1]]

    #regularizing cost function

    regularizator_cost = _lambda/(2*m)*(sum(sum(unbias_Theta1**2))+sum(sum(unbias_Theta2**2)))

    J=s/m+regularizator_cost


    delta3 = a3-y_new # 5000 x 10
    delta2 = delta3.dot(theta2) #5000 x 26
    delta2 = delta2[:, 1:delta2.shape[1]] #%5000 x 25

    delta2 = delta2 * sigmoidGradient(z2) #5000 x 25

    DEL1 = 0
    DEL2 = 0


    DEL1 = delta2.T.dot(X) # 25 X 401 - ok
    DEL2  = delta3.T.dot(a2)#; % 10 X 26   vs Theta 10 x 26
    DEL1 = DEL1/m
    DEL2  = DEL2/m

    theta1_regul = np.zeros((unbias_Theta1.shape[0],1))
    theta1_regul = np.hstack((theta1_regul, unbias_Theta1))
    theta2_regul = np.zeros((unbias_Theta2.shape[0],1))
    theta2_regul = np.hstack((theta2_regul, unbias_Theta2))



    theta1_grad = DEL1+(_lambda/m)*theta1_regul
    theta2_grad = DEL2+(_lambda/m)*theta2_regul




    grad = np.hstack((theta1_grad.flatten(), theta2_grad.flatten()))
    #print(J[0][0])
    return J[0][0], grad

