import scipy.io as sio
from displayData import displayData
from oneVsAll import oneVsAll
import scipy.optimize as opt
from ex2.costFunction import *
from predictOneVsAll import *

import numpy as np

'''
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  linear exercise. You will need to complete the following functions
%  in this exericse:
%
%     lrCostFunction.m (logistic regression cost function)
%     oneVsAll.m
%     predictOneVsAll.m
%     predict.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%
'''

# Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

#%% =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
#fprintf('Loading and Visualizing Data ...\n')
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex3/ex3/'

data = sio.loadmat(ml_dir+'ex3data1.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']

y[y==10] = 0
#print(X[0:10,200:210])
#print(y[0:10, :])

print(X.shape)
print(y.shape)


m = X.shape[0]

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m);
#sel = X[rand_indices[1:101], :]
#sel_Y = y[rand_indices[1:101], :]
sel = X[1:101, :]
sel_Y = y[1:101, :]

print(y)
displayData(sel,sel_Y)

'''
%% ============ Part 2: Vectorize Logistic Regression ============
%  In this part of the exercise, you will reuse your logistic regression
%  code from the last exercise. You task here is to make sure that your
%  regularized logistic regression implementation is vectorized. After
%  that, you will implement one-vs-all classification for the handwritten
%  digit dataset.
%
'''

#fprintf('\nTraining One-vs-All Logistic Regression...\n')

lambda_ = 0.1

all_theta = oneVsAll(X, y, num_labels, lambda_)



#%% ================ Part 3: Predict for One-Vs-All ================
#%  After ...
pred = predictOneVsAll(all_theta, X)
pred = pred.reshape(-1,1)
print('Training Set Accuracy is : ', np.mean(pred == y) * 100.)

# Randomly select 100 data points to display
rand_indices = np.random.permutation(m);

X_train = X[rand_indices[:4500], :]
y_train = y[rand_indices[:4500], :]
X_test = X[rand_indices[4501:], :]
y_test = y[rand_indices[4501:], :]

all_theta = oneVsAll(X_train, y_train, num_labels, lambda_)
pred = predictOneVsAll(all_theta, X_test)
pred = pred.reshape(-1,1)
print('Training Set Accuracy is : ', np.mean(pred == y_test) * 100.)

