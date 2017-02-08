import scipy.io as sio
import matplotlib.pyplot as plt
import scipy.optimize as opt
import numpy as np
from sklearn.svm import SVC
from plotData import plotData
from visualizeBoundary import visualizeBoundaryLinear, visualizeBoundary
from gaussianKernel import gaussianKernel
from dataset3Params import dataset3Params

'''
%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% =============== Part 1: Loading and Visualizing Data ================
%  We start the exercise by first loading and visualizing the dataset.
%  The following code will load the dataset into your environment and plot
%  the data.
%
'''

ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex6/ex6/'

data = sio.loadmat(ml_dir+'ex6data1.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']


#plotData(X, y)

#==================== Part 2: Training Linear SVM ====================
#  The following code will train a linear SVM on the dataset and plot the
#  decision boundary learned.


print('\nTraining Linear SVM ...\n')

# You should try to change the C value below and see how the decision
# boundary varies (e.g., try C = 1000)
C = 1

clf = SVC(kernel='linear')
clf.fit(X, y.flatten())



#model = svmTrain(X, y, C, @linearKernel, 1e-3, 20);
#visualizeBoundaryLinear(X, y, clf)

#fprintf('Program paused. Press enter to continue.\n');



#% =============== Part 3: Implementing Gaussian Kernel ===============
#  You will now implement the Gaussian kernel to use
#  with the SVM. You should complete the code in gaussianKernel.m
#
print('\nEvaluating the Gaussian Kernel ...\n')

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2
sim = gaussianKernel(x1, x2, sigma)



print('Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ,
         '\n\t%f\n(this value should be about 0.324652)\t', sim)


#%% =============== Part 4: Visualizing Dataset 2 ================
#%  The following code will load the next dataset into your environment and
#%  plot the data.


# Load from ex6data2:
# You will have X, y in your environment
#load('ex6data2.mat');

data = sio.loadmat(ml_dir+'ex6data2.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']

# Plot training data
#plotData(X, y)


# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
print('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

# SVM Parameters
C = 1
sigma = 0.01
gamma = 1/(2*sigma*sigma)

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

clf = SVC(kernel='rbf', C=C, gamma=gamma)

#model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
clf.fit(X, y.flatten())
visualizeBoundary(X, y, clf)


#%% =============== Part 6: Visualizing Dataset 3 ================
#%  The following code will load the next dataset into your environment and
#%  plot the data.


#fprintf('Loading and Visualizing Data ...\n')

# Load from ex6data3:
# You will have X, y in your environment
#load('ex6data3.mat');
data = sio.loadmat(ml_dir+'ex6data3.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']

Xval = data['Xval']
yval = data['yval']

# Plot training data
plotData(X, y)



# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.


# Try different SVM Parameters here
[C, sigma] = dataset3Params(X, y, Xval, yval);
gamma = 1/(2*sigma*sigma)
# Train the SVM
#model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));


clf = SVC(kernel='rbf', C=C, gamma=gamma)
clf.fit(X, y.flatten())
visualizeBoundary(X, y, clf)
print('C=', C, ' sigma=',sigma)