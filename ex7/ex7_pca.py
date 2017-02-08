import scipy.io as sio
import matplotlib.pyplot as plt
import math
from ex1.featureNormalize import featureNormalize
from pca import pca
from projectData import projectData
from recoverData import recoverData
from displayData import displayData
import scipy.misc as msc
from findClosestCentroids import findClosestCentroids
from computeCentroids import computeCentroids
from runkMeans import runkMeans
from kMeansInitCentroids import kMeansInitCentroids
from mpl_toolkits.mplot3d import Axes3D
import numpy as np


# Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.


# Initialization
#clear ; close all; clc

# ================== Part 1: Load Example Dataset  ===================
#  We start this exercise by using a small dataset that is easily to
#  visualize

print('Visualizing example dataset for PCA.\n\n')

#  The following command loads the dataset. You should now have the
#  variable X in your environment
#load ('ex7data1.mat');
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex7/ex7/'

data = sio.loadmat(ml_dir+'ex7data1.mat') # % training data stored in arrays X, y

X = data['X']

#  Visualize the example dataset
plt.plot(X[:, 0], X[:, 1], 'bo')
plt.axis([0.5, 6.5, 2, 8])



# =============== Part 2: Principal Component Analysis ===============
#  You should now implement PCA, a dimension reduction technique. You
#  should complete the code in pca.m
#
print('\nRunning PCA on example dataset.\n\n')

#  Before running PCA, it is important to first normalize X
[X_norm, mu, sigma] = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

#  Compute mu, the mean of the each feature

#  Draw the eigenvectors centered at mean of data. These lines show the
#  directions of maximum variations in the dataset.


p1 = mu + 1.5 * S[0] * U[:,0].T
p2  = mu + 1.5 * S[1] * U[:,1].T
plt.plot([mu[0],p1[0]], [mu[1],p1[1]])
plt.plot([mu[0],p2[0]], [mu[1],p2[1]])
#drawLine(mu, mu + 1.5 * S(1,1) * U(:,1)', '-k', 'LineWidth', 2);
#drawLine(mu, mu + 1.5 * S(2,2) * U(:,2)', '-k', 'LineWidth', 2);

print('Top eigenvector: \n')
print(' U(:,1) = \n', U[0,0], U[1,0])
print('\n(you should expect to see -0.707107 -0.707107)\n')

plt.show()


# =================== Part 3: Dimension Reduction ===================
#  You should now implement the projection step to map the data onto the
#   dimensional space.  This will show you what the data looks like when
#  using only the corresponding eigenvectors to reconstruct it.
#
#  You should complete the code in projectData.m

print('Dimension reduction on example dataset.\n\n')

#  Plot the normalized dataset (returned from pca)
plt.plot(X_norm[:, 0], X_norm[:, 1], 'bo')
plt.axis([-4 ,3, -4, 3])#; axis square

#  Project the data onto K = 1 dimension
K = 1
Z = projectData(X_norm, U, K)
print('Projection of the first example:\n', Z[0])
print('(this value should be about 1.481274)\n\n')

X_rec  = recoverData(Z, U, K)
print('Approximation of the first example: \n', X_rec[0, 0], X_rec[0, 1])
print('(this value should be about  -1.047419 -1.047419)\n\n');

#  Draw lines connecting the projected points to the original points
#hold on;
plt.plot(X_rec[:, 0], X_rec[:, 1], 'ro')
move_lines_x = [[c1[0], c2[0]] for (c1, c2) in zip(X_norm, X_rec)]
move_lines_y = [[c1[1], c2[1]] for (c1, c2) in zip(X_norm, X_rec)]
for i in range(X_norm.shape[0]):
    #drawLine(X_norm(i,:), X_rec(i,:), '--k', 'LineWidth', 1);
    #plt.plot(move_lines_x[i], move_lines_y[i])
    plt.plot([X_norm[i,0],X_rec[i,0]], [X_norm[i,1],X_rec[i,1]])
plt.show()


# =============== Part 4: Loading and Visualizing Face Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment
#
print('\nLoading face dataset.\n\n')

#  Load Face dataset
#load ('ex7faces.mat')
data = sio.loadmat(ml_dir+'ex7faces.mat') # % training data stored in arrays X, y

X = data['X']



#  Display the first 100 faces in the dataset
displayData(X[0:100, :])



# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
#  Run PCA and visualize the eigenvectors which are in this case eigenfaces
#  We display the first 36 eigenfaces.
#

print('\nRunning PCA on face dataset.This mght take a minute or two ...)\n\n')

#  Before running PCA, it is important to first normalize X by subtracting
#  the mean value from each feature
[X_norm, mu, sigma] = featureNormalize(X)

#  Run PCA
U, S, V = pca(X_norm)

#  Visualize the top 36 eigenvectors found
displayData(U[:, 0:36].T)


# ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors
#  If you are applying a machine learning algorithm
print('\nDimension reduction for face dataset.\n')

K = 100
Z = projectData(X_norm, U, K)

print('The projected data Z has a size of: ')
print(Z.shape)


# ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and
#  visualize only using those K dimensions
#  Compare to the original input, which is also displayed

print('Visualizing the projected (reduced dimension) faces.\n\n')

K = 100
X_rec  = recoverData(Z, U, K)

# Display normalized data
#subplot(1, 2, 1);
displayData(X_norm[0:100,:], 'Original faces')
#title('Original faces');
#axis square;

# Display reconstructed data from only k eigenfaces
#subplot(1, 2, 2);
displayData(X_rec[0:100,:],'Recovered faces' )
#title('Recovered faces');
#axis square;



# === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
#  One useful application of PCA is to use it to visualize high-dimensional
#  data. In the last K-Means exercise you ran K-Means on 3-dimensional
#  pixel colors of an image. We first visualize this output in 3D, and then
#  apply PCA to obtain a visualization in 2D.

#close all; close all; clc

# Re-load the image from the previous exercise and run K-Means on it
# For this to work, you need to complete the K-Means assignment first
#A = double(imread('bird_small.png'));

# If imread does not work for you, you can try instead
#   load ('bird_small.mat');

A = msc.imread(ml_dir+'bird_small.png')

A = A / 255
img_size = A.shape

X = A.reshape(img_size[0]*img_size[1],3)
#X = reshape(A, img_size(1) * img_size(2), 3);
K = 16
max_iters = 10
initial_centroids = kMeansInitCentroids(X, K)
[centroids, idx] = runkMeans(X, initial_centroids, max_iters)

#  Sample 1000 random indexes (since working with all the data is
#  too expensive. If you have a fast computer, you may increase this.
#sel = math.floor(math.rand(1000, 1) * X.shape[0]) + 1
sel = np.random.choice(X.shape[0], 1000)

#  Setup Color Palette
#palette = hsv(K);
#colors = palette(idx(sel), :);

colors = idx[sel]
#  Visualize the data and centroid memberships in 3D
#figure;
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
cax = ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=50, c=colors, cmap=cmhot)

#plt.scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
#plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')

plt.show()


# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Use PCA to project this cloud to 2D for visualization

# Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X)

# PCA and project the data to 2D
U, S, V = pca(X_norm)
Z = projectData(X_norm, U, 2)

# Plot in 2D
plt.scatter(Z[sel, 0], Z[sel, 1], s=50, c=colors, cmap=cmhot)
#plotDataPoints(Z[sel, :], idx(sel), K)
plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
plt.show()