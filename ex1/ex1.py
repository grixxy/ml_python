from warmUpExercise import warmUpExercise
from plotData import plotData
from gradientDescent import gradientDescent
from computeCost import computeCost
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import linear_model



ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex1/ex1/'

# ==================== Part 1: Basic Function ====================

print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')
print(warmUpExercise())

#======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.loadtxt(ml_dir+'ex1data1.txt', delimiter=',')


X = data[:, 0]
y = data[:, 1]

m = np.size(y) #number of training examples
y = y.reshape(m,1)
X = X.reshape(m,1)



clf = linear_model.LinearRegression()
clf.fit(X,y)
print('Coefficients from scikit learn LinearRegression: \n',clf.intercept_,' ',clf.coef_)

clf = linear_model.Ridge (alpha = .5)
clf.fit(X,y)
print('Coefficients from scikit learn RidgeRegression: \n',clf.intercept_,' ',clf.coef_)

clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X,y)
print('Coefficients from scikit learn Lasso: \n',clf.intercept_,' ',clf.coef_)



# Plot Data
# Note: You have to complete the code in plotData.m
plotData(X, y);

# =================== Part 3: Gradient descent ===================
print('Running Gradient Descent ...\n')
#X = [ones(m, 1), data(:,1)]; # Add a column of ones to x
ones = np.ones((m,1))
X = np.hstack((ones,X))

theta = np.zeros((2, 1)) # initialize fitting parameters

# Some gradient descent settings
iterations = 1500;
alpha = 0.01;

# compute and display initial cost
computeCost(X, y, theta)

# run gradient descent
theta_J_Hist = gradientDescent(X, y, theta, alpha, iterations)
theta = theta_J_Hist[0]
J_Hist = theta_J_Hist[1]



# print theta to screen
print('Coefficients found by gradient descent: ');
print(theta)




#Plot the linear fit
plt.hold(True)  # keep previous plot visible
plt.plot(X[:,1], X.dot(theta), '-')


#plt.legend('Training data', 'Linear regression')
plt.hold(False) # don't overlay any more plots on this figure
plt.show()

#Cost function reduction
plt.plot(np.arange(0,iterations), J_Hist)
plt.xlabel('Iterations')
plt.ylabel('Cost')



# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
print('For population = 35,000, we predict a profit of \n',predict1*10000)

predict2 = np.array([1, 7]).dot(theta)
print('For population = 70,000, we predict a profit of \n',predict2*10000)

# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((np.size(theta0_vals), np.size(theta1_vals)))

# Fill out J_vals
for i in range(1,np.size(theta0_vals)):
    for j in range(1,np.size(theta1_vals)):
        t = [theta0_vals[i], theta1_vals[j]]
        J_vals[i,j] = computeCost(X, y, t)



# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T;
# Surface plot
fig = plt.figure();
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, rstride=1, cstride=1)
#surf(theta0_vals, theta1_vals, J_vals)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()

# Contour plot
fig = plt.figure()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100

plt.contour(theta0_vals, theta1_vals, J_vals, linespace=np.logspace(-2, 3, num=100))
plt.hold(True)
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.plot(theta[0], theta[1], 'rx')
plt.show()






