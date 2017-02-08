from sklearn.svm import SVC
import numpy as np

def dataset3Params(X, y, Xval, yval):
    #EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
    #where you select the optimal (C, sigma) learning parameters to use for SVM
    #with RBF kernel
    #   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and
    #   sigma. You should complete this function to return the optimal C and
    #   sigma based on a cross-validation set.
    #

    # You need to return the following variables correctly.
    #C = 1;
    #sigma = 0.3;

    C_temp = 0
    sigma_temp = 0
    error_temp = 0


    '''% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%
    '''

    testParams = [ 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]

    for i in range(0,len(testParams)):
        for j in range(0,len(testParams)):
            C_temp = testParams[i]
            sigma_temp = testParams[j]
            gamma = 1 / (2 * sigma_temp * sigma_temp)
            clf = SVC(kernel='rbf', C=C_temp, gamma=gamma)
            clf.fit(X, y.flatten())
            #model = svmTrain(X, y, C_temp, @(x1, x2) gaussianKernel(x1, x2, sigma_temp ));
            predictions = clf.predict(Xval)
            diff = predictions.flatten() != yval.flatten()
            error = np.mean(diff)
            print('current error = ',error)
            if(i==1 & j==1):
                error_temp = error
            elif(error < error_temp):
                error_temp = error
                C = C_temp
                sigma=sigma_temp
    return C, sigma