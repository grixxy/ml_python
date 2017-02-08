import scipy.io as sio
import numpy as np
from sklearn.svm import SVC
from processEmail import processEmail, getVocabList
from emailFeatures import emailFeatures

# Machine Learning Online Class
#  Exercise 6 | Spam Classification with SVMs
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

# ==================== Part 1: Email Preprocessing ====================
#  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
#  to convert each email into a vector of features. In this part, you will
#  implement the preprocessing steps for each email. You should
#  complete the code in processEmail.m to produce a word indices vector
#  for a given email.

ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex6/ex6/'
fname = ml_dir + 'emailSample1.txt'
with open(fname) as f:
    file_contents = f.readlines()

# Extract Features
word_indices  = processEmail(file_contents)

# Print Stats
print('Word Indices: \n')
print(word_indices)


# ==================== Part 2: Feature Extraction ====================
#  Now, you will convert each email into a vector of features in R^n.
#  You should complete the code in emailFeatures.m to produce a feature
#  vector for a given email.

# Extract Features
#file_contents = readFile('emailSample1.txt');
#word_indices  = processEmail(file_contents);
vocab_length = len(getVocabList())
features      = emailFeatures(word_indices, vocab_length)

# Print Stats
print('Length of feature vector: ', np.size(features))
print('Number of non-zero entries: ', np.sum(features > 0))



# =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to determine if an
#  email is Spam or Not-Spam.

# Load the Spam Email dataset
# You will have X, y in your environment
#load('spamTrain.mat');

ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex6/ex6/'

data = sio.loadmat(ml_dir+'spamTrain.mat') # % training data stored in arrays X, y
X = data['X']
y = data['y']


#fprintf('\nTraining Linear SVM (Spam Classification)\n')
#fprintf('(this may take 1 to 2 minutes) ...\n')

C = 1

clf = SVC(kernel='linear')
clf.fit(X, y.flatten())


p = clf.predict(X)
accuracy = np.mean(p.flatten() == y.flatten())

print('Training Accuracy: ', accuracy * 100);


#C = 1
#sigma = 0.01
#gamma = 1/(2*sigma*sigma)

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

#rbf_clf = SVC(kernel='rbf', C=C, gamma=gamma)
#rbf_clf.fit(X, y.flatten())
#p = rbf_clf.predict(X)
#accuracy = np.mean(p.flatten() == y.flatten())

#print('Training Accuracy RBF: ', accuracy * 100);


# =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have
#  included a test set in spamTest.mat

# Load the test dataset
# You will have Xtest, ytest in your environment
#load('spamTest.mat');
data_test = sio.loadmat(ml_dir+'spamTest.mat') # % training data stored in arrays X, y
Xtest = data_test['Xtest']
ytest = data_test['ytest']

p = clf.predict(Xtest)
#p_rbf = rbf_clf.predict(Xtest)

accuracy = np.mean(p.flatten() == ytest.flatten())
#accuracy_rbf = np.mean(p_rbf.flatten() == ytest.flatten())

print('Test Accuracy: ', accuracy * 100)
#print('Test Accuracy RBF: ', accuracy_rbf * 100)


# ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
#  weights learned by the model to understand better how it is determining
#  whether an email is spam or not. The following code finds the words with
#  the highest weights in the classifier. Informally, the classifier
#  'thinks' that these words are the most likely indicators of spam.


# Sort the weights and obtin the vocabulary list
weights = np.zeros((vocab_length,), dtype=[('idx', int), ('weight', float)])
weights['idx'] = np.arange(vocab_length)
weights['weight'] = clf.coef_[0]
weights.sort(order='weight')
weights = weights[::-1]

vocabList = getVocabList()

print('Top predictors of spam: ')
for i in range(15):
    for word, indx in vocabList.items():
        if indx == weights['idx'][i]:
            print(word, ' ', weights['weight'][i])


# =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples.
#  The following code reads in one of these emails and then uses your
#  learned SVM classifier to determine whether the email is Spam or
#  Not Spam

# Set the file to be read in (change this to spamSample2.txt,
# emailSample1.txt or emailSample2.txt to see different predictions on
# different emails types). Try your own emails as well!
ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex6/ex6/'
fname = ml_dir + 'spamSample_my.txt'
with open(fname) as f:
    file_contents = f.readlines()



# Read and predict
word_indices  = processEmail(file_contents)
x_             = emailFeatures(word_indices, vocab_length)
x_ = x_.reshape(1,-1)
p = clf.predict(x_)

print('Processed Spam Classification: ', p)
print('(1 indicates spam, 0 indicates ham)')

