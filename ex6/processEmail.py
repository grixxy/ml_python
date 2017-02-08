import pandas as pd
import re
import nltk
from nltk.stem.porter import *


def processEmail(email_contents):
    # Load Vocabulary
    vocabulary = getVocabList()

    # Init return value
    word_indices = []
    email_contents = ''.join(email_contents)
    email_contents = email_contents.lower()

    email_contents = re.sub('<[^<>]+>',' ', email_contents)

    email_contents = re.sub('[0-9]+', 'number', email_contents)

    email_contents = re.sub('(http|https)://[^\s]*', 'httpaddr', email_contents)

    email_contents = re.sub('[^\s]+@[^\s]+', 'emailaddr', email_contents)

    email_contents = re.sub('[$]+', 'dollar', email_contents)

    tokens = nltk.word_tokenize(email_contents)

    stemmer = PorterStemmer()


    for word in tokens:
        word = re.sub('[^a-zA-Z0-9]', '', word)

        word = stemmer.stem(word)

        if len(word) > 1:
            if word in vocabulary:
                word_indices.append(vocabulary[word])
    return word_indices


def getVocabList():
    #GETVOCABLIST reads the fixed vocabulary list in vocab.txt and returns a
    #cell array of the words
    #   vocabList = GETVOCABLIST() reads the fixed vocabulary list in vocab.txt
    #   and returns a cell array of the words in vocabList.

    ml_dir = '/Users/gregory/Desktop/me/coursera/machine_learning/ml_python/machine-learning-ex6/ex6/'
    fname = ml_dir + 'vocab.txt'
    df = pd.read_csv(fname, delimiter='\t', header=None)
    df.columns = ['index', 'word']
    dic  = df.set_index('word')['index'].to_dict()
    return dic