import numpy as np

def emailFeatures(word_indices, vocab_len):
    x = np.zeros((vocab_len, 1))
    x[word_indices] = 1
    return x