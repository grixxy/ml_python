from costFunction import sigmoid

def predict(theta, X):
    return 0.5 <= sigmoid(X.dot(theta))
