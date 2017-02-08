def projectData(X, U, K):
    U_reduce = U[:, 0:K]
    Z = X.dot(U_reduce)
    return Z