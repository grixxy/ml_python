def recoverData(Z, U, K):
    U_reduce = U[:, 0:K]
    X_rec= Z.dot(U_reduce.T)
    return X_rec