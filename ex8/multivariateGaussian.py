import numpy as np
import math

def multivariateGaussian(X, mu, Sigma2):
    k = np.size(mu)

    if (len(Sigma2.shape)== 1) or (Sigma2.shape[1] == 1):
        Sigma2 = np.diag(Sigma2)

    X = X -  mu
    #denominator = (2 * math.pi) ** (k / 2) * np.linalg.det(Sigma2) ** 0.5
    denominator = (2 * math.pi) ** (- k / 2) * np.linalg.det(Sigma2) ** (-0.5)

#    f1 = X * pinv(Sigma2)
#    fprintf('size f1');
#    size(f1)
#    f2 = bsxfun( @ times, f1, X)
#    fprintf('size f2');
#    size(f2)

#    p = denom * exp(-0.5 * sum(f2, 2));

    sigma2_inv = np.linalg.pinv(Sigma2) #2x2
    f1 = X.dot(sigma2_inv)
    f2 = np.multiply(f1,X) #times
    f3 = np.sum(f2, axis=1)

    sp = np.exp(-0.5*f3)
    p =  denominator*sp
    return p