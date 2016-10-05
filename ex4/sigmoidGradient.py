from ex2.costFunction import *

def sigmoidGradient(z):
    zig = sigmoid(z)
    g = zig*(1-zig)
    return g


