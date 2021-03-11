import numpy as np
import math


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh(x):
    return np.tanh(x)


def tanh_derivate(X):
    return 1 - (tanh(X) ** 2)


def relu(x):
    return x * (x > 0)


def relu_derivative(X):
    X[X <= 0.0] = 0.0
    X[X > 0.0] = 1.0
    return X


def softmax(X):
    X = np.exp(X)
    sum = np.sum(X, axis=0)
    return X / sum


def squared_loss(y, y_hat):
    if y.shape != y_hat.shape:
        raise Exception('Two arrays should be of same shape')
    return np.sum(np.square(y - y_hat))


# def cross_entropy(y, y_hat):
#     s = 0.0
#     for y_i, y_hat_i in zip(y, y_hat):
#         s += y_i * math.log(y_hat_i + 1e-35)
#     return 0 if s==0 else -s

def cross_entropy(yhat, y_train, k):
    eIndicator = np.zeros((10, k))
    eIndicator[y_train, np.arange(k)] = 1
    eIndicator = eIndicator * yhat
    eIndicator = eIndicator.sum(axis=0)
    eIndicator = np.log(eIndicator)
    return -sum(eIndicator)
