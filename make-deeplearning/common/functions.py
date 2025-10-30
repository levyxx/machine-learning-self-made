import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    if a.ndim == 1:
        a = a.reshape(1, -1)
    
    c = np.max(a, axis=1, keepdims=True)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a

    return y


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size





