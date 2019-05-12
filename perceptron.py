import numpy as np
import random
# X: samples Y:labels
def train(X, Y):
    samples_size = len(X)
    features_size = len(X[0])
    eta = 0.1
    # weights matrix
    w = np.zeros((3, features_size))
    for t in range(0, samples_size):
        # choose id of  random example from samples set
        i = random.randint(0, samples_size - 1)
        x = np.dot(w, X[i])
        y_hat = np.argmax(x)
        y = Y[i]
        if (y_hat != y):
            w[y_hat] = w[y_hat] - eta * X[i]
            w[y] = w[y] + eta * X[i]

    return w