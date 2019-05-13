import numpy as np
import random


# X: samples Y:labels
def train(X, Y, eta, regulation):
    samples_size = len(X)
    features_size = len(X[0])
    # weights matrix
    w = np.zeros((3, features_size))
    indexes = np.arange(0, samples_size)
    epochs = 10
    for e in range(epochs):
        # shuffle the data
        random.shuffle(indexes)
        for t in range(0, samples_size):
            # choose id of random example from samples set
            i = indexes[t]
            y_hat = np.argmax(np.dot(w, X[i]))
            y = int(Y[i])
            if y != y_hat:
                other = 3 - y - y_hat
                w[y_hat] = w[y_hat] * (1 - eta * regulation) - eta * X[i]
                w[y] = w[y] * (1 - eta * regulation) + eta * X[i]
                w[other] = w[other] * (1 - eta * regulation)
    return w


def test(w, X, Y):
    err = 0
    samples_size = len(X)
    for i in range(0, samples_size):
        y_hat = np.argmax(np.dot(w, X[i]))
        y = int(Y[i])
        if y_hat != y:
            err += 1
    return float(err) / samples_size


def predict(w, input):
    return np.argmax(w, input)
