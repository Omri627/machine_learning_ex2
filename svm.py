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
            calculations = np.dot(w, X[i])
            y_hat = np.argmax(calculations)
            y = int(Y[i])
            if y != y_hat:
                if calculations[y_hat] - calculations[y] >= 1:
                    w[y_hat] = w[y_hat] + eta * [-1 * y_hat * X[i] + regulation * w[i]]
                    w[y] = w[y] - eta * [-1 * y_hat * X[i] + regulation * w[i]]
                else:
                    w[y_hat] = w[y_hat] + eta * regulation * w[y_hat]
                    w[y] = w[y] - eta * regulation * w[y_hat]
    return w


# X: samples Y:labels
def train(X, Y, eta, regulation):
    samples_size = len(X)
    features_size = len(X[0])
    # weights matrix
    w = np.zeros((3, features_size))
    for t in range(0, samples_size):
        # choose id of  random example from samples set
        i = random.randint(0, samples_size - 1)
        calculations = np.dot(w, X[i])
        y_hat = np.argmax(calculations)
        y = int(Y[i])
        if y != y_hat and abs(calculations[y] - calculations[y_hat]) >= 1:
            w[y_hat] = w[y_hat] - eta * [-1 * y_hat * X[i] + regulation * w[y_hat]]
            w[y] = w[y] + eta * [-1 * y_hat * X[i] + regulation * w[y_hat]]
        elif y != y_hat and abs(calculations[y] - calculations[y_hat]) < 1:
            w[y_hat] = w[y_hat] - eta * X[i] * w[y_hat]
            w[y] = w[y] + eta * X[i] * w[y]
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
