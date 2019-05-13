import numpy as np
import random
import math

# X: samples Y:labels
def train(X, Y, eta):
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
            normalized_x = normalize(X[i])
            tau = np.zeros(3)
            for k in range(0, 3):
                tau[k] = (max(0, 1 - y_hat * w[k] * X[i])) / (2 * normalized_x * normalized_x)
            if y_hat != y:
                w[y_hat] = w[y_hat] - tau[y_hat] * X[i]
                w[y] = w[y] + tau[y] * X[i]
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

def normalize(vector):
    sum = 0
    for x in vector:
        sum = sum + x * x
    return math.sqrt(sum)