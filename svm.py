import numpy as np
import random


# X: samples Y:labels
def train(X, Y, eta, regulation):
    classes = 3
    samples_size = len(X)
    features_size = len(X[0])
    # weights matrix
    w = np.zeros((classes, features_size))
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
                other = classes - y - y_hat
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
        print "prediction:", y_hat
        print "real result:", y
    return float(err) / samples_size

def predict(w, input):
    return np.argmax(w, input)

def getBestModel(Train_X, Train_Y, Test_X, Test_Y, eta, regulation):
    instances = 15
    min_error_rate = 1
    min_model = 0
    for i in range(0, instances):
        w = train(Train_X, Train_Y, eta, regulation)
        error_rate = test(w, Test_X, Test_Y)
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    return min_model