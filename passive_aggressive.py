import numpy as np
import random
import math


# X: samples Y:labels
def train(X, Y):
    # quantity of samples in training set
    samples_size = len(X)
    # quantity of features of each sample
    features_size = len(X[0])
    classes = 3
    # weights matrix
    w = np.zeros((classes, features_size))
    indexes = np.arange(0, samples_size)
    epochs = 10
    for e in range(0, epochs):
        # shuffle the data
        random.shuffle(indexes)
        for t in range(0, samples_size):
            # choose id of random example from samples set
            i = indexes[t]
            # prediction of the model
            classes_value = np.dot(w, X[i])
            y_hat = np.argmax(classes_value)
            # label of current sample
            y = int(Y[i])
            if y_hat != y:
                norm_x = np.linalg.norm(X[i])
                loss = max(0, 1 - classes_value[y] + classes_value[y_hat])
                tau = loss / (2 * norm_x * norm_x)
                w[y_hat] = w[y_hat] - tau * X[i]
                w[y] = w[y] + tau * X[i]
    return w


def test(w, X, Y):
    err = 0  # error counter
    samples_size = len(X)
    for i in range(0, samples_size):
        y_hat = np.argmax(np.dot(w, X[i]))
        y = int(Y[i])
        if y_hat != y:
            err += 1
    return float(err) / samples_size


def predict(w, input):
    return np.argmax(w, input)


def norm(vector):
    sum = 0
    for x in vector:
        sum = sum + (x * x)
    return math.sqrt(sum)

def getBestModel(Train_X, Train_Y, Test_X, Test_Y):
    instances = 15
    min_error_rate = 1
    min_model = 0
    for i in range(0, instances):
        w = train(Train_X, Train_Y)
        error_rate = test(w, Test_X, Test_Y)
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    return min_model