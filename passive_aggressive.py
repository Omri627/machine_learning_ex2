import numpy as np
import random
import math

def train(X, Y):
    """
    passive aggressive algorithm is online algorithm used to form a linear classification/decision boundary
    between different classes. passive-aggressive is supervised learning algorithm which analyzes
    the training data and produces a model which can generalize new cases and be used for mapping new examples.

    :param X: collection of examples (represented by features vector) from training set
    :param Y: collection of corresponding labels
    :return: returns a model which can be used for mapping new cases
    """
    classes_size = 3                        # number of classes
    samples_size = len(X)                   # quantity of samples in training set
    features_size = len(X[0])               # quantity of features of each sample
    indexes = np.arange(0, samples_size)    # indexes vector used to pick random examples from samples set
    # start with the all-zeroes weight matrix
    w = np.zeros((classes_size, features_size))
    for i in range(0, classes_size):
        for j in range(0, features_size):
            w[i][j] = 0.1
    # the number of times to run through the training data while updating the weight.
    epochs = 10
    for e in range(0, epochs):
        # shuffle the data
        random.shuffle(indexes)
        for t in range(0, samples_size):
            # choose id of random example from samples set
            i = indexes[t]
            classes_value = np.dot(w, X[i])
            # prediction of the model
            y_hat = np.argmax(classes_value)
            # label of current sample
            y = int(Y[i])
            # if the model's prediction for this sample was wrong, updates the weighted matrix
            if y_hat != y:
                # compute tau: adaptive learning rate
                norm_x = np.linalg.norm(X[i])
                loss = max(0, 1 - classes_value[y] + classes_value[y_hat])
                tau = loss / (2 * norm_x * norm_x)
                # applies multi-class passive-aggressive update rule
                w[y_hat] = w[y_hat] - tau * X[i]
                w[y] = w[y] + tau * X[i]
    # returns the model represented by weights matrix
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
    instances = 15              # number of models to train
    min_error_rate = 1          # min error rate of models
    min_model = []              # instance of model which obtained the min error rate
    for i in range(0, instances):
        # builds new model by svm algorithm on training set
        w = train(Train_X, Train_Y)
        # computes the error rate of model on validation set
        error_rate = test(w, Test_X, Test_Y)
        # in case the mode achieved the min error rate, mark the model as the best
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    # return the instance of model which obtained the min error rate
    return min_model
