import numpy as np
import random
import math
import tester

def train(X, Y, eta):
    """
    perceptron algrithm used to form a linear classification/decision boundary between different classes.
    perceptron is supervised learning algorithm which analyzes the training data and produces
    a model which can generalize new cases and be used for mapping new examples

    :param X: collection of examples (represented by features vector) from training set
    :param Y: collection of corresponding labels
    :param eta: learning rate
    :param regulation:
    :return: returns a model which can be used for mapping new cases
    """
    samples_size = len(X)                   # number of samples in data set
    features_size = len(X[0])               # number of features of input data
    w = np.zeros((3, features_size))        # start with the all-zeroes weight matrix
    indexes = np.arange(0, samples_size)    # indexes vector used to pick random examples from samples set
    # the number of times to run through the training data while updating the weight.
    epochs = 50
    for e in range(epochs):
        # shuffle the data-set
        random.shuffle(indexes)
        for t in range(0, samples_size):
            eta = 1 / math.sqrt((e + 1) * (t + 1))
            # choose id of random example from samples set
            i = indexes[t]
            # prediction of model
            y_hat = np.argmax(np.dot(w, X[i]))
            # label of current sample
            y = int(Y[i])
            # if the model's prediction for this sample was wrong, updates the weighted matrix
            if y_hat != y:
                # applies perceptron update rule on weights matrix
                w[y_hat] = w[y_hat] - eta * X[i]
                w[y] = w[y] + eta * X[i]
    # returns the model represented by weights matrix
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

def getBestModelPerShuffle(Train_X, Train_Y, Test_X, Test_Y, eta):
    instances = 15              # number of models to train
    min_error_rate = 1          # min error rate of models
    min_model = []              # instance of model which obtained the min error rate
    for i in range(0, instances):
        # builds new model by perceptron algorithm on training set
        w = train(Train_X, Train_Y, eta)
        # computes the error rate of model on validation set
        error_rate = test(w, Test_X, Test_Y)
        # in case the mode achieved the min error rate, mark the model as the best
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    # return the instance of model which obtained the min error rate
    return min_model

def getBestModel(Train_X, Train_Y, samples_size , eta):
    shuffles_amount = 15
    min_model = []
    min_error_rate = 1
    for i in range(0, shuffles_amount):
        Train_X, Train_Y = tester.unison_shuffled_copies(Train_X, Train_Y)
        split_data = np.split(Train_X, [int(0.80 * samples_size), samples_size])
        split_label = np.split(Train_Y, [int(0.80 * samples_size), samples_size])
        model, error_rate = getBestModelPerShuffle(split_data[0], split_label[0], split_data[1], split_label[1], eta)
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = model
    return min_model

