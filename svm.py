import numpy as np
import random
import math
import tester


def train(X, Y, eta, regulation):
    """
    svm algorithm used to form a linear classification/decision boundary between different classes.
    The objective of the support vector machine algorithm is to find a hyperplane/s
    in an N-dimensional space (N is the number of features) that distinctly classified the data points.
    svm algorithm find the hyperplane with maximum margin.
    the algorithm builds a model that can be used to predict new examples that the model never exposed to before.
    maximizing the margin distance provides some reinforcement so that future
    data points can be classified with more confidence.

    :param X: collection of examples (represented by features vector) from training set
    :param Y: collection of corresponding labels
    :param eta: fixed learning rate
    :return: returns a model which can be used for mapping new cases
    """
    classes_size = 3  # number of class
    samples_size = len(X)  # quantity of samples in training set
    features_size = len(X[0])  # quantity of features of each sample
    indexes = np.arange(0, samples_size)  # indexes vector used to pick random examples from samples set
    # weights matrix, start with all-zeroes weighted matrix
    w = np.zeros((classes_size, features_size))
    # the number of times to run through the training data while updating the weight.
    epochs = 10
    for e in range(epochs):
        # shuffle the data
        random.shuffle(indexes)
        for t in range(0, samples_size):
            eta = 1 / math.sqrt(t + 1)
            # choose id of random example from samples set
            i = indexes[t]
            # the prediction of model
            classes_confidence = np.dot(w, X[i])
            y_hat = np.argmax(classes_confidence)
            # the label of current sample
            y = int(Y[i])
            # if the model's prediction for this sample was wrong, updates the weighted matrix
            if y != y_hat:
                other = classes_size - y - y_hat  # get other class id
                # applies svm update rule on weights matrix
                w[y_hat] = w[y_hat] * (1 - eta * regulation) - eta * X[i]
                w[y] = w[y] * (1 - eta * regulation) + eta * X[i]
                w[other] = w[other] * (1 - eta * regulation)
            else:
                w[y] = w[y] * (1 - eta * regulation)
                second_prediction = secondmax(classes_confidence)
                other = 3 - y - second_prediction
                w[other] = w[other] * (1 - eta * regulation)
                if classes_confidence[y] < classes_confidence[second_prediction] + 1:
                    w[second_prediction] = w[second_prediction] * (1 - eta * regulation) - eta * X[i]
                else:
                    w[second_prediction] = w[second_prediction] * (1 - eta * regulation)

    # returns the model
    return w


def secondmax(vec):
    max_i = np.argmax(vec)
    vec[max_i] = -1000
    return np.argmax(vec)


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


def getBestModelPerShuffle(Train_X, Train_Y, Test_X, Test_Y, eta, regulation):
    instances = 10  # number of models to train
    min_error_rate = 1  # min error rate of models
    min_model = []  # instance of model which obtained the min error rate
    for i in range(0, instances):
        # builds new model by svm algorithm on training set
        w = train(Train_X, Train_Y, eta, regulation)
        # computes the error rate of model on validation set
        error_rate = test(w, Test_X, Test_Y)
        # in case the mode achieved the min error rate, mark the model as the best
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    # return the instance of model which obtained the min error rate
    return min_model, min_error_rate


def getBestModel(Train_X, Train_Y, samples_size, eta, regulation):
    shuffles_amount = 10
    min_model = []
    min_error_rate = 1
    for i in range(0, shuffles_amount):
        Train_X, Train_Y = tester.unison_shuffled_copies(Train_X, Train_Y)
        split_data = np.split(Train_X, [int(0.80 * samples_size), samples_size])
        split_label = np.split(Train_Y, [int(0.80 * samples_size), samples_size])
        model, error_rate = getBestModelPerShuffle(split_data[0], split_label[0], split_data[1], split_label[1], eta,
                                                   regulation)
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = model
    return min_model
