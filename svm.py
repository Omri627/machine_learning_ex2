import numpy as np
import random
import math
import tester
import ex2


def train(X, Y, samples_size, regulation):
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
    features_size = len(X[0])  # quantity of features of each sample
    indexes = np.arange(0, samples_size)  # indexes vector used to pick random examples from samples set
    # weights matrix, start with all-zeroes weighted matrix
    w = np.zeros((classes_size, features_size))
    # the number of times to run through the training data while updating the weight.
    epochs = 15
    for e in range(epochs):
        # shuffle the data
        random.shuffle(indexes)
        for t in range(0, samples_size):
            eta = 1 / math.sqrt((e + 1) * (t + 1))
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
                # applies update rule when the model predicted the label correctly
                w[y] = w[y] * (1 - eta * regulation)
                second_prediction = secondmax(classes_confidence)
                other = 3 - y - second_prediction
                w[other] = w[other] * (1 - eta * regulation)
                # find the second maximum prediction
                # if the second prediction confidence was close to the right prediction
                # that means the point is in the margin, so we make the model less right about this sample
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


def predict(w, x):
    """
    predict method gets the model and example
    and predicts his label using the model
    :param w: model trained by svm
    :param x: example to predict his label
    :return: returns the label of example predicted by model
    """
    return np.argmax(np.dot(w, x))


def getBestModelPerShuffle(train_X, train_Y, validation_X, validation_Y, regulation):
    """
    getBestModelPerShuffle method trains fixed amount of models on training set using svm algorithm
    for each model the method computes the error rate
    and pick the best model i.e the model which gains the minimum error rate on validation set
    :param train_X: set of examples in training set
    :param train_Y: set of labels in training set
    :param validation_X: set of examples of validation set
    :param validation_Y: set of labels of validation set
    :param regulation: fixed regulation constant
    :return: returns the best model which gains the minimum error rate
    """
    instances = 15  # number of models to train
    min_error_rate = 1  # min error rate of models
    min_model = []  # instance of model which obtained the min error rate
    train_amount = len(train_X)  # amount of samples in training set
    validation_amount = len(validation_X)  # amount of samples in test set
    for i in range(0, instances):
        # builds new model by svm algorithm on training set
        w = train(train_X, train_Y, train_amount, regulation)
        # computes the error rate of model on validation set
        error_rate = tester.computeErrorRate(w, validation_X, validation_Y, validation_amount)
        # in case the mode achieved the min error rate, mark the model as the best
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    # return the instance of model which obtained the min error rate
    return min_model, min_error_rate


def getBestModel(train_X, train_Y, samples_size, regulation):
    """
    getBestModel method shuffles the training set and divides/splits the whole-training set
    into two sets: train set and validation set.
    for each shuffle the method runs getBestModelPerShuffle
    that trains fixed amount of models and pick the model which gains the minimum error rate on validation set
    the method, eventually choose the best model over the optimal models trained on different shuffles
    :param train_X: set of examples of training set
    :param train_Y: set of labels of training set
    :param samples_size: the amount of samples on training set
    :param regulation:
    :return:
    """
    training_percentage = 0.80  # percentage of samples used for training
    shuffles_amount = 10  # amount of times of dividing the training set
    min_model = []  # model which gains the min error rate
    min_error_rate = 1  # minimum error rate of models
    for i in range(0, shuffles_amount):
        # shuffle the set of samples
        train_X, train_Y = ex2.unison_shuffled_copies(train_X, train_Y, samples_size)
        # split the data and labels of given samples
        split_data = np.split(train_X, [int(training_percentage * samples_size), samples_size])
        split_label = np.split(train_Y, [int(training_percentage * samples_size), samples_size])
        # get best model for this shuffle
        model, error_rate = getBestModelPerShuffle(split_data[0], split_label[0], split_data[1], split_label[1],
                                                   regulation)
        # in case this is the best model
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = model
    # return the min model i.e the model which gains the minimum error-rate
    return min_model
