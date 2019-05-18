import numpy as np
import random
import math
import tester
import ex2


def train(X, Y, samples_size):
    """
    passive aggressive algorithm is online algorithm used to form a linear classification/decision boundary
    between different classes. passive-aggressive is supervised learning algorithm which analyzes
    the training data and produces a model which can generalize new cases and be used for mapping new examples.
    :param X: collection of examples (represented by features vector) from training set
    :param Y: collection of corresponding labels
    :return: returns a model which can be used for mapping new cases
    """
    classes_size = 3  # number of classes
    features_size = len(X[0])  # quantity of features of each sample
    indexes = np.arange(0, samples_size)  # indexes vector used to pick random examples from samples set
    # start with the all-zeroes weight matrix
    w = np.zeros((classes_size, features_size))
    # the number of times to run through the training data while updating the weight.
    epochs = 20
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
                if norm_x == 0:
                    tau = 0
                else:
                    loss = max(0, 1 - classes_value[y] + classes_value[y_hat])
                    tau = loss / (2 * norm_x * norm_x)
                # applies multi-class passive-aggressive update rule
                w[y_hat] = w[y_hat] - tau * X[i]
                w[y] = w[y] + tau * X[i]
    # returns the model represented by weights matrix
    return w


def predict(w, x):
    """
    predict method gets the model and example
    and predicts his label using the model
    :param w: model trained by passive-aggressive algorithm
    :param x: example to predict his label
    :return: returns the label of example predicted by model
    """
    return np.argmax(w, x)


def norm(vector):
    """
    computes the norm of the given vector
    :param vector: specific vector
    :return: return real number specifies the norm of the vector
    """
    sum = 0
    # computes the summation of features squared of the vector
    for x in vector:
        sum = sum + (x * x)
    return math.sqrt(sum)


def getBestModelPerShuffle(train_X, train_Y, test_X, test_Y):
    """
    getBestModelPerShuffle method trains fixed amount of models on training set using perceptron algorithm
    for each model the method computes the error rate
    and pick the best model i.e the model which gains the minimum error rate on validation set
    :param train_X: set of examples in training set
    :param train_Y: set of labels in training set
    :param validation_X: set of examples of validation set
    :param validation_Y: set of labels of validation set
    :param samples_size number of samples in training set
    :return: returns the best model which gains the minimum error rate
    """
    instances = 10  # number of models to train
    min_error_rate = 1  # min error rate of models
    min_model = []  # instance of model which obtained the min error rate
    train_amount = len(train_X)  # amount of samples in training set
    test_amount = len(test_X)  # amount of samples in test set
    for i in range(0, instances):
        # builds new model by svm algorithm on training set
        w = train(train_X, train_Y, train_amount)
        # computes the error rate of model on validation set
        error_rate = tester.computeErrorRate(w, test_X, test_Y, test_amount)
        # in case the mode achieved the min error rate, mark the model as the best
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = w
    # return the instance of model which obtained the min error rate
    return min_model, min_error_rate


def getBestModel(train_X, train_Y, samples_size):
    """
    getBestModel method shuffles the training set and divides/splits the whole-training set
    into two sets: train set and validation set.
    for each shuffle the method runs getBestModelPerShuffle
    that trains fixed amount of models and pick the model which gains the minimum error rate on validation set
    the method, eventually choose the best model over the optimal models trained on different shuffles
    :param train_X: set of examples of training set
    :param train_Y: set of labels of training set
    :param samples_size: the amount of samples on training set
    :return: returns the best model over the optimal models trained on different shuffles
    """
    training_percentage = 0.80  # percentage of samples used for training
    shuffles_amount = 5  # amount of times of dividing the training set
    min_model = []  # model which gains the min error rate
    min_error_rate = 1  # minimum error rate of models
    for i in range(0, shuffles_amount):
        # shuffle the set of samples
        train_X, train_Y = ex2.unison_shuffled_copies(train_X, train_Y, samples_size)
        # split the data and labels of given samples
        split_data = np.split(train_X, [int(training_percentage * samples_size), samples_size])
        split_label = np.split(train_Y, [int(training_percentage * samples_size), samples_size])
        # get best model for this shuffle
        model, error_rate = getBestModelPerShuffle(split_data[0], split_label[0], split_data[1], split_label[1])
        # in case this is the best model
        if error_rate < min_error_rate:
            min_error_rate = error_rate
            min_model = model
    # return the min model i.e the model which gains the minimum error-rate
    return min_model
