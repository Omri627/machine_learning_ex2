import perceptron
import random
import numpy as np
import svm
import passive_aggressive as pa
import matplotlib.pyplot as plt


def computeErrorRate(model, test_X, test_Y, samples_size):
    """
    computeErrorRate gets a model and test set which the model never exposed before
    the method computes the error rate of the model over the test set.
    :param model:
    :param test_X:
    :param test_Y:
    :param samples_size:
    :return: returns real number in range [0,1] indicate the error rate of given model
    The lower the value, the better the model described the data
    """
    err = 0                                                 # error counter
    for i in range(0, samples_size):
        # the prediction of the model
        y_hat = np.argmax(np.dot(model, test_X[i]))
        y = int(test_Y[i])
        # in case the prediction is not the guinene label of this example
        if y_hat != y:
            err += 1            # increase the amount of errors
    # return the percentage of errors on test set
    return float(err) / samples_size

def print_results(m_percepton, m_svm, m_pa, test_X, samples_size):
    """
    print_results method gets the best model of each algorithm perceptron, svm and passive aggressive and test set.
    the method iterate through test-set and print out the prediction of given models for each sample

    :param m_percepton: model trained by perceptron algorithm
    :param m_svm: model trained by svm algorithm
    :param m_pa: model trained by passive-aggressive algorithm
    :param test_X: test set
    :param samples_size: the amount of samples in test set
    :return: none
    """
    # iterate through samples in test set
    for i in range(0, samples_size):
        # compute the prediction of each model
        y_perceptron = np.argmax(np.dot(m_percepton, test_X[i]))
        y_svm = np.argmax(np.dot(m_svm, test_X[i]))
        y_pa = np.argmax(np.dot(m_pa, test_X[i]))
        # prints out results
        print "perceptron: ", y_perceptron, ", svm: ", y_svm, ", pa: ", y_pa

def test_perceptron(X, Y,  testX, testY):
    """
    test perceptron method trains fixed amount of models using perceptron full algorithm.
    for each model computes the error rate
    and prints out bar graph filled with this information
    additionally print out the average error rate of models

    :param X: set of samples of training set
    :param Y: set of labels  of training set
    :param testX: set of samples of test set
    :param testY: set of labels of test set
    :return:
    """
    test_amount = 5                                 # amount models to train
    result = np.zeros(test_amount)                  # result vector of perceptron models test
    values = np.arange(0, test_amount)              # vector identification of tests
    samples_size = len(X)                           # number of samples in training set
    test_size = len(testX)                          # number of samples in test set
    # summation of error rate of all tests, used to computes average
    sum = 0
    for i in range(0, test_amount):
        # train model using perceptron algorithm
        model = perceptron.getBestModel(X, Y, samples_size)
        # computes error rate over test set
        error_rate = computeErrorRate(model, testX, testY, test_size)
        result[i] = error_rate          # store error rate value in result vector
        sum = sum + error_rate          # sum up error rate results
        print error_rate
    # prints out average of error rate
    print sum / test_amount
    # draw bar graph filled with this information
    draw_graph(result, values)

def test_svm(X, Y,  testX, testY):
    """
    test_svm method trains fixed amount of models using svm full algorithm.
    for each model computes the error rate
    and prints out bar graph filled with this information
    and additionally the average error rate of models

    :param X: set of samples of training set
    :param Y: set of labels  of training set
    :param testX: set of samples of test set
    :param testY: set of labels of test set
    :return: none
    """
    test_amount = 5                                 # amount models to train
    result = np.zeros(test_amount)                  # result vector storing the error rate of svm models
    values = np.zeros(test_amount)                  # regulation constant used in each model
    samples_size = len(X)                           # size of samples in training set
    test_size = len(testX)
    # summation of error rate of all tests, used to computes average
    sum = 0
    for i in range(0, test_amount):
        # get a random regulation in range [0,1]
        #regulation = round(random.uniform(0, 1), 2)
        # train model using svm algorithm
        model = svm.getBestModel(X, Y, samples_size, 0.25)
        # computes error rate over test set
        error_rate = computeErrorRate(model, testX, testY, test_size)
        # storing data about test
        result[i] = error_rate
        values[i] = i
        print error_rate
        # sum up error rates value
        sum = sum + error_rate
    # prints out average error rate of all models
    sum = sum / test_amount
    print sum
    # draw a graph filled with this information
    draw_graph(result, values)


def test_pa(X, Y,  testX, testY):
    """
    test_pa method trains fixed amount of models using passive-aggressive full algorithm.
    for each model computes the error rate
    and prints out bar graph filled with this information
    and additionally the average error rate of models

    :param X: set of samples of training set
    :param Y: set of labels  of training set
    :param testX: set of samples of test set
    :param testY: set of labels of test set
    :return: none
    """
    test_amount = 5                                 # amount models to train
    result = np.zeros(test_amount)                  # result vector of perceptron models test
    values = np.arange(0, test_amount)              # vector identification of tests
    samples_size = len(X)                           # number of samples in training set
    test_size = len(testX)                          # number of samples in test set
    # summation of error rate of all tests, used to computes average
    sum = 0
    for i in range(0, test_amount):
        # train model using svm algorithm
        model = pa.getBestModel(X, Y, samples_size)
        error_rate = computeErrorRate(model, testX, testY, test_size)
        # store error rate of each model
        result[i] = error_rate
        # sum up error rates of models
        sum = sum + error_rate
        print error_rate
    # computes the average of error rates
    sum = sum / test_amount
    print sum
    # draw a graph filled with this information
    draw_graph(result, values)

def draw_graph(results, titles):
    # x-coordinates of left sides of bars
    left = np.arange(0, len(results))

    # plotting a bar chart
    plt.rcParams.update({'font.size': 8})
    plt.bar(left, results, tick_label=titles,
            width=0.8, color=["blue", "green", "yellow", "red"])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('Result Chart')

    # show the plot
    plt.show()

