import perceptron
import random
import numpy as np
import svm
import passive_aggressive as pa
import matplotlib.pyplot as plt


def print_results(m_percepton, m_svm, m_pa, test_X):
    samples_size = len(test_X)
    for i in range(0, samples_size):
        y_perc = np.argmax(np.dot(m_percepton, test_X[i]))
        y_svm = np.argmax(np.dot(m_svm, test_X[i]))
        y_pa = np.argmax(np.dot(m_pa, test_X[i]))
        print "perceptron: ", y_perc, ", svm: ", y_svm, ", pa: ", y_pa


def test_peceptron(X, Y, validateX, validateY,  testX, testY):
    test_amount = 15
    result = np.zeros(test_amount)
    values = np.zeros(test_amount)
    sum = 0
    for i in range(0, test_amount):
        eta = round(random.uniform(0, 1), 2)
        model = perceptron.getBestModel(X, Y, validateX, validateY, 0.1)
        error_rate = perceptron.test(model, testX, testY)
        result[i] = error_rate
        values[i] = eta
        sum = sum + error_rate
    sum = sum / 15
    print sum
    draw_graph(result, values)

def test_svm(X, Y,  testX, testY):
    test_amount = 3
    result = np.zeros(test_amount)
    values = np.zeros(test_amount)
    sample_size = len(X)
    sum = 0
    for i in range(0, test_amount):
        eta = round(random.uniform(0, 1), 2)
        regulation = round(random.uniform(0, 1), 2)
        model = svm.getBestModel(X, Y, sample_size ,eta, 0.25)
        error_rate = svm.test(model, testX, testY)
        result[i] = error_rate
        values[i] = regulation
        sum = sum + error_rate
    sum = sum / test_amount
    print sum
    draw_graph(result, values)


def test_pa(X, Y, validateX, validateY,  testX, testY):
    test_amount = 15
    result = np.zeros(test_amount)
    values = np.zeros(test_amount)
    for i in range(0, test_amount):
        eta = round(random.uniform(0, 1), 2)
        model = pa.getBestModel(X, Y, validateX, validateY, 0.1)
        error_rate = pa.test(model, testX, testY)
        result[i] = error_rate
        values[i] = eta
    draw_graph(result, values)



def TestModel(model, Test_X, Test_Y):
    test_amount = 10
    result = np.zeros(test_amount)
    values = np.arange(0, test_amount)
    for i in range(0, test_amount):
        error_rate = perceptron.test(model, Test_X, Test_Y)
        print(error_rate)
    # draw_graph(result, error_rate)


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

    # function to show the plot
    plt.show()
