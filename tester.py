import perceptron
import random
import numpy as np
import svm
import passive_aggressive as pa
import matplotlib.pyplot as plt

def test_peceptron(X, Y, testX, testY):
    test_amount = 15
    result = np.zeros(test_amount)
    values = np.zeros(test_amount)
    for i in range(0, test_amount):
        eta = round(random.uniform(0, 1),2)
        model = perceptron.train(X, Y, eta)
        error_rate = perceptron.test(model, testX, testY)
        result[i] = error_rate
        values[i] = eta
    draw_graph(result, values)

def test_svm(X, Y, testX, testY):
    test_amount = 15
    result = np.zeros(test_amount)
    values = np.zeros(test_amount)
    for i in range(0, test_amount):
        eta = round(random.uniform(0, 1),2)
        regulation = round(random.uniform(0, 1),2)
        model = svm.train(X, Y, eta, regulation)
        error_rate = svm.test(model, testX, testY)
        result[i] = error_rate
        values[i] = eta
        print eta, regulation, error_rate
    draw_graph(result, values)

def test_pa(X, Y, testX, testY):
    test_amount = 15
    result = np.zeros(test_amount)
    values = np.zeros(test_amount)
    for i in range(0, test_amount):
        model = pa.train(X, Y)
        error_rate = pa.test(model, testX, testY)
        result[i] = error_rate
        values[i] = i
        print error_rate
    draw_graph(result, values)

def TestModel(model, Test_X, Test_Y):
    test_amount = 10
    result = np.zeros(test_amount)
    values = np.arange(0,test_amount)
    for i in range(0, test_amount):
        error_rate = perceptron.test(model, Test_X, Test_Y)
        print error_rate
    #draw_graph(result, error_rate)

def draw_graph(results, titles):
    # x-coordinates of left sides of bars
    left = np.arange(0, len(results))

    # plotting a bar chart
    plt.rcParams.update({'font.size': 8})
    plt.bar(left, results, tick_label=titles,
            width=0.8, color=["blue", "green", "yellow","red"])

    # naming the x-axis
    plt.xlabel('x - axis')
    # naming the y-axis
    plt.ylabel('y - axis')
    # plot title
    plt.title('Result Chart')

    # function to show the plot
    plt.show()