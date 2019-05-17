import sys
import numpy as np
import tester
import perceptron
import svm
import passive_aggressive as pa
from scipy import stats

def unison_shuffled_copies(samples, labels, samples_size):
    """
    unison_shuffled_copies method gets samples and labels sets
    and shuffle them together with the same permutation.
    i.e the sample and his corresponding label stored with the same indentation in samples, labels arrays respectively
    :param samples: set of samples
    :param labels: set of labels
    :return: returns shuffled vectors of samples and labels
    """
    assert len(samples) == len(labels)
    p = np.random.permutation(samples_size)
    return samples[p], labels[p]

def one_hot_encoding(data, samples_size):
    """
    One hot encoding is a process by which categorical variables are converted into a form
    that could be provided to machine learning algorithms to do a better job in prediction.
    Categorical data are variables that contain label values rather than numeric values.
    Each label values gets integer representation.
    a one-hot encoding can be applied to the integer representation.
    the label variable is removed from samples features and instead
    a feature is added for each label value.
    the method applies 1 for the feature with the original label value of current sample.

    :param data: sets of samples
    :return: returns the set of samples after described modifications
    """
    arr = np.zeros((samples_size, 3))
    for i, row in enumerate(data):
        if row[0] == 'M':
            arr[i][0] = 1
        elif row[0] == 'F':
            arr[i][1] = 1
        else:
            arr[i][2] = 1
    data2 = np.array(data[:, 1:])
    data2 = np.concatenate((data2, arr), axis=1)
    return data2


def base_encoding(data):
    """
    each unique category value is assigned an integer value.

    :param data: set of samples
    :return: returns set of samples after described modifications
    """
    encode = {
        "M": 1,
        "F": 2,
        "I": 3
    }
    for i, row in enumerate(data):
        # transform it into vector
        data[i][0] = encode[row[0]]


def z_score(data):
    """
    the method normalized the samples set using z_score algorithm
    Normalization is a technique applied as part of data preparation for machine learning.
    The goal of normalization is to change the values of numeric columns in the dataset to a common scale,
    without distorting differences in the ranges of values.

    :param data: set of samples
    :return: returns the set of samples after normalization process
    """
    data = np.array(data)
    return stats.mstats.zscore(data)


def min_max_normalization(data):
    """
    the method normalized the samples set using min_max normalization algorithm
    Normalization is a technique applied as part of data preparation for machine learning.
    The goal of normalization is to change the values of numeric columns in the dataset to a common scale,
    without distorting differences in the ranges of values.

    :param data: set of samples
    :return: returns the data after normalization process
    """
    for i, x in enumerate(data):
        data[i] = ((x - np.min(x)) / (np.max(x) - np.min(x)))


def ex2_tester():
    # read data from csv to data frame
    data_arr = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    # applies one hot enconding over samples set
    samples_size = len(data_arr)        # number of samples in training set
    data_arr = one_hot_encoding(data_arr, samples_size)

    # read labels training set data
    test_x = np.genfromtxt(sys.argv[3], delimiter=',', dtype="|U5")
    test_size = len(test_x)             # number of samples in test set
    test_x = one_hot_encoding(test_x, test_size)
    test_x = np.array(test_x, dtype=float)
    test_x = z_score(test_x)
    # normalize the data-set
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)

    # load labels file
    label_set = np.genfromtxt(sys.argv[2], delimiter=',', dtype=float)
    test_y = np.genfromtxt(sys.argv[4], delimiter=',', dtype=float)

    # test passive-aggressive algorithm
    tester.test_perceptron(data_set, label_set, test_x, test_y)
    tester.test_svm(data_set, label_set, test_x, test_y)
    tester.test_pa(data_set, label_set, test_x, test_y)

def main():

    # read data from csv to data frame
    data_arr = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    # applies one hot enconding over samples set
    samples_size = len(data_arr)        # number of samples in training set
    data_arr = one_hot_encoding(data_arr, samples_size)

    # read labels training set data
    test_x = np.genfromtxt(sys.argv[3], delimiter=',', dtype="|U5")
    test_size = len(test_x)             # number of samples in test set
    test_x = one_hot_encoding(test_x, test_size)
    test_x = np.array(test_x, dtype=float)
    test_x = z_score(test_x)
    # normalize the data-set
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)

    # load labels file
    label_set = np.genfromtxt(sys.argv[2], delimiter=',', dtype=float)

    # split the data set to 80% training set and 20% to the test set
    # trained models with perceptron, svm and passive aggressive algorithm
    regulation_constant = 0.25              # regulation constant value
    m_perc = perceptron.getBestModel(data_set, label_set, samples_size)
    m_svm = svm.getBestModel(data_set, label_set, samples_size, regulation_constant)
    m_pa = pa.getBestModel(data_set, label_set, samples_size)

    # prints out the prediction of given models for each sample in test set
    tester.print_results(m_perc, m_svm, m_pa, test_x, test_size)


if __name__ == "__main__":
    main()
