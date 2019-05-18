import sys
import numpy as np
import tester
import perceptron
import svm
import passive_aggressive as pa
from scipy import stats


def one_hot_encoding(data):
    arr = np.zeros((len(data), 3))
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


def encoding(data):
    encode = {
        "M": 1,
        "F": 2,
        "I": 3
    }
    for i, row in enumerate(data):
        # transform it into vector
        data[i][0] = encode[row[0]]


def z_score(data):
    data = np.array(data)
    return stats.mstats.zscore(data)


def min_max_normalization(data):
    for i, x in enumerate(data):
        data[i] = ((x - np.min(x)) / (np.max(x) - np.min(x)))


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def real_time():
    # read data from csv to data frame
    data_arr = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    # change the M F I to vectors
    data_arr = one_hot_encoding(data_arr)

    test_x = np.genfromtxt(sys.argv[3], delimiter=',', dtype="|U5")
    test_x = one_hot_encoding(test_x)
    test_x = np.array(test_x, dtype=float)
    test_x = z_score(test_x)
    # normalize the data-set
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)

    # load labels file
    label_set = np.genfromtxt(sys.argv[2], delimiter=',', dtype=float)

    # split the data set to 80% training set and 20% to the test set
    samples_size = len(data_set)

    m_perc = perceptron.getBestModel(data_set, label_set, samples_size, 0.1)

    m_svm = svm.getBestModel(data_set, label_set, samples_size, 0.1, 0.25)

    m_pa = pa.getBestModel(data_set, label_set, samples_size)

    tester.print_results(m_perc, m_svm, m_pa, test_x)


def main():

    # read data from csv to data frame
    data_arr = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    # change the M F I to 0 1 2
    data_arr = one_hot_encoding(data_arr)

    test_x = np.genfromtxt(sys.argv[3], delimiter=',', dtype="|U5")
    test_x = one_hot_encoding(test_x)
    test_x = np.array(test_x, dtype=float)
    test_x = z_score(test_x)
    # normalize the data-set
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)

    # load labels file
    label_set = np.genfromtxt(sys.argv[2], delimiter=',', dtype=float)
    test_y = np.genfromtxt(sys.argv[4], delimiter=',', dtype=float)

    # shuffle the data
    #data_set, label_set = unison_shuffled_copies(data_set, label_set)

    # split the data set to 80% training set and 20% to the test set
    #samples_size = len(data_set)
    #label_size = len(label_set)
    #split_data = np.split(data_set, [int(0.80 * samples_size), samples_size])
    #split_label = np.split(label_set, [int(0.80 * label_size), label_size])

    #tester.test_pa(data_set, label_set, test_x, test_y)
    tester.test_svm(data_set,label_set, test_x, test_y)
    #tester.test_perceptron(data_set,label_set, test_x, test_y)


    """
    m_perc = perceptron.getBestModel(split_data[0], split_label[0], split_data[1], split_label[1], 0.1)
    error_rate = perceptron.test(m_perc, test_x, test_y)
    print(error_rate)
    print("******************")

    m_svm = svm.getBestModel(split_data[0], split_label[0], split_data[1], split_label[1], 0.1, 0.5)
    error_rate = svm.test(m_svm, test_x, test_y)
    print(error_rate)
    print("******************")
    m_pa = pa.getBestModel(split_data[0], split_label[0], split_data[1], split_label[1])
    error_rate = pa.test(m_pa, test_x, test_y)
    print(error_rate)

    print("******************")
    #tester.print_results(m_perc, m_svm, m_pa, test_x)
    """

    real_time()


if __name__ == "__main__":
    main()
