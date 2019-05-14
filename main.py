import sys
import numpy as np
import tester
import perceptron
import svm
import passive_aggressive as pa
from scipy import stats


def one_hot_encoding(data):
    encode = {
        "M": np.array([0, 0, 1]),
        "F": np.array([0, 1, 0]),
        "I": np.array([1, 0, 0])
    }
    for i, row in enumerate(data):
        # transform it into vector
        data[i][0] = encode[row[0]]


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


def main():
    # read data from csv to data frame
    data_arr = np.genfromtxt(sys.argv[1], delimiter=',', dtype="|U5")
    # change the M F I to 0 1 2
    encoding(data_arr)

    # normalize the data-set
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)

    # load labels file
    label_set = np.genfromtxt(sys.argv[2], delimiter=',', dtype=float)

    # shuffle the data
    data_set, label_set = unison_shuffled_copies(data_set, label_set)

    # split the data set to 80% training set and 20% to the test set
    samples_size = len(data_set)
    label_size = len(label_set)
    split_data = np.split(data_set, [int(0.7 * samples_size), samples_size])
    split_label = np.split(label_set, [int(0.7 * label_size), label_size])
    tester.test_pa(split_data[0], split_label[0], split_data[1], split_label[1])
    """
    w = perceptron.train(split_data[0], split_label[0], 0.1)
    print(perceptron.test(w, split_data[1], split_label[1]))
    print("******************")
    w2 = svm.train(split_data[0], split_label[0], 0.1, 0.5)
    print(svm.test(w2, split_data[1], split_label[1]))
    print("******************")
    w3 = pa.train(split_data[0], split_label[0])
    print(pa.test(w3, split_data[1], split_label[1]))
    """

if __name__ == "__main__":
    main()
