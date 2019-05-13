import sys
import numpy as np
import perceptron
import svm
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
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)
    # load labels
    label_set = np.genfromtxt(sys.argv[2], delimiter=',', dtype=float)

    # shuffle the data
    data_set, label_set = unison_shuffled_copies(data_set, label_set)

    samples_size = len(data_set)
    label_size = len(label_set)
    # split the data set to 80% training set and 20% to the test set
    split_data = np.split(data_set, [int(0.8 * samples_size), samples_size])
    split_label = np.split(label_set, [int(0.8 * label_size), label_size])

    w = perceptron.train(split_data[0], split_label[0])
    print(perceptron.test(w, split_data[1], split_label[1]))
    print("******************")
    w2 = svm.train(split_data[0], split_label[0], 0.005, 0.005)
    print(svm.test(w2, split_data[1], split_label[1]))


if __name__ == "__main__":
    main()
