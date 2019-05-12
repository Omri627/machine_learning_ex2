import sys
import pandas as pd
import numpy as np
import perceptron
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
        "M": 0,
        "F": 1,
        "I": 2
    }
    for i, row in enumerate(data):
        # transform it into vector
        data[i][0] = encode[row[0]]


def z_score(data):
    data = np.array(data)
    return stats.mstats.zscore(data)


def main():
    # header for the data
    header = ['Sex', 'Length', 'Diameter', 'Height', 'W weight', 'S weight', 'V weight', 'Shell weight']
    # read data from csv to data frame
    df = pd.read_csv(sys.argv[1], names=header)  # df = data frame
    # transfer the data to numpy array
    data_arr = np.array(df.iloc[:, 0:],
                        dtype="|U5")  # return all the indexes, all rows from 0 to 3286 and columns from 0 to all
    # change the M F I to 0 1 2
    encoding(data_arr)
    data_set = np.array(data_arr, dtype=float)
    data_set = z_score(data_set)
    df = pd.read_csv(sys.argv[2], header=None)
    label_set = np.array(df.iloc[:, :], dtype=float)
    perceptron.train(data_set, label_set)


if __name__ == "__main__":
    main()
