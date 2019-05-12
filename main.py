import sys
import pandas as pd
import numpy as np
from scipy import __all__ as sc


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
    return sc.stats.mstats.zscore(data)


def main():
    header = ['Sex', 'Length', 'Diameter', 'Height', 'W weight', 'S weight', 'V weight', 'Shell weight']
    df = pd.read_csv(sys.argv[1], names=header)  # df = data frame
    data_arr = np.array(df.iloc[:, 0:])  # return all the indexes, all rows from 0 to 3286 and columns from 0 to all
    encoding(data_arr)
    data_arr = z_score(data_arr)
    print(data_arr)


if __name__ == "__main__":
    main()
