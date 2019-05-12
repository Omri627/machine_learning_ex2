import sys
import pandas as pd
import numpy as np


def one_hot_encoding(data):
    encode = {
        "M": np.array([0, 0, 1]),
        "F": np.array([0, 1, 0]),
        "I": np.array([1, 0, 0])
    }
    for i, row in enumerate(data):
        # transform it into vector
        data[i][0] = encode[row[0]]


header = ['Sex', 'Length', 'Diameter', 'Height', 'W weight', 'S weight', 'V weight', 'Shell weight']
df = pd.read_csv(sys.argv[1], names=header)  # df = data frame
dataArr = np.array(df.iloc[:, 0:])  # return all the indexes, all rows from 0 to 3286 and columns from 0 to all
one_hot_encoding(dataArr)
print(dataArr)
