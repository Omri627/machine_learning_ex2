import numpy as np
import random
# X: samples Y:labels
def train(X, Y):
    samples_size = len(X)
    features_size = len(X[0])
    eta = 0.1
    # weights matrix
    w = np.zeros((3, features_size))
    for t in range(0, samples_size):
        # choose id of  random example from samples set
        i = random.randint(0, samples_size - 1)
        np.argmax(w, X[i])

    data_set = zip(X, Y)
