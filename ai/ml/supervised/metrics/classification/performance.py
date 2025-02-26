import numpy as np


def accuracy(predicted, reality):
    return np.sum((predicted == reality)) / np.shape(predicted)[0]