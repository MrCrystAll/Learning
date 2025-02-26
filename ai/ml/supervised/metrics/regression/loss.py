import numpy as np


def squared_loss(prediction, reality):
    return np.sum(np.pow(reality - prediction, 2))

def mean_absolute_error(prediction, reality):
    return np.mean(np.abs(reality - prediction))

def mean_squared_error(prediction, reality):
    return np.mean(np.pow(reality - prediction, 2))

def mean_bias_error(prediction, reality):
    return np.mean(reality - prediction)

def relative_absolute_error(prediction, reality):
    top = np.sum(np.abs(reality - prediction))
    bottom = np.sum(np.abs(reality - reality.mean()))

    return top / bottom

def relative_squared_error(prediction, reality):
    top = np.sum(np.pow(reality - prediction, 2))
    bottom = np.sum(np.pow(reality - reality.mean(), 2))

    return top / bottom

def mean_absolute_percentage_error(prediction, reality):
    return np.sum(np.abs((reality - prediction) / reality))