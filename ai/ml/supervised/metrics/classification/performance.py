import numpy as np


def accuracy(predicted, reality) -> float:
    squeezed_predicted = np.squeeze(predicted)
    squeezed_reality = np.squeeze(reality)
    return np.sum((squeezed_predicted == squeezed_reality)) / np.shape(squeezed_predicted)[0]

# https://stackoverflow.com/a/48087308
def confusion_matrix(predicted, reality) -> np.ndarray:
    squeezed_prediction = np.squeeze(predicted).astype(np.int32)
    squeezed_reality = np.squeeze(reality)

    K = len(np.unique(squeezed_reality))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(squeezed_reality)):
        result[squeezed_reality[i]][squeezed_prediction[i]] += 1

    return result

def f_score(predicted, reality) -> float:
    true_p = 0
    false_p = 0
    false_n = 0

    conf_m = confusion_matrix(predicted, reality)
    for i in range(conf_m.shape[0]):
        true_p += conf_m[i][i]
        false_p += conf_m[~i][i]
        false_n += conf_m[i][~i]

    bottom = true_p + (false_p + false_n) / 2
    if bottom == 0:
        bottom = 1

    return true_p / bottom