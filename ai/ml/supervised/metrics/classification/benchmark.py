from typing import Callable, Tuple

import numpy as np

from ai.ml.base_ml import BaseModel, base_benchmark
from ai.ml.supervised.metrics.classification.performance import accuracy, confusion_matrix, f_score


def benchmark(estimator: BaseModel, get_data: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], n_iterations: int = 100):
    X_train, X_test, y_train, y_test = get_data()
    n_unique = len(np.unique(y_test))

    estimator_results = {
        "accuracy": np.zeros((n_iterations, 1)),
        "confusion": np.zeros((n_iterations, n_unique, n_unique)),
        "f_score": np.zeros((n_iterations, 1))
    }

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test, predictions = base_benchmark(estimator, get_data)

        estimator_results["accuracy"][i] = accuracy(predictions, y_test)
        estimator_results["confusion"][i] = confusion_matrix(predictions, y_test)
        estimator_results["f_score"][i] = f_score(predictions, y_test)

    return {
        "accuracy": np.mean(estimator_results["accuracy"], dtype=np.float32),
        "confusion": np.mean(estimator_results["confusion"], axis=0, dtype=np.float32),
        "f_score": np.mean(estimator_results["f_score"], dtype=np.float32)
    }
