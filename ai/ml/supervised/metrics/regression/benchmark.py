from typing import Callable, Tuple

import numpy as np

from ai.ml.base_ml import BaseModel, base_benchmark
from ai.ml.supervised.metrics.regression.loss import mean_squared_error


def benchmark(estimator: BaseModel, get_data: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], n_iterations: int = 100):
    estimator_results = {
        "loss": np.zeros((n_iterations, 1))
    }

    for i in range(n_iterations):
        X_train, X_test, y_train, y_test, predictions = base_benchmark(estimator, get_data)
        estimator_results["loss"][i] = mean_squared_error(predictions, y_test)

    return {
        "loss": np.mean(estimator_results["loss"], dtype=np.float32)
    }
