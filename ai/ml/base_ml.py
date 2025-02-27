from abc import abstractmethod
from typing import Any, Optional, Tuple, Callable

import numpy as np


class BaseModel:
    @abstractmethod
    def learn(self, inputs: np.ndarray, target: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict(self, input_: np.ndarray) -> np.ndarray:
        pass


def base_benchmark(estimator: BaseModel, get_data: Callable[[], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]):
    X_train, X_test, y_train, y_test = get_data()

    if issubclass(type(estimator), BaseModel):
        estimator.learn(X_train, y_train)
    else:
        estimator.fit(X_train, y_train)

    predictions = estimator.predict(X_test)
    return X_train, X_test, y_train, y_test, predictions