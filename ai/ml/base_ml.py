from abc import abstractmethod
from typing import Any, Optional

import numpy as np


class BaseModel:
    @abstractmethod
    def learn(self, inputs: np.ndarray, target: Optional[np.ndarray] = None):
        pass

    @abstractmethod
    def predict(self, input_: np.ndarray) -> np.ndarray:
        pass