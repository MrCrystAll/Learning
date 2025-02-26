from typing import Optional

import numpy as np

from ai.ml.base_ml import BaseModel


class Node:
    def __init__(self, feature_id, threshold, value: float = 0.0):
        self.feature_id = feature_id
        self.threshold = threshold
        self.value = value
        self.left, self.right = None, None

    def node_def(self) -> str:
        if self.left or self.right:
            return f"NODE | Split IF X[{self.feature_id}] < {self.threshold} THEN left O/W right"
        else:
            return f"LEAF | Value = {self.value}"

def mse_loss(pred):
    mean = np.mean(pred)
    return np.mean((pred - mean) ** 2)

class DecisionTreeRegressor(BaseModel):
    def __init__(self, max_depth: int, min_leaf_size: int):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.tree = None
        self.labels_in_train = None

    def learn(self, inputs: np.ndarray, target: Optional[np.ndarray] = None):
        self.tree = self.create_tree(inputs, target)

    def predict(self, input_: np.ndarray) -> np.ndarray:
        return np.array([self._predict(inp, self.tree) for inp in input_])

    def _predict(self, input_: np.ndarray, tree: Node) -> float:
        if tree.left is None and tree.right is None:
            return tree.value

        if input_[tree.feature_id] <= tree.threshold:
            return self._predict(input_, tree.left)
        else:
            return self._predict(input_, tree.right)

    def create_tree(self, inputs: np.ndarray, targets: np.ndarray, current_depth: int = 0):
        if self.max_depth <= current_depth:
            return None

        best_feature_id, best_threshold = self.find_best_split(inputs, targets)

        if best_feature_id is None:
            return Node(None, None, float(np.mean(targets)))

        node = Node(
            feature_id=best_feature_id,
            threshold=best_threshold
        )

        in_left, in_right, target_left, target_right = self.split_data(inputs, targets, best_feature_id, best_threshold)

        current_depth += 1
        node.left = self.create_tree(in_left, target_left, current_depth)
        node.right = self.create_tree(in_right, target_right, current_depth)

        return node

    def split_data(self, inputs: np.ndarray, targets: np.ndarray, feature_id, threshold):
        mask = inputs[:, feature_id] <= threshold
        return inputs[mask], inputs[~mask], targets[mask], targets[~mask]

    def _print_recursive(self, node: Node, level=0) -> None:
        if node is not None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def find_best_split(self, inputs: np.ndarray, targets: np.ndarray):
        n_features = inputs.shape[-1]

        min_mse_loss = np.inf
        best_feature_id = None
        best_threshold = None

        for feature in range(n_features):
            thresholds = np.unique(inputs[:, feature])

            for t in thresholds:
                in_left, in_right, target_left, target_right = self.split_data(inputs, targets, feature, t)

                left_mse = mse_loss(target_left)
                right_mse = mse_loss(target_right)

                if left_mse + right_mse < min_mse_loss:
                    min_mse_loss = left_mse + right_mse
                    best_feature_id = feature
                    best_threshold = t

        return best_feature_id, best_threshold