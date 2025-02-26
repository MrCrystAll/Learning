from typing import Optional, Tuple

import numpy as np

from ai.ml.base_ml import BaseModel


class Node:
    def __init__(self, data, targets, feature_id: int, feature_val: float, probabilities: np.ndarray,
                 information_gain: float):
        self.data = data
        self.targets = targets
        self.feature_id = feature_id
        self.feature_val = feature_val
        self.probabilities = probabilities
        self.information_gain = information_gain

        self.left = None
        self.right = None

    def node_def(self) -> str:

        if self.left or self.right:
            return f"NODE | Information Gain = {self.information_gain} | Split IF X[{self.feature_id}] < {self.feature_val} THEN left O/W right"
        else:
            unique_values, value_counts = np.unique(self.targets, return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])
            return f"LEAF | Label Counts = {output} | Pred Probs = {self.probabilities}"


class DecisionTreeClassifier(BaseModel):
    def __init__(self, max_depth: int, min_leaf_size: int):
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
        self.tree = None
        self.labels_in_train = None

    def predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree
        targets = None

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.probabilities
            if X[node.feature_id] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs, np.unique(self.labels_in_train)

    def _print_recursive(self, node: Node, level=0) -> None:
        if node != None:
            self._print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' + node.node_def())
            self._print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self._print_recursive(node=self.tree)

    def learn(self, inputs: np.ndarray, target: Optional[np.ndarray] = None):
        self.labels_in_train = np.unique(target)
        if isinstance(target, type(None)):
            raise ValueError(self.__class__.__name__ + " requires targets to learn, but you did not pass them")

        self.tree = self.create_tree(inputs, target)

    def _find_label_probs(self, targets: np.array) -> np.array:
        labels_as_integers = targets.astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def predict(self, input_: np.ndarray) -> np.ndarray:
        outputs = np.apply_along_axis(self.predict_one_sample, -1, input_)
        probs = outputs[:, 0]
        targets = outputs[:, 1]

        choices = np.argmax(probs, axis=-1)

        results = np.zeros(shape=(targets.shape[0], 1))

        for i in range(targets.shape[0]):
            results[i] = int(targets[i][choices[i]])

        return results

    def create_tree(self, data: np.ndarray, target: np.ndarray, current_depth: int = 0):
        if current_depth >= self.max_depth:
            return None

        splits_data, splits_target, feature_id, feature_median, entropy = self._get_best_split(data, target)

        label_probabilities = self._find_label_probs(target)
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - entropy

        node = Node(
            data=data,
            targets=target,
            feature_id=feature_id,
            feature_val=feature_median,
            probabilities=label_probabilities,
            information_gain=information_gain
        )

        if splits_data[0].shape[0] <= self.min_leaf_size or splits_data[1].shape[0] <= self.min_leaf_size:
            return node

        current_depth += 1
        node.left = self.create_tree(splits_data[0], splits_target[0], current_depth)
        node.right = self.create_tree(splits_data[1], splits_target[1], current_depth)

        return node

    def entropy(self, probabilities: np.ndarray) -> float:
        return sum([-p * np.log2(p) for p in probabilities if p > 0])

    def _get_best_split(self, data, target) -> Tuple[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], int, float, float]:

        min_entropy = 1e6
        chosen_data = None
        chosen_targets = None
        chosen_id = -1
        chosen_median = 0

        for feature in range(data.shape[1]):
            feature_median = np.median(data[:, feature])

            below_median_mask = data[:, feature] <= feature_median

            below_median_data = data[below_median_mask]
            above_median_data = data[~below_median_mask]

            below_median_target = target[below_median_mask]
            above_median_target = target[~below_median_mask]

            probs_below = self._find_label_probs(below_median_target)
            probs_above = self._find_label_probs(above_median_target)

            total_count = sum([subset.shape[0] for subset in [probs_below, probs_above]])
            part_entropy = sum(
                [self.entropy(subset) * (subset.shape[0] / total_count) for subset in [probs_below, probs_above]])

            if min_entropy > part_entropy:
                min_entropy = part_entropy
                chosen_id = feature
                chosen_median = feature_median
                chosen_data = below_median_data, above_median_data
                chosen_targets = below_median_target, above_median_target

        return chosen_data, chosen_targets, chosen_id, chosen_median, min_entropy
