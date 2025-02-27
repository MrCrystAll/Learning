import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from ai.ml.supervised.decision_tree.decision_tree_classifier import DecisionTreeClassifier as DecisionTreeClassifierMe
from ai.ml.supervised.metrics.classification.benchmark import benchmark

import seaborn as sns

if __name__ == "__main__":
    def get_data():
        iris_data, iris_target = datasets.load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, train_size=0.75)
        y_test = np.expand_dims(y_test, -1)
        return X_train, X_test, y_train, y_test

    me_results = benchmark(DecisionTreeClassifierMe(10, 10), get_data, 100)
    sk_results = benchmark(DecisionTreeClassifier(max_depth=10, min_samples_leaf=10), get_data, 100)

    print("Accuracy:")
    print("\tMe:", me_results["accuracy"], "%")
    print("\tSk:", sk_results["accuracy"], "%")

    print("Confusion matrix:")
    print("\tMe:", me_results["confusion"])
    print("\tSk:", sk_results["confusion"])

    print("F-Score:")
    print("\tMe:", me_results["f_score"])
    print("\tSk:", sk_results["f_score"])

    fig = plt.figure()
    a1, a2 = fig.subplots(1, 2)
    a1.set_title("My results")
    a2.set_title("Scikit-learn")
    # sns.heatmap(me_results["confusion"], annot=True)
    sns.heatmap(me_results["confusion"], annot=True, ax=a1)
    # sns.heatmap(me_results["confusion"], annot=True)
    sns.heatmap(sk_results["confusion"], annot=True, ax=a2)
    plt.show()
