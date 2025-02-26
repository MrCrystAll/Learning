import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from ai.ml.supervised.decision_tree.decision_tree_regressor import DecisionTreeRegressor as DecisionTreeRegressorMe
from ai.ml.supervised.decision_tree.decision_tree_classifier import DecisionTreeClassifier as DecisionTreeClassifierMe
from ai.ml.supervised.metrics.regression.loss import mean_squared_error
from ai.ml.supervised.metrics.classification.performance import accuracy

if __name__ == "__main__":
    all_me = []
    all_sk = []

    for i in range(100):
        iris_data, iris_target = datasets.load_wine(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(iris_data, iris_target, train_size=0.75)
        y_test = np.expand_dims(y_test, -1)

        tree = DecisionTreeClassifierMe(10, 10)
        sk_tree = DecisionTreeClassifier(max_depth=10, min_samples_leaf=10)

        tree.learn(X_train, y_train)
        sk_tree.fit(X_train, y_train)

        prediction_me = tree.predict(X_test)
        prediction_sk = sk_tree.predict(X_test)
        prediction_sk = np.expand_dims(prediction_sk, -1)

        all_me.append(accuracy(prediction_me, y_test))
        all_sk.append(accuracy(prediction_sk, y_test))

    print("Accuracy:")
    print("\tMe:", sum(all_me) / len(all_me) * 100, "%")
    print("\tSk:", sum(all_sk) / len(all_sk) * 100, "%")
