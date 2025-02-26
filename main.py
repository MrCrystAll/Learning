import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

from ai.ml.supervised.decision_tree import DecisionTreeClassifier

if __name__ == "__main__":
    iris_data, iris_target = datasets.load_digits(return_X_y=True)

    avg_accuracy = []

    X_train, X_test, Y_train, Y_test = train_test_split(iris_data, iris_target, train_size=0.75)
    Y_test = np.expand_dims(Y_test, -1)

    tree = DecisionTreeClassifier(100, 10)

    tree.learn(X_train, Y_train)
    tree.print_tree()
    preds = tree.predict(X_test)

    avg_accuracy.append((preds == Y_test).sum() / preds.shape[0] * 100)

    print("Accuracy:", sum(avg_accuracy) / len(avg_accuracy), "%")