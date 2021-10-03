import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import logistic_regression as LR
from sklearn import linear_model


def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy


if __name__ == "__main__":
    lr_sklearn = linear_model.LogisticRegression()

    ld = datasets.load_wine()
    X, y = ld.data, ld.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=500
    )
    lr_sklearn.fit(X_train, y_train)
    regressor = LR.LogisticRegression()
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)
    predictions2 = lr_sklearn.predict(X_test)
    print("LR classification :", accuracy(y_test, predictions))
    print("framework accuracy:", accuracy(y_test, predictions2))
