import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.w = None
        self.bias = None

    def fit(self, X, y):
        self.w = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iters):
            linear_model = np.dot(X, self.w) + self.bias
            y_predicted = sigmoid(linear_model)

            dw = np.dot(X.T, (y_predicted - y))
            db = np.sum(y_predicted - y)
            self.w -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.bias
        y_predicted_cls = list()
        y_predicted = sigmoid(linear_model)
        for i in y_predicted:
            if i > 0.5:
                y_predicted_cls.append(1)
            else:
                y_predicted_cls.append(0)
        return np.array(y_predicted_cls)
