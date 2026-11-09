import numpy as np


class RegressaoLinear:
    """Solução analítica com fórmula fechada"""

    def __init__(self) -> None:
        self.weight = 0
        self.bias = 0

    def fit(self, x, y):
        self.weight = self.calc_weight(x, y)
        self.bias = self.calc_bias(x, y)

    def calc_weight(self, x, y):
        return (x.size * (x * y).sum() - x.sum() * y.sum()) / (
            x.size * (x * x).sum() - (x.sum() * x.sum())
        )

    def calc_bias(self, x, y):
        return np.mean(y) - self.weight * np.mean(x)

    def predict(self, x):
        return self.bias + self.weight * x

    def r2(self, x, y):
        y_pred = self.predict(x)
        ss_res = ((y - y_pred) ** 2).sum()
        y_media = y.mean()
        ss_tot = ((y - y_media) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    def MSE(self, x, y):
        return np.mean((y - self.predict(x)) ** 2)


class RegressaoLinearGD:
    """Gradient Descent"""

    def __init__(self) -> None:
        self.weight = 0
        self.bias = 0
        self.loss_history = []

    def fit(self, x, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            z = self.predict(x)
            loss = self.MSE(x, y)
            self.loss_history.append(loss)

            diff = y - z
            grad_weight = np.mean(-2 * diff * x)
            grad_bias = np.mean(-2 * diff)

            self.weight -= learning_rate * grad_weight
            self.bias -= learning_rate * grad_bias

    def predict(self, x):
        return self.bias + self.weight * x

    def r2(self, x, y):
        y_pred = self.predict(x)
        ss_res = ((y - y_pred) ** 2).sum()
        y_media = y.mean()
        ss_tot = ((y - y_media) ** 2).sum()
        return 1 - (ss_res / ss_tot)

    def MSE(self, x, y):
        return np.mean((y - self.predict(x)) ** 2)
