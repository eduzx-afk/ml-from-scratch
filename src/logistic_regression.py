import numpy as np


class RegressaoLogistica:
    def __init__(self) -> None:
        self.weight = 0
        self.bias = 0
        self.loss_history = []

    def calc_sigmoid(self, z):
        return 1 / (1 + np.e ** (-z))

    def predict(self, x):
        return np.round(self.calc_sigmoid(self.bias + self.weight * x))

    def predict_proba(self, x):
        return self.calc_sigmoid(self.bias + self.weight * x)

    def fit(self, x, y, epochs=100, learning_rate=0.01):
        for epoch in range(epochs):
            z = self.bias + self.weight * x
            logits = self.calc_sigmoid(z)
            loss = self.calc_entropy(y, logits)
            self.loss_history.append(loss)

            diff = logits - y
            grad_weight = np.mean(diff * x)
            grad_bias = np.mean(diff)

            self.weight -= learning_rate * grad_weight
            self.bias -= learning_rate * grad_bias

    def calc_entropy(self, y, logits):
        logits_clip = np.clip(logits, 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(logits_clip) + (1 - y) * np.log(1 - logits_clip))