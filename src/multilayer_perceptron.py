import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

digits = load_digits()
X = digits.data / 16.0
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def Cross_Entropy_loss(y, logits):
    logits_clip = np.clip(logits, 1e-7, 1 - 1e-7)
    return -np.mean(y * np.log(logits_clip) + (1 - y) * np.log(1 - logits_clip))

def one_hot(y, num_classes=10):
    one_hot_y = np.zeros((y.shape[0], num_classes))
    one_hot_y[np.arange(y.shape[0]), y] = 1
    return one_hot_y


class ReLU:
    def __call__(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.x > 0).astype(float)


class Linear:
    def __init__(self, in_features, out_features, learning_rate = 0.1):
        k_sqrt = 1 / (in_features ** 0.5)     
        self.weight = np.random.uniform(-k_sqrt, k_sqrt, size=(in_features, out_features))
        self.bias = np.random.uniform(-k_sqrt, k_sqrt, size=out_features)
        self.lr = learning_rate
    
    def __call__(self, x):
        self.x = x
        return x @ self.weight + self.bias

    def _calc_gradients(self, grad_output):
        n = self.x.shape[0]
        
        grad_weight = self.x.T @ grad_output
        grad_bias = np.sum(grad_output, axis=0)
        return { "weight": grad_weight, "bias": grad_bias }

    def _update_params(self, gradients):
        self.weight -= gradients["weight"] * self.lr
        self.bias -= gradients["bias"] * self.lr

    def backward(self, grad_output):
        grad_input = grad_output @ self.weight.T

        gradients = self._calc_gradients(grad_output)   

        self._update_params(gradients)

        return grad_input
        

class MultilayerPerceptron:
    def __init__(self):
        self.linear1 = Linear(64, 128)
        self.linear2 = Linear(128, 64)
        self.linear3 = Linear(64, 10)

        self.relu1 = ReLU()
        self.relu2 = ReLU()
    
    def train_step(self, x, y):
        logits = self.forward(x)

        loss = Cross_Entropy_loss(y, logits)

        self.backward(y, logits)

        return loss

    def forward(self, x):
        x = self.relu1(self.linear1(x))
        x = self.relu2(self.linear2(x))

        return self.linear3(x)

    def backward(self, y, logits):
        n = y.shape[0]

        grad = (softmax(logits) - y) / n
        
        grad = self.linear3.backward(grad)
        grad = self.relu2.backward(grad)
        grad = self.linear2.backward(grad)
        grad = self.relu1.backward(grad)
        self.linear1.backward(grad)

    def predict(self, x):
        output = self.forward(x)
        return np.argmax(output, axis=1)

    def accuracy(self, x, y):
        output = self.predict(x)
        return np.mean(output == y)


model = MultilayerPerceptron()

num_epochs = 50
batch_size = 32

y_train_onehot = one_hot(y_train, 10)
y_test_onehot = one_hot(y_test, 10)

for epoch in range(num_epochs):
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train_onehot[i:i+batch_size]
        
        loss = model.train_step(X_batch, y_batch)
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    train_acc = model.accuracy(X_train, y_train)
    test_acc = model.accuracy(X_test, y_test)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")


def visualize_predictions(model, X, y, num_images=9):
    indices = np.random.choice(
        len(X),
        size=num_images,
        replace=False
    )

    X_sample = X[indices]
    y_true = y[indices]

    y_pred = model.predict(X_sample)

    fig, axes = plt.subplots(3, 3, figsize=(9, 9))

    for i, ax in enumerate(axes.flat):
        image = X_sample[i].reshape(8, 8)

        ax.imshow(image, cmap="gray")

        color = "green" if y_true[i] == y_pred[i] else "red"

        ax.set_title(
            f"y_true: {y_true[i]} | y_pred: {y_pred[i]}",
            color=color,
            fontsize=11
        )

        ax.axis("off")

    plt.tight_layout()
    plt.show()


visualize_predictions(model, X_test, y_test)
