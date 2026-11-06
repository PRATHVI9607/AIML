import numpy as np

def train(X, y, lr=0.1, iters=1000):
    """Fit weights and bias with gradient descent on log loss."""
    w, b = np.zeros(X.shape[1]), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        w -= lr * X.T @ (p - y) / len(y)
        b -= lr * np.mean(p - y)
    return w, b

def predict(X, w, b):
    return (1 / (1 + np.exp(-(X @ w + b))) >= 0.5).astype(int)

X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array([0, 0, 0, 1, 1, 1])
X_test = np.array([[2, 2], [7, 6], [4, 4]])

w, b = train(X_train, y_train)
print('Weights:', w.round(2), 'Bias:', round(b, 2))
for point, label in zip(X_test, predict(X_test, w, b)):
    print(f'{point} -> class {label}')
