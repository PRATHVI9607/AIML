import numpy as np

from iris import load_iris, split


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


# Logistic regression is binary: keep the two species that overlap (setosa is trivially separable)
X, y = load_iris()
keep = y != 'Iris-setosa'
X, y = X[keep], (y[keep] == 'Iris-virginica').astype(int)  # 0 = versicolor, 1 = virginica
X_train, y_train, X_test, y_test = split(X, y)

# Standardize with training stats only, so gradient descent treats every feature on the same scale
mean, std = X_train.mean(axis=0), X_train.std(axis=0)
X_train, X_test = (X_train - mean) / std, (X_test - mean) / std

w, b = train(X_train, y_train)
print('Weights:', w.round(2), 'Bias:', round(b, 2))
print(f'Train accuracy: {np.mean(predict(X_train, w, b) == y_train):.2%}')
print(f'Test accuracy:  {np.mean(predict(X_test, w, b) == y_test):.2%}')
