import numpy as np
from collections import Counter

from iris import load_iris, split


def knn_predict(X_train, y_train, X_test, k=3):
    """Label each test point by majority vote of its k nearest training points."""
    preds = []
    for x in X_test:
        nearest = np.argsort(np.linalg.norm(X_train - x, axis=1))[:k]
        preds.append(Counter(y_train[nearest]).most_common(1)[0][0])
    return np.array(preds)


X, y = load_iris()
X_train, y_train, X_test, y_test = split(X, y)

for k in (1, 3, 5, 7):
    acc = np.mean(knn_predict(X_train, y_train, X_test, k) == y_test)
    print(f'k={k}: test accuracy {acc:.2%}')
