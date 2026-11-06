import numpy as np
from collections import Counter

def knn_predict(X_train, y_train, X_test, k=3):
    preds = []
    for x in X_test:
        nearest = np.argsort(np.linalg.norm(X_train - x, axis=1))[:k]
        preds.append(Counter(y_train[nearest]).most_common(1)[0][0])
    return np.array(preds)

X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
X_test = np.array([[2, 2], [7, 6], [4, 4]])

for point, label in zip(X_test, knn_predict(X_train, y_train, X_test)):
    print(f'{point} -> {label}')
