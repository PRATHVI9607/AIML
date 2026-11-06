import numpy as np

def train(X, y):
    return {c: (np.mean(y == c), X[y == c].mean(axis=0), X[y == c].var(axis=0) + 1e-9) for c in np.unique(y)}

def predict(X, model):
    scores = [np.log(prior) - 0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
              for prior, mean, var in model.values()]
    return np.array(list(model))[np.argmax(scores, axis=0)]

X_train = np.array([[1, 2], [2, 3], [3, 1], [6, 5], [7, 7], [8, 6]])
y_train = np.array(['A', 'A', 'A', 'B', 'B', 'B'])
X_test = np.array([[2, 2], [7, 6], [4, 4]])

model = train(X_train, y_train)
preds = predict(X_test, model)
assert list(predict(X_train, model)) == list(y_train)  # self-check: training points classified correctly
for point, label in zip(X_test, preds):
    print(f'{point} -> {label}')