import numpy as np

from iris import load_iris, split


def train(X, y):
    """Per class: prior P(c), feature means and variances (Gaussian Naive Bayes)."""
    return {c: (np.mean(y == c), X[y == c].mean(axis=0), X[y == c].var(axis=0) + 1e-9) for c in np.unique(y)}


def predict(X, model):
    """Pick the class with the highest log P(c) + sum of log P(x_i | c)."""
    scores = [np.log(prior) - 0.5 * np.sum(np.log(2 * np.pi * var) + (X - mean) ** 2 / var, axis=1)
              for prior, mean, var in model.values()]
    return np.array(list(model))[np.argmax(scores, axis=0)]


X, y = load_iris()
X_train, y_train, X_test, y_test = split(X, y)

model = train(X_train, y_train)
preds = predict(X_test, model)
print(f'Test accuracy: {np.mean(preds == y_test):.2%}')
for c in model:
    print(f'{c}: {np.sum((preds == c) & (y_test == c))}/{np.sum(y_test == c)} correct')
