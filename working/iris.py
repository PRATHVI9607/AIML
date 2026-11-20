import csv
from pathlib import Path

import numpy as np

FEATURES = ['sepal length', 'sepal width', 'petal length', 'petal width']


def load_iris():
    with open(Path(__file__).with_name('iris.csv')) as f:
        rows = [r for r in csv.reader(f) if r]
    return np.array([r[:4] for r in rows], dtype=float), np.array([r[4] for r in rows])


def split(X, y, test_frac=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(len(X))
    n = int(len(X) * test_frac)
    return X[idx[n:]], y[idx[n:]], X[idx[:n]], y[idx[:n]]
