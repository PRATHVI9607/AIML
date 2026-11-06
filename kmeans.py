import numpy as np

def kmeans(X, k, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    centroids = X[[rng.integers(len(X))]]
    for _ in range(k - 1):
        d = np.linalg.norm(X[:, None] - centroids, axis=2).min(axis=1) ** 2
        centroids = np.vstack([centroids, X[rng.choice(len(X), p=d / d.sum())]])
    for _ in range(iters):
        labels = np.linalg.norm(X[:, None] - centroids, axis=2).argmin(axis=1)
        new = np.array([X[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
        if np.allclose(new, centroids):
            break
        centroids = new
    return centroids, labels

X = np.array([[1, 2], [1.5, 1.8], [1, 0.6], [5, 8], [8, 8], [9, 11], [8, 2], [9, 3], [10, 2]])
centroids, labels = kmeans(X, k=3)

print('Centroids:\n', centroids.round(2))
for point, label in zip(X, labels):
    print(f'{point} -> cluster {label}')
