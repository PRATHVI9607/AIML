import numpy as np

from iris import FEATURES, load_iris


def kmeans(X, k, iters=100, seed=0):
    """Returns (centroids, labels). Stops early once centroids stop moving."""
    rng = np.random.default_rng(seed)
    # k-means++ init: pick each new centroid with probability ∝ squared distance to the nearest existing one
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


def inertia(X, centroids, labels):
    """Sum of squared distances from each point to its centroid (lower = tighter clusters)."""
    return np.sum((X - centroids[labels]) ** 2)


X, y = load_iris()  # species labels are NOT used for clustering, only to check the result
# k-means can get stuck in a bad local optimum, so run 10 random starts and keep the tightest
centroids, labels = min((kmeans(X, k=3, seed=s) for s in range(10)), key=lambda r: inertia(X, *r))

print('Centroids (' + ', '.join(FEATURES) + '):\n', centroids.round(2))
print('\nHow many of each species landed in each cluster:')
for i in range(3):
    species, counts = np.unique(y[labels == i], return_counts=True)
    print(f'cluster {i}:', dict(zip(species.tolist(), counts.tolist())))
