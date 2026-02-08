import time

import numpy as np

from mnar import GaussMnar

# Parameters for synthetic data
n_samples = 5000
n_features = 3
n_clusters = 2

# Generate synthetic Gaussian data
true_means = np.array([[0, 0, 0], [5, 5, 5]])
true_covs = np.array([np.eye(n_features), 2 * np.eye(n_features)])
labels = np.random.choice(n_clusters, size=n_samples)

X = np.zeros((n_samples, n_features))
for k in range(n_clusters):
    X[labels == k] = np.random.multivariate_normal(
        true_means[k], true_covs[k], size=(labels == k).sum()
    )

# Make missing values depend on cluster and feature
missing_probs = np.array(
    [
        [0.05, 0.2, 0.1],  # cluster 0 missing probabilities for each feature
        [0.4, 0.1, 0.25],  # cluster 1 missing probabilities
    ]
)

mask = np.zeros((n_samples, n_features), dtype=bool)
for k in range(n_clusters):
    idx = labels == k
    mask[idx] = np.random.rand(idx.sum(), n_features) < missing_probs[k]

# Add missing values
X[mask] = np.nan


# Initialize and fit the model
model = GaussMnar(n_clusters=n_clusters)

start = time.time()
model.fit(X)
end = time.time()

print(f"Time taken to fit the model: {end - start:.2f} seconds\n")
print("Estimated cluster centers:\n", model.cluster_centers_)
print("Estimated cluster covariances:\n", model.cluster_cov_matrices_)
print("Estimated missing probabilities:\n", model.miss_probs_)
print(model.icl_)