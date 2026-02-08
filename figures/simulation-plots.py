import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from sklearn.datasets import make_blobs
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.metrics import adjusted_rand_score, mean_squared_error
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from tqdm import tqdm

from mnar import GaussMnar

# Set seed
np.random.seed(42)

# Define true means
true_means = np.array([[0, 0], [2, 2]])


def gen_data(n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=2,
        centers=true_means,
        cluster_std=1.0,
        random_state=None,
    )

    missing_probs = np.array([[0.4, 0.2], [0.1, 0.3]])

    X_miss = X.copy()
    mask = np.zeros((n_samples, 2), dtype=bool)
    for k in range(2):
        idx = y == k
        mask[idx] = np.random.rand(idx.sum(), 2) < missing_probs[k]

    X_miss[mask] = np.nan
    return X, X_miss, y


n_range = [125, 250, 500, 1000, 2000, 4000]
models = [
    GaussMnar(n_clusters=2, mode=mode, random_state=42)
    for mode in ["r", "z", "rj", "zj"]
] + [
    Pipeline(
        [
            ("imputer", IterativeImputer(random_state=42)),
            ("clusterer", GaussianMixture(n_components=2, random_state=42)),
        ]
    )
]
names = ["MCAR", "MNARz", "MNARzj", "MCARj", "Single Iterative Impute"]
n_repeats = 100

# Collect scores and execution times
data = []

for n in tqdm(n_range):
    for model, name in zip(models, names, strict=True):
        for _ in range(n_repeats):
            X, X_miss, y = gen_data(n)

            start_time = time.time()
            model.fit(X_miss)
            elapsed = time.time() - start_time

            if isinstance(model, Pipeline):
                score = adjusted_rand_score(y, model.predict(X_miss))
                mse = mean_squared_error(X, model["imputer"].transform(X_miss))
            else:
                score = adjusted_rand_score(y, model.labels_)
                mse = mean_squared_error(X, model.X_imputed_)

            data.append(
                {"n": n, "model": name, "ari": score, "time": elapsed, "mse": mse}
            )

# Convert to DataFrame
df = pd.DataFrame(data)

# Get best expected ARI
best_ari = (1 - 2 * norm.cdf(-np.linalg.norm(true_means[0] - true_means[1]) / 2)) ** 2

# Boxplot of ARI
plt.figure(figsize=(10, 6))
sns.boxplot(x="n", y="ari", hue="model", data=df, palette="Set2")
plt.axhline(
    y=best_ari, color="r", linestyle="--", label="Best ARI with no missing data"
)
plt.xlabel("Number of samples")
plt.ylabel("Adjusted Rand Index")
plt.title("ARI scores for different models and sample sizes")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("ari-sim.png")
plt.show()

# Boxplot of MSE
plt.figure(figsize=(10, 6))
sns.boxplot(x="n", y="mse", hue="model", data=df, palette="Set2")
plt.xlabel("Number of samples")
plt.ylabel("Mean Squared Error")
plt.title("Reconstruction error for different models and sample sizes")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("mse-sim.png")
plt.show()

# Boxplot of execution time
plt.figure(figsize=(10, 6))
sns.boxplot(x="n", y="time", hue="model", data=df, palette="Set2")
plt.yscale("log")
plt.xlabel("Number of samples")
plt.ylabel("Execution Time (seconds), log scale")
plt.title("Execution time for different models and sample sizes")
plt.legend(title="Model")
plt.tight_layout()
plt.savefig("time-sim.png")
plt.show()
