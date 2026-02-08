import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score


def gen_data(n_samples: int, d: float) -> tuple[np.ndarray, np.ndarray]:
    X, y = make_blobs(
        n_samples=n_samples,
        n_features=1,
        centers=np.array([[-d / 2], [d / 2]]),
        cluster_std=1.0,
        random_state=42,
    )

    return X, y


# Generate data
ds = np.linspace(0, 5, 50)
scores: list[float] = []
for d in ds:
    X, y = gen_data(1_000_000, d)
    scores.append(adjusted_rand_score(y, (X >= 0).ravel()))

true_scores = (1 - 2 * norm.cdf(-ds / 2)) ** 2

# Plot
plt.plot(ds, scores, "o-", label="Monte-Carlo")
plt.plot(
    ds, true_scores, "x--", label=r"$\left(1 - 2 \Phi(-\frac{d}{2 \sigma})\right)^2$"
)
plt.xlabel(r"$d / \sigma$")
plt.ylabel("ARI")
plt.title("Asymptotic ARI exact computation")
plt.legend()
plt.tight_layout()
plt.savefig("ari-computation.png")
plt.show()
