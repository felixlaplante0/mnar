from typing import NamedTuple, Self, cast

import numpy as np
from fastkmeanspp import KMeans
from sklearn.impute import SimpleImputer


class Params(NamedTuple):
    prior_probs: np.ndarray
    cluster_centers: np.ndarray
    cluster_cov_matrices: np.ndarray
    missing_probs: np.ndarray

    @classmethod
    def init_params(
        cls, n_clusters: int, X: np.ndarray, mode: str, random_state: int | None
    ) -> Self:
        """Initialize parameters for the model.

        Args:
            n_clusters (int): Number of clusters.
            X (np.ndarray): Data.
            mode (str): Mode.
            random_state (int | None): Random state.

        Returns:
            Self: Initialized parameters.
        """
        # Impute missing values and fit KMeans
        imp = SimpleImputer(strategy="mean")
        X_imp = cast(np.ndarray, imp.fit_transform(X))  # type: ignore
        km = KMeans(n_clusters, random_state=random_state).fit(X_imp)
        # Initialize parameters
        prior_probs = np.array([(km.labels_ == i).mean() for i in range(n_clusters)])
        cluster_centers = cast(np.ndarray, km.cluster_centers_)
        cluster_cov_matrices = np.stack(
            [(km.inertia_ / X.shape[0]) * np.eye(X.shape[1])] * n_clusters
        )
        missing_probs = (
            np.isnan(X).mean(keepdims=True)
            if mode in ("z", "r")
            else np.isnan(X).mean(axis=0, keepdims=True)
        )

        return cls(prior_probs, cluster_centers, cluster_cov_matrices, missing_probs)

    def numel(self, mode: str) -> int:
        """Number of elements.

        Args:
            mode (str): Mode.

        Returns:
            int: Number of elements.
        """
        k_multiplier = 1 if mode in ("r", "rj") else self.cluster_centers.shape[0]
        j_multiplier = 1 if mode in ("zj", "rj") else self.cluster_centers.shape[1]

        return sum(v.size for v in self) * k_multiplier * j_multiplier
