from numbers import Real
from typing import Self, cast

import numpy as np
import numpy.typing as npt
from scipy.special import xlog1py, xlogy
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils._param_validation import (
    Integral,
    Interval,
    StrOptions,
    validate_params,
)
from sklearn.utils.validation import check_is_fitted

from ._params import Params


class GaussMnar(BaseEstimator, ClusterMixin):
    """Gaussian Mnar EM Algorithm.

    Attributes:
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.
        prior_probs_ (np.ndarray | None): Prior probabilities.
        cluster_centers_ (np.ndarray | None): Cluster means.
        cluster_cov_matrices_ (np.ndarray | None): Cluster covariances.
        miss_probs_ (np.ndarray | None): Missing probabilities.
        mode (str): Mode. Either "r" or "z".
        X_imputed_ (np.ndarray | None): Imputed data.
        loglik_ (float | None): Log-likelihood.
        bic_ (float | None): BIC.
        icl_ (float | None): ICL.
        probs_ (np.ndarray | None): Responsibilities.
        labels_ (np.ndarray | None): Labels.
    """

    max_iter: int
    tol: float
    prior_probs_: np.ndarray | None
    cluster_centers_: np.ndarray | None
    cluster_cov_matrices_: np.ndarray | None
    miss_probs_: np.ndarray | None
    mode: str
    X_imputed_: np.ndarray | None
    loglik_: float | None
    bic_: float | None
    icl_: float | None
    probs_: np.ndarray | None
    labels_: np.ndarray | None

    @validate_params(
        {
            "n_clusters": [Interval(Integral, 1, None, closed="left")],
            "max_iter": [Interval(Integral, 1, None, closed="left")],
            "tol": [Interval(Real, 0, None, closed="left")],
            "mode": [StrOptions({"z", "r", "zj", "rj"})],
            "random_state": [None, Integral],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(
        self,
        n_clusters: int,
        *,
        max_iter: int = 100,
        tol: float = 1e-3,
        mode: str = "zj",
        random_state: int | None = None,
    ):
        """Initialize the model.

        Args:
            n_clusters (int): Number of clusters.
            max_iter (int, optional): Maximum number of iterations. Defaults to 100.
            tol (float, optional): Tolerance. Defaults to 1e-3.
            mode (str, optional): Mode. Defaults to "zj".
            random_state (int | None, optional): Random state. Defaults to None.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.mode = mode
        self.random_state = random_state
        self.tol = tol
        self.prior_probs_ = None
        self.cluster_centers_ = None
        self.cluster_cov_matrices_ = None
        self.miss_probs_ = None
        self.X_imputed_ = None
        self.loglik_ = None
        self.bic_ = None
        self.icl_ = None
        self.probs_ = None
        self.labels_ = None

    @staticmethod
    def _responsibilities(loglik: np.ndarray) -> np.ndarray:
        """Responsibilities.

        Args:
            loglik (np.ndarray): Log-likelihood.

        Returns:
            np.ndarray: Responsibilities.
        """
        # Responsibilities
        t = np.exp(loglik - loglik.max(axis=0))

        return t / t.sum(axis=0)

    def _E_step(
        self,
        X: np.ndarray,
        obs_mask: np.ndarray,
        miss_mask: np.ndarray,
        norm_factor: np.ndarray | float,
        mode: str = "full",
    ) -> np.ndarray | tuple[np.ndarray, ...]:
        """E step.

        Args:
            X (np.ndarray): Data.
            obs_mask (np.ndarray): Observed mask.
            miss_mask (np.ndarray): Missing mask.
            norm_factor (np.ndarray | float): Normalization factor.
            mode (str, optional): Whether to compute log-likelihood, values, impute.
                Defaults to "full".

        Returns:
            np.ndarray | tuple[np.ndarray, ...]: Computed values.
        """
        cluster_centers = cast(np.ndarray, self.cluster_centers_)[:, None]
        cluster_cov_matrices = cast(np.ndarray, self.cluster_cov_matrices_)[:, None]

        # Missing and observed means
        cluster_centers_miss = cluster_centers * miss_mask
        cluster_centers_obs = cluster_centers * obs_mask

        # Missing and observed covariances
        cluster_cov_matrices_miss = (
            cluster_cov_matrices * miss_mask[:, :, None] * miss_mask[:, None, :]
        )
        cluster_cov_matrices_obs = (
            cluster_cov_matrices * obs_mask[:, :, None] * obs_mask[:, None, :]
        )
        cluster_cov_matrices_obs[
            ..., np.arange(X.shape[1]), np.arange(X.shape[1])
        ] += ~obs_mask

        # Inverse of observed covariances
        cluster_cov_matrices_obs_inv = np.linalg.inv(cluster_cov_matrices_obs)

        # Product of covariances and inverse of observed covariances
        cluster_cov_matrices_miss_obs = (
            cluster_cov_matrices * miss_mask[:, :, None] * obs_mask[:, None, :]
        )
        cluster_cov_matrices_miss_obs_cluster_cov_matrices_obs_inv = np.einsum(
            "...ij,...jk->...ik",
            cluster_cov_matrices_miss_obs,
            cluster_cov_matrices_obs_inv,
        )

        # Tilde means
        cluster_centers_tilde_miss = cluster_centers_miss + np.einsum(
            "...ij,...j->...i",
            cluster_cov_matrices_miss_obs_cluster_cov_matrices_obs_inv,
            X - cluster_centers_obs,
        )
        x_tilde = X + cluster_centers_tilde_miss

        # Prior log-likelihood
        prior_loglik = np.log(cast(np.ndarray, self.prior_probs_))[:, None]

        # Gaussian log-likelihood
        centered = X - cluster_centers_obs
        gaussian_loglik = norm_factor - 0.5 * (
            np.einsum(
                "...i,...ij,...j->...", centered, cluster_cov_matrices_obs_inv, centered
            )
            + np.linalg.slogdet(cluster_cov_matrices_obs)[1]
        )

        # Missing log-likelihood
        miss_probs = cast(np.ndarray, self.miss_probs_)[:, None]
        missing_loglik = (
            xlogy(miss_mask, miss_probs) + xlog1py(obs_mask, -miss_probs)
        ).sum(axis=-1)

        # Complete log-likelihood
        loglik = prior_loglik + gaussian_loglik + missing_loglik

        if mode == "loglik":
            return loglik

        if mode == "xloglik":
            return x_tilde, loglik

        # Tilde covariances
        cluster_cov_matrices_tilde = cluster_cov_matrices_miss - np.einsum(
            "...ij,...jk->...ik",
            cluster_cov_matrices_miss_obs_cluster_cov_matrices_obs_inv,
            cluster_cov_matrices_miss_obs,
        )

        return x_tilde, cluster_cov_matrices_tilde, loglik

    def _M_step(
        self,
        x_tilde: np.ndarray,
        cluster_cov_matrices_tilde: np.ndarray,
        loglik: np.ndarray,
        miss_mask: np.ndarray,
    ) -> Params:
        """M step.

        Args:
            x_tilde (np.ndarray): X tilde.
            cluster_cov_matrices_tilde (np.ndarray): cluster_cov_matrices tilde.
            loglik (np.ndarray): Log-likelihood.
            miss_mask (np.ndarray): Missing mask.

        Returns:
            Params: New parameters.
        """
        # Compute responsibilities
        t = self._responsibilities(loglik)
        t_sum = t.sum(axis=1)
        t_red = t / t_sum[:, None]
        prior_probs = t_sum / t.shape[1]

        # M-step for means
        cluster_centers = np.einsum("...i,...ij->...j", t_red, x_tilde)

        # M-step for covariances
        centered = x_tilde - cluster_centers[:, None, :]
        cluster_cov_matrices = np.einsum(
            "...i,...ij,...ik->...jk", t_red, centered, centered
        ) + np.einsum("...i,...ijk->...jk", t_red, cluster_cov_matrices_tilde)

        # Get the missing probabilities
        if self.mode == "z":
            miss_probs = np.clip(
                np.einsum("...i,...ij->...j", t_red, miss_mask).mean(
                    axis=-1, keepdims=True
                ),
                0,
                1,
            )
        elif self.mode == "zj":
            miss_probs = np.clip(np.einsum("...i,...ij->...j", t_red, miss_mask), 0, 1)
        else:
            miss_probs = cast(np.ndarray, self.miss_probs_)

        return Params(prior_probs, cluster_centers, cluster_cov_matrices, miss_probs)

    @staticmethod
    def _validate_X(X: npt.ArrayLike) -> np.ndarray:
        """Validate X.

        Args:
            X (npt.ArrayLike): Data.

        Raises:
            ValueError: If X is not a 2D array.
            ValueError: If X contains inf values.

        Returns:
            np.ndarray: Validated data.
        """
        X = np.asarray(X)  # type: ignore

        if X.ndim != 2:  # noqa: PLR2004
            raise ValueError("X must be a 2D array.")

        if np.isinf(X).any():
            raise ValueError("X must not contain inf values.")

        return X

    @validate_params(
        {"X": ["array-like"], "y": [None]}, prefer_skip_nested_validation=True
    )
    def fit(self, X: npt.ArrayLike, y: None = None) -> Self:  # noqa: ARG002
        """Fit the model.

        Args:
            X (npt.ArrayLike): Data.
            y (None): Placeholder for scikit-learn methods. Defaults to None.

        Returns:
            Self: Fitted model.
        """
        X = self._validate_X(X)  # type: ignore

        # Observed mask and missing mask
        miss_mask = np.isnan(X)
        obs_mask = ~miss_mask
        n_valid = np.sum(obs_mask, axis=1)
        norm_factor = -0.5 * np.log(2 * np.pi) * n_valid

        # Initialize parameters
        init_params = Params.init_params(
            self.n_clusters, X, self.mode, self.random_state
        )
        (
            self.prior_probs_,
            self.cluster_centers_,
            self.cluster_cov_matrices_,
            self.miss_probs_,
        ) = init_params

        # Replace missing values with 0
        X = np.nan_to_num(X, nan=0.0)  # type: ignore

        # EM loop, check for convergence
        for _ in range(self.max_iter):
            x_tilde, cluster_cov_matrices_tilde, loglik = cast(
                tuple[np.ndarray, np.ndarray, np.ndarray],
                self._E_step(X, obs_mask, miss_mask, norm_factor),
            )
            new_params = self._M_step(
                x_tilde, cluster_cov_matrices_tilde, loglik, miss_mask
            )

            # Update parameters
            d = np.linalg.norm(new_params.cluster_centers - self.cluster_centers_)

            # If convergence
            if d < self.tol:
                break

            (
                self.prior_probs_,
                self.cluster_centers_,
                self.cluster_cov_matrices_,
                self.miss_probs_,
            ) = new_params

        # Metrics
        x_tilde, loglik = cast(
            np.ndarray, self._E_step(X, obs_mask, miss_mask, norm_factor, "xloglik")
        )
        self.X_imputed_ = x_tilde[np.argmax(loglik, axis=0), np.arange(X.shape[0])]

        n_params = init_params.numel(self.mode)
        self.loglik_ = float(loglik.sum())
        self.aic_ = -2 * self.loglik_ + 2 * n_params
        self.bic_ = -2 * self.loglik_ + np.log(X.shape[0]) * n_params
        self.probs_ = self._responsibilities(loglik).T
        self.labels_ = np.argmax(self.probs_, axis=1)
        self.icl_ = loglik[self.labels_].sum() - 0.5 * np.log(X.shape[0]) * n_params

        return self

    @validate_params({"X": ["array-like"]}, prefer_skip_nested_validation=True)
    def predict_proba(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the probability of labels of the data.

        Args:
            X (npt.ArrayLike): Data.

        Raises:
            ValueError: If X contains inf values.

        Returns:
            np.ndarray: Labels.
        """
        check_is_fitted(
            self,
            [
                "prior_probs_",
                "cluster_centers_",
                "cluster_cov_matrices_",
                "miss_probs_",
            ],
        )

        X = self._validate_X(X)  # type: ignore

        # Missing mask
        miss_mask = np.isnan(X)

        # Log-likelihood and responsibilities
        return self._responsibilities(
            cast(
                np.ndarray,
                self._E_step(
                    np.nan_to_num(X, nan=0.0), ~miss_mask, miss_mask, 0.0, "loglik"
                ),
            )
        ).T

    @validate_params({"X": ["array-like"]}, prefer_skip_nested_validation=True)
    def predict(self, X: npt.ArrayLike) -> np.ndarray:
        """Predict the labels of the data.

        Args:
            X (npt.ArrayLike): Data.

        Returns:
            np.ndarray: Labels.
        """
        return np.argmax(self.predict_proba(X), axis=1)

    @validate_params({"X": ["array-like"]}, prefer_skip_nested_validation=True)
    def impute(self, X: npt.ArrayLike) -> np.ndarray:
        """Impute missing values.

        Args:
            X (npt.ArrayLike): Data.

        Raises:
            ValueError: If X contains inf values.

        Returns:
            np.ndarray: Imputed data.
        """
        check_is_fitted(
            self,
            [
                "prior_probs_",
                "cluster_centers_",
                "cluster_cov_matrices_",
                "miss_probs_",
            ],
        )

        X = self._validate_X(X)  # type: ignore

        # Missing mask
        miss_mask = np.isnan(X)

        # Imputed data only
        x_tilde, loglik = cast(
            np.ndarray,
            self._E_step(
                np.nan_to_num(X, nan=0.0), ~miss_mask, miss_mask, 0.0, "xloglik"
            ),
        )

        return x_tilde[np.argmax(loglik, axis=0), np.arange(X.shape[0])]
