"""Black-Litterman model: reverse-optimised priors and view blending.

Equations
---------
Reverse optimisation (Black & Litterman 1992).  If the market portfolio is
mean-variance optimal for risk aversion ``delta``, equilibrium excess returns are

.. math::  \\Pi = \\delta \\, \\Sigma w_{mkt}.

Views are linear statements ``P \\mu = Q + \\epsilon`` with
``\\epsilon \\sim N(0, \\Omega)``.  With the prior ``\\mu \\sim N(\\Pi, \\tau\\Sigma)``
the posterior is Gaussian with

.. math::

    \\mu_{post} = \\left[(\\tau\\Sigma)^{-1} + P'\\Omega^{-1}P\\right]^{-1}
                 \\left[(\\tau\\Sigma)^{-1}\\Pi + P'\\Omega^{-1}Q\\right],

    \\Sigma_{post} = \\Sigma +
                 \\left[(\\tau\\Sigma)^{-1} + P'\\Omega^{-1}P\\right]^{-1}.

By default ``Omega = diag(tau * P Sigma P')`` (He & Litterman), i.e. view
uncertainty scales with the prior variance of the view portfolio.  An
equivalent, numerically cheaper form of the mean is

``mu_post = Pi + tau*Sigma*P' (P*tau*Sigma*P' + Omega)^{-1} (Q - P*Pi)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["BlackLitterman", "BLResult"]


@dataclass
class BLResult:
    """Posterior summary produced by :meth:`BlackLitterman.blend_views`."""

    prior_mean: pd.Series
    posterior_mean: pd.Series
    posterior_cov: pd.DataFrame
    omega: pd.DataFrame
    views: pd.DataFrame
    tau: float
    risk_aversion: float
    diagnostics: dict = field(default_factory=dict)


def _square(mat: pd.DataFrame | np.ndarray, label: str) -> tuple[np.ndarray, pd.Index]:
    if isinstance(mat, pd.DataFrame):
        if mat.shape[0] != mat.shape[1]:
            raise ValueError(f"{label} must be square")
        arr = mat.to_numpy(dtype=float)
        names = mat.columns
    else:
        arr = np.asarray(mat, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError(f"{label} must be a square matrix")
        names = pd.Index([f"A{i + 1:02d}" for i in range(arr.shape[0])])
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} contains NaN or infinite values")
    if not np.allclose(arr, arr.T, atol=1e-12):
        raise ValueError(f"{label} must be symmetric")
    return arr, names


class BlackLitterman:
    """Black-Litterman posterior engine.

    Parameters
    ----------
    prior_mu : Series or None
        Optional explicit prior mean.  When ``None`` the prior must be supplied
        through :meth:`implied_prior` / :meth:`blend_views`.
    tau : float, default 0.05
        Prior scale; the prior covariance of the mean is ``tau * Sigma``.
    risk_aversion : float, default 2.5
        ``delta`` used by the reverse optimisation.
    """

    def __init__(
        self,
        prior_mu: pd.Series | None = None,
        tau: float = 0.05,
        risk_aversion: float = 2.5,
    ) -> None:
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("tau must be a positive number")
        if not np.isfinite(risk_aversion) or risk_aversion <= 0:
            raise ValueError("risk_aversion must be a positive number")
        if prior_mu is not None and not isinstance(prior_mu, pd.Series):
            prior_mu = pd.Series(np.asarray(prior_mu, dtype=float))
        if prior_mu is not None and not np.isfinite(prior_mu.to_numpy()).all():
            raise ValueError("prior_mu contains NaN or infinite values")
        self.prior_mu = prior_mu
        self.tau = float(tau)
        self.risk_aversion = float(risk_aversion)

    # ------------------------------------------------------------------ #
    def implied_prior(
        self,
        cov: pd.DataFrame | np.ndarray,
        market_weights: pd.Series | np.ndarray,
    ) -> pd.Series:
        """Equilibrium excess returns ``Pi = delta * Sigma * w_mkt``."""
        sigma, names = _square(cov, "cov")
        w = self._vector(market_weights, names, "market_weights")
        if np.any(w < -1e-12):
            raise ValueError("market_weights must be non-negative")
        if abs(w.sum() - 1.0) > 1e-8:
            raise ValueError(f"market_weights must sum to 1, got {w.sum():.10f}")
        pi = self.risk_aversion * (sigma @ w)
        return pd.Series(pi, index=names, name="implied_prior")

    def implied_weights(
        self,
        cov: pd.DataFrame | np.ndarray,
        mu: pd.Series | np.ndarray,
        risk_aversion: float | None = None,
    ) -> pd.Series:
        """Unconstrained mean-variance weights ``(1/delta) Sigma^{-1} mu``, normalised.

        Inverts :meth:`implied_prior`; feeding ``Pi`` returns the market weights.
        """
        sigma, names = _square(cov, "cov")
        delta = self.risk_aversion if risk_aversion is None else float(risk_aversion)
        if delta <= 0:
            raise ValueError("risk_aversion must be positive")
        m = self._vector(mu, names, "mu")
        raw = np.linalg.pinv(sigma) @ m / delta
        total = raw.sum()
        if abs(total) < 1e-14:
            raise ValueError("implied weights sum to zero; mu is not investable")
        return pd.Series(raw / total, index=names, name="implied_weights")

    # ------------------------------------------------------------------ #
    def posterior(
        self,
        prior_mu: pd.Series | np.ndarray,
        cov: pd.DataFrame | np.ndarray,
        P: pd.DataFrame | np.ndarray,
        Q: pd.Series | np.ndarray,
        omega: pd.DataFrame | np.ndarray | None = None,
    ) -> tuple[pd.Series, pd.DataFrame]:
        """Posterior mean and covariance given views ``P mu = Q``.

        ``P`` may be empty (zero rows) in which case the prior is returned with
        covariance ``(1 + tau) * Sigma``.  ``omega`` defaults to
        ``diag(tau * P Sigma P')``.
        """
        sigma, names = _square(cov, "cov")
        n = names.size
        pi = self._vector(prior_mu, names, "prior_mu")
        p = self._views_matrix(P, n, names)
        k = p.shape[0]
        q = np.asarray(Q, dtype=float).ravel()
        if k == 0:
            if q.size not in (0,):
                raise ValueError("Q supplied without views in P")
            post_cov = sigma * (1.0 + self.tau)
            return (
                pd.Series(pi, index=names, name="posterior_mean"),
                pd.DataFrame(0.5 * (post_cov + post_cov.T), index=names, columns=names),
            )
        if q.size != k:
            raise ValueError(f"Q has length {q.size}, expected {k}")
        if not np.isfinite(q).all():
            raise ValueError("Q contains NaN or infinite values")

        if omega is None:
            om = np.diag(np.diag(self.tau * p @ sigma @ p.T))
        else:
            om = np.asarray(omega, dtype=float)
            if om.ndim == 1:
                om = np.diag(om)
            if om.shape != (k, k):
                raise ValueError(f"omega must have shape ({k}, {k})")
            if not np.isfinite(om).all():
                raise ValueError("omega contains NaN or infinite values")
            if np.any(np.diag(om) <= 0):
                raise ValueError("omega must have a strictly positive diagonal")
        om = 0.5 * (om + om.T)

        tau_sigma_inv = np.linalg.pinv(self.tau * sigma)
        om_inv = np.linalg.pinv(om)
        precision = tau_sigma_inv + p.T @ om_inv @ p
        post_prec_inv = np.linalg.pinv(precision)
        post_mean = post_prec_inv @ (tau_sigma_inv @ pi + p.T @ om_inv @ q)
        post_cov = sigma + post_prec_inv
        post_cov = 0.5 * (post_cov + post_cov.T)

        return (
            pd.Series(post_mean, index=names, name="posterior_mean"),
            pd.DataFrame(post_cov, index=names, columns=names),
        )

    # ------------------------------------------------------------------ #
    def blend_views(
        self,
        cov: pd.DataFrame | np.ndarray,
        market_weights: pd.Series | np.ndarray,
        P: pd.DataFrame | np.ndarray,
        Q: pd.Series | np.ndarray,
        omega: pd.DataFrame | np.ndarray | None = None,
        prior_mu: pd.Series | np.ndarray | None = None,
    ) -> BLResult:
        """Convenience wrapper: implied prior (or explicit prior) + views."""
        sigma, names = _square(cov, "cov")
        if prior_mu is None:
            prior = self.prior_mu if self.prior_mu is not None else self.implied_prior(
                pd.DataFrame(sigma, index=names, columns=names), market_weights
            )
            prior = pd.Series(self._vector(prior, names, "prior_mu"), index=names)
        else:
            prior = pd.Series(self._vector(prior_mu, names, "prior_mu"), index=names)
        post_mean, post_cov = self.posterior(
            prior, pd.DataFrame(sigma, index=names, columns=names), P, Q, omega
        )

        p = self._views_matrix(P, names.size, names)
        k = p.shape[0]
        if omega is None:
            om = np.diag(np.diag(self.tau * p @ sigma @ p.T)) if k else np.zeros((0, 0))
        else:
            om = np.asarray(omega, dtype=float)
            if om.ndim == 1:
                om = np.diag(om)
        views = pd.DataFrame(
            {
                "view": [f"V{i + 1}" for i in range(k)],
                "Q": np.asarray(Q, dtype=float).ravel() if k else np.array([]),
                "prior_view": (p @ prior.to_numpy()) if k else np.array([]),
            }
        )
        return BLResult(
            prior_mean=prior,
            posterior_mean=post_mean,
            posterior_cov=post_cov,
            omega=pd.DataFrame(om, index=views["view"], columns=views["view"]) if k else pd.DataFrame(),
            views=views,
            tau=self.tau,
            risk_aversion=self.risk_aversion,
            diagnostics={
                "n_views": int(k),
                "tau": self.tau,
                "risk_aversion": self.risk_aversion,
                "omega_source": "diag(tau * P Sigma P')" if omega is None else "user",
                "view_shift": (post_mean - prior).to_dict(),
            },
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _vector(values: pd.Series | np.ndarray, names: pd.Index, label: str) -> np.ndarray:
        if isinstance(values, pd.Series):
            if set(values.index) != set(names):
                raise ValueError(f"{label} index does not match cov columns")
            arr = values.reindex(names).to_numpy(dtype=float)
        else:
            arr = np.asarray(values, dtype=float).ravel()
        if arr.size != names.size:
            raise ValueError(f"{label} has length {arr.size}, expected {names.size}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{label} contains NaN or infinite values")
        return arr

    @staticmethod
    def _views_matrix(P: pd.DataFrame | np.ndarray, n: int, names: pd.Index) -> np.ndarray:
        if isinstance(P, pd.DataFrame):
            p = P.reindex(columns=names)
            if p.isna().to_numpy().any():
                raise ValueError("P has columns not present in cov")
            arr = p.to_numpy(dtype=float)
        else:
            arr = np.asarray(P, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.ndim != 2:
                raise ValueError("P must be 2-D (views x assets)")
        if arr.shape[1] != n:
            raise ValueError(f"P has {arr.shape[1]} columns, expected {n}")
        if not np.isfinite(arr).all():
            raise ValueError("P contains NaN or infinite values")
        if arr.shape[0] and not np.any(np.abs(arr).sum(axis=1) > 0):
            raise ValueError("every view must reference at least one asset")
        return arr
