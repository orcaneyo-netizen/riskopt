"""Synthetic factor-model return generation and covariance estimators.

The generator draws returns from a known ``K``-factor model

.. math::

    r_t = \\mu + B f_t + g_{b(i),t} + \\varepsilon_t,

where ``f_t`` are correlated Student-t factor innovations, ``g`` is a block
(industry) shock shared by assets in the same block, and ``\\varepsilon`` is
asset-specific Student-t noise.  Because the data-generating process is known,
every downstream estimator can be validated against the truth.

Covariance estimators
---------------------
``sample``       : classic ``X'X / (T - ddof)``.
``ledoit-wolf``  : Ledoit & Wolf (2004) analytical shrinkage towards the
                   constant-correlation target ``F_ij = rbar * s_i * s_j``.
``ewma``         : exponentially weighted covariance with a half-life.

All estimators are symmetrised and projected onto the PSD cone, so downstream
optimisers never see an indefinite matrix.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "synthetic_returns",
    "sample_cov",
    "ledoit_wolf_cov",
    "ewma_weights",
    "project_psd",
    "is_psd",
    "annualize_cov",
]

CovMethod = Literal["sample", "ledoit-wolf", "ewma"]


def _as_frame(returns: pd.DataFrame | np.ndarray, *, require_finite: bool = True) -> pd.DataFrame:
    """Coerce input to a float DataFrame with sensible default labels."""
    if isinstance(returns, pd.DataFrame):
        frame = returns.astype(float)
        if frame.columns.has_duplicates:
            raise ValueError("returns has duplicate column labels")
    else:
        arr = np.asarray(returns, dtype=float)
        if arr.ndim != 2:
            raise ValueError(f"returns must be 2-D, got {arr.ndim}-D")
        frame = pd.DataFrame(arr, columns=[f"A{i + 1:02d}" for i in range(arr.shape[1])])
    if frame.shape[0] < 2:
        raise ValueError("need at least two observations")
    if frame.shape[1] < 1:
        raise ValueError("need at least one asset")
    if require_finite and not np.isfinite(frame.to_numpy()).all():
        raise ValueError("returns contains NaN or infinite values")
    return frame


def _clean(returns: pd.DataFrame | np.ndarray, nan_policy: str) -> pd.DataFrame:
    """Drop incomplete rows (documented) or raise on non-finite data."""
    frame = _as_frame(returns, require_finite=False)
    if not np.isfinite(frame.to_numpy()).all():
        if nan_policy == "raise":
            raise ValueError("returns contains NaN or infinite values (nan_policy='raise')")
        if nan_policy != "drop":
            raise ValueError(f"unknown nan_policy {nan_policy!r}")
        before = len(frame)
        frame = frame.dropna(how="any")
        warnings.warn(
            f"dropped {before - len(frame)} rows with missing values",
            stacklevel=3,
        )
        if len(frame) < 3:
            raise ValueError("too few complete observations after dropping NaN rows")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("returns still contain infinite values")
    return frame


def project_psd(cov: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Return the nearest PSD matrix by clipping negative eigenvalues.

    Parameters
    ----------
    cov : ndarray, shape (n, n)
        Symmetric candidate covariance matrix.
    eps : float, default 1e-12
        Eigenvalue floor.

    Returns
    -------
    ndarray
        Symmetric matrix with all eigenvalues >= ``eps``.
    """
    sym = 0.5 * (np.asarray(cov, dtype=float) + np.asarray(cov, dtype=float).T)
    vals, vecs = np.linalg.eigh(sym)
    if vals.min() >= eps:
        return sym
    vals = np.clip(vals, eps, None)
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def is_psd(cov: np.ndarray, tol: float = 1e-10) -> bool:
    """True when the symmetric matrix has no eigenvalue below ``-tol``."""
    arr = np.asarray(cov, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("is_psd expects a square matrix")
    sym = 0.5 * (arr + arr.T)
    return bool(np.linalg.eigvalsh(sym).min() >= -abs(tol))


def ewma_weights(n_obs: int, halflife: float) -> np.ndarray:
    """Normalised exponential weights, oldest first, decaying by 1/2 per half-life.

    ``w_t \\propto 2^{-(n-1-t)/halflife}`` for ``t = 0 .. n-1``.
    """
    if n_obs < 1:
        raise ValueError("n_obs must be positive")
    if not np.isfinite(halflife) or halflife <= 0:
        raise ValueError("halflife must be a positive number")
    decay = 0.5 ** (1.0 / float(halflife))
    weights = decay ** np.arange(n_obs - 1, -1, -1, dtype=float)
    return weights / weights.sum()


def synthetic_returns(
    n_assets: int = 12,
    n_days: int = 1500,
    n_factors: int = 4,
    seed: int = 42,
    *,
    block_size: int = 3,
    t_df: float = 5.0,
    factor_vol: float = 0.01,
    idio_vol: float = 0.006,
    block_vol: float = 0.004,
    factor_corr: float = 0.35,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Draw returns from a known block-correlated factor model.

    Parameters
    ----------
    n_assets, n_days, n_factors : int
        Panel dimensions.  Requires ``n_factors < n_assets`` and
        ``n_days > n_assets``.
    seed : int, default 42
        Seed for :class:`numpy.random.Generator` (deterministic output).
    block_size : int, default 3
        Assets are grouped into contiguous blocks of this size; each block
        receives a shared shock, creating within-block correlation.
    t_df : float, default 5.0
        Degrees of freedom of the Student-t innovations (must exceed 2 so the
        variance is finite).  ``t_df=5`` gives excess kurtosis 6.
    factor_vol, idio_vol, block_vol : float
        Daily volatility targets for factors, specific noise and block shocks.
    factor_corr : float
        Equicorrelation of the factor innovations, in ``(-1, 1)``.

    Returns
    -------
    returns : DataFrame, shape (n_days, n_assets)
    factor_loadings : DataFrame, shape (n_assets, n_factors)
        The true ``B`` matrix.
    factor_returns : DataFrame, shape (n_days, n_factors)
        The true factor innovations ``f``.
    """
    for name, value in (("n_assets", n_assets), ("n_days", n_days), ("n_factors", n_factors)):
        if int(value) != value or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if n_factors >= n_assets:
        raise ValueError("n_factors must be smaller than n_assets")
    if n_days <= n_assets + n_factors:
        raise ValueError("n_days must exceed n_assets + n_factors")
    if t_df <= 2:
        raise ValueError("t_df must exceed 2 for a finite variance")
    if not -1.0 < factor_corr < 1.0:
        raise ValueError("factor_corr must lie in (-1, 1)")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if min(factor_vol, idio_vol, block_vol) < 0:
        raise ValueError("volatility parameters must be non-negative")

    rng = np.random.default_rng(seed)
    n_blocks = max(2, int(np.ceil(n_assets / block_size)))
    blocks = np.minimum(np.arange(n_assets) // block_size, n_blocks - 1)

    # Correlated Student-t factors, standardised to unit variance then scaled.
    corr = np.full((n_factors, n_factors), factor_corr)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    t_scale = np.sqrt(t_df / (t_df - 2.0))
    z = rng.standard_t(t_df, size=(n_days, n_factors)) / t_scale
    factors = z @ chol.T
    factors *= factor_vol / factors.std(axis=0, ddof=1)

    # Loadings: factor 1 is a positive "market" factor, the rest are mixed sign.
    loadings = rng.uniform(0.3, 1.2, size=(n_assets, n_factors))
    loadings *= rng.choice([-1.0, 1.0], size=(n_assets, n_factors), p=[0.3, 0.7])
    loadings[:, 0] = rng.uniform(0.5, 1.3, size=n_assets)

    block_shock = rng.standard_t(t_df, size=(n_days, n_blocks)) / t_scale * block_vol
    idio = rng.standard_t(t_df, size=(n_days, n_assets)) / t_scale * idio_vol

    premia = 0.0004 * (0.6 ** np.arange(n_factors))
    mu = loadings @ premia
    rets = mu + factors @ loadings.T + block_shock[:, blocks] + idio

    dates = pd.bdate_range(end="2026-11-20", periods=n_days)
    asset_names = [f"A{i + 1:02d}" for i in range(n_assets)]
    factor_names = [f"F{i + 1}" for i in range(n_factors)]
    return (
        pd.DataFrame(rets, index=dates, columns=asset_names),
        pd.DataFrame(loadings, index=asset_names, columns=factor_names),
        pd.DataFrame(factors, index=dates, columns=factor_names),
    )


def ledoit_wolf_cov(returns: pd.DataFrame | np.ndarray) -> tuple[pd.DataFrame, float]:
    """Ledoit-Wolf shrinkage towards the constant-correlation target.

    The estimator is ``(1 - delta) S + delta F`` with

    .. math::

        F_{ij} = \\bar r \\sqrt{s_{ii} s_{jj}} \\;(i \\neq j), \\qquad F_{ii} = s_{ii},

    where ``\\bar r`` is the average off-diagonal sample correlation.  The
    optimal intensity is the Ledoit & Wolf (2004) plug-in

    .. math::

        \\delta = \\mathrm{clip}\\left(\\frac{\\hat\\pi - \\hat\\rho}{\\hat\\gamma T}, 0, 1\\right),
        \\qquad \\hat\\gamma = \\sum_{ij} (f_{ij} - s_{ij})^2,

    with

    .. math::

        \\hat\\pi = \\sum_{ij}\\frac1T\\sum_t (x_{it}x_{jt} - s_{ij})^2,

        \\hat\\rho = \\sum_i \\hat\\pi_{ii}
            + \\bar r \\sum_{i \\neq j} \\sqrt{s_{jj}/s_{ii}}
              \\left(\\frac1T\\sum_t x_{it}^3 x_{jt} - s_{ii}s_{ij}\\right).

    Sample moments use the ``1/T`` normalisation of the original paper, so the
    result differs from ``sample_cov(..., ddof=1)`` by a factor
    ``T / (T - 1)``.

    Returns
    -------
    cov : DataFrame
        Shrunk covariance matrix.
    delta : float
        Shrinkage intensity in ``[0, 1]`` (0 = sample, 1 = target).
    """
    frame = _clean(returns, "raise")
    x = frame.to_numpy()
    t_obs, n = x.shape
    if n < 2:
        raise ValueError("Ledoit-Wolf shrinkage needs at least two assets")

    xc = x - x.mean(axis=0, keepdims=True)
    sample = xc.T @ xc / t_obs
    var = np.diag(sample).copy()
    if np.any(var <= 0):
        raise ValueError("an asset has zero variance; shrinkage is undefined")
    sd = np.sqrt(var)
    corr = sample / np.outer(sd, sd)
    rbar = (corr.sum() - n) / (n * (n - 1))

    target = rbar * np.outer(sd, sd)
    np.fill_diagonal(target, var)

    y = xc**2
    phi_mat = (y.T @ y) / t_obs - sample**2
    phi = float(phi_mat.sum())

    theta = ((xc**3).T @ xc) / t_obs - var[:, None] * sample
    np.fill_diagonal(theta, 0.0)
    rho = float(np.diag(phi_mat).sum()
                + rbar * (np.outer(1.0 / sd, sd) * theta).sum())

    gamma = float(((target - sample) ** 2).sum())
    if gamma <= 0.0:
        delta = 0.0
    else:
        delta = float(np.clip((phi - rho) / (gamma * t_obs), 0.0, 1.0))

    shrunk = delta * target + (1.0 - delta) * sample
    cov = project_psd(shrunk)
    return pd.DataFrame(cov, index=frame.columns, columns=frame.columns), delta


def sample_cov(
    returns: pd.DataFrame | np.ndarray,
    method: CovMethod = "sample",
    *,
    halflife: float = 63.0,
    ddof: int = 1,
    nan_policy: str = "drop",
) -> pd.DataFrame:
    """Estimate a covariance matrix with the requested method.

    Parameters
    ----------
    returns : DataFrame or ndarray, shape (T, N)
        Asset returns (rows = dates, columns = assets).
    method : {'sample', 'ledoit-wolf', 'ewma'}
        * ``sample`` -- ``X_c' X_c / (T - ddof)``.
        * ``ledoit-wolf`` -- analytical constant-correlation shrinkage.
        * ``ewma`` -- ``sum_t w_t (x_t - mu)(x_t - mu)' / (1 - sum_t w_t^2)``
          with ``w_t`` from :func:`ewma_weights`.
    halflife : float, default 63
        EWMA half-life in observations.
    ddof : int, default 1
        Degrees-of-freedom correction for the sample estimator.
    nan_policy : {'drop', 'raise'}, default 'drop'
        Rows with missing values are dropped with a warning, or rejected.

    Returns
    -------
    DataFrame
        Symmetric PSD covariance matrix labelled like ``returns.columns``.
    """
    if method not in ("sample", "ledoit-wolf", "ewma"):
        raise ValueError(f"unknown covariance method {method!r}")
    if method == "ledoit-wolf":
        cov, _ = ledoit_wolf_cov(returns)
        return cov

    frame = _clean(returns, nan_policy)
    x = frame.to_numpy()
    t_obs = x.shape[0]

    if method == "sample":
        if not 0 <= ddof < t_obs:
            raise ValueError("ddof must satisfy 0 <= ddof < T")
        xc = x - x.mean(axis=0, keepdims=True)
        cov = xc.T @ xc / (t_obs - ddof)
    else:
        w = ewma_weights(t_obs, halflife)
        mu = w @ x
        xc = x - mu
        denom = 1.0 - float(w @ w)
        if denom <= 1e-12:
            raise ValueError("EWMA effective sample size is too small")
        cov = (xc * w[:, None]).T @ xc / denom

    cov = project_psd(cov)
    return pd.DataFrame(cov, index=frame.columns, columns=frame.columns)


def annualize_cov(cov: pd.DataFrame, periods: int = 252) -> pd.DataFrame:
    """Scale a per-period covariance matrix to annual units."""
    if periods <= 0:
        raise ValueError("periods must be positive")
    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a DataFrame")
    return cov * float(periods)
