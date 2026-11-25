"""Portfolio risk analytics: factor decomposition, risk contributions, ENB.

Decomposition
-------------
With exposures ``B`` (N x K), factor covariance ``F`` (K x K) and diagonal
specific variances ``D``, the asset covariance is ``Sigma = B F B' + diag(D)``
and

.. math::  \\sigma_p^2 = w'B F B'w + w'D w = (B'w)'F(B'w) + \\sum_i w_i^2 D_i.

Risk contributions
------------------
``MCTR_i = (Sigma w)_i / sigma_p`` and ``CTR_i = w_i MCTR_i``; by Euler's
theorem ``sum_i CTR_i = sigma_p``.  Factor contributions are
``e_k (F e)_k`` with ``e = B'w`` and sum to the factor variance.

Effective number of bets
------------------------
``ENB = exp(-sum_i p_i log p_i)`` with ``p_i = CTR_i / sigma_p``.  It equals 1
for a single-asset portfolio and ``N`` for a portfolio with equal risk
contributions.  ``ENB`` is ``NaN`` when any ``CTR_i <= 0`` (short positions can
hedge risk, making the entropy undefined); this is reported explicitly rather
than silently clipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["PortfolioRisk", "RiskReport", "estimate_loadings"]


@dataclass
class RiskReport:
    """Snapshot of a portfolio's risk profile."""

    weights: pd.Series
    total_vol: float
    total_var: float
    factor_var: float
    specific_var: float
    portfolio_exposure: pd.Series
    factor_contributions: pd.Series
    specific_contributions: pd.Series
    marginal_contributions: pd.Series
    component_contributions: pd.Series
    pct_contributions: pd.Series
    diversification_ratio: float
    effective_bets: float
    beta: float | None
    tracking_error: float | None
    ex_ante_vol: float
    ex_post_vol: float | None
    cov: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """One-row-per-metric summary table for reporting."""
        rows = {
            "ex-ante vol": self.ex_ante_vol,
            "ex-post vol": self.ex_post_vol,
            "factor vol": float(np.sqrt(max(self.factor_var, 0.0))),
            "specific vol": float(np.sqrt(max(self.specific_var, 0.0))),
            "diversification ratio": self.diversification_ratio,
            "effective bets": self.effective_bets,
            "beta": self.beta,
            "tracking error": self.tracking_error,
        }
        return pd.DataFrame({"metric": list(rows), "value": list(rows.values())})


def estimate_loadings(
    asset_returns: pd.DataFrame, factor_returns: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Time-series OLS of each asset on the factors.

    For asset ``i``: ``r_i = alpha_i + B_i f + eps_i`` estimated by ordinary
    least squares over the common sample.

    Returns
    -------
    loadings : DataFrame (N x K)
    alpha : Series
    specific_var : Series
        Residual variance with ``T - K - 1`` degrees of freedom.
    """
    if not isinstance(asset_returns, pd.DataFrame) or not isinstance(factor_returns, pd.DataFrame):
        raise ValueError("asset_returns and factor_returns must be DataFrames")
    idx = asset_returns.index.intersection(factor_returns.index)
    if len(idx) < 3:
        raise ValueError("no overlapping observations between assets and factors")
    assets = asset_returns.loc[idx]
    factors = factor_returns.loc[idx]
    k = factors.shape[1]
    t_obs = len(idx)
    if t_obs <= k + 1:
        raise ValueError("not enough observations to estimate factor loadings")
    if not np.isfinite(assets.to_numpy()).all() or not np.isfinite(factors.to_numpy()).all():
        raise ValueError("returns contain NaN or infinite values")

    design = np.column_stack([np.ones(t_obs), factors.to_numpy(dtype=float)])
    coef, *_ = np.linalg.lstsq(design, assets.to_numpy(dtype=float), rcond=None)
    resid = assets.to_numpy(dtype=float) - design @ coef
    dof = t_obs - k - 1
    spec = (resid**2).sum(axis=0) / dof
    loadings = pd.DataFrame(coef[1:].T, index=assets.columns, columns=factors.columns)
    return (
        loadings,
        pd.Series(coef[0], index=assets.columns, name="alpha"),
        pd.Series(spec, index=assets.columns, name="specific_var"),
    )


class PortfolioRisk:
    """Risk engine for one portfolio.

    Parameters
    ----------
    weights : Series
        Portfolio weights (must sum to 1).
    asset_returns : DataFrame, optional
        Asset returns used for ex-post statistics, betas and (with
        ``factor_returns``) OLS exposure estimation.
    factor_returns : DataFrame, optional
        Factor returns; enables the factor/specific decomposition.
    benchmark : Series, optional
        Benchmark return series (or benchmark weights) for beta / tracking error.
    loadings, factor_cov, specific_var : optional
        Explicit factor model.  When all three are given, no estimation occurs.
    cov : DataFrame, optional
        Explicit covariance matrix.  Without a factor model the decomposition
        reports ``factor_var = 0`` and ``specific_var = total_var``.
    """

    def __init__(
        self,
        weights: pd.Series,
        asset_returns: pd.DataFrame | None = None,
        factor_returns: pd.DataFrame | None = None,
        benchmark: pd.Series | None = None,
        loadings: pd.DataFrame | None = None,
        factor_cov: pd.DataFrame | None = None,
        specific_var: pd.Series | None = None,
        cov: pd.DataFrame | None = None,
    ) -> None:
        if not isinstance(weights, pd.Series):
            raise ValueError("weights must be a pandas Series")
        if weights.empty:
            raise ValueError("weights must not be empty")
        if not np.isfinite(weights.to_numpy()).all():
            raise ValueError("weights contain NaN or infinite values")
        if abs(float(weights.sum()) - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1, got {weights.sum():.10f}")
        self.weights = weights.astype(float)
        self.names = weights.index
        self.asset_returns = self._align_assets(asset_returns)
        self.factor_returns = self._align_factors(factor_returns)
        self.benchmark = benchmark

        explicit = loadings is not None or factor_cov is not None or specific_var is not None
        if explicit:
            if loadings is None or factor_cov is None or specific_var is None:
                raise ValueError("loadings, factor_cov and specific_var must be given together")
            self.loadings = loadings.reindex(index=self.names)
            self.factor_cov = factor_cov.astype(float)
            self.specific_var = specific_var.reindex(self.names)
            if self.loadings.isna().to_numpy().any() or self.specific_var.isna().any():
                raise ValueError("factor model does not cover every asset")
        elif self.factor_returns is not None and self.asset_returns is not None:
            self.loadings, self.alpha, self.specific_var = estimate_loadings(
                self.asset_returns, self.factor_returns
            )
            self.factor_cov = self.factor_returns.cov()
        elif cov is not None:
            self.loadings = None
            self.factor_cov = None
            self.specific_var = None
            self._explicit_cov = cov.astype(float).reindex(index=self.names, columns=self.names)
        else:
            raise ValueError(
                "supply cov, or asset_returns with factor_returns, or an explicit factor model"
            )

        if getattr(self, "_explicit_cov", None) is not None:
            sigma = self._explicit_cov
        elif self.loadings is not None:
            b = self.loadings.to_numpy(dtype=float)
            f = self.factor_cov.to_numpy(dtype=float)
            d = np.diag(self.specific_var.to_numpy(dtype=float))
            sigma = pd.DataFrame(b @ f @ b.T + d, index=self.names, columns=self.names)
        else:  # pragma: no cover - defensive
            raise ValueError("could not assemble a covariance matrix")
        if sigma.isna().to_numpy().any():
            raise ValueError("covariance contains NaN")
        self._cov = 0.5 * (sigma + sigma.T)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_factor_model(
        cls,
        weights: pd.Series,
        loadings: pd.DataFrame,
        factor_cov: pd.DataFrame,
        specific_var: pd.Series,
        benchmark: pd.Series | None = None,
    ) -> "PortfolioRisk":
        """Build the engine directly from a known ``B``, ``F`` and ``D``."""
        return cls(
            weights,
            benchmark=benchmark,
            loadings=loadings,
            factor_cov=factor_cov,
            specific_var=specific_var,
        )

    # ------------------------------------------------------------------ #
    def _align_assets(self, asset_returns: pd.DataFrame | None) -> pd.DataFrame | None:
        if asset_returns is None:
            return None
        if not isinstance(asset_returns, pd.DataFrame):
            raise ValueError("asset_returns must be a DataFrame")
        missing = [n for n in self.names if n not in asset_returns.columns]
        if missing:
            raise ValueError(f"asset_returns missing assets: {missing[:5]}")
        return asset_returns.loc[:, self.names].astype(float)

    def _align_factors(self, factor_returns: pd.DataFrame | None) -> pd.DataFrame | None:
        if factor_returns is None:
            return None
        if not isinstance(factor_returns, pd.DataFrame):
            raise ValueError("factor_returns must be a DataFrame")
        if self.asset_returns is not None:
            idx = self.asset_returns.index.intersection(factor_returns.index)
            if len(idx) < 3:
                raise ValueError("asset and factor returns do not overlap")
            self.asset_returns = self.asset_returns.loc[idx]
            return factor_returns.loc[idx].astype(float)
        return factor_returns.astype(float)

    # ------------------------------------------------------------------ #
    @property
    def cov(self) -> pd.DataFrame:
        """Model covariance matrix ``B F B' + D``."""
        return self._cov

    def covariance(self) -> pd.DataFrame:
        return self._cov

    def volatility(self) -> float:
        return float(np.sqrt(self.total_variance()))

    def total_variance(self) -> float:
        w = self.weights.to_numpy()
        return float(w @ self._cov.to_numpy() @ w)

    def marginal_contributions(self) -> pd.Series:
        """``MCTR_i = (Sigma w)_i / sigma_p``."""
        vol = self.volatility()
        if vol <= 0:
            raise ValueError("portfolio volatility is zero; contributions undefined")
        mctr = (self._cov.to_numpy() @ self.weights.to_numpy()) / vol
        return pd.Series(mctr, index=self.names, name="mctr")

    def component_contributions(self) -> pd.Series:
        """``CTR_i = w_i MCTR_i``; sums to ``sigma_p``."""
        ctr = self.weights * self.marginal_contributions()
        ctr.name = "ctr"
        return ctr

    def factor_contributions(self) -> pd.Series:
        """Per-factor variance contributions ``e_k (F e)_k``."""
        if self.loadings is None:
            return pd.Series(dtype=float, name="factor_ctr")
        b = self.loadings.to_numpy()
        f = self.factor_cov.to_numpy()
        e = b.T @ self.weights.to_numpy()
        contrib = e * (f @ e)
        return pd.Series(contrib, index=self.factor_cov.columns, name="factor_ctr")

    def specific_contributions(self) -> pd.Series:
        """Per-asset specific variance contributions ``w_i^2 D_i``."""
        if self.specific_var is None:
            return pd.Series(0.0, index=self.names, name="specific_ctr")
        return (self.weights**2 * self.specific_var).rename("specific_ctr")

    def diversification_ratio(self) -> float:
        """``(sum_i w_i sigma_i) / sigma_p``."""
        vol = self.volatility()
        if vol <= 0:
            raise ValueError("portfolio volatility is zero")
        return float(self.weights @ np.sqrt(np.diag(self._cov.to_numpy())) / vol)

    def effective_bets(self) -> float:
        """Entropy-based effective number of risk bets (NaN if a CTR <= 0)."""
        ctr = self.component_contributions().to_numpy()
        vol = self.volatility()
        p = ctr / vol
        if np.any(p <= 0):
            return float("nan")
        return float(np.exp(-np.sum(p * np.log(p))))

    # ------------------------------------------------------------------ #
    def _benchmark_returns(self, benchmark: pd.Series | None) -> pd.Series:
        bench = self.benchmark if benchmark is None else benchmark
        if bench is None:
            raise ValueError("no benchmark supplied")
        if not isinstance(bench, pd.Series):
            bench = pd.Series(np.asarray(bench, dtype=float).ravel())
        if self.asset_returns is None:
            raise ValueError("asset_returns are required for beta / tracking error")
        if set(bench.index) == set(self.names):
            return (self.asset_returns @ bench.reindex(self.names)).rename("benchmark")
        if bench.index.equals(self.asset_returns.index):
            return bench.astype(float)
        shared = self.asset_returns.index.intersection(bench.index)
        if len(shared) < 3:
            raise ValueError("benchmark does not overlap asset returns")
        return bench.loc[shared].astype(float)

    def _portfolio_returns(self, index: pd.Index | None = None) -> pd.Series:
        if self.asset_returns is None:
            raise ValueError("asset_returns are required for realised returns")
        frame = self.asset_returns if index is None else self.asset_returns.loc[index]
        return (frame @ self.weights).rename("portfolio")

    def beta(self, benchmark: pd.Series | None = None) -> float:
        """Regression beta of the portfolio on the benchmark."""
        bench = self._benchmark_returns(benchmark)
        port = self._portfolio_returns(bench.index)
        if len(bench) < 3:
            raise ValueError("need at least three observations for beta")
        var_b = float(np.var(bench.to_numpy(), ddof=1))
        if var_b <= 0:
            raise ValueError("benchmark variance is zero")
        cov = float(np.cov(port.to_numpy(), bench.to_numpy(), ddof=1)[0, 1])
        return cov / var_b

    def tracking_error(self, benchmark: pd.Series | None = None) -> float:
        """Annualisation-free standard deviation of active returns."""
        bench = self._benchmark_returns(benchmark)
        port = self._portfolio_returns(bench.index)
        active = port.to_numpy() - bench.to_numpy()
        if active.size < 2:
            raise ValueError("need at least two observations for tracking error")
        return float(np.std(active, ddof=1))

    def ex_post_volatility(self) -> float:
        """Realised standard deviation of portfolio returns."""
        return float(np.std(self._portfolio_returns().to_numpy(), ddof=1))

    # ------------------------------------------------------------------ #
    def report(self, benchmark: pd.Series | None = None) -> RiskReport:
        """Compute the full risk decomposition."""
        vol = self.volatility()
        var = self.total_variance()
        if self.loadings is not None:
            fct_var = float(self.factor_contributions().sum())
            spec_var = float(self.specific_contributions().sum())
            exposure = (self.loadings.T @ self.weights).rename("exposure")
        else:
            fct_var = 0.0
            spec_var = var
            exposure = pd.Series(dtype=float, name="exposure")
        ctr = self.component_contributions()
        pct = (ctr / vol).rename("pct_ctr") if vol > 0 else ctr * np.nan

        beta: float | None
        te: float | None
        try:
            beta = self.beta(benchmark)
        except ValueError:
            beta = None
        try:
            te = self.tracking_error(benchmark)
        except ValueError:
            te = None
        try:
            ex_post = self.ex_post_volatility()
        except ValueError:
            ex_post = None

        return RiskReport(
            weights=self.weights,
            total_vol=vol,
            total_var=var,
            factor_var=fct_var,
            specific_var=spec_var,
            portfolio_exposure=exposure,
            factor_contributions=self.factor_contributions(),
            specific_contributions=self.specific_contributions(),
            marginal_contributions=self.marginal_contributions(),
            component_contributions=ctr,
            pct_contributions=pct,
            diversification_ratio=self.diversification_ratio(),
            effective_bets=self.effective_bets(),
            beta=beta,
            tracking_error=te,
            ex_ante_vol=vol,
            ex_post_vol=ex_post,
            cov=self._cov,
            diagnostics={
                "n_assets": int(self.names.size),
                "n_factors": 0 if self.loadings is None else int(self.loadings.shape[1]),
                "model": "aggregate" if self.loadings is None else "factor",
                "decomposition_residual": float(abs(var - fct_var - spec_var)),
            },
        )
