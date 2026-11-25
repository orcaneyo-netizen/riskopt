"""Turnover and transaction-cost model.

Cost of moving from ``prev`` to ``target`` weights, per asset:

.. math::

    c_i = \\frac{\\text{bps}}{10^4}\\,|\\Delta w_i|
        + \\kappa\\,|\\Delta w_i|^{3/2}\\,\\sigma_i,

where the linear term is half-spread/commission and the concave-in-size
``|dw|^{3/2}`` term is a square-root market-impact proxy scaled by asset
volatility ``sigma_i`` (Almgren et al. style).  Total cost is expressed as a
fraction of NAV; multiply by NAV for currency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TurnoverCostModel", "net_weights", "trade_list"]


def _align(prev: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> tuple[pd.Series, pd.Series]:
    if not isinstance(prev, pd.Series) or not isinstance(target, pd.Series):
        p = pd.Series(np.asarray(prev, dtype=float).ravel(), name="prev")
        t = pd.Series(np.asarray(target, dtype=float).ravel(), name="target")
        if p.size != t.size:
            raise ValueError("prev and target must have the same length")
        t.index = p.index
    else:
        p, t = prev.astype(float), target.astype(float)
        idx = p.index.union(t.index)
        p = p.reindex(idx, fill_value=0.0)
        t = t.reindex(idx, fill_value=0.0)
    if not np.isfinite(p.to_numpy()).all() or not np.isfinite(t.to_numpy()).all():
        raise ValueError("weights contain NaN or infinite values")
    return p, t


@dataclass
class TurnoverCostModel:
    """Linear-plus-impact turnover cost model.

    Parameters
    ----------
    linear_bps : float, default 5.0
        Linear cost in basis points of traded notional.
    impact_coef : float, default 0.1
        Coefficient ``kappa`` of the ``|dw|^{3/2} * sigma`` impact term.
    """

    linear_bps: float = 5.0
    impact_coef: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.linear_bps) or self.linear_bps < 0:
            raise ValueError("linear_bps must be a non-negative number")
        if not np.isfinite(self.impact_coef) or self.impact_coef < 0:
            raise ValueError("impact_coef must be a non-negative number")

    # ------------------------------------------------------------------ #
    def cost_series(
        self,
        prev: pd.Series | np.ndarray,
        target: pd.Series | np.ndarray,
        sigma: pd.Series | float | None = None,
    ) -> pd.Series:
        """Per-asset cost as a fraction of NAV.

        ``sigma`` is the per-period asset volatility; ``None`` means 1.0
        (unit-volatility convention, costs then expressed per unit of vol).
        """
        p, t = _align(prev, target)
        dw = (t - p).abs()
        if sigma is None:
            vol = pd.Series(1.0, index=dw.index)
        elif np.isscalar(sigma):
            value = float(sigma)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("sigma must be positive")
            vol = pd.Series(value, index=dw.index)
        else:
            if not isinstance(sigma, pd.Series):
                sigma = pd.Series(np.asarray(sigma, dtype=float).ravel(), index=dw.index)
            vol = sigma.astype(float).reindex(dw.index)
            if vol.isna().any():
                raise ValueError("sigma does not cover every asset")
            if (vol <= 0).any() or not np.isfinite(vol.to_numpy()).all():
                raise ValueError("sigma must be finite and strictly positive")
        cost = self.linear_bps / 1e4 * dw + self.impact_coef * dw**1.5 * vol
        cost.name = "cost"
        return cost

    def total_cost(
        self,
        prev: pd.Series | np.ndarray,
        target: pd.Series | np.ndarray,
        sigma: pd.Series | float | None = None,
    ) -> float:
        """Total cost as a fraction of NAV."""
        return float(self.cost_series(prev, target, sigma).sum())

    def turnover(self, prev: pd.Series | np.ndarray, target: pd.Series | np.ndarray) -> float:
        """One-way turnover ``0.5 * sum |dw|``."""
        p, t = _align(prev, target)
        return float(0.5 * (t - p).abs().sum())

    def net_weights(
        self,
        prev: pd.Series | np.ndarray,
        target: pd.Series | np.ndarray,
        sigma: pd.Series | float | None = None,
    ) -> pd.Series:
        """Post-trade weights net of the cost drag: ``(1 - c) * target``."""
        p, t = _align(prev, target)
        total = self.total_cost(p, t, sigma)
        if total >= 1.0:
            raise ValueError("transaction costs exceed 100% of NAV")
        out = (1.0 - total) * t
        out.name = "net_weight"
        return out

    def trade_list(
        self,
        prev: pd.Series | np.ndarray,
        target: pd.Series | np.ndarray,
        nav: float = 1.0,
        tol: float = 1e-12,
    ) -> pd.DataFrame:
        """Tidy trade blotter.

        Columns: ``asset, prev_weight, target_weight, delta_weight, notional,
        side``.  ``notional = delta_weight * nav``; rows with
        ``|delta_weight| <= tol`` are omitted.
        """
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError("nav must be positive")
        p, t = _align(prev, target)
        delta = t - p
        keep = delta.abs() > tol
        frame = pd.DataFrame(
            {
                "asset": delta.index.astype(str),
                "prev_weight": p.to_numpy(),
                "target_weight": t.to_numpy(),
                "delta_weight": delta.to_numpy(),
                "notional": delta.to_numpy() * float(nav),
            }
        )
        frame = frame.loc[keep.to_numpy()].reset_index(drop=True)
        frame["side"] = np.where(frame["delta_weight"] > 0, "BUY", "SELL")
        return frame


def net_weights(
    prev: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray,
    cost_model: TurnoverCostModel,
    sigma: pd.Series | float | None = None,
) -> pd.Series:
    """Functional wrapper around :meth:`TurnoverCostModel.net_weights`."""
    if not isinstance(cost_model, TurnoverCostModel):
        raise ValueError("cost_model must be a TurnoverCostModel")
    return cost_model.net_weights(prev, target, sigma)


def trade_list(
    prev: pd.Series | np.ndarray,
    target: pd.Series | np.ndarray,
    nav: float = 1.0,
) -> pd.DataFrame:
    """Trade blotter using default (zero) cost settings."""
    return TurnoverCostModel().trade_list(prev, target, nav)
