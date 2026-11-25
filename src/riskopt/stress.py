"""Stress testing: historical windows, factor shocks and scenario grids.

Methodology
-----------
* **Historical windows** replay realised portfolio PnL over non-overlapping
  blocks of ``window`` days and rank them, or evaluate user-specified windows.
  PnL is the arithmetic sum of daily returns inside the block (a first-order
  approximation to the compounded return, which is also reported).
* **Factor shocks** apply a vector of factor moves ``s`` through the loadings:
  ``pnl = w'B s``.  An optional convexity matrix ``H`` adds the second-order
  term ``0.5 s'Hs`` (e.g. option gamma, or a negative convexity adjustment).
* **Scenario grid** crosses equity / rates / credit shocks into a tidy table.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "FactorShockResult",
    "historical_stress",
    "factor_shock",
    "scenario_grid",
]


def _weights(weights: pd.Series | np.ndarray, n: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != n:
        raise ValueError(f"weights have length {w.size}, expected {n}")
    if not np.isfinite(w).all():
        raise ValueError("weights contain NaN or infinite values")
    return w


def _portfolio_returns(
    weights: pd.Series | np.ndarray,
    returns: pd.DataFrame,
    loadings: pd.DataFrame | None,
) -> pd.Series:
    if loadings is None:
        w = _weights(weights, returns.shape[1])
        if isinstance(weights, pd.Series) and set(weights.index) != set(returns.columns):
            raise ValueError("weights index does not match returns columns")
        if isinstance(weights, pd.Series):
            w = weights.reindex(returns.columns).to_numpy(dtype=float)
        port = returns.to_numpy(dtype=float) @ w
        return pd.Series(port, index=returns.index, name="portfolio")
    w = np.asarray(weights, dtype=float).ravel()
    if w.size != loadings.shape[0]:
        raise ValueError("weights length does not match loadings rows")
    exposure = loadings.to_numpy(dtype=float).T @ w
    if returns.shape[1] != loadings.shape[1]:
        raise ValueError("factor_returns columns do not match loadings columns")
    return pd.Series(
        returns.to_numpy(dtype=float) @ exposure, index=returns.index, name="portfolio"
    )


def historical_stress(
    weights: pd.Series | np.ndarray,
    returns: pd.DataFrame,
    windows: Sequence[tuple[object, object]] | None = None,
    window: int = 21,
    n_worst: int = 5,
    loadings: pd.DataFrame | None = None,
    step: int | None = None,
) -> pd.DataFrame:
    """Worst historical windows for the portfolio.

    Parameters
    ----------
    weights : Series or ndarray
        Portfolio weights.  When ``loadings`` is given these are asset weights
        and ``returns`` holds *factor* returns.
    returns : DataFrame
        Asset returns (T x N) or factor returns (T x K) when ``loadings`` is set.
    windows : sequence of (start, end), optional
        Explicit windows as positional integers or index labels.  ``end`` is
        exclusive for integer positions and inclusive for labels.
    window : int, default 21
        Length of the rolling non-overlapping windows when ``windows`` is None.
    n_worst : int, default 5
        Number of worst windows to return.
    loadings : DataFrame, optional
        Asset-by-factor loadings; converts factor returns to portfolio returns.
    step : int, optional
        Stride between windows (defaults to ``window``, i.e. non-overlapping).

    Returns
    -------
    DataFrame
        Columns ``window_id, start, end, n_days, pnl, loss, cum_return`` sorted
        from worst to best PnL.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")
    if returns.empty:
        raise ValueError("returns must not be empty")
    port = _portfolio_returns(weights, returns, loadings)
    n_obs = len(port)
    if n_worst < 1:
        raise ValueError("n_worst must be positive")

    spans: list[tuple[int, int]] = []
    if windows is not None:
        for pair in windows:
            if len(pair) != 2:
                raise ValueError("each window must be a (start, end) pair")
            start, end = pair
            if isinstance(start, (int, np.integer)) and isinstance(end, (int, np.integer)):
                i0, i1 = int(start), int(end)
            else:
                i0 = int(port.index.get_indexer([start])[0])
                i1 = int(port.index.get_indexer([end])[0]) + 1
            if i0 < 0 or i1 > n_obs or i0 >= i1:
                raise ValueError(f"invalid window {pair!r}")
            spans.append((i0, i1))
    else:
        if window < 1:
            raise ValueError("window must be positive")
        stride = window if step is None else int(step)
        if stride < 1:
            raise ValueError("step must be positive")
        spans = [(i, i + window) for i in range(0, n_obs - window + 1, stride)]
        if not spans:
            raise ValueError("returns are shorter than the stress window")

    rows = []
    for i0, i1 in spans:
        block = port.iloc[i0:i1]
        pnl = float(block.sum())
        rows.append(
            {
                "window_id": len(rows),
                "start": port.index[i0],
                "end": port.index[i1 - 1],
                "n_days": int(i1 - i0),
                "pnl": pnl,
                "loss": -pnl,
                "cum_return": float(np.expm1(np.log1p(block.to_numpy()).sum())),
            }
        )
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["pnl", "start"], kind="mergesort").reset_index(drop=True)
    frame = frame.head(int(n_worst)).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    return frame


@dataclass
class FactorShockResult:
    """Outcome of a deterministic factor shock."""

    linear_pnl: float
    quadratic_pnl: float
    total_pnl: float
    loss: float
    per_asset: pd.Series
    factor_contributions: pd.Series
    shocks: pd.Series

    def table(self) -> pd.DataFrame:
        """Per-factor shock / exposure / PnL table."""
        return pd.DataFrame(
            {
                "shock": self.shocks,
                "contribution": self.factor_contributions,
            }
        )


def factor_shock(
    weights: pd.Series | np.ndarray,
    loadings: pd.DataFrame,
    shocks: pd.Series | np.ndarray | Mapping[str, float],
    convexity: pd.DataFrame | np.ndarray | None = None,
) -> FactorShockResult:
    """Apply factor shocks through the loadings.

    ``pnl = w'B s + 0.5 s'Hs`` where ``H`` is the optional convexity matrix.
    For a single-factor model this reduces exactly to ``(w'B) * s``.
    """
    if not isinstance(loadings, pd.DataFrame):
        raise ValueError("loadings must be a DataFrame")
    n, k = loadings.shape
    w = _weights(weights, n)
    if isinstance(weights, pd.Series) and set(weights.index) != set(loadings.index):
        raise ValueError("weights index does not match loadings index")

    if isinstance(shocks, Mapping):
        missing = [c for c in shocks if c not in loadings.columns]
        if missing:
            raise ValueError(f"shocks reference unknown factors: {missing}")
        vec = pd.Series(0.0, index=loadings.columns)
        for name, value in shocks.items():
            vec[name] = float(value)
    elif isinstance(shocks, pd.Series):
        if set(shocks.index) != set(loadings.columns):
            raise ValueError("shocks index does not match factor columns")
        vec = shocks.reindex(loadings.columns).astype(float)
    else:
        arr = np.asarray(shocks, dtype=float).ravel()
        if arr.size != k:
            raise ValueError(f"shocks have length {arr.size}, expected {k}")
        vec = pd.Series(arr, index=loadings.columns)
    if not np.isfinite(vec.to_numpy()).all():
        raise ValueError("shocks contain NaN or infinite values")

    b = loadings.to_numpy(dtype=float)
    s = vec.to_numpy(dtype=float)
    per_asset = pd.Series(w * (b @ s), index=loadings.index, name="pnl")
    linear = float(per_asset.sum())
    if convexity is None:
        quad = 0.0
    else:
        h = np.asarray(convexity, dtype=float)
        if h.shape != (k, k):
            raise ValueError(f"convexity must have shape ({k}, {k})")
        if not np.isfinite(h).all():
            raise ValueError("convexity contains NaN or infinite values")
        quad = float(0.5 * s @ h @ s)
    exposure = b.T @ w
    return FactorShockResult(
        linear_pnl=linear,
        quadratic_pnl=quad,
        total_pnl=linear + quad,
        loss=-(linear + quad),
        per_asset=per_asset,
        factor_contributions=pd.Series(s * exposure, index=loadings.columns, name="pnl"),
        shocks=vec,
    )


def scenario_grid(
    weights: pd.Series | np.ndarray,
    loadings: pd.DataFrame,
    factor_map: Mapping[str, str] | None = None,
    shocks: Mapping[str, Iterable[float]] | None = None,
    convexity: pd.DataFrame | np.ndarray | None = None,
) -> pd.DataFrame:
    """Cross equity / rates / credit shocks into a tidy scenario table.

    Parameters
    ----------
    factor_map : mapping, optional
        ``{'equity': <column>, 'rates': <column>, 'credit': <column>}``.  By
        default columns are matched case-insensitively; otherwise the first
        three factor columns are used in order.
    shocks : mapping, optional
        Per-block shock values.  Defaults: equity ``[-0.20, -0.10, 0, 0.10]``,
        rates ``[-0.01, 0, 0.01]``, credit ``[-0.05, 0, 0.05]``.

    Returns
    -------
    DataFrame with columns ``scenario, equity, rates, credit, pnl, loss``.
    """
    if not isinstance(loadings, pd.DataFrame):
        raise ValueError("loadings must be a DataFrame")
    if loadings.shape[1] < 3:
        raise ValueError("scenario_grid needs at least three factors")
    cols = list(loadings.columns)
    mapping = dict(factor_map or {})
    if not mapping:
        lower = {str(c).lower(): c for c in cols}
        for canonical in ("equity", "rates", "credit"):
            if canonical in lower:
                mapping[canonical] = lower[canonical]
        if len(mapping) < 3:
            mapping = {"equity": cols[0], "rates": cols[1], "credit": cols[2]}
    missing = [k for k in ("equity", "rates", "credit") if k not in mapping]
    if missing:
        raise ValueError(f"factor_map missing keys: {missing}")
    for key, col in mapping.items():
        if col not in cols:
            raise ValueError(f"factor_map[{key!r}] = {col!r} is not a factor column")

    grid = shocks or {
        "equity": (-0.20, -0.10, 0.0, 0.10),
        "rates": (-0.01, 0.0, 0.01),
        "credit": (-0.05, 0.0, 0.05),
    }
    for key in ("equity", "rates", "credit"):
        if key not in grid:
            raise ValueError(f"shocks missing {key!r}")

    rows = []
    for idx, (eq, ra, cr) in enumerate(product(grid["equity"], grid["rates"], grid["credit"])):
        shock_map = {"equity": eq, "rates": ra, "credit": cr}
        applied = {mapping[k]: float(v) for k, v in shock_map.items()}
        result = factor_shock(weights, loadings, applied, convexity)
        rows.append(
            {
                "scenario": idx,
                "equity": float(eq),
                "rates": float(ra),
                "credit": float(cr),
                "pnl": result.total_pnl,
                "loss": result.loss,
            }
        )
    return pd.DataFrame(rows, columns=["scenario", "equity", "rates", "credit", "pnl", "loss"])
