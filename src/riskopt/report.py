"""One-page portfolio report: markdown for humans, JSON for machines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import sample_cov
from .risk import PortfolioRisk
from .stress import historical_stress

__all__ = ["Report", "build_report"]


def _fmt(value: Any, digits: int = 4) -> str:
    """Format a number for markdown, rendering NaN/missing as ``n/a``."""
    if value is None:
        return "n/a"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    if number != 0.0 and abs(number) < 10 ** (-digits):
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def _pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{100.0 * number:.{digits}f}%"


def _jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas objects to JSON-safe Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, pd.DataFrame):
        return [_jsonable(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


@dataclass
class Report:
    """Markdown + JSON risk report."""

    markdown: str
    data: dict

    def to_json(self, indent: int = 2) -> str:
        """Serialise the structured payload (NaN becomes ``null``)."""
        return json.dumps(_jsonable(self.data), indent=indent, allow_nan=False)

    def save(self, path: str | Path) -> Path:
        """Write the markdown report (``.json`` writes JSON instead)."""
        target = Path(path)
        if target.suffix.lower() == ".json":
            target.write_text(self.to_json(), encoding="utf-8")
        else:
            target.write_text(self.markdown, encoding="utf-8")
        return target

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return self.markdown


def build_report(
    returns: pd.DataFrame,
    weights: pd.Series,
    factor_returns: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    cov_method: str = "ledoit-wolf",
    cov: pd.DataFrame | None = None,
    stress_window: int = 21,
    n_worst: int = 3,
    top_n: int = 5,
) -> Report:
    """Aggregate weights, risk decomposition, stress and diversification.

    Parameters
    ----------
    returns : DataFrame (T x N)
        Asset returns.
    weights : Series
        Portfolio weights indexed by asset.
    factor_returns : DataFrame, optional
        Factor returns; enables the factor/specific decomposition via OLS.
    benchmark : Series, optional
        Benchmark returns (or benchmark weights) for beta / tracking error.
    cov_method : str, default 'ledoit-wolf'
        Passed to :func:`riskopt.data.sample_cov` when ``cov`` is None.
    cov : DataFrame, optional
        Explicit covariance matrix (overrides ``cov_method``).
    stress_window : int, default 21
        Length of the historical stress windows.
    n_worst : int, default 3
        Number of worst windows to report.
    top_n : int, default 5
        Number of top risk contributors to show.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")
    if not isinstance(weights, pd.Series):
        raise ValueError("weights must be a Series")
    if returns.empty:
        raise ValueError("returns must not be empty")
    missing = [n for n in weights.index if n not in returns.columns]
    if missing:
        raise ValueError(f"weights reference assets not in returns: {missing[:5]}")
    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("weights contain NaN or infinite values")
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("weights must sum to 1")

    if cov is None:
        cov = sample_cov(returns.loc[:, weights.index], method=cov_method)

    if factor_returns is not None:
        engine = PortfolioRisk(
            weights,
            asset_returns=returns.loc[:, weights.index],
            factor_returns=factor_returns,
            benchmark=benchmark,
        )
    else:
        engine = PortfolioRisk(weights, asset_returns=returns.loc[:, weights.index],
                               benchmark=benchmark, cov=cov)
    risk = engine.report()

    stress = historical_stress(
        weights, returns.loc[:, weights.index], window=stress_window, n_worst=n_worst
    )

    weights_sorted = weights.sort_values(ascending=False)
    contributors = risk.component_contributions.sort_values(ascending=False).head(top_n)

    factor_vol = float(np.sqrt(max(risk.factor_var, 0.0)))
    specific_vol = float(np.sqrt(max(risk.specific_var, 0.0)))

    lines: list[str] = ["# Portfolio risk report", ""]
    lines.append(
        f"_{len(weights)} assets · covariance `{cov_method}` · "
        f"factor model: {'yes' if risk.diagnostics['n_factors'] else 'no'}_"
    )
    lines += ["", "## Summary", "", "| metric | value |", "| --- | ---: |"]
    summary_rows = [
        ("Ex-ante volatility", _pct(risk.ex_ante_vol)),
        ("Ex-post volatility", _pct(risk.ex_post_vol)),
        ("Factor volatility", _pct(factor_vol)),
        ("Specific volatility", _pct(specific_vol)),
        ("Diversification ratio", _fmt(risk.diversification_ratio)),
        ("Effective number of bets", _fmt(risk.effective_bets, 3)),
        ("Beta vs benchmark", _fmt(risk.beta, 3)),
        ("Tracking error", _pct(risk.tracking_error)),
        ("Largest weight", _pct(float(weights.max()))),
    ]
    lines += [f"| {name} | {value} |" for name, value in summary_rows]

    lines += ["", "## Weights", "", "| asset | weight |", "| --- | ---: |"]
    lines += [f"| {asset} | {_pct(weight)} |" for asset, weight in weights_sorted.items()]

    lines += [
        "",
        "## Risk decomposition",
        "",
        "| component | variance | volatility | share of variance |",
        "| --- | ---: | ---: | ---: |",
    ]
    total_var = risk.total_var if risk.total_var > 0 else np.nan
    lines.append(
        f"| Total | {_fmt(risk.total_var, 6)} | {_pct(risk.total_vol)} | 100.00% |"
    )
    lines.append(
        f"| Factor | {_fmt(risk.factor_var, 6)} | {_pct(factor_vol)} | "
        f"{_pct(risk.factor_var / total_var)} |"
    )
    lines.append(
        f"| Specific | {_fmt(risk.specific_var, 6)} | {_pct(specific_vol)} | "
        f"{_pct(risk.specific_var / total_var)} |"
    )

    lines += [
        "",
        "## Top risk contributors",
        "",
        "| asset | weight | MCTR | CTR | % of vol |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for asset, ctr in contributors.items():
        lines.append(
            f"| {asset} | {_pct(weights[asset])} | {_fmt(risk.marginal_contributions[asset], 5)} "
            f"| {_fmt(ctr, 5)} | {_pct(risk.pct_contributions[asset])} |"
        )

    lines += [
        "",
        f"## Stress test — worst {len(stress)} non-overlapping {stress_window}-day windows",
        "",
        "| rank | start | end | days | PnL | loss |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in stress.itertuples(index=False):
        lines.append(
            f"| {int(row.rank)} | {row.start} | {row.end} | {int(row.n_days)} "
            f"| {_pct(row.pnl)} | {_pct(row.loss)} |"
        )

    lines += [
        "",
        "## Diversification",
        "",
        f"- Diversification ratio: **{_fmt(risk.diversification_ratio)}** "
        "(1 = no diversification, higher is better).",
        f"- Effective number of bets: **{_fmt(risk.effective_bets, 3)}** "
        f"out of {len(weights)} assets.",
        f"- Weight Herfindahl index: **{_fmt(float((weights**2).sum()), 4)}** "
        f"(effective N = {_fmt(1.0 / float((weights**2).sum()), 2)}).",
        "",
        "## Notes",
        "",
        "- Risk contributions are Euler allocations: they sum to portfolio volatility.",
        "- ENB is entropy-based; it is reported as `n/a` when a position contributes "
        "non-positive risk (possible with shorts).",
        "- Stress windows are non-overlapping and use arithmetic PnL.",
        "",
    ]

    data = {
        "summary": {
            "n_assets": int(len(weights)),
            "cov_method": cov_method,
            "ex_ante_vol": risk.ex_ante_vol,
            "ex_post_vol": risk.ex_post_vol,
            "factor_vol": factor_vol,
            "specific_vol": specific_vol,
            "diversification_ratio": risk.diversification_ratio,
            "effective_bets": risk.effective_bets,
            "beta": risk.beta,
            "tracking_error": risk.tracking_error,
        },
        "weights": weights_sorted.to_dict(),
        "risk": {
            "total_var": risk.total_var,
            "total_vol": risk.total_vol,
            "factor_var": risk.factor_var,
            "specific_var": risk.specific_var,
            "factor_contributions": risk.factor_contributions.to_dict(),
            "specific_contributions": risk.specific_contributions.to_dict(),
            "marginal_contributions": risk.marginal_contributions.to_dict(),
            "component_contributions": risk.component_contributions.to_dict(),
            "pct_contributions": risk.pct_contributions.to_dict(),
        },
        "stress": stress.to_dict(orient="records"),
        "diversification": {
            "ratio": risk.diversification_ratio,
            "effective_bets": risk.effective_bets,
            "herfindahl": float((weights**2).sum()),
            "effective_n": 1.0 / float((weights**2).sum()),
        },
        "meta": {
            "generated_by": "riskopt",
            "n_factors": risk.diagnostics["n_factors"],
            "model": risk.diagnostics["model"],
        },
    }
    return Report(markdown="\n".join(lines), data=data)
