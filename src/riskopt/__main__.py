"""Command line entry point: ``python -m riskopt demo``."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .allocators import equal_weight, hrp, max_sharpe, mean_cvar, min_variance, risk_parity
from .data import ledoit_wolf_cov, sample_cov, synthetic_returns
from .risk import PortfolioRisk
from .stress import historical_stress


def _cvar(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.05) -> float:
    losses = -(returns.loc[:, weights.index].to_numpy() @ weights.to_numpy())
    var = float(np.quantile(losses, 1.0 - alpha))
    tail = losses[losses >= var]
    return float(tail.mean()) if tail.size else float(losses.max())


def _turnover(prev: pd.Series, target: pd.Series) -> float:
    idx = prev.index.union(target.index)
    return float(0.5 * (target.reindex(idx, fill_value=0.0)
                        - prev.reindex(idx, fill_value=0.0)).abs().sum())


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) if rows else len(header[i])
              for i in range(len(header))]
    line = "| " + " | ".join(h.ljust(w) for h, w in zip(header, widths)) + " |"
    sep = "| " + " | ".join("-" * w for w in widths) + " |"
    body = ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join([line, sep, *body])


def demo(n_assets: int = 12, n_days: int = 1500, n_factors: int = 4, seed: int = 42,
         plot: bool = False, out: str | None = None) -> dict:
    """Run the end-to-end synthetic example and print a compact report."""
    returns, loadings, factors = synthetic_returns(
        n_assets=n_assets, n_days=n_days, n_factors=n_factors, seed=seed
    )
    cov, delta = ledoit_wolf_cov(returns)
    mu = returns.mean()

    ew = equal_weight(returns.columns)
    allocs = {
        "Equal weight": None,
        "Min variance": min_variance(cov),
        "Max Sharpe": max_sharpe(mu, cov),
        "Mean-CVaR": mean_cvar(returns, alpha=0.05),
        "Risk parity": risk_parity(cov),
        "HRP": hrp(returns),
    }
    weights = {"Equal weight": ew}
    weights.update({name: a.weights for name, a in allocs.items() if a is not None})

    print(f"riskopt {__version__} demo")
    print(f"universe: {n_assets} assets, {n_days} days, {n_factors} factors, seed={seed}")
    print(f"covariance: Ledoit-Wolf (shrinkage delta = {delta:.3f})")
    print()

    rows = []
    for name, w in weights.items():
        engine = PortfolioRisk(w, asset_returns=returns, factor_returns=factors)
        rep = engine.report()
        rows.append([
            name,
            f"{rep.ex_ante_vol:.4%}",
            f"{_cvar(returns, w):.4%}",
            f"{rep.diversification_ratio:.3f}",
            f"{rep.effective_bets:.2f}",
            f"{_turnover(ew, w):.2%}",
            f"{float(w.max()):.2%}",
        ])
    print("### Allocator comparison")
    print(_markdown_table(
        ["allocator", "ex-ante vol", "CVaR 95%", "div ratio", "ENB", "turnover", "max w"],
        rows,
    ))
    print()

    print("### Weights")
    table_rows = []
    for asset in returns.columns:
        table_rows.append([asset] + [f"{float(weights[name][asset]):.2%}" for name in weights])
    print(_markdown_table(["asset"] + list(weights), table_rows))
    print()

    rep = PortfolioRisk(weights["Min variance"], asset_returns=returns,
                        factor_returns=factors).report()
    print("### Risk decomposition (min variance)")
    print(_markdown_table(
        ["component", "variance", "volatility", "share"],
        [
            ["total", f"{rep.total_var:.6e}", f"{rep.total_vol:.4%}", "100.00%"],
            ["factor", f"{rep.factor_var:.6e}", f"{np.sqrt(max(rep.factor_var, 0)):.4%}",
             f"{rep.factor_var / rep.total_var:.2%}"],
            ["specific", f"{rep.specific_var:.6e}",
             f"{np.sqrt(max(rep.specific_var, 0)):.4%}",
             f"{rep.specific_var / rep.total_var:.2%}"],
        ],
    ))
    top = rep.component_contributions.sort_values(ascending=False).head(3)
    print("top risk contributors: " + ", ".join(
        f"{a} {rep.pct_contributions[a]:.1%}" for a in top.index))
    print()

    stress = historical_stress(weights["Min variance"], returns, window=21, n_worst=3)
    print("### Worst 21-day windows (min variance)")
    print(_markdown_table(
        ["rank", "start", "end", "PnL"],
        [[str(int(r.rank)), str(r.start.date()), str(r.end.date()), f"{r.pnl:.2%}"]
         for r in stress.itertuples(index=False)],
    ))

    if plot:
        target = Path(out or "hrp_dendrogram.png")
        _plot_dendrogram(returns, target)
        print(f"\nsaved HRP dendrogram -> {target}")

    return {
        "returns": returns,
        "weights": weights,
        "diagnostics": {name: a.diagnostics for name, a in allocs.items() if a is not None},
    }


def _plot_dendrogram(returns: pd.DataFrame, target: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import dendrogram

    alloc = hrp(returns)
    fig, ax = plt.subplots(figsize=(8, 4))
    dendrogram(np.asarray(alloc.diagnostics["linkage_matrix"]),
               labels=list(returns.columns), ax=ax)
    ax.set_title("HRP single-linkage dendrogram (correlation distance)")
    ax.set_ylabel("distance")
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="riskopt", description=__doc__)
    parser.add_argument("command", nargs="?", default="demo", choices=["demo"],
                        help="command to run (default: demo)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--assets", type=int, default=12)
    parser.add_argument("--days", type=int, default=1500)
    parser.add_argument("--factors", type=int, default=4)
    parser.add_argument("--plot", action="store_true", help="save the HRP dendrogram PNG")
    parser.add_argument("--out", type=str, default=None, help="output path for --plot")
    args = parser.parse_args(argv)
    demo(n_assets=args.assets, n_days=args.days, n_factors=args.factors,
         seed=args.seed, plot=args.plot, out=args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
