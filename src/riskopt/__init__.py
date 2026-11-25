"""riskopt — portfolio construction, allocation and risk analytics.

Quick start
-----------
>>> from riskopt import synthetic_returns, sample_cov, min_variance
>>> rets, loadings, factors = synthetic_returns(seed=42)
>>> alloc = min_variance(sample_cov(rets, "ledoit-wolf"))
>>> round(float(alloc.weights.sum()), 12)
1.0
"""

from __future__ import annotations

from .allocators import (
    Allocation,
    equal_weight,
    hrp,
    max_sharpe,
    mean_cvar,
    min_variance,
    risk_parity,
)
from .black_litterman import BLResult, BlackLitterman
from .costs import TurnoverCostModel, net_weights, trade_list
from .data import (
    annualize_cov,
    ewma_weights,
    is_psd,
    ledoit_wolf_cov,
    project_psd,
    sample_cov,
    synthetic_returns,
)
from .report import Report, build_report
from .risk import PortfolioRisk, RiskReport, estimate_loadings
from .stress import FactorShockResult, factor_shock, historical_stress, scenario_grid

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # data
    "synthetic_returns",
    "sample_cov",
    "ledoit_wolf_cov",
    "ewma_weights",
    "project_psd",
    "is_psd",
    "annualize_cov",
    # allocators
    "Allocation",
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "mean_cvar",
    "risk_parity",
    "hrp",
    # black-litterman
    "BlackLitterman",
    "BLResult",
    # costs
    "TurnoverCostModel",
    "net_weights",
    "trade_list",
    # risk
    "PortfolioRisk",
    "RiskReport",
    "estimate_loadings",
    # stress
    "historical_stress",
    "factor_shock",
    "scenario_grid",
    "FactorShockResult",
    # report
    "build_report",
    "Report",
]
