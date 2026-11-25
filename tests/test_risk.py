"""Tests for the risk engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskopt.allocators import risk_parity
from riskopt.data import sample_cov, synthetic_returns
from riskopt.risk import PortfolioRisk, estimate_loadings

NAMES = ["A", "B", "C", "D"]
B_TRUE = np.array(
    [
        [1.0, 0.0],
        [0.8, 0.3],
        [0.2, 1.0],
        [-0.5, 0.4],
    ]
)
F_TRUE = np.array([[0.04, 0.01], [0.01, 0.09]])
D_TRUE = np.array([0.010, 0.020, 0.030, 0.015])
W = np.array([0.30, 0.25, 0.25, 0.20])


def _model() -> PortfolioRisk:
    weights = pd.Series(W, index=NAMES)
    loadings = pd.DataFrame(B_TRUE, index=NAMES, columns=["F1", "F2"])
    factor_cov = pd.DataFrame(F_TRUE, index=["F1", "F2"], columns=["F1", "F2"])
    specific = pd.Series(D_TRUE, index=NAMES)
    return PortfolioRisk.from_factor_model(weights, loadings, factor_cov, specific)


def test_factor_plus_specific_equals_total_variance():
    engine = _model()
    w = W
    expected = w @ B_TRUE @ F_TRUE @ B_TRUE.T @ w + float(w**2 @ D_TRUE)
    assert engine.total_variance() == pytest.approx(expected, abs=1e-12)

    report = engine.report()
    assert report.factor_var + report.specific_var == pytest.approx(report.total_var, abs=1e-12)
    assert report.factor_contributions.sum() == pytest.approx(report.factor_var, abs=1e-12)
    assert report.specific_contributions.sum() == pytest.approx(report.specific_var, abs=1e-12)
    assert report.total_vol == pytest.approx(np.sqrt(expected), abs=1e-12)


def test_component_contributions_sum_to_volatility():
    report = _model().report()
    ctr = report.component_contributions
    assert ctr.sum() == pytest.approx(report.total_vol, abs=1e-10)
    np.testing.assert_allclose(
        ctr.to_numpy(), (W * (report.cov.to_numpy() @ W) / report.total_vol), atol=1e-12
    )
    assert report.pct_contributions.sum() == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(
        report.marginal_contributions.to_numpy(),
        (report.cov.to_numpy() @ W) / report.total_vol,
        atol=1e-12,
    )


def test_diversification_ratio_is_at_least_one():
    report = _model().report()
    assert report.diversification_ratio >= 1.0 - 1e-12
    # single asset: no diversification
    engine = PortfolioRisk.from_factor_model(
        pd.Series([1.0], index=["A"]),
        pd.DataFrame([[1.0]], index=["A"], columns=["F1"]),
        pd.DataFrame([[0.04]], index=["F1"], columns=["F1"]),
        pd.Series([0.01], index=["A"]),
    )
    assert engine.diversification_ratio() == pytest.approx(1.0)


def test_single_asset_has_effective_bets_of_one():
    engine = PortfolioRisk.from_factor_model(
        pd.Series([1.0], index=["A"]),
        pd.DataFrame([[1.0]], index=["A"], columns=["F1"]),
        pd.DataFrame([[0.04]], index=["F1"], columns=["F1"]),
        pd.Series([0.01], index=["A"]),
    )
    report = engine.report()
    assert report.effective_bets == pytest.approx(1.0, abs=1e-12)
    assert report.component_contributions.sum() == pytest.approx(report.total_vol)


def test_effective_bets_equals_n_for_equal_risk_contributions():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(6, 6))
    cov = pd.DataFrame(a @ a.T + np.diag(np.full(6, 0.5)),
                       index=[f"A{i}" for i in range(6)],
                       columns=[f"A{i}" for i in range(6)])
    weights = risk_parity(cov).weights
    engine = PortfolioRisk(weights, cov=cov)
    assert engine.effective_bets() == pytest.approx(6.0, abs=1e-6)


def test_benchmark_beta_of_benchmark_is_one():
    rets, _, _ = synthetic_returns(n_assets=4, n_days=300, n_factors=2, seed=13)
    cov = sample_cov(rets)
    bench_w = pd.Series([0.4, 0.3, 0.2, 0.1], index=rets.columns)
    bench_returns = rets @ bench_w
    engine = PortfolioRisk(bench_w, asset_returns=rets, benchmark=bench_returns, cov=cov)
    assert engine.beta() == pytest.approx(1.0, abs=1e-12)
    assert engine.tracking_error() == pytest.approx(0.0, abs=1e-18)


def test_beta_against_a_weights_benchmark():
    rets, _, _ = synthetic_returns(n_assets=4, n_days=300, n_factors=2, seed=14)
    bench_w = pd.Series([0.25, 0.25, 0.25, 0.25], index=rets.columns)
    port_w = pd.Series([0.1, 0.2, 0.3, 0.4], index=rets.columns)
    engine = PortfolioRisk(port_w, asset_returns=rets, cov=sample_cov(rets))
    port = rets @ port_w
    bench = rets @ bench_w
    manual = float(np.cov(port, bench, ddof=1)[0, 1] / np.var(bench, ddof=1))
    assert engine.beta(bench_w) == pytest.approx(manual, abs=1e-12)
    assert engine.beta(bench_w) == engine.beta(bench)


def test_ex_ante_and_ex_post_volatility():
    rets, _, _ = synthetic_returns(n_assets=5, n_days=400, n_factors=2, seed=15)
    weights = pd.Series(np.full(5, 0.2), index=rets.columns)
    engine = PortfolioRisk(weights, asset_returns=rets, cov=sample_cov(rets))
    report = engine.report()
    assert report.ex_post_vol == pytest.approx(float((rets @ weights).std(ddof=1)))
    assert report.ex_ante_vol == pytest.approx(engine.volatility())


def test_ols_loadings_recover_the_planted_model():
    rets, loadings, factors = synthetic_returns(n_assets=12, n_days=1500, n_factors=4, seed=42)
    estimated, alpha, specific = estimate_loadings(rets, factors)
    assert estimated.shape == loadings.shape
    np.testing.assert_allclose(estimated.to_numpy(), loadings.to_numpy(), atol=0.12)
    assert (specific > 0).all()
    assert np.isfinite(alpha.to_numpy()).all()


def test_aggregate_covariance_mode():
    rets, _, _ = synthetic_returns(n_assets=3, n_days=200, n_factors=1, seed=16)
    weights = pd.Series([0.5, 0.3, 0.2], index=rets.columns)
    cov = sample_cov(rets)
    engine = PortfolioRisk(weights, asset_returns=rets, cov=cov)
    report = engine.report()
    assert report.factor_var == 0.0
    assert report.specific_var == pytest.approx(report.total_var)
    assert report.diagnostics["model"] == "aggregate"
    assert report.component_contributions.sum() == pytest.approx(report.total_vol)


def test_input_validation():
    with pytest.raises(ValueError):
        PortfolioRisk(pd.Series([0.5, 0.5], index=["A", "B"]))  # no data
    cov = pd.DataFrame([[0.04, 0.01], [0.01, 0.09]], index=["A", "B"], columns=["A", "B"])
    with pytest.raises(ValueError):
        PortfolioRisk(pd.Series([0.7, 0.7], index=["A", "B"]), cov=cov)  # sum != 1
    with pytest.raises(ValueError):
        PortfolioRisk(pd.Series([0.5, 0.5], index=["A", "B"]), cov=cov).beta()
    with pytest.raises(ValueError):
        estimate_loadings(pd.DataFrame({"A": [1.0, 2.0]}), pd.DataFrame({"F": [1.0, 2.0]}))
