"""Tests for the allocators: constraints, optimality and risk budgets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskopt.allocators import (
    equal_weight,
    hrp,
    max_sharpe,
    mean_cvar,
    min_variance,
    risk_parity,
)
from riskopt.data import sample_cov, synthetic_returns


@pytest.fixture(scope="module")
def panel():
    rets, _, _ = synthetic_returns(n_assets=12, n_days=800, n_factors=4, seed=11)
    cov = sample_cov(rets, "ledoit-wolf")
    return rets, cov, rets.mean()


def _vol(weights: pd.Series, cov: pd.DataFrame) -> float:
    w = weights.to_numpy()
    return float(np.sqrt(w @ cov.to_numpy() @ w))


def _sharpe(weights: pd.Series, mu: pd.Series, cov: pd.DataFrame, rf: float = 0.0) -> float:
    w = weights.to_numpy()
    return float((mu @ w - rf) / np.sqrt(w @ cov.to_numpy() @ w))


def _cvar(returns: pd.DataFrame, weights: pd.Series, alpha: float = 0.05) -> float:
    losses = -(returns.to_numpy() @ weights.to_numpy())
    var = float(np.quantile(losses, 1.0 - alpha))
    tail = losses[losses >= var]
    return float(tail.mean())


SECTORS = {f"A{i:02d}": f"S{(i - 1) % 3}" for i in range(1, 13)}
CAPS = {"S0": 0.45, "S1": 0.45, "S2": 0.45}


def test_all_allocators_are_fully_invested(panel):
    rets, cov, mu = panel
    results = {
        "min_variance": min_variance(cov),
        "max_sharpe": max_sharpe(mu, cov),
        "mean_cvar": mean_cvar(rets),
        "risk_parity": risk_parity(cov),
        "hrp": hrp(rets),
    }
    for name, alloc in results.items():
        w = alloc.weights
        assert w.sum() == pytest.approx(1.0, abs=1e-10), name
        assert np.all(w.to_numpy() >= -1e-12), name
        assert alloc.diagnostics["sum_weights"] == pytest.approx(1.0, abs=1e-10)


def test_max_weight_is_respected(panel):
    rets, cov, mu = panel
    for alloc in (
        min_variance(cov, max_weight=0.2),
        max_sharpe(mu, cov, max_weight=0.2),
        mean_cvar(rets, max_weight=0.2),
    ):
        assert alloc.weights.max() <= 0.2 + 1e-6
        assert alloc.weights.sum() == pytest.approx(1.0, abs=1e-10)


def test_sector_caps_are_respected(panel):
    rets, cov, mu = panel
    for alloc in (
        min_variance(cov, sector_map=SECTORS, sector_caps=CAPS),
        max_sharpe(mu, cov, sector_map=SECTORS, sector_caps=CAPS),
        mean_cvar(rets, sector_map=SECTORS, sector_caps=CAPS),
    ):
        totals = alloc.weights.groupby(pd.Series(SECTORS).reindex(alloc.weights.index)).sum()
        assert (totals <= 0.45 + 1e-6).all()
        assert alloc.weights.sum() == pytest.approx(1.0, abs=1e-10)


def test_min_variance_beats_equal_weight(panel):
    _, cov, _ = panel
    ew = equal_weight(cov.columns)
    mv = min_variance(cov)
    assert _vol(mv.weights, cov) <= _vol(ew, cov) + 1e-12
    # It should be strictly better on a non-degenerate covariance matrix.
    assert _vol(mv.weights, cov) < _vol(ew, cov)


def test_max_sharpe_beats_min_variance_sharpe(panel):
    _, cov, mu = panel
    mv = min_variance(cov)
    ms = max_sharpe(mu, cov)
    assert _sharpe(ms.weights, mu, cov) >= _sharpe(mv.weights, mu, cov) - 1e-9


def test_mean_cvar_beats_equal_weight_cvar(panel):
    rets, _, _ = panel
    ew = equal_weight(rets.columns)
    alloc = mean_cvar(rets, alpha=0.05)
    assert alloc.diagnostics["cvar"] <= _cvar(rets, ew) + 1e-12
    assert _cvar(rets, alloc.weights) <= _cvar(rets, ew) + 1e-12
    assert 0.0 < alloc.diagnostics["alpha"] < 1.0


def test_mean_cvar_respects_target_return(panel):
    rets, _, _ = panel
    target = float(rets.mean().mean())  # equal-weight expected return
    alloc = mean_cvar(rets, target_return=target)
    achieved = float(rets.mean() @ alloc.weights)
    assert achieved >= target - 1e-9


def test_risk_parity_equalises_contributions(panel):
    _, cov, _ = panel
    alloc = risk_parity(cov)
    w = alloc.weights.to_numpy()
    sigma_w = cov.to_numpy() @ w
    vol = np.sqrt(w @ sigma_w)
    rc = w * sigma_w / vol
    pct = rc / vol
    assert np.all(w > 0)
    assert np.max(np.abs(pct - 1.0 / len(w))) <= 1e-8
    assert rc.sum() == pytest.approx(vol, abs=1e-10)
    assert alloc.diagnostics["rc_deviation"] <= 1e-8


def test_risk_parity_honours_budget(panel):
    _, cov, _ = panel
    n = cov.shape[0]
    budget = np.full(n, 0.3 / (n - 1))
    budget[0] = 0.7
    alloc = risk_parity(cov, budget=budget)
    w = alloc.weights.to_numpy()
    sigma_w = cov.to_numpy() @ w
    vol = np.sqrt(w @ sigma_w)
    pct = w * sigma_w / vol**2
    assert np.max(np.abs(pct - budget)) <= 1e-8


def test_hrp_weights_and_quasi_diagonalisation(panel):
    rets, _, _ = panel
    alloc = hrp(rets)
    assert alloc.weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(alloc.weights.to_numpy() >= 0)
    assert alloc.diagnostics["coherence_after"] > alloc.diagnostics["coherence_before"]

    order = alloc.diagnostics["order"]
    assert sorted(order) == sorted(rets.columns)
    corr = rets.corr().loc[order, order]
    band = sum(
        abs(corr.iloc[i, j]) for i in range(len(order)) for j in range(len(order))
        if 0 < abs(i - j) <= 2
    ) / sum(1 for i in range(len(order)) for j in range(len(order)) if 0 < abs(i - j) <= 2)
    assert band > alloc.diagnostics["coherence_before"]


def test_allocators_are_deterministic(panel):
    rets, cov, mu = panel
    pairs = [
        (min_variance(cov, max_weight=0.25), min_variance(cov, max_weight=0.25)),
        (max_sharpe(mu, cov), max_sharpe(mu, cov)),
        (mean_cvar(rets), mean_cvar(rets)),
        (risk_parity(cov), risk_parity(cov)),
        (hrp(rets), hrp(rets)),
    ]
    for first, second in pairs:
        np.testing.assert_array_equal(first.weights.to_numpy(), second.weights.to_numpy())


def test_single_asset_degenerate_case():
    cov = pd.DataFrame([[0.04]], index=["A"], columns=["A"])
    alloc = min_variance(cov)
    assert alloc.weights.tolist() == [1.0]
    rp = risk_parity(cov)
    assert rp.weights.tolist() == [1.0]


def test_infeasible_constraints_raise(panel):
    rets, cov, mu = panel
    with pytest.raises(ValueError):
        min_variance(cov, max_weight=0.05)
    with pytest.raises(ValueError):
        max_sharpe(mu, cov, sector_map=SECTORS, sector_caps={"S0": 0.2, "S1": 0.2, "S2": 0.2})
    with pytest.raises(ValueError):
        mean_cvar(rets, target_return=10.0)
    with pytest.raises(ValueError):
        risk_parity(cov, budget=np.zeros(12))
