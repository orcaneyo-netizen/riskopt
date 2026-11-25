"""Tests for historical, factor and grid stress testing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskopt.stress import factor_shock, historical_stress, scenario_grid


def _panel(n_days: int = 240, n_assets: int = 3, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2024-01-01", periods=n_days)
    return pd.DataFrame(rng.normal(0.0, 0.005, size=(n_days, n_assets)),
                        index=idx, columns=[f"A{i}" for i in range(n_assets)])


def test_single_factor_linear_shock_matches_closed_form():
    loadings = pd.DataFrame({"MKT": [1.0, 0.5, -0.25]}, index=["A", "B", "C"])
    weights = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "C"])
    result = factor_shock(weights, loadings, {"MKT": 0.10})
    expected = float(weights @ (loadings.to_numpy() @ np.array([0.10])))
    assert result.linear_pnl == pytest.approx(expected, abs=1e-15)
    assert result.total_pnl == pytest.approx(expected, abs=1e-15)
    assert result.quadratic_pnl == 0.0
    assert result.factor_contributions.sum() == pytest.approx(result.linear_pnl, abs=1e-15)
    assert result.per_asset.sum() == pytest.approx(result.linear_pnl, abs=1e-15)
    assert result.loss == pytest.approx(-result.linear_pnl)


def test_quadratic_convexity_term():
    loadings = pd.DataFrame({"MKT": [1.0, 0.5]}, index=["A", "B"])
    weights = pd.Series([0.6, 0.4], index=["A", "B"])
    gamma = np.array([[-2.0]])
    result = factor_shock(weights, loadings, np.array([0.10]), convexity=gamma)
    assert result.quadratic_pnl == pytest.approx(0.5 * 0.10**2 * -2.0)
    assert result.total_pnl == pytest.approx(result.linear_pnl + result.quadratic_pnl)


def test_multi_factor_shock_decomposes_additively():
    rng = np.random.default_rng(4)
    loadings = pd.DataFrame(rng.normal(0.5, 0.2, size=(5, 3)),
                            index=[f"A{i}" for i in range(5)],
                            columns=["EQ", "RATES", "CREDIT"])
    weights = pd.Series(np.full(5, 0.2), index=loadings.index)
    shocks = pd.Series({"EQ": -0.1, "RATES": 0.02, "CREDIT": -0.03})
    result = factor_shock(weights, loadings, shocks)
    expected = float(weights @ loadings.to_numpy() @ shocks.to_numpy())
    assert result.linear_pnl == pytest.approx(expected, abs=1e-15)
    assert result.factor_contributions.sum() == pytest.approx(expected, abs=1e-15)


def test_historical_stress_finds_planted_worst_window():
    rets = _panel(n_days=200, n_assets=3, seed=2)
    rets.iloc[100:120] = -0.05  # planted crash
    weights = pd.Series([0.4, 0.35, 0.25], index=rets.columns)
    table = historical_stress(weights, rets, window=20, n_worst=3)
    assert list(table.columns) == [
        "window_id", "start", "end", "n_days", "pnl", "loss", "cum_return", "rank"
    ]
    worst = table.iloc[0]
    assert worst["start"] == rets.index[100]
    assert worst["end"] == rets.index[119]
    assert worst["pnl"] == pytest.approx(-1.0)
    assert table["pnl"].is_monotonic_increasing
    assert (table["loss"] == -table["pnl"]).all()


def test_historical_stress_explicit_windows_and_factor_inputs():
    rets = _panel(n_days=120, n_assets=4, seed=3)
    weights = pd.Series(np.full(4, 0.25), index=rets.columns)
    table = historical_stress(
        weights, rets, windows=[(0, 10), (10, 20)], n_worst=2
    )
    assert len(table) == 2
    assert table["n_days"].tolist() == [10, 10]

    factors = pd.DataFrame(
        np.zeros((60, 2)), index=rets.index[:60], columns=["F1", "F2"]
    )
    factors.iloc[30:40, 0] = -0.02
    loadings = pd.DataFrame(
        np.ones((4, 2)), index=rets.columns, columns=["F1", "F2"]
    )
    stressed = historical_stress(
        weights, factors, window=10, n_worst=2, loadings=loadings
    )
    assert stressed.iloc[0]["start"] == factors.index[30]
    assert stressed.iloc[0]["pnl"] == pytest.approx(-0.2)


def test_scenario_grid_shape_and_columns():
    rng = np.random.default_rng(5)
    loadings = pd.DataFrame(
        rng.normal(0.5, 0.2, size=(6, 3)),
        index=[f"A{i}" for i in range(6)],
        columns=["equity", "rates", "credit"],
    )
    weights = pd.Series(np.full(6, 1.0 / 6), index=loadings.index)
    grid = scenario_grid(weights, loadings)
    assert list(grid.columns) == ["scenario", "equity", "rates", "credit", "pnl", "loss"]
    assert grid.shape == (4 * 3 * 3, 6)
    assert grid["scenario"].tolist() == list(range(36))
    np.testing.assert_allclose(grid["loss"].to_numpy(), -grid["pnl"].to_numpy())
    assert grid["pnl"].abs().sum() > 0

    custom = scenario_grid(
        weights,
        loadings,
        factor_map={"equity": "rates", "rates": "credit", "credit": "equity"},
        shocks={"equity": (-0.1,), "rates": (0.0,), "credit": (0.1,)},
    )
    assert custom.shape == (1, 6)
    assert custom.iloc[0]["equity"] == pytest.approx(-0.1)


def test_stress_validation():
    loadings = pd.DataFrame({"MKT": [1.0, 0.5]}, index=["A", "B"])
    weights = pd.Series([0.5, 0.5], index=["A", "B"])
    with pytest.raises(ValueError):
        factor_shock(weights, loadings, {"UNKNOWN": 0.1})
    with pytest.raises(ValueError):
        factor_shock(weights, loadings, np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        factor_shock(weights, loadings, 0.1, convexity=np.eye(2))
    rets = _panel(n_days=30, n_assets=2)
    with pytest.raises(ValueError):
        historical_stress(weights, rets, window=100)
    with pytest.raises(ValueError):
        historical_stress(weights, rets, windows=[(5, 5)])
    with pytest.raises(ValueError):
        scenario_grid(weights, loadings)  # only one factor
