"""Tests for the turnover / transaction-cost model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskopt.costs import TurnoverCostModel, net_weights, trade_list


@pytest.fixture()
def weights():
    prev = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2})
    target = pd.Series({"A": 0.3, "B": 0.4, "C": 0.3})
    return prev, target


def test_zero_turnover_has_zero_cost(weights):
    prev, _ = weights
    model = TurnoverCostModel(linear_bps=12.5, impact_coef=0.7)
    assert model.total_cost(prev, prev) == pytest.approx(0.0)
    assert np.allclose(model.cost_series(prev, prev).to_numpy(), 0.0)
    blotter = model.trade_list(prev, prev)
    assert blotter.empty
    assert list(blotter.columns) == [
        "asset", "prev_weight", "target_weight", "delta_weight", "notional", "side"
    ]


def test_cost_is_monotone_in_trade_size(weights):
    prev, target = weights
    model = TurnoverCostModel(linear_bps=5.0, impact_coef=0.2)
    small = pd.Series({"A": 0.5, "B": 0.3, "C": 0.2 + 0.02})
    small["A"] = 0.48
    large = pd.Series({"A": 0.30, "B": 0.30, "C": 0.40})
    assert model.total_cost(prev, small) < model.total_cost(prev, large)
    # strict monotonicity asset by asset
    c_small = model.cost_series(prev, small)
    c_large = model.cost_series(prev, large)
    assert (c_large["C"] > c_small["C"])


def test_cost_is_monotone_in_impact_coefficient(weights):
    prev, target = weights
    low = TurnoverCostModel(linear_bps=5.0, impact_coef=0.05)
    high = TurnoverCostModel(linear_bps=5.0, impact_coef=0.50)
    assert high.total_cost(prev, target) > low.total_cost(prev, target)
    # with zero linear cost, a zero impact model must be free
    free = TurnoverCostModel(linear_bps=0.0, impact_coef=0.0)
    assert free.total_cost(prev, target) == pytest.approx(0.0)


def test_cost_matches_closed_form(weights):
    prev, target = weights
    model = TurnoverCostModel(linear_bps=10.0, impact_coef=0.3)
    sigma = pd.Series({"A": 0.02, "B": 0.015, "C": 0.03})
    cost = model.cost_series(prev, target, sigma=sigma)
    dw = (target - prev).abs()
    expected = 10.0 / 1e4 * dw + 0.3 * dw**1.5 * sigma
    np.testing.assert_allclose(cost.to_numpy(), expected.to_numpy(), atol=1e-15)
    assert model.total_cost(prev, target, sigma=sigma) == pytest.approx(expected.sum())


def test_trade_list_reconciles_to_weight_change(weights):
    prev, target = weights
    model = TurnoverCostModel()
    nav = 25_000_000.0
    blotter = model.trade_list(prev, target, nav=nav)
    assert set(blotter["asset"]) == {"A", "B", "C"}
    recon = blotter.set_index("asset")["delta_weight"]
    np.testing.assert_allclose(
        recon.reindex(target.index).to_numpy(), (target - prev).to_numpy(), atol=1e-15
    )
    np.testing.assert_allclose(blotter["notional"].to_numpy(),
                               blotter["delta_weight"].to_numpy() * nav, atol=1e-6)
    assert blotter.loc[blotter["delta_weight"] > 0, "side"].eq("BUY").all()
    assert blotter.loc[blotter["delta_weight"] < 0, "side"].eq("SELL").all()
    # Fully invested book: buys and sells net to zero notional.
    assert blotter["notional"].sum() == pytest.approx(0.0, abs=1e-6)


def test_net_weights_apply_cost_drag(weights):
    prev, target = weights
    model = TurnoverCostModel(linear_bps=8.0, impact_coef=0.1)
    total = model.total_cost(prev, target)
    net = net_weights(prev, target, model)
    assert total > 0
    assert net.sum() == pytest.approx(1.0 - total, abs=1e-15)
    np.testing.assert_allclose(net.to_numpy(), (1.0 - total) * target.to_numpy())
    # net weights preserve the relative structure of the target
    np.testing.assert_allclose(net / target, np.full(len(target), 1.0 - total))


def test_turnover_metric(weights):
    prev, target = weights
    model = TurnoverCostModel()
    dw = (target - prev).abs().sum()
    assert model.turnover(prev, target) == pytest.approx(0.5 * dw)
    assert model.turnover(prev, prev) == pytest.approx(0.0)


def test_trade_list_function_and_validation(weights):
    prev, target = weights
    blotter = trade_list(prev, target, nav=1_000.0)
    assert len(blotter) == 3
    with pytest.raises(ValueError):
        TurnoverCostModel(linear_bps=-1.0)
    with pytest.raises(ValueError):
        TurnoverCostModel(impact_coef=-0.5)
    with pytest.raises(ValueError):
        TurnoverCostModel().trade_list(prev, target, nav=0.0)
    with pytest.raises(ValueError):
        net_weights(prev, target, "not a model")
