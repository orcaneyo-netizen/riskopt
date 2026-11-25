"""Tests for the markdown/JSON report builder."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from riskopt.report import build_report


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(21)
    idx = pd.bdate_range("2025-01-01", periods=300)
    rets = pd.DataFrame(
        rng.normal(0.0003, 0.008, size=(300, 3)),
        index=idx,
        columns=["EQ", "BOND", "GOLD"],
    )
    weights = pd.Series([0.5, 0.3, 0.2], index=rets.columns)
    return rets, weights


REQUIRED_SECTIONS = [
    "# Portfolio risk report",
    "## Summary",
    "## Weights",
    "## Risk decomposition",
    "## Top risk contributors",
    "## Stress test",
    "## Diversification",
    "## Notes",
]


def test_report_builds_for_three_assets(panel):
    rets, weights = panel
    report = build_report(rets, weights, cov_method="ledoit-wolf")
    for section in REQUIRED_SECTIONS:
        assert section in report.markdown
    assert "EQ" in report.markdown
    assert "Total" in report.markdown

    payload = json.loads(report.to_json())
    assert set(payload) == {
        "summary", "weights", "risk", "stress", "diversification", "meta"
    }
    assert payload["summary"]["n_assets"] == 3
    assert sum(payload["weights"].values()) == pytest.approx(1.0, abs=1e-12)
    assert payload["summary"]["ex_ante_vol"] > 0
    assert len(payload["stress"]) == 3
    assert payload["meta"]["model"] == "aggregate"
    # Round-tripping through strict JSON must not encounter NaN.
    json.dumps(payload, allow_nan=False)


def test_report_with_factor_model(panel):
    rets, weights = panel
    rng = np.random.default_rng(22)
    factors = pd.DataFrame(
        rng.normal(0.0, 0.01, size=(300, 2)),
        index=rets.index,
        columns=["MKT", "RATES"],
    )
    report = build_report(rets, weights, factor_returns=factors, n_worst=2)
    payload = json.loads(report.to_json())
    assert payload["meta"]["n_factors"] == 2
    assert payload["meta"]["model"] == "factor"
    assert payload["risk"]["factor_var"] + payload["risk"]["specific_var"] == pytest.approx(
        payload["risk"]["total_var"], abs=1e-10
    )
    assert len(payload["stress"]) == 2
    assert "factor model: yes" in report.markdown


def test_report_with_benchmark(panel):
    rets, weights = panel
    bench = rets @ pd.Series([1 / 3, 1 / 3, 1 / 3], index=rets.columns)
    report = build_report(rets, weights, benchmark=bench)
    payload = json.loads(report.to_json())
    assert payload["summary"]["beta"] is not None
    assert payload["summary"]["tracking_error"] is not None
    assert "Beta vs benchmark" in report.markdown


def test_report_saves_files(panel, tmp_path):
    rets, weights = panel
    report = build_report(rets, weights)
    md_path = report.save(tmp_path / "report.md")
    json_path = report.save(tmp_path / "report.json")
    assert md_path.read_text(encoding="utf-8").startswith("# Portfolio risk report")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"]["n_assets"] == 3
    assert str(report) == report.markdown


def test_report_validation(panel):
    rets, weights = panel
    with pytest.raises(ValueError):
        build_report(rets, weights * 0.5)  # does not sum to 1
    with pytest.raises(ValueError):
        build_report(rets, pd.Series([1.0], index=["MISSING"]))
