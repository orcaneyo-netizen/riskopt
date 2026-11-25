"""Tests for synthetic data and covariance estimators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskopt.data import (
    ewma_weights,
    is_psd,
    ledoit_wolf_cov,
    project_psd,
    sample_cov,
    synthetic_returns,
)


@pytest.fixture(scope="module")
def panel():
    return synthetic_returns(n_assets=12, n_days=1000, n_factors=4, seed=7)


def test_synthetic_returns_are_deterministic(panel):
    rets, loadings, factors = panel
    rets2, loadings2, factors2 = synthetic_returns(12, 1000, 4, seed=7)
    pd.testing.assert_frame_equal(rets, rets2)
    pd.testing.assert_frame_equal(loadings, loadings2)
    pd.testing.assert_frame_equal(factors, factors2)

    other = synthetic_returns(12, 1000, 4, seed=8)
    assert not rets.equals(other[0])


def test_synthetic_shapes_and_labels(panel):
    rets, loadings, factors = panel
    assert rets.shape == (1000, 12)
    assert loadings.shape == (12, 4)
    assert factors.shape == (1000, 4)
    assert list(loadings.index) == list(rets.columns)
    assert list(loadings.columns) == list(factors.columns)
    assert rets.index.is_monotonic_increasing
    assert np.isfinite(rets.to_numpy()).all()


def test_synthetic_innovations_are_fat_tailed(panel):
    _, _, factors = panel
    excess_kurtosis = pd.Series(factors.to_numpy().ravel()).kurtosis()
    assert excess_kurtosis > 3.0, "Student-t(5) innovations should be fat tailed"


def test_covariance_is_psd_for_every_method(panel):
    rets, _, _ = panel
    for method in ("sample", "ledoit-wolf", "ewma"):
        cov = sample_cov(rets, method=method)
        assert cov.shape == (12, 12)
        assert np.allclose(cov.to_numpy(), cov.to_numpy().T, atol=1e-12)
        assert is_psd(cov.to_numpy()), method
        assert np.all(np.linalg.eigvalsh(cov.to_numpy()) > -1e-10)


def test_ledoit_wolf_sits_between_sample_and_diagonal(panel):
    rets, _, _ = panel
    lw, delta = ledoit_wolf_cov(rets)
    sample = sample_cov(rets, "sample")
    diagonal = pd.DataFrame(np.diag(np.diag(sample.to_numpy())),
                            index=sample.index, columns=sample.columns)

    assert 0.0 <= delta <= 1.0
    assert 0.0 < delta < 1.0, "shrinkage should be non-trivial on this panel"
    assert is_psd(lw.to_numpy())
    assert not np.allclose(lw.to_numpy(), sample.to_numpy())

    # The shrunk estimate must be closer to the sample covariance than the
    # diagonal target is (it is a convex combination of sample and target).
    dist_lw = np.linalg.norm(lw.to_numpy() - sample.to_numpy())
    dist_diag = np.linalg.norm(diagonal.to_numpy() - sample.to_numpy())
    assert dist_lw < dist_diag

    # Constant-correlation shrinkage homogenises off-diagonal correlations.
    def offdiag_corr(cov: pd.DataFrame) -> np.ndarray:
        sd = np.sqrt(np.diag(cov.to_numpy()))
        corr = cov.to_numpy() / np.outer(sd, sd)
        return corr[np.triu_indices_from(corr, k=1)]

    assert offdiag_corr(lw).std() < offdiag_corr(sample).std()


def test_ewma_weights_decay_geometrically():
    weights = ewma_weights(10, halflife=2.0)
    assert weights.sum() == pytest.approx(1.0)
    assert np.all(np.diff(weights) > 0), "recent observations get more weight"
    assert np.allclose(weights[1:] / weights[:-1], 2.0 ** (1.0 / 2.0))
    assert ewma_weights(100, 10)[-1] > ewma_weights(100, 60)[-1]


def test_ewma_covariance_reacts_to_recent_volatility():
    rng = np.random.default_rng(3)
    n = 400
    data = rng.normal(0.0, 0.01, size=(n, 3))
    data[n // 2 :] *= 5.0  # volatility regime shift
    frame = pd.DataFrame(data, columns=list("ABC"))

    fast = sample_cov(frame, "ewma", halflife=10)
    slow = sample_cov(frame, "ewma", halflife=200)
    plain = sample_cov(frame, "sample")
    assert np.all(np.diag(fast) > np.diag(plain))
    assert np.all(np.diag(fast) > np.diag(slow))
    assert is_psd(fast.to_numpy())


def test_psd_projection_repairs_indefinite_matrix():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]])
    fixed = project_psd(bad)
    assert is_psd(fixed)
    assert np.allclose(fixed, fixed.T)


def test_invalid_inputs_raise(panel):
    rets, _, _ = panel
    with pytest.raises(ValueError):
        sample_cov(rets, method="nope")
    with pytest.raises(ValueError):
        synthetic_returns(n_assets=3, n_factors=3)
    with pytest.raises(ValueError):
        synthetic_returns(n_assets=5, n_days=4, n_factors=2)
    with pytest.raises(ValueError):
        ewma_weights(5, halflife=0)

    dirty = rets.copy()
    dirty.iloc[3, 2] = np.nan
    with pytest.raises(ValueError):
        sample_cov(dirty, nan_policy="raise")
    with pytest.warns(UserWarning):
        dropped = sample_cov(dirty, nan_policy="drop")
    assert dropped.shape == (12, 12)
