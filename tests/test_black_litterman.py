"""Tests for the Black-Litterman engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from riskopt.black_litterman import BlackLitterman
from riskopt.data import is_psd, sample_cov, synthetic_returns


@pytest.fixture(scope="module")
def setup():
    rets, _, _ = synthetic_returns(n_assets=6, n_days=600, n_factors=2, seed=5)
    cov = sample_cov(rets, "sample")
    n = cov.shape[0]
    market = pd.Series(np.full(n, 1.0 / n), index=cov.columns)
    return cov, market


def test_zero_views_reproduce_the_prior(setup):
    cov, market = setup
    bl = BlackLitterman(tau=0.05, risk_aversion=2.5)
    prior = bl.implied_prior(cov, market)
    p = np.empty((0, cov.shape[0]))
    post_mean, post_cov = bl.posterior(prior, cov, p, np.array([]))
    pd.testing.assert_series_equal(post_mean, prior, check_names=False)
    expected = (1.0 + bl.tau) * cov
    np.testing.assert_allclose(post_cov.to_numpy(), expected.to_numpy(), atol=1e-12)
    assert is_psd(post_cov.to_numpy())


def test_strong_positive_view_raises_posterior_mean(setup):
    cov, market = setup
    bl = BlackLitterman(tau=0.05, risk_aversion=2.5)
    prior = bl.implied_prior(cov, market)
    n = cov.shape[0]
    j = 2
    p = np.zeros((1, n))
    p[0, j] = 1.0
    q = np.array([float(prior.iloc[j]) + 0.05])  # +5% per-period view
    post_mean, post_cov = bl.posterior(prior, cov, p, q)

    assert post_mean.iloc[j] > prior.iloc[j]
    assert post_mean.iloc[j] < q[0]
    # With Omega = tau * P Sigma P' the prior and the view carry equal weight,
    # so a single-asset view lands exactly half way between the two.
    assert post_mean.iloc[j] == pytest.approx(0.5 * (prior.iloc[j] + q[0]), abs=1e-10)
    assert is_psd(post_cov.to_numpy())
    # Posterior covariance must be larger than the prior covariance.
    assert np.all(np.diag(post_cov.to_numpy()) >= np.diag(cov.to_numpy()) - 1e-12)


def test_implied_prior_round_trips_market_weights(setup):
    cov, market = setup
    bl = BlackLitterman(tau=0.05, risk_aversion=2.5)
    prior = bl.implied_prior(cov, market)
    recovered = bl.implied_weights(cov, prior)
    np.testing.assert_allclose(recovered.to_numpy(), market.to_numpy(), atol=1e-10)


def test_default_omega_is_tau_p_sigma_p(setup):
    cov, market = setup
    bl = BlackLitterman(tau=0.03, risk_aversion=3.0)
    n = cov.shape[0]
    p = np.zeros((2, n))
    p[0, 0] = 1.0
    p[1, 1] = 1.0
    p[1, 2] = -1.0
    q = np.array([0.01, 0.0])
    result = bl.blend_views(cov, market, p, q)
    expected = bl.tau * p @ cov.to_numpy() @ p.T
    np.testing.assert_allclose(result.omega.to_numpy(), np.diag(np.diag(expected)), atol=1e-14)
    assert np.allclose(result.omega.to_numpy(), np.diag(np.diag(result.omega.to_numpy())))


def test_blend_views_moves_mean_towards_view(setup):
    cov, market = setup
    bl = BlackLitterman(tau=0.05, risk_aversion=2.5)
    n = cov.shape[0]
    p = np.zeros((1, n))
    p[0, 0] = 1.0
    q = np.array([0.10])
    result = bl.blend_views(cov, market, p, q)
    assert result.posterior_mean.iloc[0] > result.prior_mean.iloc[0]
    assert result.views.shape[0] == 1
    assert result.views.iloc[0]["Q"] == pytest.approx(0.10)
    assert is_psd(result.posterior_cov.to_numpy())


def test_explicit_omega_is_used(setup):
    cov, market = setup
    bl = BlackLitterman(tau=0.05)
    n = cov.shape[0]
    p = np.zeros((1, n))
    p[0, 0] = 1.0
    q = np.array([0.05])
    tight = bl.blend_views(cov, market, p, q, omega=np.array([[1e-8]]))
    loose = bl.blend_views(cov, market, p, q, omega=np.array([[1.0]]))
    prior = bl.implied_prior(cov, market)
    assert abs(tight.posterior_mean.iloc[0] - 0.05) < abs(loose.posterior_mean.iloc[0] - 0.05)
    assert abs(tight.posterior_mean.iloc[0] - prior.iloc[0]) > abs(
        loose.posterior_mean.iloc[0] - prior.iloc[0]
    )


def test_invalid_parameters_raise(setup):
    cov, market = setup
    with pytest.raises(ValueError):
        BlackLitterman(tau=0.0)
    with pytest.raises(ValueError):
        BlackLitterman(risk_aversion=-1.0)
    bl = BlackLitterman()
    with pytest.raises(ValueError):
        bl.implied_prior(cov, np.full(cov.shape[0], 0.5))  # does not sum to 1
    n = cov.shape[0]
    prior = bl.implied_prior(cov, market)
    with pytest.raises(ValueError):
        bl.posterior(prior, cov, np.zeros((1, n - 1)), np.array([0.01]))
    with pytest.raises(ValueError):
        bl.posterior(prior, cov, np.zeros((1, n)), np.array([0.01, 0.02]))
