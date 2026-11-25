# riskopt — portfolio construction, allocation & risk analytics

Read `..\CONVENTIONS.md` first. Package name: `riskopt`.

## Goal

A portfolio construction and risk lab: classic and modern allocators (mean-
variance, Black-Litterman, HRP, risk parity, mean-CVaR), transaction-cost and
turnover awareness, factor exposure constraints, plus a proper risk/stress
reporting engine.

## Modules

`src/riskopt/`

1. `data.py` — `synthetic_returns(n_assets=12, n_days=1500, n_factors=4, seed=42)`
   → `(returns: DataFrame, factor_loadings: DataFrame, factor_returns: DataFrame)`
   built from a known factor model with block-correlated idiosyncratic terms and
   fat-tailed (Student-t) innovations; `sample_cov(returns, method='sample'|
   'ledoit-wolf'|'ewma')` with shrinkage implemented in numpy (Ledoit-Wolf
   analytical constant-correlation target).
2. `allocators.py` — every allocator returns a dataclass
   `Allocation(weights: pd.Series, diagnostics: dict)`:
   - `min_variance(cov, long_only=True, max_weight=None, sector_map=None,
     sector_caps=None)` via `scipy.optimize.minimize` (SLSQP) with analytic
     gradient; documented constraints.
   - `max_sharpe(mu, cov, rf=0.0, ...)` (tangency portfolio, same constraint
     machinery).
   - `mean_cvar(returns, alpha=0.05, target_return=None, ...)` — Rockafellar–
     Uryasev LP as linear program via `scipy.optimize.linprog` (HiGHS).
   - `risk_parity(cov, budget=None, ...)` — Newton / cyclical coordinate descent
     on the equal-risk-contribution system; assert risk contributions equal.
   - `hrp(returns, linkage='single', codep='ward'|'average')` — Hierarchical
     Risk Parity (López de Prado): correlation distance, quasi-diagonalisation,
     recursive bisection. Pure numpy/scipy.cluster.
3. `black_litterman.py` — `BlackLitterman(prior_mu=None, tau=0.05, risk_aversion
   =2.5)` with `implied_prior(cov, market_weights)` (reverse optimisation),
   `posterior(prior_mu, cov, P, Q, omega=None)` (diagonal Omega from
   `tau * P cov P'` by default) returning posterior mean + cov; `blend_views`
   convenience. Document the equations.
4. `costs.py` — `TurnoverCostModel(linear_bps, impact_coef)` applying
   `cost = bps*|Δw| + impact*|Δw|^{3/2}*sigma`; `net_weights(prev, target,
   cost_model)` and `trade_list(prev, target, nav)`.
5. `risk.py` — `PortfolioRisk`: exposures to supplied factors (OLS cross-
   sectionally per day), factor + specific risk decomposition
   (`total_var = w'B F B'w + w'D w`), marginal & component risk contributions,
   diversification ratio, effective number of bets (entropy of risk
   contributions), beta to a benchmark, tracking error, ex-ante vs ex-post vol.
6. `stress.py` — `historical_stress(weights, factor_returns, windows)` (worst
   non-overlapping windows, report per-window PnL), `factor_shock(weights,
   loadings, shocks)` (linear + optional quadratic convexity), `scenario_grid`
   (varying equity/rates/credit shocks) producing a tidy DataFrame.
7. `report.py` — `build_report(returns, weights, ...)` aggregating all of the
   above into one markdown/JSON document: weights, risk decomposition table,
   top risk contributors, stress table, diversification stats.
8. `__main__.py` — `demo`: synthetic returns → five allocators → compare weights,
   ex-ante vol, CVaR, diversification ratio, turnover vs. equal weight → print
   markdown table; `--plot` saves the HRP dendrogram ordering.

## Tests

- `test_data.py`: determinism; covariance PSD; Ledoit-Wolf between sample and
  diagonal; ewma weights decay.
- `test_allocators.py`: weights sum to 1 and respect `max_weight`/sector caps;
  min-variance vol <= equal-weight vol; max-Sharpe in-sample Sharpe >= min-var;
  mean-CVaR solves and is <= equal-weight CVaR; risk parity contributions equal
  within 1e-8 and long-only solutions positive; HRP weights sum to 1 and its
  quasi-diagonal correlation matrix has larger off-diagonal average after
  reordering than before.
- `test_black_litterman.py`: zero views reproduce the prior mean; a strong
  positive view raises the posterior mean of that asset; posterior cov PSD;
  implied prior round-trips `market_weights` under the risk-aversion convention.
- `test_costs.py`: zero turnover → zero cost; cost monotone in |Δw| and in
  impact coefficient; trade list nets to the weight change.
- `test_risk.py`: factor+specific variance equals total variance to 1e-10 for a
  known B/F/D; component risk contributions sum to total vol; single-asset
  portfolio has ENB = 1; benchmark beta of the benchmark itself is 1.
- `test_stress.py`: linear shock on a single-factor model matches
  `w' B shock` exactly; worst-window selector picks a hand-planted worst window;
  scenario grid shape/columns stable.
- `test_report.py`: report builds for 3 assets, JSON serialisable, contains all
  required sections.

## README sections

Allocator comparison table (objective, constraints, solver, complexity),
derivations for BL and CVaR LP, HRP algorithm walkthrough, risk decomposition
math, stress methodology, honest limitations (estimation error, non-stationary
correlations).
