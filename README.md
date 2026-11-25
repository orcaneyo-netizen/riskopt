# riskopt

Portfolio construction, allocation and risk analytics in pure Python.

`riskopt` is a small, self-contained research library: synthetic factor-model
data, five classic/modern allocators, Black-Litterman views, turnover costs, a
factor risk decomposition engine, stress testing and a one-page report.
Everything runs offline with **numpy / pandas / scipy** only.

```python
from riskopt import synthetic_returns, sample_cov, min_variance, build_report

returns, loadings, factors = synthetic_returns(n_assets=12, seed=42)
cov = sample_cov(returns, "ledoit-wolf")
alloc = min_variance(cov, max_weight=0.2)

report = build_report(returns, alloc.weights, factor_returns=factors)
print(report.markdown)
```

## Install

```bash
python -m pip install -e .
python -m pytest -q
python -m riskopt demo            # end-to-end synthetic example
python -m riskopt demo --plot     # also saves hrp_dendrogram.png
```

Python 3.12+, numpy, pandas, scipy. `matplotlib` is optional (only `--plot`).

## What is inside

| module | responsibility |
| --- | --- |
| `data.py` | synthetic block-factor returns (Student-t), sample / Ledoit-Wolf / EWMA covariance |
| `allocators.py` | min-variance, max-Sharpe, mean-CVaR, risk parity, HRP |
| `black_litterman.py` | reverse-optimised prior, posterior mean/covariance, view blending |
| `costs.py` | linear + `|dw|^{3/2}` impact costs, net weights, trade blotter |
| `risk.py` | factor/specific variance split, MCTR/CTR, DR, ENB, beta, tracking error |
| `stress.py` | worst historical windows, factor shocks, equity/rates/credit grid |
| `report.py` | markdown + JSON report |

## Allocator comparison

All allocators share the same constraint language: `sum(w) = 1`,
`0 <= w_i <= max_weight` (or `|w_i| <= max_weight` when shorting), and
`sum_{i in s} w_i <= sector_caps[s]`.

| allocator | objective | solver | cost per solve |
| --- | --- | --- | --- |
| `min_variance` | `min w'Σw` | SLSQP, analytic gradient `2Σw` | `O(N²)` per iteration, ~10–50 iterations |
| `max_sharpe` | `max (μ-rf)'w / sqrt(w'Σw)` | SLSQP, analytic gradient, 3 deterministic starts | as above × 3 |
| `mean_cvar` | `min η + (1/αT)Σξ_t`, `ξ_t ≥ -r_t'w - η` | linear program, HiGHS | `O(T·N)` variables, sparse LP |
| `risk_parity` | `min ½x'Σx - Σ b_i log x_i`, `w = x/Σx` | Newton + CCD polish | `O(N³)` per Newton step, ~10 steps |
| `hrp` | recursive inverse-variance bisection | `scipy.cluster.hierarchy` | `O(N² log N)` |

The minimum-variance problem is convex, so SLSQP reaches the global optimum.
Max-Sharpe is not convex in `w`; the library tries equal-weight, min-variance
and inverse-volatility starts and keeps the best feasible Sharpe, which is
deterministic and robust in practice (not a proof of global optimality).

## Math notes

### Covariance estimation

*Sample*: `S = X_c'X_c / (T - 1)`.

*Ledoit-Wolf* (2004) shrinks towards the constant-correlation target
`F_ij = r̄ sqrt(s_ii s_jj)` (diagonal unchanged):

```
Σ_LW = (1 - δ) S + δ F,      δ = clip((π - ρ) / (γ T), 0, 1)
π = Σ_ij Var(x_i x_j),       ρ = Σ_ij Cov(x_i x_j, f_ij),   γ = Σ_ij (f_ij - s_ij)²
```

with `ρ_ij = (r̄/2) E[x_i x_j ((x_i²-s_ii)/s_ii + (x_j²-s_jj)/s_jj)]`. Sample
moments use the paper's `1/T` normalisation. `δ` is returned alongside the
matrix.

*EWMA*: weights `w_t ∝ 2^{-(T-1-t)/h}`, covariance
`Σ_t w_t (x_t-μ)(x_t-μ)' / (1 - Σ_t w_t²)`.

Every estimator is symmetrised and eigen-clipped onto the PSD cone.

### Mean-CVaR LP (Rockafellar-Uryasev)

For losses `L_t(w) = -r_t'w`,

```
CVaR_α(w) = min_η  η + 1/(αT) Σ_t max(0, L_t(w) - η)
```

Introducing `ξ_t ≥ L_t(w) - η`, `ξ_t ≥ 0` gives a linear program in
`(w, η, ξ)`. The optimum `η*` is the VaR and the objective is the CVaR. Adding
`μ'w ≥ target` and the shared weight constraints keeps it an LP, solved with
HiGHS. CVaR is coherent (sub-additive) and the LP is convex in `w`.

### Black-Litterman

Reverse optimisation assumes the market portfolio is mean-variance optimal:

```
Π = δ Σ w_mkt
```

Views `P μ = Q + ε`, `ε ~ N(0, Ω)`, combined with the prior
`μ ~ N(Π, τΣ)` yield

```
μ_post = [(τΣ)^{-1} + P'Ω^{-1}P]^{-1} [(τΣ)^{-1}Π + P'Ω^{-1}Q]
Σ_post = Σ + [(τΣ)^{-1} + P'Ω^{-1}P]^{-1}
Ω      = diag(τ P Σ P')                (He-Litterman default)
```

With `Ω = τPΣP'` a single view is absorbed half-way between prior and view
(prior and view carry equal weight); with `Ω → 0` the posterior mean equals
the view exactly. Zero views return the prior mean and `(1+τ)Σ`.

### HRP (López de Prado)

1. Correlation distance `d_ij = sqrt(0.5(1 - ρ_ij))`.
2. Agglomerative clustering; quasi-diagonalisation via `leaves_list`.
3. Recursive bisection of the ordered list: for clusters `L`, `R` with
   inverse-variance variances `V_L`, `V_R`, allocate
   `α = V_R/(V_L+V_R)` to `L` and `1-α` to `R`, recursing until single assets.

HRP never inverts the covariance matrix, which makes it more stable than
mean-variance when `N` is large relative to `T`.

### Risk decomposition

With `Σ = B F B' + diag(D)`:

```
σ_p² = w'B F B'w + w'Dw = (B'w)'F(B'w) + Σ_i w_i² D_i
MCTR_i = (Σw)_i / σ_p          CTR_i = w_i MCTR_i          Σ_i CTR_i = σ_p
```

Factor contributions are `e_k (Fe)_k` with `e = B'w`; they sum to the factor
variance. The diversification ratio is `(Σ_i w_i σ_i)/σ_p ≥ 1` for long-only
books. The effective number of bets is the entropy of risk shares,
`ENB = exp(-Σ p_i log p_i)`, `p_i = CTR_i/σ_p`: `1` for a single asset and `N`
for equal risk contributions. `ENB` is `NaN` when a position contributes
non-positive risk (possible with shorts).

Exposures are estimated by time-series OLS of each asset on the factors
(`r_i = α_i + B_i f + ε_i`), with residual variance using `T-K-1` degrees of
freedom.

### Stress testing

* Historical: non-overlapping blocks of `window` days, ranked by arithmetic
  PnL (compounded return also reported). Explicit `(start, end)` windows are
  supported, and factor returns can be mapped to portfolio returns through
  loadings.
* Factor shock: `pnl = w'B s`, optionally plus `½ s'Hs` for convexity.
* Scenario grid: Cartesian product of equity / rates / credit shocks.

## Demo output

`python -m riskopt demo` prints:

* allocator comparison — ex-ante vol, historical 95% CVaR, diversification
  ratio, ENB, turnover vs equal weight, largest weight;
* the full weight matrix;
* the factor/specific risk split for the min-variance portfolio;
* the three worst 21-day windows.

## Honest limitations

* **Estimation error dominates.** Sample means are extremely noisy; max-Sharpe
  and mean-CVaR are sensitive to them. Prefer shrinkage / BL priors and treat
  point estimates as inputs, not truth.
* **Correlations are non-stationary.** Historical windows and covariance
  estimates assume a stable regime; crisis correlations rise exactly when
  diversification is needed. Stress tests are scenarios, not probabilities.
* **CVaR is in-sample.** The LP optimises the empirical tail of the supplied
  sample; out-of-sample CVaR will be worse.
* **No transaction-cost-aware optimisation.** Costs are measured after the
  fact (`costs.py`), not embedded in the objective; high-turnover allocators
  such as max-Sharpe will look better gross than net.
* **HRP is heuristic.** It has no optimality guarantee and depends on the
  linkage/distance choice.
* **Factor model is linear** with diagonal specific risk; it ignores
  nonlinear exposures, fat tails and time-varying betas.
* `max_sharpe` uses deterministic multi-start, not a global-optimality proof.

## License

MIT — see `LICENSE`.
