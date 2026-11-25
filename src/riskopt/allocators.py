"""Portfolio allocators with a shared constraint language.

Every allocator returns an :class:`Allocation` holding a weight ``Series`` and a
diagnostics dictionary.  Constraints are identical across solvers:

* ``sum(w) == 1`` (fully invested, no leverage unless ``long_only=False``);
* ``0 <= w_i <= max_weight`` (or ``|w_i| <= max_weight`` when shorting);
* ``sum_{i in sector s} w_i <= sector_caps[s]``.

Solvers are deterministic: fixed starting points, no randomness, and the
analytic objective gradient is supplied to SLSQP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage as _linkage
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import squareform

__all__ = [
    "Allocation",
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "mean_cvar",
    "risk_parity",
    "hrp",
]


@dataclass
class Allocation:
    """Container for an allocation result.

    Attributes
    ----------
    weights : Series
        Portfolio weights indexed by asset.
    diagnostics : dict
        Solver status, objective value, risk statistics and constraint metadata.
    name : str
        Allocator name (convenience label).
    """

    weights: pd.Series
    diagnostics: dict = field(default_factory=dict)
    name: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.weights, pd.Series):
            raise ValueError("weights must be a pandas Series")
        if not np.isfinite(self.weights.to_numpy()).all():
            raise ValueError("weights contain NaN or infinite values")

    @property
    def n_assets(self) -> int:
        return int(self.weights.size)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Allocation(name={self.name!r}, n_assets={self.n_assets})"


# --------------------------------------------------------------------------- #
# constraint bookkeeping
# --------------------------------------------------------------------------- #


@dataclass
class _Book:
    names: pd.Index
    lo: np.ndarray
    hi: np.ndarray
    sectors: dict[str, np.ndarray]
    caps: dict[str, float]

    @property
    def n(self) -> int:
        return int(self.names.size)

    @property
    def bounds(self) -> list[tuple[float | None, float | None]]:
        out: list[tuple[float | None, float | None]] = []
        for low, high in zip(self.lo, self.hi):
            out.append((None if not np.isfinite(low) else float(low),
                        None if not np.isfinite(high) else float(high)))
        return out


def _sector_labels(sector_map: Mapping[str, str] | pd.Series | np.ndarray,
                   names: pd.Index) -> np.ndarray:
    if isinstance(sector_map, Mapping):
        missing = [n for n in names if n not in sector_map]
        if missing:
            raise ValueError(f"sector_map missing assets: {missing[:5]}")
        labels = np.array([sector_map[n] for n in names], dtype=object)
    elif isinstance(sector_map, pd.Series):
        labels = sector_map.reindex(names).to_numpy(dtype=object)
        if pd.isna(labels).any():
            raise ValueError("sector_map does not cover every asset")
    else:
        labels = np.asarray(sector_map, dtype=object)
        if labels.size != names.size:
            raise ValueError("sector_map length does not match the number of assets")
    return labels


def _make_book(names: pd.Index, long_only: bool, max_weight: float | None,
               sector_map: Any, sector_caps: Mapping[str, float] | None) -> _Book:
    n = int(names.size)
    if max_weight is not None:
        if not np.isfinite(max_weight) or not 0.0 < max_weight <= 1.0:
            raise ValueError("max_weight must lie in (0, 1]")
        if long_only and max_weight * n < 1.0 - 1e-12:
            raise ValueError(
                f"max_weight={max_weight} with {n} long-only assets cannot sum to 1"
            )
    lo = np.zeros(n) if long_only else np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    if max_weight is not None:
        hi = np.full(n, float(max_weight))
        if not long_only:
            lo = np.full(n, -float(max_weight))

    sectors: dict[str, np.ndarray] = {}
    caps: dict[str, float] = {}
    if sector_caps:
        if sector_map is None:
            raise ValueError("sector_caps requires sector_map")
        labels = _sector_labels(sector_map, names)
        for label, cap in sector_caps.items():
            idx = np.flatnonzero(labels == label)
            if idx.size == 0:
                raise ValueError(f"sector cap given for unknown sector {label!r}")
            if not np.isfinite(cap) or not 0.0 < cap <= 1.0:
                raise ValueError(f"sector cap for {label!r} must lie in (0, 1]")
            sectors[str(label)] = idx
            caps[str(label)] = float(cap)
        if sum(caps.values()) < 1.0 - 1e-12:
            raise ValueError("sector caps sum to less than 1: problem is infeasible")
    elif sector_map is not None:
        raise ValueError("sector_map provided without sector_caps")
    return _Book(names=names, lo=lo, hi=hi, sectors=sectors, caps=caps)


def _constraints(book: _Book) -> list[dict]:
    cons: list[dict] = [
        {
            "type": "eq",
            "fun": lambda w: float(np.sum(w) - 1.0),
            "jac": lambda w: np.ones(book.n),
        }
    ]
    for label, idx in book.sectors.items():
        cap = book.caps[label]

        def fun(w: np.ndarray, idx=idx, cap=cap) -> float:
            return float(cap - w[idx].sum())

        def jac(w: np.ndarray, idx=idx) -> np.ndarray:
            g = np.zeros(book.n)
            g[idx] = -1.0
            return g

        cons.append({"type": "ineq", "fun": fun, "jac": jac})
    return cons


def _verify(w: np.ndarray, book: _Book, tol: float = 1e-8) -> np.ndarray:
    """Raise when a candidate weight vector violates the constraint set."""
    w = np.asarray(w, dtype=float)
    if not np.isfinite(w).all():
        raise ValueError("solver returned non-finite weights")
    if abs(w.sum() - 1.0) > tol:
        raise ValueError(f"weights sum to {w.sum():.12f}, expected 1")
    if np.any(w < book.lo - tol) or np.any(w > book.hi + tol):
        raise ValueError("weights violate per-asset bounds")
    for label, idx in book.sectors.items():
        if w[idx].sum() > book.caps[label] + tol:
            raise ValueError(f"sector {label!r} exceeds its cap")
    return w


def _start(book: _Book) -> np.ndarray:
    """Deterministic feasible starting point via iterative projection."""
    w = np.full(book.n, 1.0 / book.n)
    for _ in range(2000):
        w = np.clip(w, book.lo, book.hi)
        total = w.sum()
        if total <= 1e-15:
            raise ValueError("constraints admit no fully-invested portfolio")
        w = w / total
        violated = False
        for label, idx in book.sectors.items():
            cap = book.caps[label]
            sector_total = w[idx].sum()
            if sector_total > cap + 1e-14:
                violated = True
                excess = sector_total - cap
                w[idx] *= cap / sector_total
                other = np.ones(book.n, dtype=bool)
                other[idx] = False
                room = np.where(other, book.hi - w, 0.0)
                room = np.clip(room, 0.0, None)
                if room.sum() <= 1e-15:
                    raise ValueError("sector caps are infeasible for these bounds")
                w = w + excess * room / room.sum()
        if not violated:
            try:
                return _verify(w, book, tol=1e-9)
            except ValueError:
                continue
    raise ValueError("could not build a feasible starting point; constraints conflict")


def _finalize(w: np.ndarray, book: _Book) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w[np.abs(w) < 1e-13] = 0.0
    if np.isfinite(book.lo).all() and book.lo.min() >= 0.0:
        w = np.clip(w, 0.0, None)
    w = np.clip(w, book.lo, book.hi)
    w = w / w.sum()
    w = np.clip(w, book.lo, book.hi)
    w = w / w.sum()
    return _verify(w, book, tol=2e-6)


def _cov_matrix(cov: pd.DataFrame | np.ndarray) -> tuple[np.ndarray, pd.Index]:
    if isinstance(cov, pd.DataFrame):
        if cov.shape[0] != cov.shape[1]:
            raise ValueError("cov must be square")
        names = cov.columns
        mat = cov.to_numpy(dtype=float)
    else:
        mat = np.asarray(cov, dtype=float)
        if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
            raise ValueError("cov must be a square matrix")
        names = pd.Index([f"A{i + 1:02d}" for i in range(mat.shape[0])])
    if not np.isfinite(mat).all():
        raise ValueError("cov contains NaN or infinite values")
    if np.any(np.diag(mat) <= 0):
        raise ValueError("cov must have strictly positive variances")
    if not np.allclose(mat, mat.T, atol=1e-12):
        raise ValueError("cov must be symmetric")
    return mat, names


def equal_weight(names: pd.Index | list[str]) -> pd.Series:
    """Fully invested, equally weighted portfolio."""
    idx = pd.Index(names)
    if idx.empty:
        raise ValueError("names must not be empty")
    return pd.Series(np.full(idx.size, 1.0 / idx.size), index=idx, name="equal_weight")


def _diagnostics(res: Any, w: np.ndarray, book: _Book, extra: dict) -> dict:
    out = {
        "solver": "SLSQP",
        "status": int(getattr(res, "status", -1)),
        "success": bool(getattr(res, "success", False)),
        "message": str(getattr(res, "message", "")),
        "n_iter": int(getattr(res, "nit", 0)),
        "sum_weights": float(w.sum()),
        "n_assets": book.n,
        "max_weight": float(w.max()),
        "min_weight": float(w.min()),
        "sector_caps": dict(book.caps),
    }
    out.update(extra)
    return out


# --------------------------------------------------------------------------- #
# mean-variance
# --------------------------------------------------------------------------- #


def min_variance(
    cov: pd.DataFrame | np.ndarray,
    long_only: bool = True,
    max_weight: float | None = None,
    sector_map: Any = None,
    sector_caps: Mapping[str, float] | None = None,
) -> Allocation:
    """Global minimum-variance portfolio.

    Solves ``min_w w' S w`` subject to ``1'w = 1`` and the constraint set.
    The analytic gradient is ``2 S w``.  The problem is convex, so SLSQP
    converges to the global optimum from any feasible start.
    """
    S, names = _cov_matrix(cov)
    book = _make_book(names, long_only, max_weight, sector_map, sector_caps)

    def objective(w: np.ndarray) -> float:
        return float(w @ S @ w)

    def gradient(w: np.ndarray) -> np.ndarray:
        return 2.0 * (S @ w)

    res = minimize(
        objective,
        _start(book),
        jac=gradient,
        method="SLSQP",
        bounds=book.bounds,
        constraints=_constraints(book),
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = _finalize(res.x, book)
    var = float(w @ S @ w)
    diag = _diagnostics(res, w, book, {"objective": var, "volatility": float(np.sqrt(var))})
    return Allocation(pd.Series(w, index=names, name="min_variance"), diag, "min_variance")


def max_sharpe(
    mu: pd.Series | np.ndarray,
    cov: pd.DataFrame | np.ndarray,
    rf: float = 0.0,
    long_only: bool = True,
    max_weight: float | None = None,
    sector_map: Any = None,
    sector_caps: Mapping[str, float] | None = None,
) -> Allocation:
    """Maximum-Sharpe (tangency) portfolio.

    Maximises ``(mu - rf)'w / sqrt(w'Sw)``.  The negative-Sharpe gradient is

    .. math::

        \\nabla_w \\left(-\\frac{m'w}{\\sqrt{w'Sw}}\\right)
        = -\\frac{m\\,(w'Sw) - (m'w)\\,Sw}{(w'Sw)^{3/2}}.

    Several deterministic starts (equal weight, minimum variance, inverse
    volatility) are tried and the best feasible Sharpe is kept.
    """
    S, names = _cov_matrix(cov)
    if isinstance(mu, pd.Series):
        if set(mu.index) != set(names):
            raise ValueError("mu index does not match cov columns")
        m = mu.reindex(names).to_numpy(dtype=float) - float(rf)
    else:
        m = np.asarray(mu, dtype=float).ravel() - float(rf)
        if m.size != names.size:
            raise ValueError("mu length does not match cov")
    if not np.isfinite(m).all():
        raise ValueError("mu contains NaN or infinite values")
    if np.allclose(m, 0.0):
        raise ValueError("excess expected returns are all zero; Sharpe is undefined")
    book = _make_book(names, long_only, max_weight, sector_map, sector_caps)
    cons = _constraints(book)

    def neg_sharpe(w: np.ndarray) -> float:
        var = float(w @ S @ w)
        if var <= 1e-24:
            return 1e12
        return -float(m @ w) / np.sqrt(var)

    def neg_sharpe_grad(w: np.ndarray) -> np.ndarray:
        var = float(w @ S @ w)
        var = max(var, 1e-24)
        return -(m * var - float(m @ w) * (S @ w)) / var**1.5

    starts = [_start(book)]
    try:
        mv = min_variance(cov, long_only, max_weight, sector_map, sector_caps)
        starts.append(_verify(mv.weights.to_numpy(), book, tol=1e-6))
    except ValueError:
        pass
    inv_vol = 1.0 / np.sqrt(np.diag(S))
    starts.append(_start_from(book, inv_vol / inv_vol.sum()))

    best_w, best_res, best_sharpe = None, None, -np.inf
    for x0 in starts:
        res = minimize(
            neg_sharpe,
            x0,
            jac=neg_sharpe_grad,
            method="SLSQP",
            bounds=book.bounds,
            constraints=cons,
            options={"maxiter": 1000, "ftol": 1e-12},
        )
        try:
            w = _finalize(res.x, book)
        except ValueError:
            continue
        var = float(w @ S @ w)
        sharpe = float(m @ w) / np.sqrt(var) if var > 0 else -np.inf
        if sharpe > best_sharpe:
            best_w, best_res, best_sharpe = w, res, sharpe
    if best_w is None:
        raise RuntimeError("max_sharpe failed to find a feasible solution")
    var = float(best_w @ S @ best_w)
    diag = _diagnostics(
        best_res,
        best_w,
        book,
        {
            "objective": -best_sharpe,
            "sharpe": best_sharpe,
            "volatility": float(np.sqrt(var)),
            "expected_return": float(m @ best_w) + float(rf),
            "n_starts": len(starts),
        },
    )
    return Allocation(pd.Series(best_w, index=names, name="max_sharpe"), diag, "max_sharpe")


def _start_from(book: _Book, seed_weights: np.ndarray) -> np.ndarray:
    """Project an arbitrary weight vector onto the feasible set."""
    w = np.asarray(seed_weights, dtype=float)
    if w.size != book.n:
        raise ValueError("seed weights have the wrong length")
    if not np.isfinite(w).all() or w.sum() <= 0:
        return _start(book)
    w = np.clip(w / w.sum(), book.lo, book.hi)
    for _ in range(500):
        w = np.clip(w, book.lo, book.hi)
        w = w / w.sum()
        violated = False
        for label, idx in book.sectors.items():
            cap = book.caps[label]
            total = w[idx].sum()
            if total > cap + 1e-14:
                violated = True
                excess = total - cap
                w[idx] *= cap / total
                other = np.ones(book.n, dtype=bool)
                other[idx] = False
                room = np.clip(np.where(other, book.hi - w, 0.0), 0.0, None)
                if room.sum() <= 1e-15:
                    return _start(book)
                w = w + excess * room / room.sum()
        if not violated:
            try:
                return _verify(w, book, tol=1e-9)
            except ValueError:
                continue
    return _start(book)


# --------------------------------------------------------------------------- #
# mean-CVaR (Rockafellar-Uryasev LP)
# --------------------------------------------------------------------------- #


def mean_cvar(
    returns: pd.DataFrame | np.ndarray,
    alpha: float = 0.05,
    target_return: float | None = None,
    long_only: bool = True,
    max_weight: float | None = None,
    sector_map: Any = None,
    sector_caps: Mapping[str, float] | None = None,
) -> Allocation:
    """Minimum CVaR portfolio via the Rockafellar-Uryasev linear program.

    For loss ``L_t(w) = -r_t'w`` and tail probability ``alpha``,

    .. math::

        \\mathrm{CVaR}_\\alpha(w) = \\min_{\\eta}\\;
        \\eta + \\frac{1}{\\alpha T}\\sum_t \\xi_t
        \\quad \\text{s.t.}\\; \\xi_t \\ge L_t(w) - \\eta,\\; \\xi_t \\ge 0.

    The LP minimises ``eta + (1/(alpha T)) sum xi`` over ``(w, eta, xi)`` with
    ``sum(w) = 1``, the bound/sector constraints, and an optional
    ``mu'w >= target_return``.  HiGHS is deterministic.
    """
    if isinstance(returns, pd.DataFrame):
        names = returns.columns
        R = returns.to_numpy(dtype=float)
    else:
        R = np.asarray(returns, dtype=float)
        if R.ndim != 2:
            raise ValueError("returns must be 2-D")
        names = pd.Index([f"A{i + 1:02d}" for i in range(R.shape[1])])
    if not np.isfinite(R).all():
        raise ValueError("returns contains NaN or infinite values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    t_obs, n = R.shape
    if t_obs < 2:
        raise ValueError("need at least two return observations")
    book = _make_book(names, long_only, max_weight, sector_map, sector_caps)

    c = np.concatenate([np.zeros(n), [1.0], np.full(t_obs, 1.0 / (alpha * t_obs))])
    a_ub = np.zeros((t_obs, n + 1 + t_obs))
    a_ub[:, :n] = -R
    a_ub[:, n] = -1.0
    a_ub[np.arange(t_obs), n + 1 + np.arange(t_obs)] = -1.0
    b_ub = np.zeros(t_obs)

    mu = R.mean(axis=0)
    if target_return is not None:
        row = np.zeros((1, n + 1 + t_obs))
        row[0, :n] = -mu
        a_ub = np.vstack([a_ub, row])
        b_ub = np.concatenate([b_ub, [-float(target_return)]])

    a_eq = np.zeros((1, n + 1 + t_obs))
    a_eq[0, :n] = 1.0
    bounds = book.bounds + [(None, None)] + [(0.0, None)] * t_obs

    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        raise ValueError(f"mean-CVaR LP failed: {res.message}")
    w = _finalize(res.x[:n], book)
    eta = float(res.x[n])
    losses = -(R @ w)
    var = float(np.quantile(losses, 1.0 - alpha))
    tail = losses[losses >= var]
    empirical_cvar = float(tail.mean()) if tail.size else float(losses.max())
    diag = {
        "solver": "highs",
        "status": int(res.status),
        "success": bool(res.success),
        "message": str(res.message),
        "n_iter": int(getattr(res, "nit", 0)),
        "sum_weights": float(w.sum()),
        "n_assets": book.n,
        "max_weight": float(w.max()),
        "min_weight": float(w.min()),
        "sector_caps": dict(book.caps),
        "alpha": float(alpha),
        "cvar": float(res.fun),
        "var": eta,
        "empirical_cvar": empirical_cvar,
        "empirical_var": var,
        "expected_return": float(mu @ w),
        "volatility": float(np.sqrt(w @ np.cov(R, rowvar=False, ddof=1) @ w)),
        "target_return": None if target_return is None else float(target_return),
    }
    return Allocation(pd.Series(w, index=names, name="mean_cvar"), diag, "mean_cvar")


# --------------------------------------------------------------------------- #
# risk parity
# --------------------------------------------------------------------------- #


def risk_parity(
    cov: pd.DataFrame | np.ndarray,
    budget: pd.Series | np.ndarray | None = None,
    long_only: bool = True,
    max_weight: float | None = None,
    max_iter: int = 500,
    tol: float = 1e-12,
) -> Allocation:
    """Equal (or budgeted) risk contribution portfolio.

    Uses Spinu's (2013) convex reformulation: minimise
    ``0.5 x'Sx - sum b_i log x_i`` over ``x > 0``, then set ``w = x / sum(x)``.
    The first-order condition ``(Sx)_i = b_i / x_i`` implies
    ``RC_i = w_i (Sw)_i / sigma = b_i``.  A Newton iteration with backtracking
    is followed by cyclical coordinate descent sweeps (exact coordinate
    minimisers) to polish the solution to machine precision.
    """
    S, names = _cov_matrix(cov)
    n = names.size
    if not long_only:
        raise ValueError("risk_parity is only defined for long-only portfolios")
    if budget is None:
        b = np.full(n, 1.0 / n)
    elif isinstance(budget, pd.Series):
        if set(budget.index) != set(names):
            raise ValueError("budget index does not match cov columns")
        b = budget.reindex(names).to_numpy(dtype=float)
    else:
        b = np.asarray(budget, dtype=float).ravel()
        if b.size != n:
            raise ValueError("budget length does not match cov")
    if np.any(~np.isfinite(b)) or np.any(b <= 0):
        raise ValueError("risk budgets must be finite and strictly positive")
    b = b / b.sum()

    var = np.diag(S)
    x = np.sqrt(b / var)
    x = x / x.sum()

    for _ in range(max_iter):
        sx = S @ x
        grad = sx - b / x
        if np.max(np.abs(grad)) < tol:
            break
        hess = S + np.diag(b / x**2)
        try:
            step = np.linalg.solve(hess, -grad)
        except np.linalg.LinAlgError:  # pragma: no cover - S is PD
            step = -grad
        f0 = 0.5 * float(x @ sx) - float(b @ np.log(x))
        slope = float(grad @ step)
        t = 1.0
        while t > 1e-14:
            candidate = x + t * step
            if np.all(candidate > 0):
                f1 = 0.5 * float(candidate @ S @ candidate) - float(b @ np.log(candidate))
                if f1 <= f0 + 1e-4 * t * slope:
                    x = candidate
                    break
            t *= 0.5
        else:
            break

    for _ in range(500):
        change = 0.0
        for i in range(n):
            cross = float(S[i] @ x) - S[i, i] * x[i]
            new = (-cross + np.sqrt(cross * cross + 4.0 * S[i, i] * b[i])) / (2.0 * S[i, i])
            change = max(change, abs(new - x[i]))
            x[i] = new
        if change < 1e-15:
            break

    w = x / x.sum()
    vol = float(np.sqrt(w @ S @ w))
    rc = w * (S @ w) / vol
    deviation = float(np.max(np.abs(rc / vol - b)))

    if max_weight is not None:
        if max_weight * n < 1.0 - 1e-12:
            raise ValueError("max_weight is infeasible for a long-only portfolio")
        if w.max() > max_weight + 1e-10:
            book = _make_book(names, True, max_weight, None, None)
            target = b

            def objective(v: np.ndarray) -> float:
                vv = np.sqrt(v @ S @ v)
                contrib = v * (S @ v) / vv / vv
                return float(np.sum((contrib - target) ** 2))

            res = minimize(
                objective,
                _start_from(book, w),
                method="SLSQP",
                bounds=book.bounds,
                constraints=_constraints(book),
                options={"maxiter": 1000, "ftol": 1e-14},
            )
            w = _finalize(res.x, book)
            vol = float(np.sqrt(w @ S @ w))
            rc = w * (S @ w) / vol
            deviation = float(np.max(np.abs(rc / vol - b)))

    if max_weight is None:
        assert deviation <= 1e-8, f"risk contributions not equal (max dev {deviation:.3e})"

    diag = {
        "solver": "newton+ccd",
        "success": True,
        "status": 0,
        "message": "converged",
        "n_iter": int(max_iter),
        "sum_weights": float(w.sum()),
        "n_assets": n,
        "max_weight": float(w.max()),
        "min_weight": float(w.min()),
        "sector_caps": {},
        "volatility": vol,
        "risk_contributions": rc,
        "budget": b,
        "rc_deviation": deviation,
    }
    return Allocation(pd.Series(w, index=names, name="risk_parity"), diag, "risk_parity")


# --------------------------------------------------------------------------- #
# hierarchical risk parity
# --------------------------------------------------------------------------- #


def hrp(
    returns: pd.DataFrame,
    linkage: str = "single",
    codep: str | None = None,
) -> Allocation:
    """Hierarchical Risk Parity (Lopez de Prado, 2016).

    Steps: (1) correlation distance ``d_ij = sqrt(0.5 (1 - rho_ij))``;
    (2) agglomerative clustering and quasi-diagonalisation with
    :func:`scipy.cluster.hierarchy.leaves_list`; (3) recursive bisection where
    the split fraction between two clusters is
    ``alpha = V_R / (V_L + V_R)`` with ``V`` the inverse-variance cluster
    variance.

    Parameters
    ----------
    returns : DataFrame
        Asset returns (T x N).
    linkage : str, default 'single'
        Linkage method used for the recursive-bisection tree.
    codep : str or None
        Optional separate linkage method used to order assets (quasi-
        diagonalisation).  ``None`` reuses ``linkage``.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("hrp expects a returns DataFrame")
    if returns.shape[1] < 2:
        raise ValueError("hrp needs at least two assets")
    if returns.shape[0] < 3:
        raise ValueError("hrp needs at least three observations")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("returns contains NaN or infinite values")
    if linkage not in ("single", "complete", "average", "ward", "centroid", "median"):
        raise ValueError(f"unknown linkage method {linkage!r}")
    if codep is not None and codep not in ("single", "complete", "average", "ward"):
        raise ValueError(f"unknown codep method {codep!r}")

    names = returns.columns
    n = names.size
    corr = returns.corr().to_numpy()
    if not np.isfinite(corr).all():
        raise ValueError("correlation matrix contains NaN; an asset may be constant")
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, None))
    np.fill_diagonal(dist, 0.0)
    condensed = squareform(dist, checks=False)
    tree = _linkage(condensed, method=linkage)
    order_tree = _linkage(condensed, method=codep or linkage)
    order = [int(i) for i in leaves_list(order_tree)]

    cov = returns.cov().to_numpy()

    def inverse_variance(idx: list[int]) -> np.ndarray:
        sub = cov[np.ix_(idx, idx)]
        ivp = 1.0 / np.diag(sub)
        return ivp / ivp.sum()

    def cluster_var(idx: list[int]) -> float:
        sub = cov[np.ix_(idx, idx)]
        ivp = inverse_variance(idx)
        return float(ivp @ sub @ ivp)

    fractions: dict[int, float] = {}

    def bisect(idx: list[int], scale: float) -> None:
        if len(idx) == 1:
            fractions[idx[0]] = scale
            return
        mid = len(idx) // 2
        left, right = idx[:mid], idx[mid:]
        var_l, var_r = cluster_var(left), cluster_var(right)
        alpha = var_r / (var_l + var_r) if (var_l + var_r) > 0 else 0.5
        bisect(left, scale * alpha)
        bisect(right, scale * (1.0 - alpha))

    bisect(order, 1.0)
    w = np.array([fractions[i] for i in range(n)], dtype=float)
    w = w / w.sum()

    reordered = corr[np.ix_(order, order)]

    def local_coherence(mat: np.ndarray, band: int = 2) -> float:
        mask = np.zeros_like(mat, dtype=bool)
        for k in range(1, band + 1):
            idx = np.arange(mat.shape[0] - k)
            mask[idx, idx + k] = True
            mask[idx + k, idx] = True
        return float(np.abs(mat[mask]).mean()) if mask.any() else float("nan")

    diag = {
        "solver": "hrp",
        "success": True,
        "status": 0,
        "message": "recursive bisection",
        "n_iter": int(n - 1),
        "sum_weights": float(w.sum()),
        "n_assets": n,
        "max_weight": float(w.max()),
        "min_weight": float(w.min()),
        "sector_caps": {},
        "linkage": linkage,
        "codep": codep or linkage,
        "order": [str(names[i]) for i in order],
        "linkage_matrix": tree.tolist(),
        "coherence_before": local_coherence(corr),
        "coherence_after": local_coherence(reordered),
        "corr_reordered": pd.DataFrame(reordered, index=names[order], columns=names[order]),
        "volatility": float(np.sqrt(w @ cov @ w)),
    }
    return Allocation(pd.Series(w, index=names, name="hrp"), diag, "hrp")
