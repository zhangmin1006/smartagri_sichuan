"""
contract.py
===========
Numerical core of the Guo, Parlar & Zhang (2025) managerial-supervision model,
"Optimal Effort Under Managerial Supervision with Risk-Averse Participants",
re-instantiated for smart-agriculture service contracts in Sichuan.

Role mapping used throughout this package
-----------------------------------------
The paper's *owner* (principal, risk aversion c1) and *manager* (agent, risk
aversion c2) are instantiated twice, with the SAME solver:

  Layer S ("service contract", used inside the ABM)
      principal = FARMER   (owns the crop, receives x - w)          -> c1
      agent     = PROVIDER (drone pilot / machinery service centre  -> c2
                            / cooperative operator) exerting effort e
      x         = realised gross value of output on the served plot
      w(x)      = what the farmer pays the provider
      e         = service effort (timeliness in the critical window,
                  application accuracy, number of passes, warning response)

  Layer G ("policy contract", used by the Government agent)
      principal = GOVERNMENT (residual claimant on social outcome)  -> c1
      agent     = FARMER (exerting effective-use effort)            -> c2
      x         = verified outcome (yield / avoided loss / area covered)
      w(x)      = outcome-contingent support (performance subsidy,
                  voucher top-up, insurance indemnity)

Core results implemented
------------------------
Theorem 1   dw/dx = rO(x-w) / [rO(x-w) + rM(w)],   w(a) = b
Section 3   power/power utilities  =>
                dw/dx = c1 w / (c1 w + c2 (x - w))
            with the exact implicit solution
                w(x) + K1 * w(x)^(c2/c1) = x,      K1 = (a - b) / b^(c2/c1)
Prop. 1     w is linear if c1 == c2, concave if c1 < c2, convex if c1 > c2
Eq. (5)     participation constraint  E[u(w(X_e))] = v(e) + Umin  pins b(e)
Section 4   J(e) = E[B(X_e - w(X_e;e))];  e* = argmax J(e), possibly interior

Because the implicit solution is exact, no ODE integration is required: w(x)
is recovered by a monotone root-find, which is fast enough to run inside an
agent-based model with thousands of contracts per season.

Validation (see tests/test_contract.py) reproduces the paper's
Example 2 (Table 2) and Example 3 (Table 3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

__all__ = [
    "PowerUtility",
    "OutcomeDistribution",
    "LinearTiltDensity",
    "ContractProblem",
    "ContractSolution",
    "wage_curvature",
    "solve_effort_cached",
    "normalised_contract_payment",
    "theoretical_mean_outcome",
]

_EPS = 1e-12


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PowerUtility:
    """Power (CRRA) utility  U(z) = z^(1-c) / (1-c),  c > 0, c != 1.

    `c` is the Arrow-Pratt *relative* risk-aversion coefficient; the absolute
    measure is r(z) = c / z (paper, Section 3).  In this package `c` is the
    single attribute that defines an agent's risk attitude.
    """

    c: float

    def __post_init__(self) -> None:
        if self.c <= 0:
            raise ValueError("power-utility risk aversion c must be > 0")
        if abs(self.c - 1.0) < 1e-9:
            raise ValueError("c == 1 (log utility) is excluded by the paper setup")

    def u(self, z):
        z = np.maximum(z, _EPS)
        return np.power(z, 1.0 - self.c) / (1.0 - self.c)

    def u_prime(self, z):
        z = np.maximum(z, _EPS)
        return np.power(z, -self.c)

    def absolute_risk_aversion(self, z):
        """Arrow-Pratt absolute risk aversion  r(z) = c / z."""
        return self.c / np.maximum(z, _EPS)

    def inverse_u(self, value: float) -> float:
        """z such that u(z) = value (certainty equivalent)."""
        return float(np.power(max(value * (1.0 - self.c), _EPS), 1.0 / (1.0 - self.c)))


# --------------------------------------------------------------------------
# Outcome distribution  f(x; e)
# --------------------------------------------------------------------------
class OutcomeDistribution:
    """Base class for the effort-dependent outcome density f(x; e) on [x0, x1]."""

    x0: float = 0.0
    x1: float = 1.0

    def pdf(self, x, e: float):  # pragma: no cover
        raise NotImplementedError

    def nodes(self, n: int = 401):
        """Quadrature nodes on the support."""
        return np.linspace(self.x0, self.x1, n)

    def expectation(self, g, e: float, n: int = 401) -> float:
        """E[g(X_e)] by Simpson quadrature (n forced odd)."""
        n = n if n % 2 == 1 else n + 1
        x = self.nodes(n)
        vals = np.asarray(g(x), dtype=float) * self.pdf(x, e)
        h = (self.x1 - self.x0) / (n - 1)
        wq = np.ones(n)
        wq[1:-1:2] = 4.0
        wq[2:-1:2] = 2.0
        return float(h / 3.0 * np.dot(wq, vals))

    def mean(self, e: float) -> float:
        return self.expectation(lambda x: x, e)


class LinearTiltDensity(OutcomeDistribution):
    """The Example 2/3 density  f(x; e) = 1 + (x - 1/2) e  on [0, 1].

    Satisfies first-order stochastic dominance in e:
        d(survivor)/de = -x(x-1)/2 > 0  for x in (0,1).
    Mean E[X; e] = 1/2 + e/12, i.e. effort tilts probability mass upward.

    `scale` rescales the support to [0, scale] for monetary outcomes
    (gross value of output per mu); the tilt is applied to x/scale.
    """

    def __init__(self, scale: float = 1.0) -> None:
        self.scale = float(scale)
        self.x0 = 0.0
        self.x1 = float(scale)

    def pdf(self, x, e: float):
        z = np.asarray(x, dtype=float) / self.scale
        dens = 1.0 + (z - 0.5) * e
        return np.maximum(dens, 0.0) / self.scale

    def survivor(self, x: float, e: float) -> float:
        z = x / self.scale
        return 1.0 - 0.5 * e * (z * z - z) - z


# --------------------------------------------------------------------------
# Wage schedule  w(x)  from the exact implicit solution
# --------------------------------------------------------------------------
def _wage_from_implicit(x, b: float, a: float, c1: float, c2: float):
    """Solve  w + K1 * w^(c2/c1) = x  for w, given the anchor w(a) = b.

    The left-hand side is strictly increasing in w > 0, so a bracketed
    root-find is globally convergent.  Vectorised over x.
    """
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    p = c2 / c1
    if not (0.0 < b < a):
        raise ValueError("anchor must satisfy 0 < b < a (got b=%r, a=%r)" % (b, a))

    if abs(c1 - c2) < 1e-12:                       # Example 1: linear contract
        return np.clip((b / a) * x_arr, 0.0, None)

    # Work in log space and in the normalised variable t = w / x.
    # Substituting w = t*x into  w + K1 w^p = x  and dividing by x gives
    #     h(t) = t + A t^p - 1 = 0,     log A = log(a-b) - p log b + (p-1) log x
    # h is strictly increasing on (0, 1] with h(1) = A >= 0, so the only
    # difficulty is finding a lower bracket where h < 0. Because A can be
    # astronomically large when the anchor b is near zero, the bracket is
    # derived analytically rather than guessed: A t^p < 1/2 whenever
    # t < (1/(2A))^(1/p).  Computing K1 directly would overflow here.
    # The inversion is done by VECTORISED BISECTION in log-t space over all
    # quadrature nodes at once. A per-node brentq call is correct but far too
    # slow to sit inside an ABM: this version runs ~100x faster and is exact
    # to machine precision after 90 halvings.
    log_K1 = np.log(a - b) - p * np.log(b)

    out = np.zeros_like(x_arr)
    pos = x_arr > _EPS
    if not np.any(pos):
        return out
    xi = x_arr[pos]
    log_A = log_K1 + (p - 1.0) * np.log(xi)

    def h(log_t):
        """h(t) = t + A t^p - 1, evaluated from log t (monotone increasing)."""
        return np.exp(log_t) + np.exp(np.minimum(log_A + p * log_t, 700.0)) - 1.0

    # Analytic lower bracket: A t^p < 1/2  whenever  log t < (log 0.5 - log A)/p
    lo = np.minimum((np.log(0.5) - log_A) / p, -1e-9)
    lo = np.maximum(lo, -700.0)
    hi = np.zeros_like(lo)                       # log t = 0  ->  t = 1

    lo_bad = h(lo) >= 0.0                        # underflowed bracket
    lo = np.where(lo_bad, -700.0, lo)

    # Bisection to a safe bracket, then Newton polish. Plain bisection needs
    # ~90 halvings to reach machine precision from a bracket up to 700 wide,
    # and this inversion is the single hottest operation in the ABM: it runs
    # once per quadrature node per participation-constraint residual, tens of
    # thousands of times per contract solve. 30 halvings narrow the bracket to
    # ~7e-7, from which Newton -- quadratically convergent, and cheap because
    # f and f' share the same two exponentials -- reaches machine precision in
    # three steps. Newton is clamped to the retained bracket, so it can never
    # leave the interval bisection has already proved contains the root; the
    # result is identical to the 90-halving version to ~1e-15 at roughly a
    # third of the cost.
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        neg = h(mid) < 0.0
        lo = np.where(neg, mid, lo)
        hi = np.where(neg, hi, mid)

    log_t = 0.5 * (lo + hi)
    for _ in range(3):
        e_lin = np.exp(log_t)
        e_pow = np.exp(np.minimum(log_A + p * log_t, 700.0))
        f = e_lin + e_pow - 1.0
        df = e_lin + p * e_pow            # dh/d(log t) > 0 on the bracket
        step = np.where(df > 1e-300, f / df, 0.0)
        log_t = np.clip(log_t - step, lo, hi)

    t_star = np.exp(log_t)
    t_star = np.clip(t_star, 0.0, 1.0)
    out[pos] = t_star * xi
    return out


def wage_curvature(c1: float, c2: float) -> str:
    """Proposition 1: curvature of w(x) implied by the risk-attitude pair.

    Interpretation in the service-contract layer (principal = farmer):
      * c1 < c2  (farmer less risk-averse than provider) -> concave w:
        provider pay is INSENSITIVE to the outcome -> flat per-mu fee,
        the farmer absorbs production risk.
      * c1 = c2  -> linear w: proportional revenue-sharing contract.
      * c1 > c2  (farmer more risk-averse than provider) -> convex w:
        provider pay is HIGHLY outcome-sensitive -> the provider carries
        the risk, i.e. guaranteed-yield trusteeship / floor-plus-share.
    """
    if abs(c1 - c2) < 1e-12:
        return "linear"
    return "concave" if c1 < c2 else "convex"


def theoretical_mean_outcome(effort: float) -> float:
    """E[X; e] on the normalised support for the Example 2/3 density.

    f(x; e) = 1 + (x - 1/2)e on [0, 1] has mean 1/2 + e/12. The realised ABM
    outcome must be rescaled onto this support before it is fed to the wage
    schedule, otherwise the two are measured on different scales.
    """
    return 0.5 + float(np.clip(effort, 0.0, 1.0)) / 12.0


def normalised_contract_payment(outcome: float, b: float,
                                expected_payment: float,
                                c1: float, c2: float,
                                floor: float = 0.25,
                                ceiling: float = 2.50,
                                effort: float | None = None) -> float:
    """Return an outcome-contingent multiplier around the observed fee.

    The theoretical contract is solved on the normalised output support
    ``[0, 1]`` whereas the ABM's service prices are observed in currency per
    mu.  Dividing the realised theoretical wage by its model expectation
    preserves the observed fee as the *expected* payment while allowing the
    contract curvature and participation anchor to determine who bears
    realised production risk.  The bounds prevent a single noisy yield draw
    from creating an implausibly large or vanishing bill.
    """
    if (not np.isfinite(b) or not np.isfinite(expected_payment)
            or expected_payment <= _EPS):
        return 1.0

    # SCALE MATCHING. The realised ABM outcome is a fraction of expected gross
    # output and averages close to 1.0, whereas the theoretical support is
    # [0, 1] with mean 1/2 + e/12. Passing the realised value in raw pushed
    # roughly half of all outcomes onto the x = 1 ceiling, which made the wage
    # schedule FLAT above the median and destroyed exactly the convex upside
    # the outcome-contingent contract exists to represent. Centring the
    # realised outcome on the theoretical mean preserves the whole schedule.
    x_raw = float(max(outcome, 0.0))
    if effort is not None:
        x_raw *= theoretical_mean_outcome(effort)
    x = float(np.clip(x_raw, _EPS, 1.0))

    wage = float(_wage_from_implicit(x, b, 1.0, c1, c2)[0])
    return float(np.clip(wage / expected_payment, floor, ceiling))


CONTRACT_FORM = {
    "concave": "flat per-mu service fee (provider insured by farmer)",
    "linear": "proportional revenue share",
    "convex": "guaranteed-yield trusteeship (provider bears outcome risk)",
}


# --------------------------------------------------------------------------
# The contract problem
# --------------------------------------------------------------------------
@dataclass
class ContractSolution:
    """Outcome of solving the principal problem for one contracting pair."""

    e_star: float
    J_star: float
    b_star: float
    curvature: str
    e_grid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    J_grid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    b_grid: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    feasible: bool = True
    interior: bool = False
    agent_certainty_equivalent: float = float("nan")
    principal_expected_payment: float = float("nan")

    @property
    def contract_form(self) -> str:
        return CONTRACT_FORM[self.curvature]

    def wage_at(self, x, a: float, c1: float, c2: float):
        return _wage_from_implicit(x, self.b_star, a, c1, c2)


class ContractProblem:
    """Principal-agent problem with power utilities on both sides.

    Parameters
    ----------
    c1, c2 : float
        Relative risk aversion of principal and agent.  In the ABM these are
        agent attributes: c1 = farmer, c2 = provider (service layer).
    dist : OutcomeDistribution
        Effort-dependent outcome density f(x; e).
    gamma : float
        Scale of the agent convex effort disutility v(e) = gamma * e^2.
    u_min : float
        Agent reservation utility (outside option: alternative clients,
        off-farm work, or simply not serving this farmer).
    anchor : float or None
        Outcome `a` at which the initial condition w(a) = b is set.
        Defaults to the top of the support x1, as in step (ii) of the paper.
    effort_bounds : (float, float)
        Admissible effort interval [e0, e1].
    n_quad : int
        Simpson nodes for the expectation integrals.
    """

    def __init__(self, c1: float, c2: float, dist: OutcomeDistribution,
                 gamma: float = 0.1, u_min: float = 0.3,
                 anchor: float | None = None,
                 effort_bounds: tuple = (0.0, 1.0),
                 n_quad: int = 401) -> None:
        self.B = PowerUtility(c1)
        self.u = PowerUtility(c2)
        self.c1, self.c2 = float(c1), float(c2)
        self.dist = dist
        self.gamma = float(gamma)
        self.u_min = float(u_min)
        self.a = float(dist.x1 if anchor is None else anchor)
        self.e0, self.e1 = effort_bounds
        self.n_quad = n_quad if n_quad % 2 == 1 else n_quad + 1

    # -- agent side ------------------------------------------------------
    def v(self, e: float) -> float:
        """Effort disutility  v(e) = gamma e^2  (increasing, convex)."""
        return self.gamma * e * e

    def required_agent_utility(self, e: float) -> float:
        """RHS of the binding participation constraint, v(e) + Umin."""
        return self.v(e) + self.u_min

    # -- closed form for the equal-risk-aversion case --------------------
    def _b_closed_form(self, e: float) -> float:
        """Paper Eq. (8), valid only when c1 == c2 == c."""
        c = self.c1
        H = self.dist.expectation(
            lambda x: np.power(np.maximum(x, _EPS), 1.0 - c), e, self.n_quad)
        ratio = (1.0 - c) * self.required_agent_utility(e) / H
        return self.a * float(np.power(max(ratio, _EPS), 1.0 / (1.0 - c)))

    # -- general case: solve the participation constraint for b ----------
    def _pc_residual(self, b: float, e: float) -> float:
        w = lambda x: _wage_from_implicit(x, b, self.a, self.c1, self.c2)
        lhs = self.dist.expectation(lambda x: self.u.u(w(x)), e, self.n_quad)
        return lhs - self.required_agent_utility(e)

    def solve_b(self, e: float) -> float:
        """Find b(e): the anchor making the PC bind at effort e (step iii).

        Returns NaN when no feasible b in (0, a) exists, i.e. when the
        paper condition (9) fails and the principal would have to pay more
        than the output is worth.
        """
        if abs(self.c1 - self.c2) < 1e-12:
            b = self._b_closed_form(e)
            return b if 0.0 < b < self.a else float("nan")

        lo, hi = self.a * 1e-8, self.a * (1.0 - 1e-9)
        f_lo, f_hi = self._pc_residual(lo, e), self._pc_residual(hi, e)
        if f_lo > 0.0:            # even a token payment over-satisfies the PC
            return lo
        if f_hi < 0.0:            # paying the whole output cannot satisfy it
            return float("nan")
        return float(brentq(lambda b: self._pc_residual(b, e), lo, hi,
                            xtol=1e-12, rtol=1e-10, maxiter=200))

    # -- principal objective ---------------------------------------------
    def J(self, e: float, b: float | None = None) -> float:
        """Principal expected utility J(e) = E[B(X_e - w(X_e; e))]."""
        b = self.solve_b(e) if b is None else b
        if not np.isfinite(b):
            return float("-inf")
        w = lambda x: _wage_from_implicit(x, b, self.a, self.c1, self.c2)
        return self.dist.expectation(
            lambda x: self.B.u(np.maximum(x - w(x), _EPS)), e, self.n_quad)

    def expected_payment(self, e: float, b: float) -> float:
        w = lambda x: _wage_from_implicit(x, b, self.a, self.c1, self.c2)
        return self.dist.expectation(w, e, self.n_quad)

    # -- optimal effort ---------------------------------------------------
    def solve(self, e_grid=None, refine: bool = True) -> ContractSolution:
        """Full five-step procedure -> (e*, J(e*), b(e*)).

        A grid search establishes the basin (J need not be concave in e, as
        the paper stresses), optionally refined by golden section.
        """
        if e_grid is None:
            lo = max(self.e0, 1e-3)
            e_grid = np.linspace(lo, self.e1, 25)
        e_grid = np.asarray(list(e_grid), dtype=float)

        b_vals = np.array([self.solve_b(e) for e in e_grid])
        J_vals = np.array([self.J(e, b) if np.isfinite(b) else -np.inf
                           for e, b in zip(e_grid, b_vals)])

        if not np.any(np.isfinite(J_vals)):
            return ContractSolution(float("nan"), float("-inf"), float("nan"),
                                    wage_curvature(self.c1, self.c2),
                                    e_grid, J_vals, b_vals, feasible=False)

        k = int(np.argmax(np.where(np.isfinite(J_vals), J_vals, -np.inf)))
        e_star, J_star = float(e_grid[k]), float(J_vals[k])

        if refine and 0 < k < len(e_grid) - 1:
            e_star, J_star = self._golden(e_grid[k - 1], e_grid[k + 1],
                                          e_star, J_star)

        b_star = self.solve_b(e_star)
        interior = bool(self.e0 + 1e-6 < e_star < self.e1 - 1e-6)
        ce = (self.u.inverse_u(self.required_agent_utility(e_star))
              if np.isfinite(b_star) else float("nan"))

        return ContractSolution(
            e_star=e_star, J_star=J_star, b_star=float(b_star),
            curvature=wage_curvature(self.c1, self.c2),
            e_grid=e_grid, J_grid=J_vals, b_grid=b_vals,
            feasible=bool(np.isfinite(b_star)), interior=interior,
            agent_certainty_equivalent=ce,
            principal_expected_payment=(self.expected_payment(e_star, b_star)
                                        if np.isfinite(b_star) else float("nan")),
        )

    def _golden(self, lo: float, hi: float, e_best: float, J_best: float,
                tol: float = 1e-4, max_iter: int = 60):
        invphi = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = hi - invphi * (hi - lo), lo + invphi * (hi - lo)
        fc, fd = self.J(c), self.J(d)
        for _ in range(max_iter):
            if hi - lo < tol:
                break
            if fc > fd:
                hi, d, fd = d, c, fc
                c = hi - invphi * (hi - lo)
                fc = self.J(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + invphi * (hi - lo)
                fd = self.J(d)
        e_ref = c if fc > fd else d
        J_ref = max(fc, fd)
        return (e_ref, J_ref) if J_ref > J_best else (e_best, J_best)


# --------------------------------------------------------------------------
# Cached driver for use inside the ABM
# --------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def solve_effort_cached(c1: float, c2: float, gamma: float, u_min: float,
                        scale: float = 1.0, e_max: float = 1.0,
                        n_grid: int = 13, n_quad: int = 121):
    """Memoised contract solution for one contracting pair.

    Agents quantise (c1, c2, gamma, u_min) before calling, so a population of
    thousands of farmers collapses onto a few thousand cache entries and a
    full season of contracting costs milliseconds.
    """
    prob = ContractProblem(c1, c2, LinearTiltDensity(scale), gamma=gamma,
                           u_min=u_min, effort_bounds=(0.0, e_max),
                           n_quad=n_quad)
    sol = prob.solve(e_grid=np.linspace(1e-3, e_max, n_grid))
    return (sol.e_star, sol.J_star, sol.b_star,
            sol.principal_expected_payment, sol.curvature, sol.feasible)


def quantise(value: float, step: float = 0.02) -> float:
    """Round a risk/cost parameter onto a grid so the cache actually hits."""
    return round(round(value / step) * step, 6)
