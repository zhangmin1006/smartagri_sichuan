"""
Validation of contract.py against the published numbers in
Guo, Parlar & Zhang (2025).  If these pass, the solver is a faithful
implementation and everything the ABM builds on top of it inherits that.

Targets
-------
Example 2 (c1 = c2 = c, gamma = 0.1, Umin = 0.3):
    c = 0.2 -> e* = 0.60, J(e*) = 0.52
    c = 0.6 -> e* = 1.00, J(e*) = 1.92
    Table 2: the full (gamma, c) sensitivity grid.
Example 3 (c1 = 1/4, c2 = 1/3, gamma = 0.1, Umin = 0.3):
    Table 3: b(e) and J(e) for e = 0.1 .. 1.0, optimum e* = 0.9, J = 0.6999.
Proposition 1: curvature of w(x).
Theorem 1: 0 < dw/dx < 1 and 0 < w(x) < x.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from smartagri.contract import (ContractProblem, LinearTiltDensity,
                                _wage_from_implicit, wage_curvature)

DIST = LinearTiltDensity()
GAMMA, UMIN = 0.1, 0.3

# Table 2 of the paper: rows gamma = 0.1..0.9, cols c = 0.1..0.9
TABLE2 = {
    (0.1, 0.1): (0.488, 0.350), (0.1, 0.3): (0.812, 0.749),
    (0.1, 0.5): (1.000, 1.411), (0.1, 0.7): (1.000, 2.730),
    (0.1, 0.9): (1.000, 9.307),
    (0.3, 0.1): (0.163, 0.335), (0.3, 0.3): (0.278, 0.717),
    (0.3, 0.5): (0.706, 1.355), (0.3, 0.7): (1.000, 2.726),
    (0.3, 0.9): (1.000, 9.307),
    (0.5, 0.1): (0.098, 0.331), (0.5, 0.3): (0.168, 0.710),
    (0.5, 0.5): (0.459, 1.334), (0.5, 0.7): (1.000, 2.718),
    (0.5, 0.9): (1.000, 9.307),
    (0.7, 0.1): (0.070, 0.330), (0.7, 0.3): (0.120, 0.707),
    (0.7, 0.5): (0.343, 1.325), (0.7, 0.7): (1.000, 2.702),
    (0.7, 0.9): (1.000, 9.307),
    (0.9, 0.1): (0.054, 0.329), (0.9, 0.3): (0.094, 0.706),
    (0.9, 0.5): (0.276, 1.320), (0.9, 0.7): (0.898, 2.682),
    (0.9, 0.9): (1.000, 9.307),
}

# Table 3 of the paper: c1 = 1/4, c2 = 1/3
TABLE3 = {
    0.1: (0.1654, 0.6607), 0.2: (0.1654, 0.6698), 0.3: (0.1671, 0.6778),
    0.4: (0.1704, 0.6846), 0.5: (0.1753, 0.6902), 0.6: (0.1818, 0.6946),
    0.7: (0.1900, 0.6977), 0.8: (0.1998, 0.6995), 0.9: (0.2113, 0.6999),
    1.0: (0.2245, 0.6988),
}


def _report(title, rows, headers):
    print("\n" + title)
    print("-" * len(title))
    print("  ".join(f"{h:>12s}" for h in headers))
    for r in rows:
        print("  ".join(f"{v:>12}" for v in r))


def test_example2_table2():
    """Equal risk aversion: reproduce the (e*, J*) sensitivity grid."""
    rows, max_e_err, max_J_err = [], 0.0, 0.0
    for (gam, c), (e_ref, J_ref) in sorted(TABLE2.items()):
        prob = ContractProblem(c, c, DIST, gamma=gam, u_min=UMIN,
                               effort_bounds=(0.0, 1.0), n_quad=801)
        sol = prob.solve(e_grid=np.linspace(1e-4, 1.0, 201))
        de, dJ = abs(sol.e_star - e_ref), abs(sol.J_star - J_ref)
        max_e_err, max_J_err = max(max_e_err, de), max(max_J_err, dJ)
        rows.append((f"{gam:.1f}", f"{c:.1f}", f"{e_ref:.3f}",
                     f"{sol.e_star:.3f}", f"{J_ref:.3f}", f"{sol.J_star:.3f}",
                     "OK" if de < 0.01 and dJ < 0.01 else "MISMATCH"))
    _report("Example 2 / Table 2  (c1 = c2)", rows,
            ["gamma", "c", "e* paper", "e* ours", "J* paper", "J* ours", ""])
    print(f"max |de*| = {max_e_err:.4f}   max |dJ*| = {max_J_err:.4f}")
    assert max_e_err < 0.01, f"effort mismatch {max_e_err}"
    assert max_J_err < 0.01, f"objective mismatch {max_J_err}"


def test_example3_table3():
    """Unequal risk aversion c1 = 1/4 < c2 = 1/3: reproduce b(e) and J(e)."""
    c1, c2 = 0.25, 1.0 / 3.0
    prob = ContractProblem(c1, c2, DIST, gamma=GAMMA, u_min=UMIN, n_quad=2001)
    rows, max_b_err, max_J_err = [], 0.0, 0.0
    for e, (b_ref, J_ref) in sorted(TABLE3.items()):
        b = prob.solve_b(e)
        J = prob.J(e, b)
        db, dJ = abs(b - b_ref), abs(J - J_ref)
        max_b_err, max_J_err = max(max_b_err, db), max(max_J_err, dJ)
        rows.append((f"{e:.1f}", f"{b_ref:.4f}", f"{b:.4f}",
                     f"{J_ref:.4f}", f"{J:.4f}",
                     "OK" if db < 5e-3 and dJ < 5e-3 else "MISMATCH"))
    _report("Example 3 / Table 3  (c1 = 1/4 < c2 = 1/3)", rows,
            ["e", "b paper", "b ours", "J paper", "J ours", ""])
    print(f"max |db| = {max_b_err:.5f}   max |dJ| = {max_J_err:.5f}")

    sol = prob.solve(e_grid=np.round(np.arange(0.1, 1.01, 0.1), 3), refine=False)
    print(f"argmax over the paper grid: e* = {sol.e_star:.2f} "
          f"(paper 0.90), J* = {sol.J_star:.4f} (paper 0.6999)")
    assert max_b_err < 5e-3
    assert max_J_err < 5e-3
    assert abs(sol.e_star - 0.9) < 1e-6, "interior optimum not reproduced"


def test_proposition1_curvature():
    """w concave iff c1 < c2, convex iff c1 > c2, linear iff equal."""
    x = np.linspace(0.05, 1.0, 200)
    cases = [(0.2, 0.2, "linear"), (0.2, 0.5, "concave"), (0.5, 0.2, "convex")]
    rows = []
    for c1, c2, expected in cases:
        w = _wage_from_implicit(x, b=0.3, a=1.0, c1=c1, c2=c2)
        second = np.gradient(np.gradient(w, x), x)
        interior = second[5:-5]
        if expected == "linear":
            ok = np.max(np.abs(interior)) < 1e-6
        elif expected == "concave":
            ok = np.all(interior < 1e-9)
        else:
            ok = np.all(interior > -1e-9)
        rows.append((f"{c1:.2f}", f"{c2:.2f}", expected,
                     wage_curvature(c1, c2), "OK" if ok else "MISMATCH"))
        assert wage_curvature(c1, c2) == expected
        assert ok, f"numerical curvature contradicts Prop.1 for c1={c1}, c2={c2}"
    _report("Proposition 1: curvature of w(x)", rows,
            ["c1", "c2", "expected", "predicted", ""])


def test_theorem1_bounds():
    """Theorem 1: w increases in x with slope in (0,1), and 0 < w(x) < x."""
    x = np.linspace(0.02, 1.0, 300)
    for c1, c2 in [(0.2, 0.5), (0.5, 0.2), (0.3, 0.3), (0.15, 0.8)]:
        w = _wage_from_implicit(x, b=0.25, a=1.0, c1=c1, c2=c2)
        slope = np.gradient(w, x)
        assert np.all(w > 0), "wage must be positive (rM > 0)"
        assert np.all(w < x + 1e-9), "wage must not exceed the outcome (rO > 0)"
        assert np.all(slope > -1e-9) and np.all(slope < 1.0 + 1e-6), \
            f"slope out of (0,1) for c1={c1}, c2={c2}"
    print("\nTheorem 1 bounds: 0 < w(x) < x and 0 < dw/dx < 1 hold on all cases")


if __name__ == "__main__":
    test_example2_table2()
    test_example3_table3()
    test_proposition1_curvature()
    test_theorem1_bounds()
    print("\nALL VALIDATION TESTS PASSED")
