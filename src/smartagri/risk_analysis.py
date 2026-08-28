"""
risk_analysis.py
================
Identifies the effect of RISK ATTITUDE on adoption, effort and resilience,
separately from the farm characteristics it is correlated with.

Why this module exists
----------------------
In the baseline population c1 is deliberately correlated with observables
(model_params.yaml, risk_attitude.farmer.covariates): larger, better educated,
more liquid and cooperative-member farmers are less risk-averse. That is
empirically motivated, but it means a raw comparison of adoption across risk
quartiles is CONFOUNDED. The most risk-averse quartile is also the smallest
and least liquid, and those act in the opposite direction to risk itself, so
the raw gradient can be flat, non-monotone or even reversed while the
underlying risk mechanism is working perfectly.

Three designs are provided, and they answer different questions:

    R1  raw gradient        what a survey of this population would observe,
                            confounded exactly as field data would be
    R2  orthogonal design   covariates switched off, so c1 varies
                            independently of size and liquidity: the PURE
                            causal effect of risk attitude
    R3  stratified partial  raw population, compared within narrow farm-size
                            strata: the partial effect, which is what a
                            regression with controls would recover

R2 is the identified quantity. R1 minus R3 is a direct measure of how much
confounding a naive study of this system would suffer.

Run
---
    python -m smartagri.risk_analysis --all
    python -m smartagri.risk_analysis --sweep
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import STATE_ACQUIRED, STATE_USED, TECHS
from .model import SmartAgriModel

OUT = Path(__file__).resolve().parents[2] / "outputs" / "risk"


def _farmer_frame(m: SmartAgriModel) -> pd.DataFrame:
    # IDENTIFICATION NOTE. Risk attitude is now endogenous: c_risk =
    # c_risk_base + risk_stress, and risk_stress is driven by realised severe
    # loss (corr with mean loss about +0.78). Realised c_risk is therefore
    # partly an OUTCOME of the very behaviour being explained, and regressing
    # adoption on it is simultaneity-biased -- it flipped the orthogonal
    # slope's sign, from +0.050 to -0.064, which is how this was caught.
    # `c1` is the EXOGENOUS component and is the only valid regressor here;
    # the realised value is kept alongside for description only.
    return pd.DataFrame([{
        "c1": f.c_risk_base, "c1_realised": f.c_risk,
        "risk_stress": f.risk_stress,
        "area": f.area_mu, "liquidity": f.liquidity,
        "literacy": f.digital_literacy, "terrain": f.terrain,
        "irrigation": int(f.irrigation),
        "adopt": int(any(f.state[t] >= STATE_ACQUIRED for t in TECHS)),
        "used": int(any(f.state[t] >= STATE_USED for t in TECHS)),
        "effort": float(np.mean([f.effort[t] for t in TECHS])),
        "experience": int(sum(f.experience.values())),
        "loss": float(np.mean(f.loss_history)) if f.loss_history else np.nan,
        "income": float(np.mean(f.income_history)) if f.income_history else np.nan,
    } for f in m.farmers])


def _strip_covariates(m: SmartAgriModel) -> None:
    """Re-draw c1 independently of every other farmer attribute."""
    ra = m.cfg["params"]["risk_attitude"]["farmer"]
    lo, hi = ra["support"]
    rng = np.random.default_rng(m.seed + 99991)
    for f in m.farmers:
        base = float(rng.beta(*ra["params"]))
        # c_risk_base MUST be set too. Risk attitude is now endogenous: the
        # season loop recomputes c_risk from c_risk_base + risk_stress, so
        # assigning c_risk alone is silently reverted on the first season and
        # the orthogonal design would quietly revert to the raw population.
        v = float(np.clip(lo + base * (hi - lo), 0.08, 0.92))
        f.c_risk = f.c_risk_base = v


def _run(seed: int, n: int, seasons: int, orthogonal: bool) -> pd.DataFrame:
    m = SmartAgriModel("baseline", seed=seed, n_farmers=n, seasons=seasons)
    if orthogonal:
        _strip_covariates(m)
    m.run()
    df = _farmer_frame(m)
    df["orthogonal"] = orthogonal
    df["seed"] = seed
    return df


def _quartile_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["q"] = pd.qcut(d.c1, 4, labels=["Q1 least", "Q2", "Q3", "Q4 most"])
    return (d.groupby("q", observed=True)
              .agg(n=("c1", "size"), c1=("c1", "mean"), area=("area", "mean"),
                   adopt=("adopt", "mean"), used=("used", "mean"),
                   effort=("effort", "mean"))
              .round(4).reset_index())


def _slope(df: pd.DataFrame, y: str, controls: bool):
    """OLS slope of y on c1, WITH its standard error.

    Returning a bare point estimate is how a slope of +0.050 came to be read
    as a clean causal effect when it was 0.41 SE from zero. This design is
    badly underpowered: adoption is near-Bernoulli with p about 0.2 and c1
    has sd about 0.13, so at n=600 the slope SE is roughly 0.12 and only
    effects larger than about 0.24 are detectable at all. Prefer sweep(),
    a paired population-level manipulation under common random numbers,
    whenever the question is whether risk attitude moves outcomes.
    """
    d = df.dropna(subset=[y, "c1"])
    if len(d) < 20:
        return np.nan, np.nan
    cols = ["c1"]
    if controls:
        d = d.assign(log_area=np.log(d.area.clip(lower=0.5)))
        cols += ["log_area", "liquidity", "literacy", "irrigation"]
    X = np.column_stack([np.ones(len(d))] + [d[c].to_numpy(float) for c in cols])
    yv = d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta
    dof = max(len(d) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    try:
        se = float(np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1]))
    except np.linalg.LinAlgError:
        se = float("nan")
    return float(beta[1]), se


# ===========================================================================
def r1_r2_r3(n: int, seasons: int, seeds: list[int]) -> dict:
    raw = pd.concat([_run(s, n, seasons, False) for s in seeds], ignore_index=True)
    ort = pd.concat([_run(s, n, seasons, True) for s in seeds], ignore_index=True)

    print("\n[R1] RAW gradient - confounded, as a field survey would see it")
    print(_quartile_table(raw).to_string(index=False))
    print(f"     c1 correlates with log(area) at r = "
          f"{np.corrcoef(raw.c1, np.log(raw.area.clip(lower=0.5)))[0,1]:+.3f}")

    print("\n[R2] ORTHOGONAL design - c1 independent of everything else")
    print(_quartile_table(ort).to_string(index=False))
    print(f"     c1 correlates with log(area) at r = "
          f"{np.corrcoef(ort.c1, np.log(ort.area.clip(lower=0.5)))[0,1]:+.3f}")

    print("\n[R3] PARTIAL effects - OLS slope of the outcome on c1")
    rows = []
    for y in ("adopt", "used", "effort"):
        b_raw, se_raw = _slope(raw, y, False)
        b_ctl, se_ctl = _slope(raw, y, True)
        b_ort, se_ort = _slope(ort, y, False)
        t_ort = b_ort / se_ort if se_ort and se_ort == se_ort else float("nan")
        rows.append({
            "outcome": y,
            "raw": round(b_raw, 4), "raw_se": round(se_raw, 4),
            "controls": round(b_ctl, 4), "ctl_se": round(se_ctl, 4),
            "orthogonal": round(b_ort, 4), "orth_se": round(se_ort, 4),
            "orth_t": round(t_ort, 2),
            "sig": "yes" if abs(t_ort) > 1.96 else "no",
        })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False))
    print("\n     raw is what a naive study reports; orthogonal is the causal")
    print("     estimate. CHECK `sig` BEFORE INTERPRETING ANY OF THESE. This")
    print("     cross-sectional design is underpowered (slope SE about 0.12")
    print("     at n=600), so a non-significant slope must NOT be read as an")
    print("     effect in either direction. Use sweep() for a powered test.")

    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "risk_raw.csv", index=False)
    ort.to_csv(OUT / "risk_orthogonal.csv", index=False)
    tab.to_csv(OUT / "risk_partial_effects.csv", index=False)
    return {"raw": raw, "orthogonal": ort, "partial": tab}


def sweep(n: int, seasons: int, seeds: list[int]) -> pd.DataFrame:
    """Shift the whole risk distribution and watch system outcomes move.

    NOTE the realised mean does not equal the target. c1 is clipped to
    [0.08, 0.92] because the power utility degenerates at the endpoints, so
    shifting the distribution below about 0.45 piles mass on the lower clip
    and the realised mean stays above the target (target 0.25 realises near
    0.32). The sweep therefore explores roughly 0.32 to 0.87, not 0.25 to
    0.85. Always read results against `realised_mean_c1`, which is reported
    alongside the target for exactly this reason.
    """
    rows = []
    for target in (0.25, 0.40, 0.55, 0.70, 0.85):
        for s in seeds:
            m = SmartAgriModel("baseline", seed=s, n_farmers=n, seasons=seasons)
            ra = m.cfg["params"]["risk_attitude"]["farmer"]
            lo, hi = ra["support"]
            half = (hi - lo) / 2.0
            rng = np.random.default_rng(s + 7717)
            for f in m.farmers:
                base = float(rng.beta(*ra["params"]))
                v = float(np.clip(
                    (target - half) + base * (hi - lo), 0.08, 0.92))
                f.c_risk = f.c_risk_base = v   # see note in _strip_covariates
            m.run()
            d = m.to_dataframe()
            rows.append({
                "target_mean_c1": target, "seed": s,
                "realised_mean_c1": float(np.mean([f.c_risk for f in m.farmers])),
                "adopt": float(np.mean([d[f"adopt_{t}"].mean() for t in TECHS])),
                "effective_use": d.effective_use_rate.mean(),
                "effort_T3": d.effort_T3.mean(),
                "mitigation": d.mitigation_rate.mean(),
                "loss": d.mean_loss_fraction.mean(),
                "income_p10": d.income_p10.mean(),
            })
    df = pd.DataFrame(rows)
    agg = df.groupby("target_mean_c1").mean(numeric_only=True).drop(
        columns=["seed"]).round(4).reset_index()
    print("\n[R4] RISK SWEEP - whole population shifted, everything else fixed")
    print(agg.to_string(index=False))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "risk_sweep_raw.csv", index=False)
    agg.to_csv(OUT / "risk_sweep.csv", index=False)
    return agg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartagri.risk_analysis")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--designs", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n-farmers", type=int, default=250)
    ap.add_argument("--seasons", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    if a.all or not (a.designs or a.sweep):
        a.designs = a.sweep = True

    seeds = [11 + i for i in range(a.seeds)]
    t0 = time.time()
    print("=" * 74)
    print("RISK ATTITUDE ANALYSIS")
    print("=" * 74)
    if a.designs:
        r1_r2_r3(a.n_farmers, a.seasons, seeds)
    if a.sweep:
        sweep(a.n_farmers, a.seasons, seeds)
    print(f"\nOutputs -> {OUT}")
    print(f"Runtime: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
