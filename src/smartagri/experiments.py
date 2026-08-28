"""
experiments.py
==============
Runs the analyses that answer the three research questions and writes the
results to outputs/.

    RQ1  How do farmers decide adoption and effort?
         -> adoption and effort by risk-aversion quartile
         -> contract form vs the (c1, c2) risk-attitude pair (Proposition 1)
         -> the verifiability experiment: what monitoring technology buys

    RQ2  How does smart technology change resilience to shocks?
         -> loss, income drop and recovery for adopters vs non-adopters
         -> the D3 stress test, where technology availability collapses

    RQ3  How effective are policies, and what roles do they play?
         -> scenario comparison across all objectives
         -> the voucher-without-capacity congestion result

Usage
-----
    python -m smartagri.experiments --quick
    python -m smartagri.experiments --full --reps 5
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .agents import STATE_ACQUIRED, STATE_EFFECTIVE, STATE_USED, TECHS
from .contract import ContractProblem, LinearTiltDensity, wage_curvature
from .model import SmartAgriModel

OUT = Path(__file__).resolve().parents[2] / "outputs" / "experiments"


def _ensure_out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


# ===========================================================================
# RQ1
# ===========================================================================
def rq1_adoption_and_effort(n_farmers: int, seasons: int,
                            seed: int = 11) -> dict:
    """Who adopts, in which mode, and how much effort is contracted."""
    m = SmartAgriModel("baseline", seed=seed, n_farmers=n_farmers,
                       seasons=seasons)
    m.run()

    rows = []
    for f in m.farmers:
        rows.append({
            "uid": f.uid, "county": f.county, "terrain": f.terrain,
            "area_mu": f.area_mu, "c_farmer": f.c_risk,
            "digital_literacy": f.digital_literacy, "irrigation": f.irrigation,
            "coop": f.coop_member, "service_distance": f.service_distance,
            "wealth": f.wealth, "exited": f.exited,
            **{f"mode_{t}": f.mode[t] for t in TECHS},
            **{f"effort_{t}": f.effort[t] for t in TECHS},
            **{f"state_{t}": f.state[t] for t in TECHS},
            **{f"form_{t}": f.contract_form[t] for t in TECHS},
            "any_adopted": any(
                f.mode[t] != "none"
                and f.state[t] in (STATE_ACQUIRED, STATE_USED, STATE_EFFECTIVE)
                for t in TECHS
            ),
            "any_effective": any(
                f.mode[t] != "none" and f.state[t] == STATE_EFFECTIVE
                for t in TECHS
            ),
            "mean_effort": float(np.mean([f.effort[t] for t in TECHS])),
        })
    farmers = pd.DataFrame(rows)
    farmers["risk_quartile"] = pd.qcut(farmers["c_farmer"], 4,
                                       labels=["Q1 least averse", "Q2", "Q3",
                                               "Q4 most averse"])
    by_risk = farmers.groupby("risk_quartile", observed=True).agg(
        n=("uid", "size"), c_farmer=("c_farmer", "mean"),
        area_mu=("area_mu", "mean"),
        adopt_any=("any_adopted", "mean"),
        effective=("any_effective", "mean"),
        service_T3=("mode_T3", lambda s: (s == "service").mean()),
        own_T3=("mode_T3", lambda s: (s == "own").mean()),
        adopt_T2=("mode_T2", lambda s: (s != "none").mean()),
        mean_effort=("mean_effort", "mean"),
    ).reset_index()

    provs = pd.DataFrame([{"uid": p.uid, "county": p.county,
                           "c_provider": p.c_risk, "units": p.units,
                           "utilisation": p.utilisation(),
                           "mean_effort": p.mean_effort_season}
                          for p in m.providers])
    return {"farmers": farmers, "by_risk": by_risk, "providers": provs,
            "model": m}


def rq1_contract_map(grid: int = 21) -> pd.DataFrame:
    """Map the (c1, c2) plane: contract form and demanded effort.

    This is the direct ABM-facing reading of Proposition 1 and Section 4:
    what the risk-attitude PAIR implies for contract shape and effort.
    """
    cs = np.linspace(0.12, 0.88, grid)
    dist = LinearTiltDensity()
    rows = []
    for c1 in cs:
        for c2 in cs:
            prob = ContractProblem(float(c1), float(c2), dist, gamma=0.12,
                                   u_min=0.28, n_quad=121)
            sol = prob.solve(e_grid=np.linspace(1e-3, 1.0, 13))
            rows.append({
                "c1_farmer": float(c1), "c2_provider": float(c2),
                "curvature": wage_curvature(float(c1), float(c2)),
                "e_star": sol.e_star, "J_star": sol.J_star, "b_star": sol.b_star,
                "interior": sol.interior, "feasible": sol.feasible,
                "expected_payment": sol.principal_expected_payment,
            })
    return pd.DataFrame(rows)


def rq1_verifiability_experiment(n_farmers: int, seasons: int,
                                 seed: int = 13) -> pd.DataFrame:
    """What monitoring technology buys: first-best vs unverifiable effort.

    The paper's solution assumes effort and outcome are observable AND
    verifiable. This experiment switches that assumption off and measures
    the cost, which is the quantitative case for digital monitoring as an
    ENABLING condition rather than merely a productivity tool.
    """
    rows = []
    for label, mult in (("first-best (fully verifiable)", 1.0),
                        ("partial monitoring (as configured)", None),
                        ("no monitoring (shirking)", 0.0)):
        m = SmartAgriModel("baseline", seed=seed, n_farmers=n_farmers,
                           seasons=seasons)
        if mult is not None:
            v = m.cfg["params"]["contract"]["verifiability"]
            for k in ("with_T1_remote_sensing", "with_T2_sensors",
                      "with_T3_telemetry"):
                v[k] = 1.0 if mult == 1.0 else 0.0
        m.run()
        df = m.to_dataframe()
        rows.append({
            "regime": label,
            "mean_effort_T3": df["effort_T3"].mean(),
            "effective_use_rate": df["effective_use_rate"].mean(),
            "mean_loss_fraction": df["mean_loss_fraction"].mean(),
            "avoided_loss_fraction": df["avoided_loss_fraction"].mean(),
            "mean_income": df["mean_income"].mean(),
        })
    return pd.DataFrame(rows)


# ===========================================================================
# RQ2
# ===========================================================================
def rq2_resilience(n_farmers: int, seed: int = 17) -> dict:
    """Resilience under a controlled shock sequence.

    Seasons 1-3 are calm so adoption can settle, season 4 delivers the shock
    being tested, and seasons 5-8 measure recovery. Running the SAME shock
    against every technology regime is what makes the comparison causal
    within the model rather than confounded by different shock draws.
    """
    seasons = 8
    calm = {s: [] for s in range(1, seasons + 1)}

    def _design(**hits):
        d = dict(calm)
        for k, v in hits.items():
            d[int(k[1:])] = v
        return d

    designs = {
        "no shock (counterfactual)": dict(calm),
        "D1 drought": _design(s4=["D1"]),
        "D2 flood": _design(s4=["D2"]),
        "D3 compound (tech degraded)": _design(s4=["D3"]),
        "D1 then D2 (repeat, different type)": _design(s4=["D1"], s5=["D2"]),
        "D2 then D3 (escalating)": _design(s4=["D2"], s5=["D3"]),
        "D3 twice (repeated compound)": _design(s4=["D3"], s5=["D3"]),
    }

    # Paired counterfactual: identical seed and population, no shocks. Income
    # is compared to the SAME model in the SAME season without the shock, so
    # the drop is not confounded by the adoption trend, which rises steadily
    # over the first seasons and would otherwise mask the shock entirely.
    cf = SmartAgriModel("baseline", seed=seed, n_farmers=n_farmers,
                        seasons=seasons)
    cf.run(forced=dict(calm))
    cf_df = cf.to_dataframe().set_index("season")

    rows, panels = [], {}
    for label, forced in designs.items():
        m = SmartAgriModel("baseline", seed=seed, n_farmers=n_farmers,
                           seasons=seasons)
        m.run(forced=forced)
        df = m.to_dataframe()
        panels[label] = df

        pre = float(cf_df.loc[4, "mean_income"])
        shock_income = df.loc[df.season == 4, "mean_income"].mean()
        drop = (pre - shock_income) / max(abs(pre), 1e-9)
        rec = np.nan
        for s in range(5, seasons + 1):
            ref = float(cf_df.loc[s, "mean_income"])
            val = float(df.loc[df.season == s, "mean_income"].iloc[0])
            if val >= 0.95 * ref:
                rec = float(s - 4)
                break

        adopters = [
            f for f in m.farmers
            if any(
                f.mode[t] != "none"
                and f.state[t] in (STATE_USED, STATE_EFFECTIVE)
                for t in TECHS
            )
        ]
        non = [
            f for f in m.farmers
            if all(
                f.mode[t] == "none"
                or f.state[t] not in (STATE_USED, STATE_EFFECTIVE)
                for t in TECHS
            )
        ]

        def _loss4(group):
            return float(np.mean([g.loss_history[3] for g in group
                                  if len(g.loss_history) > 3])) if group else np.nan

        # Two estimates of "what the technology did", deliberately reported
        # side by side:
        #   naive  -- cross-sectional adopter vs non-adopter loss gap, which is
        #             confounded by selection (adopters are larger, on flatter
        #             land and more often already irrigated)
        #   causal -- within-farmer counterfactual computed inside the model
        #             (damage with vs without mitigation, same farmer, same
        #             shock), which is what the technology actually removed
        la, ln = _loss4(adopters), _loss4(non)
        naive = float((ln - la) / ln) if ln and ln > 1e-9 else np.nan
        causal = float(df.loc[df.season == 4, "mitigation_rate"].mean())
        if abs(drop) < 0.02:
            rec = 0.0

        rows.append({
            "design": label,
            "naive_adopter_effect": naive,
            "causal_mitigation_rate": causal,
            "selection_inflation": (naive / causal
                                    if causal and causal > 1e-9 else np.nan),
            "pre_shock_income": pre,
            "shock_season_income": shock_income,
            "income_drop_pct": 100 * drop,
            "recovery_seasons": rec,
            "loss_shock_season": df.loc[df.season == 4, "mean_loss_fraction"].mean(),
            "avoided_shock_season": df.loc[df.season == 4,
                                           "avoided_loss_fraction"].mean(),
            "loss_adopters": la,
            "loss_non_adopters": ln,
            "n_adopters": len(adopters), "n_non_adopters": len(non),
            "exit_rate_end": df["exit_rate"].iloc[-1],
            "cumulative_loss_area": float(df["mean_loss_fraction"].sum()),
        })
    return {"summary": pd.DataFrame(rows), "panels": panels}


# ===========================================================================
# RQ3
# ===========================================================================
def rq3_policies(n_farmers: int, seasons: int, reps: int,
                 seed0: int = 101) -> dict:
    """Compare policy scenarios on every objective, with replication."""
    m0 = SmartAgriModel("baseline", n_farmers=4, seasons=1)
    scenarios = [s["key"] for s in m0.cfg["params"]["scenarios"]]

    # every scenario faces the SAME shock sequence per replicate, so
    # differences are attributable to policy and not to shock luck
    rng = np.random.default_rng(seed0)
    shock_designs = []
    for _ in range(reps):
        design = {}
        for s in range(1, seasons + 1):
            # Shock mix over the three modelled hazards. Probabilities are
            # scaled up from the per-shock annual priors so that a short run
            # still exercises each hazard; the RELATIVE frequencies follow
            # D1 0.35 : D2 0.40 : D3 0.12 from disruptions.yaml.
            r = rng.random()
            if r < 0.25:
                design[s] = ["D1"]
            elif r < 0.53:
                design[s] = ["D2"]
            elif r < 0.62:
                design[s] = ["D3"]
            else:
                design[s] = []
        shock_designs.append(design)

    rows, series = [], []
    for sc in scenarios:
        for r, design in enumerate(shock_designs):
            m = SmartAgriModel(sc, seed=seed0 + r, n_farmers=n_farmers,
                               seasons=seasons)
            m.run(forced=design)
            df = m.to_dataframe()
            df["scenario"], df["rep"] = sc, r
            series.append(df)
            rows.append({
                "scenario": sc, "rep": r,
                "avoided_loss": df["avoided_loss_fraction"].mean(),
                "mean_loss": df["mean_loss_fraction"].mean(),
                "mitigation_rate": df["mitigation_rate"].mean(),
                "effective_use_rate": df["effective_use_rate"].mean(),
                "adopt_T3": df["adopt_T3"].mean(),
                "service_T3": df["service_T3"].mean(),
                "mean_effort_T3": df["effort_T3"].mean(),
                "mean_wait_days": df["mean_wait_days"].mean(),
                "peak_wait_days": df["mean_wait_days"].max(),
                "backlog_mu": df["backlog_mu"].mean(),
                "mean_income": df["mean_income"].mean(),
                "income_p10": df["income_p10"].mean(),
                "gini": df["gini_income"].mean(),
                "exit_rate": df["exit_rate"].iloc[-1],
                "fiscal_cost": df["fiscal_cumulative"].iloc[-1],
                "equity_gap": df["equity_gap"].mean(),
                "mountain_gap": df["mountain_gap"].mean(),
                "recovery_seasons": df["recovery_seasons_mean"].mean(),
                "capacity_end": df["capacity_units"].iloc[-1],
                "trust_end": df["trust"].iloc[-1],
            })
    raw = pd.DataFrame(rows)
    agg = raw.groupby("scenario").mean(numeric_only=True).drop(
        columns=["rep"]).reset_index()

    # cost effectiveness: avoided loss per unit of public money
    base = agg.loc[agg.scenario == "baseline"].iloc[0]
    agg["avoided_vs_baseline"] = agg["avoided_loss"] - base["avoided_loss"]
    agg["extra_fiscal_cost"] = agg["fiscal_cost"] - base["fiscal_cost"]
    agg["avoided_per_1k_spent"] = np.where(
        agg["extra_fiscal_cost"] > 1.0,
        agg["avoided_vs_baseline"] / (agg["extra_fiscal_cost"] / 1000.0),
        np.nan)
    return {"raw": raw, "agg": agg, "series": pd.concat(series,
                                                        ignore_index=True)}


def pareto_front(df: pd.DataFrame, objectives: dict) -> pd.DataFrame:
    """Non-dominated set. objectives: {column: 'max'|'min'}."""
    cols = list(objectives)
    sign = np.array([1.0 if objectives[c] == "max" else -1.0 for c in cols])
    vals = df[cols].to_numpy(dtype=float) * sign
    keep = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        if not keep[i]:
            continue
        dominated = np.all(vals >= vals[i], axis=1) & np.any(vals > vals[i], axis=1)
        if dominated.any():
            keep[i] = False
    out = df.loc[keep].copy()
    out["pareto"] = True
    return out


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartagri.experiments")
    ap.add_argument("--quick", action="store_true",
                    help="small, fast configuration")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--n-farmers", type=int, default=None)
    ap.add_argument("--seasons", type=int, default=None)
    ap.add_argument("--reps", type=int, default=None)
    ap.add_argument("--grid", type=int, default=None)
    args = ap.parse_args(argv)

    if args.full:
        nf, seasons, reps, grid = 500, 12, 5, 25
    else:
        nf, seasons, reps, grid = 250, 8, 3, 17
    nf = args.n_farmers or nf
    seasons = args.seasons or seasons
    reps = args.reps or reps
    grid = args.grid or grid

    out = _ensure_out()
    t0 = time.time()
    print("=" * 74)
    print(f"EXPERIMENT SUITE  (n_farmers={nf}, seasons={seasons}, "
          f"reps={reps}, contract grid={grid})")
    print("=" * 74)

    # ---------------- RQ1 ----------------
    print("\n[RQ1] adoption and effort ...")
    r1 = rq1_adoption_and_effort(nf, seasons)
    r1["farmers"].to_csv(out / "rq1_farmers.csv", index=False)
    r1["by_risk"].to_csv(out / "rq1_by_risk_quartile.csv", index=False)
    r1["providers"].to_csv(out / "rq1_providers.csv", index=False)
    print(r1["by_risk"].round(3).to_string(index=False))

    print("\n[RQ1] contract map over the (c1, c2) plane ...")
    cmap = rq1_contract_map(grid)
    cmap.to_csv(out / "rq1_contract_map.csv", index=False)
    summ = cmap.groupby("curvature").agg(
        n=("e_star", "size"), mean_effort=("e_star", "mean"),
        share_interior=("interior", "mean"),
        mean_payment=("expected_payment", "mean")).reset_index()
    print(summ.round(3).to_string(index=False))

    print("\n[RQ1] verifiability experiment ...")
    vexp = rq1_verifiability_experiment(nf, seasons)
    vexp.to_csv(out / "rq1_verifiability.csv", index=False)
    print(vexp.round(4).to_string(index=False))

    # ---------------- RQ2 ----------------
    print("\n[RQ2] resilience under controlled shocks ...")
    r2 = rq2_resilience(nf)
    r2["summary"].to_csv(out / "rq2_resilience.csv", index=False)
    cols2 = ["design", "income_drop_pct", "recovery_seasons",
             "loss_shock_season", "causal_mitigation_rate",
             "naive_adopter_effect", "selection_inflation",
             "loss_adopters", "loss_non_adopters", "cumulative_loss_area"]
    print(r2["summary"][cols2].round(3).to_string(index=False))

    # ---------------- RQ3 ----------------
    print("\n[RQ3] policy scenarios ...")
    r3 = rq3_policies(nf, seasons, reps)
    r3["agg"].to_csv(out / "rq3_scenarios.csv", index=False)
    r3["raw"].to_csv(out / "rq3_scenarios_raw.csv", index=False)
    r3["series"].to_csv(out / "rq3_series.csv", index=False)
    show = ["scenario", "mitigation_rate", "avoided_loss", "effective_use_rate",
            "mean_wait_days", "peak_wait_days", "income_p10", "exit_rate",
            "fiscal_cost", "equity_gap", "mountain_gap"]
    print(r3["agg"][show].round(4).to_string(index=False))

    pf = pareto_front(r3["agg"], {
        "mitigation_rate": "max", "fiscal_cost": "min",
        "equity_gap": "min", "effective_use_rate": "max"})
    pf.to_csv(out / "rq3_pareto.csv", index=False)
    print(f"\nPareto-non-dominated scenarios: {sorted(pf['scenario'])}")

    meta = {"n_farmers": nf, "seasons": seasons, "reps": reps,
            "contract_grid": grid, "runtime_s": round(time.time() - t0, 1)}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nOutputs -> {out}")
    print(f"Total runtime: {meta['runtime_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
