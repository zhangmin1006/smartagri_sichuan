"""
audit.py
========
Model and data audit utilities.

Five independent checks, each of which can fail and each of which reports
evidence rather than an opinion:

    A1  config coverage      -- which declared parameters are never read
    A2  hard-coded values    -- magic numbers living in code, not config
    A3  sensitivity          -- which parameters actually move the outputs
    A4  Monte Carlo stability-- how many replicates a conclusion needs
    A5  extreme conditions   -- does the model behave sanely at the limits

Run:
    python -m smartagri.audit --all
    python -m smartagri.audit --sensitivity --n-farmers 150
"""

from __future__ import annotations

import argparse
import ast
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .model import SmartAgriModel

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config"
SRC = ROOT / "src"
OUT = ROOT / "outputs" / "audit"


def _out() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    return OUT


# ===========================================================================
# A1  Config coverage: declared but never read
# ===========================================================================
def _walk_keys(node, prefix=""):
    """Yield dotted paths of every leaf key in a nested YAML structure."""
    if isinstance(node, dict):
        for k, v in node.items():
            path = f"{prefix}.{k}" if prefix else str(k)
            yield path, v
            yield from _walk_keys(v, path)
    elif isinstance(node, list):
        for item in node:
            yield from _walk_keys(item, prefix)


def audit_config_coverage() -> pd.DataFrame:
    """Find configuration keys that no source file ever reads.

    A declared-but-unread parameter is worse than a missing one: it looks
    like a modelling choice, invites reviewers to argue about its value, and
    changes nothing when changed.
    """
    code = "\n".join(p.read_text(encoding="utf-8")
                     for p in SRC.rglob("*.py") if p.name != "audit.py")

    rows = []
    for cfg_file in sorted(CONFIG.glob("*.yaml")):
        data = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
        seen = set()
        for path, val in _walk_keys(data):
            leaf = path.split(".")[-1]
            if leaf in seen or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", leaf):
                continue
            seen.add(leaf)
            # documentation-only keys are exempt by design
            if leaf in {"note", "notes", "rationale", "why_included",
                        "why_deferred", "reason", "role_in_model", "caution",
                        "abm_mechanism", "resilience_mechanism", "components",
                        "data_needed", "verify_queries", "sichuan_anchor",
                        "known_side_effects", "spatial_signature", "meta",
                        "state_warning", "interface_contract", "definition",
                        "label_en", "title_en", "title_zh", "name_en",
                        "name_zh", "issuer", "reference", "binding_on_model",
                        "key_target", "decision_variables", "objectives",
                        "optimisation", "status", "equity_flag", "evidence_grade",
                        "primary_loss_channel", "channel", "method"}:
                continue
            used = bool(re.search(rf'["\']{re.escape(leaf)}["\']', code))
            rows.append({"file": cfg_file.name, "key": leaf, "path": path,
                         "read_by_code": used,
                         "value": str(val)[:60] if not isinstance(
                             val, (dict, list)) else ""})
    df = pd.DataFrame(rows)
    return df.sort_values(["read_by_code", "file", "key"])


# ===========================================================================
# A2  Hard-coded numeric literals in decision code
# ===========================================================================
def audit_hardcoded() -> pd.DataFrame:
    """Flag numeric literals in behavioural code that are not in config.

    Not every literal is a defect -- loop bounds, tolerances and array
    indices are fine. The audit reports them so a reviewer can decide, and
    deliberately errs towards over-reporting.
    """
    targets = ["agents.py", "model.py", "shocks.py", "sd.py"]
    benign = {0, 1, 2, 3, 4, 5, 10, 12, 100, -1}
    rows = []
    for name in targets:
        path = SRC / "smartagri" / name
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        lines = src.splitlines()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant):
                continue
            if not isinstance(node.value, (int, float)) or isinstance(
                    node.value, bool):
                continue
            v = node.value
            if v in benign or abs(v) > 1e6:
                continue
            line = lines[node.lineno - 1].strip()
            if any(tok in line for tok in ("range(", "np.linspace", "round(",
                                           "maxsize", "seed", "1e-", "1e6",
                                           "percentile", "size=", "[:", "n_quad",
                                           "__", "figsize")):
                continue
            rows.append({"file": name, "line": node.lineno, "value": v,
                         "context": line[:96]})
    return pd.DataFrame(rows)


# ===========================================================================
# A3  Sensitivity analysis
# ===========================================================================
SENSITIVITY_SPEC = [
    # (label, config path, low, base, high, module)
    ("eta T2 drought", ("tech", "T2", "eta_drought"), 0.25, 0.45, 0.65, "technology"),
    ("eta T3 flood", ("tech", "T3", "eta_flood"), 0.20, 0.35, 0.50, "technology"),
    ("eta T1 flood", ("tech", "T1", "eta_flood"), 0.15, 0.28, 0.45, "technology"),
    ("T2 capex per mu", ("tech", "T2", "capex_per_mu"), 250.0, 420.0, 650.0, "technology"),
    ("T3 service price", ("tech", "T3", "service_price_per_mu"), 7.0, 12.0, 20.0, "technology"),
    ("gamma provider", ("par", "contract", "gamma_provider"), 0.05, 0.12, 0.25, "contract"),
    ("U_min provider", ("par", "contract", "u_min_provider_base"), 0.15, 0.28, 0.45, "contract"),
    ("shirk multiplier", ("par", "contract", "verifiability", "shirk_effort_multiplier"), 0.30, 0.55, 0.80, "contract"),
    ("verifiability T3", ("par", "contract", "verifiability", "with_T3_telemetry"), 0.50, 0.80, 0.95, "contract"),
    ("logit scale tau", ("par", "behaviour", "softmax_temperature"), 0.01, 0.02, 0.04, "behaviour"),
    ("social learning", ("par", "behaviour", "social_learning_weight"), 0.10, 0.30, 0.55, "behaviour"),
    ("capacity coverage", ("par", "sd", "initial_capacity_coverage"), 0.35, 0.55, 0.85, "sd"),
    ("max mitigation", ("par", "loss", "max_total_mitigation"), 0.45, 0.65, 0.85, "loss"),
    ("farmer risk mean", ("risk", "farmer"), 0.45, 0.60, 0.75, "risk"),
    ("provider risk mean", ("risk", "provider"), 0.20, 0.33, 0.50, "risk"),
    # Parameters added during the Aug-2026 debugging rounds. Every one of
    # these is a REASONED PRIOR rather than a measurement, and several now
    # drive headline results, so they must be sweepable -- a parameter that
    # moves the conclusion but cannot be varied is the blind spot this
    # section exists to prevent. Ranges are deliberately wide (roughly +/-50
    # per cent) because the uncertainty on them is genuine.
    ("routine benefit T3", ("tech", "T3", "routine_benefit_per_mu"), 11.0, 22.0, 33.0, "technology"),
    ("routine benefit T2", ("tech", "T2", "routine_benefit_per_mu"), 15.0, 30.0, 45.0, "technology"),
    ("non-power outage share", ("par", "loss", "non_power_outage_share"), 0.20, 0.40, 0.70, "loss"),
    ("learning decay", ("par", "behaviour", "frictions", "learning_decay"), 0.5, 1.0, 2.0, "behaviour"),
    ("hazard learning rate", ("par", "behaviour", "belief_update", "learning_rate_hazard"), 0.05, 0.15, 0.35, "behaviour"),
    ("stress decay", ("par", "risk_attitude", "farmer", "stress_shift", "stress_decay_per_season"), 0.15, 0.35, 0.60, "risk"),
]

OUTPUTS = ["adopt_any", "effort_T3", "effective_use_rate",
           "mitigation_rate", "mean_wait_days", "mean_income"]


def _apply(model: SmartAgriModel, path: tuple, value: float) -> None:
    """Mutate a loaded model's configuration in place.

    IMPORTANT: some quantities are consumed once, in __init__, and never read
    again from the configuration -- service capacity and the provider
    reservation utility are both of this kind. Writing only to cfg leaves them
    untouched and the sensitivity analysis silently reports zero effect for a
    parameter that in fact matters. Those cases are therefore re-applied to
    the constructed objects here.
    """
    kind = path[0]
    if kind == "tech":
        model.tech_spec[path[1]]["model"][path[2]] = value
    elif kind == "par":
        node = model.cfg["params"]
        if path[1] == "sd":
            node["sd"]["stocks"]["service_capacity"][path[2]] = value
            if path[2] == "initial_capacity_coverage":
                per_unit = (model.sd.unit_capacity_mu_per_day * 12.0
                            * model.sd.utilisation_ceiling)
                model.sd.capacity_units = max(
                    1e-3, value * model.total_area_mu / max(per_unit, 1e-9))
                model._sync_provider_capacity()
            return
        for k in path[1:-1]:
            node = node[k]
        node[path[-1]] = value
        if path[-1] == "u_min_provider_base":
            for pr in model.providers:
                pr.u_min_base = float(value)
        # Parameters cached on the instance at construction are invisible to
        # a cfg write. The model centralises those reads, so refresh them all
        # rather than enumerating them here -- three priors silently reported
        # zero sensitivity before this call existed.
        model.refresh_cached_params()
    elif kind == "risk":
        # shift the support midpoint while preserving spread
        ra = model.cfg["params"]["risk_attitude"][path[1]]
        lo, hi = ra["support"]
        half = (hi - lo) / 2.0
        ra["support"] = [max(0.05, value - half), min(0.95, value + half)]
        model.refresh_cached_params()


def _run_once(overrides, n_farmers, seasons, seed, shock_design):
    m = SmartAgriModel("baseline", seed=seed, n_farmers=n_farmers,
                       seasons=seasons)
    for path, value in overrides:
        _apply(m, path, value)
    # population must be rebuilt if a risk distribution changed
    if any(p[0] == "risk" for p, _ in overrides):
        m.rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[0])
        m._build_population(n_farmers)
        m.total_area_mu = sum(f.area_mu for f in m.farmers)
        m._sync_provider_capacity()
        m._apply_mountain_adaptation()
    m.run(forced=shock_design)
    df = m.to_dataframe()
    adopt = np.mean([df[f"adopt_{t}"].mean() for t in ("T1", "T2", "T3")])
    return {
        "adopt_any": adopt,
        "effort_T3": df["effort_T3"].mean(),
        "effective_use_rate": df["effective_use_rate"].mean(),
        "mitigation_rate": df["mitigation_rate"].mean(),
        "mean_wait_days": df["mean_wait_days"].mean(),
        "mean_income": df["mean_income"].mean(),
    }


def _shock_design(seasons, seed=99):
    rng = np.random.default_rng(seed)
    d = {}
    for s in range(1, seasons + 1):
        r = rng.random()
        d[s] = (["D1"] if r < 0.25 else ["D2"] if r < 0.53
                else ["D3"] if r < 0.62 else [])
    return d


def audit_sensitivity(n_farmers=150, seasons=6, reps=2) -> pd.DataFrame:
    """One-at-a-time elasticities on the parameters that carry the argument."""
    design = _shock_design(seasons)
    seeds = [41 + i for i in range(reps)]

    base = pd.DataFrame([_run_once([], n_farmers, seasons, s, design)
                         for s in seeds]).mean()

    rows = []
    for label, path, lo, mid, hi, module in SENSITIVITY_SPEC:
        for tag, val in (("low", lo), ("high", hi)):
            res = pd.DataFrame([_run_once([(path, val)], n_farmers, seasons,
                                          s, design) for s in seeds]).mean()
            for out in OUTPUTS:
                b, v = base[out], res[out]
                rel_par = (val - mid) / mid if mid else np.nan
                rel_out = (v - b) / abs(b) if abs(b) > 1e-9 else np.nan
                rows.append({
                    "parameter": label, "module": module, "direction": tag,
                    "param_base": mid, "param_value": val,
                    "output": out, "base": b, "value": v,
                    "pct_change_out": 100 * rel_out,
                    "elasticity": (rel_out / rel_par
                                   if rel_par and abs(rel_par) > 1e-9 else np.nan),
                })
    return pd.DataFrame(rows)


def summarise_sensitivity(df: pd.DataFrame) -> pd.DataFrame:
    """Rank parameters by the largest absolute effect they have on any output."""
    g = (df.groupby(["parameter", "module", "output"])["pct_change_out"]
           .apply(lambda s: np.nanmax(np.abs(s))).reset_index())
    top = (g.sort_values("pct_change_out", ascending=False)
             .groupby("parameter").head(1))
    wide = g.pivot_table(index=["parameter", "module"], columns="output",
                         values="pct_change_out").reset_index()
    wide["max_effect_pct"] = wide[OUTPUTS].max(axis=1)
    wide["drives"] = top.set_index("parameter").loc[
        wide["parameter"], "output"].values
    return wide.sort_values("max_effect_pct", ascending=False)


# ===========================================================================
# A4  Monte Carlo stability
# ===========================================================================
def audit_stability(n_farmers=150, seasons=6, max_reps=8) -> pd.DataFrame:
    """How many replicates before a scenario ranking stops changing?

    Reports the standard error of the mean and the running rank of three
    scenarios as replicates accumulate. If the ranking is still flipping at
    the replicate count used to produce headline results, those results are
    not yet conclusions.
    """
    scenarios = ["baseline", "voucher", "voucher_plus_capacity"]
    recs = []
    for rep in range(max_reps):
        for sc in scenarios:
            m = SmartAgriModel(sc, seed=200 + rep, n_farmers=n_farmers,
                               seasons=seasons)
            # Separate shock RNG streams make the weather path identical
            # across scenarios inside a replicate, while allowing it to vary
            # across replicates.  The earlier fixed design did not measure
            # shock-path uncertainty at all.
            m.run()
            df = m.to_dataframe()
            recs.append({"rep": rep, "scenario": sc,
                         "mitigation_rate": df["mitigation_rate"].mean(),
                         "mean_wait_days": df["mean_wait_days"].mean(),
                         "effective_use_rate": df["effective_use_rate"].mean()})
    raw = pd.DataFrame(recs)

    rows = []
    for k in range(1, max_reps + 1):
        sub = raw[raw.rep < k]
        agg = sub.groupby("scenario").agg(
            mitig_mean=("mitigation_rate", "mean"),
            mitig_sd=("mitigation_rate", "std"),
            wait_mean=("mean_wait_days", "mean")).reset_index()
        agg["sem"] = agg["mitig_sd"] / np.sqrt(k)
        order = tuple(agg.sort_values("mitig_mean", ascending=False)["scenario"])
        best = agg.loc[agg.mitig_mean.idxmax()]
        rows.append({
            "n_reps": k,
            "ranking_by_mitigation": " > ".join(s[:14] for s in order),
            "top_scenario": best["scenario"],
            "top_mitigation": best["mitig_mean"],
            "sem_top": best["sem"] if k > 1 else np.nan,
            "cv_pct": (100 * best["sem"] / best["mitig_mean"]
                       if k > 1 and best["mitig_mean"] else np.nan),
        })
    return pd.DataFrame(rows), raw


# ===========================================================================
# A5  Extreme-condition tests
# ===========================================================================
def audit_extreme(n_farmers=120, seasons=5) -> pd.DataFrame:
    """Does the model behave sanely when pushed to its limits?"""
    calm = {s: [] for s in range(1, seasons + 1)}
    allshock = {s: ["D3"] for s in range(1, seasons + 1)}
    tests = []

    def run(label, expect, scenario="baseline", design=None, mutate=None,
            seed=7):
        m = SmartAgriModel(scenario, seed=seed, n_farmers=n_farmers,
                           seasons=seasons)
        if mutate:
            mutate(m)
        m.run(forced=design if design is not None else calm)
        df = m.to_dataframe()
        tests.append({
            "test": label, "expected": expect,
            "adopt_T3": round(df["adopt_T3"].mean(), 4),
            "effort_T3": round(df["effort_T3"].mean(), 4),
            "mitigation": round(df["mitigation_rate"].mean(), 4),
            "wait_days": round(df["mean_wait_days"].mean(), 3),
            "loss": round(df["mean_loss_fraction"].mean(), 4),
            "income": round(df["mean_income"].mean(), 1),
            "exit_rate": round(df["exit_rate"].iloc[-1], 4),
        })

    run("No shocks at all", "zero loss, zero mitigation", design=calm)
    run("Every season D3", "high loss, low mitigation", design=allshock)

    def zero_eff(m):
        for t in ("T1", "T2", "T3"):
            for k in list(m.tech_spec[t]["model"]):
                if k.startswith("eta_"):
                    m.tech_spec[t]["model"][k] = 0.0
    # NB the expectation changed when the routine benefit was added. Zeroing
    # shock efficacy no longer collapses adoption, and should not: a drone
    # that saves a labour day is still worth hiring in a season with no
    # disaster. What must go to zero is MITIGATION. Adoption falling only
    # partway (about 30 per cent below baseline) is the correct behaviour,
    # not a failure.
    run("Technology efficacy = 0", "mitigation = 0; adoption falls but persists",
        design=allshock, mutate=zero_eff)

    def free_tech(m):
        m.tech_spec["T2"]["model"]["capex_per_mu"] = 0.0
        m.tech_spec["T3"]["model"]["service_price_per_mu"] = 0.0
        m.tech_spec["T2"]["model"]["opex_per_mu"] = 0.0
    run("Technology is free", "adoption near ceiling, queue binds",
        design=allshock, mutate=free_tech)

    def huge_capacity(m):
        m.sd.capacity_units *= 1000
    run("Unlimited service capacity", "wait -> minimum", design=allshock,
        mutate=huge_capacity)

    def no_capacity(m):
        m.sd.capacity_units = 1e-3
    run("No service capacity", "wait at ceiling, service adoption unusable",
        design=allshock, mutate=no_capacity)

    def no_verify(m):
        v = m.cfg["params"]["contract"]["verifiability"]
        for k in ("with_T1_remote_sensing", "with_T2_sensors",
                  "with_T3_telemetry"):
            v[k] = 0.0
    run("No monitoring at all", "effort falls to the shirking floor",
        design=allshock, mutate=no_verify)

    def broke(m):
        m.sd.budget = 0.0
        m.sd.budget_inflow_annual = 0.0
    run("Zero budget with subsidy scenario", "instrument rations to zero",
        scenario="subsidy", design=allshock, mutate=broke)

    return pd.DataFrame(tests)


# ===========================================================================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartagri.audit")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--config", action="store_true")
    ap.add_argument("--hardcoded", action="store_true")
    ap.add_argument("--sensitivity", action="store_true")
    ap.add_argument("--stability", action="store_true")
    ap.add_argument("--extreme", action="store_true")
    ap.add_argument("--n-farmers", type=int, default=150)
    ap.add_argument("--seasons", type=int, default=6)
    ap.add_argument("--reps", type=int, default=2)
    a = ap.parse_args(argv)
    if a.all:
        a.config = a.hardcoded = a.sensitivity = a.stability = a.extreme = True

    out = _out()
    t0 = time.time()
    print("=" * 76)
    print("MODEL AND DATA AUDIT")
    print("=" * 76)

    if a.config:
        print("\n[A1] configuration coverage")
        df = audit_config_coverage()
        df.to_csv(out / "a1_config_coverage.csv", index=False)
        unread = df[~df.read_by_code]
        print(f"  declared keys checked : {len(df)}")
        print(f"  never read by code    : {len(unread)}")
        for _, r in unread.head(25).iterrows():
            print(f"    - {r['file']:22s} {r['key']}")

    if a.hardcoded:
        print("\n[A2] hard-coded numeric literals in behavioural code")
        df = audit_hardcoded()
        df.to_csv(out / "a2_hardcoded.csv", index=False)
        print(f"  literals flagged: {len(df)}")
        print(df.groupby("file").size().to_string())

    if a.sensitivity:
        print("\n[A3] sensitivity analysis (this takes a few minutes)")
        df = audit_sensitivity(a.n_farmers, a.seasons, a.reps)
        df.to_csv(out / "a3_sensitivity_raw.csv", index=False)
        s = summarise_sensitivity(df)
        s.to_csv(out / "a3_sensitivity_summary.csv", index=False)
        cols = ["parameter", "module", "max_effect_pct", "drives"]
        print(s[cols].round(2).to_string(index=False))

    if a.stability:
        print("\n[A4] Monte Carlo stability")
        df, raw = audit_stability(a.n_farmers, a.seasons)
        df.to_csv(out / "a4_stability.csv", index=False)
        raw.to_csv(out / "a4_stability_raw.csv", index=False)
        print(df.round(4).to_string(index=False))

    if a.extreme:
        print("\n[A5] extreme-condition tests")
        df = audit_extreme(a.n_farmers, a.seasons)
        df.to_csv(out / "a5_extreme.csv", index=False)
        print(df.to_string(index=False))

    print(f"\nOutputs -> {out}")
    print(f"Runtime: {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
