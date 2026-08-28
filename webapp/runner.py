# -*- coding: utf-8 -*-
"""
runner.py
=========
Turns a scenario posted from the browser into model runs, and the model runs
into the summary the interface plots.

Two design decisions matter here.

**Overrides are applied by writing a patched config directory, not by mutating
the model object.** ``SmartAgriModel`` already accepts ``config_dir``, and so
does ``ShockGenerator``, which re-reads ``disruptions.yaml`` for itself. Mutating
``model.cfg`` after construction would miss the shock generator entirely and
would also miss every value ``__init__`` caches onto the instance -- the audit
log records that exact bug producing a silent ZERO effect for three parameters
in the sensitivity harness. Writing YAML costs a few milliseconds and cannot
desynchronise.

**Every policy is scored against a baseline run on the same seed.** The model
spawns independent RNG streams for behaviour, weather and outcomes precisely so
that a policy comparison is paired: the two runs see the SAME weather, so the
difference between them is the policy and not the draw. Reporting a policy
number on its own would be uninterpretable, because a single run's outcome is
dominated by whether a shock happened to land in it.
"""

from __future__ import annotations

import copy
import math
import shutil
import statistics
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"

# Season-level series sent to the browser for plotting.
SERIES_KEYS = [
    "adopt_T1", "adopt_T2", "adopt_T3", "effective_use_rate", "mitigation_rate",
    "mean_loss_fraction", "mean_wait_days", "backlog_mu", "mean_income",
    "income_p10", "gini_income", "exit_rate", "fiscal_cumulative",
    "fiscal_spend", "equity_gap", "mountain_gap", "capacity_units",
    "reliability", "trust", "capable_share",
]

# Scalar outcomes summarising a whole run.
SUMMARY_KEYS = [
    "mitigation_rate", "effective_use_rate", "mean_wait_days",
    "fiscal_cumulative", "equity_gap", "mountain_gap", "exit_rate",
    "mean_income", "mean_loss_fraction", "backlog_mu", "adopt_T1",
    "adopt_T2", "adopt_T3", "income_p10", "gini_income", "capacity_units",
]

# Summaries that are a STATE at the end of the run rather than an average over
# it. Cumulative spend averaged across seasons would understate the bill by
# roughly half, and capacity is a stock, not a flow.
_TERMINAL = {"fiscal_cumulative", "capacity_units"}


def _load_configs() -> dict:
    out = {}
    for key, name in (("params", "model_params.yaml"),
                      ("tech", "technologies.yaml"),
                      ("disrupt", "disruptions.yaml")):
        with (CONFIG_DIR / name).open(encoding="utf-8") as fh:
            out[key] = yaml.safe_load(fh)
    return out


def _set_path(node, path, value):
    for p in path[:-1]:
        node = node[p]
    node[path[-1]] = value


def build_config_dir(overrides: dict, workdir: Path) -> Path:
    """Write a config directory carrying the user's parameter overrides.

    `policies.yaml` is copied unchanged: policy settings reach the model
    through the `instruments` argument, not through the config file.
    """
    cfg = _load_configs()
    params, tech, disrupt = cfg["params"], cfg["tech"], cfg["disrupt"]

    # -- population ------------------------------------------------------
    pop_over = overrides.get("population") or {}
    for key in ("n_farmers", "n_providers"):
        if key in pop_over:
            params["population"][key] = int(pop_over[key])
    if "seasons" in pop_over:
        params["meta"]["seasons"] = int(pop_over["seasons"])

    counties = {c["id"]: c for c in params["population"]["counties"]}
    for cid, fields in (overrides.get("counties") or {}).items():
        if cid not in counties:
            continue
        for f, v in fields.items():
            if f in counties[cid]:
                counties[cid][f] = int(v) if f == "n_farmers" else float(v)

    # The county table and the headline population count are two statements of
    # the same quantity. If the user edits county sizes, the headline must
    # follow them or the model silently rescales the population it was given.
    if overrides.get("counties"):
        params["population"]["n_farmers"] = sum(
            int(c["n_farmers"]) for c in params["population"]["counties"])

    # -- risk attitude and behaviour -------------------------------------
    RISK_PATHS = {
        "farmer_alpha": ["risk_attitude", "farmer", "params", 0],
        "farmer_beta": ["risk_attitude", "farmer", "params", 1],
        "provider_alpha": ["risk_attitude", "provider", "params", 0],
        "provider_beta": ["risk_attitude", "provider", "params", 1],
    }
    BEHAV_PATHS = {
        "base_verifiability": ["contract", "verifiability", "base_verifiability"],
        "shirk_effort_multiplier": ["contract", "verifiability",
                                    "shirk_effort_multiplier"],
        "gamma_provider": ["contract", "gamma_provider"],
        "u_min_provider_base": ["contract", "u_min_provider_base"],
        "social_learning_weight": ["behaviour", "social_learning_weight"],
        "base_yield_value_per_mu": ["production", "base_yield_value_per_mu"],
    }
    for src, paths in ((overrides.get("risk") or {}, RISK_PATHS),
                       (overrides.get("behaviour") or {}, BEHAV_PATHS)):
        for k, v in src.items():
            if k in paths:
                _set_path(params, paths[k], float(v))

    # -- technologies ----------------------------------------------------
    bundles = {b["id"]: b for b in tech["bundles"]}
    for tid, fields in (overrides.get("technologies") or {}).items():
        if tid not in bundles:
            continue
        for f, v in fields.items():
            if f in bundles[tid]["model"]:
                bundles[tid]["model"][f] = float(v)

    # -- shocks ----------------------------------------------------------
    specs = {d["id"]: d for d in disrupt["tier1"]}
    for did, fields in (overrides.get("shocks") or {}).items():
        if did not in specs:
            continue
        model = specs[did]["model"]
        for f, v in fields.items():
            if f not in model:
                continue
            model[f] = float(v)
            if f == "annual_probability_prior":
                # The per-county probabilities are the calibrated object the
                # shock generator actually samples from; the provincial prior
                # is their summary. Rescaling the county field preserves the
                # measured geographic footprint -- which differs sharply
                # between hazards and drives how much simultaneous demand the
                # queue absorbs -- while honouring the user's chosen level.
                _rescale_county_probs(model, float(v))

    workdir.mkdir(parents=True, exist_ok=True)
    for name, data in (("model_params.yaml", params),
                       ("technologies.yaml", tech),
                       ("disruptions.yaml", disrupt)):
        with (workdir / name).open("w", encoding="utf-8") as fh:
            yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)
    shutil.copy(CONFIG_DIR / "policies.yaml", workdir / "policies.yaml")
    return workdir


def _rescale_county_probs(model: dict, new_annual: float) -> None:
    probs = model.get("probability_by_county")
    if not probs:
        return
    # Scale relative to the county field's own mean, not to the provincial
    # prior: the two agree to within a few thousandths in the calibrated
    # config, and using the field's mean keeps the rescaling exact even if
    # they are ever edited apart.
    vals = [float(v) for v in probs.values()]
    old = sum(vals) / len(vals) if vals else 0.0
    if old <= 1e-9:
        # A hazard calibrated to never occur has no footprint to preserve, so
        # a uniform field is the only defensible reading of "raise it to p".
        model["probability_by_county"] = {k: new_annual for k in probs}
        return
    factor = new_annual / old
    model["probability_by_county"] = {
        k: float(min(1.0, max(0.0, float(v) * factor))) for k, v in probs.items()}


# ---------------------------------------------------------------------------
def _summarise(df) -> dict:
    out = {}
    for k in SUMMARY_KEYS:
        if k not in df.columns:
            continue
        col = df[k].astype(float)
        out[k] = float(col.iloc[-1]) if k in _TERMINAL else float(col.mean())
    rec = df.get("recovery_seasons_mean")
    if rec is not None:
        valid = [float(v) for v in rec if v == v]     # drop NaN
        out["recovery_seasons_mean"] = (
            float(sum(valid) / len(valid)) if valid else float("nan"))
    return out


def _series(df) -> dict:
    out = {"season": [int(s) for s in df["season"]],
           "shocks": [str(s) for s in df["shocks"]]}
    for k in SERIES_KEYS:
        if k in df.columns:
            out[k] = [_clean(v) for v in df[k].astype(float)]
    return out


def _clean(v) -> float | None:
    v = float(v)
    return None if (math.isnan(v) or math.isinf(v)) else v


def run_scenario(config_dir: Path, instruments: dict, seed: int,
                 replicates: int, forced: dict | None,
                 progress=None, tag: str = "") -> dict:
    """Run one scenario `replicates` times and aggregate across replicates."""
    from smartagri.model import SmartAgriModel

    summaries, series_all = [], []
    for r in range(replicates):
        m = SmartAgriModel(scenario="custom", seed=seed + r,
                           config_dir=config_dir,
                           instruments=copy.deepcopy(instruments) or {})
        m.run(forced=forced or None)
        df = m.to_dataframe()
        summaries.append(_summarise(df))
        series_all.append(_series(df))
        if progress:
            progress(tag, r + 1, replicates)

    keys = sorted({k for s in summaries for k in s})
    agg = {}
    for k in keys:
        vals = [s[k] for s in summaries if k in s and s[k] == s[k]]
        if not vals:
            agg[k] = {"mean": None, "sd": None}
            continue
        agg[k] = {"mean": float(statistics.fmean(vals)),
                  "sd": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0}

    # Season series are averaged across replicates so the plotted line is the
    # expected path rather than one arbitrary draw.
    base = series_all[0]
    mean_series = {"season": base["season"],
                   "shocks": base["shocks"] if replicates == 1 else None}
    for k in SERIES_KEYS:
        if k not in base:
            continue
        cols = [s[k] for s in series_all]
        mean_series[k] = [
            _mean_ignoring_none([c[i] for c in cols])
            for i in range(len(base["season"]))]
    return {"summary": agg, "series": mean_series, "replicates": replicates}


def _mean_ignoring_none(vals):
    vs = [v for v in vals if v is not None]
    return float(sum(vs) / len(vs)) if vs else None


def compare(policy: dict, baseline: dict) -> dict:
    """Policy minus baseline, per outcome, with a crude signal-to-noise flag."""
    out = {}
    for k, pv in policy["summary"].items():
        bv = baseline["summary"].get(k)
        if not bv or pv["mean"] is None or bv["mean"] is None:
            continue
        diff = pv["mean"] - bv["mean"]
        rel = (diff / abs(bv["mean"])) if abs(bv["mean"]) > 1e-12 else None
        # Replicate spread is the only noise estimate available here. A
        # difference smaller than the combined spread is not distinguishable
        # from the draw and is flagged so the interface can say so rather than
        # invite the user to read a rounding artefact as a policy effect.
        noise = math.hypot(pv.get("sd") or 0.0, bv.get("sd") or 0.0)
        out[k] = {"policy": pv["mean"], "baseline": bv["mean"], "diff": diff,
                  "rel": rel, "noise": noise,
                  "distinguishable": bool(abs(diff) > noise) if noise > 0 else None}
    return out


def temp_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix="smartagri_cfg_"))
