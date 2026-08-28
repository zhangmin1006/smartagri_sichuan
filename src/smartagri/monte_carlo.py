"""Robust Monte Carlo experiments for policy comparisons.

This module separates two uncertainty questions:

1. *Aleatory/stochastic stability*: repeat the same parameterisation over
   paired seeds and stop only when policy-vs-baseline confidence intervals
   reach declared precision targets.
2. *Epistemic/parameter uncertainty*: draw influential parameters with a
   Latin-hypercube design, then repeat paired stochastic runs inside each
   parameter draw.

All scenarios in a replicate use the same seed.  ``SmartAgriModel`` maintains
independent behaviour, shock and outcome random streams, so a policy cannot
change its own weather merely by consuming a different number of behavioural
random draws.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from . import __version__
from .audit import SENSITIVITY_SPEC, _apply
from .model import CONFIG_DIR, SmartAgriModel


OUT = Path(__file__).resolve().parents[2] / "outputs" / "monte_carlo"

METRIC_DIRECTION = {
    "mitigation_rate": "max",
    "effective_use_rate": "max",
    "mean_wait_days": "min",
    "income_p10": "max",
    "equity_gap": "min",
    "mountain_gap": "min",
    "fiscal_cumulative": "min",
}

# Absolute Monte Carlo half-widths at which a near-zero contrast is precise
# enough to be called operationally negligible.  These are reporting
# thresholds, not empirical validation targets, and are written to the run
# manifest so a decision maker can replace them.
ABSOLUTE_TOLERANCE = {
    "mitigation_rate": 0.0010,
    "effective_use_rate": 0.0100,
    "mean_wait_days": 0.25,
    "income_p10": 100.0,
    "equity_gap": 0.0020,
    "mountain_gap": 0.0020,
    "fiscal_cumulative": 1000.0,
}

# Parameters for the epistemic envelope.  Uniform draws between audit bounds
# are deliberately labelled an envelope, not a posterior, until field data
# support defensible distributions.
EPISTEMIC_LABELS = {
    "capacity coverage", "logit scale tau", "gamma provider",
    "provider risk mean", "farmer risk mean", "U_min provider",
    "eta T3 flood", "eta T2 drought", "eta T1 flood",
    "T3 service price", "verifiability T3", "social learning",
}


@dataclass(frozen=True)
class MonteCarloConfig:
    n_farmers: int = 150
    seasons: int = 6
    min_reps: int = 30
    max_reps: int = 100
    batch_size: int = 10
    confidence: float = 0.95
    relative_half_width: float = 0.10
    seed0: int = 81001


def _config_hash() -> str:
    h = hashlib.sha256()
    for name in ("model_params.yaml", "technologies.yaml", "policies.yaml",
                 "disruptions.yaml"):
        h.update((CONFIG_DIR / name).read_bytes())
    return h.hexdigest()[:16]


def _model_hash() -> str:
    """Fingerprint model source so result tables cannot outlive code changes."""
    h = hashlib.sha256()
    source_dir = Path(__file__).resolve().parent
    for path in sorted(source_dir.glob("*.py")):
        h.update(path.name.encode("utf-8"))
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _scenario_keys() -> list[str]:
    probe = SmartAgriModel("baseline", seed=1, n_farmers=4, seasons=1)
    return [row["key"] for row in probe.cfg["params"]["scenarios"]]


def _summarise_model(model: SmartAgriModel) -> dict[str, float]:
    df = model.to_dataframe()
    return {
        "mitigation_rate": float(df["mitigation_rate"].mean()),
        "effective_use_rate": float(df["effective_use_rate"].mean()),
        "mean_wait_days": float(df["mean_wait_days"].mean()),
        "income_p10": float(df["income_p10"].mean()),
        "equity_gap": float(df["equity_gap"].mean()),
        "mountain_gap": float(df["mountain_gap"].mean()),
        "fiscal_cumulative": float(df["fiscal_cumulative"].iloc[-1]),
    }


def _run_one(scenario: str, seed: int, n_farmers: int, seasons: int,
             overrides: list[tuple[tuple, float]] | None = None) -> dict:
    model = SmartAgriModel(scenario, seed=seed, n_farmers=n_farmers,
                           seasons=seasons)
    overrides = overrides or []
    risk_overrides = [(path, value) for path, value in overrides
                      if path[0] == "risk"]
    if risk_overrides:
        for path, value in risk_overrides:
            _apply(model, path, value)
        # Risk distributions are consumed at population construction.  Reset
        # only the behaviour stream and rebuild the same-sized population.
        stream = np.random.SeedSequence(model.seed).spawn(3)[0]
        model.rng = np.random.default_rng(stream)
        model._build_population(n_farmers)
        model.total_area_mu = sum(f.area_mu for f in model.farmers)
        model._sync_provider_capacity()
        model._apply_mountain_adaptation()
    for path, value in overrides:
        if path[0] != "risk":
            _apply(model, path, value)
    model.run()
    return _summarise_model(model)


def paired_differences(raw: pd.DataFrame, baseline: str = "baseline",
                       confidence: float = 0.95) -> pd.DataFrame:
    """Paired policy-minus-baseline uncertainty by metric."""
    keys = [k for k in ("parameter_draw", "rep") if k in raw.columns]
    if "rep" not in keys:
        raise ValueError("raw results require a rep column")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    rows = []
    for metric, direction in METRIC_DIRECTION.items():
        wide = raw.pivot_table(index=keys, columns="scenario",
                               values=metric, aggfunc="first")
        if baseline not in wide:
            raise ValueError(f"baseline scenario {baseline!r} is absent")
        for scenario in wide.columns:
            if scenario == baseline:
                continue
            diff = (wide[scenario] - wide[baseline]).dropna()
            n = int(diff.size)
            mean = float(diff.mean()) if n else float("nan")
            sd = float(diff.std(ddof=1)) if n > 1 else float("nan")
            se = sd / np.sqrt(n) if n > 1 else float("nan")
            half = z * se if n > 1 else float("nan")
            better = diff > 0 if direction == "max" else diff < 0
            rows.append({
                "scenario": scenario,
                "baseline": baseline,
                "metric": metric,
                "direction": direction,
                "n_pairs": n,
                "mean_difference": mean,
                "sd_difference": sd,
                "mcse_difference": se,
                "ci_low": mean - half if n > 1 else float("nan"),
                "ci_high": mean + half if n > 1 else float("nan"),
                "ci_half_width": half,
                "probability_better": float(better.mean()) if n else float("nan"),
            })
    return pd.DataFrame(rows)


def scenario_summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scenario, group in raw.groupby("scenario"):
        for metric in METRIC_DIRECTION:
            x = group[metric].dropna()
            rows.append({
                "scenario": scenario, "metric": metric, "n": len(x),
                "mean": x.mean(), "sd": x.std(ddof=1),
                "p05": x.quantile(0.05), "median": x.median(),
                "p95": x.quantile(0.95),
            })
    return pd.DataFrame(rows)


def ranking_probability(raw: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in ("parameter_draw", "rep") if k in raw.columns]
    rows = []
    for metric, direction in METRIC_DIRECTION.items():
        wide = raw.pivot_table(index=keys, columns="scenario",
                               values=metric, aggfunc="first")
        winners = wide.idxmax(axis=1) if direction == "max" else wide.idxmin(axis=1)
        freq = winners.value_counts(normalize=True)
        for scenario in wide.columns:
            rows.append({"metric": metric, "scenario": scenario,
                         "probability_best": float(freq.get(scenario, 0.0)),
                         "n_rankings": int(len(winners))})
    return pd.DataFrame(rows)


def _precision_reached(paired: pd.DataFrame,
                       relative_half_width: float) -> bool:
    if paired.empty or paired["ci_half_width"].isna().any():
        return False
    targets = paired.apply(
        lambda r: max(ABSOLUTE_TOLERANCE[r["metric"]],
                      relative_half_width * abs(r["mean_difference"])), axis=1)
    return bool((paired["ci_half_width"] <= targets).all())


def run_stochastic(config: MonteCarloConfig,
                   scenarios: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    scenarios = scenarios or _scenario_keys()
    rows = []
    converged = False
    stop_reps = config.max_reps
    for start in range(0, config.max_reps, config.batch_size):
        end = min(start + config.batch_size, config.max_reps)
        for rep in range(start, end):
            seed = config.seed0 + rep
            for scenario in scenarios:
                result = _run_one(scenario, seed, config.n_farmers,
                                  config.seasons)
                rows.append({"rep": rep, "seed": seed,
                             "scenario": scenario, **result})
        raw = pd.DataFrame(rows)
        if end >= config.min_reps:
            paired = paired_differences(raw, confidence=config.confidence)
            if _precision_reached(paired, config.relative_half_width):
                converged = True
                stop_reps = end
                break
    return pd.DataFrame(rows), {
        "converged": converged,
        "stop_reps": stop_reps,
        "stopping_rule": (
            "all policy-vs-baseline CI half-widths <= max(metric absolute "
            "tolerance, relative_half_width * |mean difference|)"
        ),
    }


def _latin_hypercube(n: int, dimensions: int,
                     rng: np.random.Generator) -> np.ndarray:
    u = np.empty((n, dimensions), dtype=float)
    for j in range(dimensions):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def parameter_design(n_draws: int, seed: int) -> tuple[pd.DataFrame, list]:
    specs = [row for row in SENSITIVITY_SPEC if row[0] in EPISTEMIC_LABELS]
    cube = _latin_hypercube(n_draws, len(specs), np.random.default_rng(seed))
    rows, overrides = [], []
    for i in range(n_draws):
        record = {"parameter_draw": i}
        draw = []
        for j, (label, path, lo, _mid, hi, _module) in enumerate(specs):
            value = float(lo + cube[i, j] * (hi - lo))
            record[label] = value
            draw.append((path, value))
        rows.append(record)
        overrides.append(draw)
    return pd.DataFrame(rows), overrides


def run_epistemic(n_draws: int, inner_reps: int, config: MonteCarloConfig,
                  scenarios: list[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = scenarios or _scenario_keys()
    design, draws = parameter_design(n_draws, config.seed0 + 100000)
    rows = []
    for draw_id, overrides in enumerate(draws):
        for rep in range(inner_reps):
            seed = config.seed0 + draw_id * 1000 + rep
            for scenario in scenarios:
                result = _run_one(scenario, seed, config.n_farmers,
                                  config.seasons, overrides)
                rows.append({"parameter_draw": draw_id, "rep": rep,
                             "seed": seed, "scenario": scenario, **result})
    return pd.DataFrame(rows), design


def write_outputs(raw: pd.DataFrame, config: MonteCarloConfig,
                  status: dict, runtime_s: float,
                  epistemic_raw: pd.DataFrame | None = None,
                  parameter_draws: pd.DataFrame | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw.to_csv(OUT / "stochastic_raw.csv", index=False)
    scenario_summary(raw).to_csv(OUT / "stochastic_scenario_summary.csv",
                                 index=False)
    paired_differences(raw, confidence=config.confidence).to_csv(
        OUT / "stochastic_paired_differences.csv", index=False)
    ranking_probability(raw).to_csv(
        OUT / "stochastic_ranking_probability.csv", index=False)
    if epistemic_raw is not None and not epistemic_raw.empty:
        epistemic_raw.to_csv(OUT / "epistemic_raw.csv", index=False)
        scenario_summary(epistemic_raw).to_csv(
            OUT / "epistemic_scenario_summary.csv", index=False)
        paired_differences(epistemic_raw, confidence=config.confidence).to_csv(
            OUT / "epistemic_paired_differences.csv", index=False)
        ranking_probability(epistemic_raw).to_csv(
            OUT / "epistemic_ranking_probability.csv", index=False)
        parameter_draws.to_csv(OUT / "parameter_design.csv", index=False)
    manifest = {
        "model_version": __version__,
        "model_hash": _model_hash(),
        "config_hash": _config_hash(),
        "config": asdict(config),
        "scenarios": sorted(raw["scenario"].unique().tolist()),
        "stochastic_rows": int(len(raw)),
        "stochastic_replicates": int(raw["rep"].nunique()),
        "epistemic_rows": int(len(epistemic_raw)) if epistemic_raw is not None else 0,
        "epistemic_parameter_draws": (
            int(epistemic_raw["parameter_draw"].nunique())
            if epistemic_raw is not None and not epistemic_raw.empty else 0
        ),
        "epistemic_inner_replicates": (
            int(epistemic_raw["rep"].nunique())
            if epistemic_raw is not None and not epistemic_raw.empty else 0
        ),
        "metric_direction": METRIC_DIRECTION,
        "absolute_tolerance": ABSOLUTE_TOLERANCE,
        "status": status,
        "runtime_s": round(runtime_s, 3),
        "parameter_distributions": (
            "uniform Latin-hypercube audit envelopes; not empirical posteriors"
        ),
    }
    (OUT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartagri.monte_carlo")
    ap.add_argument("--n-farmers", type=int, default=150)
    ap.add_argument("--seasons", type=int, default=6)
    ap.add_argument("--min-reps", type=int, default=30)
    ap.add_argument("--max-reps", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--relative-half-width", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=81001)
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--parameter-draws", type=int, default=0)
    ap.add_argument("--inner-reps", type=int, default=5)
    args = ap.parse_args(argv)
    cfg = MonteCarloConfig(
        n_farmers=args.n_farmers, seasons=args.seasons,
        min_reps=args.min_reps, max_reps=args.max_reps,
        batch_size=args.batch_size,
        relative_half_width=args.relative_half_width, seed0=args.seed)
    t0 = time.time()
    raw, status = run_stochastic(cfg, args.scenarios)
    eraw = design = None
    if args.parameter_draws > 0:
        eraw, design = run_epistemic(args.parameter_draws, args.inner_reps,
                                     cfg, args.scenarios)
    write_outputs(raw, cfg, status, time.time() - t0, eraw, design)
    print(json.dumps({"output": str(OUT), **status}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
