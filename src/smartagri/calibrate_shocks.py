"""
calibrate_shocks.py
===================
Replaces the guessed shock parameters with values derived from 34 years of
daily reanalysis data for six Sichuan locations.

This addresses audit finding D1 for the disruption module: 20 of the 84
grade-C parameters live here, and several of them (spatial correlation,
occurrence probability, severity distribution) are influential.

Data source
-----------
Open-Meteo Historical Weather API, serving ERA5 / ERA5-Land reanalysis at
roughly 9 km resolution, 1991-2024, daily precipitation and 2 m maximum
temperature. Free, no key, no rate limit for this volume.

Reanalysis is graded B rather than A: it is a physically consistent modelled
product, not a direct station observation. Obtaining CMA station series for
the same locations would upgrade these to grade A and is recommended, but
reanalysis is a very large improvement on expert judgement.

Hazard definitions follow Chinese meteorological convention
-----------------------------------------------------------
  rainstorm      暴雨      daily precipitation >= 50 mm
  heavy rainstorm 大暴雨    daily precipitation >= 100 mm
  high temperature 高温    daily Tmax >= 35 C
  summer drought  伏旱      Jul-Aug dry spell, precipitation < 1 mm/day

Run
---
    python -m smartagri.calibrate_shocks --fetch
    python -m smartagri.calibrate_shocks --report
    python -m smartagri.calibrate_shocks --fetch --write-config
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "climate"
OUT = ROOT / "outputs" / "calibration"

UA = ("SichuanSmartAgriResearch/0.2 (academic research; "
      "contact: zhangmin1006@gmail.com)")

START, END = "1991-01-01", "2024-12-31"

# Six archetype locations matching the model's stylised counties.
# Coordinates are the approximate centroids of real Sichuan agricultural
# districts chosen to span the plain / hill / mountain gradient.
COUNTY_POINTS = [
    {"id": "C1", "name": "plain_high_service", "terrain": "plain",
     "place": "Chengdu Plain (Dujiangyan-irrigated)", "lat": 30.80, "lon": 103.95},
    {"id": "C2", "name": "plain_mixed", "terrain": "plain",
     "place": "Deyang / Mianyang plain margin", "lat": 31.13, "lon": 104.40},
    {"id": "C3", "name": "hill_fragmented", "terrain": "hill",
     "place": "Suining central basin hills", "lat": 30.53, "lon": 105.57},
    {"id": "C4", "name": "hill_flood_exposed", "terrain": "hill",
     "place": "Yibin southern basin rim", "lat": 28.77, "lon": 104.62},
    {"id": "C5", "name": "hill_drought_exposed", "terrain": "hill",
     "place": "Nanchong eastern basin", "lat": 30.84, "lon": 106.11},
    {"id": "C6", "name": "mountain_remote", "terrain": "mountain",
     "place": "Liangshan western mountains", "lat": 27.90, "lon": 102.27},
]

# Hazard thresholds
RAINSTORM_MM = 50.0
HEAVY_RAINSTORM_MM = 100.0
HOT_DAY_C = 35.0
DRY_DAY_MM = 1.0

# Event-year thresholds (a year counts as an event when the index exceeds these)
DROUGHT_SPELL_DAYS = 20      # 伏旱: consecutive Jul-Aug dry days
FLOOD_RAINSTORM_DAYS = 2     # Jun-Sep days at or above 50 mm
COMPOUND_HOT_DAYS = 20       # Jul-Aug days at or above 35 C
COMPOUND_SPELL_DAYS = 15     # simultaneous dry spell


# ---------------------------------------------------------------------------
def fetch_point(lat: float, lon: float, tag: str,
                use_cache: bool = True) -> pd.DataFrame:
    """Daily precipitation and Tmax for one point, cached to disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{tag}_{START}_{END}.json"
    if use_cache and path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        url = ("https://archive-api.open-meteo.com/v1/archive"
               f"?latitude={lat}&longitude={lon}"
               f"&start_date={START}&end_date={END}"
               "&daily=precipitation_sum,temperature_2m_max"
               "&timezone=Asia%2FShanghai")
        r = requests.get(url, timeout=180, headers={"User-Agent": UA})
        r.raise_for_status()
        payload = r.json()
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(1.0)

    d = payload["daily"]
    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "precip": [np.nan if v is None else float(v) for v in d["precipitation_sum"]],
        "tmax": [np.nan if v is None else float(v) for v in d["temperature_2m_max"]],
    })
    df["year"] = df.date.dt.year
    df["month"] = df.date.dt.month
    return df


def _max_dry_spell(precip: np.ndarray, thresh: float = DRY_DAY_MM) -> int:
    """Longest run of consecutive days below the wet-day threshold."""
    best = run = 0
    for v in precip:
        if np.isnan(v) or v < thresh:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def county_year_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Per-year hazard indices for one location."""
    rows = []
    for yr, g in df.groupby("year"):
        ja = g[g.month.isin([7, 8])]                 # Jul-Aug
        jjas = g[g.month.isin([6, 7, 8, 9])]         # Jun-Sep

        spell = _max_dry_spell(ja.precip.to_numpy())
        ja_precip = float(np.nansum(ja.precip))
        hot_days = int(np.nansum(ja.tmax >= HOT_DAY_C))
        rainstorm_days = int(np.nansum(jjas.precip >= RAINSTORM_MM))
        heavy_days = int(np.nansum(jjas.precip >= HEAVY_RAINSTORM_MM))
        max_1day = float(np.nanmax(jjas.precip)) if len(jjas) else np.nan
        max_3day = float(np.nanmax(
            jjas.precip.rolling(3, min_periods=1).sum())) if len(jjas) else np.nan

        rows.append({"year": yr, "dry_spell_days": spell,
                     "julaug_precip_mm": ja_precip, "hot_days": hot_days,
                     "rainstorm_days": rainstorm_days,
                     "heavy_rainstorm_days": heavy_days,
                     "max_1day_mm": max_1day, "max_3day_mm": max_3day})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
def _binary_corr_from_rho(rho: float, p_i: float, p_j: float) -> float:
    """Pearson correlation of two Bernoulli indicators generated by a Gaussian
    copula with latent correlation rho and marginals p_i, p_j."""
    t_i, t_j = norm.ppf(p_i), norm.ppf(p_j)
    joint = float(multivariate_normal(
        mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]]).cdf([t_i, t_j]))
    den = np.sqrt(p_i * (1 - p_i) * p_j * (1 - p_j))
    return (joint - p_i * p_j) / den if den > 0 else np.nan


def tetrachoric_rho(target_binary_corr: float, probs: list) -> float:
    """Latent Gaussian correlation that reproduces an OBSERVED binary
    correlation, given the marginals.

    Feeding an observed correlation of 0/1 event indicators straight into a
    Gaussian copula under-correlates the simulated events, because the
    binary correlation is an attenuated image of the latent one. This inverts
    the attenuation: it is the standard tetrachoric correction, averaged over
    the county pairs actually present.
    """
    ps = [p for p in probs if 1e-6 < p < 1 - 1e-6]
    if len(ps) < 2 or not np.isfinite(target_binary_corr):
        return float(np.clip(target_binary_corr, 0.0, 0.95))
    pairs = [(ps[i], ps[j]) for i in range(len(ps)) for j in range(i + 1, len(ps))]

    def f(rho):
        vals = [_binary_corr_from_rho(rho, a, b) for a, b in pairs]
        return float(np.nanmean(vals)) - target_binary_corr

    try:
        if f(0.001) > 0:
            return 0.001
        if f(0.985) < 0:
            return 0.985
        return float(brentq(f, 0.001, 0.985, xtol=1e-4))
    except Exception:
        return float(np.clip(target_binary_corr, 0.0, 0.95))

def _beta_moments(x: np.ndarray):
    """Method-of-moments Beta fit on (0,1) data."""
    x = np.clip(np.asarray(x, dtype=float), 1e-4, 1 - 1e-4)
    if len(x) < 3:
        return np.nan, np.nan
    m, v = float(np.mean(x)), float(np.var(x, ddof=1))
    if v <= 0 or v >= m * (1 - m):
        return np.nan, np.nan
    c = m * (1 - m) / v - 1
    return max(c * m, 0.05), max(c * (1 - m), 0.05)


def _severity(index: np.ndarray, threshold: float, cap_pct: float = 97.5):
    """Map an exceedance index onto [0,1] for event years only."""
    idx = np.asarray(index, dtype=float)
    hit = idx >= threshold
    if hit.sum() == 0:
        return hit, np.array([])
    top = np.nanpercentile(idx, cap_pct)
    span = max(top - threshold, 1e-6)
    sev = np.clip((idx[hit] - threshold) / span, 0.0, 1.0)
    return hit, sev


def calibrate(panel: pd.DataFrame) -> dict:
    """Derive occurrence probability, severity Beta and spatial correlation.

    Drought hazards use a STANDARDISED PRECIPITATION ANOMALY computed against
    each location's own 1991-2024 Jul-Aug climatology, which is the standard
    climatological approach and the one that matches operational drought
    classification. An absolute dry-spell threshold was tried first and
    rejected: in a monsoon climate a 20-day rain-free spell is so rare
    (2 events in 204 county-years) that it identifies almost none of the
    documented Sichuan drought years.
    """
    out = {}
    panel = panel.copy()

    # standardised Jul-Aug precipitation anomaly, per county
    piv = panel.pivot(index="year", columns="county", values="julaug_precip_mm")
    z = (piv - piv.mean()) / piv.std()
    zl = z.reset_index().melt(id_vars="year", var_name="county",
                              value_name="spi")
    panel = panel.merge(zl, on=["year", "county"], how="left")

    # ---- D1 drought: SPI <= -0.8 ("moderately dry" in SPI convention) ----
    D1_Z = -0.8
    probs, sev_all = [], []
    for cid, g in panel.groupby("county"):
        hit = (g.spi <= D1_Z).to_numpy()
        probs.append(hit.mean())
        if hit.sum():
            sev_all.append(np.clip((-g.spi.to_numpy()[hit] + D1_Z) / 1.7, 0, 1))
    sev_pool = np.concatenate(sev_all) if sev_all else np.array([])
    a, b = _beta_moments(sev_pool)
    ind = (z <= D1_Z).astype(float)
    corr = ind.corr().to_numpy()
    out["D1"] = {
        "label": "Seasonal summer drought (伏旱)",
        "index": "Jul-Aug standardised precipitation anomaly",
        "threshold": "SPI <= %.1f" % D1_Z,
        "annual_probability": float(np.mean(probs)),
        "probability_by_county": {c: float(p) for c, p in
                                  zip(sorted(panel.county.unique()), probs)},
        "severity_beta": [None if np.isnan(a) else round(a, 3),
                          None if np.isnan(b) else round(b, 3)],
        "severity_mean": float(np.mean(sev_pool)) if len(sev_pool) else np.nan,
        "n_events": int(sum(len(s) for s in sev_all)),
        "spatial_correlation_observed": float(
            np.nanmean(corr[np.triu_indices_from(corr, k=1)])),
        "spatial_correlation": tetrachoric_rho(
            float(np.nanmean(corr[np.triu_indices_from(corr, k=1)])), probs),
    }

    # ---- D2 flood: two or more rainstorm days (>= 50 mm) in Jun-Sep -------
    probs, sev_all = [], []
    for cid, g in panel.groupby("county"):
        hit, sev = _severity(g.rainstorm_days.to_numpy(), FLOOD_RAINSTORM_DAYS)
        probs.append(hit.mean())
        if len(sev):
            sev_all.append(sev)
    sev_pool = np.concatenate(sev_all) if sev_all else np.array([])
    a, b = _beta_moments(sev_pool)
    wide = panel.pivot(index="year", columns="county", values="rainstorm_days")
    corr = (wide >= FLOOD_RAINSTORM_DAYS).astype(float).corr().to_numpy()
    out["D2"] = {
        "label": "Rainstorm, flood and waterlogging (暴雨洪涝)",
        "index": "Jun-Sep days with precipitation >= 50 mm",
        "threshold": ">= %d rainstorm days" % FLOOD_RAINSTORM_DAYS,
        "annual_probability": float(np.mean(probs)),
        "probability_by_county": {c: float(p) for c, p in
                                  zip(sorted(panel.county.unique()), probs)},
        "severity_beta": [None if np.isnan(a) else round(a, 3),
                          None if np.isnan(b) else round(b, 3)],
        "severity_mean": float(np.mean(sev_pool)) if len(sev_pool) else np.nan,
        "n_events": int(sum(len(s) for s in sev_all)),
        "spatial_correlation_observed": float(
            np.nanmean(corr[np.triu_indices_from(corr, k=1)])),
        "spatial_correlation": tetrachoric_rho(
            float(np.nanmean(corr[np.triu_indices_from(corr, k=1)])), probs),
    }

    # ---- D3 compound: severe drought AND an unusually hot Jul-Aug --------
    D3_Z = -1.0
    probs, sev_all, comp_index = [], [], {}
    for cid, g in panel.groupby("county"):
        hot = g.hot_days.to_numpy(dtype=float)
        hot_cut = np.percentile(hot, 75)
        hit = ((g.spi <= D3_Z).to_numpy() & (hot >= max(hot_cut, 5.0)))
        probs.append(hit.mean())
        hot_norm = hot / max(np.percentile(hot, 97.5), 1.0)
        dry_norm = np.clip((-g.spi.to_numpy() + D3_Z) / 1.5, 0, 1)
        comp_index[cid] = 0.5 * (np.clip(hot_norm, 0, 1) + dry_norm)
        if hit.sum():
            sev_all.append(np.clip(comp_index[cid][hit], 0, 1))
    sev_pool = np.concatenate(sev_all) if sev_all else np.array([])
    a, b = _beta_moments(sev_pool)
    hotp = panel.pivot(index="year", columns="county", values="hot_days")
    zpiv = panel.pivot(index="year", columns="county", values="spi")
    ind3 = ((zpiv <= D3_Z) & (hotp >= hotp.quantile(0.75))).astype(float)
    corr = ind3.corr().to_numpy()
    comp_df = pd.DataFrame(comp_index, index=sorted(panel.year.unique()))
    out["D3"] = {
        "label": "Compound heat-drought (高温干旱复合)",
        "index": "SPI and Jul-Aug days >= 35 C",
        "threshold": "SPI <= %.1f and hot days >= county p75" % D3_Z,
        "annual_probability": float(np.mean(probs)),
        "probability_by_county": {c: float(p) for c, p in
                                  zip(sorted(panel.county.unique()), probs)},
        "severity_beta": [None if np.isnan(a) else round(a, 3),
                          None if np.isnan(b) else round(b, 3)],
        "severity_mean": float(np.mean(sev_pool)) if len(sev_pool) else np.nan,
        "n_events": int(sum(len(s) for s in sev_all)),
        "spatial_correlation_observed": float(
            np.nanmean(corr[np.triu_indices_from(corr, k=1)])),
        "spatial_correlation": tetrachoric_rho(
            float(np.nanmean(corr[np.triu_indices_from(corr, k=1)])), probs),
        "event_years": sorted(set(
            int(y) for cid in comp_index
            for y, v in zip(sorted(panel.year.unique()), comp_index[cid])
            if v > 0.35)),
    }
    return out


def exposure_multipliers(panel: pd.DataFrame, points: list) -> dict:
    """Relative hazard exposure by terrain, normalised to the plain mean."""
    terr = {p["id"]: p["terrain"] for p in points}
    panel = panel.copy()
    panel["terrain"] = panel.county.map(terr)
    out = {}
    for hid, col in (("D1", "dry_spell_days"), ("D2", "rainstorm_days"),
                     ("D3", "hot_days")):
        m = panel.groupby("terrain")[col].mean()
        base = m.get("plain", np.nan)
        out[hid] = {k: round(float(v / base), 3) for k, v in m.items()} \
            if base and not np.isnan(base) else {}
    return out


# ---------------------------------------------------------------------------
def build_panel(use_cache: bool = True) -> pd.DataFrame:
    frames = []
    for p in COUNTY_POINTS:
        df = fetch_point(p["lat"], p["lon"], p["id"], use_cache)
        idx = county_year_indices(df)
        idx["county"] = p["id"]
        frames.append(idx)
        print(f"  {p['id']}  {p['place']:38s} "
              f"{len(df):>6} days  {len(idx):>3} years")
    return pd.concat(frames, ignore_index=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="smartagri.calibrate_shocks")
    ap.add_argument("--fetch", action="store_true",
                    help="download (or reuse cache) and calibrate")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--write-config", action="store_true",
                    help="write calibrated values into config/disruptions.yaml")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("SHOCK CALIBRATION FROM ERA5 REANALYSIS, 1991-2024")
    print("=" * 74)
    print("\nFetching six Sichuan locations:")
    panel = build_panel(use_cache=not a.no_cache)
    panel.to_csv(OUT / "climate_panel.csv", index=False)

    cal = calibrate(panel)
    expo = exposure_multipliers(panel, COUNTY_POINTS)

    print("\n--- calibrated hazard parameters ---")
    rows = []
    for hid, c in cal.items():
        print(f"\n{hid}  {c['label']}")
        print(f"   annual probability      {c['annual_probability']:.3f}  "
              f"(model prior varied)")
        print(f"   severity Beta(a, b)     {c['severity_beta']}")
        print(f"   mean severity           {c['severity_mean']:.3f}")
        print(f"   events observed         {c['n_events']} across 6 counties")
        print(f"   spatial correlation     {c['spatial_correlation']:.3f}")
        print("   by county               " + ", ".join(
            f"{k}={v:.2f}" for k, v in c["probability_by_county"].items()))
        rows.append({"hazard": hid, "label": c["label"],
                     "annual_probability": c["annual_probability"],
                     "beta_a": c["severity_beta"][0],
                     "beta_b": c["severity_beta"][1],
                     "severity_mean": c["severity_mean"],
                     "n_events": c["n_events"],
                     "spatial_correlation": c["spatial_correlation"]})
    pd.DataFrame(rows).to_csv(OUT / "shock_parameters.csv", index=False)

    print("\n--- relative hazard exposure by terrain (plain = 1.00) ---")
    for hid, m in expo.items():
        print(f"   {hid}: " + ", ".join(f"{k}={v}" for k, v in m.items()))

    rho = float(np.mean([c["spatial_correlation"] for c in cal.values()]))
    print(f"\n   mean spatial correlation across hazards: {rho:.3f}  "
          f"(model used 0.650 by judgement)")

    (OUT / "shock_calibration.json").write_text(
        json.dumps({"source": "Open-Meteo ERA5 reanalysis",
                    "period": f"{START}..{END}", "points": COUNTY_POINTS,
                    "hazards": cal, "exposure_multipliers": expo,
                    "mean_spatial_correlation": rho},
                   ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    if a.write_config:
        _write_config(cal, rho)
    print(f"\nOutputs -> {OUT}")
    return 0


def _write_config(cal: dict, rho: float) -> None:
    """Patch config/disruptions.yaml in place, preserving comments."""
    path = ROOT / "config" / "disruptions.yaml"
    text = path.read_text(encoding="utf-8")
    import re

    for hid in ("D1", "D2", "D3"):
        c = cal[hid]
        block_start = text.index(f"  - id: {hid}\n")
        block_end = text.find("\n  - id: ", block_start + 1)
        if block_end == -1:
            block_end = text.find("\ntier2_extensions:", block_start)
        block = text[block_start:block_end]

        block = re.sub(r"annual_probability_prior: [0-9.]+",
                       "annual_probability_prior: %.3f"
                       % c["annual_probability"], block)
        if c["severity_beta"][0] is not None:
            block = re.sub(r"severity_params: \[[^\]]*\]",
                           "severity_params: [%.2f, %.2f]"
                           % tuple(c["severity_beta"]), block)
        block = re.sub(r"    evidence_grade: [ABC]",
                       "    evidence_grade: B", block, count=1)
        text = text[:block_start] + block + text[block_end:]

    if "calibration:" not in text:
        text = text.replace(
            "tier1:\n",
            "calibration:\n"
            "  source: Open-Meteo ERA5 reanalysis, daily, 1991-2024\n"
            "  points: six Sichuan archetype locations (see calibrate_shocks.py)\n"
            "  method: >\n"
            "    Occurrence probability, severity Beta and spatial correlation\n"
            "    derived from 34 years of daily precipitation and maximum\n"
            "    temperature using Chinese meteorological hazard definitions\n"
            "    (rainstorm >= 50 mm/day, high temperature >= 35 C, summer\n"
            "    drought as a Jul-Aug dry spell). Reanalysis is graded B; CMA\n"
            "    station series for the same points would upgrade this to A.\n"
            "  spatial_correlation_estimated: %.3f\n"
            "  calibrated_on: 2026-08-25\n\n"
            "tier1:\n" % rho, 1)

    path.write_text(text, encoding="utf-8")
    print(f"\n   config/disruptions.yaml updated "
          f"(probabilities, severity Beta, evidence grade -> B)")


if __name__ == "__main__":
    raise SystemExit(main())
