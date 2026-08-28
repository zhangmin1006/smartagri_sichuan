"""
literature_calibration.py
=========================
Extracts CANDIDATE parameter values from the empirical literature, with the
source sentence attached, so that model priors can be justified against
published estimates instead of expert judgement.

What this does
--------------
For each influential parameter it runs targeted OpenAlex queries, rebuilds the
abstract from the inverted index, finds sentences that contain BOTH an anchor
phrase for that parameter AND a number in a plausible range, and records the
value together with the sentence, DOI and citation count.

What this deliberately does NOT do
----------------------------------
It does not silently average whatever it scrapes into a "calibrated" value.
Abstract text is noisy: a percentage near the phrase "yield loss" may be a
sample share, a significance level or a completely different treatment arm.
Every extracted row therefore carries its source sentence and is marked
`verified = False`. The aggregate that the module proposes is an
INTERQUARTILE RANGE across sources, offered as a defensible prior envelope,
not a point estimate. A human must read the flagged sources before any value
is promoted to evidence grade A or B.

Run
---
    python -m evidence.literature_calibration
    python -m evidence.literature_calibration --rows 25 --param eta_flood
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

OUT = HERE.parents[2] / "outputs" / "literature"
CACHE = HERE.parents[2] / "outputs" / "_lit_cache"

UA = ("SichuanSmartAgriResearch/0.2 (academic research; "
      "contact: zhangmin1006@gmail.com)")
MAILTO = "zhangmin1006@gmail.com"


# ---------------------------------------------------------------------------
# Target parameters, ordered by the influence ranking from the model audit
# ---------------------------------------------------------------------------
TARGETS = [
    {
        "key": "c1_farmer_risk",
        "label": "Farmer relative risk aversion (CRRA)",
        "symbol": "c1",
        "model_value": "Beta(4.5, 3.0) on [0.10, 0.92], mean 0.60",
        "unit": "coefficient",
        "bounds": (0.05, 3.0),
        "anchors": ["risk aversion", "risk-aversion", "crra", "relative risk",
                    "partial risk aversion", "risk preference"],
        "require": ["coefficient", "crra", "estimate", "mean", "average",
                    "elicit"],
        "queries": [
            "risk aversion coefficient elicitation farmers China experimental",
            "CRRA relative risk aversion estimate farm households",
            "measuring farmer risk preferences systematic review",
            "multiple price list risk aversion smallholder agriculture",
        ],
    },
    {
        "key": "eta_flood_machinery",
        "label": "Loss reduction from mechanised/drone operations under flood",
        "symbol": "eta_T3_flood",
        "model_value": "0.35",
        "unit": "fraction",
        "bounds": (0.02, 0.80),
        "anchors": ["yield loss", "loss reduction", "reduced losses",
                    "avoided loss", "damage reduction", "yield increase",
                    "timeliness"],
        "require": ["reduc", "increas", "improv", "avoid", "loss", "yield"],
        "queries": [
            "mechanisation harvest timeliness yield loss reduction China",
            "unmanned aerial vehicle plant protection yield effect China",
            "agricultural machinery service yield loss flood waterlogging",
            "timely harvesting reduces yield loss rice China",
        ],
    },
    {
        "key": "eta_drought_irrigation",
        "label": "Loss reduction from irrigation / fertigation under drought",
        "symbol": "eta_T2_drought",
        "model_value": "0.45",
        "unit": "fraction",
        "bounds": (0.02, 0.90),
        "anchors": ["yield", "water use efficiency", "drought", "irrigation",
                    "water saving", "water-saving"],
        "require": ["increas", "reduc", "improv", "yield", "sav"],
        "queries": [
            "drip irrigation water saving yield increase China maize wheat",
            "water fertilizer integration yield response China",
            "supplemental irrigation drought yield loss reduction China",
            "deficit irrigation yield response water productivity China",
        ],
    },
    {
        "key": "eta_warning",
        "label": "Loss reduction from early warning / climate information",
        "symbol": "eta_T1",
        "model_value": "0.12 drought / 0.28 flood",
        "unit": "fraction",
        "bounds": (0.02, 0.60),
        "anchors": ["early warning", "climate information", "forecast",
                    "weather information", "value of information"],
        "require": ["reduc", "avoid", "loss", "benefit", "increas", "damage"],
        "queries": [
            "value of climate information agriculture loss reduction farmers",
            "early warning system agricultural damage avoided benefit",
            "seasonal forecast use farm income benefit developing country",
            "weather information services smallholder yield impact",
        ],
    },
    {
        "key": "service_price",
        "label": "Outsourced field-operation service price",
        "symbol": "price_per_mu",
        "model_value": "12 currency units per mu (T3)",
        "unit": "price",
        "bounds": (1.0, 500.0),
        "anchors": ["service price", "service fee", "cost per hectare",
                    "yuan per mu", "custom hire", "outsourc"],
        "require": ["price", "fee", "cost", "yuan", "per"],
        "queries": [
            "agricultural machinery outsourcing service price China per mu",
            "custom hire service charge mechanisation cost smallholder China",
            "drone spraying service price per hectare China",
        ],
    },
    {
        "key": "adoption_rate",
        "label": "Observed adoption rate of smart / digital agricultural technology",
        "symbol": "adopt",
        "model_value": "0.10 - 0.17 simulated",
        "unit": "fraction",
        "bounds": (0.01, 0.95),
        "anchors": ["adoption rate", "adopted", "adoption of", "share of farmers",
                    "percentage of farmers", "penetration"],
        "require": ["adopt", "farmer", "household", "%"],
        "queries": [
            "adoption rate digital agriculture technology Chinese farmers survey",
            "smart agriculture technology adoption share smallholder China",
            "precision agriculture adoption rate survey China percentage",
        ],
    },
    {
        "key": "peer_effect",
        "label": "Peer / social network effect on adoption",
        "symbol": "beta_peer",
        "model_value": "0.30 weight",
        "unit": "fraction",
        "bounds": (0.01, 1.5),
        "anchors": ["peer effect", "social network", "neighbour", "neighbor",
                    "social learning", "imitation"],
        "require": ["adopt", "increas", "effect", "probability", "point"],
        "queries": [
            "peer effects agricultural technology adoption village China estimate",
            "social network learning adoption marginal effect farmers",
        ],
    },
]

# Topics that routinely produce look-alike numbers in the same sentences but
# measure something else entirely. The first pass of this module extracted
# biochar yield gains, soil-organic-matter effects and variance-decomposition
# shares as if they were irrigation efficacy, so they are excluded explicitly.
BLOCKLIST = [
    "biochar", "organic matter", "soc", "manure", "compost", "nitrogen rate",
    "co2", "elevated co2", "tillage", "cover crop", "variability can be",
    "variability accounts", "explains", "explained by", "significance",
    "confidence interval", "p <", "p-value", "r2", "response rate",
    "sample of", "respondents", "questionnaire",
]

# A sentence must mention the TECHNOLOGY and the OUTCOME together, not merely
# sit near an anchor word.
CO_REQUIRE = {
    "eta_drought_irrigation": (["irrigat", "water-sav", "water saving",
                                "fertigation", "drip", "sprinkler",
                                "supplemental water"],
                               ["yield", "production", "loss"]),
    "eta_flood_machinery": (["mechanis", "mechaniz", "machinery", "harvest",
                             "uav", "unmanned", "drone", "timeli", "spray"],
                            ["yield", "loss", "production"]),
    "eta_warning": (["early warning", "forecast", "climate information",
                     "weather information", "advisory"],
                    ["loss", "damage", "yield", "income", "benefit"]),
    "service_price": (["service", "outsourc", "custom hire", "machinery",
                       "spray"],
                      ["price", "fee", "charge", "cost", "yuan"]),
    "c1_farmer_risk": (["risk aversion", "risk-aversion", "crra",
                        "risk preference"],
                       ["coefficient", "estimate", "elicit", "mean", "average",
                        "median", "range"]),
    "adoption_rate": (["adopt"], ["farmer", "household", "respondent", "share"]),
    "peer_effect": (["peer", "social network", "neighbo", "social learning"],
                    ["adopt", "probability", "percentage point", "increase"]),
}

NUM = r"(\d{1,3}(?:\.\d+)?)"
PCT_RE = re.compile(NUM + r"\s*(?:%|per\s?cent|percent)", re.I)
COEF_RE = re.compile(r"(?:of|is|was|were|=|:)\s*(0?\.\d+|[1-2]\.\d+)", re.I)


# ---------------------------------------------------------------------------
def _rebuild_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[k] for k in sorted(pos))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _fetch(query: str, rows: int) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"\W+", "_", query)[:80]
    path = CACHE / f"{key}_{rows}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")).get("results", [])
    url = ("https://api.openalex.org/works?search="
           + requests.utils.quote(query)
           + f"&per-page={rows}&mailto={MAILTO}")
    try:
        r = requests.get(url, timeout=40, headers={"User-Agent": UA})
        r.raise_for_status()
        payload = r.json()
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(0.8)
        return payload.get("results", [])
    except Exception as exc:
        print(f"    ! query failed ({type(exc).__name__}): {query[:50]}")
        return []


def _extract(sentence: str, spec: dict) -> list[float]:
    """Pull plausible numeric values out of one sentence."""
    lo, hi = spec["bounds"]
    vals: list[float] = []

    if spec["unit"] == "fraction":
        for m in PCT_RE.finditer(sentence):
            v = float(m.group(1)) / 100.0
            if lo <= v <= hi:
                vals.append(v)
        for m in COEF_RE.finditer(sentence):
            v = float(m.group(1))
            if lo <= v <= hi:
                vals.append(v)
    elif spec["unit"] == "coefficient":
        for m in COEF_RE.finditer(sentence):
            v = float(m.group(1))
            if lo <= v <= hi:
                vals.append(v)
    elif spec["unit"] == "price":
        for m in re.finditer(NUM, sentence):
            v = float(m.group(1))
            if lo <= v <= hi:
                vals.append(v)
    return vals


def harvest(spec: dict, rows: int) -> pd.DataFrame:
    seen, out = set(), []
    for q in spec["queries"]:
        works = _fetch(q, rows)
        for w in works:
            title = (w.get("display_name") or "").strip()
            key = re.sub(r"\W+", "", title.lower())[:80]
            if not title or key in seen:
                continue
            seen.add(key)
            abstract = _rebuild_abstract(w.get("abstract_inverted_index"))
            if not abstract:
                continue
            blob = (title + ". " + abstract)
            for sent in _sentences(blob):
                low = sent.lower()
                if any(b in low for b in BLOCKLIST):
                    continue
                tech_terms, out_terms = CO_REQUIRE.get(
                    spec["key"], (spec["anchors"], spec["require"]))
                if not any(t in low for t in tech_terms):
                    continue
                if not any(o in low for o in out_terms):
                    continue
                for v in _extract(sent, spec):
                    out.append({
                        "parameter": spec["key"], "symbol": spec["symbol"],
                        "value": v, "unit": spec["unit"],
                        "title": title[:180],
                        "year": w.get("publication_year"),
                        "cited_by": w.get("cited_by_count", 0),
                        "doi": w.get("doi") or "",
                        "oa_url": (w.get("open_access") or {}).get("oa_url") or "",
                        "sentence": sent[:320],
                        "query": q,
                        "verified": False,
                    })
        print(f"    {q[:56]:56s} -> {len(out):>3} candidate values so far")
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# HAND-VERIFIED ANCHORS
# ---------------------------------------------------------------------------
# The automated pass above is a DISCOVERY tool. When its output was read
# sentence by sentence, roughly one value in thirty was actually measuring the
# quantity it was matched to: soil-compaction yield loss, zinc-nanoparticle
# trials, grafting experiments and forecast-accuracy scores were all being
# captured as irrigation or machinery efficacy. Abstract-level extraction is
# therefore recorded but NOT used to set any parameter.
#
# The entries below are the ones that survived reading. Each records what the
# source actually measures, which is usually adjacent to, not identical with,
# the model parameter, so each carries an explicit mapping note.
VERIFIED_ANCHORS = [
    {
        "parameter": "damage_max_drought",
        "symbol": "D1 yield_damage_max",
        "model_value": 0.45,
        "source": "Global Synthesis of Drought Effects on Maize and Wheat Production",
        "reported": "wheat 20.6 pct and maize 39.3 pct yield reduction at about 40 pct water reduction",
        "value_low": 0.206, "value_high": 0.393,
        "maps_to": "Upper bound on drought yield damage before mitigation. The "
                   "model ceiling of 0.45 sits just above the maize figure, "
                   "appropriate for a CEILING at maximum severity rather than "
                   "a mean effect.",
        "verdict": "model value SUPPORTED as a ceiling",
    },
    {
        "parameter": "eta_drought_irrigation",
        "symbol": "eta_T2_drought",
        "model_value": 0.45,
        "source": "Review on Drip Irrigation: Impact on Crop Yield, Quality and Water Saving",
        "reported": "drip irrigation raises yield by 28.92, 14.55, 8.03, 2.32 and 5.17 pct relative to flooding and other conventional methods",
        "value_low": 0.023, "value_high": 0.289,
        "maps_to": "Yield gain of drip over conventional irrigation, which is "
                   "the MARGINAL gain the model attributes to smart fertigation "
                   "over an existing irrigation asset. The model value of 0.45 "
                   "lies above the whole reported range.",
        "verdict": "model value looks OPTIMISTIC - candidate for revision",
    },
]


def verified_table() -> pd.DataFrame:
    return pd.DataFrame(VERIFIED_ANCHORS)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for spec in TARGETS:
        g = df[df.parameter == spec["key"]]
        if g.empty:
            rows.append({"parameter": spec["key"], "label": spec["label"],
                         "symbol": spec["symbol"], "n_values": 0,
                         "n_sources": 0, "model_value": spec["model_value"],
                         "lit_p25": np.nan, "lit_median": np.nan,
                         "lit_p75": np.nan, "verdict": "no usable evidence"})
            continue
        v = g.value.to_numpy()
        p25, med, p75 = np.percentile(v, [25, 50, 75])
        rows.append({
            "parameter": spec["key"], "label": spec["label"],
            "symbol": spec["symbol"], "n_values": len(v),
            "n_sources": g.title.nunique(),
            "model_value": spec["model_value"],
            "lit_p25": round(float(p25), 4),
            "lit_median": round(float(med), 4),
            "lit_p75": round(float(p75), 4),
            "verdict": "UNVERIFIED - discovery only, do not calibrate on this",
        })
    return pd.DataFrame(rows)


def write_markdown(raw: pd.DataFrame, summary: pd.DataFrame, path: Path) -> None:
    L = ["# Literature-derived parameter candidates", "",
         "Candidate values extracted from published abstracts, each with the "
         "sentence it came from.", "",
         "> **Nothing here is a calibrated value.** Abstract text is noisy: a "
         "number near an anchor phrase may be a sample share, a significance "
         "level or a different treatment arm. Every row is marked "
         "`verified = False`. The ranges below are prior ENVELOPES for "
         "sensitivity analysis, and no value should be promoted to evidence "
         "grade B until the source has been read.", "",
         "## Summary", "",
         "| Parameter | Symbol | Model value | Lit p25 | median | p75 | n values | n sources |",
         "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        L.append(f"| {r['label']} | `{r['symbol']}` | {r['model_value']} | "
                 f"{r['lit_p25']} | {r['lit_median']} | {r['lit_p75']} | "
                 f"{r['n_values']} | {r['n_sources']} |")
    L.append("")
    for spec in TARGETS:
        g = raw[raw.parameter == spec["key"]]
        if g.empty:
            continue
        L += [f"## {spec['label']}  (`{spec['symbol']}`)", "",
              f"Model currently uses: **{spec['model_value']}**", "",
              "| Value | Source | Year | Cites | Sentence |", "|---|---|---|---|---|"]
        for _, r in (g.sort_values("cited_by", ascending=False)
                       .drop_duplicates("sentence").head(14).iterrows()):
            sent = str(r["sentence"]).replace("|", "/")
            title = str(r["title"]).replace("|", "/")[:90]
            L.append(f"| {r['value']} | {title} | {r['year']} | "
                     f"{r['cited_by']} | {sent} |")
        L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evidence.literature_calibration")
    ap.add_argument("--rows", type=int, default=20)
    ap.add_argument("--param", default="", help="restrict to one parameter key")
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("LITERATURE PARAMETER EXTRACTION")
    print("=" * 74)

    frames = []
    for spec in TARGETS:
        if a.param and spec["key"] != a.param:
            continue
        print(f"\n[{spec['key']}] {spec['label']}")
        frames.append(harvest(spec, a.rows))
    raw = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame()

    if raw.empty:
        print("\nno candidate values extracted")
        return 1

    summary = summarise(raw)
    ver = verified_table()
    raw.to_csv(OUT / "literature_candidates.csv", index=False)
    summary.to_csv(OUT / "literature_summary.csv", index=False)
    ver.to_csv(OUT / "verified_anchors.csv", index=False)
    print("\n=== HAND-VERIFIED ANCHORS "
          "(the only rows fit to inform a parameter) ===")
    for _, r in ver.iterrows():
        print(f"  {r['symbol']:22s} model={r['model_value']}  "
              f"lit={r['value_low']}-{r['value_high']}  -> {r['verdict']}")
    write_markdown(raw, summary, OUT / "LITERATURE_CALIBRATION.md")

    print("\n" + "=" * 74)
    print(summary[["symbol", "model_value", "lit_p25", "lit_median",
                   "lit_p75", "n_values", "n_sources"]].to_string(index=False))
    print(f"\nOutputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
