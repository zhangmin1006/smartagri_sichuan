"""
parameter_dossier.py
====================
Assembles a sourced evidence dossier for the model parameters the audit
identified as most influential, so that calibration values can be extracted
by reading primary sources rather than invented.

What this does and does not do
------------------------------
It DOES find, filter, rank and group the empirical literature that reports
each target parameter, with DOIs and open-access links where available.

It does NOT extract numeric parameter values from abstracts. Abstract text is
not a reliable source for a coefficient, and a plausible-looking number pulled
from an abstract by pattern matching would be worse than an honest gap. Every
row in the dossier is a candidate source for a human to read.

Run
---
    python -m evidence.parameter_dossier
    python -m evidence.parameter_dossier --rows 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

import pandas as pd

from evidence.harvesters import (CrossrefHarvester, OpenAlexHarvester,
                                 PoliteFetcher, _norm_title)

OUT = HERE.parents[2] / "outputs" / "dossier"

# Targets are ordered by the influence ranking established in the audit.
TARGETS = [
    {"rank": 1, "param": "Service capacity coverage and queueing",
     "symbol": "C(0), W(rho)", "effect": "164.5%",
     "note": "Best resolved by work-order records rather than literature; "
             "published queueing studies of agricultural machinery services "
             "give functional forms and plausible utilisation ranges.",
     "queries": [
         "agricultural machinery service scheduling queueing China",
         "custom hiring agricultural machinery utilisation developing country",
         "harvest timeliness loss machinery capacity constraint",
     ]},
    {"rank": 2, "param": "Choice model scale and attribute weights",
     "symbol": "tau", "effect": "122.7%",
     "note": "Requires a discrete choice experiment. Look for published DCEs "
             "on agricultural technology or service adoption in China that "
             "report the scale parameter or willingness-to-pay space "
             "estimates.",
     "queries": [
         "discrete choice experiment agricultural technology adoption China farmers",
         "choice experiment willingness to pay irrigation service farmers China",
         "mixed logit farmer technology adoption scale heterogeneity",
     ]},
    {"rank": 3, "param": "Provider effort cost",
     "symbol": "gamma", "effect": "58.5%",
     "note": "Provider-side operating cost per hectare or per hour, and the "
             "cost of raising service quality or timeliness.",
     "queries": [
         "custom hiring service provider operating cost agricultural machinery",
         "drone spraying service cost per hectare China",
         "agricultural service provider profitability contract farming",
     ]},
    {"rank": 4, "param": "Provider risk aversion",
     "symbol": "c2", "effect": "28.9%",
     "note": "Rarely elicited. Search covers agribusiness and SME risk "
             "preference elicitation as the closest available evidence.",
     "queries": [
         "risk preference elicitation agricultural service providers firms China",
         "risk aversion small business owners experimental elicitation China",
     ]},
    {"rank": 5, "param": "Provider reservation utility",
     "symbol": "U_min", "effect": "23.6%",
     "note": "Minimum acceptable price and job acceptance behaviour; also "
             "informed by the contract farming participation literature.",
     "queries": [
         "reservation wage service provider job acceptance agriculture",
         "contract farming participation constraint smallholder China",
     ]},
    {"rank": 6, "param": "Technology efficacy against flood and drought",
     "symbol": "eta_k", "effect": "21.9% / 16.0%",
     "note": "Yield-loss reduction attributable to irrigation, drone plant "
             "protection, mechanised timeliness and early warning.",
     "queries": [
         "irrigation drought yield loss reduction China maize rice",
         "unmanned aerial vehicle plant protection spraying efficacy yield China",
         "early warning system agricultural loss reduction value of information",
         "water saving irrigation technology yield response China",
     ]},
    {"rank": 7, "param": "Service price per mu",
     "symbol": "price", "effect": "19.6%",
     "note": "Market prices for outsourced field operations.",
     "queries": [
         "agricultural outsourcing service price land preparation harvesting China",
         "machinery service market price smallholder China mechanisation",
     ]},
    {"rank": 9, "param": "Social learning and peer effects",
     "symbol": "beta_peer", "effect": "15.5%",
     "note": "Peer effect magnitudes in agricultural technology diffusion.",
     "queries": [
         "social network peer effects agricultural technology adoption China village",
         "information diffusion extension farmer network experiment",
     ]},
    {"rank": 11, "param": "Farmer risk aversion",
     "symbol": "c1", "effect": "12.1%",
     "note": "CRRA or partial risk aversion coefficients elicited from "
             "Chinese farm households.",
     "queries": [
         "risk aversion elicitation Chinese farmers experimental lottery CRRA",
         "risk preferences farm households China multiple price list",
         "risk aversion technology adoption smallholder农户 China",
     ]},
    {"rank": 10, "param": "Verifiability, monitoring and contract form",
     "symbol": "nu", "effect": "12.7%",
     "note": "Directly tests the theoretical core: contract form, monitoring "
             "technology and moral hazard in agricultural services.",
     "queries": [
         "moral hazard agricultural service contract monitoring verification",
         "principal agent contract form agriculture risk sharing empirical",
         "land trusteeship service outsourcing contract China 托管",
     ]},
]


def build(rows: int = 12, max_per_target: int = 14) -> pd.DataFrame:
    fetcher = PoliteFetcher()
    cr = CrossrefHarvester(fetcher, rows=rows)
    oa = OpenAlexHarvester(fetcher, rows=rows)

    records, seen_titles = [], set()
    for t in TARGETS:
        found = []
        for q in t["queries"]:
            for rec in cr.search(q, "param") + oa.search(q, "param"):
                key = _norm_title(rec.name_en)
                if not key or key in seen_titles:
                    continue
                seen_titles.add(key)
                score = 0
                notes = rec.notes or ""
                if "relevance=" in notes:
                    try:
                        score = int(notes.split("relevance=")[1].split(";")[0])
                    except Exception:
                        score = 0
                cited = 0
                if "cited_by=" in (rec.claim or ""):
                    try:
                        cited = int(rec.claim.split("cited_by=")[1].split(";")[0])
                    except Exception:
                        cited = 0
                found.append({
                    "rank": t["rank"], "parameter": t["param"],
                    "symbol": t["symbol"], "influence": t["effect"],
                    "title": rec.name_en, "year": rec.year,
                    "source": rec.source, "url": rec.url,
                    "relevance": score, "cited_by": cited, "query": q,
                })
            print(f"  [{t['rank']:>2}] {q[:58]:58s} -> {len(found):>3} cumulative")
        found.sort(key=lambda r: (-r["relevance"], -r["cited_by"]))
        records.extend(found[:max_per_target])

    return pd.DataFrame(records)


def write_markdown(df: pd.DataFrame, path: Path) -> None:
    lines = ["# Parameter evidence dossier", "",
             "Candidate primary sources for the parameters the audit "
             "identified as most influential. Every parameter listed here is "
             "currently evidence grade C.", "",
             "> **These are candidate sources, not extracted values.** Numeric "
             "parameter values must be read from the papers. No value in this "
             "dossier has been inferred from an abstract.", ""]
    for (rank, param), g in df.groupby(["rank", "parameter"], sort=True):
        t = next(x for x in TARGETS if x["param"] == param)
        lines += [f"## {rank}. {param}  (`{t['symbol']}`)", "",
                  f"**Model influence:** {t['effect']} maximum effect on any "
                  f"output.", "", t["note"], "",
                  "| Title | Year | Cited | Link |", "|---|---|---|---|"]
        for _, r in g.head(12).iterrows():
            title = str(r["title"]).replace("|", "/")[:150]
            url = r["url"] if isinstance(r["url"], str) and r["url"] else ""
            lines.append(f"| {title} | {r['year']} | {r['cited_by']} | "
                         f"{url} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evidence.parameter_dossier")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--max-per-target", type=int, default=14)
    a = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    print("=" * 74)
    print("PARAMETER EVIDENCE DOSSIER")
    print("=" * 74)
    df = build(a.rows, a.max_per_target)

    df.to_csv(OUT / "parameter_dossier.csv", index=False)
    write_markdown(df, OUT / "PARAMETER_DOSSIER.md")

    print(f"\ncandidate sources retained: {len(df)}")
    print(df.groupby(["rank", "parameter"]).size().to_string())
    print(f"\nOutputs -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
