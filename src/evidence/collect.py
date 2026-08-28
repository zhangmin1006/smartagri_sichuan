"""
collect.py
==========
The information-collection program.

    python -m evidence.collect --offline
    python -m evidence.collect --online --targets disruptions,technologies,policies
    python -m evidence.collect --online --rows 15 --no-robots

It answers the three scoping questions by assembling and exporting:

    1. which DISRUPTIONS the model focuses on, and why
    2. which SMART TECHNOLOGIES are actually used in Sichuan
    3. which POLICIES are relevant, with their instruments and targets

Offline mode uses only the curated registry in config/ and always succeeds.
Online mode adds literature (Crossref, OpenAlex) and probes the official
source list, degrading gracefully if any host is unavailable.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
if str(HERE.parents[1]) not in sys.path:
    sys.path.insert(0, str(HERE.parents[1]))

from evidence.harvesters import (OFFICIAL_SOURCES, CrossrefHarvester,
                                 OfficialPageHarvester, OpenAlexHarvester,
                                 PoliteFetcher)
from evidence.registry import Registry
from evidence.report import summarise, write_csv_json, write_markdown

DEFAULT_OUT = HERE.parents[2] / "outputs" / "evidence"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="evidence.collect",
        description="Collect scoping evidence for the Sichuan smart-agriculture "
                    "ABM-SD model: disruptions, technologies, policies.")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--online", action="store_true",
                      help="harvest literature and probe official sources")
    mode.add_argument("--offline", action="store_true",
                      help="curated registry only (default)")
    p.add_argument("--targets", default="disruptions,technologies,policies",
                   help="comma-separated subset to collect")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="output directory")
    p.add_argument("--rows", type=int, default=10,
                   help="literature records per query per source")
    p.add_argument("--max-queries", type=int, default=6,
                   help="cap on queries per category (politeness)")
    p.add_argument("--no-robots", action="store_true",
                   help="skip robots.txt checking (use only with permission)")
    p.add_argument("--no-cache", action="store_true", help="bypass the fetch cache")
    p.add_argument("--sources", default="", help="comma-separated source keys to probe")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-7s %(name)s | %(message)s")
    log = logging.getLogger("evidence.collect")

    online = bool(args.online)
    targets = {t.strip() for t in args.targets.split(",") if t.strip()}
    outdir = Path(args.out)

    log.info("loading curated registry from config/")
    reg = Registry()
    problems = reg.integrity_check()
    if problems:
        for p in problems:
            log.warning("config integrity: %s", p)
    else:
        log.info("config integrity: all cross-references resolve")

    records = []
    if "disruptions" in targets:
        r = reg.disruption_records()
        log.info("disruptions: %d curated records", len(r))
        records += r
    if "technologies" in targets:
        r = reg.technology_records()
        log.info("technologies: %d curated records", len(r))
        records += r
    if "policies" in targets:
        r = reg.policy_records()
        log.info("policies: %d curated records", len(r))
        records += r

    if online:
        fetcher = PoliteFetcher(respect_robots=not args.no_robots)
        crossref = CrossrefHarvester(fetcher, rows=args.rows)
        openalex = OpenAlexHarvester(fetcher, rows=args.rows)

        queries = reg.search_queries()
        cat_map = {"disruptions": "disruption", "technologies": "technology",
                   "policies": "policy", "theory": "theory"}
        for target, cat in cat_map.items():
            if target not in targets:
                continue
            qs = queries.get(cat, [])[: args.max_queries]
            log.info("harvesting literature for %s (%d queries)", cat, len(qs))
            for q in qs:
                got = crossref.search(q, cat) + openalex.search(q, cat)
                log.info("  %-46s -> %d records", q[:46], len(got))
                records += got

        wanted = {s.strip() for s in args.sources.split(",") if s.strip()}
        srcs = ([s for s in OFFICIAL_SOURCES if s["key"] in wanted]
                if wanted else OFFICIAL_SOURCES)
        log.info("probing %d official sources", len(srcs))
        probe_records = OfficialPageHarvester(fetcher).probe_all(srcs)
        reachable = sum(1 for r in probe_records
                        if r.record_id.endswith("-PROBE")
                        and "reachable" in r.claim and "unreachable" not in r.claim)
        log.info("  %d/%d sources reachable, %d records captured",
                 reachable, len(srcs), len(probe_records))
        records += probe_records

    records = _dedupe(records)
    log.info("collected %s", summarise(records))

    paths = write_csv_json(records, outdir)
    md = write_markdown(records, reg, outdir, online, {"integrity": problems})
    paths["report_md"] = md

    log.info("outputs written to %s", outdir)
    for k, v in paths.items():
        log.info("  %-16s %s", k, Path(v).name)

    print()
    print("=" * 74)
    print("COLLECTION COMPLETE")
    print("=" * 74)
    print(f"mode      : {'online + curated' if online else 'curated (offline)'}")
    print(f"targets   : {', '.join(sorted(targets))}")
    print(f"summary   : {summarise(records)}")
    print(f"outputs   : {outdir}")
    for k, v in paths.items():
        print(f"            {Path(v).name}")
    print()
    print("Scope decisions recorded:")
    print(f"  disruptions modelled : "
          f"{', '.join(d['id'] + ' ' + d['name_en'] for d in reg.disruptions['tier1'])}")
    bundles = [b for b in reg.technologies["bundles"]
               if b.get("status", "in_scope") == "in_scope"]
    print(f"  technologies modelled: "
          f"{', '.join(b['id'] + ' ' + b['name_en'] for b in bundles)}")
    print(f"  instruments modelled : "
          f"{', '.join(i['id'] + ' ' + i['name_en'] for i in reg.policies['instruments'])}")
    print()
    return 0


def _dedupe(records):
    """Drop repeated ids, and collapse the same paper found by both sources."""
    from evidence.harvesters import _norm_title

    seen_ids, seen_titles, out = set(), set(), []
    for r in records:
        if r.record_id in seen_ids:
            continue
        if r.category == "literature":
            key = _norm_title(r.name_en)
            if key and key in seen_titles:
                continue
            seen_titles.add(key)
        seen_ids.add(r.record_id)
        out.append(r)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
