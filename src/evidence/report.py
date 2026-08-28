"""
report.py
=========
Turns collected evidence records into the deliverables:

    evidence.csv         every record, flat, one row each
    evidence.json        the same, structured
    parameters.csv       only claim_type == parameter, ready for calibration
    data_gaps.csv        what must still be obtained, ranked
    EVIDENCE_REPORT.md   a readable briefing with the scope decisions

The exporter enforces the discipline the evidence report insists on:
policy TARGETS and infrastructure COVERAGE are never reported in the same
column as measured ADOPTION.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from .registry import EvidenceRecord, records_to_rows

BANNER = "Observed / Estimated / Simulated must stay distinguishable at all times."


def _df(records: list[EvidenceRecord]) -> pd.DataFrame:
    return pd.DataFrame(records_to_rows(records))


def write_csv_json(records: list[EvidenceRecord], outdir: Path) -> dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = _df(records)

    paths: dict[str, Path] = {}
    p = outdir / "evidence.csv"
    df.to_csv(p, index=False, encoding="utf-8-sig")
    paths["evidence_csv"] = p

    p = outdir / "evidence.json"
    p.write_text(json.dumps([r.as_dict() for r in records], ensure_ascii=False,
                            indent=2, default=str), encoding="utf-8")
    paths["evidence_json"] = p

    params = df[df["claim_type"] == "parameter"].copy()
    if not params.empty:
        params = params[["record_id", "category", "entity_id", "name_en",
                         "claim", "value", "evidence_grade", "verify", "notes"]]
        p = outdir / "parameters.csv"
        params.to_csv(p, index=False, encoding="utf-8-sig")
        paths["parameters_csv"] = p

    gaps = df[(df["verify"]) | (df["evidence_grade"] == "C")].copy()
    if not gaps.empty:
        gaps["priority"] = gaps.apply(_gap_priority, axis=1)
        gaps = gaps.sort_values(["priority", "category"])
        gaps = gaps[["priority", "record_id", "category", "entity_id",
                     "name_en", "claim", "evidence_grade", "notes"]]
        p = outdir / "data_gaps.csv"
        gaps.to_csv(p, index=False, encoding="utf-8-sig")
        paths["data_gaps_csv"] = p

    return paths


def _gap_priority(row) -> int:
    """1 = blocks calibration, 2 = weakens a headline claim, 3 = nice to have."""
    tags = str(row.get("tags", ""))
    if row.get("claim_type") == "parameter" and "prior" in tags:
        return 1
    if row.get("claim_type") == "target":
        return 2
    if "sichuan_anchor" in tags:
        return 2
    return 3


# ---------------------------------------------------------------------------
def write_markdown(records: list[EvidenceRecord], registry, outdir: Path,
                   online: bool, notes: dict | None = None) -> Path:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = _df(records)
    notes = notes or {}

    by_cat = Counter(df["category"])
    by_grade = Counter(df["evidence_grade"])
    lines: list[str] = []
    add = lines.append

    add("# Sichuan Smart Agriculture — Evidence Collection Report")
    add("")
    add(f"Generated: {date.today().isoformat()}  ")
    add(f"Mode: {'online + curated' if online else 'curated (offline)'}  ")
    add(f"Records: {len(df)}")
    add("")
    add(f"> {BANNER}")
    add("")

    add("## 1. Scope decisions")
    add("")
    add("### 1.1 Disruptions modelled")
    add("")
    add("| ID | Shock | Family | Grade | Coupled tech | Why |")
    add("|---|---|---|---|---|---|")
    for item in registry.disruptions.get("tier1", []) or []:
        why = " ".join((item.get("why_included") or "").split())
        add(f"| {item['id']} | {item.get('name_en','')} ({item.get('name_zh','')}) "
            f"| {item.get('family','')} | {item.get('evidence_grade','')} "
            f"| {', '.join(item.get('coupled_technologies', []))} "
            f"| {why[:210]}{'…' if len(why) > 210 else ''} |")
    add("")
    deferred = registry.disruptions.get("tier2_extensions", []) or []
    excluded = registry.disruptions.get("excluded", []) or []
    if deferred:
        add("**Deferred to version 2:** " + ", ".join(
            f"{d['id']} {d.get('name_en','')}" for d in deferred))
        add("")
    if excluded:
        add("**Excluded:** " + ", ".join(
            f"{d.get('name_en','')} — {' '.join((d.get('reason') or '').split())[:120]}"
            for d in excluded))
        add("")

    add("### 1.2 Technology bundles modelled")
    add("")
    add("| ID | Bundle | Channel | Access modes | Grade | Status |")
    add("|---|---|---|---|---|---|")
    for item in registry.technologies.get("bundles", []) or []:
        add(f"| {item['id']} | {item.get('name_en','')} ({item.get('name_zh','')}) "
            f"| {item.get('channel','')} | {', '.join(item.get('access_modes', []))} "
            f"| {item.get('evidence_grade','')} | {item.get('status','in_scope')} |")
    add("")
    add("Adoption ladder enforced throughout "
        "(coverage is never reported as adoption):")
    add("")
    for st in registry.technologies.get("adoption_states", []) or []:
        add(f"{st['id']}. **{st['key']}** — {st.get('label_en','')}")
    add("")

    add("### 1.3 Policy instruments modelled")
    add("")
    add("| ID | Instrument | Targets | SD stock | Equity |")
    add("|---|---|---|---|---|")
    for inst in registry.policies.get("instruments", []) or []:
        add(f"| {inst['id']} | {inst.get('name_en','')} ({inst.get('name_zh','')}) "
            f"| {', '.join(inst.get('targets', []))} | {inst.get('sd_stock','')} "
            f"| {inst.get('equity_flag','')} |")
    add("")

    add("## 2. Policy documents inventoried")
    add("")
    add("| Key | Document | Year | Grade | Verify |")
    add("|---|---|---|---|---|")
    for scope in ("national", "provincial"):
        for doc in registry.policies.get(scope, []) or []:
            add(f"| {doc['key']} | {doc.get('title_zh', doc.get('title_en',''))} "
                f"| {doc.get('year','')} | {doc.get('evidence_grade','')} "
                f"| {'yes' if doc.get('verify') else ''} |")
    add("")

    targets = df[df["claim_type"] == "target"]
    if not targets.empty:
        add("### 2.1 Provincial targets (commitments, NOT realised adoption)")
        add("")
        add("| Entity | Target | Value |")
        add("|---|---|---|")
        for _, r in targets.iterrows():
            add(f"| {r['entity_id']} | {r['claim']} | {r['value']} |")
        add("")

    add("## 3. Record inventory")
    add("")
    add("| Category | Records |")
    add("|---|---|")
    for k, v in sorted(by_cat.items()):
        add(f"| {k} | {v} |")
    add("")
    add("| Evidence grade | Records |")
    add("|---|---|")
    for k in ("A", "B", "C"):
        add(f"| {k} | {by_grade.get(k, 0)} |")
    add("")

    lit = df[df["category"] == "literature"]
    if not lit.empty:
        add(f"## 4. Literature harvested ({len(lit)} records)")
        add("")
        for ent, grp in lit.groupby("entity_id"):
            add(f"### {ent}")
            add("")
            for _, r in grp.head(15).iterrows():
                url = f" <{r['url']}>" if r["url"] else ""
                add(f"- **{r['name_en']}** ({r['year']}) — {r['claim']}{url}")
            add("")

    srcs = df[df["category"] == "source"]
    if not srcs.empty:
        probes = srcs[srcs["record_id"].str.endswith("-PROBE")]
        add("## 5. Official source reachability")
        add("")
        add("| Source | Status | URL |")
        add("|---|---|---|")
        for _, r in probes.iterrows():
            add(f"| {r['name_en']} | {r['claim'].replace('access check: ','')} "
                f"| {r['url']} |")
        add("")
        heads = srcs[srcs["tags"].str.contains("headline", na=False)]
        if not heads.empty:
            add(f"### 5.1 Scope-matching headlines captured ({len(heads)})")
            add("")
            for _, r in heads.head(40).iterrows():
                add(f"- [{r['name_en']}] {r['name_zh']} <{r['url']}>")
            add("")

    add("## 6. Data gaps, ranked")
    add("")
    add("Priority 1 blocks calibration; 2 weakens a headline claim; "
        "3 is desirable.")
    add("")
    gaps = df[(df["verify"]) | (df["evidence_grade"] == "C")].copy()
    if not gaps.empty:
        gaps["priority"] = gaps.apply(_gap_priority, axis=1)
        counts = Counter(gaps["priority"])
        for pr in (1, 2, 3):
            add(f"- Priority {pr}: {counts.get(pr, 0)} records")
        add("")
        add("The single highest-value acquisition remains an anonymously "
            "linkable chain: **farmer and plot → policy receipt → verified "
            "technology use → shock exposure → loss → recovery**. Without it "
            "the adoption and effort rules can be structurally validated but "
            "not causally calibrated.")
        add("")

    if notes.get("integrity"):
        add("## 7. Config integrity")
        add("")
        for p in notes["integrity"]:
            add(f"- PROBLEM: {p}")
        add("")
    else:
        add("## 7. Config integrity")
        add("")
        add("All cross-references between disruptions, technologies and "
            "policy instruments resolve.")
        add("")

    path = outdir / "EVIDENCE_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def summarise(records: list[EvidenceRecord]) -> str:
    df = _df(records)
    cat = Counter(df["category"])
    grade = Counter(df["evidence_grade"])
    parts = [f"{len(df)} records"]
    parts.append("by category: " + ", ".join(f"{k}={v}" for k, v in sorted(cat.items())))
    parts.append("by grade: " + ", ".join(f"{k}={grade.get(k,0)}" for k in "ABC"))
    return "; ".join(parts)
