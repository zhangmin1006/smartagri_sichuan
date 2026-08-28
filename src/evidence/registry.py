"""
registry.py
===========
Loads the curated scope decisions (disruptions, technologies, policies) from
config/ and turns them into flat, exportable evidence records.

The registry is the OFFLINE backbone of the collector: it always produces a
complete, citable dataset even with no network. Online harvesting adds
records on top of it and never silently overwrites a curated entry.

Every record carries:
    evidence_grade  A / B / C     (strength of the underlying source)
    claim_type      target | baseline | mechanism | parameter | context
    verify          whether the record still needs primary-source checking

The distinction that matters most is claim_type == "target": provincial
action-plan figures are POLICY COMMITMENTS, not measured adoption. The
exporter refuses to emit them as adoption baselines.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"

EVIDENCE_GRADES = {"A", "B", "C"}
CLAIM_TYPES = {"target", "baseline", "mechanism", "parameter", "context",
               "literature", "policy_document"}


@dataclass
class EvidenceRecord:
    """One atomic, exportable piece of evidence."""

    record_id: str
    category: str                      # disruption | technology | policy | literature
    entity_id: str                     # D1, T2, P3, SP1 ...
    name_en: str
    name_zh: str = ""
    claim: str = ""
    claim_type: str = "context"
    value: Any = None
    unit: str = ""
    evidence_grade: str = "C"
    source: str = "curated:config"
    url: str = ""
    year: Any = ""
    verify: bool = False
    notes: str = ""
    tags: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.evidence_grade not in EVIDENCE_GRADES:
            raise ValueError(f"bad evidence grade {self.evidence_grade!r}")
        if self.claim_type not in CLAIM_TYPES:
            raise ValueError(f"bad claim type {self.claim_type!r}")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


class Registry:
    """Curated knowledge base assembled from the YAML scope decisions."""

    def __init__(self, config_dir: Path | str = CONFIG_DIR) -> None:
        self.config_dir = Path(config_dir)
        self.disruptions = self._load("disruptions.yaml")
        self.technologies = self._load("technologies.yaml")
        self.policies = self._load("policies.yaml")
        self.params = self._load("model_params.yaml")

    def _load(self, name: str) -> dict:
        path = self.config_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing config file: {path}")
        with path.open(encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    # ------------------------------------------------------------------
    # Record builders
    # ------------------------------------------------------------------
    def disruption_records(self) -> list[EvidenceRecord]:
        out: list[EvidenceRecord] = []
        for tier, key in (("tier1", "tier1"), ("tier2", "tier2_extensions"),
                          ("excluded", "excluded")):
            for item in self.disruptions.get(key, []) or []:
                did = item["id"]
                grade = item.get("evidence_grade", "C")
                out.append(EvidenceRecord(
                    record_id=f"DIS-{did}-SCOPE",
                    category="disruption", entity_id=did,
                    name_en=item.get("name_en", ""), name_zh=item.get("name_zh", ""),
                    claim=(item.get("why_included") or item.get("why_deferred")
                           or item.get("reason", "")).strip(),
                    claim_type="mechanism", evidence_grade=grade,
                    notes=f"tier={tier}",
                    tags=["scope_decision", tier] + item.get("coupled_technologies", []),
                ))
                if tier != "tier1":
                    continue
                out.append(EvidenceRecord(
                    record_id=f"DIS-{did}-CHANNEL",
                    category="disruption", entity_id=did,
                    name_en=item.get("name_en", ""), name_zh=item.get("name_zh", ""),
                    claim=item.get("primary_loss_channel", "").strip(),
                    claim_type="mechanism", evidence_grade=grade,
                    tags=["loss_channel"],
                ))
                # Parameters derived from the ERA5 calibration inherit the
                # hazard's evidence grade and are no longer flagged for
                # verification; the rest remain grade C priors.
                calibrated = {"annual_probability_prior", "severity_params",
                              "spatial_correlation", "probability_by_county",
                              "n_events_observed", "hazard_definition"}
                has_cal = bool(self.disruptions.get("calibration"))
                for pkey, pval in (item.get("model") or {}).items():
                    is_cal = has_cal and pkey in calibrated
                    out.append(EvidenceRecord(
                        record_id=f"DIS-{did}-PAR-{pkey}",
                        category="disruption", entity_id=did,
                        name_en=item.get("name_en", ""),
                        claim=f"{'calibrated' if is_cal else 'model prior'}: {pkey}",
                        claim_type="parameter",
                        value=pval,
                        evidence_grade=(grade if is_cal else "C"),
                        verify=(not is_cal),
                        notes=("derived from ERA5 reanalysis 1991-2024"
                               if is_cal else
                               "prior for calibration, not a measurement"),
                        tags=["parameter", "calibrated" if is_cal else "prior"],
                    ))
                for need in item.get("data_needed", []) or []:
                    out.append(EvidenceRecord(
                        record_id=f"DIS-{did}-DATA-{abs(hash(need)) % 10**6}",
                        category="disruption", entity_id=did,
                        name_en=item.get("name_en", ""), claim=need,
                        claim_type="context", evidence_grade="C",
                        tags=["data_requirement"],
                    ))
        return out

    def technology_records(self) -> list[EvidenceRecord]:
        out: list[EvidenceRecord] = []
        for item in self.technologies.get("bundles", []) or []:
            tid = item["id"]
            grade = item.get("evidence_grade", "C")
            status = item.get("status", "in_scope")
            out.append(EvidenceRecord(
                record_id=f"TEC-{tid}-SCOPE",
                category="technology", entity_id=tid,
                name_en=item.get("name_en", ""), name_zh=item.get("name_zh", ""),
                claim=(item.get("resilience_mechanism") or item.get("note", "")).strip(),
                claim_type="mechanism", evidence_grade=grade,
                notes=f"status={status}; channel={item.get('channel', '')}",
                tags=["scope_decision", status] + item.get("access_modes", []),
            ))
            if item.get("sichuan_anchor"):
                out.append(EvidenceRecord(
                    record_id=f"TEC-{tid}-ANCHOR",
                    category="technology", entity_id=tid,
                    name_en=item.get("name_en", ""),
                    claim=item["sichuan_anchor"].strip(),
                    claim_type="context", evidence_grade=grade, verify=True,
                    source="curated:provincial action plan and statistics",
                    tags=["sichuan_anchor"],
                ))
            for pkey, pval in (item.get("model") or {}).items():
                out.append(EvidenceRecord(
                    record_id=f"TEC-{tid}-PAR-{pkey}",
                    category="technology", entity_id=tid,
                    name_en=item.get("name_en", ""),
                    claim=f"model prior: {pkey}", claim_type="parameter",
                    value=pval, evidence_grade="C", verify=True,
                    notes="prior for calibration, not a measurement",
                    tags=["parameter", "prior"],
                ))
        for st in self.technologies.get("adoption_states", []) or []:
            out.append(EvidenceRecord(
                record_id=f"TEC-STATE-{st['id']}",
                category="technology", entity_id="STATE",
                name_en=st.get("label_en", ""),
                claim=f"adoption state {st['id']}: {st['key']}",
                claim_type="context", evidence_grade="A",
                notes=self.technologies.get("state_warning", "").strip(),
                tags=["adoption_ladder"],
            ))
        return out

    def policy_records(self) -> list[EvidenceRecord]:
        out: list[EvidenceRecord] = []
        for scope in ("national", "provincial"):
            for doc in self.policies.get(scope, []) or []:
                key = doc["key"]
                out.append(EvidenceRecord(
                    record_id=f"POL-{key}-DOC",
                    category="policy", entity_id=key,
                    name_en=doc.get("title_en", ""), name_zh=doc.get("title_zh", ""),
                    claim=doc.get("role_in_model", doc.get("note", "")).strip(),
                    claim_type="policy_document",
                    evidence_grade=doc.get("evidence_grade", "C"),
                    year=doc.get("year", ""), verify=bool(doc.get("verify", False)),
                    notes=f"scope={scope}; ref={doc.get('reference', '')}",
                    tags=["policy_document", scope],
                ))
                if doc.get("key_target"):
                    out.append(EvidenceRecord(
                        record_id=f"POL-{key}-KT",
                        category="policy", entity_id=key,
                        name_en=doc.get("title_en", ""),
                        claim=doc["key_target"].strip(), claim_type="target",
                        evidence_grade=doc.get("evidence_grade", "C"),
                        year=doc.get("year", ""), tags=["target"],
                    ))
                for block, ctype in (("targets_by_2028", "target"),
                                     ("baseline_2025", "baseline")):
                    for tkey, tval in (doc.get(block) or {}).items():
                        out.append(EvidenceRecord(
                            record_id=f"POL-{key}-{block}-{tkey}",
                            category="policy", entity_id=key,
                            name_en=doc.get("title_en", ""),
                            claim=f"{block}: {tkey}", claim_type=ctype, value=tval,
                            evidence_grade=doc.get("evidence_grade", "C"),
                            year=doc.get("year", ""),
                            verify=bool(doc.get("verify", False)),
                            notes=("policy commitment, NOT realised adoption"
                                   if ctype == "target" else "stated starting point"),
                            tags=["target" if ctype == "target" else "baseline"],
                        ))
        for inst in self.policies.get("instruments", []) or []:
            iid = inst["id"]
            out.append(EvidenceRecord(
                record_id=f"POL-{iid}-INST",
                category="policy", entity_id=iid,
                name_en=inst.get("name_en", ""), name_zh=inst.get("name_zh", ""),
                claim=inst.get("abm_mechanism", "").strip(),
                claim_type="mechanism", evidence_grade="C",
                notes=f"targets={inst.get('targets', [])}; "
                      f"sd_stock={inst.get('sd_stock', '')}; "
                      f"equity={inst.get('equity_flag', '')}",
                tags=["instrument"] + list(inst.get("targets", [])),
            ))
            for dkey, dval in (inst.get("decision_variables") or {}).items():
                out.append(EvidenceRecord(
                    record_id=f"POL-{iid}-DV-{dkey}",
                    category="policy", entity_id=iid,
                    name_en=inst.get("name_en", ""),
                    claim=f"decision variable: {dkey}", claim_type="parameter",
                    value=dval, evidence_grade="C", verify=True,
                    tags=["decision_variable"],
                ))
            for se in inst.get("known_side_effects", []) or []:
                out.append(EvidenceRecord(
                    record_id=f"POL-{iid}-SE-{abs(hash(se)) % 10**6}",
                    category="policy", entity_id=iid,
                    name_en=inst.get("name_en", ""), claim=se,
                    claim_type="mechanism", evidence_grade="C",
                    tags=["side_effect"],
                ))
        return out

    # ------------------------------------------------------------------
    def all_records(self) -> list[EvidenceRecord]:
        return (self.disruption_records() + self.technology_records()
                + self.policy_records())

    # ------------------------------------------------------------------
    def search_queries(self) -> dict[str, list[str]]:
        """Assemble the query sets the online harvesters will run.

        High-precision queries only. Broad Chinese-language keyword queries
        against Crossref and OpenAlex return large volumes of loosely matched
        domestic journal articles, so queries here always pin BOTH a topic and
        a place or method term, and the harvester applies a relevance filter
        on top.
        """
        q: dict[str, list[str]] = {
            "disruption": [
                "drought agricultural loss Sichuan Basin China",
                "flood waterlogging crop damage China remote sensing assessment",
                "2022 Yangtze heatwave drought agriculture impact China",
                "compound climate extremes agriculture power supply China",
                "fertilizer price shock farm input cost China smallholder",
                "climate shock farm household resilience recovery China",
            ],
            "technology": [
                "digital agriculture technology adoption Sichuan farmers",
                "smart agriculture adoption smallholder China determinants",
                "agricultural drone spraying plant protection service China",
                "water fertilizer integration drip irrigation adoption China",
                "BeiDou agricultural machinery precision operation China",
                "agricultural early warning information service farmer response China",
                "agricultural machinery socialized service outsourcing China",
                "precision agriculture technology effective use intensity China",
            ],
            "policy": [
                "agricultural machinery purchase subsidy China policy evaluation",
                "digital village policy China rural digital infrastructure",
                "smart agriculture policy China action plan implementation",
                "agricultural insurance remote sensing loss assessment China",
                "service voucher agricultural extension subsidy developing country",
                "agricultural technology subsidy targeting smallholder equity",
            ],
            "theory": [
                "principal agent model risk averse agent optimal effort contract",
                "risk aversion farmer technology adoption expected utility",
                "agricultural service contract moral hazard monitoring verifiability",
                "agent based model system dynamics hybrid policy simulation agriculture",
                "resilience metrics agricultural household shock recovery modelling",
            ],
        }
        # Verification queries attached to individual disruption records.
        for item in self.disruptions.get("tier1", []) or []:
            q["disruption"].extend(item.get("verify_queries", []) or [])
        for k in q:
            seen, uniq = set(), []
            for item in q[k]:
                item = (item or "").strip()
                if item and item not in seen:
                    seen.add(item)
                    uniq.append(item)
            q[k] = uniq
        return q

    # ------------------------------------------------------------------
    def integrity_check(self) -> list[str]:
        """Cross-file consistency: every referenced id must exist."""
        problems: list[str] = []
        tech_ids = {t["id"] for t in self.technologies.get("bundles", []) or []}
        pol_ids = {p["id"] for p in self.policies.get("instruments", []) or []}
        for item in self.disruptions.get("tier1", []) or []:
            for t in item.get("coupled_technologies", []) or []:
                if t not in tech_ids:
                    problems.append(
                        f"{item['id']} couples unknown technology {t}")
        for item in self.technologies.get("bundles", []) or []:
            for p in item.get("policy_levers", []) or []:
                if p not in pol_ids:
                    problems.append(
                        f"{item['id']} references unknown instrument {p}")
        for inst in self.policies.get("instruments", []) or []:
            for t in inst.get("targets", []) or []:
                if t not in tech_ids:
                    problems.append(
                        f"{inst['id']} targets unknown technology {t}")
        return problems


def records_to_rows(records: Iterable[EvidenceRecord]) -> list[dict]:
    rows = []
    for r in records:
        d = r.as_dict()
        d["tags"] = "|".join(str(t) for t in d.get("tags", []))
        if isinstance(d.get("value"), (dict, list)):
            d["value"] = str(d["value"])
        rows.append(d)
    return rows
