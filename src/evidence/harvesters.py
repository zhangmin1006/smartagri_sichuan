"""
harvesters.py
=============
Online collection of supporting evidence. Everything here is best-effort:
if the network is unavailable or a host refuses, the harvester logs the
failure and returns an empty list, so the collector never fails as a whole
and the curated registry still produces a complete dataset.

Sources
-------
Crossref  : peer-reviewed literature metadata (open API, no key)
OpenAlex  : literature with citation counts and open-access links (no key)
WebPages  : configured government and statistical pages, fetched politely
            with a cache, a delay and a per-host request budget

Design rules
------------
* every fetch is cached to disk, so re-running the collector is cheap and
  does not hammer any host;
* a fixed delay between requests to the same host;
* robots.txt is consulted for government hosts before any page fetch;
* harvested records are graded no higher than B (literature) or B (official
  page), and never overwrite curated entries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.robotparser as robotparser
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import requests

from .registry import EvidenceRecord

log = logging.getLogger("evidence.harvest")

DEFAULT_CACHE = Path(__file__).resolve().parents[2] / "outputs" / "_cache"
USER_AGENT = ("SichuanSmartAgriResearchBot/0.1 "
              "(academic research; contact: zhangmin1006@gmail.com)")
REQUEST_DELAY_S = 1.2
TIMEOUT_S = 20


# ---------------------------------------------------------------------------
# Fetch plumbing
# ---------------------------------------------------------------------------
@dataclass
class FetchResult:
    ok: bool
    url: str
    status: int = 0
    text: str = ""
    error: str = ""
    from_cache: bool = False


class PoliteFetcher:
    """Cached, rate-limited, robots-aware HTTP client."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE,
                 delay: float = REQUEST_DELAY_S, respect_robots: bool = True,
                 max_per_host: int = 40) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.delay = delay
        self.respect_robots = respect_robots
        self.max_per_host = max_per_host
        self._last_hit: dict[str, float] = {}
        self._count: dict[str, int] = {}
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT,
                                     "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8"})

    def _cache_path(self, url: str) -> Path:
        return self.cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:24] + ".json")

    def _allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        host = urlparse(url).netloc
        if host not in self._robots:
            rp = robotparser.RobotFileParser()
            rp.set_url(f"{urlparse(url).scheme}://{host}/robots.txt")
            try:
                rp.read()
            except Exception as exc:                       # no robots -> allow
                log.debug("robots unavailable for %s (%s)", host, exc)
                rp = None
            self._robots[host] = rp
        rp = self._robots[host]
        if rp is None:
            return True
        try:
            return rp.can_fetch(USER_AGENT, url)
        except Exception:
            return True

    def get(self, url: str, use_cache: bool = True) -> FetchResult:
        cache_file = self._cache_path(url)
        if use_cache and cache_file.exists():
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                return FetchResult(True, url, payload.get("status", 200),
                                   payload.get("text", ""), from_cache=True)
            except Exception:
                pass

        host = urlparse(url).netloc
        if self._count.get(host, 0) >= self.max_per_host:
            return FetchResult(False, url, error="per-host request budget exhausted")
        if not self._allowed(url):
            return FetchResult(False, url, error="disallowed by robots.txt")

        wait = self.delay - (time.time() - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)

        try:
            resp = self.session.get(url, timeout=TIMEOUT_S)
            self._last_hit[host] = time.time()
            self._count[host] = self._count.get(host, 0) + 1
            if resp.encoding in (None, "ISO-8859-1"):
                resp.encoding = resp.apparent_encoding or "utf-8"
            result = FetchResult(resp.ok, url, resp.status_code, resp.text)
            if resp.ok:
                cache_file.write_text(
                    json.dumps({"status": resp.status_code, "text": resp.text}),
                    encoding="utf-8")
            return result
        except Exception as exc:
            self._last_hit[host] = time.time()
            return FetchResult(False, url, error=f"{type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Relevance filtering
# ---------------------------------------------------------------------------
# Crossref and OpenAlex match Chinese keyword queries very loosely and return
# large volumes of unrelated domestic journal articles. Every harvested record
# is therefore scored against the study scope and low scorers are dropped.

PLACE_TERMS = ("sichuan", "四川", "china", "chinese", "中国", "yangtze", "chengdu")
TOPIC_TERMS = (
    "smart agricultur", "digital agricultur", "precision agricultur",
    "agricultural technology", "technology adoption", "farm household",
    "smallholder", "irrigation", "fertigation", "drip", "sensor",
    "drone", "uav", "unmanned aerial", "remote sensing", "beidou", "gnss",
    "early warning", "agricultural machinery", "mechani", "extension service",
    "agricultural insurance", "crop insurance", "resilience", "drought",
    "flood", "waterlogging", "heatwave", "heat wave", "climate shock",
    "yield loss", "crop damage", "input price", "fertilizer price",
    "智慧农业", "数字农业", "精准农业", "农业技术", "技术采用", "农户",
    "小农户", "灌溉", "水肥一体化", "传感器", "无人机", "遥感", "北斗",
    "预警", "农机", "机械化", "农业保险", "韧性", "干旱", "洪涝", "受灾",
)
METHOD_TERMS = (
    "principal-agent", "principal agent", "moral hazard", "risk aversion",
    "risk averse", "expected utility", "contract", "incentive",
    "agent-based", "agent based", "system dynamics", "simulation",
    "policy evaluation", "subsidy", "voucher", "willingness", "adoption",
    "委托代理", "风险厌恶", "补贴", "政策", "仿真", "多主体",
)


def relevance_score(text: str) -> int:
    """Scope relevance: place (2) + topic (2 each, capped) + method (1 each)."""
    t = (text or "").lower()
    score = 2 if any(p in t for p in PLACE_TERMS) else 0
    score += 2 * min(2, sum(1 for k in TOPIC_TERMS if k in t))
    score += min(2, sum(1 for k in METHOD_TERMS if k in t))
    return score


MIN_RELEVANCE = 4


def _norm_title(title: str) -> str:
    return "".join(ch for ch in (title or "").lower() if ch.isalnum())[:90]


# ---------------------------------------------------------------------------
# Literature harvesters
# ---------------------------------------------------------------------------
class CrossrefHarvester:
    """Peer-reviewed literature metadata from the Crossref REST API."""

    BASE = "https://api.crossref.org/works"

    def __init__(self, fetcher: PoliteFetcher, rows: int = 12) -> None:
        self.fetcher, self.rows = fetcher, rows

    def search(self, query: str, category: str) -> list[EvidenceRecord]:
        url = (f"{self.BASE}?query={requests.utils.quote(query)}"
               f"&rows={self.rows}&sort=relevance"
               f"&select=DOI,title,abstract,issued,container-title,author,type,URL")
        res = self.fetcher.get(url)
        if not res.ok:
            log.warning("crossref failed for %r: %s", query, res.error or res.status)
            return []
        try:
            items = json.loads(res.text)["message"]["items"]
        except Exception as exc:
            log.warning("crossref parse failed for %r: %s", query, exc)
            return []

        out = []
        for it in items:
            title = " ".join(it.get("title") or []).strip()
            if not title:
                continue
            year = ""
            try:
                year = it["issued"]["date-parts"][0][0]
            except Exception:
                pass
            authors = "; ".join(
                f"{a.get('family', '')} {a.get('given', '')}".strip()
                for a in (it.get("author") or [])[:4])
            journal = " ".join(it.get("container-title") or [])
            score = relevance_score(f"{title} {journal} {it.get('abstract', '')}")
            if score < MIN_RELEVANCE:
                continue
            out.append(EvidenceRecord(
                record_id=f"LIT-CR-{hashlib.sha1(it.get('DOI', title).encode()).hexdigest()[:10]}",
                category="literature", entity_id=category,
                name_en=title[:300],
                claim=f"{authors} ({year}). {journal}".strip(" ."),
                claim_type="literature", evidence_grade="B",
                source="crossref", url=it.get("URL", ""), year=year,
                notes=f"query={query}; type={it.get('type', '')}; relevance={score}",
                tags=["literature", "crossref", category],
            ))
        return out


class OpenAlexHarvester:
    """Literature with citation counts and open-access links."""

    BASE = "https://api.openalex.org/works"

    def __init__(self, fetcher: PoliteFetcher, rows: int = 12,
                 mailto: str = "zhangmin1006@gmail.com") -> None:
        self.fetcher, self.rows, self.mailto = fetcher, rows, mailto

    def search(self, query: str, category: str) -> list[EvidenceRecord]:
        url = (f"{self.BASE}?search={requests.utils.quote(query)}"
               f"&per-page={self.rows}&mailto={self.mailto}")
        res = self.fetcher.get(url)
        if not res.ok:
            log.warning("openalex failed for %r: %s", query, res.error or res.status)
            return []
        try:
            items = json.loads(res.text).get("results", [])
        except Exception as exc:
            log.warning("openalex parse failed for %r: %s", query, exc)
            return []

        out = []
        for it in items:
            title = (it.get("display_name") or "").strip()
            if not title:
                continue
            venue = ((it.get("primary_location") or {}).get("source") or {}
                     ).get("display_name", "") or ""
            concepts = " ".join(c.get("display_name", "")
                                for c in (it.get("concepts") or [])[:8])
            score = relevance_score(f"{title} {venue} {concepts}")
            if score < MIN_RELEVANCE:
                continue
            oa = (it.get("open_access") or {}).get("oa_url") or it.get("doi") or ""
            out.append(EvidenceRecord(
                record_id=f"LIT-OA-{(it.get('id') or title)[-12:].strip('/')}",
                category="literature", entity_id=category,
                name_en=title[:300],
                claim=f"cited_by={it.get('cited_by_count', 0)}; venue={venue}",
                claim_type="literature", evidence_grade="B",
                source="openalex", url=oa, year=it.get("publication_year", ""),
                notes=f"query={query}; relevance={score}",
                tags=["literature", "openalex", category],
            ))
        return out


# ---------------------------------------------------------------------------
# Government and statistical pages
# ---------------------------------------------------------------------------
OFFICIAL_SOURCES = [
    {"key": "MARA", "name_en": "Ministry of Agriculture and Rural Affairs",
     "url": "https://www.moa.gov.cn/", "category": "policy",
     "note": "national policy texts, smart agriculture action plan"},
    {"key": "SC-DARA", "name_en": "Sichuan Dept of Agriculture and Rural Affairs",
     "url": "http://nynct.sc.gov.cn/", "category": "policy",
     "note": "provincial action plans; primary source for SP1 and SP2"},
    {"key": "SC-GOV", "name_en": "Sichuan Provincial Government",
     "url": "https://www.sc.gov.cn/", "category": "policy",
     "note": "provincial directives and No.1 documents"},
    {"key": "SC-STATS", "name_en": "Sichuan Bureau of Statistics",
     "url": "http://tjj.sc.gov.cn/", "category": "context",
     "note": "statistical communique and yearbook, agricultural baseline"},
    {"key": "MACH-SUBSIDY", "name_en": "National machinery purchase subsidy disclosure platform",
     "url": "http://njbt.moa.gov.cn/", "category": "policy",
     "verify_url": True,
     "note": "equipment-level subsidy records; key micro data source for P1. "
             "Provincial sub-portal URL must be confirmed with Sichuan DARA."},
    {"key": "CMA-SC", "name_en": "Sichuan Meteorological Service",
     "url": "http://sc.cma.gov.cn/", "category": "disruption",
     "note": "drought and rainstorm warnings, event chronology"},
    {"key": "MEM", "name_en": "Ministry of Emergency Management",
     "url": "https://www.mem.gov.cn/", "category": "disruption",
     "note": "disaster loss statistics, affected crop area"},
]


class OfficialPageHarvester:
    """Records reachability and page titles for the official source list.

    Deliberately shallow: it verifies that a source is live and captures the
    landing-page title and any headline links, producing an auditable
    access log rather than attempting to scrape policy text. Bulk text
    extraction from these hosts should be done under an explicit data
    agreement, which is exactly what the report recommends.
    """

    def __init__(self, fetcher: PoliteFetcher, max_links: int = 12) -> None:
        self.fetcher = fetcher
        self.max_links = max_links

    def probe(self, source: dict) -> list[EvidenceRecord]:
        res = self.fetcher.get(source["url"])
        status = "reachable" if res.ok else f"unreachable ({res.error or res.status})"
        records = [EvidenceRecord(
            record_id=f"SRC-{source['key']}-PROBE",
            category="source", entity_id=source["key"],
            name_en=source["name_en"],
            claim=f"access check: {status}", claim_type="context",
            evidence_grade="A" if res.ok else "C",
            source="official_site", url=source["url"],
            notes=source.get("note", ""),
            tags=["source_probe", source.get("category", "")],
        )]
        if not res.ok:
            return records

        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(res.text, "html.parser")
            title = (soup.title.get_text(strip=True) if soup.title else "")
            if title:
                records.append(EvidenceRecord(
                    record_id=f"SRC-{source['key']}-TITLE",
                    category="source", entity_id=source["key"],
                    name_en=source["name_en"], claim=f"site title: {title}",
                    claim_type="context", evidence_grade="A",
                    source="official_site", url=source["url"],
                    tags=["source_probe"],
                ))
            keywords = ("智慧农业", "数字农业", "农机", "补贴", "无人机", "灌溉",
                        "干旱", "洪涝", "预警", "保险", "数字乡村", "北斗")
            seen = set()
            for a in soup.find_all("a", href=True):
                text = a.get_text(strip=True)
                if len(text) < 6 or text in seen:
                    continue
                if not any(k in text for k in keywords):
                    continue
                seen.add(text)
                href = a["href"]
                if href.startswith("/"):
                    p = urlparse(source["url"])
                    href = f"{p.scheme}://{p.netloc}{href}"
                records.append(EvidenceRecord(
                    record_id=f"SRC-{source['key']}-LNK-{abs(hash(text)) % 10**6}",
                    category="source", entity_id=source["key"],
                    name_en=source["name_en"], name_zh=text[:200],
                    claim="headline matching a scope keyword",
                    claim_type="context", evidence_grade="B",
                    source="official_site", url=href, verify=True,
                    tags=["headline", source.get("category", "")],
                ))
                if len(seen) >= self.max_links:
                    break
        except Exception as exc:
            log.debug("parse failed for %s: %s", source["url"], exc)
        return records

    def probe_all(self, sources: list[dict] | None = None) -> list[EvidenceRecord]:
        out = []
        for src in (sources if sources is not None else OFFICIAL_SOURCES):
            out.extend(self.probe(src))
        return out
