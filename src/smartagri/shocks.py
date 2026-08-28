"""
shocks.py
=========
Generates the external shocks the model exposes farmers to, from the scope
decisions in config/disruptions.yaml.

Design points
-------------
* Shocks are CORRELATED across farmers within a season. That is the whole
  point: a shock that hits everyone at once is what creates the service
  queue, and the queue is what makes a nominal adopter fail to benefit.
  Independent per-farmer shocks would hide the central mechanism.
* D3 (compound heat-drought with power and network interruption) carries a
  `tech_availability` multiplier below 1, so technology efficacy degrades
  exactly when demand for it peaks.
* D4 (input price) does not damage yield at all: it acts on cash and, through
  the liquidity channel, raises the farmer's effective risk aversion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from scipy.special import ndtr as _ndtr

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


@dataclass
class ShockEvent:
    """One realised shock in one season."""

    shock_id: str
    name_en: str
    family: str
    severity: float                 # 0..1
    tech_availability: float = 1.0  # rho multiplier during the event
    price_multiplier: float = 1.0   # D4 only
    crop_stage: str = "tasselling"
    county_severity: dict = field(default_factory=dict)
    warning_lead_days: float = 0.0
    action_window_days: float = 2.0

    @property
    def is_economic(self) -> bool:
        return self.family == "economic"

    def severity_for(self, county_id: str) -> float:
        return float(self.county_severity.get(county_id, self.severity))


def annual_to_period_probability(annual_probability: float,
                                 periods_per_year: int) -> float:
    """Convert an annual occurrence probability to one model period.

    The conversion preserves the stated annual probability under independent
    within-year periods: ``1 - (1 - p_period) ** periods_per_year = p_annual``.
    """
    p = float(np.clip(annual_probability, 0.0, 1.0))
    n = max(int(periods_per_year), 1)
    return float(1.0 - (1.0 - p) ** (1.0 / n))


class ShockGenerator:
    """Draws seasonal shock events with spatial correlation."""

    def __init__(self, rng: np.random.Generator,
                 config_dir: Path | str = CONFIG_DIR,
                 counties: list[dict] | None = None,
                 spatial_correlation: float = 0.65,
                 seasons_per_year: int = 1) -> None:
        self.rng = rng
        with (Path(config_dir) / "disruptions.yaml").open(encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)
        self.specs = {d["id"]: d for d in cfg.get("tier1", [])}
        self.counties = counties or []
        self.rho_spatial = spatial_correlation
        self.seasons_per_year = max(int(seasons_per_year), 1)

    # ------------------------------------------------------------------
    def _draw_severity(self, spec: dict) -> float:
        m = spec.get("model", {})
        kind = m.get("severity_distribution", "beta")
        params = m.get("severity_params", [2.0, 3.0])
        if kind == "beta":
            return float(self.rng.beta(params[0], params[1]))
        if kind == "lognormal":
            return float(np.clip(self.rng.lognormal(params[0], params[1]) - 1.0,
                                 0.0, 1.0))
        return float(self.rng.uniform(0.0, 1.0))

    def _county_field(self, base: float, spec: dict) -> dict:
        """Correlated county-level severities around the provincial draw."""
        out = {}
        tm = spec.get("model", {}).get("terrain_multiplier", {})
        for c in self.counties:
            idio = self.rng.normal(0.0, 0.18)
            sev = self.rho_spatial * base + (1 - self.rho_spatial) * (base + idio)
            sev *= float(tm.get(c.get("terrain", "plain"), 1.0))
            out[c["id"]] = float(np.clip(sev, 0.0, 1.0))
        return out

    # ------------------------------------------------------------------
    def _copula_field(self, spec: dict, base_sev: float,
                      force_all: bool = False) -> dict:
        """Correlated county occurrence via a Gaussian copula.

        Calibration from 34 years of ERA5 reanalysis gives, per hazard, both a
        county-specific occurrence probability and a spatial correlation. A
        single provincial Bernoulli with a shared severity cannot reproduce
        either: it forces every hazard to have the same geographic footprint.
        The measured footprints differ sharply -- rainstorm-flood is local
        (correlation 0.19) while compound heat-drought is province-wide (0.54)
        -- and that difference drives how much simultaneous demand the service
        queue has to absorb.

        Construction: a common factor u and idiosyncratic terms e_c give
            z_c = sqrt(rho)*u + sqrt(1-rho)*e_c,     z_c ~ N(0,1)
        and county c is hit when Phi(z_c) <= p_c. This reproduces the measured
        marginal probabilities and the measured correlation exactly.
        """
        m = spec.get("model", {})
        pby = m.get("probability_by_county") or {}
        rho = float(m.get("spatial_correlation", self.rho_spatial))
        rho = float(np.clip(rho, 0.0, 0.99))
        tm = m.get("terrain_multiplier", {})

        u = self.rng.normal()
        out = {}
        for c in self.counties:
            cid = c["id"]
            p_annual = float(pby.get(
                cid, m.get("annual_probability_prior", 0.0)))
            p_c = annual_to_period_probability(
                p_annual, self.seasons_per_year)
            if force_all:
                hit = p_c > 0.0
            else:
                z = np.sqrt(rho) * u + np.sqrt(1.0 - rho) * self.rng.normal()
                hit = _ndtr(z) <= p_c
            if not hit:
                out[cid] = 0.0
                continue
            # severity varies around the provincial draw, then terrain-adjusted
            sev = base_sev * float(np.clip(self.rng.normal(1.0, 0.20), 0.3, 1.7))
            sev *= float(tm.get(c.get("terrain", "plain"), 1.0))
            out[cid] = float(np.clip(sev, 0.0, 1.0))
        return out

    # ------------------------------------------------------------------
    def draw_season(self, season: int, month: int,
                    force: list[str] | None = None) -> list[ShockEvent]:
        """Return the shocks realised in this season.

        `force` overrides the stochastic draw, which is what the extreme
        condition tests and the fixed-event backtests use.
        """
        events: list[ShockEvent] = []
        for sid, spec in self.specs.items():
            m = spec.get("model", {})
            in_window = month in (spec.get("season_window") or list(range(1, 13)))
            forced = force is not None and sid in force
            if force is not None and not forced:
                continue
            if force is None and not in_window:
                continue

            sev = self._draw_severity(spec)
            # county occurrence and severity from the calibrated copula; the
            # hazard is realised this season only if at least one county is hit
            field = self._copula_field(spec, sev, force_all=forced)
            if not any(v > 0.0 for v in field.values()):
                continue
            stages = spec.get("crop_stages_at_risk") or ["tasselling"]
            lead = 0.0
            if m.get("warning_lead_time_days"):
                lo, hi = m["warning_lead_time_days"]
                lead = float(self.rng.uniform(lo, hi))

            ev = ShockEvent(
                shock_id=sid,
                name_en=spec.get("name_en", sid),
                family=spec.get("family", ""),
                severity=float(np.mean([v for v in field.values() if v > 0])),
                tech_availability=float(m.get("tech_availability_multiplier", 1.0)),
                crop_stage=str(self.rng.choice(stages)),
                county_severity=field,
                warning_lead_days=lead,
                action_window_days=float(m.get("action_window_days", 2.0)),
            )
            if spec.get("family") == "economic":
                lo, hi = m.get("price_multiplier_range", [1.0, 1.0])
                ev.price_multiplier = float(lo + (hi - lo) * sev)
            events.append(ev)
        return events


# ---------------------------------------------------------------------------
def damage_fraction(event: ShockEvent, county_id: str, terrain: str,
                    stage_vulnerability: dict, spec_model: dict,
                    irrigation_share: float = 0.0) -> float:
    """D(H, stage, terrain): fraction of base yield at risk BEFORE mitigation."""
    if event.is_economic:
        return 0.0
    sev = event.severity_for(county_id)
    dmax = float(spec_model.get("yield_damage_max", 0.4))
    stage_v = float(stage_vulnerability.get(event.crop_stage, 1.0))
    dmg = dmax * (sev ** 1.25) * stage_v
    # existing irrigation already offsets part of drought damage
    offset = float(spec_model.get("irrigation_offset", 0.0)) * irrigation_share
    return float(np.clip(dmg * (1.0 - offset), 0.0, 1.0))
