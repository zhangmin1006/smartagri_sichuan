"""
model.py
========
The coupled ABM + SD model.

Season loop
-----------
  1  SD -> ABM      budget, capacity, reliability, trust, training slots
  2  government     translate instruments into the terms farmers face
  3  adoption       each farmer chooses technology and access mode (RQ1a)
  4  shocks/dispatch correlated shocks realise; warned demand is matched locally
  5  contracting    for each matched pair solve Guo-Parlar-Zhang for e* (RQ1b)
  6  consequences   technology availability and infrastructure may drop
  7  outcomes       loss, mitigation, insurance, income, recovery (RQ2)
  8  learning       beliefs, trust, exit
  9  ABM -> SD      expenditure, demand, failures, effective use (RQ3)

Everything the three research questions need is recorded per season.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .agents import (STATE_ACQUIRED, STATE_EFFECTIVE, STATE_EXITED, STATE_USED,
                     TECHS, Cooperative, Farmer, Government, ServiceProvider)
from .shocks import (ShockGenerator, annual_to_period_probability,
                     damage_fraction)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"


def load_configs(config_dir: Path | str = CONFIG_DIR) -> dict:
    cd = Path(config_dir)
    out = {}
    for key, name in (("params", "model_params.yaml"),
                      ("tech", "technologies.yaml"),
                      ("disrupt", "disruptions.yaml"),
                      ("policy", "policies.yaml")):
        with (cd / name).open(encoding="utf-8") as fh:
            out[key] = yaml.safe_load(fh)
    return out


# ===========================================================================
@dataclass
class SeasonRecord:
    season: int
    year: float
    shocks: list
    adoption_rate: dict
    service_share: dict
    effective_use_rate: float
    mean_effort: dict
    contract_mix: dict
    mean_wait_days: float
    backlog_mu: float
    mean_loss_fraction: float
    avoided_loss_fraction: float
    mitigation_rate: float
    mean_income: float
    income_p10: float
    gini_income: float
    exit_rate: float
    fiscal_spend: float
    fiscal_cumulative: float
    equity_gap: float
    mountain_gap: float
    capacity_units: float
    reliability: float
    trust: float
    capable_share: float
    recovery_seasons_mean: float = float("nan")
    extras: dict = field(default_factory=dict)


def _require_softmax(rule: str) -> str:
    if str(rule).lower() != "softmax":
        raise ValueError(
            f"behaviour.choice_rule={rule!r} is configured but only 'softmax' "
            "is implemented. Implement the rule or correct the configuration.")
    return "softmax"


class SmartAgriModel:
    """Sichuan smart-agriculture adoption, resilience and policy model."""

    def __init__(self, scenario: str = "baseline", seed: int | None = None,
                 config_dir: Path | str = CONFIG_DIR,
                 instruments: dict | None = None,
                 n_farmers: int | None = None,
                 seasons: int | None = None) -> None:
        self.cfg = load_configs(config_dir)
        p = self.cfg["params"]
        self.seed = int(p["meta"]["seed"] if seed is None else seed)
        # Independent random streams keep policy comparisons paired.  A
        # policy that changes the number of behavioural draws must not also
        # change the weather path or every scenario contrast mixes policy
        # effects with random-number consumption.
        streams = np.random.SeedSequence(self.seed).spawn(3)
        self.rng = np.random.default_rng(streams[0])
        self.shock_rng = np.random.default_rng(streams[1])
        self.outcome_rng = np.random.default_rng(streams[2])
        self.scenario = scenario
        self.seasons_total = seasons or int(p["meta"]["seasons"])
        self.seasons_per_year = int(p["meta"]["seasons_per_year"])

        self.tech_spec = {b["id"]: b for b in self.cfg["tech"]["bundles"]}
        self.disrupt_spec = {d["id"]: d for d in self.cfg["disrupt"]["tier1"]}

        # policy instruments for this scenario
        if instruments is None:
            sc = {s["key"]: s for s in p["scenarios"]}
            instruments = dict(sc.get(scenario, {}).get("instruments", {}) or {})
        # A scenario names the instruments it switches ON and may override some
        # of their settings; any setting it does not name falls back to the
        # documented default in policies.yaml. Instruments the scenario does
        # NOT name stay off entirely. Without this merge a scenario that sets
        # only the premium subsidy leaves coverage at zero and the instrument
        # silently does nothing.
        self.instrument_defaults = {
            i["id"]: {k: v.get("default") for k, v in
                      (i.get("decision_variables") or {}).items()}
            for i in self.cfg["policy"].get("instruments", [])}
        merged = {}
        for iid, overrides in instruments.items():
            base = dict(self.instrument_defaults.get(iid, {}))
            base.update(overrides or {})
            merged[iid] = base
        self.instruments = merged
        self.gov = Government(c_risk=p["risk_attitude"]["government"]["c"],
                              instruments=self.instruments)

        from .sd import SDState
        self.sd = SDState.from_config(p)

        self.counties = {c["id"]: c for c in p["population"]["counties"]}
        self._build_population(n_farmers)

        # Scale every capacity-like stock to the simulated population so that
        # runs at different n_farmers remain comparable. Without this, halving
        # the population silently doubles per-capita service capacity and the
        # queue mechanism -- the core of RQ3 -- disappears.
        cfg_n = sum(c["n_farmers"] for c in self.counties.values())
        self.pop_scale = len(self.farmers) / max(cfg_n, 1)
        self.sd.budget *= self.pop_scale
        self.sd.budget_inflow_annual *= self.pop_scale
        self.sd.training_throughput *= self.pop_scale

        # Service capacity is set from a coverage ratio on total modelled area
        # so that the queue binds at a realistic utilisation. An absolute unit
        # count calibrated for a province leaves a 500-farmer run with 50x more
        # capacity than demand, which silently removes the congestion feedback.
        self.total_area_mu = sum(f.area_mu for f in self.farmers)
        coverage = float(p["sd"]["stocks"]["service_capacity"].get(
            "initial_capacity_coverage", 0.55))
        per_unit = (self.sd.unit_capacity_mu_per_day * 12.0
                    * self.sd.utilisation_ceiling)
        # NOTE: capacity is deliberately continuous, not floored at one whole
        # unit. A max(1.0, ...) floor silently inflates capacity by up to 4x
        # for populations below ~460 farmers, which disables the queue and
        # makes the coverage ratio inert. Fractional units are the correct
        # reading here: one unit is a capacity quantum, not an indivisible
        # machine, and the simulated population is itself a sample.
        self.sd.capacity_units = max(
            1e-3, coverage * self.total_area_mu / max(per_unit, 1e-9))
        # Providers hold fractional units for the same reason the SD stock
        # does: rounding each of a dozen providers up to one whole unit
        # reconstitutes the capacity inflation the fix above removes, and
        # makes the reported per-provider utilisation disagree with the
        # capacity the queue is actually allocating from.
        self._sync_provider_capacity()
        self.shockgen = ShockGenerator(
            self.shock_rng, config_dir, list(self.counties.values()),
            seasons_per_year=self.seasons_per_year)
        self.records: list[SeasonRecord] = []
        self._baseline_income: float | None = None
        self._non_power_share = float(
            self.cfg["params"]["loss"].get("non_power_outage_share", 1.0))
        _bu = self.cfg["params"]["behaviour"]["belief_update"]
        self._hazard_lr = float(_bu.get("learning_rate_hazard", 0.15))
        self._hazard_prior = -1.0   # set once _context has been built
        _ss = self.cfg["params"]["risk_attitude"]["farmer"]["stress_shift"]
        _thr = _ss["severe_loss_threshold"]
        if str(_thr).lower() == "auto":
            # Same definition of "severe" as the adoption-side risk framing:
            # damage_mult_severe x mean damage given a shock, where the mean
            # is built the same way _context builds it.
            _rf = self.cfg["params"]["behaviour"]["risk_framing"]
            _nat = [d for d in self.disrupt_spec.values()
                    if d["id"] in ("D1", "D2", "D3")]
            _mean_dmg = float(np.mean(
                [d["model"]["yield_damage_max"] * 0.45 for d in _nat]))
            _thr = _mean_dmg * float(_rf["damage_mult_severe"])
        self._risk_stress_cfg = {
            "shift": float(_ss["recent_severe_loss"]),
            "threshold": float(_thr),
            "decay": float(_ss["stress_decay_per_season"]),
            "cap": float(_ss["stress_cap"]),
        }
        self._expected_effort: dict = {}   # adaptive, updated each season
        self._expected_service: dict = {}  # served / requested, per tech
        self._build_outcome_noise()
        self._apply_mountain_adaptation()

    def refresh_cached_params(self) -> None:
        """Re-read every configuration value that __init__ caches on self.

        Any parameter consumed once at construction and stored on the instance
        is invisible to code that mutates cfg afterwards -- the sensitivity
        harness does exactly that, and silently reported ZERO effect for three
        parameters that had simply never been applied. Rather than patch the
        harness once per parameter, every cached read is centralised here and
        the harness calls this after mutating cfg. Extend this method, not the
        harness, whenever a new cached parameter is introduced.
        """
        p = self.cfg["params"]
        self._non_power_share = float(
            p["loss"].get("non_power_outage_share", 1.0))
        self._hazard_lr = float(
            p["behaviour"]["belief_update"].get("learning_rate_hazard", 0.15))
        _ss = p["risk_attitude"]["farmer"]["stress_shift"]
        _thr = _ss["severe_loss_threshold"]
        if str(_thr).lower() == "auto":
            _rf = p["behaviour"]["risk_framing"]
            _nat = [d for d in self.disrupt_spec.values()
                    if d["id"] in ("D1", "D2", "D3")]
            _thr = (float(np.mean([d["model"]["yield_damage_max"] * 0.45
                                   for d in _nat]))
                    * float(_rf["damage_mult_severe"]))
        self._risk_stress_cfg = {
            "shift": float(_ss["recent_severe_loss"]),
            "threshold": float(_thr),
            "decay": float(_ss["stress_decay_per_season"]),
            "cap": float(_ss["stress_cap"]),
        }

    def _build_outcome_noise(self) -> None:
        """Common random numbers for idiosyncratic yield noise.

        The outcome stream exists so that a policy cannot change a farmer's
        idiosyncratic yield draw merely by changing how many BEHAVIOURAL
        random numbers were consumed before it. Drawing sequentially inside
        the season loop would not achieve that: exits and adoption differ by
        scenario, so the draw ORDER differs and the pairing is lost anyway.

        Indexing the matrix by (farmer uid, season) instead guarantees that
        farmer i in season t receives the same shock-free yield draw under
        every scenario, which is what makes a paired policy contrast estimate
        the policy effect rather than the difference of two noise paths.
        """
        cv = float(self.cfg["params"]["production"]["yield_cv"])
        n = len(self.farmers)
        horizon = max(int(self.seasons_total) + 4, 64)
        self._yield_noise = np.clip(
            self.outcome_rng.normal(1.0, cv, size=(n, horizon)), 0.3, 1.7)

    def _yield_noise_for(self, uid: int, season: int) -> float:
        """Idiosyncratic yield multiplier for one farmer in one season."""
        nz = getattr(self, "_yield_noise", None)
        if nz is None or uid >= nz.shape[0] or season >= nz.shape[1]:
            cv = float(self.cfg["params"]["production"]["yield_cv"])
            return float(np.clip(self.outcome_rng.normal(1.0, cv), 0.3, 1.7))
        return float(nz[uid, season])

    def _apply_mountain_adaptation(self) -> None:
        """Instrument P7: light equipment and remote service points.

        Terrain fit caps what any technology can achieve on slope land, so no
        amount of subsidy closes the spatial gap without this instrument. P7
        raises that cap (lighter, hill-suitable equipment) and shortens the
        effective distance to service (remote service points).
        """
        p7 = self.instruments.get("P7", {})
        light = float(p7.get("light_equipment_share", 0.0) or 0.0)
        points = float(p7.get("remote_service_points", 0.0) or 0.0)
        transport = float(p7.get("transport_support", 0.0) or 0.0)

        self.terrain_fit_bonus = {
            "plain": 1.0,
            "hill": 1.0 + 0.35 * light,
            "mountain": 1.0 + 0.80 * light,
        }
        if points > 0 or transport > 0:
            reach = min(1.0, points / 40.0) * 0.5 + transport * 0.2
            for f in self.farmers:
                if f.terrain in ("hill", "mountain"):
                    f.service_distance = float(max(0.0,
                                                   f.service_distance * (1.0 - reach)))

    def _sync_provider_capacity(self) -> None:
        """Distribute the SD capacity stock across heterogeneous providers.

        Provider fleet weights preserve cross-provider heterogeneity, while
        their capacities sum exactly to the SD layer's available stock.  This
        prevents both global pooling and accidental capacity inflation.
        """
        if not getattr(self, "providers", None):
            return
        total_weight = sum(max(pr.capacity_weight, 1e-9)
                           for pr in self.providers)
        for pr in self.providers:
            pr.units = max(
                1e-6,
                self.sd.capacity_units * max(pr.capacity_weight, 1e-9)
                / total_weight,
            )

    def _provider_by_id(self, uid: int) -> ServiceProvider | None:
        return next((pr for pr in self.providers if pr.uid == uid), None)

    def _candidate_providers(self, f: Farmer) -> list[ServiceProvider]:
        reachable = [pr for pr in self.providers
                     if pr.county == f.county
                     or f.county in pr.service_radius_counties]
        return reachable or list(self.providers)

    def _assign_provider(self, f: Farmer, tech: str,
                         assigned_mu: dict[int, float]) -> ServiceProvider:
        """Load-balance a request inside the farmer's service market."""
        candidates = self._candidate_providers(f)
        # A stable tie-break spreads otherwise identical requests without
        # making assignment depend on Python object/list order.
        target = (f.uid * 37 + TECHS.index(tech) * 101) % max(len(self.providers), 1)
        return min(
            candidates,
            key=lambda pr: (
                assigned_mu.get(pr.uid, 0.0) / max(pr.season_capacity_mu, 1e-9)
                + (0.0 if pr.county == f.county
                   else 0.15 * (1.0 + f.service_distance)),
                abs(pr.uid - target),
                pr.uid,
            ),
        )

    # ------------------------------------------------------------------
    def _build_population(self, n_farmers_override: int | None) -> None:
        p = self.cfg["params"]
        ra = p["risk_attitude"]
        rng = self.rng

        # -- farmers ---------------------------------------------------
        self.farmers: list[Farmer] = []
        uid = 0
        scale = 1.0
        if n_farmers_override:
            scale = n_farmers_override / sum(c["n_farmers"]
                                             for c in self.counties.values())
        for cid, c in self.counties.items():
            n = max(1, int(round(c["n_farmers"] * scale)))
            for _ in range(n):
                area = float(np.clip(rng.lognormal(
                    np.log(max(c["mean_area_mu"], 0.5)), 0.55), 0.5, 400.0))
                edu = float(np.clip(rng.normal(7.5, 3.0), 0, 16))
                age = float(np.clip(rng.normal(52, 11), 20, 80))
                coop = bool(rng.random() < 0.28 + 0.25 * c["service_density"])
                lit = float(np.clip(
                    rng.beta(2.2, 3.2) + 0.02 * (edu - 7.5) + 0.06 * coop,
                    0.02, 0.98))
                wealth = float(np.clip(rng.lognormal(np.log(40000 + 900 * area),
                                                     0.5), 4000, 5e6))
                liq = float(np.clip(rng.beta(2.0, 3.0), 0.02, 0.95))

                # ---- RISK ATTITUDE: trait + observable covariates -----
                base = float(rng.beta(*ra["farmer"]["params"]))
                lo, hi = ra["farmer"]["support"]
                c_risk = lo + base * (hi - lo)
                cov = ra["farmer"]["covariates"]
                # prior_loss_experience was DECLARED as a covariate but was
                # silently missing from this sum, so the one channel linking
                # risk attitude to hazard exposure never operated. It is
                # proxied by how exposed the county is: unirrigated land in
                # broken terrain has seen more loss. Centred so it shifts the
                # distribution's shape, not its mean.
                _plp = ra["farmer"]["prior_loss_proxy"]
                exposure = (1.0 - float(c["irrigation_share"])) * float(
                    _plp["terrain_weight"].get(c["terrain"], 1.0))
                prior_loss = float(np.clip(exposure, 0.0, 1.0))
                c_risk += (cov["log_area_mu"] * np.log(max(area, 1.0))
                           + cov["education_years"] * (edu - 7.5)
                           + cov["age_years"] * (age - 52)
                           + cov["cooperative_member"] * coop
                           + cov["liquidity_ratio"] * (liq - 0.5)
                           + cov["digital_literacy"] * (lit - 0.5)
                           + cov["prior_loss_experience"]
                           * (prior_loss - float(_plp["centre"])))
                c_risk = float(np.clip(c_risk, 0.08, 0.92))

                self.farmers.append(Farmer(
                    uid=uid, county=cid, terrain=c["terrain"], area_mu=area,
                    c_risk=c_risk, c_risk_base=c_risk,
                    prior_loss_experience=prior_loss,
                    wealth=wealth, liquidity=liq,
                    education_years=edu, age_years=age, coop_member=coop,
                    digital_literacy=lit,
                    irrigation=bool(rng.random() < c["irrigation_share"]),
                    service_distance=float(np.clip(
                        1.0 - c["service_density"] + rng.normal(0, 0.12),
                        0.0, 1.0)),
                    trust=float(p["behaviour"]["trust"]["initial"]),
                    belief_efficacy={t: float(np.clip(rng.normal(
                        p["behaviour"]["belief_update"]["prior_efficacy_mean"],
                        p["behaviour"]["belief_update"]["prior_efficacy_sd"]),
                        0.01, 0.8)) for t in TECHS},
                ))
                uid += 1

        self._wire_network()

        # -- providers -------------------------------------------------
        self.providers: list[ServiceProvider] = []
        cids = list(self.counties)
        n_prov = int(p["population"]["n_providers"])
        for j in range(n_prov):
            county_index = j % len(cids)
            cid = cids[county_index]
            base = float(rng.beta(*ra["provider"]["params"]))
            lo, hi = ra["provider"]["support"]
            c2 = float(np.clip(lo + base * (hi - lo), 0.06, 0.92))
            fleet_weight = float(rng.integers(3, 12))
            self.providers.append(ServiceProvider(
                uid=j, county=cid, c_risk=c2,
                units=fleet_weight, capacity_weight=fleet_weight,
                utilisation_ceiling=float(
                    p["sd"]["stocks"]["service_capacity"]["utilisation_ceiling"]),
                unit_capacity_mu_per_day=float(
                    self.tech_spec["T3"]["model"]["capacity_mu_per_unit_per_day"]),
                price_per_mu=float(
                    self.tech_spec["T3"]["model"]["service_price_per_mu"]),
                u_min_base=float(p["contract"]["u_min_provider_base"]),
                service_radius_counties=(
                    cids[(county_index - 1) % len(cids)],
                    cids[(county_index + 1) % len(cids)],
                ),
            ))

        # -- cooperatives ----------------------------------------------
        self.cooperatives = [
            Cooperative(uid=k, county=cids[k % len(cids)])
            for k in range(int(p["population"]["n_cooperatives"]))]
        for f in self.farmers:
            if f.coop_member:
                for co in self.cooperatives:
                    if co.county == f.county:
                        co.members.append(f.uid)
                        break

    def _wire_network(self) -> None:
        """Small-world peer network with a strong within-county bias."""
        p = self.cfg["params"]["behaviour"]["peer_network"]
        k, bias = int(p["k"]), float(p["within_county_bias"])
        # p_rewire was declared (type: watts_strogatz) but never read, so the
        # graph was a fixed county-biased random graph with no rewiring and
        # the documented small-world structure did not exist.
        p_rw = float(p.get("p_rewire", 0.0))
        by_county: dict[str, list[int]] = {}
        for f in self.farmers:
            by_county.setdefault(f.county, []).append(f.uid)
        all_ids = [f.uid for f in self.farmers]
        for f in self.farmers:
            same = by_county[f.county]
            n_in = int(round(k * bias))
            peers = set()
            if len(same) > 1:
                peers.update(self.rng.choice(
                    same, size=min(n_in, len(same) - 1), replace=False).tolist())
            peers.update(self.rng.choice(
                all_ids, size=max(0, k - n_in), replace=False).tolist())
            # rewire each local tie to a random global node with prob p_rewire
            if p_rw > 0 and peers:
                local = [q for q in peers if q in same]
                for q in local:
                    if self.rng.random() < p_rw:
                        peers.discard(q)
                        peers.add(int(self.rng.choice(all_ids)))
            peers.discard(f.uid)
            f.peers = sorted(peers)

    # ------------------------------------------------------------------
    # Season loop
    # ------------------------------------------------------------------
    def step(self, season: int, force_shocks: list[str] | None = None) -> SeasonRecord:
        p = self.cfg["params"]
        rng = self.rng
        month = p["production"]["crop_calendar"]["critical_window_months"][0]
        active = [f for f in self.farmers if not f.exited]
        for f in active:
            f.service_paid = 0.0
            f.last_insurance_receipt = 0.0
            f.last_claim_amount = 0.0
            for t in TECHS:
                f.served[t] = False
                f.provider_id[t] = -1
            due = [amount for due_season, amount in f.pending_indemnities
                   if due_season <= season]
            f.pending_indemnities = [
                (due_season, amount)
                for due_season, amount in f.pending_indemnities
                if due_season > season
            ]
            f.last_insurance_receipt = float(sum(due))

        # -- 1. SD -> ABM ---------------------------------------------
        sd_state = self.sd.interface()
        gov_terms = self.gov.terms(sd_state)

        # -- 2. reliability and expected wait --------------------------
        p5 = self.instruments.get("P5", {})
        reserved = float(p5.get("reserved_capacity_share", 0.0))
        prior_demand = sum(f.area_mu for f in active) * 0.45
        wait_prior = self.sd.expected_wait_days(prior_demand, reserved)

        # device-level availability x system-level infrastructure reliability
        reliability = {t: float(np.clip(
            self.tech_spec[t]["model"]["availability_base"]
            * sd_state["reliability"], 0.1, 0.99)) for t in TECHS}

        # verifiability: monitoring technology is what makes the
        # first-best contract enforceable at all
        vcfg = p["contract"]["verifiability"]
        verifiability = {
            "T1": vcfg["with_T1_remote_sensing"],
            "T2": vcfg["with_T2_sensors"],
            "T3": vcfg["with_T3_telemetry"],
        }

        # -- 3. adoption ----------------------------------------------
        ctx = self._context(gov_terms, reliability, verifiability,
                            {t: wait_prior for t in TECHS}, sd_state)
        for f in active:
            f.decide_adoption(ctx, rng)

        # Draw the realised event before dispatch so P5's reserved emergency
        # capacity can actually respond to an actionable warning.  Adoption
        # still uses ex-ante beliefs because the draw occurs afterwards.
        events = self.shockgen.draw_season(season, month, force=force_shocks)
        lead_gain = float(p5.get("lead_time_gain_days", 0.0) or 0.0)
        if lead_gain > 0:
            for ev in events:
                if ev.warning_lead_days > 0:
                    ev.warning_lead_days += lead_gain

        def emergency_eligible(farmer: Farmer, tech: str) -> bool:
            if tech != "T3":
                return False
            return any(
                not ev.is_economic
                and ev.warning_lead_days > 0
                and ev.severity_for(farmer.county) > 0
                for ev in events
            )

        # -- 4. matching and queueing ---------------------------------
        self._sync_provider_capacity()
        for pr in self.providers:
            pr.reset_season()
        demand_mu = 0.0
        requests: list[tuple[Farmer, str]] = []
        for f in active:
            for t in TECHS:
                if f.mode[t] == "service" and t in ("T2", "T3"):
                    demand_mu += f.area_mu
                    requests.append((f, t))
        # First assign every request to a provider-specific market.  The same
        # provider ID is then used for capacity, effort, payment and revenue.
        assigned_mu = {pr.uid: 0.0 for pr in self.providers}
        provider_requests = {pr.uid: [] for pr in self.providers}
        rng.shuffle(requests)
        requests.sort(key=lambda r: (r[0].service_distance,
                                     -float(r[0].coop_member)))
        for f, t in requests:
            pr = self._assign_provider(f, t, assigned_mu)
            assigned_mu[pr.uid] += f.area_mu
            provider_requests[pr.uid].append((f, t))
            f.provider_id[t] = pr.uid

        wait_by_provider = {
            pr.uid: self.sd.wait_days_for_capacity(
                assigned_mu[pr.uid], pr.season_capacity_mu * (1.0 - reserved))
            for pr in self.providers
        }
        wait_days = (float(np.average(
            [wait_by_provider[pr.uid] for pr in self.providers],
            weights=[max(assigned_mu[pr.uid], 1e-9) for pr in self.providers]))
            if requests else 1.0)
        ctx["expected_wait_days"] = {t: wait_days for t in TECHS}
        ctx["demand_pressure"] = {
            pr.uid: assigned_mu[pr.uid] / max(pr.season_capacity_mu, 1e-9)
            for pr in self.providers
        }

        # Capacity is enforced separately for every provider.  Near and
        # cooperative farmers retain priority within each local queue.
        served_ids = set()
        for pr in self.providers:
            capacity_left = pr.season_capacity_mu * (1.0 - reserved)
            unserved = []
            for f, t in provider_requests[pr.uid]:
                if capacity_left >= f.area_mu:
                    capacity_left -= f.area_mu
                    pr.served_mu_season += f.area_mu
                    f.served[t] = True
                    served_ids.add((f.uid, t))
                else:
                    unserved.append((f, t))
            # P5 releases the reserved pool only for T3 requests in counties
            # hit by a warned event; unused reserve is not silently reassigned
            # to routine work.
            emergency_left = pr.season_capacity_mu * reserved
            still_unserved = []
            for f, t in unserved:
                if (emergency_eligible(f, t)
                        and emergency_left >= f.area_mu):
                    emergency_left -= f.area_mu
                    pr.served_mu_season += f.area_mu
                    f.served[t] = True
                    served_ids.add((f.uid, t))
                else:
                    still_unserved.append((f, t))
            for f, t in still_unserved:
                pr.backlog_mu += f.area_mu
        backlog = sum(pr.backlog_mu for pr in self.providers)

        # -- 5. contracting: the Guo-Parlar-Zhang layer ----------------
        contract_mix = {"linear": 0, "concave": 0, "convex": 0,
                        "self": 0, "infeasible": 0}
        effort_by_tech = {t: [] for t in TECHS}
        for f in active:
            for t in TECHS:
                mode = f.mode[t]
                if mode == "none":
                    f.effort[t] = 0.0
                    continue
                pr = (self._provider_by_id(f.provider_id[t])
                      if mode == "service" else None)
                if mode == "service" and t == "T1":
                    # Information services do not consume machinery capacity.
                    if pr is None:
                        pr = self._assign_provider(
                            f, t, {x.uid: x.served_mu_season for x in self.providers})
                        f.provider_id[t] = pr.uid
                    f.served[t] = True
                if mode == "service" and t in ("T2", "T3") and (f.uid, t) not in served_ids:
                    f.effort[t] = 0.0            # requested but never served
                    f.state[t] = STATE_ACQUIRED
                    continue
                (e_dem, form, anchor, expected_payment,
                 contract_c1, contract_c2) = f.contract_effort(t, pr, ctx)
                e_real = f.realised_effort(t, e_dem, ctx, mode)
                f.effort[t] = e_real
                f.contract_form[t] = form
                f.contract_anchor[t] = anchor
                f.contract_expected_payment[t] = expected_payment
                f.contract_c1[t] = contract_c1
                f.contract_c2[t] = contract_c2
                if pr is not None:
                    pr._effort_accum.append(e_real)
                    f.provider_id[t] = pr.uid
                if "infeasible" in form:
                    contract_mix["infeasible"] += 1
                    f.served[t] = False
                elif "self" in form:
                    contract_mix["self"] += 1
                elif form in contract_mix:
                    contract_mix[form] += 1
                if e_real > 0:
                    f.experience[t] = f.experience.get(t, 0) + 1
                    f.state[t] = STATE_USED
                    effort_by_tech[t].append(e_real)
                    if e_real >= 0.6 and f.digital_literacy >= self.tech_spec[t][
                            "model"]["digital_literacy_threshold"]:
                        f.state[t] = STATE_EFFECTIVE
        for pr in self.providers:
            pr.close_season()
        # roll the adaptive effort expectation forward
        for t in TECHS:
            vals = effort_by_tech.get(t) or []
            if vals:
                self._expected_effort[t] = float(np.mean(vals))
        # and the service fill rate: requested vs actually served
        for t in TECHS:
            req = [f for f in active if f.mode.get(t) == "service"]
            if req:
                self._expected_service[t] = float(
                    np.mean([1.0 if f.served.get(t) else 0.0 for f in req]))

        # -- 6. shock consequences ------------------------------------
        # Events were drawn immediately after adoption so dispatch could use
        # P5 warning lead time and emergency capacity.
        price_stress = 0.0
        tech_avail = 1.0
        infra_shock = 0.0
        for ev in events:
            if ev.is_economic:
                price_stress = max(price_stress, ev.price_multiplier - 1.0)
            tech_avail = min(tech_avail, ev.tech_availability)
            if ev.tech_availability < 0.9:
                infra_shock = max(infra_shock, ev.severity)

        # -- 7. outcomes ----------------------------------------------
        base_val = float(p["production"]["base_yield_value_per_mu"])
        stage_vuln = p["loss"]["stage_vulnerability"]
        cap_mit = float(p["loss"]["max_total_mitigation"])
        losses, incomes, avoided = [], [], []
        delivered, failures = 0, 0

        for f in active:
            county = self.counties[f.county]
            gross = base_val * f.area_mu
            cost = float(p["production"]["input_cost_per_mu"]) * f.area_mu
            cost *= (1.0 + price_stress)

            dmg = 0.0
            mitig_total = 0.0
            # Reset ONCE per farmer, not once per event: `observed` below is
            # aggregated over the whole season, so the attenuation used to
            # normalise it must correspond to the season too. Resetting inside
            # the loop kept only the last event's factors.
            f._last_atten = {}
            best_mit = -1.0
            for ev in events:
                if ev.is_economic:
                    continue
                spec_m = self.disrupt_spec[ev.shock_id].get("model", {})
                d = damage_fraction(ev, f.county, f.terrain, stage_vuln, spec_m,
                                    float(f.irrigation))
                _pending = {}
                f._atten_sink = _pending
                mit = self._mitigation(f, ev, tech_avail, reliability)
                # keep the attenuation from the event that actually drove
                # the season's mitigation, which is the one `observed` reflects
                if mit > best_mit:
                    best_mit = mit
                    f._last_atten = dict(_pending)
                mitig_total = max(mitig_total, mit)
                dmg += d * (1.0 - min(mit, cap_mit))
            dmg_nomit = sum(
                damage_fraction(ev, f.county, f.terrain, stage_vuln,
                                self.disrupt_spec[ev.shock_id].get("model", {}),
                                float(f.irrigation))
                for ev in events if not ev.is_economic)
            dmg = float(np.clip(dmg, 0.0, 1.0))
            dmg_nomit = float(np.clip(dmg_nomit, 0.0, 1.0))

            yield_noise = self._yield_noise_for(f.uid, season)
            revenue = gross * (1.0 - dmg) * yield_noise
            outcome_fraction = float(np.clip((1.0 - dmg) * yield_noise,
                                             1e-6, 1.0))

            # technology cost actually paid
            paid = 0.0
            for t in TECHS:
                if f.mode[t] == "service":
                    if not f.served[t]:
                        continue
                    price = ctx["service_price"].get(t, 0.0)
                    # The outcome-contingent payment multiplier was removed
                    # after measurement: with a service bill of ~256 against
                    # gross output of ~26,700, even the full 0.25-2.50 swing
                    # moved income by ~1.5 per cent, and gini, p10 income and
                    # mitigation were all unchanged to three decimals. It
                    # added a moving part and a scale-matching failure mode
                    # without changing any reported quantity. Contract
                    # curvature still acts where it is identified: through
                    # b(e) and the effort level e*.
                    gross_bill = price * f.area_mu
                    voucher = min(gov_terms["voucher_per_mu"], price) * min(
                        f.area_mu, gov_terms["voucher_max_mu"])
                    if gov_terms["voucher_targeting"] == "smallholder" and f.area_mu > 30:
                        voucher = 0.0
                    if gov_terms["voucher_targeting"] == "mountain" and f.terrain != "mountain":
                        voucher = 0.0
                    voucher = min(voucher, gross_bill)
                    farmer_bill = max(gross_bill - voucher, 0.0)
                    paid += farmer_bill
                    f.service_paid += farmer_bill
                    self.gov.spent_season += voucher
                    pr = self._provider_by_id(f.provider_id[t])
                    if pr is not None:
                        pr.revenue_season += gross_bill
                elif f.mode[t] == "own":
                    m = self.tech_spec[t]["model"]
                    capex = (m.get("capex_per_mu", 0.0) * f.area_mu
                             if t == "T2" else m.get("capex_own_per_unit", 0.0))
                    life = max(m.get("lifetime_years", 5), 1)
                    sub = min(capex * gov_terms["subsidy_rate"],
                              gov_terms["subsidy_cap"])
                    paid += (capex - sub) / life + m.get("opex_per_mu", 0.0) * f.area_mu
                    self.gov.spent_season += sub / life

            # insurance
            indemnity = f.last_insurance_receipt
            if gov_terms["insurance_coverage"] > 0:
                premium = (gross * 0.06 * (1 - gov_terms["insurance_premium_subsidy"]))
                paid += premium
                self.gov.spent_season += gross * 0.06 * gov_terms["insurance_premium_subsidy"]
                if dmg > 0.15:
                    basis = 1.0 - gov_terms["insurance_basis_risk"]
                    claim = gross * dmg * gov_terms["insurance_coverage"] * basis
                    f.last_claim_amount = claim
                    lag_days = max(float(gov_terms["insurance_payout_lag"]), 0.0)
                    season_days = 365.0 / max(self.seasons_per_year, 1)
                    lag_steps = int(np.ceil(lag_days / season_days)) if lag_days > 0 else 0
                    if lag_steps == 0:
                        indemnity += claim
                    else:
                        f.pending_indemnities.append((season + lag_steps, claim))
                        delay_penalty = p["behaviour"]["trust"][
                            "loss_on_payment_delay"] * min(lag_days / season_days, 1.0)
                        f.trust = max(0.05, f.trust - delay_penalty)

            # realised routine benefit, so that what a farmer receives matches
            # what adoption_value promised; without this the decision rule and
            # the accounting would disagree and income would be understated
            routine_realised = 0.0
            for t in TECHS:
                if f.state[t] not in (STATE_USED, STATE_EFFECTIVE):
                    continue
                if f.mode[t] == "service" and not f.served[t]:
                    continue
                tm = self.tech_spec[t]["model"]
                rb = float(tm.get("routine_benefit_per_mu", 0.0))
                if rb <= 0.0:
                    continue
                fit_t = min(1.0, float(tm["terrain_fit"].get(f.terrain, 1.0))
                            * float(getattr(self, "terrain_fit_bonus", {})
                                    .get(f.terrain, 1.0)))
                routine_realised += rb * f.area_mu * fit_t * reliability[t]

            income = revenue - cost - paid + indemnity + routine_realised
            f.last_income = income
            f.last_loss = dmg

            # ENDOGENOUS RISK ATTITUDE. A severe loss raises risk aversion;
            # the shift decays back toward the structural level over a few
            # calm seasons. The configuration declared this channel as
            # active ("only the loss-experience and low-liquidity channels
            # currently fire") but nothing implemented it, so c1 was a fixed
            # trait and the shock -> risk attitude -> adoption feedback that
            # the study is about could not operate.
            # learn about HAZARD FREQUENCY, not just efficacy: a farmer who
            # stops being hit should stop paying for protection
            f.update_hazard_belief(bool(dmg_nomit > 0.02),
                                   self._hazard_prior, self._hazard_lr)

            rs = self._risk_stress_cfg
            f.risk_stress *= (1.0 - rs["decay"])
            if dmg >= rs["threshold"]:
                f.risk_stress = min(f.risk_stress + rs["shift"], rs["cap"])
            f.c_risk = float(np.clip(f.c_risk_base + f.risk_stress, 0.08, 0.92))
            f.wealth = max(f.wealth + income - cost * 0.0, 1000.0)
            f.income_history.append(income)
            f.loss_history.append(dmg)
            losses.append(dmg)
            incomes.append(income)
            avoided.append(max(dmg_nomit - dmg, 0.0))

            if mitig_total > 0.05 and dmg_nomit > 0.05:
                delivered += 1
            served_but_late = any(
                f.mode[t] == "service" and f.effort[t] == 0.0 for t in TECHS)
            if served_but_late or (mitig_total <= 0.01 and dmg > 0.2):
                failures += 1

            # -- 8. learning, trust, exit ------------------------------
            f._last_observed = ((dmg_nomit - dmg) / max(dmg_nomit, 1e-6)
                                if dmg_nomit > 0.02 else None)
            if income < 0:
                f.bad_seasons += 1
            else:
                f.bad_seasons = max(0, f.bad_seasons - 1)
            ex = p["behaviour"]["exit_rule"]
            if (f.bad_seasons >= ex["consecutive_bad_seasons"]
                    and rng.random() < ex["exit_probability_when_triggered"]):
                f.exited = True
                for t in TECHS:
                    f.state[t] = STATE_EXITED

        # Beliefs update in a second pass so that peer signals reflect this
        # season's outcomes for everyone, not a within-season ordering.
        self._pop_index = {g.uid: g for g in active}
        for f in active:
            self._update_beliefs(f, events, getattr(f, "_last_observed", None))

        # recovery tracking
        if self._baseline_income is None:
            self._baseline_income = float(np.mean(incomes)) if incomes else 0.0
        self._track_recovery(active)

        # -- 9. ABM -> SD ---------------------------------------------
        eff_rate = self._effective_use_rate(active)
        n_active = max(len(active), 1)
        abm_feedback = {
            "expenditure": self.gov.spent_season,
            "requested": self.gov.spent_season * 1.15,
            "n_farmers": len(self.farmers),
            "effective_use_rate": eff_rate,
            "infrastructure_shock": infra_shock,
            "delivered_rate": delivered / n_active,
            "failure_rate": failures / n_active,
        }
        p3 = self.instruments.get("P3", {})
        p4 = self.instruments.get("P4", {})
        sd_policy = {
            "new_units_this_step": (p3.get("new_centres_per_year", 0)
                                    * p3.get("units_per_centre", 8)
                                    / self.seasons_per_year * self.pop_scale),
            "training_slots_this_step": (p4.get("slots_per_year", 0)
                                         / self.seasons_per_year * self.pop_scale),
            "skill_gain": p4.get("skill_gain", 0.12),
            "followup_support": p4.get("followup_support", False),
        }
        self.gov.spent_total += self.gov.spent_season
        new_sd = self.sd.step(1.0 / self.seasons_per_year, abm_feedback, sd_policy)

        # Training raises literacy of a sample of farmers, and that gain
        # DECAYS without follow-up support. P4 declares
        # `decay_rate_annual` and `followup_support` and states that "skill
        # decays without follow-up, so one-off training campaigns produce a
        # transient bump that the model will show fading" -- but neither key
        # was read, so every gain was permanent and a single campaign looked
        # like a durable capability increase. Decay is applied to the trained
        # increment only, so untrained farmers are unaffected.
        decay_a = float(p4.get("decay_rate_annual", 0.0))
        if decay_a > 0 and not p4.get("followup_support", False):
            per_season = 1.0 - (1.0 - decay_a) ** (1.0 / self.seasons_per_year)
            for f in active:
                gain = getattr(f, "trained_gain", 0.0)
                if gain > 1e-9:
                    lost = gain * per_season
                    f.digital_literacy = float(np.clip(
                        f.digital_literacy - lost, 0.0, 1.0))
                    f.trained_gain = gain - lost
        if sd_policy["training_slots_this_step"] > 0:
            n_tr = int(min(sd_policy["training_slots_this_step"], len(active)))
            for f in rng.choice(active, size=n_tr, replace=False):
                before = f.digital_literacy
                f.digital_literacy = float(np.clip(
                    f.digital_literacy + p4.get("skill_gain", 0.12), 0, 1))
                f.trained_gain = getattr(f, "trained_gain", 0.0) + (
                    f.digital_literacy - before)

        rec = self._record(season, events, active, losses, incomes, avoided,
                           contract_mix, effort_by_tech, wait_days, backlog,
                           new_sd)
        self.records.append(rec)
        self.gov.spent_season = 0.0
        return rec

    # ------------------------------------------------------------------
    def _context(self, gov_terms, reliability, verifiability, wait, sd_state):
        p = self.cfg["params"]
        nat = [d for d in self.disrupt_spec.values()
               if d.get("family") != "economic"]
        # probability that AT LEAST ONE natural shock occurs this season
        p_none = 1.0
        for d in nat:
            p_period = annual_to_period_probability(
                float(d["model"]["annual_probability_prior"]),
                self.seasons_per_year,
            )
            p_none *= (1.0 - p_period)
        shock_prob = 1.0 - p_none
        # expected damage CONDITIONAL on a shock, severity-weighted
        exp_damage_given = float(np.mean(
            [d["model"]["yield_damage_max"] * 0.45 for d in nat]))
        fr = p["behaviour"].get("frictions", {})
        bh = p["behaviour"].get("risk_framing", {})
        if self._hazard_prior < 0.0:
            self._hazard_prior = float(shock_prob)
        return {
            "shock_probability": shock_prob,
            "expected_damage_given_shock": exp_damage_given,
            "input_cost_per_mu": float(p["production"]["input_cost_per_mu"]),
            # Conditional damage distribution, as multiples of the mean.
            # Derived from the calibrated Beta severity parameters so
            # the tail a risk-averse farmer reacts to is preserved.
            "severe_share": float(bh.get("severe_share", 0.25)),
            "damage_mult_moderate": float(bh.get("damage_mult_moderate", 0.62)),
            "damage_mult_severe": float(bh.get("damage_mult_severe", 2.14)),
            "max_mitigation": float(p["loss"]["max_total_mitigation"]),
            "income_floor": float(p["behaviour"].get("frictions", {})
                                  .get("income_floor", 200.0)),
            "switch_share": float(fr.get("switching_cost_share", 0.25)),
            "switch_floor": float(fr.get("switching_cost_floor", 40.0)),
            "tech_spec": self.tech_spec,
            "policy": gov_terms,
            "population": {f.uid: f for f in self.farmers},
            "reliability": reliability,
            "verifiability": verifiability,
            "expected_wait_days": wait,
            # ADAPTIVE EXPECTATIONS. This was a hard-coded 0.6 while providers
            # actually delivered about 0.73, so every farmer systematically
            # understated the benefit of adopting by roughly 22 per cent and
            # the model carried a permanent pessimism bias. Farmers now use
            # last season's realised mean effort, falling back to the prior
            # only in the first season when nothing has been observed yet.
            # Fraction of service requests actually filled last season. The
            # routine benefit is only received if the provider turns up, and
            # without this farmers valued it at 100 per cent while congestion
            # withheld about 19 per cent of it -- the decision rule and the
            # accounting disagreed by exactly the queue.
            "expected_service_rate": {t: self._expected_service.get(t, 1.0)
                                      for t in TECHS},
            "expected_effort": {(t, m): self._expected_effort.get(t, 0.6)
                                for t in TECHS
                                for m in ("own", "service")},
                        "base_value_per_mu": float(p["production"]["base_yield_value_per_mu"]),
            "service_price": {
                "T2": self.tech_spec["T2"]["model"]["opex_per_mu"] * 2.2,
                "T3": self.tech_spec["T3"]["model"]["service_price_per_mu"],
                "T1": self.tech_spec["T1"]["model"]["opex_per_mu"],
            },
            "terrain_fit_bonus": getattr(self, "terrain_fit_bonus", {}),
            "access_cost_scale": float(fr.get("access_cost_scale", 260.0)),
            "learning_scale": float(fr.get("learning_scale", 900.0)),
            "learning_decay": float(fr.get("learning_decay", 1.0)),
            "subsidy_salience": float(fr.get("subsidy_salience", 0.35)),
            "peer_weight": float(p["behaviour"]["social_learning_weight"]),
            # `choice_rule` was declared but never read, implying a
            # configurability that does not exist. Softmax is the only rule
            # implemented, so an unsupported value must fail loudly rather
            # than be silently ignored.
            "choice_rule": _require_softmax(p["behaviour"]["choice_rule"]),
            "softmax_temperature": float(p["behaviour"]["softmax_temperature"]),
            "quantise_step": float(p["contract"]["quantise_step"]),
            "gamma_provider": float(p["contract"]["gamma_provider"]),
            "gamma_farmer_selfuse": float(p["contract"]["gamma_farmer_selfuse"]),
            "u_min_scaling_with_demand": float(
                p["contract"]["u_min_scaling_with_demand"]),
            "effort_max": float(p["contract"]["effort_max"]),
            "shirk_effort_multiplier": float(
                p["contract"]["verifiability"]["shirk_effort_multiplier"]),
            "base_verifiability": float(
                p["contract"]["verifiability"]["base_verifiability"]),
            "demand_pressure": {pr.uid: 0.0 for pr in self.providers},
            "price_stress": 0.0,
            "rng": self.rng,
        }

    def _nearest_provider(self, f: Farmer) -> ServiceProvider:
        """Compatibility helper returning a stable local provider.

        Service allocation uses :meth:`_assign_provider`, which also accounts
        for load.  This helper intentionally never selects the first county
        provider merely because of list order.
        """
        candidates = self._candidate_providers(f)
        target = f.uid % max(len(self.providers), 1)
        return min(candidates, key=lambda pr: (abs(pr.uid - target), pr.uid))

    def _mitigation(self, f: Farmer, ev, tech_avail: float, reliability) -> float:
        """Realised loss reduction: eta_k * effort * availability * terrain fit.

        Never a fixed percentage. The availability term is what makes D3
        expose the fragility of technology-based resilience claims.
        """
        key = {"D1": "eta_drought", "D2": "eta_flood",
               "D3": "eta_compound"}.get(ev.shock_id, "eta_drought")
        total = 0.0
        for t in TECHS:
            if (f.state[t] not in (STATE_USED, STATE_EFFECTIVE)
                    or f.mode[t] == "none" or f.effort[t] <= 0):
                continue
            m = self.tech_spec[t]["model"]
            eta = float(m.get(key, 0.0))
            fit = float(m["terrain_fit"].get(f.terrain, 1.0)) * float(
                getattr(self, "terrain_fit_bonus", {}).get(f.terrain, 1.0))
            fit = min(fit, 1.0)
            # D3 cuts mains power and the mobile network. Equipment that
            # needs the grid stops; the rest degrades only through platform
            # downtime. `power_dependent` was declared per technology and
            # never read, so every technology failed identically.
            if m.get("power_dependent", False):
                avail_t = tech_avail
            else:
                avail_t = 1.0 - (1.0 - tech_avail) * self._non_power_share
            rho = reliability[t] * avail_t
            if t == "T2" and not f.irrigation:
                continue
            contrib = eta * f.effort[t] * fit * rho
            # Record the attenuation actually applied, so belief updating can
            # divide it back out. What a farmer OBSERVES is the realised
            # mitigation fraction eta*fit*rho*e, but belief_efficacy is used
            # in adoption_value as the intrinsic eta and multiplied by
            # fit*rho*e again. Storing the observation raw applied the
            # attenuation TWICE and understated expected benefit by about
            # 45 per cent, which is a large part of why adoption value was
            # negative for nearly every farmer.
            atten = f.effort[t] * fit * rho
            if t == "T1":
                # warning is only worth something if it arrives before the
                # action window closes AND the farmer can act on it
                usable = min(1.0, ev.warning_lead_days / max(ev.action_window_days, 0.1))
                can_act = 1.0 if (f.mode["T3"] != "none" or f.mode["T2"] != "none") else 0.35
                contrib *= usable * can_act
                atten *= usable * can_act
            getattr(f, "_atten_sink", f._last_atten)[t] = float(atten)
            total += contrib
        return float(total)

    def _update_beliefs(self, f: Farmer, events, observed) -> None:
        """Own-experience and peer belief updating, plus trust.

        Called in a second pass over the population so that peer signals
        reflect the whole season's outcomes rather than a within-season
        iteration order.
        """
        b = self.cfg["params"]["behaviour"]
        lr = b["belief_update"]["learning_rate_own"]
        lr_peer = b["belief_update"].get("learning_rate_peer", 0.0)
        tr = b["trust"]
        if not events:
            return
        atten_all = getattr(f, "_last_atten", {})
        for t in TECHS:
            if (f.state[t] in (STATE_USED, STATE_EFFECTIVE)
                    and f.mode[t] != "none" and observed is not None):
                # Normalise the observation back to INTRINSIC efficacy.
                # `observed` is the realised mitigation fraction, which
                # already carries effort, terrain fit and reliability;
                # belief_efficacy is consumed as eta and re-multiplied by
                # those same factors in adoption_value. Dividing the
                # attenuation back out is what makes the belief converge on
                # eta rather than on eta*fit*rho*e.
                at = float(atten_all.get(t, 0.0))
                if at <= 1e-3:
                    continue          # too little signal to attribute
                signal_eta = float(np.clip(observed / at, 0.01, 0.9))
                f.belief_efficacy[t] = float(np.clip(
                    (1 - lr) * f.belief_efficacy[t] + lr * signal_eta,
                    0.01, 0.9))
        # Audit finding M6: peer belief updating was declared in the
        # configuration but never implemented, so peers influenced the
        # adoption choice while carrying no information about efficacy.
        # A non-user learns about a technology only by observing peers.
        if lr_peer > 0 and f.peers:
            pop = self._pop_index
            for t in TECHS:
                seen = []
                for j in f.peers:
                    q = pop.get(j)
                    if (q is None or q.state.get(t, 0) not in
                            (STATE_USED, STATE_EFFECTIVE)
                            or q.mode.get(t) == "none"
                            or q._last_observed is None):
                        continue
                    at_q = float(getattr(q, "_last_atten", {}).get(t, 0.0))
                    if at_q <= 1e-3:
                        continue
                    seen.append(float(np.clip(
                        q._last_observed / at_q, 0.01, 0.9)))
                if not seen:
                    continue
                signal = float(np.mean(seen))
                f.belief_efficacy[t] = float(np.clip(
                    (1 - lr_peer) * f.belief_efficacy[t] + lr_peer * signal,
                    0.01, 0.9))
        if observed is not None and observed > 0.15:
            f.trust = min(0.95, f.trust + tr["gain_on_success"])
        elif observed is not None and observed < 0.02:
            f.trust = max(0.05, f.trust - tr["loss_on_service_failure"])
        # T1's own declared false_alarm_rate drives this, not a hard-coded
        # 0.5. The rate was declared in technologies.yaml ("erodes trust when
        # warnings do not verify") and never read.
        far = float(self.tech_spec["T1"]["model"].get("false_alarm_rate", 0.5))
        if any(ev.shock_id == "D2" and ev.warning_lead_days < 0.6 for ev in events):
            f.trust = max(0.05, f.trust - tr["loss_on_false_alarm"] * far)

    def _track_recovery(self, active) -> None:
        base = self._baseline_income or 0.0
        thresh = 0.8 * base
        for f in active:
            if not f.income_history:
                continue
            if f.income_history[-1] < thresh:
                f.seasons_below_baseline += 1
            elif f.seasons_below_baseline > 0:
                f.recovery_seasons = float(f.seasons_below_baseline)
                f.seasons_below_baseline = 0

    def _effective_use_rate(self, active) -> float:
        if not active:
            return 0.0
        n = sum(1 for f in active
                if any(f.state[t] == STATE_EFFECTIVE and f.mode[t] != "none"
                       for t in TECHS))
        return n / len(active)

    # ------------------------------------------------------------------
    def _record(self, season, events, active, losses, incomes, avoided,
                contract_mix, effort_by_tech, wait_days, backlog, sd_state):
        arr_inc = np.array(incomes) if incomes else np.array([0.0])
        adoption = {t: float(np.mean([
            f.mode[t] != "none"
            and f.state[t] in (STATE_ACQUIRED, STATE_USED, STATE_EFFECTIVE)
            for f in active])) if active else 0.0
                    for t in TECHS}
        service_share = {t: float(np.mean([f.mode[t] == "service"
                                           for f in active])) if active else 0.0
                         for t in TECHS}
        small = [f for f in active if f.area_mu <= np.percentile(
            [g.area_mu for g in active], 25)] if active else []
        large = [f for f in active if f.area_mu >= np.percentile(
            [g.area_mu for g in active], 75)] if active else []
        mtn = [f for f in active if f.terrain == "mountain"]
        pln = [f for f in active if f.terrain == "plain"]

        def _avoid(group):
            if not group:
                return 0.0
            idx = {f.uid for f in group}
            vals = [a for f, a in zip(active, avoided) if f.uid in idx]
            return float(np.mean(vals)) if vals else 0.0

        recs = [f.recovery_seasons for f in self.farmers
                if not np.isnan(f.recovery_seasons)]

        return SeasonRecord(
            season=season, year=season / self.seasons_per_year,
            shocks=[e.shock_id for e in events],
            adoption_rate=adoption, service_share=service_share,
            effective_use_rate=self._effective_use_rate(active),
            mean_effort={t: float(np.mean(v)) if v else 0.0
                         for t, v in effort_by_tech.items()},
            contract_mix=dict(contract_mix),
            mean_wait_days=float(wait_days), backlog_mu=float(backlog),
            mean_loss_fraction=float(np.mean(losses)) if losses else 0.0,
            avoided_loss_fraction=float(np.mean(avoided)) if avoided else 0.0,
            mitigation_rate=_mitigation_rate(losses, avoided),
            mean_income=float(arr_inc.mean()),
            income_p10=float(np.percentile(arr_inc, 10)),
            gini_income=_gini(arr_inc),
            exit_rate=float(np.mean([f.exited for f in self.farmers])),
            fiscal_spend=float(self.gov.spent_season),
            fiscal_cumulative=float(self.gov.spent_total),
            equity_gap=float(_avoid(large) - _avoid(small)),
            mountain_gap=float(_avoid(pln) - _avoid(mtn)),
            capacity_units=float(sd_state["capacity_units"]),
            reliability=float(sd_state["reliability"]),
            trust=float(sd_state["trust"]),
            capable_share=float(sd_state["capable_share"]),
            recovery_seasons_mean=float(np.mean(recs)) if recs else float("nan"),
        )

    # ------------------------------------------------------------------
    def run(self, seasons: int | None = None,
            forced: dict[int, list[str]] | None = None) -> list[SeasonRecord]:
        n = seasons or self.seasons_total
        forced = forced or {}
        for s in range(1, n + 1):
            self.step(s, force_shocks=forced.get(s))
        return self.records

    def to_dataframe(self):
        import pandas as pd
        rows = []
        for r in self.records:
            row = {
                "season": r.season, "year": r.year,
                "shocks": "|".join(r.shocks),
                "effective_use_rate": r.effective_use_rate,
                "mean_wait_days": r.mean_wait_days, "backlog_mu": r.backlog_mu,
                "mean_loss_fraction": r.mean_loss_fraction,
                "avoided_loss_fraction": r.avoided_loss_fraction,
                "mitigation_rate": r.mitigation_rate,
                "mean_income": r.mean_income, "income_p10": r.income_p10,
                "gini_income": r.gini_income, "exit_rate": r.exit_rate,
                "fiscal_spend": r.fiscal_spend,
                "fiscal_cumulative": r.fiscal_cumulative,
                "equity_gap": r.equity_gap, "mountain_gap": r.mountain_gap,
                "capacity_units": r.capacity_units, "reliability": r.reliability,
                "trust": r.trust, "capable_share": r.capable_share,
                "recovery_seasons_mean": r.recovery_seasons_mean,
            }
            row.update({f"adopt_{k}": v for k, v in r.adoption_rate.items()})
            row.update({f"service_{k}": v for k, v in r.service_share.items()})
            row.update({f"effort_{k}": v for k, v in r.mean_effort.items()})
            row.update({f"contract_{k}": v for k, v in r.contract_mix.items()})
            rows.append(row)
        return pd.DataFrame(rows)


def _mitigation_rate(losses, avoided) -> float:
    """Share of POTENTIAL loss that technology actually removed.

    Reported alongside the absolute avoided-loss fraction because the
    absolute number is dominated by how often shocks occur, whereas this
    ratio isolates how well the technology performs when a shock does hit.
    """
    if not losses or not avoided:
        return 0.0
    realised = float(np.sum(losses))
    prevented = float(np.sum(avoided))
    denom = realised + prevented
    return float(prevented / denom) if denom > 1e-12 else 0.0


def _gini(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x - x.min() + 1e-9 if x.min() < 0 else x + 1e-9
    x = np.sort(x)
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))
