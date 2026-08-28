"""
sd.py
=====
The selective System Dynamics layer.

It carries ONLY slow, aggregate, stock-and-flow quantities that would be
clumsy to represent agent by agent:

    government_budget          fiscal space and its carry-over
    service_capacity           equipment and operator stock, with lags
    capable_farmers            trained-skill stock, with decay
    infrastructure_reliability network, power and platform reliability
    institutional_trust        slow-moving reputation of policy and platforms

Interface discipline (config: sd.interface_contract)
----------------------------------------------------
Waiting time is computed ONCE, HERE, from SD capacity and ABM demand. The
ABM never recomputes it. Individual adoption, effort, beliefs and trust live
in the ABM and are never duplicated here; what the SD layer keeps is only
their aggregate, slow-moving residue.

This is the rule that stops the hybrid from double-counting, which is the
main failure mode of ABM-SD couplings.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SDState:
    """Stocks. Everything here is aggregate and slow."""

    budget: float
    budget_inflow_annual: float
    budget_growth: float
    carryover_share: float

    capacity_units: float
    unit_capacity_mu_per_day: float
    depreciation_rate: float
    commissioning_lag_years: float
    utilisation_ceiling: float

    capable_share: float
    training_throughput: float
    skill_decay: float
    learning_by_doing: float

    reliability: float
    maintenance_rate: float
    degradation_rate: float
    shock_damage: float
    repair_time_years: float

    trust: float
    trust_gain: float
    trust_loss: float
    trust_decay: float

    # bookkeeping
    pipeline: list = field(default_factory=list)     # (seasons_remaining, units)
    budget_requested_last: float = 0.0
    budget_spent_total: float = 0.0
    history: list = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict) -> "SDState":
        s = cfg["sd"]["stocks"]
        b, c, f, i, t = (s["government_budget"], s["service_capacity"],
                         s["capable_farmers"], s["infrastructure_reliability"],
                         s["institutional_trust"])
        return cls(
            budget=float(b["initial"]),
            budget_inflow_annual=float(b["annual_inflow"]),
            budget_growth=float(b["growth_rate"]),
            carryover_share=float(b["carryover_share"]),
            capacity_units=float(c["initial_units"]),
            unit_capacity_mu_per_day=float(c["unit_capacity_mu_per_day"]),
            depreciation_rate=float(c["depreciation_rate"]),
            commissioning_lag_years=float(c["commissioning_lag_years"]),
            utilisation_ceiling=float(c["utilisation_ceiling"]),
            capable_share=float(f["initial_share"]),
            training_throughput=float(f["training_throughput_per_year"]),
            skill_decay=float(f["skill_decay_rate"]),
            learning_by_doing=float(f["learning_by_doing_gain"]),
            reliability=float(i["initial"]),
            maintenance_rate=float(i["maintenance_rate"]),
            degradation_rate=float(i["degradation_rate"]),
            shock_damage=float(i["shock_damage"]),
            repair_time_years=float(i["repair_time_years"]),
            trust=float(t["initial"]),
            trust_gain=float(t["gain_per_delivered_outcome"]),
            trust_loss=float(t["loss_per_failure"]),
            trust_decay=float(t["decay_to_mean"]),
        )

    # ------------------------------------------------------------------
    @property
    def season_capacity_mu(self) -> float:
        """Capacity available inside a critical operating window."""
        return (self.capacity_units * self.unit_capacity_mu_per_day * 12.0
                * self.utilisation_ceiling)

    def expected_wait_days(self, demand_mu: float,
                           reserved_share: float = 0.0) -> float:
        """Queue delay from the ratio of demand to available capacity.

        Convex in utilisation: below capacity the wait is short, and it grows
        sharply as demand approaches the ceiling. This is the mechanism that
        makes a demand-side voucher backfire when it is not paired with
        capacity expansion.
        """
        cap = self.season_capacity_mu * (1.0 - reserved_share)
        return self.wait_days_for_capacity(demand_mu, cap)

    @staticmethod
    def wait_days_for_capacity(demand_mu: float, capacity_mu: float) -> float:
        """Queue delay for a specific provider or capacity pool."""
        cap = float(capacity_mu)
        if cap <= 0:
            return 30.0
        rho = float(demand_mu) / cap
        if rho < 0.85:
            return float(1.0 + 6.0 * rho ** 2)
        return float(min(30.0, 1.0 + 6.0 * rho ** 2 + 40.0 * (rho - 0.85) ** 2))

    # ------------------------------------------------------------------
    def step(self, dt_years: float, abm_feedback: dict, policy: dict) -> dict:
        """Advance every stock by dt and return the SD -> ABM interface dict."""
        # ---- budget --------------------------------------------------
        inflow = self.budget_inflow_annual * dt_years
        spent = float(abm_feedback.get("expenditure", 0.0))
        spent = min(spent, max(self.budget, 0.0))
        self.budget = (self.budget + inflow - spent) * (
            1.0 if dt_years < 1.0 else self.carryover_share)
        self.budget *= (1.0 + self.budget_growth * dt_years)
        self.budget = max(self.budget, 0.0)
        self.budget_spent_total += spent
        self.budget_requested_last = float(abm_feedback.get("requested", 0.0))

        # ---- service capacity: commissioning lag, then depreciation ---
        new_units = float(policy.get("new_units_this_step", 0.0))
        if new_units > 0:
            lag_steps = max(1, int(round(self.commissioning_lag_years / dt_years)))
            self.pipeline.append([lag_steps, new_units])
        arrived = 0.0
        still = []
        for entry in self.pipeline:
            entry[0] -= 1
            if entry[0] <= 0:
                arrived += entry[1]
            else:
                still.append(entry)
        self.pipeline = still
        self.capacity_units += arrived
        self.capacity_units *= (1.0 - self.depreciation_rate * dt_years)
        # Floor at a small positive value, NOT at one whole unit: a one-unit
        # floor re-inflates capacity every step for small populations and
        # silently disables the queue feedback (see audit finding M1).
        self.capacity_units = max(self.capacity_units, 1e-3)

        # ---- capable farmers: training in, decay out ------------------
        n_farmers = max(float(abm_feedback.get("n_farmers", 1.0)), 1.0)
        trained = min(policy.get("training_slots_this_step", 0.0), n_farmers)
        gain = trained / n_farmers * float(policy.get("skill_gain", 0.12))
        doing = (self.learning_by_doing * dt_years
                 * float(abm_feedback.get("effective_use_rate", 0.0)))
        decay = self.skill_decay * dt_years * self.capable_share
        if policy.get("followup_support"):
            decay *= 0.5
        self.capable_share = float(np.clip(
            self.capable_share + gain + doing - decay, 0.0, 1.0))

        # ---- infrastructure reliability ------------------------------
        damage = (self.shock_damage
                  * float(abm_feedback.get("infrastructure_shock", 0.0)))
        repair = (self.maintenance_rate * dt_years / max(self.repair_time_years, 1e-6)
                  * (1.0 - self.reliability))
        self.reliability = float(np.clip(
            self.reliability + repair - self.degradation_rate * dt_years - damage,
            0.25, 0.995))

        # ---- institutional trust -------------------------------------
        delivered = float(abm_feedback.get("delivered_rate", 0.0))
        failed = float(abm_feedback.get("failure_rate", 0.0))
        self.trust += (self.trust_gain * delivered - self.trust_loss * failed
                       - self.trust_decay * dt_years * (self.trust - 0.5))
        self.trust = float(np.clip(self.trust, 0.05, 0.95))

        state = self.interface()
        self.history.append(dict(state, budget=self.budget,
                                 capacity_units=self.capacity_units,
                                 spent=spent))
        return state

    def interface(self) -> dict:
        """The ONLY quantities the ABM may read from the SD layer."""
        return {
            "budget_available": self.budget,
            "budget_requested_last": self.budget_requested_last,
            "capacity_units": self.capacity_units,
            "season_capacity_mu": self.season_capacity_mu,
            "capable_share": self.capable_share,
            "reliability": self.reliability,
            "trust": self.trust,
        }
