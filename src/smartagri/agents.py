"""
agents.py
=========
Agents for the Sichuan smart-agriculture ABM.

RISK ATTITUDE IS THE DEFINING ATTRIBUTE. Every Farmer carries c_risk (the
paper's c1 when the farmer is the principal in a service contract) and every
ServiceProvider carries c_risk (the paper's c2). The PAIR determines, through
Guo, Parlar & Zhang (2025):

    * the shape of the payment schedule w(x)          -- Proposition 1
    * the anchor b(e) via the participation constraint -- Eq. (5)
    * the effort level e* the principal optimally demands -- Section 4

and e* is what feeds the loss function, so risk attitude propagates all the
way to resilience outcomes.

Three answers the module produces, one per research question:
  RQ1  adoption mode + effort level          -> decide_adoption(), contract_effort()
  RQ2  loss and recovery under shocks        -> apply_shock(), recover()
  RQ3  response to policy instruments        -> policy terms enter both above
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .contract import (LinearTiltDensity, PowerUtility,
                       quantise, solve_effort_cached, wage_curvature)

ACCESS_MODES = ("none", "own", "service")
TECHS = ("T1", "T2", "T3")

STATE_NOT_ACCESSIBLE = 0
STATE_ACCESSIBLE = 1
STATE_ACQUIRED = 2
STATE_USED = 3
STATE_EFFECTIVE = 4
STATE_EXITED = 5


# ===========================================================================
# Service provider  (the AGENT in the service contract: risk aversion c2)
# ===========================================================================
@dataclass
class ServiceProvider:
    uid: int
    county: str
    c_risk: float                       # c2
    units: float = 6.0
    capacity_weight: float = 1.0
    utilisation_ceiling: float = 0.85
    unit_capacity_mu_per_day: float = 300.0
    price_per_mu: float = 12.0
    u_min_base: float = 0.28
    service_radius_counties: tuple = ()
    backlog_mu: float = 0.0
    served_mu_season: float = 0.0
    revenue_season: float = 0.0
    mean_effort_season: float = 0.0
    _effort_accum: list = field(default_factory=list)

    @property
    def season_capacity_mu(self) -> float:
        """Capacity inside the critical operating window, not annual capacity."""
        return (self.units * self.unit_capacity_mu_per_day * 12.0
                * self.utilisation_ceiling)

    def reservation_utility(self, demand_pressure: float,
                            scaling: float = 0.35) -> float:
        """Outside option rises when the provider is in demand.

        This is the channel through which system-level congestion feeds back
        into the individual contract: a busy provider requires more to accept
        any given client, which raises b(e) and lowers the effort the farmer
        finds worth paying for.
        """
        return self.u_min_base * (1.0 + scaling * max(0.0, demand_pressure))

    def utilisation(self) -> float:
        cap = self.season_capacity_mu
        return float(self.served_mu_season / cap) if cap > 0 else 0.0

    def reset_season(self) -> None:
        self.served_mu_season = 0.0
        self.revenue_season = 0.0
        self._effort_accum = []
        self.backlog_mu = 0.0

    def close_season(self) -> None:
        self.mean_effort_season = (float(np.mean(self._effort_accum))
                                   if self._effort_accum else 0.0)


# ===========================================================================
# Farmer  (the PRINCIPAL in the service contract: risk aversion c1)
# ===========================================================================
@dataclass
class Farmer:
    uid: int
    county: str
    terrain: str
    area_mu: float
    c_risk: float                       # c1  -- the defining attribute
    wealth: float
    liquidity: float
    education_years: float
    age_years: float
    coop_member: bool
    digital_literacy: float
    irrigation: bool
    service_distance: float             # 0 = adjacent, 1 = remote
    peers: list = field(default_factory=list)

    # technology state
    state: dict = field(default_factory=lambda: {t: STATE_ACCESSIBLE for t in TECHS})
    mode: dict = field(default_factory=lambda: {t: "none" for t in TECHS})
    effort: dict = field(default_factory=lambda: {t: 0.0 for t in TECHS})
    contract_form: dict = field(default_factory=lambda: {t: "" for t in TECHS})
    provider_id: dict = field(default_factory=lambda: {t: -1 for t in TECHS})
    # Risk attitude has a STRUCTURAL component (trait plus observables, fixed
    # at construction) and a TRANSIENT stress component driven by recent
    # severe loss. c_risk is the effective value actually used in decisions;
    # c_risk_base is the structural part it decays back toward. The config
    # declared this channel as active but nothing implemented it, so risk
    # attitude was entirely exogenous.
    c_risk_base: float = 0.0
    risk_stress: float = 0.0
    # Subjective hazard probability. Beliefs previously covered technology
    # EFFICACY but not shock PROBABILITY: q came straight from the config
    # prior, so farmers could never learn that shocks had become more or less
    # frequent. A5 exposed this -- adoption was HIGHEST in the no-shock
    # scenario, because farmers kept buying protection against shocks that
    # had stopped happening. Negative if not yet initialised.
    belief_shock_prob: float = -1.0
    trained_gain: float = 0.0     # literacy gain from P4, subject to decay
    prior_loss_experience: float = 0.0
    served: dict = field(default_factory=lambda: {t: False for t in TECHS})
    experience: dict = field(default_factory=lambda: {t: 0 for t in TECHS})
    disadopted: dict = field(default_factory=lambda: {t: False for t in TECHS})
    contract_anchor: dict = field(default_factory=lambda: {t: np.nan for t in TECHS})
    contract_expected_payment: dict = field(
        default_factory=lambda: {t: np.nan for t in TECHS})
    contract_c1: dict = field(default_factory=lambda: {t: np.nan for t in TECHS})
    contract_c2: dict = field(default_factory=lambda: {t: np.nan for t in TECHS})

    # beliefs, trust, memory
    belief_efficacy: dict = field(default_factory=dict)
    trust: float = 0.55
    income_history: list = field(default_factory=list)
    loss_history: list = field(default_factory=list)
    bad_seasons: int = 0
    exited: bool = False
    seasons_below_baseline: int = 0
    pre_shock_income: float = 0.0
    recovery_seasons: float = np.nan

    # season accounting
    _last_observed: float | None = None   # realised mitigation, read by peers
    _last_atten: dict = field(default_factory=dict)  # per-tech attenuation
    last_income: float = 0.0
    last_loss: float = 0.0
    last_wait_days: float = 0.0
    last_served: bool = False
    subsidy_received: float = 0.0
    service_paid: float = 0.0
    pending_indemnities: list = field(default_factory=list)
    last_insurance_receipt: float = 0.0
    last_claim_amount: float = 0.0

    # ------------------------------------------------------------------
    def effective_risk_aversion(self, price_stress: float = 0.0,
                                stress_shift: float = 0.08,
                                loss_shift: float = 0.06) -> float:
        """c1 adjusted for transient liquidity stress and recent losses.

        Risk aversion is treated as a stable trait plus a state component:
        a cash squeeze or a fresh bad season makes a farmer behave more
        risk-averse without changing who they are. This is what lets D4
        (a pure price shock) change contract form and effort.
        """
        c = self.c_risk
        if price_stress > 0:
            c += stress_shift * price_stress
        if self.bad_seasons > 0:
            c += loss_shift * min(self.bad_seasons, 3) / 3.0
        if self.liquidity < 0.2:
            c += 0.05
        return float(np.clip(c, 0.05, 0.95))

    def utility(self, wealth: float) -> float:
        return PowerUtility(self.c_risk).u(max(wealth, 1e-9))

    # ------------------------------------------------------------------
    # RQ1a: which technologies, in which access mode
    # ------------------------------------------------------------------
    def adoption_value(self, tech: str, mode: str, ctx: dict) -> float:
        """Money-metric value of (technology, access mode) this season.

        V = CE[ seasonal income lottery | tech, mode ] + peer + trust

        Everything is expressed in CURRENCY by taking the certainty
        equivalent of the expected-utility term, so that the monetary terms
        (cost, subsidy, access) and the behavioural terms are commensurable.
        Adding a behavioural nudge directly to a utility level, as a naive
        implementation does, makes it numerically irrelevant next to u(W)
        and silently produces universal adoption.

        Risk aversion enters twice and in opposite directions: through the
        curvature of u on the loss lottery, and through the
        contract in `contract_effort`.
        """
        if mode == "none":
            gross0 = ctx["base_value_per_mu"] * self.area_mu
            base0 = max(gross0 - ctx["input_cost_per_mu"] * self.area_mu, 1.0)
            return float(self._loss_lottery_ce(
                base0, 0.0, gross0 * ctx["expected_damage_given_shock"],
                0.0, ctx))

        spec = ctx["tech_spec"][tech]
        pol = ctx["policy"]
        area = self.area_mu
        # A minimum-holding eligibility rule excludes the smallest farms from
        # capital support entirely; it does not scale their award down.
        eligible = area >= pol.get("eligibility_min_area_mu", 0.0)

        # --- capital and running cost -----------------------------------
        if mode == "own":
            if tech == "T1":
                return -np.inf                     # information is not owned
            if tech == "T2":
                if not self.irrigation:
                    return -np.inf                 # nothing to control
                capex = spec["model"]["capex_per_mu"] * area
            else:
                capex = spec["model"]["capex_own_per_unit"]
            subsidy = min(capex * (pol.get("subsidy_rate", 0.0) if eligible else 0.0),
                          pol.get("subsidy_cap", 1e12))
            # a reimbursement lag makes the nominal subsidy worth less
            lag_discount = 1.0 / (1.0 + pol.get("subsidy_lag_days", 0) / 365.0 * 0.35)
            subsidy_eff = subsidy * lag_discount
            net_capex = (capex - subsidy_eff) / max(spec["model"].get("lifetime_years", 5), 1)
            opex = spec["model"].get("opex_per_mu", 0.0) * area
            if capex - subsidy_eff > self.liquidity * self.wealth:
                return -np.inf                     # liquidity constraint binds
            k_cost = net_capex + opex
            learning = spec["model"].get("learning_cost_own",
                                         spec["model"]["learning_cost"])
        else:                                       # service
            if tech == "T2" and not self.irrigation:
                return -np.inf                     # no water source to control
            price = ctx["service_price"].get(tech, spec["model"].get(
                "service_price_per_mu", spec["model"].get("opex_per_mu", 0.0)))
            voucher = min(pol.get("voucher_per_mu", 0.0) if eligible else 0.0,
                          price) * min(area, pol.get("voucher_max_mu", 0.0))
            k_cost = price * area - voucher
            learning = spec["model"]["learning_cost"]
            subsidy_eff = voucher

        # --- capability screen ------------------------------------------
        if self.digital_literacy < spec["model"]["digital_literacy_threshold"]:
            learning *= 2.2

        # --- access cost -------------------------------------------------
        wait = ctx["expected_wait_days"].get(tech, 0.0)
        access_cost = (ctx["access_cost_scale"]
                       * (self.service_distance + 0.06 * wait))
        if mode == "own":
            access_cost *= 0.35

        # --- expected benefit -------------------------------------------
        eta = self.belief_efficacy.get(tech, spec["model"].get("eta_drought", 0.2))
        terrain_fit = spec["model"]["terrain_fit"].get(self.terrain, 1.0)
        terrain_fit = min(1.0, terrain_fit * ctx.get("terrain_fit_bonus", {}
                                                     ).get(self.terrain, 1.0))
        rho = ctx["reliability"].get(tech, 0.9)
        e_expect = ctx["expected_effort"].get((tech, mode), 0.6)
        # damage CONDITIONAL on a shock occurring; the probability itself is
        # handled by the certainty-equivalent calculation below
        exposure = ctx["expected_damage_given_shock"]
        gross = ctx["base_value_per_mu"] * area
        # NB the benefit is applied below as `mitig_frac` inside the certainty
        # equivalent, not as a separate expected-value term. An earlier
        # `avoided = gross * exposure * ...` line here was left over from the
        # expected-value formulation, was never consumed, and duplicated the
        # T1 can-act adjustment; it has been removed.

        # Learning is ONE-OFF skill acquisition, not a recurring levy. It was
        # being charged in full every season, which made it about a third of
        # total cost forever and permanently suppressed adoption. It now
        # decays with accumulated seasons of use, so a farmer pays to learn
        # once and thereafter pays only a small maintenance component.
        exp_t = int(self.experience.get(tech, 0))
        decay = 1.0 / (1.0 + ctx["learning_decay"] * exp_t)
        learn_cost = learning * ctx["learning_scale"] * decay

        # Switching friction: search, learning to transact, trust-building.
        # Proportional to the transaction it applies to, with a small fixed
        # floor. Previously a flat 1125 currency units, which exceeded the
        # entire expected benefit and made non-adoption near-absorbing.
        switch_cost = 0.0
        if self.mode.get(tech) != mode:
            switch_cost = max(ctx["switch_floor"],
                              ctx["switch_share"] * (k_cost + learn_cost))

        # ---- risk valuation: a LOSS lottery over seasonal income --------
        # Technology does not add a small certain gain; it truncates a large
        # downside. Evaluating it as a gain on top of lifetime WEALTH made the
        # gamble about 1 per cent of the reference, at which scale CRRA is
        # nearly risk-neutral for any c, so risk attitude fell to 3 per cent
        # influence. It was also inconsistent with the contract core, where
        # utility is taken over the project outcome y = x - w(x), not over
        # wealth plus the project outcome.
        #
        # The lottery is therefore stated as it actually faces the farmer,
        # over SEASONAL NET INCOME and with the shock state included:
        #     no shock : base - cost
        #     shock    : base - cost - gross * D * (1 - mitigation)
        # against the no-adoption alternative that carries the full loss.
        # A more risk-averse farmer now values loss truncation MORE, which is
        # the economically correct direction and the one the study is about.
        cost_total = k_cost + learn_cost + access_cost + switch_cost
        base_income = max(gross - ctx["input_cost_per_mu"] * area, 1.0)
        # ROUTINE BENEFIT: labour and inputs saved every season, shock or no
        # shock. Valuing technology purely as loss mitigation inverted the
        # real economics -- drone spraying is bought to displace labour, and
        # the resilience gain is a by-product. It is attenuated by terrain
        # fit and availability because equipment that cannot reach the plot
        # or is out of service delivers nothing, but NOT by shock
        # probability: it is a certain benefit, which is also why a
        # risk-averse farmer values it more than the contingent one.
        routine = (spec["model"].get("routine_benefit_per_mu", 0.0)
                   * area * terrain_fit * rho)
        if mode == "service":
            # only received if the request is actually filled
            routine *= float(ctx.get("expected_service_rate", {}).get(tech, 1.0))
        base_income += routine
        loss_if_shock = gross * exposure

        if mode == "none":
            mitig_frac = 0.0
            cost_total = 0.0
        else:
            mitig_frac = float(np.clip(
                eta * terrain_fit * rho * e_expect, 0.0, ctx["max_mitigation"]))
            if tech == "T1" and not (
                    self.mode.get("T2", "none") != "none"
                    or self.mode.get("T3", "none") != "none"
                    or self.irrigation):
                mitig_frac *= 0.35

        ce = self._loss_lottery_ce(base_income, cost_total, loss_if_shock,
                                   mitig_frac, ctx)

        # behavioural terms, in currency
        ce += ctx["subsidy_salience"] * subsidy_eff
        ce += ctx["peer_weight"] * self._peer_signal(tech, ctx) * gross * 0.02
        ce += (self.trust - 0.55) * gross * 0.03
        return float(ce)

    def subjective_hazard(self, ctx: dict) -> float:
        """The farmer's own belief about how likely a damaging season is.

        Falls back to the configured prior until the farmer has lived through
        a season, so a fresh population behaves exactly as before.
        """
        if self.belief_shock_prob < 0.0:
            return float(ctx["shock_probability"])
        return float(np.clip(self.belief_shock_prob, 0.0, 1.0))

    def update_hazard_belief(self, hit: bool, prior: float, lr: float) -> None:
        """Exponential-smoothing update toward realised shock frequency."""
        if self.belief_shock_prob < 0.0:
            self.belief_shock_prob = float(prior)
        self.belief_shock_prob = float(np.clip(
            (1.0 - lr) * self.belief_shock_prob + lr * (1.0 if hit else 0.0),
            0.0, 1.0))

    def _loss_lottery_ce(self, base_income: float, cost: float,
                         loss_if_shock: float, mitigation: float,
                         ctx: dict) -> float:
        """Certainty equivalent of the seasonal income lottery.

        Two states, over seasonal net income rather than lifetime wealth:
            prob 1-q :  base - cost
            prob q   :  base - cost - loss * (1 - mitigation)

        Because the shock state is a LARGE downside (the loss is a sizeable
        fraction of seasonal income) rather than a small gain, the curvature
        of u actually bites here. A farmer with high c1 assigns much more
        value to truncating the bad state, so risk aversion raises the
        willingness to adopt protective technology -- the direction the
        underlying theory predicts.
        """
        q = self.subjective_hazard(ctx)
        u = PowerUtility(self.c_risk)
        floor = float(ctx.get("income_floor", 200.0))

        # THREE states, not two. Collapsing damage to its conditional mean
        # discards precisely the tail that risk aversion responds to: a
        # two-point lottery on the mean makes a very risk-averse farmer look
        # almost like a risk-neutral one. The damage distribution conditional
        # on a shock is therefore represented by a moderate and a severe
        # branch drawn from the calibrated severity distribution.
        p_sev = float(ctx.get("severe_share", 0.25))
        mult_mod = float(ctx.get("damage_mult_moderate", 0.62))
        mult_sev = float(ctx.get("damage_mult_severe", 2.14))

        good = max(base_income - cost, floor)
        mod = max(base_income - cost
                  - loss_if_shock * mult_mod * (1.0 - mitigation), floor)
        sev = max(base_income - cost
                  - loss_if_shock * mult_sev * (1.0 - mitigation), floor)
        eu = ((1.0 - q) * u.u(good)
              + q * (1.0 - p_sev) * u.u(mod)
              + q * p_sev * u.u(sev))
        return float(u.inverse_u(eu))

    def _certainty_equivalent(self, w_base: float, risky_gain: float,
                              ctx: dict) -> float:
        """Legacy gain-framed CE, retained for the A/B comparison only."""
        q = self.subjective_hazard(ctx)
        u = PowerUtility(self.c_risk)
        lo = max(w_base, 1.0)
        hi = max(w_base + risky_gain, 1.0)
        eu = (1.0 - q) * u.u(lo) + q * u.u(hi)
        return float(u.inverse_u(eu))

    def _peer_signal(self, tech: str, ctx: dict) -> float:
        if not self.peers:
            return 0.0
        pop = ctx["population"]
        vals = [1.0 for p in self.peers
                if pop[p].state.get(tech, 0) in (STATE_USED, STATE_EFFECTIVE)
                and pop[p].mode.get(tech) != "none" and not pop[p].exited]
        return len(vals) / max(len(self.peers), 1)

    def decide_adoption(self, ctx: dict, rng: np.random.Generator) -> None:
        """Softmax over (mode) for each technology, then set adoption state."""
        if self.exited:
            return
        # logit scale in CURRENCY, proportional to the value of output on the
        # holding: a given absolute money difference matters less to a large
        # operator than to a smallholder
        scale = max(ctx["softmax_temperature"]
                    * ctx["base_value_per_mu"] * self.area_mu, 1.0)
        for tech in TECHS:
            spec = ctx["tech_spec"][tech]
            modes = [m for m in ACCESS_MODES if m in spec.get("access_modes", [])
                     or m == "none"]
            vals = np.array([self.adoption_value(tech, m, ctx) for m in modes])
            finite = np.isfinite(vals)
            if not finite.any():
                self.mode[tech] = "none"
                self.state[tech] = STATE_ACCESSIBLE
                continue
            v = np.where(finite, vals, -np.inf)
            v = v - np.max(v[finite])
            p = np.where(finite, np.exp(np.clip(v / scale, -60.0, 0.0)), 0.0)
            p = p / p.sum()
            choice = modes[int(rng.choice(len(modes), p=p))]

            prev = self.mode[tech]
            self.mode[tech] = choice
            if choice == "none":
                # Stopping a technology is not the same event as the farmer
                # exiting agriculture.  STATE_EXITED is reserved for the
                # farmer-level exit rule; keeping it out of this ordinal
                # technology ladder prevents a disadopter from satisfying
                # state >= ACQUIRED/EFFECTIVE reporting tests.
                self.disadopted[tech] = prev != "none"
                self.state[tech] = STATE_ACCESSIBLE
            else:
                self.disadopted[tech] = False
                self.state[tech] = max(self.state[tech], STATE_ACQUIRED)

    # ------------------------------------------------------------------
    # RQ1b: how much EFFORT -- the Guo-Parlar-Zhang layer
    # ------------------------------------------------------------------
    def contract_effort(self, tech: str, provider: ServiceProvider | None,
                        ctx: dict) -> tuple[float, str, float, float, float, float]:
        """Solve the principal-agent problem for this (farmer, provider) pair.

        Returns ``(effort, form, anchor, expected_payment, c1, c2)``.

        Service mode  -> farmer is principal (c1), provider is agent (c2):
                         exactly the paper's owner-manager problem.
        Own mode      -> the farmer supplies effort to themselves, so the
                         problem collapses to a single-agent effort choice
                         with no participation constraint.
        """
        mode = self.mode[tech]
        if mode == "none":
            return 0.0, "", np.nan, np.nan, np.nan, np.nan

        c1 = quantise(self.effective_risk_aversion(ctx.get("price_stress", 0.0)),
                      ctx["quantise_step"])

        if mode == "own":
            gamma = ctx["gamma_farmer_selfuse"]
            e = self._own_effort(c1, gamma, ctx)
            return (e, "self-supplied effort (no contract)", np.nan,
                    np.nan, c1, np.nan)

        assert provider is not None
        c2 = quantise(provider.c_risk, ctx["quantise_step"])
        pressure = ctx["demand_pressure"].get(provider.uid, 0.0)
        u_min = quantise(provider.reservation_utility(
            pressure, ctx["u_min_scaling_with_demand"]), ctx["quantise_step"])
        gamma = quantise(ctx["gamma_provider"], ctx["quantise_step"])

        e_star, _J, b_star, expected_payment, curvature, feasible = solve_effort_cached(
            c1, c2, gamma, max(u_min, 0.02), 1.0, ctx["effort_max"])

        if not feasible:
            # No wage schedule satisfies the provider's participation
            # constraint without exceeding the value of output: no contract.
            return (0.0, "infeasible (PC cannot be met)", np.nan,
                    np.nan, c1, c2)
        return (float(e_star), wage_curvature(c1, c2), float(b_star),
                float(expected_payment), c1, c2)

    def _own_effort(self, c1: float, gamma: float, ctx: dict) -> float:
        """max_e E[u(W + pi(e))] - v(e) for self-supplied effort."""
        dist = LinearTiltDensity(1.0)
        u = PowerUtility(c1)
        grid = np.linspace(0.0, ctx["effort_max"], 21)
        best_e, best_v = 0.0, -np.inf
        for e in grid:
            eu = dist.expectation(lambda x: u.u(np.maximum(x, 1e-9)), e, 81)
            val = eu - gamma * e * e
            if val > best_v:
                best_v, best_e = val, float(e)
        return best_e

    def realised_effort(self, tech: str, demanded: float, ctx: dict,
                        mode: str = "service") -> float:
        """Effort actually delivered, after verifiability and availability.

        The paper assumes the first-best case: effort and outcome are
        observable AND verifiable. In the field that holds only when a
        monitoring technology is present. Without it the contract cannot be
        enforced and the agent shirks, which is precisely why digital
        monitoring is the enabling condition for the theoretical solution.

        SELF-PROVISION HAS NO AGENCY PROBLEM. Under `own` the farmer is both
        principal and agent, so there is no one to shirk against and effort
        is delivered as chosen. This function previously applied the shirking
        lottery regardless of mode, which made owner-operators withhold
        effort from themselves 20 to 35 per cent of the time and understated
        the value of ownership relative to contracting.
        """
        if mode == "own":
            return float(np.clip(demanded, 0.0, ctx["effort_max"]))
        # `base_verifiability` is the documented no-monitoring counterfactual
        # and is the correct fallback for any technology that carries no
        # monitoring channel of its own.
        v = ctx["verifiability"].get(tech, ctx["base_verifiability"])
        if ctx["rng"].random() > v:
            demanded *= ctx["shirk_effort_multiplier"]
        return float(np.clip(demanded, 0.0, ctx["effort_max"]))


# ===========================================================================
# Cooperative: lowers access cost and raises literacy for members
# ===========================================================================
@dataclass
class Cooperative:
    uid: int
    county: str
    members: list = field(default_factory=list)
    access_discount: float = 0.45
    literacy_bonus: float = 0.08
    training_slots: int = 40


# ===========================================================================
# Government: allocates instruments subject to the SD budget stock
# ===========================================================================
@dataclass
class Government:
    c_risk: float = 0.45
    instruments: dict = field(default_factory=dict)
    spent_season: float = 0.0
    spent_total: float = 0.0
    applications: int = 0

    def terms(self, sd_state: dict) -> dict:
        """Translate active instruments into the terms farmers actually face.

        Budget exhaustion scales the subsidy rate down rather than switching
        it off, representing pro-rata rationing of an over-subscribed scheme.
        """
        inst = self.instruments
        avail = sd_state.get("budget_available", 0.0)
        req = max(sd_state.get("budget_requested_last", 1.0), 1.0)
        rationing = float(np.clip(avail / req, 0.0, 1.0)) if req > 0 else 1.0

        p1 = inst.get("P1", {})
        p2 = inst.get("P2", {})
        p6 = inst.get("P6", {})
        return {
            "subsidy_rate": p1.get("subsidy_rate", 0.0) * rationing,
            "subsidy_cap": p1.get("cap_per_farm", 25000.0),
            "subsidy_lag_days": p1.get("disbursement_lag_days", 120),
            "voucher_per_mu": p2.get("voucher_value_per_mu", 0.0) * rationing,
            "voucher_max_mu": p2.get("max_mu_per_farm", 30.0),
            "voucher_targeting": p2.get("targeting", "universal"),
            # Declared as an instrument parameter but never read, so a scheme
            # with a minimum-holding rule was indistinguishable from a
            # universal one -- the targeting question RQ3 asks about.
            "eligibility_min_area_mu": float(
                p1.get("eligibility_min_area_mu",
                       p2.get("eligibility_min_area_mu", 0.0))),
            "insurance_premium_subsidy": p6.get("premium_subsidy_rate", 0.0),
            "insurance_coverage": p6.get("coverage_ratio", 0.0),
            "insurance_payout_lag": p6.get("payout_lag_days", 60),
            "insurance_basis_risk": p6.get("basis_risk", 0.15),
            "rationing": rationing,
        }
