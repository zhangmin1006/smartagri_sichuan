# Audit round 2 — unwired mechanisms

Date: 2026-08-26

The A1 configuration-coverage check reported 299 declared keys, 74 never read
by code. Triage separated three groups:

* **metadata** (`calibrated_on`, `region`, `tier1_rationale`, `decision_date`,
  `selection_rationale`, policy target counts) — legitimately not model inputs;
* **dormant by design** (`input_price_shock`, `risk_aversion_shift`,
  `liquidity_stress_elasticity`, `price_multiplier_range`) — belong to the D4
  input-price shock deferred in the three-shock revision;
* **declared mechanisms that were never implemented** — the real finding.

The third group mattered because the Word deliverables describe mechanisms that
the code did not contain. Each is now either wired or removed.

---

## 1. Endogenous risk attitude  (`recent_severe_loss`, `prior_loss_experience`)

**Was:** `c_risk` was a fixed trait set at construction. Two declared channels
did nothing. The config comment asserted the opposite — *"only the
loss-experience and low-liquidity channels currently fire"* — so the file
documented behaviour the code did not have.

**Now:** risk attitude is split into a structural component (`c_risk_base`,
trait plus observables) and a transient `risk_stress` driven by severe loss,
decaying geometrically toward the structural level.

* `prior_loss_experience` is proxied by county exposure — `(1 - irrigation_share)`
  scaled by terrain — and centred so it shapes the distribution rather than
  shifting its mean. Grades C1 0.09 → C6 1.00.
* `severe_loss_threshold: auto` derives the trigger from the SAME definition
  the adoption-side risk framing uses (`damage_mult_severe` × mean damage
  given a shock), so the two meanings of "severe" in one config file cannot
  drift apart.

**Calibration check.** A fixed threshold of 0.15 was tried first and fired on
36.7% of farmer-seasons — i.e. on essentially *any* shock — because losses are
bimodal (median 0, p90 ≈ 0.47). The derived threshold (0.514) fires on 8.1%,
against the ~9% implied independently by `severe_share` (0.25) × shock
probability (~0.37). Two separately specified quantities agreeing to within a
percentage point is a genuine internal-consistency check, not a fitted result.

**Why it matters:** this closes the shock → loss → risk attitude → adoption
feedback. Risk attitude was previously exogenous, so the loop the study is
about could not operate. Correlation between county exposure and accumulated
stress is +0.32, so the channel adds heterogeneity rather than a uniform shift.

## 2. Owner-operators shirking against themselves

`realised_effort` applied the moral-hazard lottery regardless of access mode.
Under `own` the farmer is both principal and agent, so there is no one to
shirk against. Owner-operators withheld effort from themselves 20–35% of the
time, biasing the own-vs-service margin toward service — a margin RQ1 turns on.
`realised_effort` is now mode-aware.

## 3. D3 power dependence  (`power_dependent`)

`tech_avail` was applied uniformly to every technology, so a solar-charged
drone failed exactly as hard as a mains-powered irrigation pump. D3 is in the
model specifically to expose the fragility of technology-based resilience
claims, and applying outage uniformly is the one thing that hides differential
fragility. Power-dependent equipment now takes the full outage; the rest
degrades by `non_power_outage_share` (0.40) through platform and network
downtime only.

## 4. Small-world network  (`p_rewire`)

Declared `type: watts_strogatz, p_rewire: 0.10`; the code built a fixed
county-biased random graph with no rewiring, so the documented small-world
structure did not exist and the R3 peer-diffusion loop ran on the wrong
topology. Rewiring is now applied. Cross-county tie share 0.209.

## 5. `false_alarm_rate`

T1's declared false-alarm rate ("erodes trust when warnings do not verify")
was unread; trust erosion used a hard-coded 0.5. Now driven by the tech spec.

## 6. `eligibility_min_area_mu`

A minimum-holding rule was declared as a policy parameter but never read, so a
targeted scheme was indistinguishable from a universal one — precisely the
targeting question RQ3 asks. Now excludes sub-threshold farms from capital
support (exclusion, not a scaled-down award).

## 7. `base_verifiability`

The documented no-monitoring counterfactual (0.25) was unreachable. Now the
fallback for any technology carrying no monitoring channel of its own.

## 8. Config hygiene

* `choice_rule` — declarative only; softmax is the sole implemented rule. Now
  validated at construction so an unsupported value fails loudly.
* `outcome_scale_per_mu` — dead no-op left over from before the
  `theoretical_mean_outcome` scale fix. Removed.
* Dead `avoided` expected-value computation in `adoption_value`, superseded by
  the certainty-equivalent path and duplicating the T1 can-act adjustment.
  Removed. Unused imports removed across four modules.

---

## Known residuals (not fixed)

* `congestion_sensitive` and `requires_water_source` are enforced by hard-coded
  technology-name checks in `model.py` rather than read from the spec.
  Functionally correct today, brittle if a technology is added.
* A2 flags 158 hard-coded numeric literals in behavioural code (model.py 97,
  agents.py 39, sd.py 14, shocks.py 8). Not a correctness issue; each is a
  parameter that cannot currently be swept in sensitivity analysis.
* `abm_to_sd` / `sd_to_abm` / `owned_by_abm` / `owned_by_sd` document the
  coupling but are not read — the linkage is implemented directly in code.
  Harmless, but the declaration is not the source of truth it appears to be.
* Crop-calendar keys (`sowing_month`, `jointing`, `grain_filling`, `maturity`,
  `harvest_month`) are unread; seasonality is handled more coarsely.
