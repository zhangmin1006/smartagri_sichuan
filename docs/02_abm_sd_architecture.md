# ABM + SD architecture and how it answers the three research questions

## 0. The division of labour

```
                    ┌──────────────────────────────────────────┐
                    │  SD LAYER  (slow, aggregate, annual)     │
                    │  budget · service capacity · trained     │
                    │  farmers · infrastructure · trust        │
                    └───────┬──────────────────────▲───────────┘
       subsidy rate,        │                      │  expenditure, service
       capacity, wait time, │                      │  demand, failures,
       training slots,      │                      │  effective-use rate,
       reliability, trust   ▼                      │  avoided loss
                    ┌──────────────────────────────┴───────────┐
                    │  ABM LAYER  (fast, individual, seasonal) │
                    │  farmers · providers · cooperatives ·    │
                    │  government · shock environment          │
                    │                                          │
                    │   ┌────────────────────────────────┐     │
                    │   │ CONTRACT CORE                  │     │
                    │   │ Guo, Parlar & Zhang (2025)     │     │
                    │   │ (c1, c2) -> w(x), b(e), e*     │     │
                    │   └────────────────────────────────┘     │
                    └──────────────────────────────────────────┘
```

**Why not ABM alone.** Adoption, peer diffusion, queueing and loss are all
individual-level and belong in the ABM. But budget stocks, equipment
commissioning lags, skill decay, infrastructure depreciation and institutional
trust are stock-and-flow quantities with multi-year feedback. Representing
them agent by agent adds parameters without adding insight.

**Why not SD alone.** The central mechanisms are heterogeneous: who adopts
depends on risk aversion, plot size, slope, literacy and distance; contract
form depends on a *pair* of risk attitudes; and queueing outcomes depend on
who asks for service first. An aggregate model cannot represent any of that.

**The rule that keeps it honest.** Every quantity has exactly one owner.
Waiting time is computed once, in the SD layer, from SD capacity and ABM
demand. Duplicating a state variable across the two layers is the standard
failure mode of hybrid models, and the interface contract in
`model_params.yaml` (`sd.interface_contract`) forbids it explicitly.

---

## 1. RQ1 — How do farmers decide adoption and effort?

**Two nested decisions, not one.**

*Adoption* is a discrete choice over `{none, own, service}` per technology
bundle, on a money-metric value function (certainty equivalent of ending
wealth, plus subsidy salience, peer signal and trust). Farm size, slope,
irrigation, literacy, liquidity, distance and risk aversion all enter.

*Effort* is then the solution of the principal–agent problem for the specific
`(farmer, provider)` pair — this is where the paper does the work. The output
is the demanded effort `e*`, the contract form (Proposition 1) and the anchor
`b(e*)` from the binding participation constraint.

**What the model produces.**

| Risk quartile | mean `c1` | mean area (mu) | adopt any | effective use | mean effort |
|---|---|---|---|---|---|
| Q1 least averse | 0.25 | 20.8 | 0.746 | 0.603 | 0.175 |
| Q2 | 0.39 | 14.8 | 0.645 | 0.435 | 0.130 |
| Q3 | 0.51 | 11.6 | 0.532 | 0.403 | 0.091 |
| Q4 most averse | 0.64 | 8.5 | 0.317 | 0.270 | 0.059 |

Adoption and effort both fall as risk aversion rises. Note this runs *against*
the naive intuition that risk-averse farmers should buy more protection: the
avoided loss is itself a risky payoff (collected only if a shock occurs), so a
more risk-averse farmer discounts it more heavily, while also being smaller,
more liquidity-constrained and further from service.

**The contract map** over the `(c1, c2)` plane:

| Region | Contract form | mean demanded effort | share interior |
|---|---|---|---|
| `c1 < c2` | concave — flat per-mu fee | 0.965 | 0.11 |
| `c1 = c2` | linear — revenue share | 0.861 | 0.35 |
| `c1 > c2` | convex — guaranteed-yield trusteeship | 0.796 | 0.56 |

The convex region is where the paper's central result bites: over half the
optima are **interior**, i.e. the farmer optimally demands *less* than maximum
effort, because inducing more costs more in payment than it returns in
avoided loss.

**The verifiability result** (why smart technology is theoretically load-bearing):

| Monitoring regime | delivered effort | effective-use rate |
|---|---|---|
| First-best (fully verifiable) | 0.720 | 0.423 |
| Partial monitoring (as configured) | 0.647 | 0.376 |
| No monitoring (shirking) | 0.395 | 0.220 |

Removing monitoring cuts delivered effort by about 40 per cent and the
effective-use rate by nearly half.

---

## 2. RQ2 — How does smart technology change resilience to shocks?

Resilience is measured on five dimensions, not one: **resistance** (loss
magnitude), **recovery** (seasons to return to the counterfactual path),
**adaptation** (next-season behaviour change), **cumulative** loss area, and
**survival** (exit).

The design uses a **paired counterfactual**: the same population, same seed,
same seasons, with and without the shock. Without that pairing the rising
adoption trend swamps the shock and income appears to *rise* in shock seasons.

| Shock design | income drop | recovery (seasons) | loss | causal mitigation |
|---|---|---|---|---|
| D1 drought | 4.9% | 1 | 0.056 | 0.013 |
| D2 flood | 12.4% | 1 | 0.087 | 0.041 |
| **D3 compound (tech degraded)** | **23.0%** | 1 | **0.173** | **0.009** |
| D1 then D2 (repeat, different type) | 4.9% | **2** | 0.056 | 0.013 |
| D2 then D3 (escalating) | 12.4% | **2** | 0.087 | 0.041 |
| D3 twice (repeated compound) | 23.0% | **2** | 0.173 | 0.009 |

**The headline finding is D3.** Compound heat–drought with power and network
interruption produces a 23.0 per cent income drop against 4.9 per cent for
drought and 12.4 per cent for flood, *and* mitigation falls to 0.9 per cent
against 4.1 per cent under D2 — because technology availability collapses
exactly when demand for it peaks. Any model that applies a fixed percentage
loss reduction to smart technology will overstate resilience precisely where it
matters most.

**Every repeated shock doubles the recovery time.** All three two-season
designs take 2 seasons to return to the counterfactual path against 1 for a
single event, and cumulative loss area roughly doubles. Recovery capacity, not
resistance to the first blow, is what a second shock tests.

**A methodological finding worth reporting on its own.** The model computes
two estimates of the technology effect side by side:

- *naive* — cross-sectional adopter vs non-adopter loss gap;
- *causal* — within-farmer counterfactual (same farmer, same shock, damage
  with vs without mitigation).

The naive estimate exceeds the causal one by roughly **34x** for D1 and **9x**
for D3. Adopters are larger, flatter, better irrigated and closer to service,
so a cross-sectional comparison mostly measures selection, not treatment.

Under D2 the naive estimate now collapses to **approximately zero** (−0.003)
while the true mitigation is 4.1 per cent — its highest value of any shock.
This is the queue at work: flood is the shock where service capacity binds
hardest, so adopters who cannot be reached inside the action window end the
season no better off than non-adopters. An observational study run on a flood
year would conclude the technology does nothing, and would be wrong.
This quantifies, inside a controlled setting, exactly the caution the evidence
report raises about extrapolating from demonstration cases.

---

## 3. RQ3 — How effective are policies, and what roles do they play?

Every scenario faces the **same shock sequence** in each replicate, so
differences are attributable to policy rather than to shock luck. Results are
reported across all objectives simultaneously and a Pareto front is emitted;
no single weighted score is produced, because the weights are a political
choice belonging to the government user.

Results over 9 scenarios x 8 replicates (250 farmers, 8 seasons, three shocks,
common shock sequence per replicate):

| Scenario | mitigation | avoided loss | effective use | mean wait (d) | peak wait (d) | income p10 | fiscal cost | equity gap | mountain gap |
|---|---|---|---|---|---|---|---|---|---|
| baseline | 0.0144 | 0.0013 | 0.329 | 7.36 | 13.82 | 2241 | 0 | 0.0029 | 0.0022 |
| insurance | 0.0142 | 0.0013 | 0.330 | 7.30 | 13.60 | 2247 | 1,392,902 | 0.0029 | 0.0022 |
| equipment subsidy | 0.0172 | 0.0018 | 0.341 | 4.81 | 6.24 | 2112 | 703,585 | 0.0039 | 0.0028 |
| service voucher | 0.0152 | 0.0014 | 0.333 | **9.23** | **17.76** | 2249 | 50,867 | 0.0031 | 0.0025 |
| **voucher + capacity** | 0.0188 | 0.0018 | 0.371 | 1.44 | 4.49 | 2244 | 52,805 | 0.0037 | 0.0028 |
| training + maintenance | 0.0169 | 0.0017 | 0.354 | 7.67 | 12.75 | 2223 | 0 | 0.0039 | 0.0030 |
| warning + reserved capacity | **0.0134** | 0.0012 | 0.310 | **12.75** | **21.15** | 2203 | 0 | 0.0031 | 0.0023 |
| mountain adaptation | 0.0159 | 0.0015 | 0.337 | 7.74 | 12.39 | 2238 | 0 | 0.0032 | 0.0025 |
| **integrated package** | **0.0194** | 0.0018 | **0.400** | 1.63 | 6.02 | 2226 | 40,644 | 0.0036 | 0.0027 |

**Capacity is the binding constraint, and it is now unambiguous.** The only two
scenarios that materially improve mitigation are the two that expand service
capacity: voucher + capacity (+31% over baseline) and the integrated package
(+35%, and the best effective-use rate at 0.400). Both cost around 40–53
thousand — roughly one-fourteenth of the equipment subsidy, which buys less.

> **Read the equity columns with care.** Baseline shows the *lowest* equity and
> mountain gaps of any scenario, and appears on the Pareto front for that
> reason. That is not an achievement: gaps are small at baseline because almost
> nobody is protected. Any instrument that raises protection also widens the
> gap between those who can reach service and those who cannot. Equity gap must
> therefore be read jointly with the level of protection, never on its own —
> which is precisely why the model refuses to emit a single weighted score.

Three roles emerge, and they are not substitutes:

1. **Demand-side instruments convert money into queue.** The voucher alone
   raises mean wait from 7.36 to 9.23 days and peak wait from 13.8 to 17.8,
   while lifting mitigation by only 5 per cent. It moves farmers into adoption
   without any ability to serve them. The equipment subsidy does better on
   mitigation (+19 per cent) but costs 703,585 — thirteen times the voucher and
   seventeen times the integrated package — and it delivers the *worst*
   tenth-percentile income of any scenario, because it rewards farmers who
   could pre-finance the purchase anyway.

2. **Supply-side instruments are the only ones that move the outcome.**
   Voucher + capacity cuts mean wait from 7.36 to 1.44 days and lifts
   mitigation 31 per cent. The integrated package does slightly better on both
   and adds the largest gain in effective use (0.329 to 0.400) at the lowest
   cost of any funded scenario. Mountain adaptation raises mitigation 10 per
   cent without spending budget, by lifting the terrain-fit ceiling that caps
   what any subsidy can achieve on slope land.

3. **Reserving capacity without expanding it is actively harmful.** The
   warning-plus-reserved-capacity scenario is the only one that finishes *below
   baseline* on mitigation (0.0134 against 0.0144), while pushing mean wait to
   12.75 days and peak wait to 21.15. Holding capacity back for emergencies
   starves the routine queue, and the reserved capacity cannot compensate
   because the warning lead-time lever is not yet wired through to the shock
   (see the audit finding on unimplemented instrument levers). This scenario
   should be treated as **not yet a fair test of the instrument**.

**Insurance does not reduce loss, and is not designed to.** Its mitigation rate
is indistinguishable from baseline because it changes the payoff distribution
rather than the production function. What it buys is downside protection, and
at 1.39 million it is the most expensive scenario in the set. Judging it on
avoided loss is a category error; judging it on tenth-percentile income and
recovery speed is the right test, and on those it is roughly neutral here
because the payout lag still exceeds the replanting window.

---

## 4. Validation status

| Level | Status |
|---|---|
| Contract core against published results | **Done** — Tables 2 and 3 reproduced, incl. the interior optimum |
| Structural validation | Partial — agent set and rules follow the roadmap; needs practitioner review |
| Extreme-condition tests | Partial — zero subsidy, forced shocks, no-monitoring regime all run |
| Micro validation | **Not done** — requires the farmer panel and platform logs |
| Macro validation | **Not done** — requires county adoption and loss series |
| External validation | **Not done** — requires a held-out county or shock year |

Everything the model currently produces is a **conditional scenario**, not a
prediction. The parameters carrying the most weight — `η_k` efficacy, shock
probabilities, effort disutility `γ`, reservation utility `U_min`, and the
risk-aversion distributions — are documented priors flagged for calibration in
`outputs/evidence/parameters.csv`.
