# Model Design: from Guo, Parlar & Zhang (2025) to farmer adoption and effort in Sichuan

This note sets out how the managerial-supervision model in
`2_by_2_case (5).pdf` is re-instantiated as the decision core of the ABM, and
how risk attitude becomes the organising agent attribute.

---

## 1. The source model in one page

Guo, Parlar & Zhang (2025) study an **owner** (principal, risk aversion `c1`)
who hires a **manager** (agent, risk aversion `c2`) to exert effort `e` on a
project whose outcome `X_e` is random with density `f(x; e)`, increasing in `e`
in the sense of first-order stochastic dominance. Effort and outcome are
observable and verifiable (the *first-best* case). The owner designs a wage
schedule `w(x)` and chooses the effort level to demand.

| Result | Content |
|---|---|
| Theorem 1 | `dw/dx = r_O(x−w) / [r_O(x−w) + r_M(w)]`, with `w(a) = b`; `w` increases in `x` with slope in `(0,1)` |
| Section 3 | With power utilities: `dw/dx = c1·w / (c1·w + c2·(x−w))`, exact implicit solution `w + K1·w^(c2/c1) = x` |
| Proposition 1 | `w` is **linear** if `c1 = c2`, **concave** if `c1 < c2`, **convex** if `c1 > c2` |
| Eq. (5) | Participation constraint `E[u(w(X_e))] = v(e) + U_min` binds and pins the anchor `b(e)` |
| Section 4 | `J(e) = E[B(X_e − w(X_e; e))]`; the optimum `e*` **need not be the maximum effort** |
| Table 2 | As both parties become more risk-averse, the demanded effort rises to the ceiling |
| Table 3 | With `c1 = 1/4 < c2 = 1/3` the optimum is interior at `e* = 0.9` |

The implementation in [contract.py](../src/smartagri/contract.py) reproduces
Table 2 to three decimals and Table 3 to about `1e-3`, including the interior
optimum — see [tests/test_contract.py](../tests/test_contract.py).

---

## 2. Two instantiations of the same problem

The paper's structure is used **twice**, with one solver.

### Layer S — the service contract (inside the ABM, every season)

| Paper | Sichuan smart agriculture |
|---|---|
| Owner (principal), `c1` | **Farmer** — owns the crop, receives `x − w` |
| Manager (agent), `c2` | **Service provider** — drone pilot, machinery service centre, cooperative operator |
| Effort `e` | Service effort: timeliness inside the critical window, application accuracy, number of passes, responsiveness to a warning |
| Outcome `x` | Realised gross value of output on the served plot |
| Wage `w(x)` | What the farmer pays the provider |
| `v(e) = γe²` | Provider's cost of working harder: fuel, hours, wear, foregone clients |
| `U_min` | Provider's outside option: other clients, other counties, idle capacity |

This is the natural reading for Sichuan, because the evidence is unambiguous
that the dominant adoption pathway for smallholders is **buying a service**,
not owning equipment: over 60 per cent of plots are below 0.07 ha, basin hill
plots average under 0.03 ha, and roughly 70 per cent of cultivated land sits
above 6 degrees of slope. Modelling ownership alone would misrepresent how the
technology actually reaches farmers.

### Layer G — the policy contract (the government agent)

| Paper | Policy application |
|---|---|
| Owner (principal), `c1` | **Government** — residual claimant on the social outcome |
| Manager (agent), `c2` | **Farmer** — supplies effective-use effort |
| Effort `e` | Effective use: correct, timely, sustained use that changes a production action |
| Outcome `x` | Verified outcome: yield, avoided loss, area covered |
| Wage `w(x)` | Outcome-contingent support: performance subsidy, voucher top-up, insurance indemnity |

Layer G is what makes Section 4 of the paper directly policy-relevant: the
result that **the highest effort is not always optimal for the principal**
becomes the statement that *it is not always optimal for government to push
maximum adoption intensity*, because the transfer required to induce it rises
faster than the avoided loss it buys.

---

## 3. Risk attitude as the defining agent attribute

Every agent carries one number, its power-utility coefficient `c`, with
absolute risk aversion `r(z) = c/z`.

**Farmers** (`c1`) draw from `Beta(4.5, 3.0)` mapped onto `[0.10, 0.92]`
(mean ≈ 0.60), then shifted by observables:

```
c1_i = base_i − 0.055·log(area) − 0.012·education − 0.06·coop
              + 0.0035·age − 0.10·liquidity − 0.04·digital_literacy
```

**Providers** (`c2`) draw from `Beta(2.2, 4.5)` on `[0.08, 0.85]`
(mean ≈ 0.33) — systematically less risk-averse, because they diversify
across many plots and clients.

Risk aversion also has a **state component**: low liquidity or a recent bad
season temporarily raises measured risk aversion without changing the
underlying trait, which in turn shifts contract form and contracted effort.

> **Scope note (Aug 2026).** An input-price shock was previously modelled as D4
> and drove this channel directly. It has been moved to the deferred set so
> that all three modelled shocks act through the same biophysical loss channel,
> which keeps the efficacy parameters jointly identifiable from meteorological,
> remote-sensing and loss records. The price pathway into risk aversion is
> therefore dormant; the state component now fires only on loss experience and
> low liquidity. Re-enabling it is a config change, not a code change.

### Why the *pair* matters, not the level

Proposition 1 turns the pair `(c1, c2)` into an observable contract form, and
each form corresponds to a contract that actually exists in Chinese
agricultural service markets:

| Risk-attitude pair | `w(x)` | Real contract | Who bears production risk |
|---|---|---|---|
| `c1 < c2` (farmer less averse) | concave | flat per-mu service fee | the farmer |
| `c1 = c2` | linear | proportional revenue share | shared pro rata |
| `c1 > c2` (farmer more averse) | convex | guaranteed-yield trusteeship, "floor plus share" | the provider |

This yields a falsifiable prediction: **contract form should sort on the
relative risk aversion of the two parties**, and the model's contract map
shows the effort consequence is large — mean demanded effort is 0.96 in the
concave region against 0.79 in the convex region, where over half the optima
are interior rather than at the ceiling.

---

## 4. The verifiability bridge — why smart technology matters theoretically

The paper's solution rests on effort and outcome being **observable and
verifiable**. In the field they usually are not: a farmer cannot cheaply
confirm that a drone flew the whole plot at the right altitude on the right
day, nor that the yield shortfall was weather rather than sloppy work.

This is where smart technology enters as more than a productivity tool:

- BeiDou terminals and intelligent operation monitoring make **effort**
  verifiable (operated area, track, timing, hours);
- remote sensing and plot-level crop-condition products make **outcome**
  verifiable, which is exactly what precision insurance loss assessment does.

So **monitoring technology is the enabling condition for the paper's
first-best contract**. Without it the contract is unenforceable and collapses
to a flat fee with shirking; with it, outcome-contingent contracts become
feasible and the effort predicted by Section 4 can actually be demanded.

The model implements this as a `verifiability` probability per technology.
When a contract is drawn unverifiable, delivered effort is multiplied by
`shirk_effort_multiplier`. The experiment in
[experiments.py](../src/smartagri/experiments.py) runs three regimes and
finds delivered effort of about **0.90 under full verifiability, 0.80 as
configured, and 0.48 with no monitoring**, with the effective-use rate falling
from roughly 0.45 to 0.23. That gap is the quantitative case for treating
monitoring as infrastructure rather than as a gadget.

---

## 5. The decision sequence for a farmer agent

1. **Access screen** — network, terminal, eligibility, liquidity, a water
   source for irrigation technology, a provider within reach.
2. **Mode choice** — for each technology bundle, compare `none`, `own` and
   `service` on a **money-metric** value:

   ```
   V = CE[ u(W − K − C_learn − C_access − C_switch + avoided_loss) ]
       + salience·subsidy + peer + trust
   ```

   where `CE` is the certainty equivalent of a lottery in which the avoided
   loss is only collected if a shock occurs. Risk aversion therefore discounts
   the benefit of the technology directly: **a more risk-averse farmer values
   protection against a risk they may not face less highly, not more**, because
   the payoff itself is uncertain. Choice is a logit whose scale is
   proportional to the value of output on the holding.

3. **Effort** — if the mode is `service`, solve the Guo–Parlar–Zhang problem
   for the `(farmer, provider)` pair to get demanded effort `e*`, the contract
   form and the anchor `b(e*)`. If the mode is `own`, the farmer supplies
   effort to themselves and the problem collapses to
   `max_e E[u(W + π(e))] − v(e)` with no participation constraint.

4. **Delivery** — realised effort is demanded effort, degraded by
   verifiability (shirking) and by availability (device failure, network or
   power outage, queue delay).

5. **Outcome** — loss is
   `Loss = Y_base · D(H, stage, terrain) · (1 − Σ_k η_k·e_k·ρ_k)`,
   capped at `max_total_mitigation`. Technology **never** delivers a fixed
   percentage benefit.

6. **Learning** — beliefs about efficacy, trust in platforms and policy, and
   exit decisions update from realised outcomes, peer outcomes, false alarms,
   service failures and payment delays.

---

## 6. Where the ABM ends and the SD layer begins

| Owned by the ABM | Owned by the SD layer |
|---|---|
| individual adoption state and effort | aggregate budget stock and disbursement |
| individual beliefs and trust | aggregate service capacity, with commissioning lag and depreciation |
| service requests and work orders | stock of trained, capable farmers, with skill decay |
| realised loss and recovery per farmer | infrastructure reliability |
| | aggregate institutional trust |

**The interface rule that prevents double counting:** waiting time is computed
once, in the SD layer, from SD capacity and ABM demand. Neither layer
recomputes it. The same discipline applies to budget, capacity and trust: a
quantity has exactly one owner.

This is why the hybrid is worth the complexity. The queue is a *system*
property that no individual farmer controls, but it changes every individual's
realised efficacy — and it is the reason a demand-side voucher issued without
capacity expansion can lengthen waits and reduce the loss it was meant to
prevent.

---

## 7. Reproducing the paper's numbers

```bash
cd smartagri_sichuan
python tests/test_contract.py
```

Prints Table 2, Table 3, the Proposition 1 curvature check and the Theorem 1
bounds check, and asserts agreement with the published values.
