# Scope decisions: disruptions, technologies, policies

This memo records the three scoping choices you asked the collection program
to settle, and the reasoning behind each. The machine-readable versions live
in `config/disruptions.yaml`, `config/technologies.yaml` and
`config/policies.yaml`; the generated inventory is in
`outputs/evidence/EVIDENCE_REPORT.md`.

---

## 1. Which disruptions?

**Three modelled explicitly (Tier 1).**

| ID | Shock | Why it earns a place |
|---|---|---|
| **D1** | Seasonal summer drought (伏旱) | The recurrent climatic signature of the Sichuan Basin, and the shock with the cleanest technological counter-measure — smart irrigation acts directly on soil-moisture deficit, so the efficacy parameter is identifiable |
| **D2** | Rainstorm, flood, waterlogging (暴雨洪涝) | Fast onset, so the binding constraint is **warning lead time and machinery queue**, not equipment ownership. This is what makes voucher-versus-capacity policies separate |
| **D3** | Compound heat–drought with power/network interruption | **The diagnostic shock.** The only one that degrades the technology itself — the August 2022 heat–drought episode with power rationing is the reference event |

**Selection rule applied:** recurrent enough to give a usable frequency;
mechanistically coupled to at least one modelled technology; *discriminating*
between policy options rather than shifting everyone equally; and measurable
from obtainable data.

**Scope revision, August 2026.** An input-price shock was previously modelled
as D4. It has been moved to the deferred set. It was the only shock acting
through the **cash** constraint rather than the yield function, so it shared no
loss channel with D1–D3, no yield-side technology could address its damage, and
calibrating it required a separate price-series data chain needed nowhere else
in the model. Confining the set to hydro-meteorological and compound
infrastructure hazards keeps the three efficacy parameters jointly identifiable
from meteorological, remote-sensing and loss records. The consequence to hold
in mind is that the liquidity-to-risk-aversion channel is now dormant: the
state component of risk aversion fires only on loss experience and low
liquidity. Re-enabling D4 is a configuration change, not a code change.

D1 and D2 sit at opposite ends of the water axis, so no single technology can
be good at both — that is deliberate, and it is what forces the model to
distinguish technology bundles rather than treat "smart agriculture" as one
thing.

**D3 deserves the emphasis it gets.** Extreme heat raises irrigation demand at
the moment hydropower shortfall cuts the electricity that pumps, sensors,
gateways and base stations depend on. Technology efficacy is therefore *not
exogenous*: it collapses exactly when it is most needed. In the runs, D3
produces a 23.0% income drop against 4.9% for drought and 12.4% for flood,
while mitigation falls to 0.9% against 4.1% under D2. This is the strongest argument in
the whole model for redundancy, maintenance and reserved capacity over pure
equipment subsidy.

**Deferred:** input price shock (D4, see the scope revision above); migratory
pest outbreak (fall armyworm — excellent fit for early warning and drone
spraying, but outbreak records are patchier than meteorological series);
continuous overcast rain and hail at harvest; output price shocks (hog cycle —
belongs with the livestock module).

**Excluded on purpose:** earthquake (high salience in Sichuan but essentially
orthogonal to adoption and effort decisions — the response channel is
emergency management, not agronomy) and trade/export demand shocks (act on
marketing margins, not on production effort and disaster loss).

---

## 2. Which smart technologies?

**Three bundles, chosen because their resilience mechanisms are structurally
different and their access modes differ.**

| ID | Bundle | Acts on | Access mode | Provincial anchor |
|---|---|---|---|---|
| **T1** | Digital early warning and agro-situation information | **Information** — lead time | service / free | Micro-climate stations from 36 to 66 counties; cultivated-land digital base map; near-universal village 5G |
| **T2** | Smart irrigation, fertigation, field sensing | **Biophysical state** — soil moisture | own / rent / service | Smart fertigation from ~540k mu in 13 counties towards ~1.65m mu in 40 counties |
| **T3** | Drone and BeiDou machinery operating services | **Operating capacity** — work done inside a narrow window | service (mostly) | >200 regional service centres, ~26k terminals upgraded, >15k drones, >5k licensed pilots, 60% intelligent-equipment rate |

**Why this trio and not a longer list.** Each acts through a different causal
channel, so no single policy dominates and the model has something to
discriminate. Just as important, they differ in *how farmers get them*: T1 is
information-like and near-free, T2 is capital-intensive and ownership-like, T3
is bought per mu as a **service**.

That last point is the one the Sichuan evidence insists on. With over 60 per
cent of plots below 0.07 ha, basin hill plots averaging under 0.03 ha, and
roughly 70 per cent of cultivated land above 6 degrees of slope, individual
ownership of heavy equipment is uneconomic for most farmers. **Service-type
adoption, not ownership-type adoption, is the main pathway** — so the model
records `own`, `rent`, `service`, `use` and `effective use` separately.

**T3 is the only capacity-rationed bundle**, and that is where the SD layer
earns its place: when a correlated shock hits everyone at once, the queue
lengthens, service arrives after the action window closes, and realised
efficacy falls even though every farmer is nominally an adopter.

**Out of scope for the MVP:** livestock and aquaculture monitoring (T4 — real
adoption path, but different technology, shocks and instruments); platform,
traceability and e-commerce (T5 — acts on marketing margins, not production
effort and loss). **Precision insurance (T6) is modelled as a policy
instrument (P6)**, not a technology, because from the farmer's point of view it
changes the payoff distribution rather than the production function.

**The adoption ladder is enforced throughout:**
`not accessible → accessible → acquired → used → effectively used → exited`.
5G coverage, platform registration, training headcount and demonstration-park
counts are evidence for *accessible* and *acquired* only. They are never used
to initialise or validate *used* or *effectively used*.

---

## 3. Which policies?

**Documents inventoried.** National: Digital Agriculture and Rural Development
Plan (2019–2025); Digital Village Development Action Plan (2022–2025); National
Smart Agriculture Action Plan (2024–2028, informatisation rate 27.6% → 32%);
the machinery purchase and application subsidy scheme. Provincial: Sichuan
Smart Agriculture Action Plan 2025–2028 (川农发〔2025〕20号); Implementation
Plan for Vigorously Developing Smart Agriculture (川农发〔2025〕19号); the 14th
Five-Year informatisation plan; Sichuan machinery purchase and *operation*
subsidy pilots (31 counties, 2019–2022); policy agricultural insurance; the
annual provincial No.1 Document.

> **Caveat carried through the code:** several 2025 provincial documents are
> reachable mainly through policy-database mirrors. Verify article numbers and
> clause wording against the Sichuan Department of Agriculture and Rural
> Affairs originals before quoting them in a published evaluation. The
> collector flags these records with `verify = true`.

**Seven instruments modelled**, each mapped to the mechanism by which it
changes a decision:

| ID | Instrument | Changes | SD stock drawn on | Equity |
|---|---|---|---|---|
| P1 | Equipment purchase subsidy | acquisition cost `K`; disbursement lag interacts with liquidity | budget | poor — favours those who can pre-finance |
| P2 | Per-mu service voucher | service price; shifts choice from ownership to service | budget | favourable |
| P3 | Service capacity expansion | queue length and expected wait | capacity | depends on allocation |
| P4 | Digital skills training | learning cost, probability of *effective* use | capable farmers | favourable if targeted |
| P5 | Warning + reserved emergency capacity | lead time **and** ability to act on it | infrastructure + capacity | favourable |
| P6 | Precision insurance with remote-sensing loss assessment | the payoff distribution, hence effective `c1` | budget | mixed — basis risk excludes atypical plots |
| P7 | Mountain and hill adaptation | terrain-fit ceiling and effective distance | capacity | strongly favourable |

**Targets are treated as commitments, never as adoption.** Every provincial
target figure enters the model as an **upper bound on supply**, and the
exporter refuses to emit it as an adoption baseline. The scenario question is
not "will the targets be met" but "if they are met, does loss actually fall,
and for whom".

**Evaluation is multi-objective by construction.** Avoided loss, recovery
time, fiscal cost, effective-use rate, equity gap, service backlog and exit
rate are reported together with a Pareto front. No weighted-sum "optimal
policy" is emitted: the weights are a political choice belonging to the
government user, not a modelling assumption.

---

## 4. What the collection program does with all this

```bash
cd smartagri_sichuan/src
python -m evidence.collect --offline                       # curated only
python -m evidence.collect --online --targets disruptions,technologies,policies,theory
```

Offline mode uses the curated registry and always succeeds. Online mode adds
peer-reviewed literature from Crossref and OpenAlex (relevance-filtered and
de-duplicated across sources) and probes the official source list, degrading
gracefully when a host is unavailable.

Outputs: `evidence.csv`, `evidence.json`, `parameters.csv` (every model prior
flagged for calibration), `data_gaps.csv` (ranked by whether the gap blocks
calibration) and `EVIDENCE_REPORT.md`.

**Known access limitation from this machine:** of seven official sources, only
the Ministry of Emergency Management responded. moa.gov.cn disallows the
crawler in `robots.txt`; the Sichuan provincial government, agriculture
department and statistics bureau hosts timed out; the meteorological service
returned HTTP 406. These are recorded as access-check records rather than
silently dropped, and the provincial policy content in `config/policies.yaml`
therefore remains at evidence grade B pending primary-source verification.
