# Sichuan Smart Agriculture: adoption, effort, resilience and policy

An ABM + SD model of smart-agriculture technology adoption in Sichuan, whose
decision core is the risk-averse principal–agent model of **Guo, Parlar &
Zhang (2025)**, *Optimal Effort Under Managerial Supervision with Risk-Averse
Participants* — plus a program that collects the evidence needed to scope and
calibrate it.

Risk attitude is the organising agent attribute: every farmer and every
service provider carries a power-utility coefficient, and the **pair**
determines the contract form, the effort level, and ultimately the loss the
farmer suffers when a shock arrives.

---

## Quick start

```bash
pip install -r requirements.txt
cd src

# 1. Validate the contract core against the paper's published tables
python ../tests/test_contract.py

# 2. Collect scoping evidence (disruptions, technologies, policies)
python -m evidence.collect --offline
python -m evidence.collect --online --targets disruptions,technologies,policies,theory

# 3. Run the experiments that answer the three research questions
python -m smartagri.experiments --quick
python -m smartagri.experiments --full --reps 5

# 4. Run paired stochastic + parameter-envelope Monte Carlo analysis
python -m smartagri.monte_carlo --min-reps 30 --max-reps 100 \
  --batch-size 10 --parameter-draws 30 --inner-reps 5
```

## Streamlit app

The deployable Chinese-language policy simulator uses the same model, YAML
configuration, paired baseline design, and precomputed contract cache:

```bash
streamlit run streamlit_app.py
```

The Streamlit Community Cloud entry point is `streamlit_app.py` at the
repository root. No secrets are required.

---

## What is here

```
config/            scope decisions and parameters, all human-readable
  disruptions.yaml     which shocks, and why
  technologies.yaml    which technology bundles, and why
  policies.yaml        policy documents + the 7 modelled instruments
  model_params.yaml    population, risk-attitude distributions, SD stocks

src/smartagri/     the model
  contract.py          Guo-Parlar-Zhang solver (validated against the paper)
  agents.py            Farmer, ServiceProvider, Cooperative, Government
  shocks.py            correlated shock generator
  sd.py                System Dynamics stocks and the ABM interface
  model.py             the coupled season loop
  experiments.py       RQ1 / RQ2 / RQ3 analyses
  monte_carlo.py       paired stochastic + epistemic robustness experiments

src/evidence/      the information-collection program
  registry.py          curated knowledge base from config/
  harvesters.py        Crossref, OpenAlex, official-source probes
  report.py            CSV / JSON / Markdown exports
  collect.py           CLI

docs/
  01_model_design.md          theory -> ABM mapping
  02_abm_sd_architecture.md   architecture and the three RQs
  03_scope_decisions.md       disruptions / technologies / policies

tests/test_contract.py        reproduces Tables 2 and 3 of the paper
outputs/                      generated evidence and experiment results
```

---

## The three scope decisions

**Disruptions** — D1 summer drought (伏旱), D2 rainstorm/flood, **D3 compound
heat–drought with power and network interruption**. D3 is the diagnostic case:
it is the only shock that degrades the technology itself, and it is where
fixed-percentage benefit assumptions fail. All three act through the same
biophysical loss channel, which keeps the efficacy parameters jointly
identifiable. An input-price shock (D4) was deferred in the Aug-2026 scope
revision because it acted on cash rather than yield.

**Technologies** — T1 digital early warning (acts on *information*), T2 smart
irrigation and fertigation (acts on *biophysical state*), T3 drone and BeiDou
machinery services (acts on *operating capacity*). Three different causal
channels, three different access modes. Service-type adoption, not ownership,
is treated as the main pathway.

**Policies** — seven instruments: equipment subsidy, per-mu service voucher,
service capacity expansion, digital skills training, warning plus reserved
emergency capacity, precision insurance, and mountain adaptation.

Full reasoning in [docs/03_scope_decisions.md](docs/03_scope_decisions.md).

---

## Headline results

The values in this section came from the original illustrative experiment
suite.  Because model version 0.2.0 corrected service dispatch, annual hazard
conversion, outcome-contingent billing and insurance timing, regenerate the
suite before using numerical values in a report.  The Monte Carlo outputs carry
both configuration and source-code fingerprints to make stale results visible.

**Contract core is faithful.** Table 2 reproduced to three decimals, Table 3
to ~1e-3, including the interior optimum `e* = 0.9` — the paper's result that
demanding maximum effort is not always optimal.

**Adoption and effort fall monotonically as risk aversion rises** (adoption
0.75 → 0.32 across risk quartiles; effective use 0.60 → 0.27; effort
0.175 → 0.059). This runs against the naive intuition: the avoided loss is
itself a risky payoff, so a more risk-averse farmer discounts it *more*.

**Contract form sorts on the risk-attitude pair**, and effort follows:
concave region (flat per-mu fee) 0.965 mean demanded effort; convex region
(guaranteed-yield trusteeship) 0.796, with **56% of optima interior** — the
paper's result that maximum effort is not always optimal, reproduced at
population scale.

**Monitoring technology is load-bearing, not decorative.** Delivered effort
falls from 0.720 under full verifiability to 0.395 with no monitoring, and the
effective-use rate from 0.423 to 0.220.

**The compound shock breaks technology-based resilience.** D3 produces a 23.0%
income drop against 4.9% (drought) and 12.4% (flood), while mitigation falls to
0.9% against 4.1% under D2 — availability collapses exactly when demand peaks.
Every repeated shock doubles recovery time, from 1 season to 2.

**Cross-sectional adopter comparisons overstate the effect by up to 34×** — and
under flood the naive estimate collapses to ~zero while true mitigation is at
its highest. That is the queue: adopters who cannot be reached inside the
action window end the season no better off than non-adopters, so an
observational study run on a flood year would conclude the technology does
nothing.

**Capacity is the binding constraint.** The only scenarios that materially
improve outcomes are the two that expand service capacity: voucher + capacity
(mitigation +31%, mean wait 7.36 → 1.44 days) and the integrated package
(+35%, effective use 0.329 → 0.400), each costing about one-fourteenth of the
equipment subsidy. A voucher issued *alone* raises mean wait to 9.23 days for a
5% mitigation gain, and reserving emergency capacity without expanding it is
the only scenario that finishes **below baseline**.

---

## Status and honest limits

The contract core is **validated against published results**. Everything
downstream is a **conditional scenario, not a prediction**:

- shock probabilities, technology efficacies `η_k`, effort disutility `γ`,
  reservation utility `U_min` and the risk-aversion distributions are
  documented **priors**, listed for calibration in
  `outputs/evidence/parameters.csv`;
- micro, macro and external validation all require data not yet in hand —
  chiefly an anonymously linkable chain of *farmer and plot → policy receipt →
  verified technology use → shock exposure → loss → recovery*;
- provincial 2025 policy details are at evidence grade B pending
  primary-source verification (several provincial hosts were unreachable from
  the machine that ran the collector).

Results should be labelled **Observed / Estimated / Simulated** wherever they
are presented, and simulated outputs described as conditional scenarios.

The Monte Carlo parameter layer is an **audit envelope**, not a fitted
posterior: its uniform Latin-hypercube draws express plausible parameter
uncertainty until interview, administrative and secondary data support
empirical distributions. A short debug run is not decision-grade; use the
adaptive production command above and inspect `outputs/monte_carlo/run_manifest.json`
for the convergence flag and achieved replicate count.

---

## Reference

Guo, H., Parlar, M. and Zhang, M. (2025) *Optimal Effort Under Managerial
Supervision with Risk-Averse Participants*. Queen's Business School, Queen's
University Belfast; DeGroote School of Business, McMaster University.
