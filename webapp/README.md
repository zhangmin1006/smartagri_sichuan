# 四川智慧农业政策仿真平台 (Web app)

A Chinese-language browser interface over the Sichuan smart-agriculture
ABM–SD model. Users specify farmer attributes, technology parameters, policy
instruments and shocks, then run the model and read the policy effect against
a paired baseline.

---

## 启动 / Running it

```bash
cd smartagri_sichuan
python webapp/app.py                 # http://127.0.0.1:8000
python webapp/app.py --host 0.0.0.0 --port 8080   # 对外提供服务
```

No new dependencies: the app uses Flask, which was already installed.
Everything else (numpy, scipy, pandas, pyyaml) is already in
`requirements.txt`.

The bundled Flask server is a development server. For a real deployment put a
production WSGI server in front of it:

```bash
pip install waitress
cd webapp && waitress-serve --host 0.0.0.0 --port 8080 app:app   # gunicorn app:app on Linux
```

Note the app keeps job state **in process memory**, so it must run as a single
worker process. Multiple workers would each see only their own jobs and the
browser's polling would intermittently 404.

---

## 界面 / What the interface exposes

Five tabs, all driven by `config/`:

| Tab | Controls |
|---|---|
| **政策工具** | All 7 instruments P1–P7 with their decision variables, plus the 9 preset packages from `model_params.yaml` |
| **冲击情景** | D1/D2/D3 annual probability, damage ceiling, spatial correlation, D3 tech-availability collapse; plus a **forced-shock scheduler** to pin a given hazard to a given season |
| **技术参数** | T1/T2/T3 efficacies (η per hazard), capex/opex, routine benefit, learning cost, availability |
| **农户属性** | Population size, seasons, the six counties (size, area, service density, irrigation), risk-aversion Beta distributions for farmers and providers, contract parameters (verifiability, γ, reservation utility) |
| **运行设置** | Seed, replicates, baseline toggle |

Every field's range and default is read from the YAML config, not duplicated
here, so a recalibration in `config/` propagates to the interface
automatically.

---

## 结果如何解读 / How results are computed

Each run executes **two** simulations on the **same seed**: the user's policy
and a no-policy baseline. The model spawns independent RNG streams for
behaviour, weather and outcomes precisely so this comparison is paired — both
arms see the identical weather path, so the difference is the policy and not
the draw.

With `replicates > 1` the whole paired experiment repeats on consecutive
seeds. The spread across replicates is the only noise estimate available, and
any difference smaller than that spread is greyed out in the interface and
labelled *「与随机波动无法区分」*. A single replicate cannot distinguish a
policy effect from whether a shock happened to land in the window.

---

## 性能 / Performance

Essentially all model runtime is the Guo–Parlar–Zhang contract solver. Two
changes make the app interactive:

1. **`_wage_from_implicit` was optimised** (`src/smartagri/contract.py`):
   90 bisection halvings replaced by 30 halvings plus 3 Newton steps, clamped
   to the retained bracket. **3.4× faster**, and verified byte-identical —
   `tests/test_contract.py` reproduces the paper's Tables 2 and 3 to the same
   digits, and a three-scenario A/B produced identical CSV output.

2. **`contract_cache.py` persists solved contracts to disk.** The solver's own
   `lru_cache` dies with the process, and a new seed draws new risk-aversion
   values that miss it entirely — measured, baseline seed 1 cost 66 s cold and
   0.5 s warm, but seed 7 immediately cost 66 s again. A solved contract is a
   pure mathematical fact independent of scenario, policy and weather, so the
   store is shared across every run, seed, user and restart.

`webapp/cache/contract_solutions.pkl` ships pre-warmed with ~9,000 solved
contracts (225 runs over 25 seeds × 9 scenarios). Regenerate with:

```bash
python webapp/warm_cache.py
```

Measured: a paired 300-farmer × 8-season run completes in **~10 s**.

---

## ⚠ 标定问题 / Known calibration limits in the underlying model

Found by sweeping each instrument across its configured range. These are
**pre-existing properties of the model**, not of the web layer, and are
surfaced in the interface help text rather than silently corrected — fixing
them is a modelling decision that affects the written-up results.

| Instrument | Behaviour | Cause |
|---|---|---|
| **P3 服务运力扩容** | Saturates at **1 centre/year**. Wait falls 12.1 → 2.05 days at one centre; 1 and 80 are indistinguishable. | `initial_capacity_coverage` calibrates capacity to the *simulated sample's* area (≈1 unit for 500 farmers), while `new_centres_per_year × units_per_centre` is a *provincial* target (200 units/yr). `pop_scale` rescales against the 500-farmer config but never converts province → sample. The injection is ≈87× the entire stock, every season. |
| **P4 数字技能培训** | Saturates at ≈**1,000 slots/year**; `capable_share` hits an 0.80 ceiling. 1,000 and 20,000 identical. | Same province-vs-sample scale mismatch. |
| **P6 精准保险** | Moves income and fiscal cost, but adoption and mitigation are **bit-identical** at every setting. | The risk-attitude channel `policies.yaml` documents — "lowers the effective c1 and therefore shifts the equilibrium contract form and the effort" — is **not implemented**. Insurance currently only moves cash. |
| **P6 payout_lag_days** | Inert across its whole configured range [7, 180]. | The time step is one season (182.5 days), so `ceil(lag/182.5) == 1` for every value in range. Only `0` behaves differently. |

Consequence for users: **P3 and P4 can only be read as on/off switches**, not
as dose–response levers, and P6 cannot be evaluated on adoption or resilience
outcomes at all. P1, P2, P5 and P7 respond continuously — P5 correctly
reproduces the model's documented finding that reserving emergency capacity
*without* expanding it finishes below baseline.

---

## API

| Endpoint | Purpose |
|---|---|
| `GET /api/schema` | Parameter schema with Chinese labels, ranges and defaults |
| `POST /api/run` | Start a job → `{job_id}`. Body: `overrides`, `instruments`, `forced_shocks`, `seed`, `replicates`, `compare_baseline` |
| `GET /api/job/<id>` | Poll status, progress, stage, result |
| `GET /api/job/<id>/csv` | Season-by-season CSV (UTF-8 BOM, opens correctly in Excel) |
| `GET /api/health` | Cache statistics and job count |

Jobs run on a background thread and are polled, because a cold parameter
combination can exceed browser and proxy timeouts.

`MAX_WORK_UNITS` caps `n_farmers × seasons × replicates` and returns HTTP 400
rather than occupying the worker for many minutes.

---

## 文件 / Files

```
webapp/
  app.py              Flask server, job queue, CSV export
  runner.py           override application, paired policy/baseline runs
  schema.py           Chinese parameter schema derived from config/
  contract_cache.py   persistent disk-backed contract solution store
  warm_cache.py       pre-warms that store
  test_webapp.py      regression tests for this layer
  cache/              contract_solutions.pkl (ships pre-warmed)
  static/             index.html, app.js, charts.js, styles.css
```

`charts.js` draws the plots as inline SVG with no charting library, so the
interface works unchanged on an isolated network where a CDN would silently
fail and leave blank panels.

---

## 结果性质 / Status of the numbers

The contract core is validated against published results. Everything
downstream is a **conditional scenario, not a prediction**. Shock
probabilities, technology efficacies η, effort disutility γ, reservation
utility and the risk-aversion distributions are documented **priors** awaiting
calibration. The shape of the conclusions — which mechanism binds, where
congestion appears, who is left behind — is more trustworthy than any
particular number. Label outputs 模拟值 / Simulated.
