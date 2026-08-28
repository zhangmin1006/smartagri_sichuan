# Sichuan Smart Agriculture — Evidence Collection Report

Generated: 2026-08-25  
Mode: curated (offline)  
Records: 183

> Observed / Estimated / Simulated must stay distinguishable at all times.

## 1. Scope decisions

### 1.1 Disruptions modelled

| ID | Shock | Family | Grade | Coupled tech | Why |
|---|---|---|---|---|---|
| D1 | Seasonal summer drought (伏旱) | hydro-meteorological | B | T2, T1 | The classic recurrent climatic feature of the Sichuan Basin and the shock with the clearest technological counter-measure: smart irrigation and fertigation act directly on the loss mechanism (soil-moisture defi… |
| D2 | Rainstorm, flood and waterlogging (暴雨洪涝) | hydro-meteorological | B | T1, T3 | Fast onset means the binding constraint is the LEAD TIME of warning and the QUEUE for machinery in the operating window, not equipment ownership. This is precisely the mechanism the service-capacity SD module a… |
| D3 | Compound heat-drought with power and network interruption (高温干旱复合冲击（伴电力/网络中断）) | compound / infrastructure | B | T1, T2, T3 | THE DIAGNOSTIC SHOCK OF THIS STUDY. Extreme heat raises irrigation demand at the same moment that hydropower shortfall and grid stress cut the electricity that pumps, sensors, base stations and platforms depend… |

**Deferred to version 2:** D4 Input price shock (fertiliser, diesel, electricity), D5 Migratory pest outbreak (fall armyworm), D6 Continuous overcast rain and hail at harvest, D7 Output price shock (hog cycle)

**Excluded:** Earthquake — High salience in Sichuan but essentially orthogonal to smart agricultural technology adoption decisions and effort level, Trade and export demand shocks — Acts on marketing and e-commerce channels rather than on production effort and disaster loss, which is the causal chain 

### 1.2 Technology bundles modelled

| ID | Bundle | Channel | Access modes | Grade | Status |
|---|---|---|---|---|---|
| T1 | Digital early warning and agro-situation information (数字预警与农情信息服务) | information | none, service | B | in_scope |
| T2 | Smart irrigation, fertigation and field sensing (智慧灌溉、水肥一体化与传感器) | biophysical | none, own, rent, service | B | in_scope |
| T3 | Drone and BeiDou smart machinery operating services (无人机植保与北斗智能农机社会化服务) | operating_capacity | none, own, service | B | in_scope |
| T4 | Livestock and aquaculture intelligent monitoring (智慧牧场与智慧渔场) | biophysical | own, service | B | out_of_scope_mvp |
| T5 | Platform, traceability and e-commerce (电商、追溯与经营平台) | market |  | A | out_of_scope_mvp |
| T6 | Precision agricultural insurance with remote sensing loss assessment (精准农业保险与遥感核损) | financial |  | B | modelled_as_policy_instrument |

Adoption ladder enforced throughout (coverage is never reported as adoption):

0. **not_accessible** — No access (network, terminal, service or eligibility missing)
1. **accessible** — Access exists but nothing acquired
2. **acquired** — Purchased, registered or granted eligibility
3. **used** — Verifiable use in the production season
4. **effectively_used** — Correct, timely, sustained use that changes a production action
5. **exited** — Disadopted after failure, cost or disappointment

### 1.3 Policy instruments modelled

| ID | Instrument | Targets | SD stock | Equity |
|---|---|---|---|---|
| P1 | Equipment purchase subsidy (设备购置补贴) | T2, T3 | government_budget | poor_favours_large_operators |
| P2 | Per-mu service voucher (按亩服务券) | T3, T2 | government_budget | favourable |
| P3 | Service capacity expansion (regional machinery service centres) (区域农机社会化服务中心扩容) | T3 | service_capacity | depends_on_allocation |
| P4 | Digital skills training (数字技能培训) | T1, T2, T3 | capable_farmers | favourable_if_targeted |
| P5 | Warning plus emergency operation guarantee (预警＋应急作业保障) | T1, T3 | ['infrastructure', 'service_capacity'] | favourable |
| P6 | Precision insurance with remote sensing loss assessment (精准农业保险与遥感核损) | T6 | government_budget | mixed |
| P7 | Mountain and hill adaptation investment (山地适配投资) | T2, T3 | service_capacity | strongly_favourable |

## 2. Policy documents inventoried

| Key | Document | Year | Grade | Verify |
|---|---|---|---|---|
| NP1 | 数字农业农村发展规划（2019—2025年） | 2020 | A |  |
| NP2 | 数字乡村发展行动计划（2022—2025年） | 2022 | A |  |
| NP3 | 全国智慧农业行动计划（2024—2028年） | 2024 | A |  |
| NP4 | 农机购置与应用补贴政策 | rolling | A |  |
| SP1 | 四川省智慧农业行动计划（2025—2028年） | 2025 | B | yes |
| SP2 | 大力发展智慧农业实施方案 | 2025 | B | yes |
| SP3 | 四川省农业农村信息化"十四五"规划 | 2021 | B |  |
| SP4 | 四川省农机购置补贴实施方案 / 农机作业补贴试点 | rolling | B |  |
| SP5 | 政策性农业保险（三大粮食作物完全成本保险等） | rolling | A |  |
| SP6 | 四川省委一号文件 | annual | A |  |

### 2.1 Provincial targets (commitments, NOT realised adoption)

| Entity | Target | Value |
|---|---|---|
| NP3 | Agricultural production informatisation rate from 27.6 per cent (2022) to 32 per cent by 2028. | None |
| SP1 | targets_by_2028: smart_farms | 200 |
| SP1 | targets_by_2028: smart_pastures | 50 |
| SP1 | targets_by_2028: smart_fisheries | 20 |
| SP1 | targets_by_2028: digital_government_scenarios | >= 7 |
| SP1 | targets_by_2028: precision_insurance_counties | 176 |
| SP1 | targets_by_2028: microclimate_station_counties | 66 |
| SP1 | targets_by_2028: smart_fertigation_counties | 40 |
| SP1 | targets_by_2028: smart_fertigation_area_mu | 1650000 |
| SP1 | targets_by_2028: regional_machinery_service_centres | >= 200 |
| SP1 | targets_by_2028: intelligent_machinery_equipment_rate | 0.6 |
| SP1 | targets_by_2028: drones | 15000 |
| SP1 | targets_by_2028: licensed_pilots | 5000 |
| SP1 | targets_by_2028: low_altitude_scenarios | 100 |
| SP1 | targets_by_2028: digital_farm_factories | >= 50 |
| SP1 | targets_by_2028: upgraded_modern_parks | 50 |
| SP1 | targets_by_2028: smart_agriculture_leading_zones | [1, 3] |

## 3. Record inventory

| Category | Records |
|---|---|
| disruption | 56 |
| policy | 73 |
| technology | 54 |

| Evidence grade | Records |
|---|---|
| A | 14 |
| B | 61 |
| C | 108 |

## 6. Data gaps, ranked

Priority 1 blocks calibration; 2 weakens a headline claim; 3 is desirable.

- Priority 1: 53 records
- Priority 2: 19 records
- Priority 3: 64 records

The single highest-value acquisition remains an anonymously linkable chain: **farmer and plot → policy receipt → verified technology use → shock exposure → loss → recovery**. Without it the adoption and effort rules can be structurally validated but not causally calibrated.

## 7. Config integrity

All cross-references between disruptions, technologies and policy instruments resolve.
