# -*- coding: utf-8 -*-
"""
schema.py
=========
Builds the Chinese-language parameter schema the browser renders, DERIVED FROM
the YAML configuration rather than duplicated from it.

The point of deriving it is that defaults, admissible ranges and the policy
instrument list are stated once, in config/, where the modelling team already
maintains them alongside their justifications. A hand-written copy in the web
layer would drift the first time a parameter was recalibrated, and the
interface would then offer a range the model no longer supports.

Only the Chinese display labels and help strings live here, because the config
files carry `name_zh` for policies, technologies and shocks but not for the
individual numeric fields inside them.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"


def _load(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Chinese labels for the technology `model` fields we expose.
# (field, label, min, max, step, help)
# ---------------------------------------------------------------------------
TECH_FIELDS = [
    ("eta_drought", "干旱减损效率 eta", 0.0, 0.9, 0.01,
     "该技术在伏旱(D1)条件下可消除的产量损失比例"),
    ("eta_flood", "洪涝减损效率 eta", 0.0, 0.9, 0.01,
     "该技术在暴雨洪涝(D2)条件下可消除的产量损失比例"),
    ("eta_compound", "复合冲击减损效率 eta", 0.0, 0.9, 0.01,
     "高温干旱复合冲击(D3)下的减损效率。实际效果还会被断电导致的可用性下降进一步削弱"),
    ("routine_benefit_per_mu", "常规收益 (元/亩/季)", 0.0, 80.0, 1.0,
     "无论是否发生灾害都会产生的日常收益，例如省工、省肥。这才是农户真正的采纳动机"),
    ("opex_per_mu", "运行成本 (元/亩/季)", 0.0, 100.0, 1.0, "每亩每季的运行与维护成本"),
    ("learning_cost", "学习成本", 0.0, 1.0, 0.01, "掌握该技术的难度，直接抑制有效使用率"),
    ("digital_literacy_threshold", "数字素养门槛", 0.0, 1.0, 0.01,
     "数字素养低于该门槛的农户难以有效使用该技术"),
    ("availability_base", "基础可用性", 0.3, 1.0, 0.01, "正常年份技术可正常工作的概率"),
]

# Fields that exist on only some bundles; included when present.
TECH_OPTIONAL = [
    ("capex_per_mu", "购置成本 (元/亩)", 0.0, 2000.0, 10.0, "每亩一次性投资"),
    ("capex_own_per_unit", "自购成本 (元/台)", 0.0, 200000.0, 1000.0,
     "自有模式下每台设备的购置价"),
    ("service_price_per_mu", "服务价格 (元/亩)", 0.0, 60.0, 0.5, "购买社会化服务的市场价"),
    ("failure_rate_annual", "年故障率", 0.0, 0.6, 0.01, "设备年度故障概率"),
    ("capacity_mu_per_unit_per_day", "单机日作业能力 (亩/天)", 20.0, 800.0, 10.0,
     "决定服务排队长度的核心参数"),
]

SHOCK_FIELDS = [
    ("annual_probability_prior", "年发生概率", 0.0, 1.0, 0.005,
     "该灾害每年发生的概率，已由 1991-2024 年 ERA5 再分析资料校准"),
    ("yield_damage_max", "最大产量损失", 0.0, 1.0, 0.01, "严重度为 1 时的产量损失上限"),
    ("spatial_correlation", "空间相关性", 0.0, 0.99, 0.01,
     "各县同时受灾的程度。相关性越高，同期服务需求越集中，排队越拥堵"),
]

SHOCK_OPTIONAL = [
    ("tech_availability_multiplier", "技术可用性乘数", 0.05, 1.0, 0.01,
     "灾害期间技术仍可使用的比例。D3 的断电断网使其骤降，这是本研究的诊断性机制"),
    ("irrigation_offset", "灌溉抵消系数", 0.0, 1.0, 0.01, "既有灌溉设施已经抵消的旱灾损失"),
    ("action_window_days", "可行动窗口 (天)", 0.5, 10.0, 0.5, "灾害锁定损失前可采取行动的时间"),
]

COUNTY_FIELDS = [
    ("n_farmers", "农户数", 0, 600, 10),
    ("mean_area_mu", "户均面积 (亩)", 1.0, 80.0, 0.5),
    ("service_density", "服务密度", 0.0, 1.0, 0.05),
    ("irrigation_share", "灌溉比例", 0.0, 1.0, 0.05),
]

COUNTY_LABELS = {
    "C1": "平原 - 服务密集", "C2": "平原 - 混合", "C3": "丘陵 - 细碎",
    "C4": "丘陵 - 易涝", "C5": "丘陵 - 易旱", "C6": "山区 - 偏远",
}

TERRAIN_ZH = {"plain": "平原", "hill": "丘陵", "mountain": "山区"}

RISK_FIELDS = [
    {"key": "farmer_alpha", "label": "农户风险规避 Beta alpha", "min": 0.5, "max": 12.0,
     "step": 0.1, "path": ["risk_attitude", "farmer", "params", 0],
     "help": "农户风险规避系数 c1 的分布形状。alpha 越大，整体越规避风险"},
    {"key": "farmer_beta", "label": "农户风险规避 Beta beta", "min": 0.5, "max": 12.0,
     "step": 0.1, "path": ["risk_attitude", "farmer", "params", 1],
     "help": "beta 越大，整体越不规避风险。风险规避程度上升时采纳率与努力水平同步下降"},
    {"key": "provider_alpha", "label": "服务商风险规避 Beta alpha", "min": 0.5, "max": 12.0,
     "step": 0.1, "path": ["risk_attitude", "provider", "params", 0],
     "help": "服务商风险规避系数 c2 的分布形状"},
    {"key": "provider_beta", "label": "服务商风险规避 Beta beta", "min": 0.5, "max": 12.0,
     "step": 0.1, "path": ["risk_attitude", "provider", "params", 1],
     "help": "c1 与 c2 的相对位置决定契约形态（凹或凸）与均衡努力，是模型的理论核心"},
]

BEHAVIOUR_FIELDS = [
    {"key": "base_verifiability", "label": "作业可核查性（无监测）", "min": 0.0, "max": 1.0,
     "step": 0.05, "path": ["contract", "verifiability", "base_verifiability"],
     "help": "没有数字监测手段时作业质量可被核查的程度。可核查性是契约理论解成立的前提，"
             "它决定服务商是否会偷懒。模型显示该参数是决定性的，而非装饰性的"},
    {"key": "shirk_effort_multiplier", "label": "偷懒时的努力折扣", "min": 0.1, "max": 1.0,
     "step": 0.05, "path": ["contract", "verifiability", "shirk_effort_multiplier"],
     "help": "作业无法核查时，服务商实际交付的努力占承诺水平的比例"},
    {"key": "gamma_provider", "label": "服务商努力成本 gamma", "min": 0.02, "max": 0.9,
     "step": 0.01, "path": ["contract", "gamma_provider"],
     "help": "服务商努力的负效用系数 v(e)=gamma·e²，越高越不愿投入努力"},
    {"key": "u_min_provider_base", "label": "服务商保留效用", "min": 0.05, "max": 0.9,
     "step": 0.01, "path": ["contract", "u_min_provider_base"],
     "help": "服务商的外部选择价值。过高会使参与约束无法满足，契约不成立"},
    {"key": "social_learning_weight", "label": "社会学习权重", "min": 0.0, "max": 1.0,
     "step": 0.05, "path": ["behaviour", "social_learning_weight"],
     "help": "同伴示范效应在采纳决策中的权重"},
    {"key": "base_yield_value_per_mu", "label": "亩均产值 (元)", "min": 400.0,
     "max": 3000.0, "step": 50.0, "path": ["production", "base_yield_value_per_mu"],
     "help": "正常年份每亩产出的毛价值"},
]


def _num(key, label, value, lo, hi, step, help_=""):
    return {"key": key, "label": label, "default": value, "min": lo,
            "max": hi, "step": step, "help": help_, "type": "number"}


def _dig(cfg: dict, path: list):
    cur = cfg
    for p in path:
        cur = cur[p]
    return cur


def _step_for(lo, hi) -> float:
    span = float(hi) - float(lo)
    if span <= 1.5:
        return 0.01
    if span <= 40:
        return 0.5
    if span <= 400:
        return 1.0
    return max(round(span / 200.0), 1)


def build_schema() -> dict:
    params = _load("model_params.yaml")
    tech = _load("technologies.yaml")
    disrupt = _load("disruptions.yaml")
    policy = _load("policies.yaml")

    # -- population ------------------------------------------------------
    pop = params["population"]
    population = {
        "fields": [
            _num("n_farmers", "农户数量", pop["n_farmers"], 100, 2000, 50,
                 "模拟的农户总数。数量越大结果越稳定但运行越慢；"
                 "运力等存量会按人口自动等比缩放，因此不同规模的结果可比"),
            _num("seasons", "模拟季数", params["meta"]["seasons"], 2, 30, 1,
                 "每年 2 季，12 季即 6 年"),
            _num("n_providers", "服务商数量", pop["n_providers"], 2, 60, 1,
                 "提供社会化服务的经营主体数量"),
        ],
        "counties": [
            {"id": c["id"],
             "label": COUNTY_LABELS.get(c["id"], c["id"]),
             "terrain": TERRAIN_ZH.get(c.get("terrain"), c.get("terrain")),
             "fields": [_num(c["id"] + "." + f, lab, c[f], lo, hi, st)
                        for f, lab, lo, hi, st in COUNTY_FIELDS]}
            for c in pop["counties"]
        ],
        "risk": [dict(f, default=_dig(params, f["path"]), type="number")
                 for f in RISK_FIELDS],
        "behaviour": [dict(f, default=_dig(params, f["path"]), type="number")
                      for f in BEHAVIOUR_FIELDS],
    }

    # -- technologies ----------------------------------------------------
    techs = []
    for b in tech["bundles"]:
        if b["id"] not in ("T1", "T2", "T3"):
            continue        # T4-T6 are declared in config but out of scope
        m = b["model"]
        fields = [_num(b["id"] + "." + f, lab, m[f], lo, hi, st, hp)
                  for f, lab, lo, hi, st, hp in TECH_FIELDS if f in m]
        fields += [_num(b["id"] + "." + f, lab, m[f], lo, hi, st, hp)
                   for f, lab, lo, hi, st, hp in TECH_OPTIONAL if f in m]
        techs.append({
            "id": b["id"], "name_zh": b["name_zh"], "name_en": b["name_en"],
            "channel": CHANNEL_ZH.get(b["channel"], b["channel"]),
            "mechanism": MECHANISM_ZH.get(
                b["id"], " ".join(str(b.get("resilience_mechanism", "")).split())),
            "fields": fields,
        })

    # -- shocks ----------------------------------------------------------
    shocks = []
    for d in disrupt["tier1"]:
        m = d["model"]
        fields = [_num(d["id"] + "." + f, lab, m[f], lo, hi, st, hp)
                  for f, lab, lo, hi, st, hp in SHOCK_FIELDS if f in m]
        fields += [_num(d["id"] + "." + f, lab, m[f], lo, hi, st, hp)
                   for f, lab, lo, hi, st, hp in SHOCK_OPTIONAL if f in m]
        shocks.append({
            "id": d["id"], "name_zh": d["name_zh"], "name_en": d["name_en"],
            "family": FAMILY_ZH.get(d["family"], d["family"]),
            "why": MECHANISM_ZH.get(
                d["id"], " ".join(str(d.get("why_included", "")).split())),
            "fields": fields,
        })

    # -- policy instruments ---------------------------------------------
    instruments = []
    for i in policy.get("instruments", []):
        dvs = []
        for name, spec in (i.get("decision_variables") or {}).items():
            entry = {"key": name, "label": DV_LABELS.get(name, name),
                     "default": spec.get("default"),
                     "help": DV_HELP.get(name, "")}
            if "range" in spec:
                lo, hi = spec["range"]
                entry.update(type="number", min=lo, max=hi, step=_step_for(lo, hi))
            elif "options" in spec:
                opts = spec["options"]
                if set(type(o) for o in opts) == {bool}:
                    entry.update(type="bool")
                else:
                    entry.update(type="select", options=[
                        {"value": o, "label": OPT_LABELS.get(o, str(o))}
                        for o in opts])
            else:
                entry.update(type="number", min=0, max=1, step=0.01)
            dvs.append(entry)
        instruments.append({
            "id": i["id"], "name_zh": i["name_zh"], "name_en": i["name_en"],
            "mechanism": MECHANISM_ZH.get(
                i["id"], " ".join(str(i.get("abm_mechanism", "")).split())),
            "side_effects": [SIDE_EFFECT_ZH.get(x, x)
                             for x in i.get("known_side_effects", [])],
            "equity_flag": EQUITY_ZH.get(i.get("equity_flag"), i.get("equity_flag")),
            "variables": dvs,
        })

    presets = [{"key": s["key"], "label": PRESET_ZH.get(s["key"], s["label"]),
                "instruments": s.get("instruments") or {}}
               for s in params.get("scenarios", [])]

    return {"population": population, "technologies": techs, "shocks": shocks,
            "instruments": instruments, "presets": presets,
            "objectives": OBJECTIVES, "metrics": METRIC_LABELS,
            "model_version": params["meta"]["version"],
            "default_seed": params["meta"]["seed"]}


CHANNEL_ZH = {"information": "信息渠道", "biophysical": "生物物理渠道",
              "operating_capacity": "作业运力渠道"}

FAMILY_ZH = {"hydro-meteorological": "水文气象类",
             "compound / infrastructure": "复合 / 基础设施类"}

# ---------------------------------------------------------------------------
# Chinese renderings of the mechanism narratives.
#
# The config files carry `name_zh` for every instrument, technology and shock,
# but the explanatory prose -- abm_mechanism, resilience_mechanism,
# why_included -- exists only in English, and it is the part that actually
# tells a policy user WHY an instrument behaves as it does. Translating it
# here keeps config/ as the single source of truth for the model while giving
# the interface the Chinese it needs; anything without an entry falls back to
# the English original rather than showing nothing.
# ---------------------------------------------------------------------------
MECHANISM_ZH = {
    "P1": "降低采纳价值函数中的购置成本 K。拨付滞后与流动性约束相互作用："
          "农户须先垫资、等待数月才能报销，其实际成本远高于名义净价。"
          "这正是把拨付滞后设为决策变量而非固定常数的原因。",
    "P2": "降低每亩服务价格，使采纳选择从自购转向服务模式。"
          "由于它只提高需求而不提高运力，单独实施会拉长排队；"
          "在相关性冲击下甚至会降低所有人的实际效果。"
          "P2 与 P3 的交互是本模型中最具政策含义的非线性关系。",
    "P3": "提高决定排队长度与预期等待时间的运力存量，"
          "并通过代理人保留效用与农户可及成本进入服务商的努力决策问题。",
    "P4": "提高数字素养，从而降低学习成本项、提高农户进入「有效使用」状态的概率。"
          "缺乏跟踪辅导时技能会衰减，因此一次性培训只会产生一个逐渐消退的短暂上升，"
          "模型会把这一衰减过程显示出来。",
    "P5": "把预警信息与预留作业运力结合起来，使预警在相关性冲击期间真正能够被执行。"
          "本工具专门用于打破 D2 与 D3 的失效模式："
          "所有人同时收到预警，却没有任何人能够被服务。",
    "P6": "改变的是农户面对的结果分布而非生产函数，因此直接作用于风险规避渠道："
          "它降低有效的 c1，并通过命题 1 改变均衡契约形态与服务商所供给的努力水平。"
          "赔付速度决定灾后复种与恢复时间。",
    "P7": "提高丘陵与山区经营主体的地形适配乘数 —— 若不提高，"
          "无论投入多少补贴，可达到的效果都存在上限。"
          "这是最直接针对空间公平目标的政策工具。",
    "T1": "争取提前量。把无预期的冲击转变为有预期的冲击，"
          "使抢排、抢灌、抢收或调整投入成为可能。"
          "其价值完全取决于农户是否有能力（劳动力、机械可及性）在可行动窗口内真正采取行动 —— "
          "这正是 T1 与 T3 互补而非互替的原因。",
    "T2": "直接作用于 D1 的损失机制，即在生殖生长期维持根区土壤水分。"
          "三者中资本密集度最高，因此最易受流动性约束制约，也最依赖后续维护。"
          "它需要既有水源：在没有水源的坡地上边际价值接近于零，"
          "这是收益空间异质性的主要来源。",
    "T3": "压缩完成关键作业所需的时间，这正是把预警转化为避免损失的关键环节。"
          "这是唯一在系统层面受运力配给的技术束：当大量农户同时受灾时，"
          "排队变长、服务在可行动窗口关闭之后才到达，"
          "即使名义上人人都是采纳者，实际效果仍然下降。"
          "这一拥堵反馈是 SD 层存在的首要理由。",
    "D1": "四川盆地典型的经常性气候特征，也是技术对策最明确的冲击："
          "智慧灌溉与水肥一体化直接作用于其损失机制（土壤水分亏缺），"
          "因此技术效率参数是可识别的。",
    "D2": "快速起始意味着真正的约束是预警提前量与作业窗口内的机械排队，"
          "而非设备是否自有。这正是服务运力 SD 模块与契约努力变量所要刻画的机制，"
          "因此 D2 是区分「服务券」与「购置补贴」两类政策的关键冲击。",
    "D3": "本研究的诊断性冲击。极端高温在抬高灌溉需求的同时，"
          "水电短缺与电网压力切断了水泵、传感器、基站与平台所依赖的电力。"
          "因此技术效率并非外生变量：它恰恰在最被需要的时刻崩溃。"
          "2022 年 8 月四川高温干旱伴工业限电事件是参照事件。"
          "任何对智慧技术套用固定百分比减损假设的模型，都会恰恰在此处高估韧性 —— "
          "因此这一冲击是对整个建模主张最尖锐的检验，"
          "也是「冗余与维护」优于「单纯设备补贴」的最有力论据。",
}

SIDE_EFFECT_ZH = {
    "biases towards larger operators with the cash to pre-finance":
        "偏向有现金垫资能力的大户",
    "can produce idle or duplicated equipment": "可能造成设备闲置或重复购置",
    "does nothing about maintenance, which is where failure actually bites":
        "完全不解决维护问题，而故障恰恰发生在维护环节",
    "congestion and delayed operations at peak": "高峰期拥堵与作业延迟",
    "remote farmers still served last despite holding vouchers":
        "偏远农户即使持券仍然最后才被服务",
    "basis risk and algorithmic exclusion of atypical plots":
        "基差风险，以及算法对非典型地块的排除",
    "claim processing congestion after correlated events":
        "相关性灾害之后理赔处理拥堵",
}

DV_LABELS = {
    "subsidy_rate": "补贴比例", "cap_per_farm": "每户补贴上限 (元)",
    "eligibility_min_area_mu": "最小申领面积 (亩)", "disbursement_lag_days": "拨付滞后 (天)",
    "voucher_value_per_mu": "服务券面值 (元/亩)", "max_mu_per_farm": "每户最高亩数",
    "targeting": "瞄准对象", "validity_days": "有效期 (天)",
    "new_centres_per_year": "年新建服务中心数", "units_per_centre": "每中心装备台数",
    "mountain_allocation_share": "山区分配比例",
    "slots_per_year": "年培训名额", "followup_support": "配套跟踪辅导",
    "skill_gain": "技能提升幅度", "decay_rate_annual": "技能年衰减率",
    "lead_time_gain_days": "预警提前量 (天)", "reserved_capacity_share": "应急预留运力比例",
    "false_alarm_penalty": "误报信任损失",
    "premium_subsidy_rate": "保费补贴比例", "payout_lag_days": "理赔滞后 (天)",
    "coverage_ratio": "保障水平", "basis_risk": "基差风险",
    "light_equipment_share": "轻简装备比例", "remote_service_points": "偏远服务点数",
    "transport_support": "运输补助强度",
}

DV_HELP = {
    "disbursement_lag_days": "农户须先垫资再报销。滞后越长，实际成本越高于名义净价，"
                             "这正是补贴对缺乏现金的小农失效的原因",
    "reserved_capacity_share": "从既有运力中划出应急预留。若不同时扩容，"
                               "这只是把运力从平时挪到灾时，总量并未增加",
    "voucher_value_per_mu": "只提高需求而不提高供给能力。单独发券会拉长排队",
    "new_centres_per_year": "直接扩充运力存量，是缓解排队的唯一直接手段。"
                            "⚠ 注意：本参数在当前标定下于约 1 个中心/年即已饱和 —— "
                            "初始运力按模拟样本耕地面积标定（约 1 个作业单元），"
                            "而本参数是全省级政策目标（25 中心 × 8 台 = 200 台/年），"
                            "两者不在同一量纲上。因此 1 与 80 的结果几乎相同，"
                            "该滑块目前只能当作「是否扩容」的开关使用。"
                            "若需比较不同扩容力度，须先解决这一量纲问题",
    "basis_risk": "遥感核损与实际损失之间的偏差，会造成理赔遗漏",
    "followup_support": "无跟踪辅导时技能会衰减，一次性培训只产生短暂效果",
    "slots_per_year": "提高有能力农户存量，降低学习成本。"
                      "⚠ 注意：本参数在约 1000 名额/年即已饱和（有能力农户占比触及 0.80 上限），"
                      "1000 与 20000 的结果完全相同。与 P3 同源的量纲问题："
                      "名额数是全省级目标，而培训存量按模拟样本标定",
    "premium_subsidy_rate": "保险改变的是农户面对的结果分布而非生产函数。"
                            "⚠ 注意：当前代码中保险只作用于现金流（保费、赔付、财政支出），"
                            "policies.yaml 所述的「降低有效 c1、进而改变契约形态与努力水平」"
                            "这一风险规避渠道尚未实现，因此本工具不会改变采纳率或减损率",
    "payout_lag_days": "理赔速度决定灾后复种与恢复。"
                       "⚠ 注意：模型以季（182.5 天）为时间步长，"
                       "7 至 180 天的任何取值都被取整为「1 个季度」，结果完全相同；"
                       "只有设为 0 才会当季赔付。该滑块目前实际不起作用",
}

OPT_LABELS = {"universal": "普惠", "smallholder": "小农户", "remote": "偏远地区",
              "mountain": "山区"}

EQUITY_ZH = {
    "poor_favours_large_operators": "不利公平：偏向大户",
    "favourable": "有利公平", "favourable_if_targeted": "定向瞄准时有利公平",
    "depends_on_allocation": "取决于分配方式", "mixed": "影响不一",
    "strongly_favourable": "显著有利公平",
}

PRESET_ZH = {
    "baseline": "基准情景（无新政策）", "subsidy": "仅设备购置补贴",
    "voucher": "仅按亩服务券", "voucher_plus_capacity": "服务券 + 运力扩容",
    "training_maintenance": "培训 + 维护保障", "warning_response": "预警 + 应急预留运力",
    "insurance": "精准保险（快速理赔）", "mountain_equity": "山地适配一揽子",
    "integrated": "综合政策包（固定预算）",
}

OBJECTIVES = [
    {"key": "mitigation_rate", "label": "减损率", "direction": "max", "fmt": "pct",
     "help": "技术实际消除的潜在损失占比，剔除了灾害发生频率的影响"},
    {"key": "effective_use_rate", "label": "有效使用率", "direction": "max", "fmt": "pct",
     "help": "达到真正会用状态的采纳者占比，而非仅仅完成购置"},
    {"key": "mean_wait_days", "label": "平均等待天数", "direction": "min", "fmt": "num",
     "help": "服务排队时长。一旦超过可行动窗口，采纳者与非采纳者的结果并无差别"},
    {"key": "fiscal_cumulative", "label": "累计财政支出", "direction": "min", "fmt": "money",
     "help": "公共财政累计投入"},
    {"key": "equity_gap", "label": "规模公平差距", "direction": "min", "fmt": "num",
     "help": "最小与最大农场四分位之间的减损差距"},
    {"key": "mountain_gap", "label": "山区平原差距", "direction": "min", "fmt": "num",
     "help": "空间公平指标"},
    {"key": "exit_rate", "label": "退出率", "direction": "min", "fmt": "pct",
     "help": "退出生产的农户占比"},
    {"key": "mean_income", "label": "平均收入", "direction": "max", "fmt": "money",
     "help": "农户平均收入"},
]

METRIC_LABELS = {
    "adopt_T1": "T1 采纳率", "adopt_T2": "T2 采纳率", "adopt_T3": "T3 采纳率",
    "service_T1": "T1 服务模式占比", "service_T2": "T2 服务模式占比",
    "service_T3": "T3 服务模式占比",
    "effort_T1": "T1 平均努力", "effort_T2": "T2 平均努力", "effort_T3": "T3 平均努力",
    "effective_use_rate": "有效使用率", "mitigation_rate": "减损率",
    "mean_loss_fraction": "平均损失率", "avoided_loss_fraction": "避免损失率",
    "mean_wait_days": "平均等待天数", "backlog_mu": "积压作业量 (亩)",
    "mean_income": "平均收入", "income_p10": "收入 P10", "gini_income": "收入基尼系数",
    "exit_rate": "退出率", "fiscal_spend": "当季财政支出",
    "fiscal_cumulative": "累计财政支出", "equity_gap": "规模公平差距",
    "mountain_gap": "山区平原差距", "capacity_units": "服务运力",
    "reliability": "技术可靠性", "trust": "信任度", "capable_share": "有能力农户占比",
    "recovery_seasons_mean": "平均恢复季数",
}
