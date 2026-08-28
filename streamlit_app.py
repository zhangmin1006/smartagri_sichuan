# -*- coding: utf-8 -*-
"""Streamlit entry point for the Sichuan smart-agriculture policy model."""

from __future__ import annotations

import io
import shutil
import sys
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

import contract_cache  # noqa: E402
import runner  # noqa: E402
from schema import build_schema  # noqa: E402


st.set_page_config(
    page_title="四川智慧农业政策仿真平台",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: #f7faf7; }
      .hero {
        padding: 1.25rem 1.5rem; border-radius: 18px; margin-bottom: 1rem;
        color: white; background: linear-gradient(120deg, #174d36, #2b7a52);
        box-shadow: 0 10px 28px rgba(23,77,54,.16);
      }
      .hero h1 { margin: 0; font-size: 2rem; }
      .hero p { margin: .45rem 0 0; color: #e8f4ec; }
      .note {
        padding: .85rem 1rem; border-left: 4px solid #d7a72d;
        background: #fff8e6; border-radius: 8px; color: #51430f;
      }
      div[data-testid="stMetric"] {
        background: white; border: 1px solid #dbe8de; border-radius: 12px;
        padding: .8rem 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_schema() -> dict:
    return build_schema()


@st.cache_resource
def install_contract_cache():
    return contract_cache.install()


SCHEMA = load_schema()
CACHE = install_contract_cache()


def _field_key(prefix: str, field: dict) -> str:
    return f"{prefix}_{field['key'].replace('.', '_')}"


def _set_default(key: str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def initialise_state() -> None:
    for field in SCHEMA["population"]["fields"]:
        _set_default(_field_key("population", field), field["default"])
    for field in SCHEMA["population"]["risk"]:
        _set_default(_field_key("risk", field), field["default"])
    for field in SCHEMA["population"]["behaviour"]:
        _set_default(_field_key("behaviour", field), field["default"])
    for county in SCHEMA["population"]["counties"]:
        for field in county["fields"]:
            _set_default(_field_key("county", field), field["default"])
    for tech in SCHEMA["technologies"]:
        for field in tech["fields"]:
            _set_default(_field_key("technology", field), field["default"])
    for shock in SCHEMA["shocks"]:
        for field in shock["fields"]:
            _set_default(_field_key("shock", field), field["default"])
    for inst in SCHEMA["instruments"]:
        _set_default(f"enabled_{inst['id']}", False)
        for field in inst["variables"]:
            _set_default(_field_key(f"policy_{inst['id']}", field), field["default"])

    _set_default("preset", "")
    _set_default("forced_pairs", [])
    _set_default("seed", int(SCHEMA["default_seed"]))
    _set_default("replicates", 1)
    _set_default("compare_baseline", True)


def reset_controls() -> None:
    for field in SCHEMA["population"]["fields"]:
        st.session_state[_field_key("population", field)] = field["default"]
    for group in ("risk", "behaviour"):
        for field in SCHEMA["population"][group]:
            st.session_state[_field_key(group, field)] = field["default"]
    for county in SCHEMA["population"]["counties"]:
        for field in county["fields"]:
            st.session_state[_field_key("county", field)] = field["default"]
    for group_name, items in (("technology", SCHEMA["technologies"]),
                              ("shock", SCHEMA["shocks"])):
        for item in items:
            for field in item["fields"]:
                st.session_state[_field_key(group_name, field)] = field["default"]
    for inst in SCHEMA["instruments"]:
        st.session_state[f"enabled_{inst['id']}"] = False
        for field in inst["variables"]:
            st.session_state[_field_key(f"policy_{inst['id']}", field)] = field["default"]
    st.session_state["preset"] = ""
    st.session_state["forced_pairs"] = []
    st.session_state["seed"] = int(SCHEMA["default_seed"])
    st.session_state["replicates"] = 1
    st.session_state["compare_baseline"] = True
    st.session_state.pop("last_result", None)


def apply_preset(key: str) -> None:
    selected = next((p for p in SCHEMA["presets"] if p["key"] == key), None)
    values = selected["instruments"] if selected else {}
    for inst in SCHEMA["instruments"]:
        inst_values = values.get(inst["id"], {})
        st.session_state[f"enabled_{inst['id']}"] = inst["id"] in values
        for field in inst["variables"]:
            state_key = _field_key(f"policy_{inst['id']}", field)
            st.session_state[state_key] = inst_values.get(field["key"], field["default"])


def render_field(field: dict, key: str, disabled: bool = False):
    help_text = field.get("help") or None
    if field["type"] == "bool":
        return st.checkbox(field["label"], key=key, help=help_text, disabled=disabled)
    if field["type"] == "select":
        options = [item["value"] for item in field["options"]]
        labels = {item["value"]: item["label"] for item in field["options"]}
        return st.selectbox(
            field["label"], options, key=key, format_func=lambda x: labels.get(x, str(x)),
            help=help_text, disabled=disabled,
        )

    values = (field["default"], field["min"], field["max"], field["step"])
    integer = all(isinstance(v, int) and not isinstance(v, bool) for v in values)
    cast = int if integer else float
    return st.number_input(
        field["label"], min_value=cast(field["min"]), max_value=cast(field["max"]),
        step=cast(field["step"]), key=key, help=help_text, disabled=disabled,
    )


def collect_spec() -> dict:
    population = {
        f["key"]: st.session_state[_field_key("population", f)]
        for f in SCHEMA["population"]["fields"]
    }
    risk = {
        f["key"]: st.session_state[_field_key("risk", f)]
        for f in SCHEMA["population"]["risk"]
    }
    behaviour = {
        f["key"]: st.session_state[_field_key("behaviour", f)]
        for f in SCHEMA["population"]["behaviour"]
    }
    counties = {}
    for county in SCHEMA["population"]["counties"]:
        counties[county["id"]] = {
            f["key"].split(".", 1)[1]: st.session_state[_field_key("county", f)]
            for f in county["fields"]
        }

    technologies = {}
    for tech in SCHEMA["technologies"]:
        technologies[tech["id"]] = {
            f["key"].split(".", 1)[1]: st.session_state[_field_key("technology", f)]
            for f in tech["fields"]
        }

    shocks = {}
    for shock in SCHEMA["shocks"]:
        shocks[shock["id"]] = {
            f["key"].split(".", 1)[1]: st.session_state[_field_key("shock", f)]
            for f in shock["fields"]
        }

    instruments = {}
    for inst in SCHEMA["instruments"]:
        if st.session_state[f"enabled_{inst['id']}"]:
            instruments[inst["id"]] = {
                f["key"]: st.session_state[_field_key(f"policy_{inst['id']}", f)]
                for f in inst["variables"]
            }

    forced = {}
    for choice in st.session_state["forced_pairs"]:
        season_text, shock_id = choice.split(" · ", 1)
        season = int(season_text.removeprefix("第 ").removesuffix(" 季"))
        forced.setdefault(season, []).append(shock_id.split(" ", 1)[0])

    return {
        "overrides": {
            "population": population, "counties": counties, "risk": risk,
            "behaviour": behaviour, "technologies": technologies, "shocks": shocks,
        },
        "instruments": instruments,
        "forced_shocks": forced,
        "seed": int(st.session_state["seed"]),
        "replicates": int(st.session_state["replicates"]),
        "compare_baseline": bool(st.session_state["compare_baseline"]),
    }


def run_model(spec: dict) -> dict:
    progress_bar = st.progress(0.02, text="准备配置")
    workdir = runner.temp_workdir()
    try:
        cfg_dir = runner.build_config_dir(spec["overrides"], workdir)
        reps = spec["replicates"]
        arms = 2 if spec["compare_baseline"] else 1
        completed = 0

        def progress(tag, i, total):
            nonlocal completed
            completed += 1
            value = min(0.95, 0.05 + 0.9 * completed / (reps * arms))
            progress_bar.progress(value, text=f"{tag}：第 {i}/{total} 次重复")

        policy = runner.run_scenario(
            cfg_dir, spec["instruments"], spec["seed"], reps,
            spec["forced_shocks"], progress, "政策情景",
        )
        baseline = comparison = None
        if spec["compare_baseline"]:
            baseline = runner.run_scenario(
                cfg_dir, {}, spec["seed"], reps, spec["forced_shocks"],
                progress, "基准情景",
            )
            comparison = runner.compare(policy, baseline)
        CACHE.save()
        progress_bar.progress(1.0, text="仿真完成")
        return {
            "policy": policy, "baseline": baseline, "comparison": comparison,
            "instruments": spec["instruments"], "seed": spec["seed"],
            "replicates": reps, "cache": CACHE.stats(),
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def format_value(value, style: str | None = None, signed: bool = False) -> str:
    if value is None:
        return "—"
    sign = "+" if signed and value > 0 else ""
    if style == "pct":
        return f"{sign}{value:.1%}"
    if style == "money":
        return f"{sign}¥{value:,.0f}"
    return f"{sign}{value:,.3g}"


def result_csv(result: dict) -> bytes:
    policy = result["policy"]["series"]
    data = {"season": policy["season"]}
    shocks = policy.get("shocks")
    if shocks is not None:
        data["shocks"] = shocks
    for key in runner.SERIES_KEYS:
        if key in policy:
            data[f"policy_{key}"] = policy[key]
    baseline = (result.get("baseline") or {}).get("series")
    if baseline:
        for key in runner.SERIES_KEYS:
            if key in baseline:
                data[f"baseline_{key}"] = baseline[key]
    buf = io.StringIO()
    pd.DataFrame(data).to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8-sig")


def render_results(result: dict) -> None:
    st.subheader("仿真结果")
    enabled = ", ".join(result["instruments"]) or "无新增政策工具"
    st.caption(
        f"随机种子 {result['seed']} · 重复 {result['replicates']} 次 · "
        f"启用工具 {enabled} · 契约缓存 {result['cache']['entries']} 条"
    )

    objective_map = {o["key"]: o for o in SCHEMA["objectives"]}
    comparison = result.get("comparison")
    if comparison:
        objectives = [o for o in SCHEMA["objectives"] if o["key"] in comparison]
        for start in range(0, len(objectives), 4):
            cols = st.columns(4)
            for col, objective in zip(cols, objectives[start:start + 4]):
                item = comparison[objective["key"]]
                with col:
                    st.metric(
                        objective["label"],
                        format_value(item["policy"], objective["fmt"]),
                        format_value(item["diff"], objective["fmt"], signed=True),
                        delta_color="normal" if objective["direction"] == "max" else "inverse",
                    )
                    if item["distinguishable"] is False:
                        st.caption("与重复间波动无法区分")
    else:
        st.info("本次未运行基准情景，因此只显示政策情景结果。")

    policy_series = result["policy"]["series"]
    available = [k for k in runner.SERIES_KEYS if k in policy_series]
    default_charts = [k for k in (
        "mitigation_rate", "effective_use_rate", "mean_wait_days", "mean_income"
    ) if k in available]
    selected = st.multiselect(
        "逐季图表指标", available, default=default_charts,
        format_func=lambda k: SCHEMA["metrics"].get(k, k), key="result_chart_metrics",
    )
    baseline_series = (result.get("baseline") or {}).get("series")
    for start in range(0, len(selected), 2):
        cols = st.columns(2)
        for col, metric in zip(cols, selected[start:start + 2]):
            chart_data = pd.DataFrame({
                "季": policy_series["season"], "政策情景": policy_series[metric],
            })
            if baseline_series and metric in baseline_series:
                chart_data["基准情景"] = baseline_series[metric]
            with col:
                st.markdown(f"**{SCHEMA['metrics'].get(metric, metric)}**")
                st.line_chart(chart_data, x="季", y=[c for c in chart_data if c != "季"])

    rows = []
    for key, policy_item in result["policy"]["summary"].items():
        if policy_item["mean"] is None:
            continue
        objective = objective_map.get(key, {})
        style = objective.get("fmt")
        row = {
            "指标": SCHEMA["metrics"].get(key, key),
            "政策情景": format_value(policy_item["mean"], style),
            "重复间标准差": format_value(policy_item["sd"], style),
        }
        if comparison and key in comparison:
            item = comparison[key]
            row.update({
                "基准情景": format_value(item["baseline"], style),
                "差异": format_value(item["diff"], style, signed=True),
                "相对变化": "—" if item["rel"] is None else f"{item['rel']:+.1%}",
            })
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.download_button(
        "下载逐季结果 CSV", result_csv(result), "sichuan_smartagri_results.csv",
        "text/csv", width="stretch",
    )


initialise_state()

st.markdown(
    """
    <div class="hero">
      <h1>四川智慧农业政策仿真平台</h1>
      <p>智慧农业技术采用、服务努力、灾害韧性与政策组合的 ABM–SD 情景实验</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("运行设置")
    st.number_input("随机种子", min_value=1, max_value=99_999_999, step=1, key="seed")
    st.number_input("重复次数", min_value=1, max_value=10, step=1, key="replicates",
                    help="建议至少重复 3 次，以判断政策效应是否超过随机波动。")
    st.checkbox("同时运行配对基准情景", key="compare_baseline",
                help="政策与基准使用同一随机种子和天气路径。")
    st.caption(f"模型版本 {SCHEMA['model_version']}")
    st.caption(f"契约缓存：{CACHE.stats()['entries']} 条")
    if st.button("恢复全部默认值", width="stretch"):
        reset_controls()
        st.rerun()

tab_policy, tab_shock, tab_tech, tab_farm = st.tabs([
    "政策工具", "冲击情景", "技术参数", "农户与契约",
])

with tab_policy:
    left, right = st.columns([3, 1])
    preset_labels = {p["key"]: p["label"] for p in SCHEMA["presets"]}
    with left:
        st.selectbox(
            "政策预设", [""] + list(preset_labels), key="preset",
            format_func=lambda x: "选择预设情景…" if not x else preset_labels[x],
        )
    with right:
        st.write("")
        if st.button("应用预设", width="stretch", disabled=not st.session_state["preset"]):
            apply_preset(st.session_state["preset"])
            st.rerun()

    for inst in SCHEMA["instruments"]:
        enabled_key = f"enabled_{inst['id']}"
        title = f"{inst['id']} · {inst['name_zh']}"
        with st.expander(title, expanded=st.session_state[enabled_key]):
            st.checkbox("启用该政策工具", key=enabled_key)
            st.caption(inst["equity_flag"])
            if inst.get("mechanism"):
                st.markdown(f"**作用机制：** {inst['mechanism']}")
            for side_effect in inst.get("side_effects", []):
                st.warning(side_effect)
            cols = st.columns(2)
            for index, field in enumerate(inst["variables"]):
                with cols[index % 2]:
                    render_field(
                        field, _field_key(f"policy_{inst['id']}", field),
                        disabled=not st.session_state[enabled_key],
                    )

with tab_shock:
    for shock in SCHEMA["shocks"]:
        with st.expander(f"{shock['id']} · {shock['name_zh']} — {shock['name_en']}",
                         expanded=shock["id"] == "D3"):
            if shock.get("why"):
                st.markdown(f"**纳入原因：** {shock['why']}")
            cols = st.columns(2)
            for index, field in enumerate(shock["fields"]):
                with cols[index % 2]:
                    render_field(field, _field_key("shock", field))

    seasons = int(st.session_state[_field_key("population", SCHEMA["population"]["fields"][1])])
    shock_names = {s["id"]: s["name_zh"] for s in SCHEMA["shocks"]}
    forced_options = [
        f"第 {season} 季 · {shock_id} {shock_names[shock_id]}"
        for season in range(1, seasons + 1) for shock_id in shock_names
    ]
    st.multiselect(
        "强制冲击排程（可选）", forced_options, key="forced_pairs",
        help="用于把指定灾害固定安排在指定季节；留空则由概率过程生成。",
    )

with tab_tech:
    for tech in SCHEMA["technologies"]:
        with st.expander(f"{tech['id']} · {tech['name_zh']} — {tech['name_en']}"):
            st.caption(tech["channel"])
            if tech.get("mechanism"):
                st.markdown(f"**韧性机制：** {tech['mechanism']}")
            cols = st.columns(2)
            for index, field in enumerate(tech["fields"]):
                with cols[index % 2]:
                    render_field(field, _field_key("technology", field))

with tab_farm:
    st.subheader("人口与规模")
    cols = st.columns(3)
    for index, field in enumerate(SCHEMA["population"]["fields"]):
        with cols[index % 3]:
            render_field(field, _field_key("population", field))

    st.subheader("风险态度分布")
    cols = st.columns(2)
    for index, field in enumerate(SCHEMA["population"]["risk"]):
        with cols[index % 2]:
            render_field(field, _field_key("risk", field))

    st.subheader("契约与行为参数")
    cols = st.columns(2)
    for index, field in enumerate(SCHEMA["population"]["behaviour"]):
        with cols[index % 2]:
            render_field(field, _field_key("behaviour", field))

    st.subheader("六个代表性县域")
    st.caption("修改县域农户数量后，总农户数量将以六县之和为准。")
    for county in SCHEMA["population"]["counties"]:
        with st.expander(f"{county['id']} · {county['label']} — {county['terrain']}"):
            cols = st.columns(2)
            for index, field in enumerate(county["fields"]):
                with cols[index % 2]:
                    render_field(field, _field_key("county", field))

st.markdown(
    """
    <div class="note">
      <strong>结果性质说明：</strong>
      契约核心已对照 Guo, H. (2015) <em>Optimal ordering decision and incentives for yield improvement under random demand</em>.
      PhD thesis. McMaster University. 结果验证通过；其余为<strong>条件性情景推演，而非预测</strong>。
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")

if st.button("运行仿真", type="primary", width="stretch"):
    spec = collect_spec()
    work_units = (int(spec["overrides"]["population"]["n_farmers"])
                  * int(spec["overrides"]["population"]["seasons"])
                  * spec["replicates"])
    if work_units > 2000 * 30 * 3:
        st.error("运行规模过大：农户数 × 季数 × 重复次数超过上限，请减少其中之一。")
    else:
        try:
            with st.spinner("模型正在运行，请勿关闭页面…"):
                st.session_state["last_result"] = run_model(spec)
        except Exception as exc:  # noqa: BLE001
            st.exception(exc)

if "last_result" in st.session_state:
    render_results(st.session_state["last_result"])
