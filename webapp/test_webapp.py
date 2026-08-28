# -*- coding: utf-8 -*-
"""
Regression tests for the web layer.

These cover the parts of the app that can break silently: an override that is
accepted by the API but never reaches the model would leave the interface
looking correct while every scenario returned the same answer.

    python webapp/test_webapp.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

import yaml

import contract_cache
import runner
from schema import build_schema

contract_cache.install()

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(("  PASS  " if cond else "  FAIL  ") + name + (f"  [{detail}]" if detail else ""))


def read(cfg, name):
    with (cfg / name).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
def test_schema():
    print("\nschema")
    s = build_schema()
    check("7 instruments", len(s["instruments"]) == 7, str(len(s["instruments"])))
    check("3 technologies", len(s["technologies"]) == 3)
    check("3 shocks", len(s["shocks"]) == 3)
    check("presets present", len(s["presets"]) >= 9)
    check("all instruments have Chinese names",
          all(i["name_zh"] for i in s["instruments"]))
    check("all fields carry a range",
          all("min" in f and "max" in f
              for t in s["technologies"] for f in t["fields"]))
    # A default outside its own slider range would clamp on first render and
    # silently change the scenario the user thought they were running.
    bad = [f["key"] for t in s["technologies"] for f in t["fields"]
           if not (f["min"] <= f["default"] <= f["max"])]
    check("defaults lie inside their ranges", not bad, ",".join(bad))
    bad2 = [f"{i['id']}.{v['key']}" for i in s["instruments"] for v in i["variables"]
            if v["type"] == "number" and not (v["min"] <= v["default"] <= v["max"])]
    check("instrument defaults inside ranges", not bad2, ",".join(bad2))


def test_overrides():
    print("\noverride plumbing")
    wd = runner.temp_workdir()
    cfg = runner.build_config_dir({
        "population": {"n_farmers": 220, "seasons": 7},
        "counties": {"C6": {"n_farmers": 55}},
        "risk": {"farmer_alpha": 6.5},
        "behaviour": {"base_verifiability": 0.8},
        "technologies": {"T2": {"eta_drought": 0.61}},
        "shocks": {"D1": {"annual_probability_prior": 0.5}},
    }, wd)

    p = read(cfg, "model_params.yaml")
    t = read(cfg, "technologies.yaml")
    d = read(cfg, "disruptions.yaml")

    check("seasons override", p["meta"]["seasons"] == 7)
    counties = {c["id"]: c for c in p["population"]["counties"]}
    check("county override", counties["C6"]["n_farmers"] == 55)
    # Editing county sizes must re-derive the headline total, or the model
    # rescales the population it was handed.
    total = sum(c["n_farmers"] for c in p["population"]["counties"])
    check("headline population re-derived from counties",
          p["population"]["n_farmers"] == total, str(total))
    check("risk Beta override", p["risk_attitude"]["farmer"]["params"][0] == 6.5)
    check("verifiability override",
          p["contract"]["verifiability"]["base_verifiability"] == 0.8)
    T2 = {b["id"]: b for b in t["bundles"]}["T2"]
    check("technology override", T2["model"]["eta_drought"] == 0.61)

    D1 = {x["id"]: x for x in d["tier1"]}["D1"]
    check("shock probability override",
          D1["model"]["annual_probability_prior"] == 0.5)
    # The county field is what the generator samples; it must move with the
    # headline probability or the override does nothing.
    mean_c = sum(D1["model"]["probability_by_county"].values()) / 6
    check("county probabilities rescaled to match",
          abs(mean_c - 0.5) < 0.02, f"mean={mean_c:.3f}")
    check("policies.yaml copied through",
          (cfg / "policies.yaml").exists())


def test_overrides_change_results():
    print("\noverrides actually change model output")
    wd1, wd2 = runner.temp_workdir(), runner.temp_workdir()
    base_ov = {"population": {"n_farmers": 150, "seasons": 4}}
    hi_ov = {"population": {"n_farmers": 150, "seasons": 4},
             "shocks": {"D1": {"annual_probability_prior": 0.95},
                        "D2": {"annual_probability_prior": 0.95}}}
    a = runner.run_scenario(runner.build_config_dir(base_ov, wd1), {}, 4, 1, None)
    b = runner.run_scenario(runner.build_config_dir(hi_ov, wd2), {}, 4, 1, None)
    la = a["summary"]["mean_loss_fraction"]["mean"]
    lb = b["summary"]["mean_loss_fraction"]["mean"]
    check("raising shock probability raises losses", lb > la,
          f"{la:.4f} -> {lb:.4f}")


def test_paired_and_forced():
    print("\npaired comparison and forced shocks")
    wd = runner.temp_workdir()
    cfg = runner.build_config_dir({"population": {"n_farmers": 150, "seasons": 5}}, wd)

    pol = runner.run_scenario(cfg, {"P3": {"new_centres_per_year": 20}}, 3, 1, None)
    base = runner.run_scenario(cfg, {}, 3, 1, None)
    cmp_ = runner.compare(pol, base)
    check("comparison produced", "mean_wait_days" in cmp_)
    check("capacity expansion reduces wait",
          cmp_["mean_wait_days"]["diff"] < 0,
          f"{cmp_['mean_wait_days']['diff']:.2f} days")

    # Same seed, same config, no policy -> identical weather. If this drifts,
    # the pairing that every reported policy effect depends on is broken.
    again = runner.run_scenario(cfg, {}, 3, 1, None)
    check("baseline is reproducible on the same seed",
          base["series"]["shocks"] == again["series"]["shocks"] and
          abs(base["summary"]["mean_income"]["mean"]
              - again["summary"]["mean_income"]["mean"]) < 1e-9)

    forced = runner.run_scenario(cfg, {}, 3, 1, {2: ["D3"], 4: ["D3"]})
    sh = forced["series"]["shocks"]
    check("forced shock lands in season 2", "D3" in (sh[1] or ""), sh[1])
    check("forced shock lands in season 4", "D3" in (sh[3] or ""), sh[3])


def test_replicates():
    print("\nreplicates")
    wd = runner.temp_workdir()
    cfg = runner.build_config_dir({"population": {"n_farmers": 120, "seasons": 4}}, wd)
    r = runner.run_scenario(cfg, {}, 11, 3, None)
    check("three replicates recorded", r["replicates"] == 3)
    check("standard deviation reported",
          r["summary"]["mean_income"]["sd"] is not None
          and r["summary"]["mean_income"]["sd"] > 0)
    check("series averaged over replicates",
          len(r["series"]["season"]) == 4)


def test_api():
    print("\nHTTP API")
    import app as webapp
    c = webapp.app.test_client()

    check("index served", c.get("/").status_code == 200)
    check("static js served", c.get("/app.js").status_code == 200)
    s = c.get("/api/schema")
    check("schema endpoint", s.status_code == 200 and len(s.get_json()["instruments"]) == 7)
    check("health endpoint", c.get("/api/health").get_json()["ok"] is True)
    check("unknown job -> 404", c.get("/api/job/nope").status_code == 404)
    over = c.post("/api/run", json={"overrides": {"population":
                  {"n_farmers": 2000, "seasons": 30}}, "replicates": 5})
    check("oversized run refused with 400", over.status_code == 400)

    import time
    r = c.post("/api/run", json={
        "overrides": {"population": {"n_farmers": 120, "seasons": 4}},
        "instruments": {"P2": {"voucher_value_per_mu": 5.0}},
        "seed": 2, "replicates": 1, "compare_baseline": True})
    jid = r.get_json()["job_id"]
    for _ in range(300):
        j = c.get(f"/api/job/{jid}").get_json()
        if j["status"] in ("done", "error"):
            break
        time.sleep(1)
    check("job completes", j["status"] == "done", j.get("error") or "")
    if j["status"] == "done":
        check("comparison returned", bool(j["result"]["comparison"]))
        csv = c.get(f"/api/job/{jid}/csv")
        check("csv export", csv.status_code == 200 and b"season" in csv.data)
        check("csv has BOM for Excel", csv.data.startswith("﻿".encode()))


if __name__ == "__main__":
    test_schema()
    test_overrides()
    test_overrides_change_results()
    test_paired_and_forced()
    test_replicates()
    test_api()
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for f in FAIL:
            print("  FAILED:", f)
    sys.exit(1 if FAIL else 0)
