# -*- coding: utf-8 -*-
"""
app.py
======
四川智慧农业政策仿真平台 -- web server.

Serves the Chinese-language scenario interface and runs the ABM-SD model
behind it.

Runs are executed on a background thread and polled, rather than answered
synchronously. Even with the persistent contract cache a cold parameter
combination takes tens of seconds, which exceeds every default proxy and
browser timeout, and a blocked request would give the user no way to tell a
slow run from a hung one.
"""

from __future__ import annotations

import io
import sys
import threading
import traceback
import uuid
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "webapp"))

from flask import Flask, Response, jsonify, request, send_from_directory

import contract_cache
import runner
from schema import build_schema

# Install the disk-backed contract cache before any model import touches the
# solver, so every run in this process shares the accumulated store.
CACHE = contract_cache.install()

# static_url_path="" serves the assets from the site root, so index.html can
# reference them as "app.js" rather than "/static/app.js" and the page works
# unchanged if it is ever mounted under a sub-path by a reverse proxy.
app = Flask(__name__, static_folder=str(Path(__file__).parent / "static"),
            static_url_path="")
app.config["JSON_AS_ASCII"] = False
app.json.ensure_ascii = False

# Ceiling on a single request's work. A 2000-farmer, 30-season, 10-replicate
# job is ~30x the default and would occupy the worker for many minutes; the
# limit turns that into an immediate, explainable refusal.
MAX_WORK_UNITS = 2000 * 30 * 3

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
@app.after_request
def _utf8(resp: Response) -> Response:
    # Windows consoles default to cp1252; being explicit here keeps the
    # Chinese interface from depending on the browser guessing correctly.
    if resp.mimetype in ("application/json", "text/html", "text/csv"):
        resp.headers["Content-Type"] = f"{resp.mimetype}; charset=utf-8"
    return resp


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/schema")
def api_schema():
    return jsonify(build_schema())


@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "cache": CACHE.stats(),
                    "jobs": len(JOBS)})


# ---------------------------------------------------------------------------
@app.post("/api/run")
def api_run():
    spec = request.get_json(force=True, silent=True) or {}

    pop = (spec.get("overrides") or {}).get("population") or {}
    n = int(pop.get("n_farmers", 500))
    seasons = int(pop.get("seasons", 12))
    reps = max(1, min(int(spec.get("replicates", 1)), 10))
    if n * seasons * reps > MAX_WORK_UNITS:
        return jsonify({"error": "运行规模过大：农户数 × 季数 × 重复次数 超出上限，"
                                 "请减少其中之一"}), 400

    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0.0,
                        "stage": "排队中", "created": datetime.now().isoformat(),
                        "result": None, "error": None}
    t = threading.Thread(target=_execute, args=(job_id, spec), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def api_job(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "任务不存在"}), 404
    return jsonify(job)


@app.get("/api/job/<job_id>/csv")
def api_job_csv(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if not job or job["status"] != "done":
        return jsonify({"error": "结果尚未就绪"}), 404
    return Response(_to_csv(job["result"]), mimetype="text/csv",
                    headers={"Content-Disposition":
                             f'attachment; filename="scenario_{job_id}.csv"'})


# ---------------------------------------------------------------------------
def _set(job_id: str, **kw) -> None:
    with JOBS_LOCK:
        if job_id in JOBS:
            JOBS[job_id].update(kw)


def _execute(job_id: str, spec: dict) -> None:
    workdir = None
    try:
        overrides = spec.get("overrides") or {}
        instruments = spec.get("instruments") or {}
        seed = int(spec.get("seed", 20260825))
        reps = max(1, min(int(spec.get("replicates", 1)), 10))
        want_baseline = bool(spec.get("compare_baseline", True))
        forced = {int(k): list(v) for k, v in
                  (spec.get("forced_shocks") or {}).items() if v}

        _set(job_id, status="running", stage="准备配置", progress=0.02)
        workdir = runner.temp_workdir()
        cfg_dir = runner.build_config_dir(overrides, workdir)

        # Total replicate count across both arms, for a single progress bar.
        total = reps * (2 if want_baseline else 1)
        done = {"n": 0}

        def progress(tag, i, of):
            done["n"] += 1
            _set(job_id, progress=0.05 + 0.9 * done["n"] / total,
                 stage=f"{tag} 第 {i}/{of} 次重复")

        _set(job_id, stage="运行政策情景", progress=0.05)
        policy = runner.run_scenario(cfg_dir, instruments, seed, reps, forced,
                                     progress, "政策情景")

        baseline = comparison = None
        if want_baseline:
            _set(job_id, stage="运行基准情景（同一随机种子，配对比较）")
            # Same seed, no instruments: paired against identical weather, so
            # the difference is the policy rather than the draw.
            baseline = runner.run_scenario(cfg_dir, {}, seed, reps, forced,
                                           progress, "基准情景")
            comparison = runner.compare(policy, baseline)

        CACHE.save()
        _set(job_id, status="done", progress=1.0, stage="完成",
             result={"policy": policy, "baseline": baseline,
                     "comparison": comparison,
                     "instruments": instruments, "seed": seed,
                     "replicates": reps,
                     "cache": CACHE.stats()})
    except Exception as exc:                     # noqa: BLE001
        traceback.print_exc()
        _set(job_id, status="error", stage="出错",
             error=f"{type(exc).__name__}: {exc}")
    finally:
        if workdir is not None:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


def _to_csv(result: dict) -> str:
    buf = io.StringIO()
    buf.write("﻿")           # BOM so Excel on Windows reads UTF-8
    pol = result["policy"]["series"]
    base = (result.get("baseline") or {}).get("series")
    keys = [k for k in runner.SERIES_KEYS if k in pol]

    head = ["season", "shocks"] + [f"policy_{k}" for k in keys]
    if base:
        head += [f"baseline_{k}" for k in keys]
    buf.write(",".join(head) + "\n")

    for i, s in enumerate(pol["season"]):
        shocks = (pol.get("shocks") or [""] * len(pol["season"]))[i] or ""
        row = [str(s), f'"{shocks}"']
        row += [_fmt(pol[k][i]) for k in keys]
        if base:
            row += [_fmt(base[k][i]) for k in keys]
        buf.write(",".join(row) + "\n")
    return buf.getvalue()


def _fmt(v) -> str:
    return "" if v is None else f"{v:.6g}"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="四川智慧农业政策仿真平台")
    ap.add_argument("--host", default="127.0.0.1",
                    help="绑定地址；对外提供服务时使用 0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"合约求解缓存: {CACHE.stats()}")
    print(f"启动: http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
