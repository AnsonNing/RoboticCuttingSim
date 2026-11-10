#!/usr/bin/env python3
"""Run a perf acceptance locally, without the external agentic optimizer (P1-6 R0, decision D-11).

Plays the caller's part of the interface: runs ``run_perf_acceptance.py`` for every selected plan entry and G9
ABBA block of the config (one process per side and run, the caller's monotonic clock on the ``RCS_PERF_MARK``
lines), optional CPU suites, writes ``manifest.json`` (``rcs_perf_accept_manifest_v1``) and runs
``judge_perf_acceptance.py`` from ``--judge-checkout`` (default: the baseline checkout) into ``OUT/judge/``.

Hold the GPU lock for the whole call (one RCS GPU job per GPU):

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env TMPDIR=/mnt/data/ningan/tmp \\
      .deps/isaacsim-venv/bin/python scripts/rl_ready/run_perf_acceptance_local.py \\
        --config configs/rl_benchmark/acceptance_v1.json --baseline-checkout B --candidate-checkout C \\
        --proposal-json P.json --gpu-index 1 --out artifacts/rl_ready/p1_6_r0/<name>

``--only`` selects plan ids and ``perf_n<N>`` blocks (default: everything). The acceptance of record is the
external caller's; this runner exists so RCS can reproduce one (and for the R0 smoke).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robotic_cutting_sim.rl import perf_acceptance as PA  # noqa: E402


def git_head(checkout):
    return subprocess.run(["git", "-C", str(checkout), "rev-parse", "HEAD"], capture_output=True, text=True,
                          check=True).stdout.strip()


def run_harness(cmd, run_dir, env, cwd, timeout_s):
    run_dir.mkdir(parents=True)
    marks = []
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    timer = threading.Timer(timeout_s, proc.kill)
    timer.start()
    with (run_dir / "stdout.log").open("x") as log, (run_dir / "outer_clock.jsonl").open("x") as clock:
        for line in proc.stdout:
            t = time.monotonic()
            log.write(f"[{t - t0:10.3f}] {line}")
            mark = PA.parse_mark(line)
            if mark is not None:
                row = {"t_mono": t, "t_wall": time.time(), "mark": mark}
                marks.append(row)
                clock.write(json.dumps(row) + "\n")
    rc = proc.wait()
    timer.cancel()
    return rc, marks, time.monotonic() - t0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--baseline-checkout", required=True)
    p.add_argument("--candidate-checkout", required=True)
    p.add_argument("--proposal-json", required=True, help="the candidate's open_vector_env options")
    p.add_argument("--judge-checkout", default=None)
    p.add_argument("--gpu-index", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--sim-python", default=str(ROOT / ".deps" / "isaacsim-venv" / "bin" / "python"))
    p.add_argument("--judge-python", default=str(ROOT / ".venv" / "bin" / "python"))
    p.add_argument("--wp17", default=str(ROOT / "artifacts" / "wp17"))
    p.add_argument("--only", default="", help="comma list of plan ids and perf_n<N> (default: all)")
    p.add_argument("--cpu-tests-args", default=None,
                   help="override cpu_tests args (space separated; '' skips the CPU suites)")
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists (never overwritten)")
    cfg = PA.load_config(a.config)
    only = {x for x in a.only.split(",") if x}
    sides = {"baseline": Path(a.baseline_checkout).resolve(), "candidate": Path(a.candidate_checkout).resolve()}
    commits = {s: git_head(c) for s, c in sides.items()}
    out.mkdir(parents=True)
    proposal = json.loads(Path(a.proposal_json).read_text())
    vdir = out / "variants"
    vdir.mkdir()
    vfiles = {}
    for key, value in (("reference", cfg["reference_variant"]), ("proposal", proposal), ("default", {})):
        vfiles[key] = vdir / f"{key}.json"
        vfiles[key].write_text(json.dumps(value) + "\n")
    runs = []

    def one(rid, side, gate, layer, mode, n, seeds, warm, steps, vkey, extra):
        rdir = out / "runs" / rid
        cmd = [a.sim_python, str(sides[side] / "scripts" / "rl_ready" / "run_perf_acceptance.py"), "--variant-json",
               str(vfiles[vkey]), "--mode", mode, "--num-envs", str(n), "--seeds", ",".join(map(str, seeds)),
               "--warmup-steps", str(warm), "--steps", str(steps), "--device", a.device, "--gpu-index",
               str(a.gpu_index), "--wp17", a.wp17, "--out", str(rdir / "harness")]
        env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"))}
        env.update(PYTHONPATH=str(sides[side] / "src"), PYTHONDONTWRITEBYTECODE="1",
                   CUDA_VISIBLE_DEVICES=str(a.gpu_index))
        print(f"[{time.strftime('%H:%M:%S')}] {rid}", flush=True)
        rc, marks, wall = run_harness(cmd, rdir, env, sides[side], float(cfg.get("run_timeout_s", 1800)))
        rp = rdir / "harness" / "result.json"
        res = json.loads(rp.read_text()) if rp.is_file() else None
        outer = PA.outer_rate(marks)
        inner = next((x.get("env_steps_per_s") for x in (res or {}).get("rows") or [] if x.get("num_envs") == n),
                     None)
        clock = None
        if outer and inner:
            ratio = inner / outer["env_steps_per_s"]
            clock = dict(inner=inner, outer=outer["env_steps_per_s"], ratio=ratio,
                         consistent=abs(ratio - 1) <= cfg["thresholds"]["clock_tolerance"])
        co = ((res or {}).get("rows") or [{}])[-1].get("co_tenant") or {}
        s = dict(id=rid, side=side, commit=commits[side], gate=gate, layer=layer, mode=mode, num_envs=n,
                 seeds=list(seeds), variant=vkey, rc=rc, wall_s=round(wall, 3), result=str(rp) if res else None,
                 infra=None if res else f"no result.json (exit {rc})", outer=outer, clock=clock, marks=len(marks),
                 gpu=dict(label=(res or {}).get("gpu_sharing") or "UNKNOWN_GPU",
                          util_p50=co.get("total_util_pct_mean"), source="harness telemetry (local runner)"),
                 environment_problems=[], gpu_guard=None, waited_s=0.0, cmd=cmd, **extra)
        (rdir / "run.json").write_text(json.dumps(s, indent=1) + "\n")
        runs.append(s)
        print(f"    rc {rc}, outer {outer and round(outer['env_steps_per_s'], 4)}, clock {clock and round(clock['ratio'], 4)},"
              f" wall {wall:.1f} s", flush=True)

    for e in cfg["plan"]:
        if only and e["id"] not in only:
            continue
        layer = e.get("layer") or ("PERFORMANCE" if e["gate"] == "G9" else "PHYSICS")
        for r in range(int(e.get("repeats", 1))):
            for side in e.get("sides") or ("baseline", "candidate"):
                vkey = "default" if e["variant"] == "default" else ("reference" if side == "baseline" else "proposal")
                one(f"{e['id']}_{side}_r{r}", side, e["gate"], layer, e["mode"], int(e["num_envs"]), e["seeds"],
                    int(e["warmup_steps"]), int(e["steps"]), vkey, dict(plan_id=e["id"], repeat=r))
    perf = cfg["performance"]
    for n in perf["num_envs"]:
        if only and f"perf_n{n}" not in only:
            continue
        steps = int((perf.get("steps_by_num_envs") or {}).get(str(n), perf["steps"]))
        for i, ch in enumerate(perf["order"]):
            side = "baseline" if ch == "B" else "candidate"
            one(f"perf_n{n}_{i:02d}_{ch}", side, "G9", "PERFORMANCE", "throughput", int(n), perf["seeds"],
                int(perf["warmup_steps"]), steps, "reference" if side == "baseline" else "proposal",
                dict(plan_id=f"perf_n{n}", pair=i // 2, order=i))
    cpu = {"baseline": {}, "candidate": {}}
    args = cfg["cpu_tests"]["args"] if a.cpu_tests_args is None else a.cpu_tests_args.split()
    if args:
        for side, checkout in sides.items():
            o = out / "cpu" / side
            o.mkdir(parents=True)
            env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
            env.update(PYTHONPATH=str(checkout / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
            with (o / "pytest.log").open("x") as log:
                rc = subprocess.run([a.judge_python, "-m", "pytest", "-p", "no:cacheprovider", "-q", *args,
                                     f"--junitxml={o / 'junit.xml'}"], cwd=str(checkout), env=env, stdout=log,
                                    stderr=subprocess.STDOUT).returncode
            cpu[side]["cpu_tests"] = dict(PA.junit_summary(o / "junit.xml"), rc=rc, junit=str(o / "junit.xml"),
                                          args=args)
    cfg_bytes = Path(a.config).read_bytes()
    manifest = dict(schema=PA.SCHEMA_MANIFEST, time=time.strftime("%Y-%m-%dT%H:%M:%S%z"), config=cfg["version"],
                    config_path=str(Path(a.config).resolve()), config_sha256=PA.sha256_bytes(cfg_bytes),
                    baseline=dict(commit=commits["baseline"], variant=cfg["reference_variant"],
                                  checkout=str(sides["baseline"])),
                    candidate=dict(commit=commits["candidate"], variant=proposal, checkout=str(sides["candidate"])),
                    evaluator=dict(commit=commits["baseline"], files_sha256=None, files=None,
                                   note="local runner: the evaluator file set is not hashed here"),
                    runs=runs, cpu_tests=cpu, wp17=a.wp17, seeds=cfg["seeds"], blocked=None,
                    clock_tolerance=cfg["thresholds"]["clock_tolerance"], runner="run_perf_acceptance_local.py",
                    selection=sorted(only) or "all")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    judge_checkout = Path(a.judge_checkout or sides["baseline"]).resolve()
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "VIRTUAL_ENV"))}
    env.update(PYTHONPATH=str(judge_checkout / "src"), PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="")
    t0 = time.monotonic()
    with (out / "judge.log").open("x") as log:
        rc = subprocess.run([a.judge_python, str(judge_checkout / "scripts" / "rl_ready" / "judge_perf_acceptance.py"),
                             "--config", str(Path(a.config).resolve()), "--manifest", str(out / "manifest.json"),
                             "--out", str(out / "judge")], cwd=str(judge_checkout), env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    verdict = out / "judge" / "verdict.json"
    v = json.loads(verdict.read_text()) if verdict.is_file() else None
    print(json.dumps(dict(judge_rc=rc, judge_wall_s=round(time.monotonic() - t0, 1),
                          layers=v and v["layers"], failed_gates=v and v["failed_gates"],
                          ratios=v and {k: x["median"] for k, x in v["performance"]["ratios"].items()})))
    return 0 if v else 1


if __name__ == "__main__":
    sys.exit(main())
