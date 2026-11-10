#!/usr/bin/env python3
"""P1-4 supervisor (T4.1): run one pre-declared run of ``configs/rl_benchmark/long_rollout_v1.json`` in child
processes (``scripts/rl_ready/run_long_rollout.py``), one attempt per process, with the watchdog, per-PID VRAM /
RSS sampling, classification and restart policy of ``rl.supervision.Supervisor``.

GPU use (the caller holds the lock; this script never takes or touches locks)::

    cd /home/ningan/RoboticCuttingSim-rl-p14
    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock \\
      env CUDA_VISIBLE_DEVICES=1 TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src \\
      .deps/isaacsim-venv/bin/python scripts/rl_ready/supervise_rollout.py --run LR-A --gpu-index 1 --name lr_a

Before launch it checks free VRAM on the physical GPU (own bound + margin, D-07); if short it waits up to
``--free-vram-wait-s`` and then ends BLOCKED_INSUFFICIENT_FREE_VRAM without launching. Raw output: a new
``artifacts/rl_ready/p1_4/<name>_NNN/`` (never overwritten). ``--backend fake`` (CPU, tests) needs no GPU.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from robotic_cutting_sim.rl import supervision as S  # noqa: E402

CHILD = ROOT / "scripts" / "rl_ready" / "run_long_rollout.py"


def git(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def new_run_dir(out_root, name):
    out_root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = out_root / f"{name}_{index:03d}"
        if not path.exists():
            return path
    raise RuntimeError("no free run directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "rl_benchmark" / "long_rollout_v1.json"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--name")
    parser.add_argument("--out-root", default=str(ROOT / "artifacts" / "rl_ready" / "p1_4"))
    parser.add_argument("--backend", choices=("trim", "fake"), default="trim")
    parser.add_argument("--gpu-index", type=int, help="physical nvidia-smi index (VRAM telemetry, free check)")
    parser.add_argument("--no-gpu-telemetry", action="store_true")
    parser.add_argument("--child-python", default=sys.executable)
    parser.add_argument("--fault-inject", help="RCS_FAULT_INJECT for attempt 1 (fault tests / LR-D only)")
    parser.add_argument("--max-global-steps", type=int, default=0, help="smoke only (REDUCED run)")
    parser.add_argument("--free-vram-wait-s", type=float, default=1800.)
    parser.add_argument("--startup-timeout-s", type=float)
    parser.add_argument("--hang-min-s", type=float)
    parser.add_argument("--max-wall-s", type=float)
    parser.add_argument("--poll-s", type=float)
    parser.add_argument("--sample-interval-s", type=float)
    parser.add_argument("--term-grace-s", type=float)
    parser.add_argument("--child-arg", action="append", default=[], help="extra child argument (fake backend)")
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    run = config["runs"][args.run]
    if not args.max_global_steps and run.get("max_global_steps"):
        args.max_global_steps = int(run["max_global_steps"])      # a declared reduced run (LR-D fault probes)
    gpu = None if (args.no_gpu_telemetry or args.backend == "fake") else args.gpu_index
    if args.backend == "trim" and gpu is None:
        parser.error("--gpu-index is required for the trim backend (VRAM telemetry and the free-VRAM check)")
    override = {k: v for k, v in dict(startup_timeout_s=args.startup_timeout_s, hang_min_s=args.hang_min_s,
                                      max_wall_s=args.max_wall_s, poll_s=args.poll_s,
                                      sample_interval_s=args.sample_interval_s,
                                      term_grace_s=args.term_grace_s).items() if v is not None}
    policy = S.SupervisorPolicy.from_config(config, run, **override)
    run_dir = new_run_dir(Path(args.out_root), args.name or args.run.lower().replace("-", "_"))
    meta = dict(run=args.run, config=os.path.relpath(args.config, ROOT), backend=args.backend,
                config_sha256=hashlib.sha256(config_bytes).hexdigest(), git_head=git("rev-parse", "HEAD"),
                git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")), gpu_index=gpu,
                cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), tmpdir=os.environ.get("TMPDIR"),
                argv=sys.argv, max_global_steps=args.max_global_steps, reduced=bool(args.max_global_steps),
                fault_inject=args.fault_inject, mode=run.get("mode", "rollout"),
                split=config.get("classification", {}).get("split", "p1_4_v1"))
    free_check = None
    if gpu is not None:
        need = float(run.get("own_vram_bound_mib", 0)) + float(config["common"].get("free_vram_margin_mib", 4096))
        deadline = time.time() + args.free_vram_wait_s
        while True:
            free = S.free_vram_mib(gpu)
            free_check = dict(free_mib=free, required_mib=need, t=time.time())
            if free is not None and free >= need:
                break
            if time.time() > deadline:
                run_dir.mkdir(parents=True)
                with S.Ledger(run_dir / "supervisor.jsonl", schema=S.SUPERVISOR_SCHEMA) as ledger:
                    ledger.write("final", status="BLOCKED_INSUFFICIENT_FREE_VRAM", free_check=free_check, meta=meta)
                print(json.dumps(dict(run_dir=str(run_dir), status="BLOCKED_INSUFFICIENT_FREE_VRAM", **free_check)))
                return 3
            time.sleep(60.)
    meta["free_vram_check"] = free_check

    def command(attempt, attempt_dir, resume_dirs):
        argv = [args.child_python, str(CHILD), "--config", args.config, "--run", args.run,
                "--attempt-dir", str(attempt_dir), "--backend", args.backend]
        for d in resume_dirs:
            argv += ["--resume-from", str(d)]
        if args.max_global_steps:
            argv += ["--max-global-steps", str(args.max_global_steps)]
        if gpu is not None:
            argv += ["--gpu-index", str(gpu)]
        return argv + list(args.child_arg)

    sampler = S.nvidia_smi_sampler(gpu) if gpu is not None else None
    sup = S.Supervisor(run_dir, command, policy=policy, sampler=sampler, fault_inject=args.fault_inject, meta=meta)
    final = sup.run()
    print(json.dumps(dict(run_dir=str(run_dir), **final)))
    return 0 if final["status"].startswith("COMPLETE") else 2


if __name__ == "__main__":
    sys.exit(main())
