"""DIAGNOSTIC (never a protocol row): develop the RL-05 v4 hold phase on the FULL-resolution GPU model, development seeds.

One batch of N = len(variants) x len(seeds) envs of ``rubber_strip_drag_v2`` (the declared task, rubber_ref_v1, GPU,
CUDA graphs) runs 80-step episodes (10 preload) with ``ScriptedPivotHoldV1`` whose parameters differ per env
(variant), plus per-step tip distance, hold flag and action status. Seeds must be >= 5000 (development); the formal
RL-05 v4 seeds are declared separately and never used here.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
      TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/diag_rl05_gpu_dev.py \
      --out artifacts/rl_ready/p2_6/<dir> --variants V.json --seeds 5000-5003 --slots L150_t5,L125_t6,L100_t6,L100_t4
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True, help="JSON file: list of parameter dicts")
    ap.add_argument("--seeds", default="5000-5003")
    ap.add_argument("--slots", default="L150_t5,L125_t6,L100_t6,L100_t4")
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo)+1))
    if min(seeds) < 5000:
        sys.exit("development seeds are >= 5000")
    slots = a.slots.split(",")
    variants = json.loads(Path(a.variants).read_text())
    free = float(subprocess.run(["nvidia-smi", "--id="+os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                                 "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True,
                                text=True).stdout.split()[0])
    if free < 4096:
        sys.exit(f"BLOCKED: {free} MB free < 4096")
    out.mkdir(parents=True)
    import torch
    import warp as wp
    wp.config.log_level = 30
    from robotic_cutting_sim.rl.rubber_env import RubberVectorEnv
    from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily
    from robotic_cutting_sim.rl.rubber_scripted import ScriptedPivotHoldV1
    env_slots, env_seeds, env_var = [], [], []
    for vi in range(len(variants)):
        for si, s in enumerate(seeds):
            env_slots.append(slots[si % len(slots)])
            env_seeds.append(s)
            env_var.append(vi)
    N = len(env_slots)
    fam = RubberStripTaskFamily(env_slots, max_episode_steps=a.steps)
    env = RubberVectorEnv(N, task=fam, device=a.device, strict=False)
    params = {k: [variants[v].get(k, getattr(ScriptedPivotHoldV1.__init__, "__kwdefaults__")[k]) for v in env_var]
              for k in ScriptedPivotHoldV1.NAMES}
    ctrl = ScriptedPivotHoldV1(N, env.torch_device, fam.max_translation, fam.max_rotation, fam.control_dt, **params)
    git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    meta = {"label": "DIAGNOSTIC rl05 v4 hold-phase development (full resolution, GPU)", "argv": sys.argv,
            "commit": git, "variants": variants, "env_slots": env_slots, "env_seeds": env_seeds, "env_variant": env_var,
            "free_mb": free, "start": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    obs, _ = env.reset(seeds=env_seeds)
    ctrl.reset()
    rows = []
    t0 = time.perf_counter()
    for k in range(a.steps):
        act = ctrl(obs["policy"])
        obs, r, term, trunc, info = env.step(act)
        rows.append({"k": k+1, "d": info["tip_distance_m"].cpu().tolist(), "hold": ctrl.hold[:, 0].cpu().tolist(),
                     "status": info["action_status"].cpu().tolist(), "r": r.cpu().tolist()})
    ev = env.evaluate()
    res = {"wall_s": time.perf_counter()-t0, "rows": rows,
           "envs": [{"env": e, "variant": env_var[e], "slot": env_slots[e], "seed": env_seeds[e],
                     "return": float(sum(q["r"][e] for q in rows)), "success": ev[e]["success"],
                     "final_d": ev[e]["tip_distance_m"], "within_streak": ev[e]["within_streak"],
                     "invalid": ev[e]["invalid_reason"]} for e in range(N)]}
    (out/"result.json").write_text(json.dumps(res, default=float))
    for vi, v in enumerate(variants):
        es = [x for x in res["envs"] if x["variant"] == vi]
        print(vi, v, "successes", sum(x["success"] for x in es), "final_d_mm",
              [round(x["final_d"]*1e3, 1) for x in es], "streak", [x["within_streak"] for x in es], flush=True)
    env.close()


if __name__ == "__main__":
    main()
