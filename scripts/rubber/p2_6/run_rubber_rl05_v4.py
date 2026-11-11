"""RL-05 GPU sanity of ``configs/rubber/rubber_rl_tests_v4.json`` (supersedes RL-05 only; the last scripted attempt).

The P6.3 runner (``run_rubber_rl_gpu.py``: meta, GPU sampling, SHARED / EXCLUSIVE label, BLOCKED below 4 GB, refuses to
overwrite) bound to the v4 protocol, with row ``RL05v4``: N 8 (slot_seed 0), the declared fresh seeds, 80-step episodes
(10 preload), one episode with ``rubber_scripted_pivot_hold_v1`` (parameters read from the protocol) and one with the
zero action from the same seeds; pass:
scripted mean return > zero-action mean return in >= 7 of 8 envs and >= 1 evaluator success (unchanged from v2).

    scripts/rubber/p2_6/queue_gpu_rows.sh is for the P6.3 runner; for this one:
    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
      TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_rubber_rl05_v3.py \
      --rows RL05v4 --out artifacts/rl_ready/p2_6/<dir> --device cuda:0 --gpu-index 0
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("rl_gpu", ROOT/"scripts"/"rubber"/"p2_6"/"run_rubber_rl_gpu.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
G.PROTOCOL = ROOT/"configs"/"rubber"/"rubber_rl_tests_v4.json"
G.PROTOCOL_SHA = "ce8dcc1a5bfab15d96e02e2f94433f273229584b6295d8929b5ba5c9d93acc0b"
G.ROWS = ("RL05v4",)


class RunnerV4(G.Runner):
    def RL05v4(self, steps=80):
        import torch
        from robotic_cutting_sim.rl.rubber_scripted import ScriptedPivotHoldV1
        t5 = json.loads(G.PROTOCOL.read_text())["tests"]["RL-05"]
        cfg = t5["controller_v4"]
        params = {k: cfg[k] for k in ScriptedPivotHoldV1.NAMES}
        seed0 = int(t5["formal_seed_base"])
        fam = self.family(num_envs=8, slot_seed=0, max_episode_steps=steps)
        env = self.env(fam)
        ctrl = ScriptedPivotHoldV1(8, env.torch_device, fam.max_translation, fam.max_rotation, fam.control_dt,
                                   **params)
        seeds = [seed0+e for e in range(8)]
        out = {}
        for policy in ("scripted", "zero"):
            obs, _ = env.reset(seeds=seeds)
            ctrl.reset()
            ret = torch.zeros(8, dtype=torch.float64, device=env.torch_device)
            t0 = time.perf_counter()
            tips = []
            for k in range(steps):
                a = ctrl(obs["policy"]) if policy == "scripted" else torch.zeros((8, 6), dtype=torch.float32,
                                                                                 device=env.torch_device)
                obs, r, term, trunc, info = env.step(a)
                ret += r.double()
                tips.append(info["tip_distance_m"].cpu().tolist())
            ev = env.evaluate()
            out[policy] = {"returns": ret.cpu().tolist(), "success": [x["success"] for x in ev],
                           "tip_distance_m": [x["tip_distance_m"] for x in ev],
                           "tip_distance_every_10_steps": tips[9::10],
                           "termination": [x["termination_reason"] for x in ev], "invalid": [x["invalid_reason"]
                                                                                            for x in ev],
                           "wall_s": time.perf_counter()-t0}
            self.note({"RL05v4": policy, "returns": out[policy]["returns"], "success": out[policy]["success"]})
        better = [a > b for a, b in zip(out["scripted"]["returns"], out["zero"]["returns"])]
        succ = sum(out["scripted"]["success"])
        res = {"verdict": "PASS" if sum(better) >= 7 and succ >= 1 else "FAIL", "scripted_better": better,
               "scripted_better_count": sum(better), "scripted_successes": succ, "episodes": out,
               "slots": list(fam.env_slots), "seeds": seeds, "steps": steps, "controller": ctrl.params(),
               "reset_walls_s": env.reset_walls_s}
        env.close()
        return res


G.Runner = RunnerV4

if __name__ == "__main__":
    if "--rows" not in sys.argv:
        sys.argv += ["--rows", "RL05v4"]
    G.main()
