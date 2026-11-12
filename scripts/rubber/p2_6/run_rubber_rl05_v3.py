"""RL-05 GPU sanity of ``configs/rubber/rubber_rl_tests_v3.json`` (the only row v3 supersedes).

The P6.3 runner (``run_rubber_rl_gpu.py``: meta, GPU sampling, SHARED / EXCLUSIVE label, BLOCKED below 4 GB, refuses to
overwrite) bound to the v3 protocol, with row ``RL05v3``: N 8 (slot_seed 0), reset seeds 7000 + e, 80-step episodes
(10 preload), one episode with ``rubber_scripted_pivot_v1`` and one with the zero action from the same seeds; pass:
scripted mean return > zero-action mean return in >= 7 of 8 envs and >= 1 evaluator success (unchanged from v2).

    scripts/rubber/p2_6/queue_gpu_rows.sh is for the P6.3 runner; for this one:
    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 \
      TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_rubber_rl05_v3.py \
      --rows RL05v3 --out artifacts/rl_ready/p2_6/<dir> --device cuda:0 --gpu-index 0
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
G.PROTOCOL = ROOT/"configs"/"rubber"/"rubber_rl_tests_v3.json"
G.PROTOCOL_SHA = "7dc077639ab8651e4ba6a92eac037486e4268834ec9f47f32b635cb4a93c6e2f"
G.ROWS = ("RL05v3",)


class RunnerV3(G.Runner):
    def RL05v3(self, steps=80):
        import torch
        from robotic_cutting_sim.rl.rubber_scripted import PIVOT_V1, scripted_pivot_v1
        cfg = json.loads(G.PROTOCOL.read_text())["tests"]["RL-05"]["controller_v3"]
        declared = {k.replace("_m", "") if k in ("r_fine_m", "eps_m") else k: v for k, v in cfg.items()
                    if k not in ("id", "class")}
        if declared != PIVOT_V1:
            raise RuntimeError(f"controller parameters {PIVOT_V1} differ from the declaration {cfg}")
        fam = self.family(num_envs=8, slot_seed=0, max_episode_steps=steps)
        env = self.env(fam)
        ctrl = scripted_pivot_v1(8, env.torch_device, fam.max_translation, fam.max_rotation)
        seeds = [7000+e for e in range(8)]
        out = {}
        for policy in ("scripted", "zero"):
            obs, _ = env.reset(seeds=seeds)
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
            self.note({"RL05v3": policy, "returns": out[policy]["returns"], "success": out[policy]["success"]})
        better = [a > b for a, b in zip(out["scripted"]["returns"], out["zero"]["returns"])]
        succ = sum(out["scripted"]["success"])
        res = {"verdict": "PASS" if sum(better) >= 7 and succ >= 1 else "FAIL", "scripted_better": better,
               "scripted_better_count": sum(better), "scripted_successes": succ, "episodes": out,
               "slots": list(fam.env_slots), "seeds": seeds, "steps": steps, "controller": ctrl.params(),
               "reset_walls_s": env.reset_walls_s}
        env.close()
        return res


G.Runner = RunnerV3

if __name__ == "__main__":
    if "--rows" not in sys.argv:
        sys.argv += ["--rows", "RL05v3"]
    G.main()
