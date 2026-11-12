"""RL-12 host-memory growth over resets of ``configs/rubber/rubber_rl_tests_v5.json`` (P6.7 review Z-a / F1).

The P6.3 runner (``run_rubber_rl_gpu.py``: meta, GPU sampling, SHARED / EXCLUSIVE label, BLOCKED below 4 GB, refuses to
overwrite) bound to the v5 protocol, with row ``RL12``: N 8 (slot_seed 0), full reset with seeds 12000 + e, then 80
iterations of (one zero-action batch step, partial reset of env i mod 8 with seed 12100 + i); after each reset
gc.collect() and a sample of the process VmRSS and of the F8 inverse cache. Pass: cache <= its cap at every sample,
RSS growth over resets 20..80 <= 32 MiB, finite outputs and no infrastructure error.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock env CUDA_VISIBLE_DEVICES=<N> OPENBLAS_NUM_THREADS=1 \
      TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_rubber_rl12_hostmem.py \
      --rows RL12 --out artifacts/rl_ready/p2_6/<dir> --device cuda:0 --gpu-index <N>
"""
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
_spec = importlib.util.spec_from_file_location("rl_gpu", ROOT/"scripts"/"rubber"/"p2_6"/"run_rubber_rl_gpu.py")
G = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(G)
G.PROTOCOL = ROOT/"configs"/"rubber"/"rubber_rl_tests_v5.json"
G.PROTOCOL_SHA = "5200ed1be70e5eadae3e90a65f98f3a724c159516fb7e5da11522091c631f0eb"
G.ROWS = ("RL12",)


def rss_mib():
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])/1024.0
    raise RuntimeError("VmRSS not found")


class RunnerV5(G.Runner):
    def RL12(self, iterations=80, warmup=20, bound_mib=32.0):
        torch = self.torch
        t12 = json.loads(G.PROTOCOL.read_text())["tests"]["RL-12"]
        assert "32 MiB" in t12["pass"] and "80 iterations" in t12["procedure"]
        N = 8
        fam = self.family(num_envs=N, slot_seed=0, max_episode_steps=1000)
        env = self.env(fam)
        m = env.material
        cap = m.proj_inv_cache_max
        obs, _ = env.reset(seeds=[12000+e for e in range(N)])
        zero = torch.zeros((N, 6), dtype=torch.float32, device=env.torch_device)
        gc.collect()
        samples = [{"reset": 0, "rss_mib": rss_mib(), "cache_entries": len(m._proj_inv_cache), "cache_mb": 0.0}]
        finite_bad, vram = [], []
        t0 = time.perf_counter()
        for i in range(iterations):
            obs, r, term, trunc, info = env.step(zero)
            if not (bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(r).all())):
                finite_bad.append(i)
            e = i % N
            obs, _ = env.reset(env_ids=[e], seeds=[12100+i])
            gc.collect()
            cache = m._proj_inv_cache
            samples.append({"reset": i+1, "env": e, "rss_mib": rss_mib(), "cache_entries": len(cache),
                            "cache_mb": sum(v.nbytes for v in cache.values())/1e6,
                            "density_mass": float(m.stage_mass.numpy()[e]), "wall_s": time.perf_counter()-t0})
            if (i+1) % 10 == 0:
                self.sync()
                vram.append({"reset": i+1, "own_mib": self.gpu.own_mib()})
                self.note({"RL12": i+1, "rss_mib": samples[-1]["rss_mib"], "cache_entries": len(cache)})
        scored = [s for s in samples if s["reset"] >= warmup]
        rss = np.array([s["rss_mib"] for s in scored])
        x = np.array([s["reset"] for s in scored], float)
        slope = float(np.polyfit(x, rss, 1)[0])
        growth = float(rss.max()-rss[0])
        max_entries = max(s["cache_entries"] for s in samples)
        masses = [s["density_mass"] for s in samples[1:]]
        walls = [w["wall_s"] for w in env.reset_walls_s[1:]]
        res = {"iterations": iterations, "warmup_resets": warmup, "bound_mib": bound_mib, "cache_cap": cap,
               "cache_max_entries": max_entries, "rss_growth_scored_mib": growth, "rss_slope_mib_per_reset": slope,
               "rss_first_scored_mib": float(rss[0]), "rss_max_scored_mib": float(rss.max()),
               "distinct_masses": len(set(masses)), "samples": samples, "vram_samples": vram,
               "finite_bad_iterations": finite_bad, "infrastructure_error": env.infrastructure_error,
               "reset_median_s": float(np.median(walls)) if walls else None,
               "reset_p95_s": float(np.percentile(walls, 95)) if walls else None,
               "wall_s": time.perf_counter()-t0, "slots": list(fam.env_slots)}
        res["verdict"] = "PASS" if (max_entries <= cap and growth <= bound_mib and not finite_bad
                                    and env.infrastructure_error is None) else "FAIL"
        env.close()
        return res


G.Runner = RunnerV5

if __name__ == "__main__":
    if "--rows" not in sys.argv:
        sys.argv += ["--rows", "RL12"]
    G.main()
