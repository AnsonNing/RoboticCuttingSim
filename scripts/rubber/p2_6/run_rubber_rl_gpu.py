"""P6.3 GPU rows of ``configs/rubber/rubber_rl_tests_v1.json`` (P2-6, ADR 0023 §23.2 decision 7).

Rows (``--rows``, comma list, run in this order in one process): RL01 (contract N 1 / N 8, both profiles), RL02b (L2
every step, N 8), RL02c (L3 rubber_l3_band_v4: N 8 batch vs 20 + 10 single GPU repeats for envs 0 / 1), RL03 (device
parameter rows repeat for the same seeds; in range), RL04b (the -09 v2 rule at N 4, k 5 / 5 interleaved), RL05
(scripted vs zero-action episodes, 80 steps), RL06 (long rollout >= 400 batch steps), RL07 (throughput / VRAM, S1 and
plane-only contact cost, Phase 1 workload A re-measured with the frozen harness), RL08 (dt / spacing, single env).

Run under the GPU lock (one RCS job per GPU) in the Torch venv:

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 TMPDIR=/mnt/data/ningan/tmp \
      PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_rubber_rl_gpu.py \
        --rows RL01,RL02b --out artifacts/rl_ready/p2_6/<dir>

Refuses < 4 GB free (BLOCKED), refuses an existing --out, checks the protocol / task config SHA pins, records the
commit, the dirty flag, driver, GPU, co-tenant nvidia-smi samples (SHARED_GPU / EXCLUSIVE_AT_SAMPLES) and per-row
wall times. Every row writes ``rows/<row>.json``; ``progress.log`` holds one JSON line per event.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_rl_tests_v1.json"
PROTOCOL_SHA = "8e4b2593e968532e10808d4828f093df964747c18c8b98d260d1eae68535fd4c"
TASK_SHA = "fba43424372e17fe926fad697ff8336ea7c5ed0c2d8ecdbdefb9978a994f4397"
MIN_FREE_MB = 4096
ROWS = ("RL01", "RL02b", "RL02c", "RL03", "RL04b", "RL05", "RL06", "RL07", "RL08")
FULL_ORDER = ("L100_t4", "L100_t5", "L100_t6", "L125_t4", "L125_t5", "L125_t6", "L150_t4", "L150_t5", "L150_t6")
L3_SLOTS = ["L100_t4", "L100_t6", "L125_t4", "L125_t5", "L125_t6", "L150_t4", "L150_t5", "L150_t6"]


def sh(*cmd):
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()


class Gpu:
    def __init__(self, index):
        self.index = index
        self.uuid = sh("nvidia-smi", f"--id={index}", "--query-gpu=uuid", "--format=csv,noheader")
        self.samples = []

    def free_mb(self):
        return float(sh("nvidia-smi", f"--id={self.index}", "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits").splitlines()[0])

    def apps(self):
        rows = sh("nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits")
        out = []
        for line in rows.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3 and parts[1] == self.uuid:
                out.append({"pid": int(parts[0]), "used_mib": float(parts[2])})
        return out

    def sample(self, tag):
        q = sh("nvidia-smi", f"--id={self.index}", "--query-gpu=memory.used,memory.total,utilization.gpu",
               "--format=csv,noheader")
        s = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "tag": tag, "gpu": q, "apps": self.apps()}
        self.samples.append(s)
        return s

    def own_mib(self):
        return max([a["used_mib"] for a in self.apps() if a["pid"] == os.getpid()] or [float("nan")])

    def sharing(self, since=0):
        others = sorted({a["pid"] for s in self.samples[since:] for a in s["apps"] if a["pid"] != os.getpid()})
        return {"label": "SHARED_GPU" if others else "EXCLUSIVE_AT_SAMPLES", "co_tenant_pids": others,
                "samples": self.samples[since:]}


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class Runner:
    def __init__(self, args):
        import torch
        import warp as wp
        self.torch, self.wp, self.args = torch, wp, args
        self.device = args.device
        self.out = Path(args.out)
        self.gpu = Gpu(args.gpu_index)
        self.log = open(self.out/"progress.log", "w")

    def note(self, obj):
        line = json.dumps(obj, default=float)
        print(line, flush=True)
        self.log.write(line+"\n")
        self.log.flush()

    def sync(self):
        self.wp.synchronize_device(self.device)
        self.torch.cuda.synchronize()

    # ------------------------------------------------------------------ helpers

    def family(self, slots=None, **kw):
        from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily
        return RubberStripTaskFamily(slots, **kw)

    def env(self, fam, profile="state", strict=False):
        from robotic_cutting_sim.rl.rubber_env import RubberVectorEnv
        return RubberVectorEnv(fam.num_envs, task=fam, device=self.device, strict=strict, observation_profile=profile)

    def scripted(self, env, obs):
        """rubber_scripted_drag_v1 (protocol common.scripted_policies)."""
        torch = self.torch
        N = env.num_envs
        tip = obs["policy"][:, 20+3*8:20+3*8+3]          # marker 8 (markers start after 7 + 6 + 1 + 6 = 20 dims)
        a = torch.zeros((N, 6), dtype=torch.float32, device=env.torch_device)
        a[:, :2] = ((env.goal[:, :2]-tip[:, :2].double())/env.family.max_translation).clamp(-1, 1).float()
        return a

    def open_loop(self, env, k):
        from robotic_cutting_sim.rl.rubber_contract import contract_actions
        return contract_actions(env.num_envs, k, env.torch_device)

    def stats_batch(self, env, e):
        from robotic_cutting_sim.tlmpm_rubber_v1 import DIAG_COLUMNS as C
        m = env.material
        s = m.summary.numpy()[e]
        out = {f"marker{k}_{a}": float(v) for k, row in enumerate(m.markers()[e]) for a, v in zip("xyz", row)}
        out.update({f"grip_wrench{i}": float(v) for i, v in enumerate(m.wrench.numpy()[e, 0])})
        out.update({f"plane_wrench{i}": float(v) for i, v in enumerate(m.contact.plane_wrench.numpy()[e, 0])})
        out["psi_eq"] = float(m.visco_summary.numpy()[e, 0])
        for c in ("kinetic_particle", "gravity", "Jbar_min", "Jbar_max", "cfl_eff_max"):
            out[c] = float(s[C.index(c)])
        return out

    def stats_single(self, live):
        d = live.last_diagnostics
        out = {f"marker{k}_{a}": float(v) for k, row in enumerate(live.markers()) for a, v in zip("xyz", row)}
        out.update({f"grip_wrench{i}": float(v) for i, v in enumerate(np.asarray(live.last_wrench)[0])})
        out.update({f"plane_wrench{i}": float(v) for i, v in enumerate(live.last_contact["plane_wrench"][0])})
        out["psi_eq"] = float(live.last_visco["psi_eq"])
        for c in ("kinetic_particle", "gravity", "Jbar_min", "Jbar_max", "cfl_eff_max"):
            out[c] = float(d[c])
        return out

    # ------------------------------------------------------------------ rows

    def RL01(self):
        from robotic_cutting_sim.rl import vector_env as VE
        from robotic_cutting_sim.rl.rubber_contract import check_rubber_vector_env_contract
        reports = []
        for N, slots in ((1, ["L125_t4"]), (8, None)):
            for profile in ("state", "privileged_state"):
                fam = self.family(slots, num_envs=N, slot_seed=0)
                t0 = time.perf_counter()
                with VE.open_vector_env(N, task=fam, backend="kinematic_grip_batched", device=self.device,
                                        observation_profile=profile, strict=False) as env:
                    build = time.perf_counter()-t0
                    try:
                        rep = check_rubber_vector_env_contract(env)
                    except AssertionError as err:
                        rep = {"status": "FAIL", "clause": str(err)}
                    rep.update(N=N, profile=profile, slots=list(fam.env_slots), build_s=build,
                               wall_s=time.perf_counter()-t0)
                reports.append(rep)
                self.note({"RL01": {k: rep.get(k) for k in ("N", "profile", "status", "build_s", "wall_s")}})
                gc.collect()
        return {"verdict": "PASS" if all(r["status"] == "PASS" for r in reports) else "FAIL", "reports": reports}

    def RL02b(self, steps=60):
        from robotic_cutting_sim.batched.rubber_contact_batch import contact_audit
        torch = self.torch
        fam = self.family(num_envs=8, slot_seed=0)
        env = self.env(fam)
        obs, _ = env.reset(seeds=[1000+e for e in range(8)])
        mass0 = env.material.solver.mass.numpy().copy()
        rows, ok = [], True
        for k in range(steps):
            obs, r, term, trunc, info = env.step(self.scripted(env, obs))
            audits = env.material.audits()
            caud = contact_audit(env.material.solver)
            row = {"step": k+1, "obs_finite": bool(torch.isfinite(obs["policy"]).all()),
                   "reward_finite": bool(torch.isfinite(r).all()),
                   "state_finite": bool(np.isfinite(env.material.solver.x.numpy()).all()),
                   "mass_constant": bool(np.array_equal(env.material.solver.mass.numpy(), mass0)),
                   "audits_zero": all(int(v) == 0 for v in audits.values()),
                   "contact_audit_zero": all(int(v) == 0 for v in (caud.values() if isinstance(caud, dict) else [caud])),
                   "invalid": env.material.invalid_reason.numpy().tolist()}
            row["pass"] = (row["obs_finite"] and row["reward_finite"] and row["state_finite"] and row["mass_constant"]
                           and row["audits_zero"] and row["contact_audit_zero"] and not any(row["invalid"]))
            ok &= row["pass"]
            rows.append(row)
        ev = env.evaluate()
        res = {"verdict": "PASS" if ok else "FAIL", "steps": steps, "slots": list(fam.env_slots),
               "failing_steps": [r["step"] for r in rows if not r["pass"]], "rows": rows,
               "final_tip_distance_m": [x["tip_distance_m"] for x in ev], "receipt": env.receipt()}
        env.close()
        return res

    def RL02c(self, steps=50, n_ref=20, n_held=10):
        from robotic_cutting_sim import rubber_l3
        from robotic_cutting_sim.rl.rubber_env import drive_single_reference
        fam = self.family(L3_SLOTS)
        env = self.env(fam)
        env.reset(seeds=[1100+e for e in range(8)])
        poses = {0: [], 1: []}
        batch = {0: [], 1: []}
        t0 = time.perf_counter()
        for k in range(steps):
            env.step(self.open_loop(env, k))
            q, b, f = env.q1_t.cpu().numpy(), env.b1_t.cpu().numpy(), env.traction_t.cpu().numpy()
            for e in (0, 1):
                poses[e].append((q[e].copy(), b[e].copy(), f[e].copy()))
                batch[e].append(self.stats_batch(env, e))
        batch_wall = time.perf_counter()-t0
        materials = {e: env.samples[e]["material"] for e in (0, 1)}
        invalid_batch = env.material.invalid_reason.numpy().tolist()
        env.close()
        del env
        gc.collect()
        result = {"rule": rubber_l3.RULE_ID, "alpha_family": 0.05, "steps": steps, "slots": L3_SLOTS[:2],
                  "batch_wall_s": batch_wall, "invalid_batch": invalid_batch, "envs": {}}
        ref_all, cand_all = {}, {"batch": {}}
        for e in (0, 1):
            live = fam.build_life(L3_SLOTS[e], materials[e], device=self.device)
            runs = []
            t1 = time.perf_counter()
            for i in range(n_ref+n_held):
                live.reset()
                series = []
                for q, b, f in poses[e]:
                    drive_single_reference(live, q, b, f)
                    series.append(self.stats_single(live))
                runs.append({"series": series, "invalid": int(live.invalid_reason)})
            result["envs"][e] = {"single_wall_s": time.perf_counter()-t1, "single_invalid": [r["invalid"] for r in runs],
                                 "particles": int(live.scene.n), "substeps": int(live.substeps)}
            names = list(batch[e][0])
            for n in names:
                key = f"env{e}.{n}"
                ref_all[key] = np.array([[s[n] for s in r["series"]] for r in runs[:n_ref]])
                cand_all["batch"][key] = np.array([s[n] for s in batch[e]])
                for j, r in enumerate(runs[n_ref:]):
                    cand_all.setdefault(f"heldout_{j}", {})[key] = np.array([s[n] for s in r["series"]])
            del live
            gc.collect()
        # held-out candidates need every statistic: evaluate per candidate set
        ev = rubber_l3.evaluate(ref_all, {"batch": cand_all["batch"]}, alpha_family=0.05)
        held = rubber_l3.evaluate(ref_all, {k: v for k, v in cand_all.items() if k != "batch"}, alpha_family=0.05)
        result.update(statistics=len(ref_all), batch=ev["runs"]["batch"],
                      worst=sorted(ev["rows"], key=lambda r: -r["M_over_upper"])[:10],
                      heldout={k: v for k, v in held["runs"].items()},
                      heldout_fail=sum(v["verdict"] != "PASS" for v in held["runs"].values()),
                      bounds_methods={k: v["method"] for k, v in ev["bounds"].items()})
        result["verdict"] = ev["runs"]["batch"]["verdict"] if not any(invalid_batch) else "FAIL"
        return result

    def RL03(self):
        fam = self.family(num_envs=8, slot_seed=0)
        env = self.env(fam)
        seeds = [4000+e for e in range(8)]
        m = env.material
        reads = []
        for _ in range(2):
            env.reset(seeds=seeds)
            reads.append({"params": m.solver.params.numpy().copy(), "mass": m.solver.mass.numpy().copy(),
                          "params64": m.solver.params64.numpy().copy(), "traction": env.traction_t.cpu().numpy(),
                          "goal": env.goal.cpu().numpy(), "preload": env.preload.cpu().numpy()})
            env.step(self.open_loop(env, 0))
        same = {k: bool(reads[0][k].tobytes() == reads[1][k].tobytes()) for k in reads[0]}
        in_range = []
        for e in range(8):
            d = env.samples[e]["draws"]
            in_range.append(all(fam.ranges[k][0] <= d[k] <= fam.ranges[k][1] for k in d))
            fam.check_material(env.samples[e]["material"], fam.slot_set[fam.env_slots[e]]["n_t"])
        env.close()
        return {"verdict": "PASS" if all(same.values()) and all(in_range) else "FAIL", "identical": same,
                "in_range": in_range, "slots": list(fam.env_slots)}

    def RL04b(self, steps=15, k_side=5):
        band = load_script("cont09_v2", ROOT/"scripts"/"rubber"/"p2_4b"/"gpu_contact_batched_check_v2.py")
        cfg = json.loads(PROTOCOL.read_text())["tests"]["RL-04"]["thresholds"]
        slots = ["L100_t4", "L100_t5", "L125_t4", "L100_t6"]
        fam = self.family(slots)
        env = self.env(fam)
        runs = {"A": [], "B": []}
        t0 = time.perf_counter()
        for i in range(2*k_side):
            side = "A" if i % 2 == 0 else "B"
            seeds = [1200, 1300 if side == "B" else 1201, 1202, 1203]
            env.reset(seeds=seeds)
            rec = []
            for k in range(steps):
                a = self.open_loop(env, k)
                if side == "B":
                    a[1] = -a[1]
                env.step(a)
                m = env.material
                mk = m.markers()
                mv = m.marker_v.numpy().reshape(4, 9, 3)
                w, pw, s = m.wrench.numpy(), m.contact.plane_wrench.numpy(), m.summary.numpy()
                rec.append({e: {"markers": mk[e], "marker_v": mv[e], "grip_wrench": w[e, 0], "plane_wrench": pw[e, 0],
                                "summary": s[e]} for e in (0, 2, 3)})
            runs[side].append({"obs": rec, "invalid": env.material.invalid_reason.numpy().tolist()})
        ok, rows, n, summ = band.band_v2_all(runs["A"], runs["B"], (0, 2, 3), cfg["lam"], cfg["eps_rel"])
        inval = [r["invalid"] for r in runs["A"]+runs["B"]]
        env.close()
        return {"verdict": "PASS" if ok and not any(any(v) for v in inval) else "FAIL", "summary": summ,
                "worst_rows": rows[:10], "invalid": inval, "wall_s": time.perf_counter()-t0, "slots": slots,
                "thresholds": cfg}

    def RL05(self, steps=80):
        torch = self.torch
        fam = self.family(num_envs=8, slot_seed=0, max_episode_steps=steps)
        env = self.env(fam)
        seeds = [2000+e for e in range(8)]
        out = {}
        for policy in ("scripted", "zero"):
            obs, _ = env.reset(seeds=seeds)
            ret = torch.zeros(8, dtype=torch.float64, device=env.torch_device)
            t0 = time.perf_counter()
            for k in range(steps):
                a = self.scripted(env, obs) if policy == "scripted" else torch.zeros((8, 6), dtype=torch.float32,
                                                                                    device=env.torch_device)
                obs, r, term, trunc, info = env.step(a)
                ret += r.double()
            ev = env.evaluate()
            out[policy] = {"returns": ret.cpu().tolist(), "success": [x["success"] for x in ev],
                           "tip_distance_m": [x["tip_distance_m"] for x in ev],
                           "termination": [x["termination_reason"] for x in ev], "wall_s": time.perf_counter()-t0}
            self.note({"RL05": policy, "returns": out[policy]["returns"], "success": out[policy]["success"]})
        better = [a > b for a, b in zip(out["scripted"]["returns"], out["zero"]["returns"])]
        succ = sum(out["scripted"]["success"])
        env.close()
        return {"verdict": "PASS" if sum(better) >= 7 and succ >= 1 else "FAIL", "scripted_better": better,
                "scripted_successes": succ, "episodes": out, "slots": list(fam.env_slots), "steps": steps}

    def RL06(self, steps=400, horizon=60):
        torch = self.torch
        from robotic_cutting_sim.rl.termination import decode_termination
        fam = self.family(num_envs=8, slot_seed=0, max_episode_steps=horizon)
        env = self.env(fam)
        N = 8
        resets = np.zeros(N, int)
        obs, _ = env.reset(seeds=[3000+e for e in range(N)])
        gen = torch.Generator(device=env.torch_device)
        gen.manual_seed(3)
        finite_bad, reasons, invalid_terms, partial = [], {}, 0, 0
        mem = []
        t0 = time.perf_counter()
        for k in range(steps):
            a = self.scripted(env, obs)
            a[4:] = torch.rand((4, 6), generator=gen, device=env.torch_device)*2-1
            obs, r, term, trunc, info = env.step(a)
            if not (bool(torch.isfinite(obs["policy"]).all()) and bool(torch.isfinite(r).all())):
                finite_bad.append(k+1)
            done = (term | trunc).cpu().numpy()
            stagger = [e for e in range(1, N) if k+1 == 7*e]
            ids = sorted(set(np.flatnonzero(done).tolist()) | set(stagger))
            for e in np.flatnonzero(term.cpu().numpy()):
                key = ",".join(decode_termination(int(info["termination_reason"][e])))
                reasons[key] = reasons.get(key, 0)+1
                invalid_terms += int(int(info["invalid_reason"][e]) != 0)
            if ids:
                resets[ids] += 1
                obs, _ = env.reset(env_ids=ids, seeds=[3000+100*int(resets[e])+e for e in ids])
                partial += len(ids)
            if (k+1) % 10 == 0 or k+1 == 20:
                self.sync()
                mem.append({"step": k+1, "own_mib": self.gpu.own_mib(), "wall_s": time.perf_counter()-t0})
        walls = [w["wall_s"] for w in env.reset_walls_s[1:]]
        warm = [m["own_mib"] for m in mem if m["step"] >= 20]
        growth = (max(warm)-warm[0]) if warm else float("nan")
        res = {"steps": steps, "horizon": horizon, "partial_resets": partial, "finite_bad_steps": finite_bad,
               "infrastructure_error": env.infrastructure_error, "vram_samples": mem, "vram_growth_mib": growth,
               "reset_p95_s": float(np.percentile(walls, 95)) if walls else None,
               "reset_median_s": float(np.median(walls)) if walls else None, "terminations": reasons,
               "invalid_terminations": invalid_terms, "invalid_termination_rate_per_episode":
               invalid_terms/max(partial, 1), "wall_s": time.perf_counter()-t0, "slots": list(fam.env_slots)}
        res["verdict"] = "PASS" if (not finite_bad and env.infrastructure_error is None and growth <= 64
                                    and partial >= 40) else "FAIL"
        env.close()
        return res

    def _throughput_row(self, label, fam, warm=5):
        env = self.env(fam)
        env.reset(seeds=[4000+e for e in range(fam.num_envs)])
        walls = []
        for k in range(1+warm):
            self.sync()
            t0 = time.perf_counter()
            env.step(self.open_loop(env, k))
            self.sync()
            walls.append(time.perf_counter()-t0)
        own = self.gpu.own_mib()
        m = env.material
        w = float(np.median(walls[1:]))
        N = fam.num_envs
        po = m.packed.particle_offset
        row = {"label": label, "N": N, "slots": list(fam.env_slots),
               "particles_per_env_mean": float(np.mean(np.diff(po))), "particles_total": int(po[-1]),
               "control_hz": 1.0/fam.control_dt, "material_dt_s": float(m.solver.dt),
               "substeps": int(m.solver.substeps), "n_t": sorted(set(env.n_t)), "beta": 20.0, "M": int(m.solver.M),
               "contact_law": fam.law, "cameras": 0, "recording": False, "cold_s": walls[0], "warm_walls_s": walls[1:],
               "env_steps_per_s": N/w, "aggregate_sim_time_per_wall": N*fam.control_dt/w,
               "ms_per_substep": 1e3*w/m.solver.substeps, "vram_peak_process_mib": own,
               "warp_mempool_high_mib": (self.wp.get_mempool_used_mem_high(self.device)/2**20
                                         if hasattr(self.wp, "get_mempool_used_mem_high") else None),
               "material_memory_bytes": m.memory_bytes(),
               "reset_s": env.reset_walls_s[-1]["wall_s"], "build_s": env.build_wall_s,
               "physics_verdict": "finite, invalid 0" if not any(m.invalid_reason.numpy()) else
               f"invalid {m.invalid_reason.numpy().tolist()}",
               "reduced_fidelity": fam.resolution, "graph": m.capture_receipt}
        env.close()
        del env
        gc.collect()
        self.note({"RL07": {k: row[k] for k in ("label", "N", "env_steps_per_s", "vram_peak_process_mib",
                                                "ms_per_substep")}})
        return row

    def _cost_arms(self, N=8, warm=5):
        from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
        fam = self.family(["L125_t4"]*N)

        def lives(arm):
            out = []
            for _ in range(N):
                if arm == "s1_off":
                    old = fam.profile["self_penetration_check"]
                    fam.profile["self_penetration_check"] = "off_v1"
                    out.append(fam.build_life("L125_t4"))
                    fam.profile["self_penetration_check"] = old
                elif arm == "contact_off":
                    orig = fam.contact_spec
                    fam.contact_spec = lambda: None
                    out.append(fam.build_life("L125_t4"))
                    fam.contact_spec = orig
                else:
                    out.append(fam.build_life("L125_t4"))
            return out
        arms = {a: {"batch": BatchedRubberMaterial(lives(a), device=self.device, strict=False), "walls": []}
                for a in ("profile", "s1_off", "contact_off")}
        order = list(arms)
        for k in range(1+warm):
            for a in order:
                b = arms[a]["batch"]
                self.sync()
                t0 = time.perf_counter()
                b.advance()
                self.sync()
                arms[a]["walls"].append(time.perf_counter()-t0)
            order.reverse()
        out = {}
        for a, v in arms.items():
            w = float(np.median(v["walls"][1:]))
            out[a] = {"ms_per_substep": 1e3*w/v["batch"].solver.substeps, "warm_walls_s": v["walls"][1:],
                      "cold_s": v["walls"][0]}
        out["s1_cost_rel"] = out["profile"]["ms_per_substep"]/out["s1_off"]["ms_per_substep"]-1
        out["plane_contact_cost_rel"] = out["profile"]["ms_per_substep"]/out["contact_off"]["ms_per_substep"]-1
        out["arms"] = ("profile: contact v3 table plane + S1; s1_off: contact, no detector; contact_off: S1, no "
                       "ContactSpec (the strip falls through the table: cost only, never physics)")
        del arms
        gc.collect()
        return out

    def _workload_a(self, N):
        run = self.out/f"workload_a_n{N}"
        var = self.out/"variant_default.json"
        if not var.exists():
            var.write_text("{}\n")
        cmd = [sys.executable, str(ROOT/"scripts"/"rl_ready"/"run_perf_acceptance.py"), "--variant-json", str(var),
               "--mode", "throughput", "--num-envs", str(N), "--seeds", "512", "--warmup-steps", "5", "--steps", "20",
               "--device", self.device, "--gpu-index", str(self.args.gpu_index), "--out", str(run)]
        t0 = time.perf_counter()
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, PYTHONPATH=str(ROOT/"src")))
        res = {"cmd": " ".join(cmd), "rc": p.returncode, "wall_s": time.perf_counter()-t0, "stderr_tail": p.stderr[-2000:]}
        rj = run/"result.json"
        if rj.exists():
            r = json.loads(rj.read_text())
            res["result_status"] = r.get("status")
            res["throughput"] = r.get("throughput") or {k: r[k] for k in r if "step" in k.lower()}
        return res

    def RL07(self):
        rows = []
        for N in (1, 4, 8, 16):
            rows.append(self._throughput_row("full_slot_set", self.family([FULL_ORDER[e % 9] for e in range(N)])))
        for N in (1, 4, 8, 16):
            rows.append(self._throughput_row("uniform_n_t4_L125_t4", self.family(["L125_t4"]*N)))
        for N in (1, 8):
            rows.append(self._throughput_row("rubber_rl_v1_reduced_fidelity",
                                             self.family([("L100_t4", "L150_t4")[e % 2] for e in range(N)],
                                                         resolution="rubber_rl_v1")))
        cost = self._cost_arms()
        self.note({"RL07_cost": {k: v for k, v in cost.items() if not isinstance(v, dict)}})
        wa = {N: self._workload_a(N) for N in (1, 8)}
        full = {r["N"]: r for r in rows if r["label"] == "full_slot_set"}
        gain = full[8]["env_steps_per_s"]/full[1]["env_steps_per_s"]
        vram8 = full[8]["vram_peak_process_mib"]
        verdict = "PASS" if gain > 1.0 and vram8 <= 4096 else "FAIL"
        return {"verdict": verdict, "scored": {"n8_over_n1_env_steps_gain": gain, "vram_peak_n8_mib": vram8,
                                               "vram_limit_mib": 4096}, "rows": rows, "cost_arms": cost,
                "phase1_workload_a": wa}

    def RL08(self, n_lift=10, n_hold=20, n_drag=30, lift=0.01, drag=0.03):
        from robotic_cutting_sim.rubber_gci import study
        from robotic_cutting_sim.rl.rubber_env import smootherstep
        fam = self.family(["L125_t4"])
        L = fam.slot_set["L125_t4"]["length_m"]
        cases = {"nt4_cfl0.5": dict(), "nt4_cfl0.25": dict(cfl=0.25), "nt6_cfl0.5": dict(n_t=6),
                 "nt8_cfl0.5": dict(n_t=8)}
        out = {}
        for name, kw in cases.items():
            live = fam.build_life("L125_t4", device=self.device, **kw)
            o = np.asarray(live.scene.groups[0]["origin"], float)
            tips, forces = [], []
            t0 = time.perf_counter()
            total = n_lift+n_hold+n_drag
            for k in range(1, total+1):
                z = lift*smootherstep(min(k/n_lift, 1.0))
                x = drag*smootherstep(min(max(k-n_lift-n_hold, 0)/n_drag, 1.0))
                w = live.advance(motions={0: {"translation": np.array([x, 0.0, z])}},
                                 traction={0: np.zeros(3, np.float32)})
                tips.append(live.markers()[-1].copy())
                forces.append(float(np.linalg.norm(np.asarray(w)[0, :3])))
            O1 = tips[n_lift+n_hold-1]
            O3 = float(tips[-1][0]-tips[n_lift+n_hold-1][0])
            out[name] = {"O1_tip_m": O1.tolist(), "O2_peak_grip_force_n": max(forces), "O3_tip_slip_m": O3,
                         "invalid": int(live.invalid_reason), "particles": int(live.scene.n),
                         "substeps": int(live.substeps), "dx_m": float(live.scene.dx), "wall_s": time.perf_counter()-t0,
                         "origin": o.tolist()}
            self.note({"RL08": name, **{k: out[name][k] for k in ("O2_peak_grip_force_n", "O3_tip_slip_m", "invalid")}})
            del live
            gc.collect()
        a, b = out["nt4_cfl0.5"], out["nt4_cfl0.25"]
        dO1 = float(np.linalg.norm(np.subtract(a["O1_tip_m"], b["O1_tip_m"])))/L
        dO3 = abs(a["O3_tip_slip_m"]-b["O3_tip_slip_m"])/abs(b["O3_tip_slip_m"])
        h = (1/8, 1/6, 1/4)                                     # dx relative (n_t 8, 6, 4), fine first
        sp = {}
        for q, f in (("O1_tip_z", lambda r: r["O1_tip_m"][2]), ("O1_tip_x", lambda r: r["O1_tip_m"][0]),
                     ("O2", lambda r: r["O2_peak_grip_force_n"]), ("O3", lambda r: r["O3_tip_slip_m"])):
            sp[q] = study(h, [f(out["nt8_cfl0.5"]), f(out["nt6_cfl0.5"]), f(out["nt4_cfl0.5"])])
        dt_ok = dO1 <= 1e-3 and dO3 <= 0.01 and not any(r["invalid"] for r in out.values())
        return {"verdict": "PASS" if dt_ok else "FAIL", "dt": {"dO1_over_L": dO1, "dO3_rel": dO3,
                                                               "limits": {"dO1_over_L": 1e-3, "dO3_rel": 0.01}},
                "spacing_reported": sp, "spacing_rule": "rubber_gci.study (RUBBER-PHYS-11 rule block; reported only)",
                "cases": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-index", type=int, default=None, help="physical nvidia-smi index (default from "
                                                                     "CUDA_VISIBLE_DEVICES)")
    args = ap.parse_args()
    rows = [r for r in args.rows.split(",") if r]
    bad = [r for r in rows if r not in ROWS]
    if bad:
        sys.exit(f"unknown rows {bad}")
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    for path, pin in ((PROTOCOL, PROTOCOL_SHA), (ROOT/"configs"/"rubber"/"rl"/"rubber_strip_drag_v1.json", TASK_SHA)):
        if hashlib.sha256(path.read_bytes()).hexdigest() != pin:
            sys.exit(f"{path} does not match its pin")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if args.gpu_index is None:
        args.gpu_index = int(vis.split(",")[0]) if vis.strip() else 0
    gpu = Gpu(args.gpu_index)
    fm = gpu.free_mb()
    if fm < MIN_FREE_MB:
        sys.exit(f"BLOCKED: GPU{args.gpu_index} has {fm:.0f} MB free < {MIN_FREE_MB} MB; not running")
    out.mkdir(parents=True)
    (out/"rows").mkdir()
    import torch
    import warp as wp
    wp.config.log_level = 30
    wp.init()
    status = sh("git", "status", "--porcelain", "--", "src", "scripts", "configs", "tests")
    from robotic_cutting_sim.rl.rubber_env import source_sha256
    meta = {"protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": PROTOCOL_SHA, "task_config_sha256": TASK_SHA,
            "commit": sh("git", "rev-parse", "HEAD"), "dirty_tree": bool(status), "dirty_paths": status.splitlines(),
            "command": " ".join([sys.executable]+sys.argv), "pid": os.getpid(), "device": args.device,
            "cuda_visible_devices": vis, "physical_gpu": args.gpu_index,
            "gpu": sh("nvidia-smi", f"--id={args.gpu_index}", "--query-gpu=name,driver_version", "--format=csv,noheader"),
            "warp": wp.config.version, "torch": torch.__version__, "python": platform.python_version(),
            "free_mb_at_start": fm, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "rows": rows,
            "sources_sha256": source_sha256(), "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
            "lock": "caller holds flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<N>.lock"}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    runner = Runner(args)
    runner.gpu = gpu
    runner.note({"meta": meta})
    for row in rows:
        since = len(gpu.samples)
        gpu.sample(f"{row}:start")
        t0 = time.perf_counter()
        try:
            res = getattr(runner, row)()
        except Exception as err:                                  # recorded, never hidden
            import traceback
            res = {"verdict": "ERROR", "error": repr(err), "traceback": traceback.format_exc()}
        gpu.sample(f"{row}:end")
        res["wall_s_row"] = time.perf_counter()-t0
        res["gpu_sharing"] = gpu.sharing(since)
        res["protocol_sha256"] = PROTOCOL_SHA
        res["commit"], res["dirty_tree"] = meta["commit"], meta["dirty_tree"]
        res["driver_gpu"] = meta["gpu"]
        (out/"rows"/f"{row}.json").write_text(json.dumps(res, indent=1, default=float))
        runner.note({"row": row, "verdict": res.get("verdict"), "wall_s": res["wall_s_row"],
                     "sharing": res["gpu_sharing"]["label"]})
        gc.collect()
    meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out/"meta.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
