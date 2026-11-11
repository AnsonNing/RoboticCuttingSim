"""DIAGNOSTIC (never a protocol row): develop the RL-05 scripted controller on the CPU on DEVELOPMENT seeds only.

A coarse CPU stand-in of ``rubber_strip_drag_v2``: the declared task (control step 20 ms, preload, randomisation,
goal, reward, evaluator, contact law v2 at mu 0.3, profile options) with the strip compiled at dx = 2 mm (n_t 2 for
t 4 mm, n_t 3 for t 6 mm; the t 5 mm slots are not on that lattice and are left out) so one 80-step episode of one env
runs in minutes on one core. The coarse grid is stiffer in bending (reduced fidelity) and its friction rows are outside
the contact envelope (the envelope check is bypassed here): numbers are for controller development only.

One process per (controller, slot, seed); the zero action is the baseline. Development seeds: 5000 ... (never the
formal RL-05 seeds 2000 + e).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src \
      .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/diag_rl05_controller.py --out artifacts/rl_ready/p2_6/<dir> \
      --controllers zero,drag_v1,tow:hybrid --seeds 5000-5011 [--jobs 24] [--steps 80]
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
DEV_SLOTS = ("L100_t4", "L125_t4", "L150_t4", "L100_t6", "L125_t6", "L150_t6")


def _family(slot, steps):
    from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily
    from robotic_cutting_sim.rubber_scene import RubberStripSpec
    fam = RubberStripTaskFamily([slot], max_episode_steps=steps)
    s = fam.slot_set[slot]
    L, t = float(s["length_m"]), float(s["thickness_m"])

    def spec(slot_, n_t=None):
        dx = 0.002
        return RubberStripSpec(length=L, width=fam.width, thickness=t, n_t=int(round(t/dx)), dx=dx, case="A",
                               jaw_length=2*dx, traction_regions=(("tip", (L-fam.tip_region, -1.0, -1.0),
                                                                    (1.0, 1.0, 1.0)),), resolution_profile=None)
    fam.spec = spec
    fam.check_material = lambda m, n_t: None          # DIAGNOSTIC: coarse n_t has no envelope row
    fam._geometry = {}
    return fam


def _controller(name, fam, device):
    from robotic_cutting_sim.rl import rubber_scripted as RS
    import torch
    if name == "zero":
        return lambda obs: torch.zeros((obs.shape[0], 6), dtype=torch.float32, device=device)
    if name == "drag_v1":
        return lambda obs: RS.scripted_drag_v1(obs, fam.max_translation)
    kind, _, rest = name.partition(":")
    kw = {}
    for item in filter(None, rest.split(",")):
        k, _, v = item.partition("=")
        kw[k] = v if k == "mode" else float(v)
    return RS.ScriptedTowV1(1, device, fam.max_translation, fam.max_rotation, **kw)


def job(item):
    controller, slot, seed, steps = item
    import torch
    torch.set_num_threads(1)
    import warp as wp
    wp.config.log_level = 30
    from robotic_cutting_sim.rl.rubber_env import RubberVectorEnv
    t0 = time.perf_counter()
    try:
        fam = _family(slot, steps)
        env = RubberVectorEnv(1, task=fam, device="cpu", parity_mode=True, strict=False)
        ctrl = _controller(controller, fam, "cpu")
        obs, _ = env.reset(seeds=[seed])
        ret, rows = 0.0, []
        for k in range(steps):
            obs, r, term, trunc, info = env.step(ctrl(obs["policy"]))
            ret += float(r[0])
            m = env.marker_x_t[0].numpy()
            rows.append({"k": k+1, "d": float(info["tip_distance_m"][0]), "status": int(info["action_status"][0]),
                         "grip": env.grip_p[0].numpy().tolist(), "tip": m[8].tolist(), "base": m[0].tolist()})
        ev = env.evaluate()[0]
        return {"controller": controller, "slot": slot, "seed": seed, "return": ret, "success": ev["success"],
                "final_d": ev["tip_distance_m"], "goal": env.goal[0].numpy().tolist(),
                "draws": {k: env.samples[0]["draws"][k] for k in ("goal_radius_m", "goal_angle_rad",
                                                                  "traction_horizontal_fraction",
                                                                  "traction_direction_rad")},
                "invalid": ev["invalid_reason"], "rows": rows, "wall_s": time.perf_counter()-t0}
    except Exception as exc:
        import traceback
        return {"controller": controller, "slot": slot, "seed": seed, "error": repr(exc),
                "traceback": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--controllers", required=True)
    ap.add_argument("--seeds", default="5000-5011")
    ap.add_argument("--slots", default=",".join(DEV_SLOTS))
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=24)
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi or lo)+1))
    if any(s < 5000 for s in seeds):
        sys.exit("development seeds are >= 5000 (the formal RL-05 seeds 2000 + e are held out)")
    slots = a.slots.split(",")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    items = [(c, slots[i % len(slots)], s, a.steps) for c in a.controllers.split(";") for i, s in enumerate(seeds)]
    (out/"meta.json").write_text(json.dumps({"label": "DIAGNOSTIC rl05 controller development (coarse CPU stand-in)",
                                             "argv": sys.argv, "items": len(items), "pid": os.getpid(),
                                             "start": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))
    res = []
    with ProcessPoolExecutor(a.jobs) as ex:
        for r in ex.map(job, items):
            res.append(r)
            print(r["controller"], r["slot"], r["seed"], r.get("error") or
                  f"ret {r['return']:.2f} final_d {r['final_d']*1e3:.1f} mm success {r['success']} "
                  f"{r['wall_s']:.0f} s", flush=True)
    (out/"results.json").write_text(json.dumps(res, default=float))


if __name__ == "__main__":
    main()
