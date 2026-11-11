"""P2-4 B2-v2: RUBBER-CONT-09 (GPU per-env wrench and state isolation of the batched contact path) under protocol
``configs/rubber/rubber_contact_tests_v2.json`` (sha256 89bc7680...a339d0, pinned below) with the v2 law
``grid_node_penalty_coulomb_v2``, and the batched RUBBER-CONT-11 cost rows of the v2 law (reported, not scored).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \\
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \\
        scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py --out artifacts/rl_ready/p2_4b/<dir>

The -09 configuration is the v1 one (``gpu_contact_batched_check.py``; workload E, case A clamp 150 x 20 x 4 mm, n_t 3,
``rubber_rl_v1``, 50 Hz, gravity -z, elastic mu0 0.4 MPa, kappa 8 MPa, rho 1150, friction_table 0.8, friction_pad 1.0,
options F8 + kick-average + F9 + F10, guard 1.0; table plane + two pad frames of one box each; N = 4, env 1 pinched and
on the table z = 0, envs 0 / 2 / 3 with pads +1 m and table z = -1 m and the same frame motions; squeeze ramped over 3
control steps then held; A: 20 um for every env, B: env 1 30 um; 6 control steps per episode) with the contact law
``grid_node_penalty_coulomb_v2`` (protocol ``common.contact_law`` and RUBBER-CONT-09.b: "the -09 v1 configuration with
the v2 law").

v2 rule (protocol RUBBER-CONT-09.criteria, thresholds k_per_side 5, lam 4, eps_rel 1e-6, read from the protocol):
  a) every contact row of envs 0 / 2 / 3 exactly 0.0 (obstacle ids -1) after every control step of every episode;
  b) k = 5 A and 5 B episodes in ONE batch / process, interleaved A, B, A, B, ... each from a full reset; for every
     control step, env e in {0, 2, 3}, observable (markers, energies summary, group / clamp wrench, plane / box / pad /
     pending wrenches) and component c: d = max_{i,j} |A_i - B_j|, s = max(range_i A_i, range_j B_j);
     PASS iff d <= lam s + eps_rel max_{i, c'} |A_i[c']| (eps_rel times the largest |component| of the observable
     vector over the A episodes);
  c) env-ownership audits 0 (nodes, cells, markers, groups, tractions, contact rows), finite, invalid bits 0.
Validity (not a criterion, the run is void without it): env 1's contact rows nonzero every step and env 1's pad wrench
differs between A and B.

Cost rows (RUBBER-CONT-11, v2 law, batched, reported): N = 4 no contact, N = 4 the -09 configuration, N = 4 every env
pinched, N = 1 pinched; columns as the v1 rows (``gpu_contact_batched_check.py``). GPU label SHARED_GPU when another
compute process is seen in the co-tenant samples before / after a row, else EXCLUSIVE_GPU (exclusive at the samples
only). Refuses < 4 GB free and a protocol whose sha256 differs from the pinned one.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v2.json"
PROTOCOL_SHA256 = "89bc7680bb9c2ed6322279b198899cef0faf31c4751b6bd849b22ecd24a339d0"
LAW = "grid_node_penalty_coulomb_v2"
MIN_FREE_MB = 4096
CONTROL_DT = 0.02
GEOM = {"L": 0.15, "W": 0.02, "T": 0.004, "N_T": 3, "PROFILE": "rubber_rl_v1", "SMOKE": False}


# ---------------------------------------------------------------------- the v2 rule (pure numpy; CPU unit-tested)
def band_v2(A, B, lam, eps_rel):
    """RUBBER-CONT-09 b v2 for one observable vector at one (step, env).

    ``A``, ``B``: [k_A, n] and [k_B, n] samples (episodes x components). Returns per component d (cross spread),
    s (larger within-side range), the floor eps_rel * max_{i, c} |A_i[c]| (one value for the vector), the limit
    lam s + floor and the per-component pass flags."""
    A = np.asarray(A, np.float64).reshape(len(A), -1)
    B = np.asarray(B, np.float64).reshape(len(B), -1)
    if A.shape[1] != B.shape[1]:
        raise ValueError("A and B have different component counts")
    if A.shape[1] == 0:
        z = np.zeros(0)
        return {"d": z, "s": z, "floor": 0.0, "limit": z, "pass": np.zeros(0, bool)}
    d = np.maximum(A.max(0)-B.min(0), B.max(0)-A.min(0))
    s = np.maximum(np.ptp(A, axis=0), np.ptp(B, axis=0))
    floor = float(eps_rel*np.abs(A).max())
    limit = lam*s+floor
    return {"d": d, "s": s, "floor": floor, "limit": limit, "pass": d <= limit}


def band_v2_all(runs_a, runs_b, envs, lam, eps_rel):
    """Apply ``band_v2`` to every (step, env in ``envs``, observable). ``runs_*``: lists of episode records with
    ``obs[k][e][name]`` arrays. Returns (pass, rows sorted worst first by d / limit, scalar count, summary)."""
    rows, n_scalars, n_fail, n_s0 = [], 0, 0, 0
    steps = len(runs_a[0]["obs"])
    for k in range(steps):
        for e in envs:
            for name in runs_a[0]["obs"][k][e]:
                A = np.stack([np.asarray(r["obs"][k][e][name], np.float64).ravel() for r in runs_a])
                B = np.stack([np.asarray(r["obs"][k][e][name], np.float64).ravel() for r in runs_b])
                res = band_v2(A, B, lam, eps_rel)
                n = res["d"].size
                if not n:
                    continue
                n_scalars += n
                fails = int((~res["pass"]).sum())
                n_fail += fails
                n_s0 += int((res["s"] == 0).sum())
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.where(res["limit"] > 0, res["d"]/np.where(res["limit"] > 0, res["limit"], 1.0),
                                     np.where(res["d"] > 0, np.inf, 0.0))
                c = int(np.argmax(ratio))
                rows.append({"step": k, "env": e, "observable": name, "components": n, "failing_components": fails,
                             "worst_component": c, "d": float(res["d"][c]), "s": float(res["s"][c]),
                             "floor": res["floor"], "limit": float(res["limit"][c]),
                             "d_over_limit": float(ratio[c]), "max_d": float(res["d"].max()),
                             "max_s": float(res["s"].max()), "pass": fails == 0})
    rows.sort(key=lambda r: -r["d_over_limit"])
    summary = {"scalars": n_scalars, "scalars_failing": n_fail, "scalars_with_s_zero": n_s0,
               "rows": len(rows), "rows_failing": sum(not r["pass"] for r in rows),
               "max_d_over_limit": rows[0]["d_over_limit"] if rows else 0.0}
    return n_fail == 0, rows, n_scalars, summary


# ---------------------------------------------------------------------- scene (the v1 -09 configuration, v2 law)
def pads(shift):
    from robotic_cutting_sim.rubber_contact import ContactFrame
    L, W, T = GEOM["L"], GEOM["W"], GEOM["T"]
    top = ContactFrame(kind="pad", name="top", origin=(L-0.005+shift, W/2, T+0.001),
                       boxes=(((L-0.005+shift, W/2, T+0.001), (0.006, W/2+0.002, 0.001)),))
    bot = ContactFrame(kind="pad", name="bottom", origin=(L-0.005+shift, W/2, -0.001),
                       boxes=(((L-0.005+shift, W/2, -0.001), (0.006, W/2+0.002, 0.001)),))
    return (top, bot)


def make_lives(reach, contact=True, law=LAW):
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_contact import ContactSpec
    from robotic_cutting_sim.rubber_material import MaterialValues
    m = MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=0.8, friction_pad=1.0)
    spec = rs.RubberStripSpec(length=GEOM["L"], width=GEOM["W"], thickness=GEOM["T"], n_t=GEOM["N_T"], case="A",
                              resolution_profile=GEOM["PROFILE"])
    opts = dict(dirichlet="consistent_mass_projection_v1", constraint_work="kick_average_v1",
                traction_work="grid_displacement_v1", state_precision="f64_shadow_v1", stable_step_guard=1.0)
    lives = []
    for r in reach:
        c = None
        if contact:
            c = (ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), frames=pads(0.0), law=law) if r else
                 ContactSpec(planes=((0.0, 0.0, 1.0, 1.0),), frames=pads(1.0), law=law))
        lives.append(LiveRubberMaterial(spec, material=m, control_dt=CONTROL_DT, device="cpu", gravity=(0, 0, -9.81),
                                        strict=False, contact=c, **opts))
    return lives


def frame_inputs(k, N, depth):
    tr = np.zeros((N, 2, 3))
    r = min(1.0, (k+1)/3)
    for e in range(N):
        tr[e, 0] = [0.0, 0.0, -depth[e]*r]
        tr[e, 1] = [0.0, 0.0, depth[e]*r]
    return tr


def contact_rows(batch, e):
    s, c = batch.solver, batch.contact
    snap = s.snapshot(e)
    out = {name: snap[name] for name in ("ct_work", "ct_pen_max", "ct_counts", "ct_energy", "ct_plane_reaction",
                                         "ct_box_reaction", "ct_jn", "ct_jt", "ct_xi", "ct_vobs")}
    for name in ("plane_wrench", "box_wrench", "pad_wrench", "pending_wrench"):
        out[name] = getattr(c, name).numpy()[e]
    out["ct_obst_plus_1"] = snap["ct_obst"]+1
    return out


def observables(batch, e):
    """The v1 observables named by the v2 protocol: markers, energies summary, group (clamp) wrench, plane / box / pad /
    pending wrenches (the v1 script did not include the pending wrench; it is zero by construction in rubber v1)."""
    out = {"markers": batch.markers()[e], "summary": batch.summary.numpy()[e], "wrench": batch.wrench.numpy()[e]}
    for name in ("plane_wrench", "box_wrench", "pad_wrench", "pending_wrench"):
        out[name] = getattr(batch.contact, name).numpy()[e]
    return out


def episode(batch, depth, steps, sync):
    import warp as wp
    batch.reset()
    N = batch.num_envs
    rec = {"obs": [], "zero_rows": [], "walls": [], "invalid": [], "finite": [], "pad_force_z_env": []}
    for k in range(steps):
        tr = frame_inputs(k, N, depth)
        sync()
        t = time.perf_counter()
        batch.advance(frame_translation=tr)
        sync()
        rec["walls"].append(time.perf_counter()-t)
        rec["obs"].append([observables(batch, e) for e in range(N)])
        zr = {}
        for e in range(N):
            rows = contact_rows(batch, e)
            nz = {name: int(np.count_nonzero(v)) for name, v in rows.items()}
            zr[e] = {"nonzero_entries": int(sum(nz.values())), "by_array": {n: v for n, v in nz.items() if v}}
        rec["zero_rows"].append(zr)
        rec["invalid"].append(batch.invalid_reason.numpy().tolist())
        diag = batch.diagnostics_read()
        rec["finite"].append([bool(d["nonfinite"] == 0.0) for d in diag])
        rec["pad_force_z_env"].append(batch.contact.box_wrench.numpy()[:, :, 0, 2].tolist())
        wp.synchronize()
    return rec


def gpu_label(samples, pid):
    others = sorted({a.split(",")[0].strip() for s in samples for a in s["apps"]
                     if a.strip() and int(a.split(",")[0]) != pid})
    return ("SHARED_GPU" if others else "EXCLUSIVE_GPU"), others


def cost_row(label, reach, contact, steps, device, peak, meta_common):
    import warp as wp
    from robotic_cutting_sim.batched import rubber_contact_batch as rcb
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    t0 = time.perf_counter()
    lives = make_lives(reach, contact)
    host_build = time.perf_counter()-t0
    t1 = time.perf_counter()
    batch = BatchedRubberMaterial(lives, device=device, strict=False)
    del lives
    wp.synchronize_device(device)
    gpu_build = time.perf_counter()-t1
    peak.update()
    N = batch.num_envs
    depth = [2e-5]*N
    walls = []
    for k in range(steps):
        tr = frame_inputs(k, N, depth) if contact else None
        wp.synchronize_device(device)
        t = time.perf_counter()
        batch.advance(frame_translation=tr) if contact else batch.advance()
        wp.synchronize_device(device)
        walls.append(time.perf_counter()-t)
        peak.update()
    resets = []
    for r in range(20):
        wp.synchronize_device(device)
        t = time.perf_counter()
        batch.reset([r % N])
        wp.synchronize_device(device)
        resets.append(time.perf_counter()-t)
    diag = batch.diagnostics_read()
    invalid = batch.invalid_reason.numpy().tolist()
    audits = batch.audits()
    law = None
    if contact:
        audits.update(rcb.contact_audit(batch.solver))
        law = batch.solver.ct_law
    finite = all(d["nonfinite"] == 0.0 for d in diag)
    ok = finite and not any(invalid) and all(v == 0 for v in audits.values())
    warm = float(np.median(walls[1:])) if len(walls) > 1 else walls[0]
    s = batch.solver
    row = {"row": label, "workload": (f"E (rubber_material_v1, case A {GEOM['L']*1e3:g} x {GEOM['W']*1e3:g} x "
                                      f"{GEOM['T']*1e3:g} mm, n_t {GEOM['N_T']}, {GEOM['PROFILE']})"),
           "boundary": "batched_headless (BatchedRubberMaterial, material only)", "N": N,
           "particles_per_env": int(batch.packed.particle_counts[0]), "nodes_per_env": int(batch.packed.node_counts[0]),
           "control_hz": 1.0/CONTROL_DT, "material_dt_s": s.dt, "substeps_per_control_step": s.substeps,
           "cameras": "none", "recording": "none", "robot_dof": 0, "contact_law": law,
           "contacts": ("table plane + 2 pad frames x 1 box per env, envs in reach: "
                        f"{[i for i, r in enumerate(reach) if r]}" if contact else "none (no ContactSpec)"),
           "graph": batch.capture_receipt, "host_build_s": host_build, "gpu_build_capture_s": gpu_build,
           "control_step_wall_s": walls, "cold_wall_s": walls[0], "warm_wall_s_median": warm,
           "warm_steps": len(walls)-1, "ms_per_substep_warm": 1e3*warm/s.substeps, "env_steps_per_s_warm": N/warm,
           "aggregate_sim_time_per_wall_warm": N*CONTROL_DT/warm,
           "reset_p95_ms": 1e3*float(np.percentile(resets, 95)), "reset_median_ms": 1e3*float(np.median(resets)),
           "reset_kind": "masked partial reset of one env (no material resampling), 20 resets",
           "vram_peak_mb_process_cumulative": peak.value, "memory_bytes_arrays": batch.memory_bytes(),
           "finite": bool(finite), "invalid_reason": invalid, "audits": audits,
           "physics_verdict": "L2_INVARIANTS_PASS" if ok else "L2_INVARIANTS_FAIL", **meta_common}
    del batch
    gc.collect()
    wp.synchronize_device(device)
    return row


def free_mb(index=0):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=6, help="control steps per -09 episode (v1 configuration: 6)")
    ap.add_argument("--cost-steps", type=int, default=4, help="control steps per cost row (1 cold + warm)")
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE (reduced_fidelity): 30 x 10 x 4 mm strip, n_t 2, script check only; never the formal row")
    args = ap.parse_args()
    if args.smoke:
        GEOM.update(L=0.03, W=0.01, T=0.004, N_T=2, SMOKE=True)
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    proto_sha = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    if proto_sha != PROTOCOL_SHA256:
        sys.exit(f"protocol sha256 {proto_sha} != pinned {PROTOCOL_SHA256}; not running")
    proto = json.loads(PROTOCOL.read_text())
    th = proto["tests"]["RUBBER-CONT-09"]["thresholds"]
    K, LAM, EPS = int(th["k_per_side"]), float(th["lam"]), float(th["eps_rel"])
    if proto["common"]["contact_law"] != LAW:
        sys.exit(f"protocol contact law {proto['common']['contact_law']} != {LAW}")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = int(vis.split(",")[0]) if vis.strip() else 0
    fm = free_mb(phys)
    if fm < MIN_FREE_MB:
        sys.exit(f"GPU{phys}: {fm:.0f} MB free < {MIN_FREE_MB} MB; not running")
    out.mkdir(parents=True)
    sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
    sys.path.insert(0, str(ROOT/"src"))
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    from gpu_smoke_rubber import Peak, gpu_sample
    from robotic_cutting_sim.batched import rubber_contact_batch as rcb
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    wp.init()
    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa
    t_start = time.perf_counter()
    meta = {"label": "RUBBER-CONT-09 v2 rule (GPU per-env isolation, batched contact, law v2) + RUBBER-CONT-11 "
                     "batched cost rows of the v2 law (reported, not scored)",
            "command": " ".join(sys.argv), "cuda_visible_devices": vis, "device": args.device,
            "device_name": wp.get_device(args.device).name, "physical_gpu_index": phys,
            "driver": subprocess.run(["nvidia-smi", f"--id={phys}", "--query-gpu=driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
            "pid": os.getpid(), "commit": git("rev-parse", "HEAD"),
            "dirty_tree": bool(git("status", "--porcelain", "--untracked-files=no")),
            "protocol": str(PROTOCOL.relative_to(ROOT)), "protocol_sha256": proto_sha, "contact_law": LAW,
            "thresholds": {"k_per_side": K, "lam": LAM, "eps_rel": EPS}, "warp": wp.config.version,
            "python": sys.version.split()[0], "free_mb_at_start": fm, "co_tenant_before": gpu_sample(),
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED",
            "smoke_reduced_fidelity": GEOM["SMOKE"], "geometry_m": [GEOM["L"], GEOM["W"], GEOM["T"]],
            "n_t": GEOM["N_T"]}
    peak = Peak()
    sync = lambda: wp.synchronize_device(args.device)  # noqa: E731

    # ------------------------------------------------------------------ RUBBER-CONT-09 (v2 rule)
    reach = [False, True, False, False]
    t0 = time.perf_counter()
    lives = make_lives(reach)
    batch = BatchedRubberMaterial(lives, device=args.device, strict=False)
    del lives
    build = time.perf_counter()-t0
    peak.update()
    N = batch.num_envs
    law_built = batch.solver.ct_law
    if law_built != LAW:
        sys.exit(f"batch built law {law_built} != {LAW}")
    depth_A = [2e-5, 2e-5, 2e-5, 2e-5]
    depth_B = [2e-5, 3e-5, 2e-5, 2e-5]
    order, runs_a, runs_b = [], [], []
    for i in range(K):
        for side, depth, lst in (("A", depth_A, runs_a), ("B", depth_B, runs_b)):
            rec = episode(batch, depth, args.steps, sync)
            lst.append(rec)
            order.append(f"{side}{i+1}")
            peak.update()
            print(json.dumps({"episode": f"{side}{i+1}", "walls": rec["walls"],
                              "pad_force_z_env1": [p[1] for p in rec["pad_force_z_env"]]}), flush=True)
    every = runs_a+runs_b
    out_envs = [e for e, r in enumerate(reach) if not r]
    zero_ok = all(r["zero_rows"][k][e]["nonzero_entries"] == 0 for r in every
                  for k in range(args.steps) for e in out_envs)
    zero_fail = [{"episode": order[(2*i if i < K else 2*(i-K)+1)], "step": k, "env": e,
                  "by_array": r["zero_rows"][k][e]["by_array"]}
                 for i, r in enumerate(every) for k in range(args.steps) for e in out_envs
                 if r["zero_rows"][k][e]["nonzero_entries"]][:30]
    reached = all(r["zero_rows"][k][1]["nonzero_entries"] > 0 for r in every for k in range(args.steps))
    band_ok, band_rows, n_scalars, band_summary = band_v2_all(runs_a, runs_b, out_envs, LAM, EPS)
    env1_differs = all(any(float(np.abs(np.asarray(a["obs"][k][1]["pad_wrench"]) -
                                        np.asarray(b["obs"][k][1]["pad_wrench"])).max()) > 0
                           for k in range(args.steps)) for a in runs_a for b in runs_b)
    audits = batch.audits()
    audits.update(rcb.contact_audit(batch.solver))
    finite = all(all(f) for r in every for f in r["finite"])
    invalid_zero = all(not any(v) for r in every for v in r["invalid"])
    guard = batch.solver.cfl_violations.numpy().tolist()
    c_ok = all(v == 0 for v in audits.values()) and finite and invalid_zero
    # DIAGNOSTIC (never part of the verdict): run-to-run determinism of A and of B per observable, and the env-1
    # (perturbed env) A-vs-B shift seen by the same rule (it must reject there, showing the rule has power)
    bitwise_A = {name: all(np.array_equal(np.asarray(runs_a[0]["obs"][k][e][name]), np.asarray(r["obs"][k][e][name]))
                           for r in runs_a[1:] for k in range(args.steps) for e in range(N))
                 for name in runs_a[0]["obs"][0][0]}
    _, env1_rows, _, env1_summary = band_v2_all(runs_a, runs_b, [1], LAM, EPS)
    cont09 = {"N": N, "particles_per_env": int(batch.packed.particle_counts[0]),
              "nodes_per_env": int(batch.packed.node_counts[0]), "substeps": batch.solver.substeps,
              "dt_s": batch.solver.dt, "control_steps_per_episode": args.steps, "build_s": build,
              "graph": batch.capture_receipt, "contact_law_built": law_built,
              "options_receipt": batch.solver.options_receipt(), "reach": reach,
              "squeeze_depth_A_m": depth_A, "squeeze_depth_B_m": depth_B, "episode_order": order,
              "k_per_side": K, "lam": LAM, "eps_rel": EPS,
              "a_zero_rows": {"pass": bool(zero_ok), "failures_first_30": zero_fail},
              "b_perturbation_band": {"pass": bool(band_ok), **band_summary,
                                      "declared_scalar_bound_for_fwer": 648,
                                      "rows_worst_first": band_rows[:30]},
              "c_audits": {"pass": bool(c_ok), "audits": audits, "finite_every_step": bool(finite),
                           "invalid_bits_zero_every_step": bool(invalid_zero), "guard_violations_after": guard},
              "validity": {"env1_contact_rows_nonzero_every_step": bool(reached),
                           "env1_pad_wrench_differs_every_A_B_pair": bool(env1_differs)},
              "diagnostic_not_verdict": {
                  "A_episodes_bitwise_identical_per_observable_all_envs": bitwise_A,
                  "env1_same_rule_summary": env1_summary, "env1_same_rule_rows_worst_first": env1_rows[:6]},
              "pad_force_z": {"A": [r["pad_force_z_env"] for r in runs_a], "B": [r["pad_force_z_env"] for r in runs_b]},
              "walls_s": {o: r["walls"] for o, r in zip(order, [x for p in zip(runs_a, runs_b) for x in p])}}
    valid = reached and env1_differs
    cont09["verdict"] = ("PASS" if (zero_ok and band_ok and c_ok) else "FAIL") if valid else "FAIL_INVALID_RUN"
    print(json.dumps({"RUBBER-CONT-09": cont09["verdict"], "a": zero_ok, "b": band_ok, "c": c_ok,
                      "env1_reached": reached, "env1_differs": env1_differs, "b_summary": band_summary,
                      "audits": audits}, default=float), flush=True)
    del batch
    gc.collect()
    sync()
    (out/"cont09_partial.json").write_text(json.dumps({"meta": meta, "cont09": cont09}, default=float))

    # ------------------------------------------------------------------ RUBBER-CONT-11 batched cost rows (v2 law)
    common = {"smoke_reduced_fidelity": GEOM["SMOKE"], "gpu": meta["device_name"], "gpu_index_physical": phys,
              "driver": meta["driver"], "commit": meta["commit"], "dirty_tree": meta["dirty_tree"],
              "protocol_sha256": meta["protocol_sha256"], "warp": meta["warp"]}
    rows = []
    for label, r, contact in (("N4_no_contact", [False]*4, False), ("N4_contact_cont09_config", reach, True),
                              ("N4_contact_all_pinched", [True]*4, True), ("N1_contact_pinched", [True], True)):
        before = gpu_sample()
        row = cost_row(label, r, contact, args.cost_steps, args.device, peak, common)
        row["co_tenant_before_row"] = before
        row["co_tenant_after_row"] = gpu_sample()
        row["gpu_label"], row["co_tenant_pids"] = gpu_label([before, row["co_tenant_after_row"]], os.getpid())
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("row", "N", "contact_law", "ms_per_substep_warm", "env_steps_per_s_warm",
                                              "aggregate_sim_time_per_wall_warm", "reset_p95_ms",
                                              "vram_peak_mb_process_cumulative", "physics_verdict", "gpu_label")},
                         default=float), flush=True)
    meta["co_tenant_after"] = gpu_sample()
    meta["label_gpu"], meta["co_tenant_pids"] = gpu_label([meta["co_tenant_before"], meta["co_tenant_after"]]
                                                          + [x for r in rows for x in (r["co_tenant_before_row"],
                                                                                       r["co_tenant_after_row"])],
                                                          os.getpid())
    meta["shared_gpu"] = meta["label_gpu"] == "SHARED_GPU"
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    meta["wall_s"] = time.perf_counter()-t_start
    (out/"result.json").write_text(json.dumps({"meta": meta, "cont09": cont09, "cost_rows": rows}, indent=1,
                                              default=float))
    print(json.dumps({"RUBBER-CONT-09": cont09["verdict"], "label_gpu": meta["label_gpu"]}))


if __name__ == "__main__":
    main()
