"""P2-4 B2: RUBBER-CONT-09 (GPU per-env wrench and state isolation of the batched contact path) and the batched rows
of RUBBER-CONT-11 (cost, reported, not scored). Protocol ``configs/rubber/rubber_contact_tests_v1.json``.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \\
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4b/gpu_contact_batched_check.py \\
        --out artifacts/rl_ready/p2_4b/<dir>

Workload E geometry (as ``scripts/rubber/p2_4/gpu_contact_check.py``): case A clamp, 150 x 20 x 4 mm, n_t 3
(``rubber_rl_v1``), 50 Hz, gravity -z, elastic mu0 0.4 MPa, beta 20, rho 1150, friction_table 0.8, friction_pad 1.0,
the protocol's options F8 + kick-average + F9 + F10, guard 1.0; contact law grid_node_penalty_coulomb_v1 defaults
(table plane + two pad frames of one box each). BatchedRubberMaterial on cuda:0, CUDA graph chunks of 500 substeps.

RUBBER-CONT-09 (N = 4): env 1 pinched by its pads (squeeze ramped over 3 control steps, then held) and resting on the
table z = 0; envs 0 / 2 / 3 with the same pads and table translated out of reach (pads +1 m in x, table z = -1 m) and
the same frame motions. Runs on ONE batch: A, then a full reset and A' (the same inputs: repeat band), then a full
reset and B (only env 1's squeeze depth differs).
  a) every contact row of envs 0 / 2 / 3 (plane / box / pad / pending wrenches, ct_work, ct_pen_max, ct_counts,
     ct_energy, the plane / box reactions, and the per-node impulses, slider state and node obstacle ids) is exactly
     0.0 (obstacle ids -1) after every control step of A, A' and B;
  b) for envs 0 / 2 / 3 every observable (markers, diagnostics summary = energies, group wrench, contact wrenches)
     satisfies max |A - B| <= max |A - A'| + 1e-6 max(|A|) after every control step;
  c) env-ownership audits 0 (``audits()``: nodes, cells, markers, groups, tractions; ``contact_audit``: contact rows),
     finite, invalid bits 0.
Cost rows (RUBBER-CONT-11 batched; every row: N, particles/env, control Hz, material dt, substeps, cameras none,
recording none, robot DOF 0, contacts, env-steps/s, aggregate sim-time / wall, VRAM peak (process, cumulative), reset
p95, physics verdict, warm / cold): N = 4 no contact, N = 4 the -09 configuration, N = 4 every env pinched, N = 1
contact. Co-tenant samples before / after; SHARED_GPU whenever another process is present. Refuses < 4 GB free.
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
sys.path.insert(0, str(ROOT/"scripts"/"rubber"))
sys.path.insert(0, str(ROOT/"src"))
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v1.json"
MIN_FREE_MB = 4096
L, W, T = 0.15, 0.02, 0.004
N_T, PROFILE = 3, "rubber_rl_v1"
CONTROL_DT = 0.02
SMOKE = False


def free_mb(index=0):
    out = subprocess.run(["nvidia-smi", f"--id={index}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip()
    return float(out.splitlines()[0])


def pads(shift):
    from robotic_cutting_sim.rubber_contact import ContactFrame
    top = ContactFrame(kind="pad", name="top", origin=(L-0.005+shift, W/2, T+0.001),
                       boxes=(((L-0.005+shift, W/2, T+0.001), (0.006, W/2+0.002, 0.001)),))
    bot = ContactFrame(kind="pad", name="bottom", origin=(L-0.005+shift, W/2, -0.001),
                       boxes=(((L-0.005+shift, W/2, -0.001), (0.006, W/2+0.002, 0.001)),))
    return (top, bot)


def make_lives(reach, contact=True):
    """``reach``: per env, True = pads + table in reach (table z = 0), False = out of reach (+1 m, z = -1 m)."""
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_contact import ContactSpec
    from robotic_cutting_sim.rubber_material import MaterialValues
    m = MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=0.8, friction_pad=1.0)
    spec = rs.RubberStripSpec(length=L, width=W, thickness=T, n_t=N_T, case="A", resolution_profile=PROFILE)
    opts = dict(dirichlet="consistent_mass_projection_v1", constraint_work="kick_average_v1",
                traction_work="grid_displacement_v1", state_precision="f64_shadow_v1", stable_step_guard=1.0)
    lives = []
    for r in reach:
        c = None
        if contact:
            c = (ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), frames=pads(0.0)) if r else
                 ContactSpec(planes=((0.0, 0.0, 1.0, 1.0),), frames=pads(1.0)))
        lives.append(LiveRubberMaterial(spec, material=m, control_dt=CONTROL_DT, device="cpu", gravity=(0, 0, -9.81),
                                        strict=False, contact=c, **opts))
    return lives


def frame_inputs(k, N, depth):
    """[N, 2, 3] END translations at control step k: squeeze ramped over 3 steps then held; ``depth`` per env."""
    tr = np.zeros((N, 2, 3))
    r = min(1.0, (k+1)/3)
    for e in range(N):
        tr[e, 0] = [0.0, 0.0, -depth[e]*r]
        tr[e, 1] = [0.0, 0.0, depth[e]*r]
    return tr


def contact_rows(batch, e):
    """Every contact row of env e (host), as one dict of arrays."""
    s, c = batch.solver, batch.contact
    snap = s.snapshot(e)
    out = {name: snap[name] for name in ("ct_work", "ct_pen_max", "ct_counts", "ct_energy", "ct_plane_reaction",
                                         "ct_box_reaction", "ct_jn", "ct_jt", "ct_xi", "ct_vobs")}
    for name in ("plane_wrench", "box_wrench", "pad_wrench", "pending_wrench"):
        out[name] = getattr(c, name).numpy()[e]
    out["ct_obst_plus_1"] = snap["ct_obst"]+1
    return out


def observables(batch, e):
    out = {"markers": batch.markers()[e], "summary": batch.summary.numpy()[e], "wrench": batch.wrench.numpy()[e]}
    for name in ("plane_wrench", "box_wrench", "pad_wrench"):
        out[name] = getattr(batch.contact, name).numpy()[e]
    return out


def episode(batch, depth, steps, sync):
    """One episode from a full reset: per step the -09 a zero-row check of the out-of-reach envs, the observables
    of every env, invalid bits, finiteness."""
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


def band_check(A, A2, B, envs, rel=1e-6):
    """-09 b: for every step, env in ``envs`` and observable: max|A-B| <= max|A-A'| + rel max|A|."""
    worst = []
    ok = True
    for k, (oa, oa2, ob) in enumerate(zip(A["obs"], A2["obs"], B["obs"])):
        for e in envs:
            for name in oa[e]:
                a, a2, b = (np.asarray(x[e][name], np.float64) for x in (oa, oa2, ob))
                d_ab = float(np.abs(a-b).max()) if a.size else 0.0
                d_rep = float(np.abs(a-a2).max()) if a.size else 0.0
                scale = float(np.abs(a).max()) if a.size else 0.0
                lim = d_rep+rel*scale
                passed = d_ab <= lim
                ok &= passed
                worst.append({"step": k, "env": e, "observable": name, "max_abs_A_minus_B": d_ab,
                              "max_abs_A_minus_Aprime": d_rep, "scale_max_abs_A": scale, "limit": lim,
                              "pass": bool(passed)})
    worst.sort(key=lambda r: -(r["max_abs_A_minus_B"]/r["limit"] if r["limit"] > 0 else
                               (np.inf if r["max_abs_A_minus_B"] > 0 else 0.0)))
    return ok, worst


def repeat_envelope(runs_a, runs_b, envs, rel=1e-6):
    """DIAGNOSTIC (not the -09 b verdict; added after the SMOKE run gpu_cont09_smoke_001 showed GPU float-atomic
    nondeterminism of the out-of-reach envs' clamp wrench of the same size as the single A vs A' sample): with several
    repeats of A and of B, compare max over the A-A pairs with every A-B pair, per (step, env, observable); and the
    per-observable reading aggregated over all steps with the formal A / A' / B episodes."""
    import itertools
    rows, inside = [], 0
    steps = len(runs_a[0]["obs"])
    for k in range(steps):
        for e in envs:
            for name in runs_a[0]["obs"][k][e]:
                A = [np.asarray(r["obs"][k][e][name], np.float64) for r in runs_a]
                Bv = [np.asarray(r["obs"][k][e][name], np.float64) for r in runs_b]
                if not A[0].size:
                    continue
                aa = max(float(np.abs(x-y).max()) for x, y in itertools.combinations(A, 2))
                ab = [float(np.abs(x-y).max()) for x in A for y in Bv]
                lim = aa+rel*float(np.abs(A[0]).max())
                ok = max(ab) <= lim
                inside += ok
                rows.append({"step": k, "env": e, "observable": name, "max_AA": aa, "min_AB": min(ab),
                             "max_AB": max(ab), "limit": lim, "max_AB_within_AA_envelope": bool(ok)})
    return {"repeats_A": len(runs_a), "repeats_B": len(runs_b), "rows": len(rows), "rows_within": int(inside),
            "rows_outside": [r for r in rows if not r["max_AB_within_AA_envelope"]][:30]}


def aggregated_band(A, A2, B, envs, rel=1e-6):
    """DIAGNOSTIC alternative reading of -09 b: per (env, observable), max over all steps."""
    out = []
    for e in envs:
        for name in A["obs"][0][e]:
            d_ab = max(float(np.abs(np.asarray(a[e][name], float)-np.asarray(b[e][name], float)).max())
                       for a, b in zip(A["obs"], B["obs"]) if np.asarray(a[e][name]).size) \
                if np.asarray(A["obs"][0][e][name]).size else 0.0
            d_rep = max(float(np.abs(np.asarray(a[e][name], float)-np.asarray(b[e][name], float)).max())
                        for a, b in zip(A["obs"], A2["obs"]) if np.asarray(a[e][name]).size) \
                if np.asarray(A["obs"][0][e][name]).size else 0.0
            scale = max(float(np.abs(np.asarray(a[e][name], float)).max()) for a in A["obs"]) \
                if np.asarray(A["obs"][0][e][name]).size else 0.0
            out.append({"env": e, "observable": name, "max_AB": d_ab, "max_AAprime": d_rep,
                        "limit": d_rep+rel*scale, "pass": bool(d_ab <= d_rep+rel*scale)})
    return {"pass_all": all(r["pass"] for r in out), "rows": out}


def cost_row(label, reach, contact, steps, device, peak, meta_common, depth_value=2e-5):
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
    depth = [depth_value]*N
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
    if contact:
        audits.update(rcb.contact_audit(batch.solver))
    finite = all(d["nonfinite"] == 0.0 for d in diag)
    ok = finite and not any(invalid) and all(v == 0 for v in audits.values())
    warm = float(np.median(walls[1:])) if len(walls) > 1 else walls[0]
    s = batch.solver
    row = {"row": label, "workload": "E (rubber_material_v1, case A 150 x 20 x 4 mm, n_t 3, rubber_rl_v1)",
           "boundary": "batched_headless (BatchedRubberMaterial, material only)", "N": N,
           "particles_per_env": int(batch.packed.particle_counts[0]), "nodes_per_env": int(batch.packed.node_counts[0]),
           "control_hz": 1.0/CONTROL_DT, "material_dt_s": s.dt, "substeps_per_control_step": s.substeps,
           "cameras": "none", "recording": "none", "robot_dof": 0,
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=6, help="control steps per -09 episode")
    ap.add_argument("--cost-steps", type=int, default=4, help="control steps per cost row (1 cold + warm)")
    ap.add_argument("--smoke", action="store_true",
                    help="SMOKE (reduced_fidelity): 30 x 10 x 4 mm strip, n_t 2, script check only; never the formal row")
    args = ap.parse_args()
    if args.smoke:
        global L, W, T, N_T, SMOKE
        L, W, T, N_T, SMOKE = 0.03, 0.01, 0.004, 2, True
    out = Path(args.out)
    if out.exists():
        sys.exit(f"refusing to overwrite {out}")
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    phys = int(vis.split(",")[0]) if vis.strip() else 0
    fm = free_mb(phys)
    if fm < MIN_FREE_MB:
        sys.exit(f"GPU{phys}: {fm:.0f} MB free < {MIN_FREE_MB} MB; not running")
    out.mkdir(parents=True)
    import warp as wp
    wp.config.log_level = wp.LOG_WARNING
    from gpu_smoke_rubber import Peak, gpu_sample
    from robotic_cutting_sim.batched import rubber_contact_batch as rcb
    from robotic_cutting_sim.batched.rubber_material_batch import BatchedRubberMaterial
    wp.init()
    git = lambda *c: subprocess.run(["git", *c], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa
    meta = {"label": "RUBBER-CONT-09 (GPU per-env isolation, batched contact) + RUBBER-CONT-11 batched cost rows "
                     "(reported, not scored)",
            "command": " ".join(sys.argv), "cuda_visible_devices": vis, "device": args.device,
            "device_name": wp.get_device(args.device).name, "physical_gpu_index": phys,
            "driver": subprocess.run(["nvidia-smi", f"--id={phys}", "--query-gpu=driver_version",
                                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
            "pid": os.getpid(), "commit": git("rev-parse", "HEAD"),
            "dirty_tree": bool(git("status", "--porcelain", "--untracked-files=no")),
            "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(), "warp": wp.config.version,
            "free_mb_at_start": fm, "co_tenant_before": gpu_sample(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "smoke_reduced_fidelity": SMOKE,
            "geometry_m": [L, W, T], "n_t": N_T}
    peak = Peak()
    sync = lambda: wp.synchronize_device(args.device)  # noqa: E731

    # ------------------------------------------------------------------ RUBBER-CONT-09
    reach = [False, True, False, False]
    t0 = time.perf_counter()
    lives = make_lives(reach)
    batch = BatchedRubberMaterial(lives, device=args.device, strict=False)
    del lives
    build = time.perf_counter()-t0
    peak.update()
    N = batch.num_envs
    depth_A = [2e-5, 2e-5, 2e-5, 2e-5]
    depth_B = [2e-5, 3e-5, 2e-5, 2e-5]
    runs = {}
    for name, depth in (("A", depth_A), ("A_repeat", depth_A), ("B", depth_B)):
        runs[name] = episode(batch, depth, args.steps, sync)
        peak.update()
        print(json.dumps({"episode": name, "walls": runs[name]["walls"],
                          "pad_force_z_env1": [p[1] for p in runs[name]["pad_force_z_env"]]}), flush=True)
    out_envs = [e for e, r in enumerate(reach) if not r]
    zero_ok = all(r["zero_rows"][k][e]["nonzero_entries"] == 0 for r in runs.values()
                  for k in range(args.steps) for e in out_envs)
    reached = all(r["zero_rows"][k][1]["nonzero_entries"] > 0 for r in runs.values() for k in range(args.steps))
    band_ok, band = band_check(runs["A"], runs["A_repeat"], runs["B"], out_envs)
    env1_differs = any(float(np.abs(np.asarray(a[1]["pad_wrench"])-np.asarray(b[1]["pad_wrench"])).max()) > 0
                       for a, b in zip(runs["A"]["obs"], runs["B"]["obs"]))
    audits = batch.audits()
    audits.update(rcb.contact_audit(batch.solver))
    finite = all(all(f) for r in runs.values() for f in r["finite"])
    invalid_zero = all(not any(v) for r in runs.values() for v in r["invalid"])
    guard = batch.solver.cfl_violations.numpy().tolist()
    c_ok = all(v == 0 for v in audits.values()) and finite and invalid_zero
    # DIAGNOSTIC repeats (after the formal episodes; never part of the verdict)
    for name, depth in (("A_repeat2", depth_A), ("B_repeat", depth_B)):
        runs[name] = episode(batch, depth, args.steps, sync)
        print(json.dumps({"diagnostic_episode": name, "walls": runs[name]["walls"]}), flush=True)
    envelope = repeat_envelope([runs["A"], runs["A_repeat"], runs["A_repeat2"]], [runs["B"], runs["B_repeat"]],
                               out_envs)
    aggregated = aggregated_band(runs["A"], runs["A_repeat"], runs["B"], out_envs)
    zero_ok_diag = all(r["zero_rows"][k][e]["nonzero_entries"] == 0 for n, r in runs.items()
                       for k in range(args.steps) for e in out_envs)
    repeat_identical = all(all(np.array_equal(np.asarray(a[e][n]), np.asarray(b[e][n])) for e in range(N)
                               for n in a[e]) for a, b in zip(runs["A"]["obs"], runs["A_repeat"]["obs"]))
    cont09 = {"N": N, "particles_per_env": int(batch.packed.particle_counts[0]), "substeps": batch.solver.substeps,
              "dt_s": batch.solver.dt, "control_steps_per_episode": args.steps, "build_s": build,
              "graph": batch.capture_receipt, "reach": reach, "squeeze_depth_A_m": depth_A, "squeeze_depth_B_m": depth_B,
              "a_zero_rows": {"pass": bool(zero_ok), "env1_contact_rows_nonzero_every_step": bool(reached),
                              "per_episode_step_env": {n: r["zero_rows"] for n, r in runs.items()}},
              "b_perturbation_band": {"pass": bool(band_ok), "env1_differs_A_vs_B": bool(env1_differs),
                                      "repeat_A_vs_Aprime_bitwise_identical_all_envs": bool(repeat_identical),
                                      "rows_worst_first": band[:30], "rows_total": len(band),
                                      "rows_failing": sum(1 for r in band if not r["pass"])},
              "c_audits": {"pass": bool(c_ok), "audits": audits, "finite_every_step": bool(finite),
                           "invalid_bits_zero_every_step": bool(invalid_zero), "guard_violations_after": guard},
              "diagnostic_not_verdict": {
                  "note": "added after the SMOKE run (reduced geometry) showed GPU float-atomic nondeterminism of the "
                          "out-of-reach clamp wrench comparable to the single A vs A' sample; the -09 b verdict above "
                          "is the declared per-step reading implemented before any run",
                  "repeat_envelope_3A_2B": envelope, "aggregated_over_steps": aggregated,
                  "zero_rows_including_diagnostic_episodes": bool(zero_ok_diag)},
              "pad_force_z": {n: r["pad_force_z_env"] for n, r in runs.items()},
              "walls_s": {n: r["walls"] for n, r in runs.items()}}
    cont09["verdict"] = "PASS" if (zero_ok and band_ok and c_ok and reached and env1_differs) else "FAIL"
    print(json.dumps({"RUBBER-CONT-09": cont09["verdict"], "a": zero_ok, "b": band_ok, "c": c_ok,
                      "env1_reached": reached, "env1_differs": env1_differs, "repeat_identical": repeat_identical,
                      "audits": audits}), flush=True)
    del batch
    gc.collect()
    sync()
    (out/"cont09_partial.json").write_text(json.dumps({"meta": meta, "cont09": cont09}, default=float))

    # ------------------------------------------------------------------ RUBBER-CONT-11 batched cost rows
    common = {"smoke_reduced_fidelity": SMOKE, "gpu": meta["device_name"], "gpu_index_physical": phys, "driver": meta["driver"],
              "commit": meta["commit"], "dirty_tree": meta["dirty_tree"], "protocol_sha256": meta["protocol_sha256"]}
    rows = []
    for label, r, contact in (("N4_no_contact", [False]*4, False), ("N4_contact_cont09_config", reach, True),
                              ("N4_contact_all_pinched", [True]*4, True), ("N1_contact_pinched", [True], True)):
        before = gpu_sample()
        row = cost_row(label, r, contact, args.cost_steps, args.device, peak, common)
        row["co_tenant_before_row"] = before
        row["co_tenant_after_row"] = gpu_sample()
        others = [a for a in before["apps"]+row["co_tenant_after_row"]["apps"]
                  if a and int(a.split(",")[0]) != os.getpid()]
        row["gpu_label"] = "SHARED_GPU" if others else "EXCLUSIVE_AT_SAMPLES"
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("row", "N", "ms_per_substep_warm", "env_steps_per_s_warm",
                                              "aggregate_sim_time_per_wall_warm", "reset_p95_ms",
                                              "vram_peak_mb_process_cumulative", "physics_verdict", "gpu_label")},
                         default=float), flush=True)
    meta["co_tenant_after"] = gpu_sample()
    others = [a for a in meta["co_tenant_before"]["apps"]+meta["co_tenant_after"]["apps"]
              if a and int(a.split(",")[0]) != os.getpid()]
    meta["shared_gpu"] = bool(others)
    meta["label_gpu"] = "SHARED_GPU" if others else "EXCLUSIVE_AT_SAMPLES"
    meta["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    (out/"result.json").write_text(json.dumps({"meta": meta, "cont09": cont09, "cost_rows": rows}, indent=1,
                                              default=float))
    print(json.dumps({"RUBBER-CONT-09": cont09["verdict"], "shared": meta["shared_gpu"]}))


if __name__ == "__main__":
    main()
