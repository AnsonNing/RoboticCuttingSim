"""WP-12 Phase 3: cut_events_v1 GPU check on the scripted CUT-01-R kinematic through-cut (tlmpm_v5, law 8).

Runs the same episode with the recorder off and on (interleaved: off, on, off, on), compares final codes and
positions against the documented GPU nondeterminism band (ADR 0012: equal final codes, |dx| <= 0.06 mm),
measures the recorder overhead (episode A/B and an isolated graph of recorder-only launches) and writes
the log. Never overwrites an existing output directory.

    TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_cut_events_gpu.py --device cuda:1 \
        --out artifacts/wp12/cut_events_001 --evidence docs/evidence/wp12/phase3_cut_events_001.json
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

BAND_M = 6e-5   # ADR 0012: GPU nondeterminism band |dx| <= 0.06 mm with equal final codes


def run_episode(option, args):
    import warp as wp
    from robotic_cutting_sim.kinematic_env import KinematicConfig, KinematicCuttingBackend
    backend = KinematicCuttingBackend(KinematicConfig("CUT-01-R", seed=args.seed, particle_spacing_m=args.spacing,
                                                      material_dt=args.material_dt, device=args.device,
                                                      cutting_law=8, solver="tlmpm_v5", cut_events=option))
    start = time.perf_counter()
    receipt = backend.reset(seed=args.seed)
    material, task = backend.material, backend.task
    walls, steps = [], 0
    while backend.time_s+task.control_dt_s <= task.duration_s+1e-9:
        _, info = backend.step()
        walls.append(info["material"]["material_wall_s"])
        steps += 1
    wp.synchronize_device(args.device)
    state = material.debug_state()
    out = {"option": option, "steps": steps, "substeps_per_step": material.substeps, "particles": int(material.bed.n),
           "episode_wall_s": time.perf_counter()-start, "reset_wall_s": float(receipt["reset_wall_s"]),
           "material_wall_s": np.asarray(walls), "state": state, "slots": material.bed.check_slots(),
           "labelled": int(np.count_nonzero(state["codes"]))}
    if option is not None:
        recorder = material.cut_event_recorder
        out.update(metadata=recorder.metadata(), log=recorder.log(), activation=recorder.activation(),
                   neighbours=recorder.neighbours_host, reference=recorder.reference_host, lattice=recorder.lattice)
        out["isolated"] = isolated_recorder_cost(material, args)
    backend.close()
    return out


def isolated_recorder_cost(material, args, repeats=20):
    """Recorder-only launches (copy + 2 kernels) for one control step, captured in a graph, on the final state."""
    import warp as wp
    recorder, bed = material.cut_event_recorder, material.bed
    saved = {name: getattr(recorder, name).numpy().copy() for name in ("substep", "count", "overflow", "scratch",
                                                                        "summary_overflow")}
    with wp.ScopedCapture(device=args.device) as capture:
        for _ in range(material.substeps):
            recorder.before_step()
            recorder.after_step()
    walls = []
    for _ in range(repeats):
        wp.synchronize_device(args.device)
        start = time.perf_counter()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(args.device)
        walls.append(time.perf_counter()-start)
    for name, value in saved.items():   # restore the counters (the log itself is untouched: code == code_prev)
        getattr(recorder, name).assign(value)
    return {"control_steps": repeats, "substeps_per_graph": material.substeps,
            "median_ms_per_substep": float(np.median(walls)/material.substeps*1e3),
            "scope": "copy(code_prev, code) + _record_cut_events + _record_cut_summary per substep on the final "
                     "state (code == code_prev, so the event kernel exits early: the steady-state cost)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--spacing", type=float, default=.001)
    parser.add_argument("--material-dt", type=float, default=1e-4)
    parser.add_argument("--out", type=Path, default=ROOT/"artifacts/wp12/cut_events_001")
    parser.add_argument("--evidence", type=Path, default=ROOT/"docs/evidence/wp12/phase3_cut_events_001.json")
    parser.add_argument("--order", default="off,on,off,on")
    args = parser.parse_args()
    if args.out.exists() or args.evidence.exists():
        raise SystemExit(f"Refusing to overwrite {args.out} / {args.evidence}")
    args.out.mkdir(parents=True)
    import warp as wp
    from robotic_cutting_sim import cut_events as ce
    command = " ".join([sys.executable, *sys.argv])
    runs = []
    for k, label in enumerate(args.order.split(",")):
        option = {"off": None, "on": ce.CUT_EVENTS_VERSION}[label]
        result = run_episode(option, args)
        runs.append(result)
        tag = f"run{k}_{label}"
        np.savez_compressed(args.out/f"{tag}_final_state.npz", positions=result["state"]["positions"],
                            velocities=result["state"]["velocities"], codes=result["state"]["codes"],
                            material_wall_s=result["material_wall_s"])
        if option is not None:
            np.save(args.out/f"{tag}_cut_events_log.npy", result["log"])
            np.save(args.out/f"{tag}_cut_events_activation.npy", result["activation"])
            (args.out/f"{tag}_cut_events_metadata.json").write_text(json.dumps(result["metadata"], indent=2))
        print(f"{tag}: {result['episode_wall_s']:.1f} s, labelled {result['labelled']}, "
              f"mean step {result['material_wall_s'].mean()*1e3:.2f} ms", flush=True)

    off = [r for r in runs if r["option"] is None]
    on = [r for r in runs if r["option"] is not None]
    reference = off[0]["state"]

    def compare(a, b):
        dx = np.abs(a["positions"].astype(np.float64)-b["positions"]).max()
        dv = np.abs(a["velocities"].astype(np.float64)-b["velocities"]).max()
        return {"codes_equal": bool(np.array_equal(a["codes"], b["codes"])), "max_abs_dx_m": float(dx),
                "max_abs_dv_m_s": float(dv), "within_band": bool(np.array_equal(a["codes"], b["codes"]) and dx <= BAND_M)}

    comparisons = {f"run{runs.index(r)}_vs_run0": compare(reference, r["state"]) for r in runs[1:]}
    # Audits on the first recorder run.
    first = on[0]
    log, codes, neighbours = first["log"], first["state"]["codes"], first["neighbours"]
    conflicting = ce.links_conflicting_in_cut(codes, neighbours, 0)
    logged = ce.log_links(log, 0)
    plane_x = 0.   # CUT-01-R: scripted edge at workpiece-local x = 0 (lattice columns at +-0.5 mm)
    offset = np.abs(log["hermite_reference"][:, 0]-plane_x)/args.spacing
    ok = (log["flags"] & ce.FLAG_SIGN_INCONSISTENT) == 0
    angle = np.degrees(np.arccos(np.clip(log["normal_reference"]@np.array([1., 0, 0]), -1, 1)))
    steps = first["steps"]*first["substeps_per_step"]
    mean_ms = lambda rs: float(np.mean([r["material_wall_s"].mean() for r in rs])*1e3)  # noqa: E731
    median_ms = lambda rs: float(np.median(np.concatenate([r["material_wall_s"] for r in rs]))*1e3)  # noqa: E731
    sub = first["substeps_per_step"]
    timing = {"mean_control_step_ms_off": mean_ms(off), "mean_control_step_ms_on": mean_ms(on),
              "median_control_step_ms_off": median_ms(off), "median_control_step_ms_on": median_ms(on),
              "mean_ms_per_substep_off": mean_ms(off)/sub, "mean_ms_per_substep_on": mean_ms(on)/sub,
              "overhead_ms_per_substep_mean": (mean_ms(on)-mean_ms(off))/sub,
              "overhead_ms_per_substep_median": (median_ms(on)-median_ms(off))/sub,
              "overhead_fraction_mean": (mean_ms(on)-mean_ms(off))/mean_ms(off),
              "isolated_recorder_ms_per_substep": [r["isolated"]["median_ms_per_substep"] for r in on],
              "episode_wall_s": {f"run{k}_{'on' if r['option'] else 'off'}": r["episode_wall_s"]
                                 for k, r in enumerate(runs)},
              "scope": "material_wall_s per control step (CUDA graph launch + sync + slot check; with the recorder "
                       "also its overflow check); interleaved runs on one GPU; another agent may share the GPU"}
    log_file = next(args.out.glob("run*_on_cut_events_log.npy"))
    summary = {
        "experiment": "wp12_phase3_cut_events_001", "recorder": ce.CUT_EVENTS_VERSION,
        "command": command, "cwd": str(ROOT),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                                         text=True).stdout.strip()),
        "warp": wp.__version__, "python": platform.python_version(), "device": args.device,
        "device_name": wp.get_device(args.device).name,
        "task": "CUT-01-R scripted reference edge (kinematic ideal tracking)", "seed": args.seed,
        "particle_spacing_m": args.spacing, "material_dt_s": args.material_dt, "solver": "tlmpm_v5",
        "cutting_law": 8, "max_cuts": 1, "particles": first["particles"], "control_steps": first["steps"],
        "substeps": steps, "run_order": args.order,
        "labelled": {f"run{k}": r["labelled"] for k, r in enumerate(runs)},
        "final_comparisons": comparisons, "band": {"max_abs_dx_m": BAND_M, "source": "ADR 0012 (GPU nondeterminism)"},
        "neutrality_pass": all(c["within_band"] for c in comparisons.values()),
        "recorder_metadata": first["metadata"],
        "audits": {"logged_links": len(logged), "conflicting_links_final_codes": len(conflicting),
                   "logged_equals_conflicting": logged == conflicting,
                   "sign_inconsistent": int((~ok).sum()), "t_range": [float(log["t"][ok].min()), float(log["t"][ok].max())],
                   "reference_plane_offset_spacings": {"p50": float(np.percentile(offset, 50)),
                                                       "p95": float(np.percentile(offset, 95)),
                                                       "max": float(offset.max())},
                   "reference_normal_angle_to_x_deg": {"p50": float(np.percentile(angle, 50)),
                                                       "p95": float(np.percentile(angle, 95)),
                                                       "max": float(angle.max())},
                   "plane": "workpiece-local x = 0 (scripted edge line; reference configuration)"},
        "log_size": {"events": int(len(log)), "log_bytes": int(log.nbytes), "log_file_bytes": log_file.stat().st_size,
                     "activation_bytes": int(first["activation"].nbytes),
                     "recorder_device_bytes": first["metadata"]["device_bytes"]},
        "timing": timing,
        "artifacts": str(args.out.relative_to(ROOT)) if args.out.is_relative_to(ROOT) else str(args.out),
    }
    (args.out/"summary.json").write_text(json.dumps(summary, indent=2, default=str))
    args.evidence.parent.mkdir(parents=True, exist_ok=True)
    args.evidence.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ("final_comparisons", "neutrality_pass", "audits", "log_size", "timing")},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
