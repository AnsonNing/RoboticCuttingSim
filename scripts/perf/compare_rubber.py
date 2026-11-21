"""Compare matched Rubber benchmark traces and write compact drag/release evidence."""
import argparse
import json
from pathlib import Path

import numpy as np


def load_run(path):
    return tuple(json.loads((path / f"{name}.json").read_text()) for name in ("benchmark", "trace"))


def compare(reference, candidate):
    ref, ref_trace = reference
    cand, cand_trace = candidate
    n = min(ref["num_envs"], cand["num_envs"])
    for key in ("slots", "seeds"):
        if ref[key][:n] != cand[key][:n]:
            raise ValueError(f"unmatched paired {key}")
    for key in ("control_dt_s", "warmup_steps", "measured_drag_steps", "release_steps", "drive"):
        if ref.get(key) != cand.get(key):
            raise ValueError(f"unmatched paired {key}")
    if [(row["step"], row["phase"]) for row in ref_trace] != [(row["step"], row["phase"]) for row in cand_trace]:
        raise ValueError("unmatched time or phase sequences")

    def columns(rows, key):
        return np.asarray([row[key][:n] for row in rows])

    ref_markers = columns(ref_trace, "markers_m")
    cand_markers = columns(cand_trace, "markers_m")
    compare_mask = np.asarray([row["phase"] != "warmup" for row in ref_trace])
    drag = np.asarray([row["phase"] == "drag" for row in ref_trace])
    release = np.asarray([row["phase"] == "release" for row in ref_trace])
    tip_error = np.linalg.norm((cand_markers - ref_markers)[:, :, -1, :], axis=-1)
    initial = ref["warmup_steps"] - 1
    last_drag = initial + ref["measured_drag_steps"]

    def trajectory(rows, markers):
        forces = columns(rows, "plane_force_n")
        tip = markers[:, :, -1, :]
        span = np.linalg.norm(tip - markers[:, :, 0, :], axis=-1)
        sag = (markers[:, :, :, 2] - markers[:, :, :1, 2]).min(axis=-1)
        floor = np.asarray([r["min_particle_z_m"][:n] for r in rows if "min_particle_z_m" in r])
        dt = ref["control_dt_s"]
        out = {
            "drag_end_tip_m": tip[last_drag].tolist(),
            "drag_tip_xy_shift_m": (tip[last_drag, :, :2] - tip[initial, :, :2]).tolist(),
            "drag_extreme_marker_span_m": np.stack([span[drag].min(0), span[drag].max(0)], axis=-1).tolist(),
            "drag_peak_sag_m": sag[drag].min(0).tolist(),
            "drag_normal_impulse_abs_n_s": (np.abs(forces[drag, :, 2]).sum(0) * dt).tolist(),
            "drag_shear_impulse_abs_n_s": (np.linalg.norm(forces[drag, :, :2], axis=-1).sum(0) * dt).tolist(),
            "peak_node_penetration_m": columns(rows, "penetration_m")[compare_mask].max(0).tolist(),
            "sampled_min_particle_z_m": floor.min(0).tolist() if floor.size else None,
            "max_particle_speed_m_s": columns(rows, "speed_max_m_s")[compare_mask].max(0).tolist(),
            "min_J": columns(rows, "J_min")[compare_mask].min(0).tolist(),
            "max_J": columns(rows, "J_max")[compare_mask].max(0).tolist(),
        }
        if release.any():
            out.update(release_final_tip_m=tip[-1].tolist(),
                       release_max_tip_motion_m=np.linalg.norm(tip[release] - tip[last_drag], axis=-1).max(0).tolist(),
                       release_final_tip_speed_m_s=rows[-1]["tip_speed_m_s"][:n],
                       release_peak_abs_grip_force_n=np.linalg.norm(columns(rows, "grip_force_n")[release],
                                                                      axis=-1).max(0).tolist(),
                       release_loaded_steps=(np.abs(forces[release, :, 2]) > 1e-8).sum(0).tolist())
        return out

    return {"ref_profile": ref["profile"], "candidate_profile": cand["profile"], "paired_envs": n,
            "slots": ref["slots"][:n], "seeds": ref["seeds"][:n],
            "reference_status": ref["status"], "candidate_status": cand["status"],
            "drag_tip_error_max_m": tip_error[drag].max(axis=0).tolist(),
            "drag_tip_error_rms_m": np.sqrt(np.mean(tip_error[drag]**2, axis=0)).tolist(),
            "release_tip_error_max_m": tip_error[release].max(axis=0).tolist() if release.any() else None,
            "reference": trajectory(ref_trace, ref_markers),
            "candidate": trajectory(cand_trace, cand_markers)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = compare(load_run(args.reference), load_run(args.candidate))
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
