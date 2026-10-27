#!/usr/bin/env python3
"""Audit all 18 frozen gates using actual recorded live Isaac episode data."""
import argparse
from pathlib import Path
import shutil

import numpy as np

from robotic_cutting_sim.live_acceptance import audit_live_episode, compare_standalone_reference, sha256
from robotic_cutting_sim.recording import ROOT, new_output, write_json


def plot_trace(data, output, source):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trajectory, robot = data["trajectory"], data["robot"]
    if "position_error_m" not in trajectory:
        return
    time = robot["time_s"]
    command, actual = trajectory["local_command_poses_xyzw"], trajectory["local_actual_poses_xyzw"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    axes[0, 0].plot(command[:, 0]*1000, command[:, 1]*1000, label="commanded edge")
    axes[0, 0].plot(actual[:, 0]*1000, actual[:, 1]*1000, label="measured edge")
    axes[0, 0].add_patch(plt.Circle((0, 0), 15, fill=False, color="gray"))
    axes[0, 0].set(xlabel="workpiece X [mm]", ylabel="workpiece Y [mm]", aspect="equal")
    axes[0, 1].plot(time, command[:, 2]*1000, label="commanded edge")
    axes[0, 1].plot(time, actual[:, 2]*1000, label="measured edge")
    axes[0, 1].axhline(data["target"].height_m*1000, color="gray", linestyle="--", label="local target")
    axes[0, 1].set(xlabel="simulation time [s]", ylabel="local edge Z [mm]")
    axes[1, 0].plot(time, trajectory["position_error_m"]*1000)
    axes[1, 0].axhline(1, linestyle="--", color="gray")
    axes[1, 0].set(xlabel="simulation time [s]", ylabel="3D tracking error [mm]")
    axes[1, 1].plot(time, np.rad2deg(trajectory["rotation_error_rad"]))
    axes[1, 1].set(xlabel="simulation time [s]", ylabel="orientation error [degrees]")
    for ax in axes[0]:
        ax.legend(fontsize=8)
    fig.suptitle(f"Actual recorded {source} — command versus measured knife edge")
    fig.tight_layout(); fig.savefig(output/"planned_vs_actual_edge.png", dpi=140); plt.close(fig)
    surface = data["surface"]
    if surface is not None:
        error = np.where(surface["valid_heightfield_debug"], (surface["measured_heightfield_local_m_debug"]-data["target"].height_m)*1000, np.nan)
        fig, ax = plt.subplots(figsize=(7, 4))
        view = ax.imshow(error, origin="lower", extent=[-60, 60, -40, 40], vmin=-2, vmax=2, cmap="coolwarm")
        ax.set(xlabel="workpiece X [mm]", ylabel="workpiece Y [mm]", title="Actual mechanical-substrate mesh signed target error [mm]")
        fig.colorbar(view, ax=ax); fig.tight_layout(); fig.savefig(output/"substrate_surface_error.png", dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recording-dir", type=Path, required=True)
    p.add_argument("--protocol", type=Path, default=ROOT/"configs/isaac_acceptance.json")
    p.add_argument("--contact-device", default="none", help="actual saved-state contact kernel recheck device, e.g. cuda:0; none disables")
    p.add_argument("--feedback-off-dir", type=Path)
    p.add_argument("--repeat-dir", type=Path)
    p.add_argument("--standalone-reference-dir", type=Path, help="descriptive same-recipe comparison; never passes a live gate")
    p.add_argument("--native-camera-controls", type=Path, help="exact-version independent native Kit known-geometry audit result")
    p.add_argument("--live-camera-audit", type=Path, help="independent measured-state scene/native camera consistency audit")
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    import json
    protocol = json.loads(args.protocol.read_text())
    output = new_output(args.output_dir)
    (output/"validator_source").mkdir()
    validator_paths = (Path(__file__), ROOT/"src/robotic_cutting_sim/live_acceptance.py",
        ROOT/"src/robotic_cutting_sim/scene_frames.py", ROOT/"src/robotic_cutting_sim/scene_validation.py", args.protocol)
    for path in validator_paths:
        shutil.copyfile(path, output/"validator_source"/path.name)
    result, debug = audit_live_episode(args.recording_dir, protocol, protocol_path=args.protocol,
        contact_device=None if args.contact_device == "none" else args.contact_device,
        feedback_off_dir=args.feedback_off_dir, repeat_dir=args.repeat_dir, native_camera_controls=args.native_camera_controls,
        live_camera_audit=args.live_camera_audit)
    result["validator_sha256"] = {label: sha256(output/"validator_source"/path.name)
        for label, path in zip(("driver", "live_acceptance", "scene_frames", "scene_validation"), validator_paths[:4])}
    result["validator_sources_unchanged_during_analysis"] = all(
        sha256(path) == sha256(output/"validator_source"/path.name) for path in validator_paths)
    if args.standalone_reference_dir:
        result["standalone_reference_comparison"] = compare_standalone_reference(args.recording_dir, args.standalone_reference_dir)
    trajectory = debug["trajectory"]
    if "position_error_m" in trajectory:
        np.savez_compressed(output/"trajectory_errors.npz", time_s=debug["robot"]["time_s"],
            position_error_m=trajectory["position_error_m"], rotation_error_rad=trajectory["rotation_error_rad"],
            active_mask=trajectory["active_mask"], measured_edge_local_xyzw=trajectory["local_actual_poses_xyzw"],
            commanded_edge_local_xyzw=trajectory["local_command_poses_xyzw"])
    surface = debug["surface"]
    if surface is not None:
        np.savez_compressed(output/"surface_height_measurements.npz",
            measured_height_local_m=surface["measured_heightfield_local_m_debug"], valid=surface["valid_heightfield_debug"])
    write_json(output/"result.json", result)
    plot_trace(debug, output, result["runtime_source"])
    print(json.dumps({"gate_counts": result["gate_counts"], "execution_pass": result["execution_pass"],
        "physical_cutting_pass": result["physical_cutting_pass"], "surface_flattening_pass": result["surface_flattening_pass"],
        "all_18_gates_pass": result["all_18_gates_pass"], "result": str(output/"result.json")}))
    return 0 if result["all_18_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
