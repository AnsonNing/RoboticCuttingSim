#!/usr/bin/env python3
"""Record CPU task geometry/plans; never report physical cutting or Isaac success."""
import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path

import numpy as np

from robotic_cutting_sim.cut_topology import reference_components
from robotic_cutting_sim.recording import ROOT, new_output, write_json
from robotic_cutting_sim.tasks import PROTOCOL_PATH, make_task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = new_output(args.output_dir)
    rows = {}
    for name in ("CUT-01", "TRIM-01"):
        task = make_task(name, args.seed)
        points, mass, lattice = task.sample_particles()
        count, _ = reference_components(points, lattice, np.zeros(len(points), int), 1)
        time = np.linspace(0, task.duration_s, round(task.duration_s/task.control_dt_s)+1)
        np.savez_compressed(out/(name+"_initial.npz"), points_workpiece_local_m=points,
            mass_kg=mass, lattice=lattice)
        np.savez_compressed(out/(name+"_planned.npz"), time_s=time,
            commanded_edge_world_xyzw=[task.reference_edge_pose_world(t) for t in time],
            derived_commanded_hand_world_xyzw=[task.reference_hand_pose_world(t) for t in time],
            scripted_cut_index=[task.active_cut_index(t) for t in time])
        rows[name] = {"task": task.to_dict(), "particle_count": len(points), "mass_kg": float(mass.sum()),
            "geometric_six_neighbor_components": count, "mechanical_components": "NOT_RUN; actual live stencils required",
            "initial_full_cell_support_bounds_local_m": [(points.min(0)-task.particle_spacing_m/2).tolist(),
                (points.max(0)+task.particle_spacing_m/2).tolist()],
            "final_commanded_edge_local_xyzw": task.reference_edge_pose_local(task.active_cut_interval_s[1]),
            "final_derived_commanded_hand_world_xyzw": task.reference_hand_pose_world(task.active_cut_interval_s[1]),
            "fixture_boxes_local": [asdict(box) for box in task.fixture_boxes_local], "physics_executed": False}
    write_json(out/"result.json", {"scope": "CPU generation, geometric contiguity and scripted plans only", "tasks": rows,
        "protocol_sha256": hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
        "task_source_sha256": hashlib.sha256((ROOT/"src/robotic_cutting_sim/tasks.py").read_bytes()).hexdigest(),
        "physical_cutting": "NOT_RUN", "isaac_runtime": "NOT_RUN", "full_benchmark_ready": False})
    print({name: {"particles": row["particle_count"], "mass_kg": row["mass_kg"],
        "geometric_components": row["geometric_six_neighbor_components"]} for name, row in rows.items()})


if __name__ == "__main__":
    main()
