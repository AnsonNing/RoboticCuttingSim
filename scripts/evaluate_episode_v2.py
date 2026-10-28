#!/usr/bin/env python3
"""Reevaluate immutable CUT/TRIM recordings with all Evaluation v2 profiles.

Exit zero means the requested analysis completed, never task success. Read the
separate episode, integrity, goal and quality statuses in result.json.
"""
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT))

from robotic_cutting_sim.evaluation import PROTOCOL_PATH, evaluate_recording, goal_for_task
from robotic_cutting_sim.live_acceptance import graph_audit, load_npz, resolve_recorded_source, sha256
from robotic_cutting_sim.recording import new_output, write_json
from robotic_cutting_sim.scene_frames import pose_errors
from robotic_cutting_sim.surface_evaluation import TargetGeometry, evaluate_3d_surface
from robotic_cutting_sim.tasks import make_task, PROTOCOL_PATH as LEGACY_PROTOCOL
from scripts.audit_cutting_task_episodes import load_surface


def load_episode(directory):
    directory = Path(directory).resolve()
    manifest_path, setup_path = directory/"manifest.json", directory/"setup.json"
    manifest, setup = [json.loads(p.read_text()) for p in (manifest_path, setup_path)]
    task_path = directory/"task.json"
    recorded = json.loads(task_path.read_text()) if task_path.exists() else setup["task"]
    input_pose = recorded.get("randomization", {}).get("input_world_from_workpiece_xyzw", recorded["world_from_workpiece_xyzw"])
    task = make_task(recorded["task_id"], recorded["seed"], recorded["variant"],
        world_from_workpiece_xyzw=input_pose, particle_spacing_m=recorded["particle_spacing_m"],
        workpiece_kind=recorded.get("workpiece", {}).get("kind", recorded.get("workpiece_kind", "rectangle")))
    generated = json.loads(json.dumps(asdict(task)))
    recorded_fields = {k:recorded[k] for k in generated}
    errors = pose_errors(generated.pop("world_from_workpiece_xyzw"), recorded_fields.pop("world_from_workpiece_xyzw"))
    if generated != recorded_fields or max(errors) > 1e-12 or recorded["protocol_sha256"] != sha256(LEGACY_PROTOCOL):
        raise ValueError("Recorded task differs from frozen legacy recipe/protocol")
    task = replace(task, world_from_workpiece_xyzw=tuple(recorded["world_from_workpiece_xyzw"]))
    oracle = directory/"oracle_debug" if (directory/"oracle_debug").is_dir() else directory
    physics_path, diagnostics_path = oracle/"physics_state.npz", oracle/"diagnostics.json"
    data = load_npz(physics_path) if physics_path.exists() else {}
    paths = [manifest_path, setup_path]+([physics_path] if physics_path.exists() else [])
    if task_path.exists():
        paths.append(task_path)
    graph, labels = graph_audit(data, 1)
    metadata = None
    try:
        surface, metadata, mesh_paths = load_surface(directory, True)
        data.update(surface)
        paths += mesh_paths
    except (FileNotFoundError, IndexError):
        pass  # Explicitly unmeasurable surface; physics/outcome remain usable.
    codes_hash = hashlib.sha256(data["codes"][-1].tobytes()).hexdigest() if "codes" in data else None
    data["surface_graph_verified"] = bool(metadata is not None and graph["status"] == "PASS"
        and metadata.get("stencil_sha256") == graph["stencil_sha256"] and metadata.get("codes_sha256") == codes_hash
        and metadata.get("future_component_ids_used") is False and metadata.get("renderer_partition_changes_physics") is False)
    diagnostics = json.loads(diagnostics_path.read_text()) if diagnostics_path.exists() else []
    if diagnostics_path.exists():
        paths.append(diagnostics_path)
    verified = {path:resolve_recorded_source(directory, path, digest)
        for path,digest in manifest.get("implementation_sha256", {}).items()}
    data["source_evidence"] = {"implementation_verified": bool(verified) and all(v["verified"] for v in verified.values()),
        "physics_state_from_recording": physics_path.exists(),
        "contact_deformation_traceable": bool(diagnostics and "final_F" in data
            and all("material_contact_statistics" in row for row in diagnostics))}
    data["geometry_frame_id"] = "workpiece_local"
    _, expected_mass, _ = task.sample_particles()
    data["expected_reference_mass_kg"] = float(expected_mass.sum())
    if "final_F" in data:
        data["deformation_jacobian"] = np.linalg.det(data["final_F"])
    original_path = oracle/"task_evaluation.json"
    original = json.loads(original_path.read_text()) if original_path.exists() else None
    if original_path.exists():
        paths.append(original_path)
    return task, data, paths, verified, original


def load_represented_target(path, task):
    path = Path(path)
    protocol_path, target_path = path/"protocol.json", path/"fixed_target.npz"
    protocol = json.loads(protocol_path.read_text())
    target = protocol["target"]
    if (target.get("no_bump") is not True or target.get("no_material_dynamics") is not True
            or not np.allclose(target["dimensions_m"], [task.geometry.length_m, task.geometry.width_m, task.geometry.base_height_m], atol=1e-12, rtol=0)
            or target["particle_spacing_m"] != task.particle_spacing_m):
        raise ValueError("Represented target must independently match initial resolution and geometry")
    arrays = load_npz(target_path)
    return TargetGeometry("mesh", str(target_path.resolve()), protocol["protocol_id"],
        parameters={"vertices_m": arrays["vertices_local_m"], "triangles": arrays["triangles"],
            "independent_uncut_target": True}), [target_path, protocol_path]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--represented-target", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    sources = [Path(__file__).resolve(), ROOT/"scripts/audit_cutting_task_episodes.py", PROTOCOL_PATH, LEGACY_PROTOCOL,
        *[ROOT/"src/robotic_cutting_sim"/name for name in ("evaluation.py", "surface_evaluation.py", "surface_distance.py",
            "tasks.py", "scene_validation.py", "cut_topology.py", "live_acceptance.py", "scene_frames.py")]]
    source_hashes = {str(p.relative_to(ROOT)):sha256(p) for p in sources}
    snapshots = output/"implementation"; snapshots.mkdir()
    for source in sources:
        shutil.copyfile(source, snapshots/source.name)
    write_json(output/"analysis_start.json", {"source_sha256":source_hashes, "command":sys.argv})
    task, data, inputs, verified, original = load_episode(args.episode)
    goal = goal_for_task(task)
    represented = None
    if args.represented_target:
        represented, target_paths = load_represented_target(args.represented_target, task)
        inputs += target_paths
    input_hashes = {str(p.resolve()):sha256(p) for p in inputs}
    profiles = {}
    for profile in ("diagnostic", "research", "precision"):
        result, raw = evaluate_recording(data, task=task, goal=goal, profile=profile, represented_target=represented)
        if profile == "diagnostic":
            np.savez_compressed(output/"raw_geometry_samples.npz", **raw)
        # Geometry has its own hashed NPZ; do not duplicate mesh arrays in JSON.
        if represented is not None:
            result["represented_target_provenance"] = {"source":represented.source, "version":represented.version,
                "sha256":sha256(Path(represented.source)), "independent_uncut_target":True}
        write_json(output/(profile+".json"), result)
        profiles[profile] = result
    write_json(output/"episode_manifest_v2.json", {
        "schema":"research_episode_evaluation_v2", "episode":str(args.episode.resolve()),
        "task_scene_configuration":task.to_dict(), "seed":task.seed,
        "initial_geometry":task.workpiece.to_dict(), "goal":profiles["diagnostic"]["goal"],
        "trajectory_actions_source":str(args.episode.resolve()/"robot_state.npz"),
        "observation_source":str(args.episode.resolve()/("observations" if (args.episode/"observations").exists() else "rgbd")),
        "simulation_time_s":np.asarray(data.get("time_s",[])).tolist(),
        "evaluations":{profile:{"path":profile+".json","sha256":sha256(output/(profile+".json"))} for profile in profiles},
        "failed_and_partial_episodes_retained":True,"units":profiles["diagnostic"]["units"],
        "source_contract":"References original captured assets; no observation/action is regenerated or oracle state inserted into vision observations"})
    if represented is not None and data.get("surface_graph_verified"):
        from robotic_cutting_sim.scene_validation import material_task_metrics
        from robotic_cutting_sim.scene_frames import WorkpieceTarget
        _, labels = graph_audit(data, 1)
        trim = material_task_metrics(data["reference_positions"],data["positions"][-1],data["mass_kg"],labels,"actual current MPM graph",
            WorkpieceTarget(task.world_from_workpiece_xyzw,task.target_origin_local_m[2]),task.geometry)
        selection = np.isin(data["surface_triangle_component_ids"],trim["reference_substrate_component_ids"])
        distance, raw = evaluate_3d_surface(task.points_to_local(data["surface_vertices_world"]),
            data["surface_triangles"][selection],represented,actual_frame_id="workpiece_local")
        write_json(output/"represented_3d_distance.json",distance)
        np.savez_compressed(output/"raw_3d_samples.npz",**raw)
    legacy = profiles["precision"]["legacy_precision_benchmark"]
    same = None if original is None else all(legacy.get(k)==original.get(k) for k in ("status","checks","metrics"))
    input_stable = all(sha256(Path(p)) == digest for p,digest in input_hashes.items())
    source_stable = all(sha256(ROOT/p) == digest for p,digest in source_hashes.items())
    compact = {"status":"ANALYSIS_COMPLETE" if input_stable and source_stable and same is not False else "FAIL_EVIDENCE_IDENTITY",
        "episode":str(args.episode.resolve()), "task_id":task.task_id, "seed":task.seed, "variant":task.variant,
        "protocol_id":profiles["diagnostic"]["protocol_id"], "profiles":{p:{"file":p+".json","sha256":sha256(output/(p+".json")),
            "success":r["success"],"assessment":r["profile_assessment"]} for p,r in profiles.items()},
        "physical_integrity":profiles["diagnostic"]["physical_integrity"],
        "cutting_outcome":profiles["diagnostic"]["cutting_outcome"],
        "legacy_score_matches_recorded_status_checks_metrics":same, "input_files_unchanged":input_stable,
        "analysis_sources_unchanged":source_stable, "input_sha256":input_hashes, "analysis_source_sha256":source_hashes,
        "recorded_runtime_source_verification":verified, "full_physics_acceptance":False,
        "recording_assets":"Original observations/actions/trajectory remain in the episode; no raw media or physics state rewritten"}
    write_json(output/"result.json",compact)
    print(compact["status"],task.task_id, {p:{"goal":r["profile_assessment"]["goal_status"],
        "surface_quality":r["profile_assessment"]["surface_quality_status"],"success":r["success"]}
        for p,r in profiles.items()},flush=True)
    return int(compact["status"].startswith("FAIL"))


if __name__ == "__main__":
    raise SystemExit(main())
