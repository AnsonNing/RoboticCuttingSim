#!/usr/bin/env python3
"""Export real online cut-history surfaces, exact blade and optional robot replay."""
import argparse
import hashlib
import json
import time
from dataclasses import fields
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, Vt
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition, CuttingSurface, wedge_mesh
from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.tlmpm import TLMPM, TLMPMConfig
from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge


def digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("physics_run", type=Path)
    parser.add_argument("workcell", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--robot-replay", type=Path)
    parser.add_argument("--samples", type=int, default=26)
    parser.add_argument("--voxel-size", type=float, default=.001)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.samples < 2 or args.voxel_size <= 0:
        parser.error("Need at least two snapshots and a positive voxel size")
    output = new_output(args.output_dir)
    (output / "surface_debug").mkdir()
    source = args.physics_run / "physics_state.npz"
    state = np.load(source)
    source_manifest = json.loads((args.physics_run / "manifest.json").read_text())
    config = source_manifest["config"]
    solver_config = config.get("solver")
    if solver_config is None:
        # Coupled runs record their exact material parameters in diagnostics/setup.
        setup = json.loads((args.physics_run / "setup.json").read_text())
        solver_config = setup["material"]
    solver = TLMPMConfig(**{k: v for k, v in solver_config.items() if k in {f.name for f in fields(TLMPMConfig)}})
    spacing = solver.dx / 2
    if not np.array_equal(state["positions"][0], state["reference_positions"]):
        raise ValueError("Initial particle state differs from the recorded uncut reference")
    if np.any(state["codes"][0] != 0):
        raise ValueError("The first snapshot must be an initially uncut continuum")
    # Build the solver's actual reference interpolation, without stepping physics.
    bed = TLMPM(state["reference_positions"], state["mass_kg"], spacing, solver, "cpu")
    partition = CurrentMechanicalPartition(bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy(), solver.cuts)
    np.savez_compressed(output / "surface_debug/reference_stencil.npz", nodes=bed.nodes.numpy(),
                         weights=bed.weights.numpy(), gradients=bed.gradients.numpy())
    surface = CuttingSurface(partition, spacing, args.voxel_size, args.device)
    stage = Usd.Stage.CreateNew(str(output / "cutting_workcell_replay.usda"))
    layers = [str(args.workcell.resolve())]
    if args.robot_replay:
        layers.insert(0, str(args.robot_replay.resolve()))
    stage.GetRootLayer().subLayerPaths = layers
    UsdGeom.SetStageMetersPerUnit(stage, 1.0); UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetTimeCodesPerSecond(60); stage.SetFramesPerSecond(60)
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    bridge = DynamicSurfaceBridge(stage)
    # A stronger Mesh opinion replaces the old box proxy. Its dimensions and edge
    # frame are identical to the physical TLMPM wedge, with no inherited scale.
    blade = UsdGeom.Mesh.Define(stage, "/World/KinematicBlade")
    vertices, faces = wedge_mesh(solver.blade_length, solver.blade_height, solver.blade_half_width)
    blade.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices))
    blade.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(faces), 3, np.int32)))
    blade.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.ravel()))
    blade.CreateSubdivisionSchemeAttr("none")
    xform = UsdGeom.Xformable(blade); xform.ClearXformOpOrder()
    translate = xform.AddTranslateOp(opSuffix="cutting")
    orient = xform.AddOrientOp(opSuffix="cutting")
    if args.robot_replay and stage.GetPrimAtPath("/World/Knife"):
        stage.OverridePrim("/World/Knife").SetActive(False)
    tool_source = source
    if "tool_pose_xyzw" in state.files:
        tool_times, tool_poses = state["time_s"], state["tool_pose_xyzw"]
    else:
        tool_source = args.physics_run / "robot_state.npz"
        robot = np.load(tool_source)
        tool_times, tool_poses = robot["time_s"], robot["tool_pose_xyzw"]
    for timestamp, pose in zip(tool_times, tool_poses):
        translate.Set(Gf.Vec3d(*map(float, pose[:3])), float(timestamp) * 60)
        orient.Set(Gf.Quatf(float(pose[6]), Gf.Vec3f(*map(float, pose[3:6]))), float(timestamp) * 60)
    translate.Set(Gf.Vec3d(*map(float, tool_poses[0, :3])))
    orient.Set(Gf.Quatf(float(tool_poses[0, 6]), Gf.Vec3f(*map(float, tool_poses[0, 3:6]))))
    # Separate debug camera can resolve a millimeter-scale cut. Native D455s and
    # the overview global camera are retained with their authored transforms.
    from robotic_cutting_sim.camera import calibration
    closeup = UsdGeom.Camera.Define(stage, "/World/CutCloseupCamera")
    c = calibration([.16, -.20, .13], target=(.02, 0, .012), hfov_deg=35)
    transform = np.asarray(c["T_world_camera"]); transform[:3, :3] @= np.diag([1, -1, -1])
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    UsdGeom.Xformable(closeup).AddTranslateOp().Set(Gf.Vec3d(*map(float, transform[:3, 3])))
    UsdGeom.Xformable(closeup).AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*map(float, q[:3]))))
    closeup.CreateHorizontalApertureAttr(20.955)
    closeup.CreateVerticalApertureAttr(20.955 * 480 / 640)
    closeup.CreateFocalLengthAttr(20.955 / (2 * np.tan(np.deg2rad(35) / 2)))
    closeup.CreateClippingRangeAttr(Gf.Vec2f(.01, 2))
    indices = np.unique(np.linspace(0, len(state["time_s"]) - 1, min(args.samples, len(state["time_s"])), dtype=int))
    rows, history = [], []
    tick = time.perf_counter()
    for index in range(len(state["time_s"])):
        count, labels, provenance = partition.partition(state["codes"][index])
        history.append({"source_frame": index, "physics_timestamp_s": float(state["time_s"][index]),
                        "mechanical_components": count, "particle_counts": np.bincount(labels).tolist(), **provenance})
        if index not in indices:
            continue
        frame_start = time.perf_counter()
        vertices, triangles, receipt = surface.extract(state["positions"][index], state["codes"][index])
        timestamp = float(state["time_s"][index])
        staged = bridge.update(vertices, triangles, timestamp, time_code=timestamp * 60)
        if not rows:
            bridge.mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
            bridge.mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
            bridge.mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.ravel()))
            bridge.mesh.GetExtentAttr().Set([Gf.Vec3f(*map(float, vertices.min(0))), Gf.Vec3f(*map(float, vertices.max(0)))])
        np.savez_compressed(output / "surface_debug" / f"{index:04d}.npz", vertices_m=vertices, triangles=triangles,
                            physics_timestamp_s=timestamp, current_mechanical_component=labels)
        rows.append({"source_frame": int(index), **receipt, **staged, "wall_s": time.perf_counter() - frame_start})
        print(f"surface={index} t={timestamp:.3f}s components={count} triangles={len(triangles)}", flush=True)
    stage.SetStartTimeCode(float(state["time_s"][0]) * 60)
    stage.SetEndTimeCode(float(state["time_s"][-1]) * 60)
    stage.GetRootLayer().customLayerData = {"status": "MEASURED_EXPERIMENTAL_CUTTING_REPLAY",
        "physics_source_sha256": digest(source), "component_provenance": "current MPM stencils and code history only",
        "future_component_ids_used": False, "physics_acceptance": "not established by rendering"}
    stage.GetRootLayer().Save()
    source_hashes = {"physics_state.npz": digest(source), "manifest.json": digest(args.physics_run / "manifest.json"),
                     "tool_source": digest(tool_source), "workcell": digest(args.workcell)}
    if args.robot_replay:
        source_hashes["robot_replay"] = digest(args.robot_replay)
    result = {"status": "PASS", "scope": "measured current-mechanical-component surfaces and exact physical blade to USD",
              "source_run": str(args.physics_run.resolve()), "source_hashes": source_hashes,
              "stencil_sha256": partition.stencil_sha256, "frames": rows, "mechanical_history": history,
              "first_mechanical_components": history[0]["mechanical_components"],
              "final_mechanical_components": history[-1]["mechanical_components"],
              "future_component_ids_used": False, "wall_s": time.perf_counter() - tick,
              "rtx_capture": "NOT_RUN", "physical_cutting_acceptance": "NOT_ESTABLISHED_BY_RENDERER"}
    write_json(output / "result.json", result)
    write_json(output / "manifest.json", manifest({"physics_run": str(args.physics_run.resolve()),
        "source_hashes": source_hashes, "samples": args.samples, "device": args.device,
        "surface_voxel_size_m": args.voxel_size, "radius_m": spacing / 2,
        "surface_method": "Newton ParticleSurface, no point/mesh smoothing, separate current mechanical components"},
        "experimental_cutting_surface_replay"))
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "mechanical_history")}, indent=2))


if __name__ == "__main__":
    main()
