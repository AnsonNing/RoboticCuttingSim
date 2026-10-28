#!/usr/bin/env python3
"""Actual isolated PhysX contact/timing control, not a Franka-grasp validation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    implementation = output/"implementation"; implementation.mkdir()
    paths = [Path(__file__), ROOT/"src/robotic_cutting_sim/physical_contacts.py", ROOT/"configs/isaac_sim_cutting.kit"]
    hashes = {}
    for path in paths:
        (implementation/path.name).write_bytes(path.read_bytes())
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    protocol = {"scope": "Actual native CPU PhysX with GPU Kit startup: floor support, two-sided V-cradle contact and unsupported gravity; no Franka/gripper or material validation",
        "dt_s": .002, "steps": 350, "mass_kg": .1, "gravity_m_s2": 9.81, "gpu": args.gpu,
        "floor_external_lateral_force_n": .2,
        "source_sha256": hashes, "acceptance": {"finite_raw_force": True, "floor_weight_relative_error_max": .05,
            "fresh_bilateral_cradle_steps_min": 20, "unsupported_downward_displacement_min_m": .1,
            "floor_friction_relative_error_max": .05}}
    (output/"protocol.json").write_text(json.dumps(protocol,indent=2)+"\n")
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    previous_argv = sys.argv
    try:
        sys.argv = sys.argv[:1]
        app = SimulationApp({"headless": True, "active_gpu": args.gpu, "physics_gpu": args.gpu,
            "multi_gpu": False, "width": 320, "height": 240,
            "extra_args": ["--ext-folder", str(ROOT/".deps/isaacsim-extension-cache")]},
            experience=str(ROOT/"configs/isaac_sim_cutting.kit"))
    finally:
        sys.argv = previous_argv
    result = {"status": "FAIL", "physical_grasp_validated": False}
    try:
        import numpy as np
        import omni.timeline
        import omni.usd
        from pxr import Gf, UsdGeom, UsdPhysics
        from isaacsim.core.experimental.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from robotic_cutting_sim.physical_contacts import NativeToolContacts
        context = omni.usd.get_context(); context.new_stage()
        stage = context.get_stage(); UsdGeom.SetStageMetersPerUnit(stage, 1.); UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.Xform.Define(stage, "/World")
        SimulationManager.setup_simulation(dt=.002)
        SimulationManager.enable_gpu_dynamics(False)
        SimulationManager.enable_fabric(True)

        def box(path, center, dimensions, *, dynamic=False, angle=0.):
            cube=UsdGeom.Cube.Define(stage,path); cube.CreateSizeAttr(1.)
            cube.AddTranslateOp().Set(Gf.Vec3d(*center))
            if angle: cube.AddRotateYOp().Set(angle)
            cube.AddScaleOp().Set(Gf.Vec3f(*dimensions))
            prim=cube.GetPrim(); UsdPhysics.CollisionAPI.Apply(prim)
            if dynamic:
                UsdPhysics.RigidBodyAPI.Apply(prim)
                UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.1)
            return prim

        box("/World/Table", (0,0,-.01), (.3,.3,.02))
        box("/World/FloorTool", (0,0,.12), (.02,.02,.02),dynamic=True)
        box("/World/Cradle/Left", (.95,0,0), (.14,.12,.01),angle=45)
        box("/World/Cradle/Right", (1.05,0,0), (.14,.12,.01),angle=-45)
        box("/World/CradleTool", (1,0,.12), (.02,.02,.02),dynamic=True)
        box("/World/FreeTool", (2,0,1.), (.02,.02,.02),dynamic=True)
        fingers={"left":"/World/Cradle/Left","right":"/World/Cradle/Right"}
        floor=NativeToolContacts(stage,"/World/FloorTool",fingers,"/World/UnusedHand",.002)
        cradle=NativeToolContacts(stage,"/World/CradleTool",fingers,"/World/UnusedHand",.002)
        bodies=RigidPrim(["/World/FloorTool","/World/CradleTool","/World/FreeTool"])
        timeline=omni.timeline.get_timeline_interface(); timeline.play()
        SimulationManager.initialize_physics()
        initial_native_time=SimulationManager.get_simulation_time()
        rows=[]
        for i in range(350):
            bodies.apply_forces_and_torques_at_pos(forces=np.array([[.2,0,0],[0,0,0],[0,0,0]],np.float32),local_frame=False)
            SimulationManager.step(steps=1,update_fabric=True)
            positions,orientations=bodies.get_world_poses()
            rows.append({"step":i+1,"time_s":SimulationManager.get_simulation_time(),"positions":positions.numpy().tolist(),
                "floor":floor.sample(),"cradle":cradle.sample()})
        (output/"trace.json").write_text(json.dumps(rows,indent=2)+"\n")
        settled=rows[-50:]
        floor_force=float(np.mean([r["floor"]["sensor_force_magnitude_n"] for r in settled]))
        bilateral=sum(r["cradle"]["bilateral_tool_contact"] for r in rows)
        drop=float(rows[0]["positions"][2][2]-rows[-1]["positions"][2][2])
        friction_x=float(np.mean([sum(c["force_world_on_tool_n"][0] for c in r["floor"]["tensor_forces"]["records"] if c["kind"]=="friction") for r in settled]))
        tensor_floor=np.mean([r["floor"]["tensor_forces"]["total_filtered_force_world_on_tool_n"] for r in settled],axis=0)
        checks={"finite_raw_force":all(np.isfinite(c["raw_force_world_n"]).all() for r in rows for name in ("floor","cradle") for c in r[name]["contacts"]),
            "floor_weight":abs(floor_force-.981)/.981 <= .05,"bilateral_actual_cradle_contact":bilateral>=20,
            "unsupported_gravity_motion":drop>=.1,
            "floor_friction_balances_external_lateral_load":abs(friction_x+.2)/.2<=.05}
        result={"status":"PASS_ISOLATED_NATIVE_CONTACT_CONTROL" if all(checks.values()) else "FAIL_ISOLATED_NATIVE_CONTACT_CONTROL",
            "checks":checks,"floor_force_n":floor_force,"expected_weight_n":.981,"fresh_bilateral_cradle_steps":bilateral,
            "free_body_drop_m":drop,"steps":350,"simulation_s":SimulationManager.get_simulation_time(),
            "initial_native_time_s":initial_native_time,"stepped_simulation_s":SimulationManager.get_simulation_time()-initial_native_time,
            "floor_friction_x_n":friction_x,"floor_total_contact_force_world_n":tensor_floor.tolist(),
            "raw_contact_times_minus_native_s":sorted(set(round(c["time_s"]-r["time_s"],9) for r in rows for name in ("floor","cradle") for c in r[name]["contacts"])),
            "physical_grasp_validated":False,"scope":protocol["scope"]}
        floor.close();cradle.close();timeline.stop()
    except BaseException as error:
        import traceback
        (output/"failure.txt").write_text(traceback.format_exc());result["error"]=str(error)
    finally:
        (output/"result.json").write_text(json.dumps(result,indent=2)+"\n")
        app.close(wait_for_replicator=False)
    print(json.dumps(result))
    return 0 if result["status"].startswith("PASS_") else 1


if __name__ == "__main__":
    raise SystemExit(main())
