#!/usr/bin/env python3
"""Optional native tool skill using the retained actual rigid/contact loop.

Starts pregrasped, places the independent tool on a static fork rack, opens,
withdraws, approaches the measured resting tool, closes, lifts and holds. No
tool pose is written after reset and TLMPM is never advanced in this probe.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.scene_frames import matrix_pose, pose_matrix
from run_physical_tool_controls import material_fingerprint, smooth_joint_targets


def author_rack(stage, spec, center):
    """A static fork supports the handle end face; the shaft fits its open slot.

    This first rack supports the nominal downward TOOL-Z mounting. Different
    handle shapes/orientations require their own rack-clearance validation.
    """
    from pxr import Gf, UsdGeom, UsdPhysics, UsdShade
    hx, hy, hz = spec.handle_half_extents_m
    if hx < .007 or hy < .01:
        raise NotImplementedError("This fork requires the verified straight-handle dimensions")
    x, y, z = center
    top = z-hz
    material = UsdShade.Material.Define(stage, "/World/ToolRack/Material")
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(.6); physics.CreateDynamicFrictionAttr(.6)
    physics.CreateRestitutionAttr(0.)
    boxes = [("Base", (x,y,.005), (.03,.03,.005))]
    # Inner9mm gap clears the actual6mm-wide shaft at the mating face.
    for side in (-1,1):
        boxes.append(("Rail"+str(side), (x+side*.00625,y,(top+.01)/2), (.00175,.015,(top-.01)/2)))
    for name, position, half in boxes:
        cube = UsdGeom.Cube.Define(stage, "/World/ToolRack/"+name)
        cube.CreateSizeAttr(2.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*position)); cube.AddScaleOp().Set(Gf.Vec3d(*half))
        cube.CreateDisplayColorAttr([Gf.Vec3f(.18,.28,.34)])
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material,materialPurpose="physics")
    return {"root":"/World/ToolRack", "handle_center_world_m":list(center), "support_top_world_z_m":top,
            "shaft_slot_width_m":.009, "static_collision":True, "tool_attachment":False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--gpu",type=int,default=0)
    p.add_argument("--protocol",type=Path,default=ROOT/"configs/tool_rack_demo_v1.json")
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    out = new_output(args.output_dir)
    write_json(out/"protocol.json",protocol)
    implementation = out/"implementation"; implementation.mkdir()
    files = [Path(__file__),ROOT/"scripts/run_physical_tool_controls.py"]
    files += [ROOT/"src/robotic_cutting_sim"/name for name in (
        "isaac_sim_runner.py","native_tool.py","physical_tool.py","physical_contacts.py","grasp_monitor.py","robot.py","live_material.py","tlmpm.py")]
    hashes = {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
    for path in files:shutil.copyfile(path,implementation/path.name)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA","YES")
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, LiveIsaacSimCutting, launch_app
    scene_dir = out/"workcell"
    config = IsaacSimCuttingConfig(task_id="CUT-01", gpu=args.gpu,device=f"cuda:{args.gpu}",
        tool_attachment_mode="physical_grasp",tool_init_mode="pregrasped",
        grasp={"initial_finger_position_m":protocol["initial_finger_position_m"]},
        camera_layout="wrist_d405_global_d455",camera_mode="native_single_view",scene=str((scene_dir/"workcell.usda").resolve()))
    write_json(out/"manifest.json",{**manifest(asdict(config),"wp08_native_rigid_tool_skill"),"implementation_sha256":hashes})
    app, env = None, None
    result = {"status":"NOT_RUN", "full_physics_acceptance":False, "material_cutting":"NOT_RUN_RIGID_SKILL"}
    try:
        app = launch_app(config,out)
        from argparse import Namespace
        from pxr import Usd
        from robotic_cutting_sim.workcell import export
        from robotic_cutting_sim.physical_tool import make_tool_spec
        from robotic_cutting_sim.tasks import make_task
        scene_dir.mkdir()
        export(Namespace(variant="D0",family="dome",seed=42,spacing=.002,asset_root=None),scene_dir)
        stage = Usd.Stage.Open(str(scene_dir/"workcell.usda"))
        spec = make_tool_spec(make_task("CUT-01"))
        rack = author_rack(stage,spec,protocol["rack_handle_center_world_m"])
        stage.GetRootLayer().Save()
        result["rack"] = rack
        env = LiveIsaacSimCutting(app,config,out); env.reset()
        initial_material = material_fingerprint(env)
        state = env.proprioception()
        rows, capture_index = [], 0
        (out/"rgbd").mkdir()

        def capture(phase):
            nonlocal capture_index
            frames,_ = env.capture()
            payload={"timestamp_s":env.time_s}
            for name, values in frames.items():
                for field in ("rgb","depth_m","valid_mask"):payload[name+"_"+field]=values[field]
            np.savez_compressed(out/"rgbd"/f"frame_{capture_index:05d}.npz",**payload)
            write_json(out/"rgbd"/f"calibration_{capture_index:05d}.json",env.calibration)
            write_json(out/"rgbd"/f"phase_{capture_index:05d}.json",{"phase":phase,"timestamp_s":env.time_s})
            capture_index += 1

        def step(target,phase):
            state,_ = env.rigid_probe_step(target)
            row={"phase":phase,"time_s":env.time_s,"commanded_joint_positions_rad":target.copy(),"state":state}
            rows.append(row)
            with (out/"control_trace.jsonl").open("a") as stream:
                stream.write(json.dumps(row,default=lambda v: v.tolist() if isinstance(v,np.ndarray) else v.item())+"\n")
            return state

        def move(edge,phase):
            q, info = env.solve_edge_ik(edge)
            if (info["position_error_m"] > protocol["transfer"]["maximum_ik_position_error_m"]
                    or info["rotation_error_rad"] > protocol["transfer"]["maximum_ik_rotation_error_rad"]):
                raise RuntimeError("Tool skill IK failed: "+str(info))
            start = env.proprioception()["q"][0]
            for target in smooth_joint_targets(start,q,protocol["transfer"],config.control_dt):step(target,phase)
            capture(phase)

        def hold(seconds,phase):
            target = env.proprioception()["q"][0]
            start = len(env.grasp_trace)
            for _ in range(round(seconds/config.control_dt)):step(target,phase)
            capture(phase)
            return env.grasp_trace[start:]

        capture("pregrasped")
        clearance = state["tool_pose"].copy(); clearance[2] = protocol["clearance_edge_height_m"]
        move(clearance,"lift_before_transfer")
        rest_body = np.r_[protocol["rack_handle_center_world_m"], [1.,0.,0.,0.]]
        rest_edge = matrix_pose(pose_matrix(rest_body) @ spec.T_tool_edge)
        over = rest_edge.copy(); over[2] = clearance[2]
        move(over,"transfer_above_rack")
        place = rest_edge.copy(); place[2] += protocol["release_clearance_m"]
        move(place,"place_tool")
        env.set_gripper("release"); hold(protocol["release_wait_s"],"release_onto_rack")
        rested = env.proprioception()["tool_body_pose_xyzw"].copy()
        retract = env.proprioception()["tool_pose"].copy(); retract[2] += protocol["open_retract_m"]
        move(retract,"withdraw_open_hand")
        # Approach the actually resting tool. This changes only an IK target.
        pickup = matrix_pose(pose_matrix(env.proprioception()["tool_body_pose_xyzw"]) @ spec.T_tool_edge)
        move(pickup,"approach_measured_handle")
        env.set_gripper("close"); hold(protocol["close_wait_s"],"close_on_handle")
        before_lift = env.proprioception()["tool_body_pose_xyzw"].copy()
        lift = env.proprioception()["tool_pose"].copy(); lift[2] += protocol["lift_m"]
        move(lift,"lift_tool_from_rack")
        samples = hold(protocol["hold_s"],"hold_reacquired_tool")
        relative = np.array([r["relative_tool_pose_xyzw"] for r in samples])
        drift = np.linalg.norm(relative[:,:3]-relative[0,:3],axis=1).max()
        angle = (Rotation.from_quat(relative[:,3:])*Rotation.from_quat(relative[0,3:]).inv()).magnitude().max()
        bilateral = np.mean([r["contacts"]["bilateral_loaded_handle_contact"] for r in samples])
        penetration = max([max(0.,-v["separation_m"]) for r in samples for v in r["contacts"]["tensor_forces"]["records"] if v.get("separation_m") is not None] or [0.])
        actual = env.proprioception()
        checks={"material_unchanged":material_fingerprint(env)==initial_material,
            "placed_on_rack":np.linalg.norm(rested[:3]-rest_body[:3]) <= protocol["maximum_placement_center_error_m"],
            "actually_lifted":actual["tool_body_pose_xyzw"][2]-before_lift[2] >= protocol["minimum_lift_m"],
            "bilateral_handle_support":bilateral >= protocol["minimum_bilateral_loaded_fraction"],
            "stable_relative_translation":drift <= protocol["maximum_relative_translation_m"],
            "stable_relative_rotation":angle <= protocol["maximum_relative_rotation_rad"],
            "finger_penetration":penetration <= protocol["maximum_grasp_penetration_m"],
            "no_rack_table_or_palm_support_after_lift":all(r["contacts"]["unintended_contact_count"]==0 for r in samples),
            "source_bytes_unchanged":all(hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==sha for path,sha in hashes.items())}
        result.update(status="PASS_NATIVE_TOOL_SKILL_ENGINEERING" if all(checks.values()) else "FAIL_NATIVE_TOOL_SKILL",
            checks={k:bool(v) for k,v in checks.items()},measured_resting_body_pose_xyzw=rested,
            lifted_body_pose_xyzw=actual["tool_body_pose_xyzw"],bilateral_loaded_handle_fraction=float(bilateral),
            hold_translation_drift_m=float(drift),hold_rotation_drift_rad=float(angle),maximum_penetration_m=penetration,
            actual_rigid_steps=len(env.grasp_trace),control_steps=len(rows),camera_frames=capture_index,
            actual_skill_duration_s=env.time_s,task_clock_scope="rigid skill probe, not8second CUT episode",newton_pickup="NOT_RUN")
        env.stage.Export(str(out/"final_live_scene.usda"))
    except Exception as error:
        import traceback
        (out/"failure.log").write_text(traceback.format_exc())
        result.update(status="FAIL_NATIVE_TOOL_SKILL",error=str(error))
    finally:
        write_json(out/"result.json",result)
        if env:env.close()
        elif app:app.close(wait_for_replicator=False)
    print(json.dumps({"status":result["status"],"output":str(out)}))
    return 0 if result["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
