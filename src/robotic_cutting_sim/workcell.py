"""Static research workcell with official asset references, without robot control.

The default asset cache preserves the official relative USD hierarchy. Downloads
are bounded and kept under ignored artifacts; vendor assets are never packaged.
"""
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

from .recording import ROOT, manifest, write_json


ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2"
ASSETS = {
    "franka": "Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda",
    "d455": "Isaac/Sensors/RealSense/D455/rsd455.usd",
}
PARK_JOINTS = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, np.pi / 4]
USD_SUFFIXES = (".usd", ".usda", ".usdc")


def _cache_assets(output):
    """Fetch only USD composition dependencies, at most 150 MiB and 40 files."""
    from pxr import UsdUtils

    cache = ROOT / "artifacts/workcell_asset_cache"
    pending, seen, records, total = list(ASSETS.values()), set(), [], 0
    while pending:
        relative = pending.pop(0)
        if relative in seen:
            continue
        seen.add(relative)
        path, url = cache / relative, f"{ASSET_ROOT}/{relative}"
        record = {"url": url, "relative_path": relative}
        try:
            if len(seen) > 40:
                raise RuntimeError("USD dependency count exceeds the 40-file limit")
            remaining = 150 * 1024**2 - total
            if path.is_file():
                if path.stat().st_size > remaining:
                    raise RuntimeError("USD composition cache exceeds the 150 MiB limit")
                data = path.read_bytes()
                record["source"] = "existing_local_cache"
            else:
                with urllib.request.urlopen(url, timeout=20) as response:
                    size = int(response.headers.get("Content-Length") or 0)
                    if size > remaining:
                        raise RuntimeError("Asset would exceed the 150 MiB download limit")
                    data = response.read(remaining + 1)
                if len(data) > remaining:
                    raise RuntimeError("Asset exceeded the 150 MiB download limit")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                record["source"] = "official_https"
            total += len(data)
            record.update(status="PASS", bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(path))
            dependencies = list(sublayers) + list(references) + list(payloads)
            record["external_assets"] = dependencies
            for dependency in dependencies:
                if dependency.lower().endswith(USD_SUFFIXES):
                    resolved = urllib.parse.urljoin(url, dependency)
                    if not resolved.startswith(ASSET_ROOT + "/"):
                        raise RuntimeError(f"USD dependency leaves the selected official asset root: {dependency}")
                    pending.append(resolved[len(ASSET_ROOT) + 1:])
        except Exception as error:
            record.update(status="BLOCKED", error=str(error))
        records.append(record)
        write_json(output / "asset_fetch.json", {
            "asset_root": ASSET_ROOT, "total_bytes": total, "files": records,
            "scope": "USD composition only; MDL shader packages and renderer are not validated",
        })
    return cache


def _local_composition(path):
    """Inspect local layers without invoking a network asset resolver."""
    from pxr import Sdf, UsdUtils

    pending, seen, missing, shaders = [Path(path)], set(), [], set()
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        if len(seen) > 100:
            missing.append("USD dependency audit stopped at 100 layers")
            break
        if not current.is_file():
            missing.append(str(current))
            continue
        layer = Sdf.Layer.FindOrOpen(str(current))
        if layer is None:
            missing.append(f"Unreadable USD layer: {current}")
            continue
        sublayers, references, payloads = UsdUtils.ExtractExternalReferences(str(current))
        for dependency in list(sublayers) + list(references) + list(payloads):
            if dependency.lower().endswith(USD_SUFFIXES):
                if "://" in dependency:
                    missing.append(f"Remote USD dependency not locally validated: {dependency}")
                else:
                    pending.append((current.parent / dependency).resolve())
            else:
                shaders.add(dependency)
    return {"status": "PASS" if not missing else "BLOCKED", "layers_checked": len(seen),
            "missing_usd_dependencies": sorted(set(missing)),
            "external_shader_assets_not_validated": sorted(shaders)}


def export(args, output):
    from pxr import Gf, Usd, UsdGeom
    from scipy.spatial.transform import Rotation

    from . import scene
    from .camera import calibration

    output = Path(output)
    base_output = output / "physics_preview"
    base_output.mkdir()
    scene.export(args, base_output)
    source_stage = Usd.Stage.Open(str(base_output / "physics_testbed.usda"))
    path = output / "workcell.usda"
    source_stage.GetRootLayer().Export(str(path))
    stage = Usd.Stage.Open(str(path))

    override = getattr(args, "asset_root", None)
    selected_root = str(override).rstrip("/") if override else str(_cache_assets(output).resolve())
    remote = "://" in selected_root
    if not remote:
        selected_root = str(Path(selected_root).expanduser().resolve())
    sources, checks = {}, {}
    for name, relative in ASSETS.items():
        sources[name] = selected_root + "/" + relative
        checks[name] = ({"status": "NOT_RUN", "reason": "Remote composition requires the Isaac asset resolver"}
                        if remote else _local_composition(sources[name]))

    def pose(prim, transform):
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*transform[:3, 3]))
        q = Rotation.from_matrix(transform[:3, :3]).as_quat()
        xform.AddOrientOp().Set(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))

    # A 1.8 x 1.2 m table gives the two static robot bases adequate support.
    table = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Table"))
    for op in table.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            op.Set(Gf.Vec3d(0, 0, -.035))
        elif op.GetOpType() == UsdGeom.XformOp.TypeScale:
            op.Set(Gf.Vec3f(.9, .6, .025))
    # Remove duplicate ideal views; the workcell exposes actual D455 optical prims.
    for name in ("IdealCamera0", "IdealCamera1", "GlobalCamera"):
        stage.RemovePrim("/World/" + name)

    robots = {}
    for index, (position, yaw) in enumerate([([-.46, .12, -.01], 0), ([.46, .12, -.01], np.pi)]):
        prim_path = f"/World/Franka_{index}"
        robot = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        robot.GetReferences().AddReference(sources["franka"], "/panda")
        # Geometry and authored joint definitions only; no PhysX/IK initialization.
        robot.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("none")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
        transform[:3, 3] = position
        pose(robot, transform)
        robot.SetCustomDataByKey("research:role", "cutting_arm" if index == 0 else "parked_secondary_arm")
        robots[f"franka_{index}"] = {"prim_path": prim_path, "asset": sources["franka"],
            "position_m": position, "yaw_rad": yaw,
            "nominal_park_joint_names": [f"panda_joint{i}" for i in range(1, 8)],
            "nominal_park_joint_positions_rad": PARK_JOINTS,
            "nominal_finger_positions_m": [.04, .04], "nominal_park_applied": False,
            "display_pose": "official asset authored link transforms", "physics_variant": "none"}

    # Measure the source camera frame: D455's body +X is its forward optical axis.
    # Reusing a USD camera orientation directly on the housing would point it away.
    d455_inspection = (Path(sources["d455"]) if not remote else
                       ROOT / "artifacts/assets_audit_002/d455.usd")
    optical_in_asset = None
    native_lens = {}
    if d455_inspection.is_file():
        sensor_stage = Usd.Stage.Open(str(d455_inspection))
        optical = sensor_stage.GetPrimAtPath("/Root/RSD455/Camera_Pseudo_Depth")
        if optical:
            optical_in_asset = np.asarray(UsdGeom.Xformable(optical).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())).T
            camera = UsdGeom.Camera(optical)
            native_lens = {"horizontal_aperture_mm": float(camera.GetHorizontalApertureAttr().Get()),
                           "vertical_aperture_mm": float(camera.GetVerticalApertureAttr().Get()),
                           "focal_length_mm": float(camera.GetFocalLengthAttr().Get()),
                           "scope": "authored USD lens; render-product resolution and depth pipeline NOT_RUN"}
    if optical_in_asset is None:
        # Canonical 6.2 asset transform, retained for remote asset-root operation.
        optical_in_asset = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)

    cameras = {}
    for index, eye in enumerate(([0, -.5, .55], [.5, -.4, .25])):
        name = f"d455_{index}"
        camera_spec = calibration(eye)
        cv = np.array(camera_spec["T_world_camera"])
        target_camera = cv.copy()
        target_camera[:3, :3] = cv[:3, :3] @ np.diag([1, -1, -1])
        mount = target_camera @ np.linalg.inv(optical_in_asset)
        prim_path = f"/World/D455_{index}"
        sensor = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
        sensor.GetReferences().AddReference(sources["d455"], "/Root")
        pose(sensor, mount)
        cameras[name] = {**camera_spec, "asset": sources["d455"], "prim_path": prim_path,
            "depth_camera_path": prim_path + "/RSD455/Camera_Pseudo_Depth",
            "color_camera_path": prim_path + "/RSD455/Camera_OmniVision_OV9782_Color",
            "T_world_asset": mount.tolist(),
            "distance_to_material_target_m": float(np.linalg.norm(np.array(eye) - [0, 0, .015])),
            "calibration_scope": "nominal pinhole target frame; native D455 intrinsics retained in USD",
            "native_depth_camera_lens": native_lens,
            "sensor_output": "NOT_RUN"}

    global_calibration = calibration([1.65, -1.75, 1.5], target=(0, 0, .4), hfov_deg=65)
    global_transform = np.array(global_calibration["T_world_camera"])
    global_transform[:3, :3] = global_transform[:3, :3] @ np.diag([1, -1, -1])
    global_camera = UsdGeom.Camera.Define(stage, "/World/GlobalCamera")
    pose(global_camera.GetPrim(), global_transform)
    global_camera.CreateHorizontalApertureAttr(20.955)
    global_camera.CreateVerticalApertureAttr(20.955 * 480 / 640)
    global_camera.CreateFocalLengthAttr(float(20.955 / (2 * np.tan(np.deg2rad(65) / 2))))
    global_camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 20))
    cameras["global"] = {**global_calibration, "prim_path": "/World/GlobalCamera"}

    local_composed = all(check["status"] == "PASS" for check in checks.values())
    capabilities = {"static_workcell_authored": True, "official_assets_composed": local_composed,
        "robot_articulation_validated": False, "ik_validated": False, "robot_tool_coupling_validated": False,
        "robot_force_feedback_validated": False, "rtx_rgbd_validated": False,
        "physical_cutting_validated": False, "physics_proven": False, "vision_proven": False,
        "workcell_proven": False, "benchmark_ready": False}
    spec = {"status": "STATIC_RESEARCH_WORKCELL", "units": "m,kg,s; quaternion XYZW; world Z up",
        "scene": str(path.resolve()), "asset_root": selected_root, "asset_root_override": override,
        "canonical_official_asset_root": ASSET_ROOT, "asset_composition": checks,
        "robots": robots, "cameras": cameras, "capabilities": capabilities,
        "material": json.loads((base_output / "specimen.json").read_text()),
        "tool": {"prim_path": "/World/KinematicBlade", "representation": "detached box proxy",
                 "attached_to_robot": False},
        "limitations": ["Static composition does not initialize robot articulation or apply nominal park joints",
            "D455 native camera prims are composed; no sensor annotators or RTX outputs were validated",
            "External MDL shader packages are not downloaded or validated by the bounded USD cache",
            "Opening the USD does not advance Newton MPM; use the physics or environment commands"]}
    stage.GetRootLayer().customLayerData = {"status": "STATIC_RESEARCH_WORKCELL",
        "robot_control": "NOT_RUN", "rtx_rgbd": "NOT_RUN", "cutting": "NOT_PROVEN",
        "material": "uncalibrated", "scene_spec": "scene_spec.json"}
    stage.GetRootLayer().Save()
    topology = {"status": "NOT_RUN", "scope": "static composition and transforms only"}
    if local_composed:
        topology["status"] = "PASS"
        for index in (0, 1):
            arm = stage.GetPrimAtPath(f"/World/Franka_{index}")
            meshes = [p for p in Usd.PrimRange(arm, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
            if len(meshes) < 10:
                raise RuntimeError(f"Franka_{index} has only {len(meshes)} composed mesh prims")
            topology[f"franka_{index}_mesh_prims"] = len(meshes)
            camera = stage.GetPrimAtPath(cameras[f"d455_{index}"]["depth_camera_path"])
            if not camera.IsA(UsdGeom.Camera):
                raise RuntimeError(f"D455_{index} native depth camera did not compose")
            transform = np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())).T
            toward = np.array([0, 0, .015]) - transform[:3, 3]
            toward /= np.linalg.norm(toward)
            alignment = float(np.dot(-transform[:3, 2], toward))
            if alignment < .99999:
                raise RuntimeError(f"D455_{index} optical axis points away from the material")
            topology[f"d455_{index}_forward_alignment"] = alignment
    write_json(output / "static_validation.json", topology)
    write_json(output / "scene_spec.json", spec)
    write_json(output / "cameras.json", cameras)
    write_json(output / "manifest.json", manifest({"scene_spec": spec}, "static_usd_workcell"))
    result = {"status": "PASS" if local_composed else "BLOCKED",
        "scope": "static research workcell; no robot control or cutting acceptance",
        "scene": str(path.resolve()), "asset_composition": checks, **capabilities}
    write_json(output / "result.json", result)
    return result
