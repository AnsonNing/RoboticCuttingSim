"""Measured-hand wrist cameras and a fixed official D455 world camera.

The D405 housing is a dimensioned visual prototype. Its pinhole geometry is not
a calibrated hardware D405 or a validated D405 noise/occlusion model. All camera
mounts are visual-only and introduce no rigid bodies, contacts, mass or inertia.
Matrices act on column vectors; camera frames use OpenCV +Z forward, +Y down.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


WRIST_LAYOUT = "wrist_d405_global_d455"
LEGACY_LAYOUT = "legacy_fixed_d455"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs/wrist_camera_rig_v5.json"
RIG_PROFILES = ('wrist_camera_rig_v5', 'wrist_camera_rig_v6', 'wrist_camera_rig_v7', 'wrist_camera_rig_v8')
_CV_TO_USD = np.diag([1., -1., -1., 1.])


def pose_matrix(pose_xyzw):
    pose = np.asarray(pose_xyzw, float)
    if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-9:
        raise ValueError("A hand pose must be finite XYZ+XYZW with a nonzero quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(pose[3:]).as_matrix()
    result[:3, 3] = pose[:3]
    return result


def _look_at(eye, target, up=(0., 0., 1.)):
    eye, target, up = map(lambda v: np.asarray(v, float), (eye, target, up))
    forward = target-eye
    if np.linalg.norm(forward) < 1e-9:
        raise ValueError("Camera eye and target coincide")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-8:
        up = np.array([0., 1., 0.])
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    result = np.eye(4)
    result[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    result[:3, 3] = eye
    return result


def _intrinsics(width=848, height=480, hfov_deg=87.):
    focal = width/(2*np.tan(np.deg2rad(hfov_deg)/2))
    return [[focal, 0., width/2], [0., focal, height/2], [0., 0., 1.]]


def _bracket_obb(transform, anchor):
    rear = transform[:3, 3] - .020*transform[:3, 2]
    anchor = np.asarray(anchor, float)
    axis = rear-anchor
    length = np.linalg.norm(axis)
    frame = _look_at((rear+anchor)/2, rear, up=(0., 0., 1.))
    return {"name": "Bracket", "T_hand_obb": frame.tolist(),
            "dimensions_m": [.008, .008, float(length+.008)],
            "scope": "Visual bracket; no contact or mass model"}


class CameraRig:
    def __init__(self, task_id="TRIM-01", camera_layout=WRIST_LAYOUT,
                 initial_hand_poses_world_xyzw=None, rig_profile='wrist_camera_rig_v5'):
        if rig_profile not in RIG_PROFILES:
            raise ValueError(f'Unknown camera rig profile: {rig_profile}')
        if camera_layout in (LEGACY_LAYOUT, 'fixed_d455') and rig_profile != RIG_PROFILES[0]:
            raise ValueError('The versioned Global rig requires the wrist/global layout')
        if camera_layout == "fixed_d455":
            camera_layout = LEGACY_LAYOUT
        if camera_layout not in (WRIST_LAYOUT, LEGACY_LAYOUT):
            raise ValueError(f"Unknown camera layout: {camera_layout}")
        self.task_id, self.camera_layout = str(task_id), camera_layout
        hands = None
        if initial_hand_poses_world_xyzw is not None:
            hands = np.asarray(initial_hand_poses_world_xyzw, float)
            if hands.shape != (2, 7):
                raise ValueError("Expected two measured XYZ+XYZW hand poses")
            for pose in hands:
                pose_matrix(pose)
        self.specs = {}
        if camera_layout == LEGACY_LAYOUT:
            # Legacy stages and sensor initialization remain owned by the caller.
            return
        for index in (0, 1):
            # Both cameras use the same hardware mount, parallel to hand axes.
            # No wrist camera aims at a world target or nominal cutting edge.
            mount = np.eye(4)
            mount[:3, 3] = [-.085, .045, .020]
            housing = np.eye(4)
            # The RGB/depth optical origin is the left imager. A nominal 18mm
            # stereo baseline places the housing center 9mm to its right.
            housing[:3, 3] = [.009, 0., -.0117]
            housing_hand = mount @ housing
            obbs = [{"name": "Housing", "T_hand_obb": housing_hand.tolist(),
                     "T_optical_cv_obb": housing.tolist(), "dimensions_m": [.042, .042, .023],
                     "scope": "Dimensioned D405 prototype housing; no contact or mass model"},
                    _bracket_obb(mount, [-.040, 0., .003])]
            self.specs[f"wrist_d405_{index}"] = {
                "camera_id": f"wrist_d405_{index}", "model": "RealSense D405 geometry prototype",
                "sensor_model": "ideal_d405_prototype", "capability": "ideal_d405_prototype",
                "d405_sensor_model_validated": False, "hardware_calibration_validated": False,
                "parent_hand_index": index, "arm_index": index, "width": 848, "height": 480,
                "near_m": .07, "far_m": .50, "K": _intrinsics(),
                "render_near_m": .001, "render_far_m": 20.,
                "T_hand_optical_cv": mount.tolist(), "T_hand_camera": mount.tolist(),
                "physical_dimensions_m": {"width": .042, "height": .042, "depth": .023},
                "rig_profile": "wrist_camera_rig_v5",
                "visibility_role": "optional_parked_wrist" if index == 1 else "cutting_wrist",
                "dimensions_source": "Official D400 datasheet Table 3-54: width42mm, height42mm, depth23mm",
                "visual_obbs": obbs, "rgb_depth_optical_transform": np.eye(4).tolist(),
                "T_left_right_optical_cv": [[1., 0., 0., .018], [0., 1., 0., 0.],
                                             [0., 0., 1., 0.], [0., 0., 0., 1.]],
                "stereo_baseline_m": .018,
                "stereo_scope": "Right optical reference only; no second imager, disparity or stereo noise implementation",
                "nominal_product_fov_deg": [87., 58.],
                "effective_pinhole_fov_deg": [87., float(np.rad2deg(2*np.arctan(480/(2*_intrinsics()[1][1]))))],
                "rgb_source": "D405 left color depth imager with onboard ISP; co-located ideal optical frames",
                "calibration_scope": "848x480 square-pixel pinhole with nominal 87 degree HFOV and effective 56.5 degree VFOV; no factory intrinsics or physical calibration",
                "minimum_z_scope": "Official 70mm MinZ mode at 848x480; prototype optical range only, no validated stereo matching",
                "mount_scope": "Fixed hand-local transform; camera follows actual measured hand, without a gimbal",
                "physics_scope": "Visual-only housing/bracket; no colliders, mass or inertia; clearance unvalidated",
                "initial_hand_pose_world_xyzw": None if hands is None else hands[index].tolist(),
            }
        global_target = np.array([0., 0., .020])
        transform = _look_at([.28, -.42, .46], global_target)
        self.specs["global_d455"] = {
            "camera_id": "global_d455", "model": "Official Isaac Sim 6.2 RealSense D455 asset",
            "rig_profile": "wrist_camera_rig_v5",
            "view_goal": "World-fixed close view of the workpiece, knife and cutting region",
            "look_at_target_world_m": global_target.tolist(),
            "working_distance_to_target_m": float(np.linalg.norm(transform[:3, 3]-global_target)),
            "whole_workcell_coverage_required": False,
            "sensor_model": "native_single_view_d455_provisional", "capability": "native_single_view_d455_provisional",
            "d455_hardware_calibration_validated": False, "parent_hand_index": None, "arm_index": None,
            "width": 640, "height": 480, "near_m": .60, "far_m": 6.,
            "native_near_m": .52, "native_far_m": 6., "render_near_m": .001, "render_far_m": 20.,
            "K": [[640*1.929999947/3.895999908, 0., 320.],
                  [0., 480*1.929999947/2.453000068, 240.], [0., 0., 1.]],
            "T_world_camera": transform.tolist(),
            "physical_dimensions_m": {"width": .124, "height": .029, "depth": .025},
            "physics_scope": "World-fixed visual-only official housing; rigid-body/collision/mass APIs disabled",
            "calibration_scope": "Authored official USD pseudo-depth lens; runtime render-product K must be recorded independently",
            "visual_obbs": [],
        }
        if rig_profile == 'wrist_camera_rig_v6':
            path = CONFIG_PATH.with_name(rig_profile+'.json')
            config = json.loads(path.read_text())
            view = config['global_d455']
            transform = _look_at(view['eye_world_m'], view['target_world_m'])
            width, height = view['resolution_wh']
            self.specs['global_d455'].update(
                rig_profile=rig_profile, width=width, height=height,
                K=_intrinsics(width,height,view['hfov_deg']),
                T_world_camera=transform.tolist(), look_at_target_world_m=view['target_world_m'],
                working_distance_to_target_m=float(np.linalg.norm(transform[:3,3]-view['target_world_m'])),
                near_m=view['near_m'], far_m=view['far_m'],
                native_near_m=view['native_near_m'], native_far_m=view['far_m'],
                native_maximum_disparity=view['maximum_disparity_px'],
                calibration_scope='Versioned square-pixel simulated D455 lens; actual render-product K recorded; physical calibration unvalidated')
        elif rig_profile in ('wrist_camera_rig_v7','wrist_camera_rig_v8'):
            config=json.loads(CONFIG_PATH.with_name(rig_profile+'.json').read_text())['global_d455']
            transform=_look_at(config['eye_world_m'], config['target_world_m'])
            self.specs['global_d455'].update(
                rig_profile=rig_profile,T_world_camera=transform.tolist(),
                look_at_target_world_m=config['target_world_m'],
                working_distance_to_target_m=float(np.linalg.norm(transform[:3,3]-config['target_world_m'])),
                optical_profile='unchanged validated v5 SingleView 640x480')

    def calibrations(self, hand_poses_world_xyzw):
        hands = np.asarray(hand_poses_world_xyzw, float)
        if hands.shape != (2, 7):
            raise ValueError("Expected two measured XYZ+XYZW hand poses")
        transforms = [pose_matrix(pose) for pose in hands]
        result = {}
        for name, spec in self.specs.items():
            index = spec["parent_hand_index"]
            transform = (np.asarray(spec["T_world_camera"], float) if index is None
                         else transforms[index] @ np.asarray(spec["T_hand_optical_cv"], float))
            result[name] = {**deepcopy(spec), "T_world_camera": transform.tolist(),
                            "convention": "OpenCV +Z forward, +Y down", "depth": "Optical Z, meters; invalid=0"}
        return result

    def author_usd(self, stage, hand_prim_paths, *, global_d455_asset_path=None):
        from pxr import Gf, Usd, UsdGeom
        from .usd_surface_bridge import usd_camera_calibration
        if self.camera_layout == LEGACY_LAYOUT:
            return {}
        if len(hand_prim_paths) != 2 or any(not stage.GetPrimAtPath(p) for p in hand_prim_paths):
            raise ValueError("Two existing measured Panda hand USD prim paths are required")
        if not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1.):
            raise ValueError("Camera rig requires an SI USD stage")
        for path in ("/World/D455_0", "/World/D455_1", "/World/GlobalCamera"):
            prim = stage.GetPrimAtPath(path)
            if prim:
                prim.SetActive(False)
        for index, hand_path in enumerate(hand_prim_paths):
            spec = self.specs[f"wrist_d405_{index}"]
            root = str(hand_path) + f"/WristD405_{index}"
            mount = np.asarray(spec["T_hand_optical_cv"])
            _set_transform(UsdGeom.Xform.Define(stage, root).GetPrim(), mount)
            for name in ("DepthOptical", "ColorOptical"):
                camera = UsdGeom.Camera.Define(stage, root+"/"+name)
                _set_transform(camera.GetPrim(), _CV_TO_USD)
                camera.CreateProjectionAttr("perspective")
                # Actual Kit RTX ignores RenderProduct pixelAspectRatio and
                # enforces square pixels. Use an official 848x480 MinZ mode,
                # retaining its actual effective VFOV rather than claiming58deg.
                camera.CreateFocalLengthAttr(10.)
                camera.CreateHorizontalApertureAttr(float(20*np.tan(np.deg2rad(87.)/2)))
                camera.CreateVerticalApertureAttr(float(20*np.tan(np.deg2rad(87.)/2)*480/848))
                # Keep close grippers/brackets as real occluders and retain RGB
                # outside the logical depth range. Sensor validity is filtered
                # after first-hit rendering, never by hiding near geometry.
                camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 20.))
            right_reference = np.asarray(spec["T_left_right_optical_cv"]) @ _CV_TO_USD
            _set_transform(UsdGeom.Xform.Define(stage, root+"/RightOpticalReference").GetPrim(), right_reference)
            for obb in spec["visual_obbs"]:
                box_path = root+"/"+obb["name"]
                local = np.linalg.inv(mount) @ np.asarray(obb["T_hand_obb"])
                box = UsdGeom.Cube.Define(stage, box_path)
                box.CreateSizeAttr(1.)
                _set_transform(box.GetPrim(), local, scale=obb["dimensions_m"])
                box.CreateDisplayColorAttr([Gf.Vec3f(.12, .13, .15) if obb["name"] == "Housing"
                                           else Gf.Vec3f(.28, .29, .31)])
                obb["prim_path"] = box_path
            spec.update(prim_path=root, parent_hand_prim_path=str(hand_path),
                        optical_prim_path=root+"/DepthOptical", depth_camera_path=root+"/DepthOptical",
                        color_camera_path=root+"/ColorOptical", right_optical_reference_path=root+"/RightOpticalReference")
        asset = str(global_d455_asset_path or "")
        if not asset:
            raise ValueError("The fixed global camera requires the cached official D455 USD asset")
        root = "/World/WristCameraRig/GlobalD455"
        UsdGeom.Xform.Define(stage, "/World/WristCameraRig")
        prim = UsdGeom.Xform.Define(stage, root).GetPrim()
        prim.GetReferences().AddReference(asset, "/Root")
        depth_path = root+"/RSD455/Camera_Pseudo_Depth"
        color_path = root+"/RSD455/Camera_OmniVision_OV9782_Color"
        optical = stage.GetPrimAtPath(depth_path)
        if not optical or not stage.GetPrimAtPath(color_path):
            raise ValueError("Official D455 reference lacks expected depth/color optical prims")
        optical_in_root = np.asarray(UsdGeom.Xformable(optical).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        root_world = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        optical_in_root = np.linalg.inv(root_world) @ optical_in_root
        global_spec = self.specs["global_d455"]
        _set_transform(prim, np.asarray(global_spec["T_world_camera"]) @ _CV_TO_USD @ np.linalg.inv(optical_in_root))
        for path in (depth_path, color_path):
            camera = UsdGeom.Camera.Get(stage, path)
            camera.CreateProjectionAttr("perspective")
            camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 20.))
            if global_spec['rig_profile'] == 'wrist_camera_rig_v6':
                camera.CreateFocalLengthAttr(10.)
                camera.CreateHorizontalApertureAttr(10.*global_spec['width']/global_spec['K'][0][0])
                camera.CreateVerticalApertureAttr(10.*global_spec['height']/global_spec['K'][1][1])
        disabled = _disable_physics_and_products(stage, prim)
        authored = usd_camera_calibration(stage, depth_path, global_spec['width'], global_spec['height'])
        global_spec.update(prim_path=root, optical_prim_path=depth_path, depth_camera_path=depth_path,
                           color_camera_path=color_path, asset=asset, K=authored["K"],
                           T_world_camera=authored["T_world_camera"], disabled_source_apis=disabled)
        # Measure the composed official visual envelope in its actual body
        # frame, keeping its transform rather than assuming optical/body origins.
        body = stage.GetPrimAtPath(root+"/RSD455")
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"],
                                  useExtentsHint=False).ComputeUntransformedBound(body).ComputeAlignedRange()
        minimum, maximum = np.asarray(bounds.GetMin()), np.asarray(bounds.GetMax())
        obb_world = np.asarray(UsdGeom.Xformable(body).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        center = np.eye(4)
        center[:3, 3] = (minimum+maximum)/2
        obb_world = obb_world @ center
        obb_cv = np.linalg.inv(np.asarray(authored["T_world_camera"])) @ obb_world
        global_spec["visual_obbs"] = [{"name": "OfficialHousingConservativeOBB", "prim_path": root+"/RSD455",
            "T_optical_cv_obb": obb_cv.tolist(), "T_world_obb": obb_world.tolist(),
            "dimensions_m": (maximum-minimum).tolist(), "scope": "Composed official visual body local bounding box; no native contact claim"}]
        return deepcopy(self.specs)


def _set_transform(prim, transform, scale=None):
    from pxr import Gf, UsdGeom
    transform = np.asarray(transform, float)
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(transform.T.tolist()))
    if scale is not None:
        xform.AddScaleOp().Set(Gf.Vec3f(*map(float, scale)))


def _disable_physics_and_products(stage, root):
    from pxr import Usd
    disabled = []
    for prim in list(Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
        if prim.GetTypeName() in ("RenderProduct", "RenderSettings", "RenderVar"):
            prim.SetActive(False)
            disabled.append({"prim_path": str(prim.GetPath()), "action": "deactivate_source_render_prim"})
            continue
        for schema in list(prim.GetAppliedSchemas()):
            if "Physics" in schema or "Physx" in schema:
                prim.RemoveAPI(schema)
                disabled.append({"prim_path": str(prim.GetPath()), "removed_api": schema})
        for attr_name in ("physics:rigidBodyEnabled", "physics:collisionEnabled"):
            attr = prim.GetAttribute(attr_name)
            if attr:
                attr.Set(False)
    return disabled


def make_rig(task_id="TRIM-01", camera_layout=WRIST_LAYOUT, initial_hand_poses_world_xyzw=None,
             rig_profile='wrist_camera_rig_v5'):
    return CameraRig(task_id, camera_layout, initial_hand_poses_world_xyzw, rig_profile)


def author_camera_rig(stage, hand_paths, d455_asset, task_id="TRIM-01", *,
                      camera_layout=WRIST_LAYOUT, initial_hand_poses_world_xyzw=None,
                      rig_profile='wrist_camera_rig_v5'):
    if initial_hand_poses_world_xyzw is None:
        from pxr import Usd, UsdGeom
        poses = []
        for path in hand_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise ValueError(f"Missing measured hand prim: {path}")
            transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            poses.append(np.r_[transform[:3, 3], Rotation.from_matrix(transform[:3, :3]).as_quat()])
        initial_hand_poses_world_xyzw = poses
    rig = make_rig(task_id, camera_layout, initial_hand_poses_world_xyzw, rig_profile)
    return rig.author_usd(stage, hand_paths, global_d455_asset_path=d455_asset)
