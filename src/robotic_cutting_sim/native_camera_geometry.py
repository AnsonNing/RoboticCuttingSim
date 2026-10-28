"""Measured native-hand camera transforms, independent of the Kit runtime.

All transforms use column vectors and OpenCV optical axes. This module validates
the current rigid hierarchy; it never writes a commanded camera world pose.
"""
import numpy as np
from scipy.spatial.transform import Rotation

from .scene_frames import pose_matrix, validate_transform


def camera_physics_receipt(stage, specs, hand_paths):
    """Reject detached wrist hierarchies or active camera rigid/contact bodies."""
    from pxr import Usd, UsdPhysics
    roots={}
    enabled=[]
    for name,spec in specs.items():
        arm=spec.get("parent_hand_index")
        boundary=hand_paths[arm] if arm is not None else "/World"
        path=spec["optical_prim_path"]
        if not path.startswith(boundary+"/"):
            raise ValueError(f"Optical camera is not a child of its actual parent: {name}: {path}")
        root=stage.GetPrimAtPath(path)
        if not root:
            raise ValueError(f"Missing authored optical camera: {name}")
        while str(root.GetParent().GetPath())!=boundary:
            root=root.GetParent()
            if not root or root.IsPseudoRoot():
                raise ValueError(f"Camera hierarchy did not reach its declared parent: {name}")
        roots[name]=str(root.GetPath())
        for prim in Usd.PrimRange(root):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and UsdPhysics.RigidBodyAPI(prim).GetRigidBodyEnabledAttr().Get():
                enabled.append({"camera":name,"prim":str(prim.GetPath()),"enabled":"RigidBodyAPI"})
            if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
                enabled.append({"camera":name,"prim":str(prim.GetPath()),"enabled":"CollisionAPI"})
    if enabled:
        raise RuntimeError(f"Camera housing/bracket must be visual-only and fixed to its parent: {enabled}")
    return {"camera_roots":roots,"enabled_camera_rigid_or_contact_prims":enabled,
            "camera_mass_and_contact_model":"Visual housing/bracket only; no additional rigid mass or collision model"}


def transform_error(expected, actual):
    expected, actual = validate_transform(expected), validate_transform(actual)
    return {"translation_error_m":float(np.linalg.norm(actual[:3,3]-expected[:3,3])),
            "rotation_error_rad":float(Rotation.from_matrix(expected[:3,:3].T@actual[:3,:3]).magnitude())}


def measured_camera_extrinsics(specs, hand_poses_xyzw, usd_world_optical, fixed_world_optical,
                               time_s, native_physics_time_s):
    """Compose both measured hands with immutable hand-to-optical transforms."""
    hands=np.asarray(hand_poses_xyzw,float)
    if hands.shape!=(2,7) or not np.isfinite(hands).all():
        raise ValueError("Camera rig requires both measured finite hand poses")
    if not np.isfinite([time_s,native_physics_time_s]).all():
        raise ValueError("Camera transform timestamps must be finite")
    cameras={}
    for name,spec in specs.items():
        arm=spec.get("parent_hand_index")
        if arm is None:
            hand,mount=None,None
            world=validate_transform(fixed_world_optical[name])
            parent="world"
        else:
            if arm not in (0,1):
                raise ValueError("A wrist camera must refer to measured hand0 or hand1")
            hand=pose_matrix(hands[arm])
            mount=validate_transform(spec["T_hand_optical_cv"])
            world=hand@mount
            parent=f"measured_hand_{arm}"
        authored=validate_transform(usd_world_optical[name])
        discrepancy=transform_error(world,authored)
        if discrepancy["translation_error_m"]>1e-5 or discrepancy["rotation_error_rad"]>1e-4:
            raise RuntimeError(f"Current optical USD prim differs from measured camera rig: {name}: {discrepancy}")
        cameras[name]={"parent_frame":parent,"parent_hand_index":arm,
            "T_world_hand":hand.tolist() if hand is not None else None,
            "T_hand_camera":mount.tolist() if mount is not None else None,
            "T_world_camera":world.tolist(),"T_camera_world":np.linalg.inv(world).tolist(),
            "T_world_camera_from_current_USD":authored.tolist(),
            "measured_vs_current_USD":discrepancy,
            "physics_timestamp_s":float(time_s),"native_physics_timestamp_s":float(native_physics_time_s),
            "sensor_model":spec["sensor_model"],
            "source":"Actual native PhysX hand tensor + fixed hand optical mount" if arm is not None
                     else "Fixed world D455 optical transform; current USD checked"}
    return {"timestamp_s":float(time_s),"native_physics_timestamp_s":float(native_physics_time_s),
            "convention":"Column-vector meters; hand pose XYZW; optical OpenCV +X right,+Y down,+Z forward",
            "cameras":cameras}
