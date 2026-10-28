"""Blunt pad bounds derived from actual composed USD colliders and measured bodies.

The TLMPM box interface uses conservative edge-aligned bounds of the collider
OBBs, updated at each coupling boundary. These are not calibrated finger skin.
"""
import itertools
import numpy as np

from .scene_frames import invert_transform, matrix_pose, pose_matrix, transform_points


def discover_pads(stage, hand_path):
    from pxr import Usd, UsdGeom, UsdPhysics
    hand = stage.GetPrimAtPath(hand_path)
    if not hand:
        raise ValueError('Actual hand prim is missing')
    pads = []
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(hand):
        if not prim.IsA(UsdGeom.Cube) or not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if 'finger_pad' not in prim.GetName():
            continue
        body = prim.GetParent()
        while body and not body.HasAPI(UsdPhysics.RigidBodyAPI):
            body = body.GetParent()
        if not body:
            raise ValueError('Pad collider has no actual rigid finger body')
        relative = np.asarray(cache.ComputeRelativeTransform(prim, body)[0], float).T
        half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
        unit_corners = np.asarray(list(itertools.product((-half, half), repeat=3)))
        # USD's authored collider scale is intentional geometry, not a rigid pose.
        corners = unit_corners @ relative[:3, :3].T + relative[:3, 3]
        pads.append(dict(collider_path=str(prim.GetPath()), body_path=str(body.GetPath()),
                         corners_body_m=corners.tolist(), source='Composed USD CollisionAPI Cube'))
    if len(pads) != 2:
        raise ValueError(f'Expected two actual Franka finger-pad cubes, got {len(pads)}')
    return sorted(pads, key=lambda pad: pad['collider_path'])


def pad_boxes_edge(pads, body_poses_world, edge_pose_world):
    inverse = invert_transform(pose_matrix(edge_pose_world))
    boxes = []
    for pad in pads:
        pose = body_poses_world[pad['body_path']]
        corners = transform_points(inverse @ pose_matrix(pose), np.asarray(pad['corners_body_m']))
        low, high = corners.min(0), corners.max(0)
        boxes.append([(low + high) / 2, (high - low) / 2])
    return np.asarray(boxes, np.float32)


def pad_plane_clearance(pads, body_poses_world, plane_origin_world, plane_normal_world):
    normal = np.asarray(plane_normal_world, float)
    normal = normal / np.linalg.norm(normal)
    return {pad['collider_path']: float(np.min((transform_points(pose_matrix(body_poses_world[pad['body_path']]),
                    np.asarray(pad['corners_body_m'])) - plane_origin_world) @ normal)) for pad in pads}
