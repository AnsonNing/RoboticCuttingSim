import numpy as np
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.scene_frames import pose_matrix
from robotic_cutting_sim.wrist_camera_clearance import (intentional_attachment,
    surface_owner, triangle_obb_intersections, clipped_triangle_vertices_in_box)


def test_nested_body_ownership_excludes_camera_visuals_and_path_prefix_aliases():
    palm = "/World/Robot/link0/link7/panda_hand"
    finger = palm+"/panda_leftfinger"
    bodies = ["/World/Robot/link0", palm, finger]
    assert surface_owner(finger+"/visuals/mesh", bodies) == finger
    assert surface_owner(palm+"/visuals/mesh", bodies) == palm
    assert surface_owner(palm+"/D405/Housing", bodies, [palm+"/D405"]) is None
    assert surface_owner("/World/Robot/link01/mesh", bodies) is None


def test_attachment_exception_does_not_hide_housing_or_finger_intersections():
    palm = "/Robot/panda_hand"
    assert intentional_attachment("Bracket", palm, palm)
    assert not intentional_attachment("Housing", palm, palm)
    assert not intentional_attachment("Bracket", palm+"/panda_leftfinger", palm)
    assert not intentional_attachment("Bracket", "/Other/panda_hand", palm)


def test_triangle_box_test_detects_crossing_and_separation_under_rigid_frames():
    triangles = np.array([[[-.2, 0, 0], [.2, 0, 0], [0, .2, 0]],
                          [[.06, 0, 0], [.06, .1, 0], [.06, 0, .1]]])
    identity = np.eye(4)
    assert triangle_obb_intersections(triangles, identity, [.1]*3).tolist() == [True, False]
    frame = pose_matrix([.4, -.2, .3, *Rotation.from_euler("xyz", [.3, -.7, .2]).as_quat()])
    transformed = triangles@frame[:3, :3].T+frame[:3, 3]
    assert triangle_obb_intersections(transformed, frame, [.1]*3).tolist() == [True, False]


def test_distance_tolerance_is_in_meters_instead_of_triangle_area_units():
    triangle = np.array([[[.050000005, 0, 0], [.050000005, .1, 0], [.050000005, 0, .1]]])
    assert triangle_obb_intersections(triangle, np.eye(4), [.1]*3, 1e-8)[0]
    assert not triangle_obb_intersections(triangle, np.eye(4), [.1]*3, 1e-10)[0]


def test_clipped_intersection_extent_retains_box_edges_without_inventing_volume():
    triangle = np.array([[[-.2, 0, 0], [.2, 0, 0], [0, .2, 0]]])
    points = clipped_triangle_vertices_in_box(triangle, np.eye(4), [.1]*3)
    np.testing.assert_allclose(points.min(0), [-.05, 0, 0], atol=1e-15)
    np.testing.assert_allclose(points.max(0), [.05, .05, 0], atol=1e-15)
    frame = pose_matrix([.4, -.2, .3, *Rotation.from_euler("xyz", [.3, -.7, .2]).as_quat()])
    transformed = triangle@frame[:3, :3].T+frame[:3, 3]
    other = clipped_triangle_vertices_in_box(transformed, frame, [.1]*3)
    np.testing.assert_allclose(other.min(0), points.min(0), atol=1e-15)
    np.testing.assert_allclose(other.max(0), points.max(0), atol=1e-15)
    separated = triangle+np.array([0, 0, .1])
    assert not len(clipped_triangle_vertices_in_box(separated, np.eye(4), [.1]*3))
