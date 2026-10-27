import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.coupled_robot import CouplingConfig, interpolate_measured_poses
from robotic_cutting_sim.robot import compose_tool_pose, wedge_mesh


def test_downward_hand_with_fixed_pi_mount_advances_wedge_in_world_x():
    pose = compose_tool_pose([0, 0, .2, 1, 0, 0, 0], [0, 0, .18], [1, 0, 0, 0])
    np.testing.assert_allclose(pose[:3], [0, 0, .02], atol=1e-12)
    np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), np.eye(3), atol=1e-12)


def test_fixed_wedge_is_closed_and_has_positive_exact_volume():
    vertices, faces = wedge_mesh()
    edges = {}
    for face in faces:
        for a, b in zip(face, np.roll(face, -1)):
            key = tuple(sorted((int(a), int(b))))
            edges[key] = edges.get(key, 0) + 1
    assert set(edges.values()) == {2}
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6
    assert volume == pytest.approx(.5 * .012 * (.004 + .0004) * .05, rel=1e-6)


def test_measured_pose_interpolation_has_world_angular_velocity_and_matching_endpoints():
    previous = np.r_[[0, 0, .1], Rotation.from_euler("x", np.pi).as_quat()]
    current_rotation = Rotation.from_euler("z", np.pi / 2) * Rotation.from_quat(previous[3:])
    current = np.r_[[.001, 0, .1], current_rotation.as_quat()]
    poses, twists = interpolate_measured_poses(previous, current, 20, .02)
    np.testing.assert_allclose(poses[0, :3], previous[:3], atol=1e-8)
    np.testing.assert_allclose(poses[-1, :3], current[:3], atol=1e-8)
    np.testing.assert_allclose(twists[:, :3], np.tile([.05, 0, 0], (21, 1)), atol=1e-6)
    np.testing.assert_allclose(twists[:, 3:], np.tile([0, 0, np.pi / .04], (21, 1)), atol=1e-5)


@pytest.mark.parametrize("kwargs", [{"material_dt": 0}, {"control_dt": -1}, {"control_dt": .02003}])
def test_partitioned_coupling_requires_integral_positive_time_grid(kwargs):
    with pytest.raises(ValueError):
        CouplingConfig(**kwargs)
