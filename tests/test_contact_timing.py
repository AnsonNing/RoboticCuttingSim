"""Known-pose CPU controls; contact-event timing is not inferred from a score."""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

path = Path(__file__).resolve().parents[1] / "scripts/diagnose_tool_contact_timing.py"
spec = importlib.util.spec_from_file_location("contact_timing_test", path)
timing = importlib.util.module_from_spec(spec); spec.loader.exec_module(timing)


def test_exact_pre_step_points_can_fail_post_step_aabb_without_any_grasp_slip():
    before = np.array([0., 0., 0., 0., 0., 0., 1.])
    after = before.copy(); after[0] += 5e-6
    half = np.array([.008, .011, .020])
    contact = np.array([[-.008, -.011, 0.]])
    assert timing.point_excess(contact, before, half).max() == 0
    assert timing.point_excess(contact, after, half).max() > timing.MARGIN_M
    # Conversely, post-step-generated contact points require the post-step pose.
    # A universal 'always use previous pose' heuristic would also be incorrect.
    new_contact = np.array([[.008 + 5e-6, -.011, 0.]])
    assert timing.point_excess(new_contact, after, half).max() <= 1e-18
    assert timing.point_excess(new_contact, before, half).max() > timing.MARGIN_M


def test_rotational_timing_changes_region_test_while_exact_source_geometry_stays_valid():
    half = np.array([.008, .011, .020])
    before = np.r_[[.2, -.1, .3], Rotation.from_euler("xyz", [10, 25, -15], degrees=True).as_quat()]
    after = np.r_[before[:3], (Rotation.from_euler("z", .001) * Rotation.from_quat(before[3:])).as_quat()]
    points = Rotation.from_quat(before[3:]).apply([[-.008, -.011, .01], [.008, .011, -.01]]) + before[:3]
    assert timing.point_excess(points, before, half).max() < 1e-15
    assert timing.point_excess(points, after, half).max() > timing.MARGIN_M
    midpoint = timing.midpoint_pose(before, after)
    assert np.isclose(np.linalg.norm(midpoint[3:]), 1.)


def test_diagnosis_preserves_recorded_failure_and_separates_unintended_bodies():
    hand, tool = "/Hand", "/World/Tools/CuttingTool_0"
    before = [0., 0., 0., 0., 0., 0., 1.]
    after = [5e-6, 0., 0., 0., 0., 0., 1.]
    contacts = [{"body0": tool, "body1": hand + "/panda_" + side + "finger", "dt_s": .002,
        "time_s": .002, "position_world_m": [-.008, sign * .011, 0.],
        "raw_impulse_world_ns": [0., .002, 0.]} for side, sign in [("left", -1), ("right", 1)]]
    contacts.append({**contacts[0], "body1": "/Table", "position_world_m": [0., 0., -.02]})
    row = {"phase": "coupled_episode", "native_time_s": .002, "tool_body_pose_xyzw": after,
        "contacts": {"sensor_valid": True, "sensor_fresh": True, "new_physics_step": True,
                     "bilateral_loaded_handle_contact": False, "contacts": contacts}}
    result = timing.diagnose([{"tool_body_pose_xyzw": before, "native_time_s": 0.}, row],
                             {"handle_half_extents_m": [.008, .011, .020]}, hand)
    assert result["bilateral_fraction"]["recorded"] == 0.
    assert result["bilateral_fraction"]["post"] == 0.
    assert result["bilateral_fraction"]["pre"] == 1.
    assert result["loaded_nonfinger_pairs"] == 1
    assert not result["threshold_or_benchmark_score_changed"]
    assert not result["physical_grasp_acceptance"]
