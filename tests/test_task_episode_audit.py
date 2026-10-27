"""Known-motion/impulse controls for the offline episode auditor."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.scene_frames import transform_poses
from robotic_cutting_sim.tasks import make_task


spec = importlib.util.spec_from_file_location("task_episode_audit",
    Path(__file__).resolve().parents[1]/"scripts/audit_cutting_task_episodes.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_tracking_uses_measured_and_issued_poses_in_rotated_workpiece():
    pose = [1., -.4, .7, *Rotation.from_euler("zyx", [.4, -.2, .3]).as_quat()]
    task = make_task("TRIM-01", world_from_workpiece_xyzw=pose)
    time = np.array([0., 2., 4., 8.])
    command = np.stack([task.reference_edge_pose_world(t) for t in time])
    measured_local = np.stack([task.reference_edge_pose_local(t) for t in time])
    measured_local[1, :3] += [0., 0., .0004]
    measured_local[2, 3:] = Rotation.from_euler("y", .5, degrees=True).as_quat()
    measured = transform_poses(task.T_world_workpiece, measured_local)
    result, _ = audit.tracking_metrics(time, measured, command, np.array([False, True, True, False]), task)
    assert result["active"]["maximum_edge_position_error_m"] == pytest.approx(.0004)
    assert result["active"]["maximum_edge_local_z_error_m"] == pytest.approx(.0004)
    assert result["active"]["maximum_edge_rotation_error_deg"] == pytest.approx(.5)
    assert result["commanded_vs_task_schedule_max_position_m"] == 0


def test_com_uses_actual_components_and_reference_mass_after_withdrawal():
    x0 = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.]])
    positions = np.stack([x0, x0, x0+[[0, 0, .1], [0, 0, .1], [0, 0, -.2]]])
    data = {"time_s": np.array([0., 6.4, 8.]), "positions": positions, "mass_kg": np.array([1., 3., 2.])}
    result, arrays = audit.component_motion(data, np.array([5, 5, 9]), 6.4)
    assert result["components"][0]["mass_kg"] == 4
    assert result["components"][0]["com_world_at_withdrawal_m"] == [1.5, 0., 0.]
    assert result["components"][0]["final_com_displacement_m"] == pytest.approx(.1)
    assert result["components"][1]["final_com_displacement_m"] == pytest.approx(.2)
    assert arrays["component_9_com_world_m"][-1, 2] == pytest.approx(-.2)
    assert audit.component_motion(data, np.array([5, 5, 9]), 8.)[0]["status"] == "NOT_RUN"


def test_reaction_uses_actual_delayed_force_not_next_or_desired_motion():
    time = np.array([0., .1, .2])
    edge = np.tile([0., 0., 0., 0., 0., 0., 1.], (3, 1))
    edge[:, 0] = [0., .01, .03]
    rows = []
    for nxt, applied in [(2., 0.), (4., 2.)]:
        rows.append({"next_material_wrench_world": [nxt, 0, 0, 0, 0, 0],
            "applied_previous_material_wrench_world": [applied, 0, 0, 0, 0, 0],
            "current_material_reaction_impulse_world": [nxt*.1, 0, 0, 0, 0, 0],
            "material_contact_statistics": [0., 3., -.02, .0001, 7., .3, .0002]})
    result = audit.reaction_metrics(rows, {"time_s": time, "tool_pose_xyzw": edge})
    assert result["robot_applied_material_wrench_work_J"] == pytest.approx(.04)
    assert result["material_boundary_work_J"] == pytest.approx(-.04)
    assert result["interface_work_residual_fraction"] == pytest.approx(0)
    assert result["sum_reaction_impulse_world_Ns_Nms"][0] == pytest.approx(.6)
    assert result["previous_interval_feedback_carry_max_abs_error"] == 0
    assert result["tool_contact_particle_substep_events"] == 6
    assert result["material_pair_events_unattributed"] == 14
    rows[1]["current_material_reaction_impulse_world"][0] += .01
    assert audit.reaction_metrics(rows, {"time_s": time, "tool_pose_xyzw": edge})[
        "reaction_force_times_dt_impulse_max_abs_error"] == pytest.approx(.01)


def test_missing_interval_instrumentation_is_unknown_and_json_serializable():
    import json
    assert audit.interval_state_metrics([{}]) == {"minimum_detF": None, "maximum_speed_m_s": None}
    measured = audit.interval_state_metrics([{"min_detF": .98, "max_speed_m_s": .3},
        {"min_detF": .94, "max_speed_m_s": .5}])
    assert measured == {"minimum_detF": .94, "maximum_speed_m_s": .5}
    assert json.loads(json.dumps(audit.interval_state_metrics([{}]), allow_nan=False))["minimum_detF"] is None
