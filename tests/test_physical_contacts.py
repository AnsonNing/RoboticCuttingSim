"""Contact bookkeeping controls; these do not simulate or certify a grasp."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.physical_contacts import NativeToolContacts, contact_records, summarize_contacts


PATHS = {1: "/Tool", 2: "/Hand/Left", 3: "/Hand/Right", 4: "/Hand", 5: "/Table"}
ROLES = {"left_finger": ["/Hand/Left"], "right_finger": ["/Hand/Right"], "palm": ["/Hand"], "table": ["/Table"]}


def report(pair=(1, 2), impulse=(.004, 0, 0), position=(0, 0, 0), time=.1):
    return dict(body0=pair[0], body1=pair[1], impulse=impulse, position=position,
                normal=(1, 0, 0), dt=.002, time=time)


def records(raw, **kwargs):
    return contact_records(raw, tool_path="/Tool", role_paths=ROLES, physics_time_s=.1,
                           dt=.002, decode_body=PATHS.__getitem__, **kwargs)


def test_opposing_contacts_are_distinct_and_do_not_cancel_gripping_pressure():
    raw = [report(), report((3, 1), (-.006, 0, 0)), report((1, 4)), report((5, 1))]
    contacts = records(raw)
    summary = summarize_contacts(contacts, valid=True, fresh=True)
    assert summary["bilateral_tool_contact"]
    assert summary["roles"]["left_finger"]["force_magnitude_sum_n"] == 2.
    assert summary["roles"]["right_finger"]["force_magnitude_sum_n"] == 3.
    assert summary["unintended_contact_count"] == 2
    assert contacts[1]["tool_is_body0"] is False
    np.testing.assert_array_equal(contacts[1]["raw_force_world_n"], [-3., 0., 0.])


def test_stale_or_invalid_sensor_cannot_certify_bilateral_contact():
    contacts = records([report(), report((3, 1), time=.09)])
    assert contacts[1]["fresh"] is False
    assert not summarize_contacts(contacts, valid=True, fresh=True)["bilateral_tool_contact"]
    assert not summarize_contacts(records([report(), report((3, 1))]), valid=False, fresh=True)["bilateral_tool_contact"]
    assert records([report(time=.098)])[0]["fresh"] is False


def test_zero_impulse_touch_is_not_certified_as_a_loaded_grip():
    contacts = records([report(impulse=(0, 0, 0)), report((3, 1))])
    summary = summarize_contacts(contacts, valid=True, fresh=True)
    assert summary["bilateral_tool_contact"] and not summary["bilateral_loaded_tool_contact"]


def test_handle_attribution_uses_measured_contact_point_in_transformed_tool_frame():
    rotation = Rotation.from_euler("z", 90, degrees=True)
    pose = np.r_[[1., 2., 3.], rotation.as_quat()]
    handle = rotation.apply([.01, 0., 0.])+pose[:3]
    blade = rotation.apply([.10, 0., 0.])+pose[:3]
    contacts = records([report(position=handle), report((3, 1), position=blade)],
        tool_pose_xyzw=pose, handle_bounds_tool_m=([-.02, -.01, -.01], [.02, .01, .01]))
    assert contacts[0]["in_handle_region"] is True and contacts[1]["in_handle_region"] is False
    summary = summarize_contacts(contacts, valid=True, fresh=True)
    assert summary["bilateral_tool_contact"] and not summary["bilateral_handle_contact"]


def test_missing_timing_and_nonfinite_impulses_fail_instead_of_fabricating_force():
    raw = report(); raw["dt"] = 0
    with pytest.raises(ValueError, match="time/dt"):
        records([raw])
    with pytest.raises(ValueError, match="finite"):
        records([report(impulse=(np.nan, 0, 0))])


def test_sibling_name_prefix_does_not_become_tool_contact():
    decode = {**PATHS, 6: "/ToolOther"}.__getitem__
    contacts = contact_records([report((6, 2))], tool_path="/Tool", role_paths=ROLES,
        physics_time_s=.1, dt=.002, decode_body=decode)
    assert contacts[0]["tool_in_pair"] is False
    assert summarize_contacts(contacts, valid=True, fresh=True)["roles"]["left_finger"]["contact_count"] == 0


def test_detection_geometry_is_separate_from_measured_poststep_actor_pose():
    # A moving handle advances100um in one2ms step. Its raw contact point
    # remains on the detection geometry. A poststep transform mislabels it;
    # using the explicitly saved pre-step pose requires no tolerance change.
    from types import SimpleNamespace
    monitor = object.__new__(NativeToolContacts)
    monitor.tool_path, monitor.dt, monitor.sensor_path = "/Tool", .002, "/Tool/Sensor"
    monitor.role_paths, monitor.decode_body = ROLES, PATHS.__getitem__
    monitor.handle_bounds_tool_m = ([-.008, -.011, -.02], [.008, .011, .02])
    monitor.manager = SimpleNamespace(get_simulation_time=lambda: .1, get_num_physics_steps=lambda: 50)
    monitor.sensor = SimpleNamespace(get_sensor_reading=lambda: SimpleNamespace(
        is_valid=True, time=.1, in_contact=True, value=2.), get_raw_data=lambda: [
            report(position=(0, -.011, 0)), report((3, 1), position=(0, .011, 0))])
    monitor._tensor_sample, monitor._last_step = lambda pose: {"status":"UNIT_TEST_NO_NATIVE_DATA"}, None
    before = np.array([0., 0., 0., 0., 0., 0., 1.])
    after = before.copy(); after[1] += .0001
    old = monitor.sample(tool_pose_xyzw=after)
    assert not old["bilateral_loaded_handle_contact"]
    aligned = monitor.sample(tool_pose_xyzw=after, contact_geometry_pose_xyzw=before)
    assert aligned["bilateral_loaded_handle_contact"]
    assert aligned["contact_geometry_frame"] == "PRE_STEP_MEASURED_TOOL_POSE"
    assert aligned["contact_geometry_time_s"] == pytest.approx(.098)
    np.testing.assert_array_equal(after, [0., .0001, 0., 0., 0., 0., 1.])
