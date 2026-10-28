import importlib.util
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

SPEC = importlib.util.spec_from_file_location("tool_momentum", Path(__file__).resolve().parents[1] / "scripts/audit_tool_momentum.py")
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
LIMITS = {"linear_absolute_budget_n_s": 2e-5, "angular_absolute_budget_n_m_s": 1e-6, "relative_budget": .05}
IDENTITY = np.array([0., 0., 0., 0., 0., 0., 1.])


def evaluate(**changes):
    values = dict(mass=.1, inertia=np.diag([.001, .002, .003]), pose_before=IDENTITY,
                  pose_after=IDENTITY, velocity_before=np.zeros(6), velocity_after=np.zeros(6),
                  gravity=np.array([0., 0., -10.]), applied_wrench=np.zeros(6), contact_wrenches=[],
                  contact_points=[], com_before=np.zeros(3), dt=.002, limits=LIMITS)
    values.update(changes)
    return audit.balance_step(**values)


def test_freefall_uses_actual_com_velocity_and_gravity_once():
    row = evaluate(velocity_after=np.array([0., 0., -.02, 0., 0., 0.]))
    assert row["linear"]["pass"]
    assert np.linalg.norm(row["linear"]["residual"]) < 1e-15
    assert not evaluate()["linear"]["pass"]


def test_normal_and_friction_both_balance_contact_load():
    row = evaluate(contact_wrenches=[[0., 0., 1., 0., 0., 0.], [-.2, 0., 0., 0., 0., 0.]],
                   contact_points=[[0., 0., 0.], [0., 0., 0.]],
                   applied_wrench=np.array([.2, 0., 0., 0., 0., 0.]))
    assert row["linear"]["pass"]
    assert not evaluate(contact_wrenches=[[0., 0., 1., 0., 0., 0.]], contact_points=[[0., 0., 0.]],
                        applied_wrench=np.array([.2, 0., 0., 0., 0., 0.]))["linear"]["pass"]


def test_contact_force_lever_and_intrinsic_torque_shift_to_com():
    row = evaluate(gravity=np.zeros(3), contact_wrenches=[[1., 0., 0., 0., .01, 0.]],
                   contact_points=[[0., 0., .02]], velocity_after=np.array([.02, 0., 0., 0., .03, 0.]))
    assert row["linear"]["pass"] and row["angular"]["pass"]
    assert np.linalg.norm(row["angular"]["residual"]) < 1e-15


def test_angular_momentum_uses_rotating_full_inertia_not_I_times_delta_omega():
    inertia = np.diag([.001, .002, .003])
    pose = IDENTITY.copy()
    pose[3:] = Rotation.from_euler("z", .4).as_quat()
    before = np.array([0., 0., 0., 1., .3, .1])
    l0 = inertia @ before[3:]
    r = Rotation.from_quat(pose[3:]).as_matrix()
    after = np.r_[np.zeros(3), np.linalg.solve(r @ inertia @ r.T, l0)]
    row = evaluate(gravity=np.zeros(3), pose_after=pose, velocity_before=before, velocity_after=after)
    assert row["angular"]["pass"]
    assert np.linalg.norm(row["angular"]["residual"]) < 1e-15


def test_summary_retains_worst_single_step_failure():
    good = evaluate(velocity_after=np.array([0., 0., -.02, 0., 0., 0.]))
    bad = evaluate()
    result = audit.summary([good] * 100 + [bad], "linear", absolute_budget=2e-5)
    assert result["status"] == "FAIL_IMPULSE_BALANCE"
    assert result["failing_steps"] == 1
    assert result["first_failing_row_indices"] == [100]
    assert result["standalone_absolute_component_diagnostic"]["exceeding_steps"] == 1
