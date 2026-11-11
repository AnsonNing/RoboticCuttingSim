"""P1-1b B2 follow-up: the opt-in ``mjwarp_grip_v1`` grip contact-solver profile of workload B.

Options (any venv): the profile exists only for the B2 physical grasp, is listed as a difference (never silently
applied), and the default B1 / the single env's profiles carry no solver profile.

Lab (Isaac Lab venv, Warp CPU device, one build per process through ``scripts/rl_ready/probe_lab_b2_grasp.py``):
the profile reaches the MuJoCo-Warp model (``DisableBit.NATIVECCD`` set: box-box pairs use MuJoCo's ``box_box``
collider; ``impratio`` 10), the default profile does not; under the WP-17 seed-512 recorded TRIM commands
(HOME -> skim reorientation, 150 control steps, robot only) the settled grasp keeps 4 contacts per finger pad and
the tool stays inside the grasp monitor's warning gates (2 mm, pi/60), while the default point profile's manifold
collapses to 1-2 points per pad and the tool slides more than 5 mm (the B2 root cause, measured on CPU and GPU).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT/"scripts/rl_ready/probe_lab_b2_grasp.py"
WP17 = ROOT/"artifacts/wp17/native_optimized_001/seed_512/box/env/recording/attempt_000/actions.jsonl"


def test_profile_options_and_fidelity_listing():
    from robotic_cutting_sim.batched import lab_scene as L
    from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
    assert L.GRIP_PROFILES == ("legacy_point_contact", "finite_pad_patch_v3", "mjwarp_grip_v1")
    assert L.contact_solver_profile("legacy_point_contact") is None
    assert L.contact_solver_profile("finite_pad_patch_v3") is None
    profile = L.contact_solver_profile(L.MJWARP_GRIP_V1)
    assert profile["mjwarp_disable_bits"] == ("NATIVECCD",) and profile["impratio"] == 10.
    options = L.BatchedLabOptions(tool_mount="physical_grasp_v1", grip_contact_profile=L.MJWARP_GRIP_V1,
                                  device="cpu", cpu_test_mode=True).validate()
    listed = options.fidelity()["listed_differences"]
    assert listed["grip_contact_profile"] == "mjwarp_grip_v1"
    assert listed["contact_solver_profile"]["impratio"] == 10.
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_TOOL_MOUNT"):
        L.BatchedLabOptions(tool_mount="rigid_mount_v1", grip_contact_profile=L.MJWARP_GRIP_V1).validate()
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_ACTUATOR_OVERRIDE"):
        L.BatchedLabOptions(tool_mount="physical_grasp_v1", grip_contact_profile="mjwarp_grip_v2").validate()
    default = L.BatchedLabOptions().validate()
    assert default.grip_contact_profile == "legacy_point_contact"
    assert "contact_solver_profile" not in default.fidelity()["listed_differences"]


def _run(tmp_path, name, profile, steps):
    out = tmp_path/name
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT/"src")}
    proc = subprocess.run([sys.executable, str(PROBE), "--device", "cpu", "--scene-seeds", "512", "--steps", str(steps),
                           "--non-strict", "--contacts-every", "0", "--grip-profile", profile, "--out", str(out)],
                          env=env, cwd=ROOT, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        raise AssertionError(f"probe failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return json.loads((out/"row.json").read_text()), np.load(out/"diag_trace.npz")


@pytest.mark.cpu_sim
def test_lab_mjwarp_grip_v1_holds_the_reorientation_and_point_profile_collapses(tmp_path):
    pytest.importorskip("isaaclab", reason="Lab build needs the Isaac Lab venv (NOT_RUN in .venv)")
    if not WP17.is_file():
        pytest.skip("WP-17 recorded commands not available (artifacts/wp17)")
    steps = 150
    row, trace = _run(tmp_path, "grip", "mjwarp_grip_v1", steps)
    solver = row["contact_solver"]
    assert solver["profile"] == "mjwarp_grip_v1" and solver["nativeccd_disabled"] is True
    assert solver["impratio"] == 10. and solver["box_box_collider"] == "mujoco_box_box_primitive"
    assert row["grasp_initialization"]["status"] == "PASS_PHYSICAL_PREGRASP"
    assert row["steps_completed"] == steps and row["termination_reason"] == [0]
    counts = np.concatenate([trace["count_left"][:, 0], trace["count_right"][:, 0]])
    assert np.mean(counts == 4) >= .95                               # the 4-corner pad manifold persists
    slip = np.linalg.norm(trace["slip_hand"][:, 0], axis=-1)
    rotation = np.linalg.norm(trace["rotvec_hand"][:, 0], axis=-1)
    assert slip.max() < .002 and rotation.max() < np.pi/60           # inside the monitor's warning gates
    assert not (trace["monitor_status"] == 2).any()
    point, point_trace = _run(tmp_path, "point", "legacy_point_contact", steps)
    assert point["contact_solver"]["nativeccd_disabled"] is False and point["contact_solver"]["impratio"] == 1.
    point_counts = np.concatenate([point_trace["count_left"][:, 0], point_trace["count_right"][:, 0]])
    assert np.mean(point_counts <= 2) > .5                            # manifold collapsed to 1-2 points per pad
    assert np.linalg.norm(point_trace["slip_hand"][:, 0], axis=-1).max() > .005


@pytest.mark.cpu_sim
def test_open_vector_env_passes_mjwarp_grip_v1_to_the_lab_env(tmp_path):
    """P1 consolidation: ``open_vector_env(backend="isaac_lab_newton", tool_mount="physical_grasp_v1",
    grip_contact_profile="mjwarp_grip_v1")`` reaches the MuJoCo-Warp model of the batched Lab env and is recorded in
    ``receipt()["grip"]`` and ``fidelity()`` (worker ``run_p1_2b_smoke.py``, Warp CPU, N=1, coarse 2.5 mm, 2 steps,
    contract incl. a partial reset). Review finding 3: the run uses a non-default ``material_dt_s`` 2e-4 (the
    review's scenario) and ``fidelity()`` must report it, the halved substeps and the coarse spacing (never full)."""
    pytest.importorskip("isaaclab", reason="Lab build needs the Isaac Lab venv (NOT_RUN in .venv)")
    out = tmp_path/"vector_grip"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT/"src")}
    proc = subprocess.run([sys.executable, str(ROOT/"scripts/rl_ready/run_p1_2b_smoke.py"), "--backend",
                           "isaac_lab_newton", "--device", "cpu", "--tasks", "512", "--steps", "2", "--reset-step", "1",
                           "--reset-envs", "0", "--spacing", "0.0025", "--max-episode-steps", "2",
                           "--tool-mount", "physical_grasp_v1", "--grip-profile", "mjwarp_grip_v1", "--non-strict",
                           "--material-dt", "2e-4", "--out", str(out)], env=env, cwd=ROOT, capture_output=True,
                          text=True, timeout=1800)
    row = json.loads((out/"row.json").read_text()) if (out/"row.json").exists() else {}
    if proc.returncode != 0:
        raise AssertionError(f"worker failed ({proc.returncode}): {row.get('error')}\n{proc.stderr[-3000:]}")
    assert row["status"] == "PASS" and row["grip_profile"] == "mjwarp_grip_v1"
    assert row["grip"]["grip_contact_profile"] == "mjwarp_grip_v1" and row["grip"]["applies"] is True
    assert row["grip"]["contact_solver_profile"]["impratio"] == 10.
    assert row["fidelity"]["grip_contact_profile"] == "mjwarp_grip_v1"
    assert row["fidelity"]["contact_solver_profile"] == "mjwarp_grip_v1"
    assert row["fidelity"]["listed_differences"]["contact_solver_profile"]["version"] == "mjwarp_grip_v1"
    fid = row["fidelity"]
    assert fid["reduced_fidelity"] is True and fid["fidelity_version"] == "lab_fidelity_v2"
    assert fid["actual"]["material_dt_s"] == pytest.approx(2e-4) and fid["actual"]["substeps_per_control_step"] == 100
    for prefix in ("material_dt_s=0.0002", "substeps_per_control_step=100", "particle_spacing_m=0.0025",
                   "episode_steps=2"):
        assert any(r.startswith(prefix) for r in fid["reasons"]), (prefix, fid["reasons"])
    solver = row["contact_solver"]                     # read back from the live MuJoCo-Warp model
    assert solver["profile"] == "mjwarp_grip_v1" and solver["nativeccd_disabled"] is True and solver["impratio"] == 10.
