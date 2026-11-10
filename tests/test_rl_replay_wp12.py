"""P1-5 T5.5: replay adapter for the WP-12 Native recording format (rl.replay_wp12).

A synthetic WP-12 run directory is always tested; the real Demo 1 recording (main checkout, read-only, via the
``artifacts/wp12`` link) is tested on its first 20 steps when present, otherwise SKIPPED (NOT_RUN). CPU only."""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.finger_geometry import pad_boxes_edge
from robotic_cutting_sim.rl import replay_wp12 as W
from robotic_cutting_sim.rl.replay import MeasuredTrajectory
from robotic_cutting_sim.scene_frames import compose_pose

ROOT = Path(__file__).resolve().parents[1]
DEMO1 = ROOT / "artifacts" / "wp12" / "demos_001" / "DEMO1_A" / "run" / "episode_000"
PATHS = ["/W/link0", "/W/hand/leftfinger", "/W/hand/rightfinger", "/W/Tools/Knife"]
HALF = np.array([0.008, 0.007, 0.006])
CORNERS = (np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]) * HALF
           + np.array([0.0, 0.01, 0.04])).tolist()
HANDLE = [[-0.028, -0.09, 0.0], [0.012, 0.03, 0.011]]


def _pose(position, rotvec=(0, 0, 0)):
    return np.r_[position, Rotation.from_rotvec(rotvec).as_quat()]


def make_run(tmp_path, *, steps=5, cut=(False, True, True, False, False), gz=True, corrupt_cut=False,
             tool_from_edge=(0.028, 0., 0.09, 0., 0., 0., 1.)):
    episode, runtime = tmp_path / "run" / "episode_000", tmp_path / "run" / "runtime"
    episode.mkdir(parents=True)
    runtime.mkdir(parents=True)
    tool_from_edge = np.asarray(tool_from_edge, float)
    body = np.zeros((steps + 1, len(PATHS), 7))
    for t in range(steps + 1):
        rot = (0.03 * t, -0.01 * t, 0.05)
        body[t, 0] = _pose([-0.46, 0.12, -0.01])
        body[t, 3] = _pose([0.002 * t, 0.001, 0.28 - 0.001 * t], rot)
        body[t, 1] = _pose([0.002 * t - 0.01, 0.011, 0.33 - 0.001 * t], rot)
        body[t, 2] = _pose([0.002 * t + 0.01, -0.011, 0.33 - 0.001 * t], rot)
    edges = compose_pose(body[:, 3], tool_from_edge)
    pads = [dict(collider_path=PATHS[i] + "/pad", body_path=PATHS[i], corners_body_m=CORNERS) for i in (1, 2)]
    reset_boxes = np.asarray([HANDLE, *pad_boxes_edge(pads, {PATHS[1]: body[0, 1], PATHS[2]: body[0, 2]}, edges[0])],
                             np.float32)
    payload = {"protocol_id": "rcs_configured_task_v1", "task_id": "CFG-synthetic", "seed": 42,
               "scene": {"solver": {"cutting_law": 8, "grid_dx_m": 0.004, "max_cuts": 4, "module": "tlmpm_v5",
                                    "particle_spacing_m": 0.001}}}
    config = dict(seed=42, control_dt=0.02, material_dt=0.0001, dx=0.004, feedback=True, max_cuts=4,
                  surface_profile="current_volume_v1", configured_task=json.dumps(payload))
    recipients = [None, PATHS[1], PATHS[2]]
    (runtime / "setup.json").write_text(json.dumps(dict(
        config=config, physical_tool=dict(tool_from_edge_xyz_xyzw=tool_from_edge.tolist()),
        physical_knife_body_index=3,
        home_reset=dict(measured_after_pregrasp=dict(tool_edge_pose_xyzw=edges[0].tolist())))))
    (runtime / "finger_geometry_00.json").write_text(json.dumps(dict(pads=pads, boxes_edge=reset_boxes.tolist())))
    (episode / "task.json").write_text(json.dumps(dict(task_id="CFG-synthetic")))
    (episode / "material.json").write_text(json.dumps(dict(
        solver="tlmpm_v5", cutting_law=8, max_cuts=4, slots_per_node=8, code_encoding="packed2",
        obstacle_boxes_edge=reset_boxes.tolist(), obstacle_reaction_recipients=recipients)))
    lines = []
    for t in range(steps):
        active = bool(cut[t % len(cut)])
        submitted = dict(mode="cartesian_edge_delta", cut_enabled=active, target=[0.] * 6)
        resolved = 0 if active else -1
        if corrupt_cut and t == 1:
            resolved = 1
        lines.append(json.dumps(dict(
            step=t + 1, submitted=submitted, policy_phase="CUT:cut0" if active else "APPROACH:cut0",
            privileged_diagnostics=dict(
                timestamp_s=0.02 * (t + 1), measured_tool_pose=edges[t + 1].tolist(), active_cut=active,
                cut_index=resolved, current_total_material_reaction_impulse_world=[0.] * 5 + [float(active)],
                next_finger_wrenches_about_edge_world={PATHS[1]: [0.] * 6, PATHS[2]: [0.] * 6}))) + "\n")
    if gz:
        with gzip.open(episode / "actions.jsonl.gz", "wt") as stream:
            stream.writelines(lines)
    else:
        (episode / "actions.jsonl").write_text("".join(lines))
    np.savez_compressed(episode / "robot_state.npz", time_s=0.02 * np.arange(steps + 1), body_pose=body,
                        body_prim_paths=np.asarray(PATHS))
    return episode, body, edges, pads


@pytest.mark.parametrize("gz", [True, False])
def test_extracts_recorded_edges_cut_flags_and_pad_boxes(tmp_path, gz):
    episode, body, edges, pads = make_run(tmp_path, gz=gz)
    source = (episode / ("actions.jsonl.gz" if gz else "actions.jsonl")).read_bytes()
    traj = W.extract_wp12_trajectory(episode)
    assert (episode / ("actions.jsonl.gz" if gz else "actions.jsonl")).read_bytes() == source   # never rewritten
    assert traj.steps == 5 and traj.options["solver"] == "tlmpm_v5" and traj.options["max_cuts"] == 4
    assert traj.options["material_diagnostics"] == "snapshot" and traj.options["recording_format"] == W.RECORDING_FORMAT
    np.testing.assert_array_equal(traj.edge_pose_world_xyzw, edges[1:])
    np.testing.assert_array_equal(traj.reset_edge_pose_world_xyzw, edges[0])
    np.testing.assert_array_equal(traj.resolved_cut, [-1, 0, 0, -1, -1])
    for t in range(5):
        expected = pad_boxes_edge(pads, {PATHS[1]: body[t + 1, 1], PATHS[2]: body[t + 1, 2]}, edges[t + 1])
        np.testing.assert_array_equal(traj.pad_boxes_edge[t, 1:], np.asarray(expected, np.float32))
        np.testing.assert_array_equal(traj.pad_boxes_edge[t, 0], np.asarray(HANDLE, np.float32))
    c = traj.checks
    assert c["pad_boxes_verified"] and c["cutting_steps"] == 2 and c["native_reaction_steps"] == 2
    assert c["native_finger_wrench_steps"] == 0 and c["submitted_modes"] == ["cartesian_edge_delta"]
    assert traj.pad_recipients == (None, PATHS[1], PATHS[2])
    traj.save(tmp_path / "traj")
    again = MeasuredTrajectory.load(tmp_path / "traj")
    np.testing.assert_array_equal(again.pad_boxes_edge, traj.pad_boxes_edge)


def test_refuses_inconsistent_recordings(tmp_path):
    episode, *_ = make_run(tmp_path / "a", corrupt_cut=True)
    with pytest.raises(ValueError, match="cut flags"):
        W.extract_wp12_trajectory(episode)
    episode, *_ = make_run(tmp_path / "b")
    (episode / "task.json").write_text(json.dumps(dict(task_id="CFG-other")))
    with pytest.raises(ValueError, match="task id"):
        W.extract_wp12_trajectory(episode)
    episode, *_ = make_run(tmp_path / "c")
    material = json.loads((episode / "material.json").read_text())
    material["max_cuts"] = 2
    (episode / "material.json").write_text(json.dumps(material))
    with pytest.raises(ValueError, match="max_cuts"):
        W.extract_wp12_trajectory(episode)
    episode, *_ = make_run(tmp_path / "d")
    setup = json.loads((episode.parent / "runtime" / "setup.json").read_text())
    setup["config"]["surface_profile"] = "cut_aware_surface_v1"
    (episode.parent / "runtime" / "setup.json").write_text(json.dumps(setup))
    with pytest.raises(NotImplementedError):
        W.extract_wp12_trajectory(episode)


def test_native_samples_view(tmp_path):
    episode, *_ = make_run(tmp_path)
    n = 4
    arrays = dict(time_s=np.array([0., .2]), positions=np.zeros((2, n, 3)), velocities=np.zeros((2, n, 3)),
                  codes=np.zeros((2, n), np.int32), mass_kg=np.ones(n), reference_positions=np.zeros((n, 3)),
                  reference_positions_local=np.zeros((n, 3), np.float32), reference_indices=np.zeros((n, 3), np.int32),
                  reference_stencil_nodes=np.zeros((n, 8), np.int32), reference_stencil_weights=np.zeros((n, 8), np.float32),
                  reference_stencil_gradients=np.zeros((n, 8, 3), np.float32))
    np.savez_compressed(episode / "physics_state.npz", **arrays)
    samples, checkpoint = W.native_samples(episode)
    assert checkpoint == dict(sample_count=2, first_time_s=0., last_time_s=.2, complete=True)
    assert samples["codes"].shape == (2, n) and "positions" in samples
    samples.close()


# ---------------------------------------------------------------- the real WP-12 Demo 1 recording (read-only)

def _demo1():
    if not (DEMO1 / "actions.jsonl.gz").exists():
        pytest.skip("WP-12 Demo 1 recording not present (NOT_RUN)")
    return DEMO1


def test_demo1_extraction_first_20_steps():
    episode = _demo1()
    before = hashlib.sha256((episode / "actions.jsonl.gz").read_bytes()).hexdigest()
    traj = W.extract_wp12_trajectory(episode)
    assert hashlib.sha256((episode / "actions.jsonl.gz").read_bytes()).hexdigest() == before
    assert traj.steps == 740 and traj.task_payload["task_id"] == "CFG-bar_flat_board-sever_mid-c32f94a4"
    o = traj.options
    assert (o["solver"], o["cutting_law"], o["max_cuts"], o["slots_per_node"], o["substeps_per_control_step"]) == \
        ("tlmpm_v5", 8, 4, 8, 200)
    c = traj.checks
    assert c["pad_boxes_verified"] and c["knife_body_to_edge_max_position_diff_m"] < 1e-12
    assert c["reset_pad_boxes_rebuilt_max_abs_diff_m"] == 0. and c["material_json_obstacle_boxes_max_abs_diff_m"] == 0.
    assert (c["cutting_steps"], c["first_cutting_step"], c["last_cutting_step"]) == (137, 207, 343)
    assert c["return_home_steps"] == 40 and c["native_finger_wrench_steps"] == 0
    # first 20 steps against the raw rows
    _, rows = W.read_action_rows(episode)
    for t in range(20):
        np.testing.assert_array_equal(traj.edge_pose_world_xyzw[t], rows[t]["edge"])
        assert traj.resolved_cut[t] == -1 and traj.time_s[t] == pytest.approx(.02 * (t + 1), abs=1e-12)
    assert np.all(traj.pad_boxes_edge[:20, :, 1] > 0)
    np.testing.assert_array_equal(traj.pad_boxes_edge[:20, 0], np.repeat(traj.reset_pad_boxes_edge[None, 0], 20, 0))


def test_demo1_reference_arrays_match_the_rebuilt_frozen_v5_material():
    episode = _demo1()
    pytest.importorskip("warp")
    from robotic_cutting_sim.rl.replay import build_material
    traj = W.extract_wp12_trajectory(episode)
    material, task = build_material(traj, "cpu")
    material.reset(traj.reset_edge_pose_world_xyzw, traj.options["seed"])
    samples, checkpoint = W.native_samples(episode)
    try:
        assert checkpoint["sample_count"] == 75 and checkpoint["last_time_s"] == pytest.approx(14.8)
        np.testing.assert_array_equal(material.masses, samples["mass_kg"])
        np.testing.assert_array_equal(material.world_points, samples["reference_positions"])
        np.testing.assert_array_equal(material.bed.nodes.numpy(), samples["reference_stencil_nodes"])
        np.testing.assert_array_equal(material.bed.weights.numpy(), samples["reference_stencil_weights"])
        np.testing.assert_array_equal(material.bed.gradients.numpy(), samples["reference_stencil_gradients"])
        assert not np.asarray(samples["codes"][0]).any()
        assert material.solver == "tlmpm_v5" and np.shape(material.obstacle_boxes_edge) == (3, 2, 3)
    finally:
        samples.close()


def test_source_equivalence_audit_of_the_v5_path():
    if not (ROOT / ".git").exists():
        pytest.skip("no git checkout (NOT_RUN)")
    audit = W.source_equivalence_audit("a960dc6c1aea1060234a94ea7d4a310370484773", ROOT)
    if audit["files"]["src/robotic_cutting_sim/tlmpm_v5.py"]["recording"] is None:
        pytest.skip("recording commit a960dc6 not in this repository (NOT_RUN)")
    assert audit["files"]["src/robotic_cutting_sim/tlmpm_v5.py"]["identical"]
    assert set(audit["changed_files"]) <= {"src/robotic_cutting_sim/live_material.py",
                                           "src/robotic_cutting_sim/scene_config.py"}
    assert audit["runner_pad_semantics_equal"]
