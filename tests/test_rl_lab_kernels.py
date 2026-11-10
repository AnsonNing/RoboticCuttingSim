"""ADR 0022 §8 C1: workload B robot kernels on the Warp CPU device (torch-free; ``.venv``).

* index tables resolved from labels, asserted per world (bitwise);
* edge kernel (B1 hand chain, B2 tool chain, nominal hand mount) against the frozen NumPy compose (gate <= 1 f64
  ulp; measured 0);
* pad-box kernel against ``finger_geometry.pad_boxes_edge`` (gate <= 1 f32 ulp);
* wrench kernel: a nonzero wrench in env k changes only env k's tool and finger rows (bitwise), values against a
  NumPy lever-arm reference (float32 tolerance); the material buffer is read in place (bitwise record);
* contact sample / grasp monitor / validity kernels stay inside their world.
"""
from functools import lru_cache

import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

pytest.importorskip("pxr", reason="needs OpenUSD (pxr): .venv or Lab venv; NOT_RUN in the Torch venv")

# NOT_RUN where the robot-batching inputs are absent (e.g. CI): the official Franka USD asset cache and the Newton
# USD importer (`newton-usd-schemas`, the `robot` extra) are local-only. Only the tests that build the USD scene
# carry these markers; option refusals, fidelity labels and pure kernel checks run everywhere.
import importlib.util as _ilu  # noqa: E402
from robotic_cutting_sim.recording import ROOT as _ROOT  # noqa: E402
from robotic_cutting_sim.workcell import ASSETS as _ASSETS  # noqa: E402
_FRANKA_CACHE = (_ROOT/"artifacts/workcell_asset_cache"/_ASSETS["franka"]).exists()
_NEWTON_USD = _ilu.find_spec("newton_usd_schemas") is not None
NEEDS_FRANKA_CACHE = pytest.mark.skipif(
    not _FRANKA_CACHE, reason="NOT_RUN: Franka USD asset cache (artifacts/workcell_asset_cache) not present")
NEEDS_USD_SCENE = pytest.mark.skipif(
    not (_FRANKA_CACHE and _NEWTON_USD),
    reason="NOT_RUN: needs the Franka USD asset cache and the Newton USD importer (newton-usd-schemas)")


from robotic_cutting_sim.batched import lab_scene
from robotic_cutting_sim.batched import robot_kernels as rk

wp.config.quiet = True
DEVICE = "cpu"
N = 3


@lru_cache(maxsize=None)
def task():
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(512), goal_config(), seed=42, particle_spacing_m=.0025)


@lru_cache(maxsize=None)
def scene(mount, n=N):
    return lab_scene.build_newton_rigid_scene(task(), n, mount, device=DEVICE)


def fk_body_q(sc, q=None):
    import newton
    state = sc.model.state()
    if q is not None:
        state.joint_q.assign(q)
    newton.eval_fk(sc.model, state.joint_q, state.joint_qd, state)
    return state


def ulp64(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    ia, ib = a.view(np.int64), b.view(np.int64)
    ia = np.where(ia < 0, np.int64(-2**63)-ia, ia)
    ib = np.where(ib < 0, np.int64(-2**63)-ib, ib)
    return np.abs(ia-ib)


def ulp32(a, b):
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    ia, ib = a.view(np.int32).astype(np.int64), b.view(np.int32).astype(np.int64)
    ia = np.where(ia < 0, -2**31-ia, ia)
    ib = np.where(ib < 0, -2**31-ib, ib)
    return np.abs(ia-ib)


@NEEDS_USD_SCENE
@pytest.mark.parametrize("mount", lab_scene.TOOL_MOUNTS)
def test_index_tables_resolve_per_world(mount):
    sc = scene(mount)
    m = sc.model
    tables = rk.resolve_index_tables(m.body_label, m.body_world.numpy(), N, mount)
    world = m.body_world.numpy()
    for name in ("hand", "tool", "park_hand"):
        assert np.array_equal(world[tables[name]], np.arange(N))
    assert np.array_equal(world[tables["finger"]], np.repeat(np.arange(N)[:, None], 2, 1))
    labels = list(m.body_label)
    for e in range(N):
        assert labels[tables["hand"][e]].startswith(f"/World/envs/env_{e}/Franka_0/")
        assert labels[tables["finger"][e][0]].endswith("/panda_leftfinger")
        assert labels[tables["finger"][e][1]].endswith("/panda_rightfinger")
        tool = labels[tables["tool"][e]]
        assert tool.endswith("/panda_hand/CuttingTool") if mount == "rigid_mount_v1" else tool == f"/World/envs/env_{e}/Tool"
    # the welded B1 tool is interleaved in the link order (arithmetic layouts are not trusted)
    if mount == "rigid_mount_v1":
        assert tables["tool"][0] == tables["hand"][0]+1 and tables["finger"][0][0] == tables["hand"][0]+2
    bad = world.copy()
    bad[tables["tool"][1]] = 0
    with pytest.raises(RuntimeError, match="crosses worlds"):
        rk.resolve_index_tables(m.body_label, bad, N, mount)


def _edge_launch(body_q_np, frames_chain, hand_from_edge, source, hand):
    M = len(source)
    with wp.ScopedDevice(DEVICE):
        meas = wp.zeros((M, 7), dtype=wp.float64)
        nom = wp.zeros((M, 7), dtype=wp.float64)
        status = wp.zeros(M, dtype=wp.int32)
        mount = np.asarray(hand_from_edge, float)
        wp.launch(rk.edge_pose_kernel, M, inputs=[
            wp.array(body_q_np, dtype=wp.transform), wp.array(source, dtype=wp.int32), wp.array(hand, dtype=wp.int32),
            wp.array(np.stack([c[:3] for c in frames_chain]), dtype=wp.vec3d),
            wp.array(np.stack([c[3:] for c in frames_chain]), dtype=wp.vec4d), len(frames_chain),
            wp.vec3d(*mount[:3]), wp.vec4d(*mount[3:]), meas, nom, status])
    return meas.numpy(), nom.numpy(), status.numpy()


@NEEDS_USD_SCENE
@pytest.mark.parametrize("mount", lab_scene.TOOL_MOUNTS)
def test_edge_kernel_matches_frozen_numpy_compose(mount):
    sc = scene(mount)
    spec = sc.spec
    chain = ([np.asarray(spec.hand_from_tool_xyz_xyzw, float), np.asarray(spec.tool_from_edge_xyz_xyzw, float)]
             if mount == "rigid_mount_v1" else [np.asarray(spec.tool_from_edge_xyz_xyzw, float)])
    M = 4096
    rng = np.random.default_rng(11)
    q = Rotation.random(M, random_state=3).as_quat().astype(np.float32)
    body_q = np.c_[rng.uniform(-1, 1, (M, 3)), q].astype(np.float32)
    idx = np.arange(M, dtype=np.int32)
    meas, nom, status = _edge_launch(body_q, chain, sc.hand_from_edge, idx, idx)
    ref_m, ref_n = rk.numpy_edge_reference(body_q, idx, idx, chain, sc.hand_from_edge)
    assert np.all(status == 0)
    assert ulp64(meas, ref_m).max() <= 1, "measured edge differs from scene_frames.compose_pose by > 1 ulp"
    assert ulp64(nom, ref_n).max() <= 1, "nominal edge differs from robot.compose_tool_pose by > 1 ulp"
    # real model poses: the B1 chain equals hand o task.hand_from_edge (same rigid transform, different rounding)
    state = fk_body_q(sc)
    tables = rk.resolve_index_tables(sc.model.body_label, sc.model.body_world.numpy(), N, mount)
    source = tables["hand"] if mount == "rigid_mount_v1" else tables["tool"]
    meas, nom, status = _edge_launch(state.body_q.numpy(), chain, sc.hand_from_edge, source, tables["hand"])
    ref_m, ref_n = rk.numpy_edge_reference(state.body_q.numpy(), source, tables["hand"], chain, sc.hand_from_edge)
    assert np.all(status == 0) and ulp64(meas, ref_m).max() <= 1 and ulp64(nom, ref_n).max() <= 1
    if mount == "rigid_mount_v1":
        assert np.abs(meas[:, :3]-nom[:, :3]).max() < 1e-9
        dq = np.abs(np.abs(np.sum(meas[:, 3:]*nom[:, 3:], 1))-1.)
        assert dq.max() < 1e-12
    else:
        # the free tool starts at the nominal hand mount (authored at HOME hand pose o hand_from_tool)
        assert np.abs(meas[:, :3]-nom[:, :3]).max() < 2e-6
    # non-finite input is flagged, never raised
    bad = state.body_q.numpy().copy()
    bad[source[1], 0] = np.nan
    _, _, status = _edge_launch(bad, chain, sc.hand_from_edge, source, tables["hand"])
    assert status.tolist() == [0, 1, 0]


def _pads(sc):
    from robotic_cutting_sim.finger_geometry import discover_pads
    hand = lab_scene.hand_body_path(sc.stage, "/World/envs/env_0/Franka_0")
    return discover_pads(sc.stage, hand)


@NEEDS_USD_SCENE
@pytest.mark.parametrize("mount", lab_scene.TOOL_MOUNTS)
def test_pad_box_kernel_matches_pad_boxes_edge(mount):
    from robotic_cutting_sim.finger_geometry import pad_boxes_edge
    from robotic_cutting_sim.tool_profile import handle_box_edge
    sc = scene(mount)
    pads = _pads(sc)
    tables = rk.resolve_index_tables(sc.model.body_label, sc.model.body_world.numpy(), N, mount)
    frames = rk.RobotFrames(tables, sc.spec, sc.hand_from_edge, mount, device=DEVICE,
                            pad_corners=np.asarray([p["corners_body_m"] for p in pads], float),
                            handle_box=handle_box_edge(task().tool_profile))
    rng = np.random.default_rng(5)
    worst = 0
    for trial in range(40):
        q = sc.initial_q.copy()
        if trial:
            q = q+rng.normal(0, .05, q.shape).astype(np.float32)
            # keep the free-joint quaternions unit (B2)
            if mount == "physical_grasp_v1":
                starts = sc.model.joint_q_start.numpy()
                for j in np.flatnonzero(sc.model.joint_type.numpy() == 4):
                    s = starts[j]
                    q[s+3:s+7] /= np.linalg.norm(q[s+3:s+7])
        state = fk_body_q(sc, q.astype(np.float32))
        frames.launch_edge(state.body_q)
        frames.launch_pad_boxes(state.body_q)
        body_q = state.body_q.numpy()
        edge = frames.measured_edge.numpy()
        centers = frames.box_centers.numpy().reshape(N, 3, 3)
        halves = frames.box_halves.numpy().reshape(N, 3, 3)
        for e in range(N):
            poses = {pads[j]["body_path"]: body_q[tables["finger"][e][j]] for j in range(2)}
            ref = np.asarray([handle_box_edge(task().tool_profile), *pad_boxes_edge(pads, poses, edge[e])], np.float32)
            worst = max(worst, ulp32(centers[e], ref[:, 0]).max(), ulp32(halves[e], ref[:, 1]).max())
    assert worst <= 1, f"pad boxes differ from finger_geometry.pad_boxes_edge by {worst} f32 ulp"


@NEEDS_USD_SCENE
@pytest.mark.parametrize("mount", lab_scene.TOOL_MOUNTS)
def test_wrench_kernel_isolated_per_world_and_lever_arm(mount):
    sc = scene(mount)
    tables = rk.resolve_index_tables(sc.model.body_label, sc.model.body_world.numpy(), N, mount)
    frames = rk.RobotFrames(tables, sc.spec, sc.hand_from_edge, mount, device=DEVICE)
    state = fk_body_q(sc)
    base = np.random.default_rng(2).normal(0, 1, (sc.model.body_count, 6)).astype(np.float32)
    for k in range(N):
        pending = np.zeros((N, 6))
        pad = np.zeros((N, 3, 6))
        external = np.zeros((N, 6))
        pending[k] = [1.5, -2., 3.25, .1, -.2, .05]
        pad[k, 1] = [.3, .2, -.1, .01, 0., .02]
        pad[k, 2] = [-.3, .1, .4, 0., .03, -.01]
        external[k] = [0., 0., -.5, 0., 0., 0.]
        frames.pending_wrench.assign(pending)
        frames.pad_wrench.assign(pad)
        frames.external_wrench.assign(external)
        body_f = wp.array(base, dtype=wp.spatial_vector, device=DEVICE)
        frames.launch_wrench(state.body_q, sc.model.body_com, body_f, .002)
        after = body_f.numpy()
        changed = np.flatnonzero(np.any(after != base, 1))
        expected = sorted([int(tables["tool"][k]), *map(int, tables["finger"][k])])
        assert changed.tolist() == expected, f"env {k} wrench touched bodies {changed.tolist()}"
        assert np.array_equal(frames.applied_material.numpy()[k], pending[k])
        assert not frames.applied_material.numpy()[np.arange(N) != k].any()
        ref = rk.numpy_wrench_reference(state.body_q.numpy(), sc.model.body_com.numpy(), tables["tool"],
                                        tables["hand"], tables["finger"], sc.spec.tool_from_edge_xyz_xyzw[:3],
                                        pending, pad, external)
        for body in expected:
            np.testing.assert_allclose(after[body]-base[body], ref[body], rtol=2e-6, atol=2e-6)
        impulse = frames.applied_impulse.numpy()
        assert np.all(impulse[np.arange(N) != k] == 0) and np.any(impulse[k] != 0)
        frames.applied_impulse.zero_()
    # feedback off: material rows are not applied, external still is
    frames.pending_wrench.assign(np.ones((N, 6)))
    frames.external_wrench.zero_()
    body_f = wp.zeros(sc.model.body_count, dtype=wp.spatial_vector, device=DEVICE)
    frames.launch_wrench(state.body_q, sc.model.body_com, body_f, .002, feedback=False)
    assert not body_f.numpy().any() and not frames.applied_material.numpy().any()


@NEEDS_USD_SCENE
def test_masked_reset_of_wrench_buffers_touches_only_masked_rows():
    sc = scene("rigid_mount_v1")
    tables = rk.resolve_index_tables(sc.model.body_label, sc.model.body_world.numpy(), N, "rigid_mount_v1")
    frames = rk.RobotFrames(tables, sc.spec, sc.hand_from_edge, "rigid_mount_v1", device=DEVICE)
    for array in (frames.pending_wrench, frames.external_wrench, frames.applied_material):
        array.assign(np.arange(N*6, dtype=float).reshape(N, 6)+1.)
    frames.pad_wrench.assign(np.ones((N, 3, 6)))
    frames.applied_impulse.assign(np.ones((N, 6), np.float32))
    mask = wp.array(np.array([False, True, False]), dtype=wp.bool, device=DEVICE)
    frames.reset_masked(mask)
    for array in (frames.pending_wrench, frames.external_wrench, frames.applied_material, frames.applied_impulse):
        values = array.numpy()
        assert not values[1].any() and values[0].all() and values[2].all()
    pad = frames.pad_wrench.numpy()
    assert not pad[1].any() and pad[0].all() and pad[2].all()


@NEEDS_USD_SCENE
def test_rigid_validity_flags_only_the_bad_world():
    sc = scene("rigid_mount_v1")
    tables = rk.resolve_index_tables(sc.model.body_label, sc.model.body_world.numpy(), N, "rigid_mount_v1")
    state = fk_body_q(sc)
    q = state.joint_q.numpy().copy()
    per = len(q)//N
    q[per+3] = np.nan
    with wp.ScopedDevice(DEVICE):
        joint_q = wp.array(q, dtype=float)
        reason = wp.zeros(N, dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        wp.launch(rk.rigid_validity_kernel, N, inputs=[
            joint_q, wp.array(np.arange(N+1, dtype=np.int32)*per, dtype=wp.int32), state.body_q,
            wp.array(tables["tool"], dtype=wp.int32), wp.array([5], dtype=int), 256,
            wp.array(np.array([0, 0, 3], np.int32), dtype=wp.int32), 3, wp.zeros(N, dtype=wp.int32), reason, count])
    assert reason.numpy().tolist() == [0, rk.RIGID_NONFINITE_STATE, rk.RIGID_IK_STREAK]
    assert count.numpy()[0] == 2
    assert rk.decode_rigid_reason(reason.numpy()[1]) == ["RIGID_NONFINITE_STATE"]


@NEEDS_USD_SCENE
def test_grasp_monitor_kernel_matches_host_monitor():
    from robotic_cutting_sim.grasp_monitor import GraspMonitor
    from robotic_cutting_sim.scene_frames import compose_pose, invert_transform, pose_matrix
    sc = scene("physical_grasp_v1")
    tables = rk.resolve_index_tables(sc.model.body_label, sc.model.body_world.numpy(), N, "physical_grasp_v1")
    state = fk_body_q(sc)
    body_q = state.body_q.numpy().copy()
    hand = body_q[tables["hand"][0]].astype(float)
    tool = body_q[tables["tool"][0]].astype(float)
    reference = invert_transform(pose_matrix(hand)) @ pose_matrix(tool)
    # world 1 slips 5 mm along the handle, world 2 rotates 40 deg (loss)
    body_q[tables["tool"][1]] = compose_pose(body_q[tables["tool"][1]].astype(float), [0, 0, .005, 0, 0, 0, 1])
    body_q[tables["tool"][2]] = compose_pose(body_q[tables["tool"][2]].astype(float),
                                             [0, 0, 0, *Rotation.from_euler("x", 40, degrees=True).as_quat()])
    with wp.ScopedDevice(DEVICE):
        integral = np.zeros((N, 16), np.float32)
        integral[:, 8] = 10.
        integral[:, 9] = 10.
        out = {k: wp.zeros(N, dtype=float) for k in ("unsupported", "slip", "angle")}
        lost = wp.zeros(N, dtype=wp.int32)
        status = wp.zeros(N, dtype=wp.int32)
        g = __import__("robotic_cutting_sim.grasp_monitor", fromlist=["GraspLossConfig"]).GraspLossConfig()
        wp.launch(rk.grasp_monitor_kernel, N, inputs=[
            wp.array(body_q, dtype=wp.transform), wp.array(tables["hand"], dtype=wp.int32),
            wp.array(tables["tool"], dtype=wp.int32), wp.array(np.repeat(reference[None, :3, :3], N, 0), dtype=wp.mat33d),
            wp.array(np.repeat(reference[None, :3, 3], N, 0), dtype=wp.vec3d), wp.array(integral, dtype=float), .02,
            g.translation_warning_m, g.rotation_warning_rad, g.translation_loss_m, g.rotation_loss_rad,
            g.unsupported_loss_s, 10, out["unsupported"], lost, out["slip"], out["angle"], status])
    for e in range(N):
        monitor = GraspMonitor(hand, tool, sc.spec)
        host = monitor.update(body_q[tables["hand"][e]].astype(float), body_q[tables["tool"][e]].astype(float),
                              time_s=.02, bilateral_loaded_tool_fraction=1.)
        assert abs(host["slip_since_settling_m"]-out["slip"].numpy()[e]) < 2e-6
        assert abs(host["rotation_since_settling_rad"]-out["angle"].numpy()[e]) < 2e-5
        assert rk.GRASP_STATUS_NAMES[status.numpy()[e]] == host["status"]
        assert bool(lost.numpy()[e]) == host["grasp_lost"]


def test_settle_gates_match_frozen_check_dictionary():
    """grasp_batch gates (device) decide as isaac_lab_env._settle_physical_grasp's checks (host) per world."""
    from robotic_cutting_sim.batched import grasp_batch
    from robotic_cutting_sim.physical_tool import ToolGraspConfig
    grasp = ToolGraspConfig()
    rng = np.random.default_rng(9)
    worlds = []
    for e in range(64):
        rows = []
        for _ in range(int(rng.integers(0, 4)) if e % 9 == 0 else 26):
            rows.append({"bilateral_substep_fraction": float(rng.choice([1., .9, .97])),
                         "relative_translation_drift_m": float(rng.uniform(0, .0025)),
                         "relative_rotation_drift_rad": float(rng.uniform(0, .06)),
                         "tool_linear_velocity_world_m_s": rng.uniform(-.013, .013, 3),
                         "tool_angular_velocity_world_rad_s": rng.uniform(-.13, .13, 3),
                         "max_tool_rigid_penetration_m": float(rng.uniform(0, .0011)),
                         "unintended_tool_contact_slots": int(rng.random() < .02)})
        worlds.append(rows)
    acc = np.zeros((len(worlds), grasp_batch.ACC_WIDTH), np.float32)
    for e, rows in enumerate(worlds):
        for r in rows:
            acc[e] += [1., r["bilateral_substep_fraction"], 0, 0, 0, 0, 0, r["unintended_tool_contact_slots"]]
            acc[e, 2:7] = np.maximum(acc[e, 2:7], [r["relative_translation_drift_m"], r["relative_rotation_drift_rad"],
                                                   np.linalg.norm(r["tool_linear_velocity_world_m_s"]),
                                                   np.linalg.norm(r["tool_angular_velocity_world_rad_s"]),
                                                   r["max_tool_rigid_penetration_m"]])
    with wp.ScopedDevice(DEVICE):
        failed = wp.zeros(len(worlds), dtype=wp.int32)
        count = wp.zeros(1, dtype=wp.int32)
        grasp_batch.launch_gates(wp.array(acc, dtype=float), grasp, failed, count, DEVICE)
    decided = 0
    for e, rows in enumerate(worlds):
        host = grasp_batch.host_gate_reference(rows, grasp)
        device = set(grasp_batch.decode_checks(failed.numpy()[e]))
        host_failed = {k for k, ok in host.items() if not ok}
        if not rows:
            assert "nonempty_stability_window" in device
            continue
        # float32 accumulation may only flip a decision exactly at a threshold; none of these samples sit there
        assert device == host_failed, (e, device, host_failed)
        decided += 1
    assert decided > 40 and count.numpy()[0] == sum(bool(failed.numpy()[e]) for e in range(len(worlds)))
