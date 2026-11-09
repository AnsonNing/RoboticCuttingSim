"""ADR 0022 §8 C3: MuJoCo-Warp on the Warp CPU device, RCS rigid scene (2 Frankas, B1/B2 tool, board, table),
60 substeps of 2 ms (torch-free; ``.venv``). Promotes the ADR 0022 §1.3 probes to tests:

(a) N=1 equals world 0 of N=3 bitwise (``separate_worlds=True`` in both);
(b) world 1 driven into the board (and given a wrench through the per-world wrench kernel): worlds 0 and 2 stay
    bitwise equal to the undisturbed run; world 1 changes;
(c) masked reset of world 2 at substep 30 (joint write + ``SolverMuJoCo.reset(world_mask, flags=0)``): worlds 0
    and 1 stay bitwise equal to the run without reset;
contacts > 0 are asserted (the tool / fingers / links touch the shared statics or each other).
"""
from functools import lru_cache

import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim.batched import lab_scene
from robotic_cutting_sim.batched import robot_kernels as rk

wp.config.quiet = True
DEVICE = "cpu"
SUBSTEPS = 60
DT = .002


@lru_cache(maxsize=None)
def task():
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(512), goal_config(), seed=42, particle_spacing_m=.0025)


@lru_cache(maxsize=None)
def scene(mount, n):
    return lab_scene.build_newton_rigid_scene(task(), n, mount, device=DEVICE)


def world_slices(model, n):
    jw = model.joint_world.numpy()
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    out = []
    for e in range(n):
        joints = np.flatnonzero(jw == e)
        q0, qd0 = q_start[joints.min()], qd_start[joints.min()]
        last = joints.max()+1
        q1 = q_start[last] if last < len(jw) else model.joint_coord_count
        qd1 = qd_start[last] if last < len(jw) else model.joint_dof_count
        out.append((slice(q0, q1), slice(qd0, qd1), np.flatnonzero(model.body_world.numpy() == e)))
    return out


def run(mount, n, *, drive_world=None, reset_world=None, wrench_world=None):
    """60 substeps; Franka 0 of every world tracks a slow joint ramp; returns per-substep q, qd, body_q, nacon."""
    sc = scene(mount, n)
    model = sc.model
    solver = sc.make_solver()
    state0, state1, control = model.state(), model.state(), model.control()
    slices = world_slices(model, n)
    tables = rk.resolve_index_tables(model.body_label, model.body_world.numpy(), n, mount)
    frames = rk.RobotFrames(tables, sc.spec, sc.hand_from_edge, mount, device=DEVICE)
    if wrench_world is not None:
        pending = np.zeros((n, 6))
        pending[wrench_world] = [0., 0., 25., 0., .5, 0.]
        frames.pending_wrench.assign(pending)
    target = model.joint_target_q.numpy() if hasattr(model, "joint_target_q") else model.joint_q.numpy().copy()
    base_target = control.joint_target_q.numpy().copy() if control.joint_target_q is not None else target.copy()
    q_hist, qd_hist, bq_hist, nacon = [], [], [], []
    for k in range(SUBSTEPS):
        targets = base_target.copy()
        for e, (qs, _, _) in enumerate(slices):
            arm = np.arange(qs.start, qs.start+7)
            targets[arm[1]] += .002*k       # all worlds: shoulder pitch ramp (same in every world)
            if e == drive_world:
                targets[arm[1]] += .02*k    # world e: drive the knife down into the board
                targets[arm[3]] += .015*k
        control.joint_target_q.assign(targets)
        if reset_world is not None and k == SUBSTEPS//2:
            q = state0.joint_q.numpy()
            qd = state0.joint_qd.numpy()
            qs, qds, _ = slices[reset_world]
            q[qs] = sc.initial_q[qs]
            qd[qds] = 0.
            state0.joint_q.assign(q)
            state0.joint_qd.assign(qd)
            mask = np.zeros(n+1, bool)
            mask[reset_world] = True
            solver.reset(state0, world_mask=wp.array(mask, dtype=wp.bool, device=DEVICE), flags=0)
        state0.clear_forces()
        frames.launch_wrench(state0.body_q, model.body_com, state0.body_f, DT)
        solver.step(state0, state1, control, None, DT)
        state0, state1 = state1, state0
        q_hist.append(state0.joint_q.numpy().copy())
        qd_hist.append(state0.joint_qd.numpy().copy())
        bq_hist.append(state0.body_q.numpy().copy())
        nacon.append(int(solver.mjw_data.nacon.numpy()[0]))
    return {"q": np.asarray(q_hist), "qd": np.asarray(qd_hist), "body_q": np.asarray(bq_hist), "nacon": nacon,
            "slices": slices, "contact_world": solver.mjw_data.contact.worldid.numpy()[:nacon[-1]].copy()}


def world(rollout, e):
    qs, qds, bodies = rollout["slices"][e]
    return rollout["q"][:, qs], rollout["qd"][:, qds], rollout["body_q"][:, bodies]


def assert_world_equal(a, b, ea, eb, label):
    for name, x, y in zip(("q", "qd", "body_q"), world(a, ea), world(b, eb)):
        assert np.array_equal(x, y), f"{label}: {name} of world {ea} vs {eb} differs (max {np.abs(x-y).max()})"


@pytest.fixture(scope="module", params=lab_scene.TOOL_MOUNTS)
def base3(request):
    return request.param, run(request.param, 3)


def test_n1_equals_world0_of_n3_bitwise(base3):
    mount, three = base3
    one = run(mount, 1)
    assert_world_equal(one, three, 0, 0, "N=1 vs N=3")
    assert np.all(np.asarray(three["nacon"]) > 0), "the rigid scene must have active contacts"
    # identical copies: every world of N=3 equals world 0
    for e in (1, 2):
        assert_world_equal(three, three, 0, e, "identical copies")


def test_driven_and_wrenched_world_leaves_others_bitwise(base3):
    mount, three = base3
    driven = run(mount, 3, drive_world=1, wrench_world=1)
    for e in (0, 2):
        assert_world_equal(driven, three, e, e, "drive world 1")
    q1, _, _ = world(driven, 1)
    q1_base, _, _ = world(three, 1)
    assert np.abs(q1-q1_base).max() > 1e-3, "world 1 must actually respond to the drive"
    assert max(driven["nacon"]) > 0


def test_masked_reset_of_world2_leaves_others_bitwise(base3):
    mount, three = base3
    reset = run(mount, 3, reset_world=2)
    for e in (0, 1):
        assert_world_equal(reset, three, e, e, "masked reset of world 2")
    q2, _, _ = world(reset, 2)
    q2_base, _, _ = world(three, 2)
    assert not np.array_equal(q2[SUBSTEPS//2:], q2_base[SUBSTEPS//2:]), "world 2 must actually be reset"
