"""Review R1 (P2_PREREQ_P2_3_REVIEW.md): the ``motion_table_v1`` BC launch must use the current group motion when the
caller steps the solver directly (``solver.step()``) after ``begin_control_step()`` or after a full / partial
``reset()``, without ``set_group_motions()``. Before the fix the default launch replayed the previous control step's
table (the grip lifted another 0.2 mm, max |dx| 2.0e-4 m). Each sequence is compared byte-wise, every array of every
env, with ``all_nodes_v1`` (the reference launch, which evaluates the group maps directly). Warp CPU device.
"""
import numpy as np
import pytest

from test_rubber_batched_options import ALL, _inputs, _live, _rotations, _same_bytes

from robotic_cutting_sim.batched import rubber_material_batch as rmb

LAUNCHES = ("all_nodes_v1", "motion_table_v1")


def _pair(options, case):
    kws = [dict(case=case), dict(case=case, L=0.009, mu=6e5, rho=1050.0)]
    return [rmb.BatchedRubberMaterial([_live(options, **kw) for kw in kws], device="cpu", parity_mode=True,
                                      bc_launch=launch) for launch in LAUNCHES]


def _lift(m, case):
    """One control step through advance(): grip 0 lifts 0.2 mm (case C: grip 1 also rotates)."""
    N, G = m.num_envs, m.G
    tr = np.zeros((N, G, 3))
    tr[:, 0, 2] = 2e-4
    kwargs = dict(translation=tr)
    if case == "C":
        kwargs["rotation"] = _rotations(2, N, G)
    m.advance(**kwargs)


def _direct(m, n):
    for _ in range(n):
        m.solver.step()


def _assert_equal(a, b, tag):
    assert a.solver.bc_launch == "all_nodes_v1" and b.solver.bc_launch == "motion_table_v1"
    for e in range(a.num_envs):
        sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
        assert sa.keys() == sb.keys()
        for name in sa:
            assert _same_bytes(sa[name], sb[name]), (tag, e, name)


def _sequence(m, case, seq):
    _lift(m, case)
    n = m.solver.substeps
    if seq == "begin_direct":
        m.begin_control_step()
        _direct(m, n)
    elif seq == "full_reset_direct":
        m.reset()
        _direct(m, n)
    elif seq == "partial_reset_direct":           # shared index past the table: direct evaluation
        m.reset([1])
        _direct(m, n)
    elif seq == "begin_partial_reset_mid_table":  # partial reset in the middle of a hold step (index inside table)
        m.begin_control_step()
        _direct(m, n//2)
        m.reset([1])
        _direct(m, n-n//2)
    elif seq == "moving_partial_reset_mid_table":  # manual moving step, env 1 reset halfway: env 1 must hold
        m.begin_control_step()
        tr = np.zeros((m.num_envs, m.G, 3))
        tr[:, 0, 2] = 4e-4
        m.set_group_motions(translation=tr)
        _direct(m, n//2)
        m.reset([1])
        _direct(m, n-n//2)
    else:
        raise AssertionError(seq)


SEQUENCES = ("begin_direct", "full_reset_direct", "partial_reset_direct", "begin_partial_reset_mid_table",
             "moving_partial_reset_mid_table")


@pytest.mark.parametrize("seq", SEQUENCES)
@pytest.mark.parametrize("case,options", [("A", {}), ("C", ALL)], ids=["A_defaults", "C_F8_F9_F10"])
def test_direct_steps_use_the_current_group_motion(case, options, seq):
    a, b = _pair(options, case)
    for m in (a, b):
        _sequence(m, case, seq)
    _assert_equal(a, b, seq)
    # an advance() after the direct steps is still byte-identical between the launches
    tr, force = _inputs(3, a.num_envs, a.G)
    for m in (a, b):
        m.advance(translation=tr, traction=force)
    _assert_equal(a, b, seq+"+advance")
    assert _same_bytes(a.wrench.numpy(), b.wrench.numpy()) and _same_bytes(a.summary.numpy(), b.summary.numpy())


def test_begin_control_step_makes_the_table_a_hold():
    _, b = _pair({}, "C")
    _lift(b, "C")
    s = b.solver
    M_before = s.motion_M.numpy()
    assert not np.array_equal(M_before[0], M_before[-1]) or not np.array_equal(s.motion_b.numpy()[0],
                                                                              s.motion_b.numpy()[-1])
    b.begin_control_step()
    M, bb = s.motion_M.numpy(), s.motion_b.numpy()
    assert all(np.array_equal(M[k], M[0]) and np.array_equal(bb[k], bb[0]) for k in range(M.shape[0]))
    b.reset()
    M, bb = s.motion_M.numpy(), s.motion_b.numpy()
    assert np.array_equal(M, np.broadcast_to(np.eye(3), M.shape)) and not np.any(bb)
