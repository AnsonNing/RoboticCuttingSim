"""P2-3 prerequisites in the batched rubber path (ADR 0023 §20): the P2-2 single-env opt-in options run batched.

Options (single-env ids, ADR 0023 §19.1): F8 ``dirichlet="consistent_mass_projection_v1"``, F9
``traction_work="grid_displacement_v1"``, F10 ``state_precision="f64_shadow_v1"``. A batch runs the options its single
references were built with. Checked here, on the Warp CPU device:

- L0: every array of every env, including the option state (shadows, projection rows / blocks / inverses / scratch,
  the discarded v1 traction work) and diagnostics, markers, wrenches, is bitwise equal to the single-env reference
  after every control step: N = 1 and N = 3 heterogeneous, per option and all together; the two-grip case C (partial
  component masks, two groups, rotation) with the projection;
- L1: perturbing one env leaves the others bitwise unchanged; the masked partial reset (vars()-complete, option
  arrays included) equals a fresh env and the reset env then evolves like a fresh env; a reset with resampled
  material (the mass enters the F8 blocks) equals the single reference built with that material;
- defaults: no option array, receipt reports the v1 ids; mixed options in one batch are refused;
- the constrained-node BC launch (``constrained_list_v1``, default) is bitwise equal to the all-node launch.
"""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_batch as rb
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.rubber_layout import GROUP_ARRAYS, NODE_ARRAYS, PARTICLE_ARRAYS, TRACTION_ARRAYS
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_material import MaterialValues

STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)
F8 = {"dirichlet": "consistent_mass_projection_v1"}
F9 = {"traction_work": "grid_displacement_v1"}
F10 = {"state_precision": "f64_shadow_v1"}
ALL = {**F8, **F9, **F10}
OPTION_SETS = {"F8": F8, "F9": F9, "F10": F10, "F8_F9_F10": ALL}
HETERO = [dict(), dict(L=0.009, W=0.006, mu=7e5, rho=1050.0, ft=0.4), dict(L=0.015, mu=3e5, rho=1200.0, ft=1.2)]
OPTION_NAMES = ("work_scratch", "F64", "x64", "proj_row_node", "proj_row_comp", "proj_row_group", "proj_row_block",
                "proj_row_local", "proj_rhs", "proj_vpresc", "proj_mu", "proj_block_offset", "proj_block_size",
                "proj_inv_offset", "proj_inv", "proj_mu_node")


def _live(options, L=0.012, W=0.003, t=0.003, n_t=2, mu=4e5, rho=1150.0, ft=0.8, case="A", guard=None):
    if case == "A":
        spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="A",
                                  traction_regions=(("tip", (L-0.0015, -1, -1), (1, 1, 1)),))
    else:
        spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=n_t, case="C", end_mask="x",
                                  traction_regions=(("mid", (L/2-0.00075, -1, -1), (L/2+0.00075, 1, 1)),))
    mat = MaterialValues(mu0=mu, kappa=20*mu, density=rho, friction_table=ft)
    return LiveRubberMaterial(spec, material=mat, control_dt=2e-4, gravity=(0, 0, -9.81), step_material=STEP_MATERIAL,
                              table_plane=(0.0, 0.0, 1.0, 0.0), stable_step_guard=guard, **options)


def _inputs(k, N, G=1):
    tr = np.zeros((N, G, 3))
    for e in range(N):
        tr[e, 0] = [0.0, 0.0, 2e-5*(k+1)*(e+1)]
        if G > 1:
            tr[e, 1] = [1.5e-5*(k+1)*(e+1), 0.0, 0.0]
    force = np.zeros((N, 3))
    force[:, 0] = 1e-3*np.arange(1, N+1)
    force[:, 2] = -2e-3*np.arange(1, N+1)
    return tr, force


def _rotations(k, N, G):
    from scipy.spatial.transform import Rotation
    R = np.tile(np.eye(3), (N, G, 1, 1))
    for e in range(N):
        R[e, G-1] = Rotation.from_rotvec([0.02*(k+1)*(e+1), 0.0, 0.0]).as_matrix()
    return R


def _single_array(s, name, G, Gt):
    a = getattr(s, name).numpy()
    if name in GROUP_ARRAYS:
        return a[:G]
    if name in TRACTION_ARRAYS:
        return a[:Gt]
    if name.startswith("proj_row") or name in ("proj_rhs", "proj_vpresc", "proj_mu"):
        return a[:s.proj_rows]
    return a


def _compare(batch, e, live):
    snap = batch.solver.snapshot(e)
    s = live.solver
    G, Gt = s.G, s.Gt
    names = PARTICLE_ARRAYS+NODE_ARRAYS+GROUP_ARRAYS+TRACTION_ARRAYS+("cell_node", "cell_offsets", "cell_particles",
                                                                       "index", "work", "violations", "cfl_violations")
    option_names = [n for n in OPTION_NAMES if hasattr(s, n)]
    assert set(option_names) == {n for n in OPTION_NAMES if n in snap}, "option array set"
    for name in names+tuple(option_names):
        assert np.array_equal(snap[name], _single_array(s, name, G, Gt)), name
    assert np.array_equal(batch.summary.numpy()[e], s.diagnostics.summary.numpy()[0]), "diagnostics"
    K = batch.K
    assert np.array_equal(batch.marker_x.numpy()[e*K:(e+1)*K], s.diagnostics.marker_x.numpy()), "markers"
    assert np.array_equal(batch.marker_v.numpy()[e*K:(e+1)*K], s.diagnostics.marker_v.numpy()), "marker velocity"
    assert np.array_equal(batch.wrench.numpy()[e, :G], live.last_wrench), "wrench"


def _advance_all(batch, singles, k, case="A"):
    N, G = batch.num_envs, batch.G
    tr, force = _inputs(k, N, G)
    rot = _rotations(k, N, G) if case == "C" else None
    active = np.ones((N, G), np.int32)
    if case == "C" and k == 1 and N > 1:
        active[1, G-1] = 0                       # env 1 releases its second grip for one control step
    batch.advance(translation=tr, traction=force, rotation=rot, active=active)
    for e, live in enumerate(singles):
        motions = {g: {"translation": tr[e, g], "active": bool(active[e, g])} for g in range(G)}
        if rot is not None:
            for g in range(G):
                motions[g]["rotation"] = rot[e, g]
        live.advance(motions, traction={0: force[e]})


def _run_l0(options, kwargs_list, steps=3, bc_launch=None):
    singles = [_live(options, **kw) for kw in kwargs_list]
    batch = rmb.BatchedRubberMaterial([_live(options, **kw) for kw in kwargs_list], device="cpu", parity_mode=True,
                                      bc_launch=bc_launch)
    case = kwargs_list[0].get("case", "A")
    for k in range(steps):
        _advance_all(batch, singles, k, case)
        for e, live in enumerate(singles):
            _compare(batch, e, live)
    return batch, singles


@pytest.mark.parametrize("label", sorted(OPTION_SETS))
def test_l0_single_env_batch(label):
    batch, _ = _run_l0(OPTION_SETS[label], [dict()])
    receipt = batch.receipt()["solver"]["options"]
    for key, value in OPTION_SETS[label].items():
        assert receipt[key] == value


@pytest.mark.parametrize("label", sorted(OPTION_SETS))
def test_l0_heterogeneous_batch(label):
    batch, singles = _run_l0(OPTION_SETS[label], HETERO)
    assert len(set(batch.packed.particle_counts.tolist())) == 3
    assert any(np.abs(s.solver.impulse.numpy()).max() > 0 for s in singles)     # table contact active
    if "dirichlet" in OPTION_SETS[label]:
        assert batch.solver.proj_rows == sum(s.solver.proj_rows for s in singles) > 0
        assert len(set(s.solver.proj_rows for s in singles)) > 1                # heterogeneous constrained sets


def test_l0_two_grips_partial_masks_with_rotation():
    kws = [dict(case="C"), dict(case="C", L=0.009, mu=6e5, rho=1050.0), dict(case="C", L=0.015, mu=3e5)]
    batch, singles = _run_l0(ALL, kws, steps=3)
    assert batch.G == 2
    blocks = singles[0].solver.projection_receipt["blocks"]
    assert {b["component"] for b in blocks} == {"x", "y", "z"} and len({b["nodes"] for b in blocks}) > 1


def test_l0_with_guard_and_options():
    _run_l0(ALL, [dict(guard=1.0), dict(L=0.009, W=0.006, mu=7e5, rho=1050.0, ft=0.4, guard=1.0)], steps=2)


def test_l1_perturbation_isolation_with_options():
    a = rmb.BatchedRubberMaterial([_live(ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True)
    b = rmb.BatchedRubberMaterial([_live(ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True)
    for k in range(3):
        tr, force = _inputs(k, 3)
        a.advance(translation=tr, traction=force)
        tr2, force2 = tr.copy(), force.copy()
        tr2[1, 0, 2] += 3e-4
        force2[1, 0] = 0.05
        b.advance(translation=tr2, traction=force2)
    for e in (0, 2):
        sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
        assert set(OPTION_NAMES) <= set(sa)
        for name in sa:
            assert np.array_equal(sa[name], sb[name]), (e, name)
        assert np.array_equal(a.summary.numpy()[e], b.summary.numpy()[e])
    for name in ("x", "x64", "proj_mu", "work"):
        assert not np.array_equal(a.solver.snapshot(1)[name], b.solver.snapshot(1)[name]), name


def test_reset_classification_is_vars_complete_with_options():
    batch = rmb.BatchedRubberMaterial([_live(ALL)], device="cpu", parity_mode=True)
    names = set(batch.solver.device_arrays())
    declared = set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS)
    for table in (rmb.OPTION_RESET_ARRAYS, rmb.OPTION_CONSTANT_ARRAYS):
        for value in table.values():
            declared |= set(value)
    assert names == declared, names ^ declared


def test_partial_reset_with_options_equals_fresh_env_and_keeps_others():
    batch = rmb.BatchedRubberMaterial([_live(ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True)
    fresh = rmb.BatchedRubberMaterial([_live(ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True)
    for k in range(2):
        tr, force = _inputs(k, 3)
        batch.advance(translation=tr, traction=force)
    before = {e: batch.solver.snapshot(e) for e in (0, 2)}
    assert np.abs(before[0]["proj_mu"]).max() > 0 and np.abs(before[0]["work_scratch"]).max() > 0
    batch.reset([1])
    snap, ref = batch.solver.snapshot(1), fresh.solver.snapshot(1)
    for name in snap:
        if name != "index":
            assert np.array_equal(snap[name], ref[name]), name
    for name in rmb.MATERIAL_RESET_ARRAYS:
        assert np.array_equal(getattr(batch, name).numpy()[1], getattr(fresh, name).numpy()[1]), name
    for e in (0, 2):
        after = batch.solver.snapshot(e)
        for name in after:
            if name != "index":
                assert np.array_equal(after[name], before[e][name]), (e, name)
    tr, force = _inputs(0, 3)
    batch.advance(translation=tr, traction=force)
    fresh.advance(translation=tr, traction=force)
    for name, value in batch.solver.snapshot(1).items():
        assert np.array_equal(value, fresh.solver.snapshot(1)[name]), name


def test_reset_with_resampled_material_rebuilds_the_projection_blocks():
    ranges = {"mu0": (3e5, 1e6), "density": (1000.0, 1200.0), "friction_table": (0.3, 1.5)}
    batch = rmb.BatchedRubberMaterial([_live(ALL, **kw) for kw in HETERO], device="cpu", parity_mode=True,
                                      ranges=ranges)
    inv0 = batch.solver.snapshot(1)["proj_inv"]
    tr, force = _inputs(0, 3)
    batch.advance(translation=tr, traction=force)
    new = MaterialValues(mu0=8e5, kappa=1.6e7, density=1010.0, friction_table=1.1)
    batch.reset([1], materials=[new])
    single = _live(ALL, **{**HETERO[1], "mu": 8e5, "rho": 1010.0, "ft": 1.1})
    assert not np.array_equal(batch.solver.snapshot(1)["proj_inv"], inv0)
    for k in range(2):
        tr, force = _inputs(k, 3)
        batch.advance(translation=tr, traction=force)
        single.advance({0: {"translation": tr[1, 0]}}, traction={0: force[1]})
        _compare(batch, 1, single)
    batch.reset([1])                                     # back to the packed material: the packed inverse again
    assert np.array_equal(batch.solver.snapshot(1)["proj_inv"], inv0)


def test_defaults_add_no_option_array_and_report_v1_ids():
    batch = rmb.BatchedRubberMaterial([_live({})], device="cpu", parity_mode=True)
    assert not any(hasattr(batch.solver, n) for n in OPTION_NAMES+("reference64", "env_of_row"))
    opts = batch.receipt()["solver"]["options"]
    assert (opts["dirichlet"], opts["traction_work"], opts["state_precision"]) == (
        "lumped_v1", "particle_velocity_v1", "float32_v1")
    assert opts["bc_launch"] == rb.DEFAULT_BC_LAUNCH == "constrained_list_v1"
    assert opts["bc_nodes"] == int(np.count_nonzero(batch.packed.host["node_group"] >= 0)) > 0


def test_mixed_options_are_refused():
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live({}), _live(F8)], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial([_live(F10), _live(F9)], device="cpu", parity_mode=True)
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_SOLVER"):
        rmb.BatchedRubberMaterial([_live({})], device="cpu", parity_mode=True, bc_launch="nope")


@pytest.mark.parametrize("launch", ["constrained_list_v1", "motion_table_v1"])
@pytest.mark.parametrize("options", [{}, ALL], ids=["defaults", "F8_F9_F10"])
def test_bc_launch_variants_equal_the_all_node_launch(options, launch):
    kws = [dict(case="C"), dict(case="C", L=0.009, mu=6e5, rho=1050.0)]
    a = rmb.BatchedRubberMaterial([_live(options, **kw) for kw in kws], device="cpu", parity_mode=True,
                                  bc_launch="all_nodes_v1")
    b = rmb.BatchedRubberMaterial([_live(options, **kw) for kw in kws], device="cpu", parity_mode=True,
                                  bc_launch=launch)
    assert b.solver.bc_node_count < b.solver.total_nodes
    for k in range(3):
        tr, force = _inputs(k, 2, 2)
        rot = _rotations(k, 2, 2)
        for m in (a, b):
            m.advance(translation=tr, traction=force, rotation=rot)
        for e in range(2):
            sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
            for name in sa:
                assert np.array_equal(sa[name], sb[name]), (k, e, name)
        assert np.array_equal(a.wrench.numpy(), b.wrench.numpy()) and np.array_equal(a.summary.numpy(),
                                                                                     b.summary.numpy())
    for m in (a, b):                              # direct substeps past the control step (beyond the motion table)
        m.solver.step()
        m.solver.step()
    for e in range(2):
        sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
        assert int(sa["index"][0]) > a.solver.substeps
        for name in sa:
            assert np.array_equal(sa[name], sb[name]), ("past", e, name)
