"""ADR 0023 §22.2 / §22.5 (P2-4 task B1, RUBBER-CONT-08 a-c): the batched rubber contact path, Warp CPU device.

``BatchedRubberMaterial`` built from ``LiveRubberMaterial(contact=ContactSpec(...))`` references runs
``rubber_contact_batch.BatchedTLMPMRubberContact``: the single solver's contact kick launch and the shared
``rubber_contact`` kernels over all envs. Checked here, byte-wise (dtype, shape, raw bytes):

- L0 (-08 a): N = 1 and N = 3 heterogeneous (geometry, mu_table, mu_pad, material incl. P2, planes, pad / fixture frames
  and their motions) for the incline (tilted gravity, table + moving fixture frame), impact (P2 rod onto a wall with
  per-env initial velocities) and pinch (case A strip, table + two pad frames of two boxes, squeeze / pull / roll, a pad
  released for one step; elastic and P2) scenes: every array of every env (base solver, options, visco state, every
  contact array incl. ct_node_u, ct_xi, ct_obst, ct_vobs, ct_work, the reactions), diagnostics, visco diagnostics,
  markers, group wrenches, the contact block (plane / box wrenches, work columns, contact energy, penetration, counts)
  and the per-pad wrenches [N, G, B, 6] equal the single-env ``TLMPMRubberContact`` reference after every control step;
- L1 (-08 b): perturbing one env's pad motion, friction or plane changes no byte of the other envs; the masked
  partial reset (vars()-complete with the contact arrays) equals a fresh env and evolves like one; a reset with
  resampled frictions equals the single reference built with them;
- refusals (-08 c): mixed contact / no contact, laws, law parameters, obstacle counts, frame kinds;
- contact off: lives without a ContactSpec build the unchanged ``BatchedTLMPMRubber`` (no contact array); an EMPTY
  ContactSpec batch is byte-identical to the batch without contact.

Two placeholder arrays of the single reference have no batched counterpart and are excluded (asserted unused): the
elastic ``TLMPMRubberContact`` (M = 0, a ``TLMPMRubberVisco``) carries ``params64 = [[1.0]]`` (g_inf only), which no
elastic kernel reads, and with ``dirichlet = consistent_mass_projection_v1`` but no constrained node (free scenes) the
single solver keeps placeholder projection arrays and skips the projection, while the batch allocates none (as for
every such batch since P2-3).
"""
import math

import numpy as np
import pytest

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim import rubber_visco_protocol as vp
from robotic_cutting_sim.batched import rubber_contact_batch as rcb
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
from robotic_cutting_sim.batched.rubber_batch import BatchedTLMPMRubber
from robotic_cutting_sim.batched.rubber_layout import CONTACT_ARRAY_KINDS
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_contact import PROJECTION_LAW, ContactFrame, ContactSpec
from robotic_cutting_sim.rubber_material import MaterialValues

P2 = ((0.2, 0.005), (0.15, 0.05))
SLS = ((0.3, 0.01),)
STEP_MATERIAL = MaterialValues(mu0=1e6, kappa=2e7, density=1000.0)
# the protocol's options (rubber_contact_tests_v1 common.options)
OPTS = dict(stable_step_guard=1.0, dirichlet="consistent_mass_projection_v1", constraint_work="kick_average_v1",
            traction_work="grid_displacement_v1", state_precision="f64_shadow_v1")
THETA = math.atan(0.5)
G_INCLINE = (9.81*math.sin(THETA), 0.0, -9.81*math.cos(THETA))
same_bytes = vp.same_bytes


def _material(mu0=4e5, rho=1150.0, terms=(), mu_table=0.5, mu_pad=1.0, M=None):
    prony = tuple(terms)
    if M is not None and len(prony) < M:
        prony = prony+((0.0, 1.0),)*(M-len(prony))
    return MaterialValues(mu0=mu0, kappa=20*mu0, density=rho, friction_table=mu_table, friction_pad=mu_pad,
                          prony=prony)


def _live(spec, material, contact, gravity, control_dt=2e-4, options=OPTS):
    energy = "branch_work_kick_average_v2" if material.prony else "off_v1"
    return LiveRubberMaterial(spec, material=material, control_dt=control_dt, gravity=gravity, contact=contact,
                              step_material=STEP_MATERIAL, strict=False, visco_energy=energy, **options)


# ----------------------------------------------------------------------------- scenes


def _inc(L=0.006, W=0.004, t=0.002, mu_t=0.5, d=0.0, gap=2e-6, mu0=4e5, rho=1150.0, options=OPTS):
    """Block on the table z = -d under tilted gravity (tan theta = 0.5, downslope +x) and a fixture frame (kind
    fixture, friction_table) placed `gap` beyond the block's +x face, moved towards it."""
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=2, case="free")
    stop = ContactFrame(kind="fixture", name="stop", origin=(L+gap+0.0005, W/2, t/2),
                        boxes=(((L+gap+0.0005, W/2, t/2), (0.0005, W, t)),))
    return _live(spec, _material(mu0, rho, mu_table=mu_t), ContactSpec(planes=((0.0, 0.0, 1.0, d),), frames=(stop,)),
                 G_INCLINE, options=options)


def _imp(L=0.012, d=0.0, mu_t=0.0, mu0=4e5, rho=1150.0, terms=P2):
    """P2-type rod (Prony terms padded to M = 2) onto the wall x = -d, no gravity."""
    spec = rs.RubberStripSpec(length=L, width=0.002, thickness=0.002, n_t=2, case="free")
    return _live(spec, _material(mu0, rho, terms, mu_table=mu_t, M=2), ContactSpec(planes=((1.0, 0.0, 0.0, d),)),
                 (0.0, 0.0, 0.0), control_dt=1e-4)


def _pinch(L=0.012, W=0.004, t=0.002, mu_p=0.8, mu_t=0.5, d=0.0, mu0=4e5, rho=1150.0, terms=()):
    """Case A strip (clamp group 0), table z = -d, two pad frames (kind pad) of two boxes each around the tip."""
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=2, case="A")

    def pad(z, name):
        return ContactFrame(kind="pad", name=name, origin=(L-0.003, W/2, z),
                            boxes=(((L-0.004, W/2, z), (0.001, W/2+0.001, 0.0005)),
                                   ((L-0.002, W/2, z), (0.001, W/2+0.001, 0.0005))))
    contact = ContactSpec(planes=((0.0, 0.0, 1.0, d),), frames=(pad(t+0.0005, "top"), pad(-0.0005, "bottom")))
    return _live(spec, _material(mu0, rho, terms, mu_table=mu_t, mu_pad=mu_p, M=2 if terms else None), contact,
                 (0.0, 0.0, -9.81))


INC_HETERO = [dict(), dict(L=0.008, W=0.005, mu_t=0.8, d=-1e-6, mu0=6e5, rho=1050.0), dict(L=0.005, mu_t=0.3, d=2e-6,
                                                                                            mu0=3e5, gap=5e-6)]
IMP_HETERO = [dict(), dict(L=0.009, d=1e-6, mu_t=0.4, mu0=6e5, rho=1050.0, terms=SLS),
              dict(L=0.015, d=-1e-6, mu_t=0.8, mu0=3e5, terms=((0.1, 0.002), (0.25, 0.02)))]
PINCH_HETERO = [dict(), dict(L=0.010, mu_p=0.5, mu_t=0.3, d=-1e-6, mu0=6e5, rho=1050.0),
                dict(L=0.015, W=0.005, mu_p=1.2, mu_t=0.9, d=2e-4, mu0=3e5)]        # env 2: table out of reach
IMP_VELOCITY = [(-0.2, 0.01, 0.0), (-0.3, 0.0, 0.02), (-0.15, -0.02, 0.0)]
SCENES = {"inc": (_inc, INC_HETERO), "imp": (_imp, IMP_HETERO), "pinch": (_pinch, PINCH_HETERO),
          "pinch_p2": (lambda **kw: _pinch(**{"terms": P2, **kw}),
                       [dict(), dict(PINCH_HETERO[1], terms=SLS), dict(PINCH_HETERO[2], terms=((0.1, 0.002),
                                                                                                (0.25, 0.02)))])}


def _frame_inputs(scene, k, N, F):
    """[N, F, ...] END poses of the contact frames at control step k (per-env different motions)."""
    from scipy.spatial.transform import Rotation
    tr = np.zeros((N, F, 3))
    R = np.tile(np.eye(3), (N, F, 1, 1))
    active = np.ones((N, F), np.int32)
    for e in range(N):
        if scene == "inc":
            tr[e, 0] = [-3e-6*(k+1)*(e+1), 0.0, 0.0]
        elif scene.startswith("pinch"):
            sq = 1e-5*(e+1)*min(k+1, 3)
            px = 2e-5*max(0, k-1)*(e+1)
            tr[e, 0] = [px, 0.0, -sq]
            tr[e, 1] = [px, 0.0, sq]
            if e == 1:
                R[e, 0] = Rotation.from_rotvec([0.01*(k+1), 0.0, 0.0]).as_matrix()
            if e == 2 and k == 2:
                active[e, 1] = 0                     # env 2 releases its bottom pad for one control step
    return tr, R, active


def _advance(scene, batch, singles, k, perturb=None):
    N, F = batch.num_envs, batch.solver.ct_F
    tr, R, active = _frame_inputs(scene, k, N, F) if F else (None, None, None)
    if perturb is not None:
        tr, R, active = perturb(tr, R, active)
    if F:
        batch.advance(frame_translation=tr, frame_rotation=R, frame_active=active)
    else:
        batch.advance()
    for e, live in enumerate(singles or ()):
        frames = {f: {"translation": tr[e, f], "rotation": R[e, f], "active": bool(active[e, f])}
                  for f in range(F)} if F else None
        live.advance(frames=frames)


def _build(scene, kws, initial=True):
    make, _ = SCENES[scene]
    lives = [make(**kw) for kw in kws]
    batch = rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)
    if initial and scene == "imp":
        batch.set_initial_velocity(range(len(kws)), IMP_VELOCITY[:len(kws)])
    return batch


def _singles(scene, kws):
    make, _ = SCENES[scene]
    singles = [make(**kw) for kw in kws]
    if scene == "imp":
        for live, vel in zip(singles, IMP_VELOCITY):
            live.solver.set_initial_velocity(vel)
    return singles


def compare_contact(batch, e, live):
    """Byte-wise comparison of batched env e with its single contact reference (module docstring); returns the
    mismatching names (empty = L0)."""
    s = live.solver
    bad = vp.compare_env(batch, e, live)
    if not s.M:
        assert same_bytes(s.params64.numpy(), np.ones((1, 1))) and not hasattr(batch.solver, "params64")
        bad = [b for b in bad if b not in ("params64", "option/visco array set")]
    if getattr(s, "dirichlet", "") == "consistent_mass_projection_v1" and s.proj_rows == 0:
        # F8 with no constrained row (free scenes): the single solver skips the projection and keeps placeholder
        # arrays, the batch allocates none (projection_rows_nonzero, ADR 0023 §20.1)
        assert not batch.solver.projection and not np.any(s.proj_mu_node.numpy())
        unused = {n for n in vp.OPTION_NAMES if n.startswith("proj_")}
        bad = [b for b in bad if b not in unused and b != "option/visco array set"]
    snap = batch.solver.snapshot(e)
    rows = {"plane": s.ct_P, "frame": s.ct_F, "box": s.ct_F*s.ct_B}
    for name, kind in CONTACT_ARRAY_KINDS.items():
        if kind == "alias_node_offset":
            continue
        a = getattr(s, name).numpy()
        if kind in rows:
            a = a[:rows[kind]]
        if name not in snap or not same_bytes(snap[name], a):
            bad.append(name)
    c = batch.contact_read(e)
    ref = live.last_contact
    if set(c) != set(ref):
        bad.append("contact keys")
    for key, value in ref.items():
        if not same_bytes(np.asarray(c.get(key)), np.asarray(value)):
            bad.append(f"contact.{key}")
    pads = [f for f, kind in enumerate(batch.solver.frame_kinds) if kind == "pad"]
    if pads and not same_bytes(batch.pad_wrenches()[e], np.ascontiguousarray(ref["box_wrench"][pads])):
        bad.append("pad_wrench")
    if np.any(batch.pending_wrench.numpy()[e] != 0.0):
        bad.append("pending_wrench")
    return bad


def _run_l0(scene, kws, steps=4):
    singles = _singles(scene, kws)
    batch = _build(scene, kws)
    for k in range(steps):
        _advance(scene, batch, singles, k)
        for e, live in enumerate(singles):
            assert compare_contact(batch, e, live) == [], (scene, k, e)
    return batch, singles


# ----------------------------------------------------------------------------- L0 (RUBBER-CONT-08 a)


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_l0_single_env_batch(scene):
    batch, singles = _run_l0(scene, [dict()])
    assert type(batch.solver) is rcb.BatchedTLMPMRubberContact
    c = singles[0].last_contact
    assert c["impulse_events"] > 0 and np.abs(c["plane_wrench"]).max() > 0     # contact is active
    if scene.startswith("pinch"):
        assert batch.pad_wrenches().shape == (1, 2, 2, 6) and np.abs(c["box_wrench"][:, :, 2]).min() > 0


@pytest.mark.parametrize("scene", sorted(SCENES))
def test_l0_heterogeneous_batch(scene):
    _, kws = SCENES[scene]
    batch, singles = _run_l0(scene, kws)
    assert len(set(batch.packed.particle_counts.tolist())) == 3
    for live in singles:
        assert live.last_contact["impulse_events"] > 0
    works = [tuple(live.solver.ct_work.numpy()[0]) for live in singles]
    assert len(set(works)) == 3                                           # heterogeneous contact histories
    if scene == "inc":
        # the moving fixture was reached in every env (obstacle-side normal work: planes are static)
        assert all(live.solver.ct_work.numpy()[0, 2] != 0.0 for live in singles)
    if scene == "pinch_p2":
        assert batch.solver.M == 2 and all(live.solver.M == 2 for live in singles)


def test_l0_default_options_and_projection_law():
    """The v1 solver options (no F8 / F9 / F10, no guard) and the rejected projection law (kept for the A/B record)
    run byte-identical too."""
    kws = INC_HETERO[:2]
    singles = [_inc(**kw, options={}) for kw in kws]
    batch = rmb.BatchedRubberMaterial([_inc(**kw, options={}) for kw in kws], device="cpu", parity_mode=True)
    for k in range(3):
        _advance("inc", batch, singles, k)
        for e, live in enumerate(singles):
            assert compare_contact(batch, e, live) == [], (k, e)
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    proj = lambda mu: _live(spec, _material(mu_table=mu), ContactSpec(law=PROJECTION_LAW,          # noqa: E731
                                                                      planes=((0.0, 0.0, 1.0, 0.0),)), G_INCLINE)
    singles = [proj(0.3), proj(0.8)]
    batch = rmb.BatchedRubberMaterial([proj(0.3), proj(0.8)], device="cpu", parity_mode=True)
    assert not batch.solver.ct_penalty
    for k in range(3):
        _advance("imp", batch, singles, k)           # no frames
        for e, live in enumerate(singles):
            assert compare_contact(batch, e, live) == [], ("projection", k, e)


# ----------------------------------------------------------------------------- L1 (RUBBER-CONT-08 b)


def _env_bytes(batch, e):
    out = dict(batch.solver.snapshot(e))
    out["summary"] = batch.summary.numpy()[e]
    out["markers"] = batch.markers()[e]
    out["wrench"] = batch.wrench.numpy()[e]
    for name in rcb.CONTACT_MATERIAL_RESET_ARRAYS:
        out[name] = getattr(batch, name).numpy()[e]
    if batch.visco_summary is not None:
        out["visco_summary"] = batch.visco_summary.numpy()[e]
    for name in rmb.MATERIAL_RESET_ARRAYS:
        out[f"material.{name}"] = getattr(batch, name).numpy()[e]
    return out


def _assert_envs_equal(a, b, envs, skip=("index",)):
    """Byte equality of env rows of two batches (``skip``: shared index, or the diagnostics scratch recomputed at the
    end of every control step and not part of the reset state)."""
    for e in envs:
        ea, eb = _env_bytes(a, e), _env_bytes(b, e)
        assert set(ea) == set(eb)
        for name in ea:
            if name not in skip:
                assert same_bytes(ea[name], eb[name]), (e, name)


def _pad_motion_perturbation(tr, R, active):
    tr = tr.copy()
    tr[1, 0, 2] -= 2e-5                           # env 1 squeezes deeper
    tr[1, 1, 0] += 1e-5
    return tr, R, active


@pytest.mark.parametrize("what", ["pad_motion", "friction", "plane"])
def test_l1_perturbing_one_env_leaves_the_others_byte_unchanged(what):
    kws = [dict(kw) for kw in PINCH_HETERO]
    kws_b = [dict(kw) for kw in PINCH_HETERO]
    if what == "friction":
        kws_b[1].update(mu_p=1.5, mu_t=0.1)
    elif what == "plane":
        kws_b[1].update(d=-2e-6)                  # env 1's table moved up into the strip
    a, b = _build("pinch", kws), _build("pinch", kws_b)
    for k in range(4):
        _advance("pinch", a, None, k)
        _advance("pinch", b, None, k, perturb=_pad_motion_perturbation if what == "pad_motion" else None)
    _assert_envs_equal(a, b, (0, 2))
    sa, sb = a.solver.snapshot(1), b.solver.snapshot(1)
    assert not same_bytes(sa["x"], sb["x"]) and not same_bytes(sa["ct_work"], sb["ct_work"])
    if what == "plane":
        assert not same_bytes(a.plane_wrench.numpy()[1], b.plane_wrench.numpy()[1])
    else:
        assert not same_bytes(a.pad_wrench.numpy()[1], b.pad_wrench.numpy()[1])


def test_reset_classification_is_vars_complete_with_contact():
    batch = _build("pinch_p2", [dict()])
    s = batch.solver
    declared = set(rmb.SOLVER_RESET_ARRAYS+rmb.SOLVER_CONSTANT_ARRAYS+rmb.SOLVER_SHARED_ARRAYS
                   + rmb.SOLVER_RECOMPUTED_ARRAYS)
    for table in (rmb.OPTION_RESET_ARRAYS, rmb.OPTION_CONSTANT_ARRAYS):
        for value in table.values():
            declared |= set(value)
    declared |= set(rmb.MODEL_STATE_RESET_ARRAYS["neo_hookean_jbar_prony_v1"]+rmb.VISCO_ENERGY_RESET_ARRAYS
                    + rmb.VISCO_ENERGY_V2_RESET_ARRAYS)
    declared |= set(rcb.CONTACT_SOLVER_RESET_ARRAYS+rcb.CONTACT_SOLVER_CONSTANT_ARRAYS
                    + rcb.CONTACT_SOLVER_ALIAS_ARRAYS)
    names = set(s.device_arrays())
    assert names == declared, names ^ declared
    assert s.ct_env_of_node is s.env_of_node and s.ct_node_offset is s.node_offset
    assert set(batch.contact.arrays()) == set(rcb.CONTACT_MATERIAL_RESET_ARRAYS+rcb.CONTACT_MATERIAL_CONSTANT_ARRAYS)
    # every non-alias contact array of the single solver has a batched row layout
    single_names = {k for k, kind in CONTACT_ARRAY_KINDS.items()}
    assert set(rcb.CONTACT_SOLVER_RESET_ARRAYS+rcb.CONTACT_SOLVER_CONSTANT_ARRAYS
               + rcb.CONTACT_SOLVER_ALIAS_ARRAYS) == single_names


@pytest.mark.parametrize("scene", ["pinch_p2", "inc"])
def test_partial_reset_equals_fresh_env_and_keeps_others(scene):
    _, kws = SCENES[scene]
    batch, fresh = _build(scene, kws), _build(scene, kws)
    for k in range(3):
        _advance(scene, batch, None, k)
    snap = batch.solver.snapshot(1)
    assert np.abs(snap["ct_node_u"]).max() > 0 and np.abs(snap["ct_work"]).max() > 0
    assert np.any(snap["ct_obst"] >= 0) and np.abs(snap["ct_xi"]).max() > 0
    before = {e: _env_bytes(batch, e) for e in (0, 2)}
    batch.reset([1])
    _assert_envs_equal(batch, fresh, (1,), skip=("index", "summary", "markers", "visco_summary"))
    for e in (0, 2):
        after = _env_bytes(batch, e)
        for name in after:
            if name != "index":
                assert same_bytes(after[name], before[e][name]), (e, name)
    # the reset env evolves like a fresh env (index: the full fresh batch is at the start of a control step too)
    _advance(scene, batch, None, 0)
    _advance(scene, fresh, None, 0)
    _assert_envs_equal(batch, fresh, (1,))


def test_reset_with_resampled_friction_equals_the_single_reference():
    ranges = {"friction_table": (0.1, 1.5), "friction_pad": (0.2, 2.0)}
    lives = [_pinch(**kw) for kw in PINCH_HETERO]
    batch = rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True, ranges=ranges)
    _advance("pinch", batch, None, 0)
    kw = dict(PINCH_HETERO[1], mu_p=1.7, mu_t=0.15)
    new = _material(kw["mu0"], kw["rho"], mu_table=kw["mu_t"], mu_pad=kw["mu_p"])
    batch.reset([1], materials=[new])
    assert same_bytes(batch.solver.ct_fric.numpy()[1], np.array([0.15, 1.7], np.float32))
    single = _pinch(**kw)
    for k in range(3):
        tr, R, active = _frame_inputs("pinch", k, 3, 2)
        batch.advance(frame_translation=tr, frame_rotation=R, frame_active=active)
        single.advance(frames={f: {"translation": tr[1, f], "rotation": R[1, f], "active": bool(active[1, f])}
                               for f in range(2)})
        assert compare_contact(batch, 1, single) == [], k
    batch.reset([1])                                     # back to the packed frictions
    assert same_bytes(batch.solver.ct_fric.numpy()[1], np.array([0.3, 0.5], np.float32))


def test_episodes_repeat_bytewise_after_full_reset():
    batch = _build("pinch", PINCH_HETERO)

    def episode():
        batch.reset()
        for k in range(3):
            _advance("pinch", batch, None, k)
        return {e: _env_bytes(batch, e) for e in range(3)}

    a, b = episode(), episode()
    for e in range(3):
        for name in a[e]:
            assert same_bytes(a[e][name], b[e][name]), (e, name)


# ----------------------------------------------------------------------------- refusals (RUBBER-CONT-08 c)


def _refused(lives):
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)


def test_mixed_contact_configurations_are_refused():
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    m = _material()
    plane = ((0.0, 0.0, 1.0, 0.0),)
    box = ContactFrame(kind="pad", boxes=(((0.003, 0.002, 0.0025), (0.001, 0.001, 0.0005)),))
    box2 = ContactFrame(kind="pad", boxes=(((0.003, 0.002, 0.0025), (0.001, 0.001, 0.0005)),
                                           ((0.001, 0.002, 0.0025), (0.001, 0.001, 0.0005))))
    fixture = ContactFrame(kind="fixture", boxes=(((0.003, 0.002, 0.0025), (0.001, 0.001, 0.0005)),))
    base = ContactSpec(planes=plane, frames=(box,))
    variants = {"no_contact": None, "law": ContactSpec(law=PROJECTION_LAW, planes=plane, frames=(box,)),
                "zeta": ContactSpec(planes=plane, frames=(box,), damping_ratio=0.3),
                "zeta_n": ContactSpec(planes=plane, frames=(box,), normal_damping_ratio=0.3),
                "kappa": ContactSpec(planes=plane, frames=(box,), normal_stiffness=0.2),
                "planes": ContactSpec(planes=plane+((1.0, 0.0, 0.0, 0.0),), frames=(box,)),
                "frames": ContactSpec(planes=plane, frames=(box, box)),
                "boxes": ContactSpec(planes=plane, frames=(box2,)),
                "kind": ContactSpec(planes=plane, frames=(fixture,))}
    ref = _live(spec, m, base, (0.0, 0.0, -9.81))
    for name, other in variants.items():
        _refused([ref, _live(spec, m, other, (0.0, 0.0, -9.81))])
    # heterogeneous values with the same structure are accepted (plane offset, box geometry, frame origin)
    moved = ContactFrame(kind="pad", origin=(0.001, 0.0, 0.0), boxes=(((0.0031, 0.002, 0.0026), (0.0011, 0.001, 0.0005)),))
    rmb.BatchedRubberMaterial([ref, _live(spec, m, ContactSpec(planes=((0.0, 0.0, 1.0, 1e-6),), frames=(moved,)),
                                          (0.0, 0.0, -9.81))], device="cpu", parity_mode=True)


def test_unknown_single_contact_array_is_refused():
    """A single-solver contact array without a declared batched row kind (e.g. a new law version) is refused."""
    import warp as wp
    live = _inc()
    live.solver.ct_new_state = wp.zeros(3, dtype=float, device="cpu")
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_SOLVER: contact arrays without a batched"):
        rmb.BatchedRubberMaterial([live], device="cpu", parity_mode=True)


def test_frame_motions_need_contact_frames():
    batch = _build("imp", [dict()])
    with pytest.raises(IndexError):
        batch.advance(frame_translation=np.zeros((1, 1, 3)))
    small = vp.strip_spec(L=0.012, W=0.003, t=0.003, n_t=2, case="A")
    plain = rmb.BatchedRubberMaterial([vp.make_live(small, vp.material(4e5, P2, M=2),
                                                    step_material=STEP_MATERIAL)], device="cpu", parity_mode=True)
    with pytest.raises(ValueError, match="ContactSpec"):
        plain.advance(frame_translation=np.zeros((1, 1, 3)))
    with pytest.raises(ValueError, match="ContactSpec"):
        plain.pad_wrench


# ----------------------------------------------------------------------------- contact off


def test_contact_off_builds_the_unchanged_batched_solver():
    lives = [vp.make_live(vp.strip_spec(L=0.012, W=0.003, t=0.003, n_t=2, case="A"), vp.material(4e5, P2, M=2),
                          step_material=STEP_MATERIAL)]
    batch = rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)
    assert type(batch.solver) is BatchedTLMPMRubber and batch.contact is None
    assert batch.packed.uniform["contact"] is None
    assert not any(name.startswith("ct_") for name in batch.solver.device_arrays())
    assert not any(name.startswith("contact.") for name in batch.device_arrays())
    assert "contact" not in batch.receipt()


@pytest.mark.parametrize("terms", [(), P2], ids=["elastic", "prony_p2"])
def test_empty_contact_spec_batch_is_byte_identical_to_the_batch_without_contact(terms):
    def lives(contact, n):
        out = []
        for e in range(n):
            L = 0.012-0.002*e
            spec = rs.RubberStripSpec(length=L, width=0.003, thickness=0.002, n_t=2, case="A",
                                      traction_regions=(("tip", (L-0.0015, -1, -1), (1, 1, 1)),))
            out.append(_live(spec, _material(4e5+1e5*e, terms=terms, M=2 if terms else None), contact,
                             (0.0, 0.0, -9.81)) if contact is not None else
                       LiveRubberMaterial(spec, material=_material(4e5+1e5*e, terms=terms, M=2 if terms else None),
                                          control_dt=2e-4, gravity=(0.0, 0.0, -9.81), step_material=STEP_MATERIAL,
                                          strict=False, visco_energy="branch_work_kick_average_v2" if terms
                                          else "off_v1", **OPTS))
        return out
    a = rmb.BatchedRubberMaterial(lives(None, 2), device="cpu", parity_mode=True)
    b = rmb.BatchedRubberMaterial(lives(ContactSpec(), 2), device="cpu", parity_mode=True)
    assert type(a.solver) is BatchedTLMPMRubber and type(b.solver) is rcb.BatchedTLMPMRubberContact
    for k in range(3):
        tr = np.zeros((2, 1, 3))
        tr[:, 0, 2] = 2e-5*(k+1)
        force = np.array([[0.0, 0.0, -1e-3], [1e-3, 0.0, 0.0]])
        a.advance(translation=tr, traction=force)
        b.advance(translation=tr, traction=force)
        for e in range(2):
            sa, sb = a.solver.snapshot(e), b.solver.snapshot(e)
            for name in sa:
                assert same_bytes(sa[name], sb[name]), (k, e, name)
            for name in ("summary", "wrench", "marker_x", "marker_v"):
                assert same_bytes(getattr(a, name).numpy(), getattr(b, name).numpy()), name
            if terms:
                assert same_bytes(a.visco_summary.numpy(), b.visco_summary.numpy())


def test_receipt_reports_the_contact_path():
    batch = _build("pinch", PINCH_HETERO[:2])
    r = batch.receipt()
    assert r["solver"]["solver"] == "tlmpm_rubber_contact_v1_batch"
    assert r["solver"]["options"]["contact_law"] == "grid_node_penalty_coulomb_v1"
    assert r["solver"]["contact"]["kick_launch"].startswith("TLMPMRubberContact._launch_contact")
    assert r["contact"]["pad_frames"] == 2 and len(r["contact"]["specs"]) == 2
    assert r["calibration"] == "SIMULATOR_ONLY_UNCALIBRATED"
    assert batch.pad_wrench.shape == (2, 4, 6) and batch.pending_wrench.shape == (2, 6)
    assert batch.box_wrench.shape == (2, 2, 2, 6) and batch.plane_wrench.shape == (2, 1, 6)
