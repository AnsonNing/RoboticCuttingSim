"""ADR 0023 §22.11 (P2-4 task X3): contact law ``grid_node_penalty_coulomb_v3`` (opt-in).

v3 = ``grid_node_penalty_coulomb_v2`` plus the patch slip-rate equalisation: a SLIPPING node on a contact patch with
mu > 0 gets the impulse -dt c_d (v_t - vbar), c_d = 2 zeta_d sqrt(k_t m_I), vbar the mass-weighted mean realised
tangential velocity of the patch's slipping nodes in the same substep, outside the Coulomb cap and outside the slider.
A patch is one plane or one face of one box, per env (rows e*Q + q, Q = P + 6 F B). Checked here (Warp CPU, no local
artifacts):

- v1 / v2 kernels unchanged (v1 pins of ``test_rubber_contact_v2``, v2 pin here), v1 stays the default, v3 is selected
  only by ``law=``; the v3 parameter is validated and appears only in v3 receipts;
- kernel level: the equalisation impulses of a patch sum to zero (the patch's total tangential impulse and reaction
  force equal those of zeta_d = 0 to rounding; the reaction torque differs), the slider state does not depend on zeta_d, and the per-patch sums
  are env-indexed (perturbing one env's nodes changes no byte of another env's results);
- zeta_d = 0 reproduces the v2 state byte-wise (positions, velocities, deformation, slider) on an incline;
- a sliding block keeps its bottom face in contact and slides at the rigid Coulomb acceleration (v2 flutters there);
- batched (RUBBER-CONT-08 under v3): L0 byte-wise for N = 1 and N = 3 heterogeneous batches of the B1 scenes, L1
  isolation, masked reset equal to a fresh env, vars()-complete classification; mixing v2 and v3 is refused.
"""
import hashlib
import inspect
import math

import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim import rubber_contact as rct
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_contact import ContactSpec
from robotic_cutting_sim.rubber_material import MaterialValues

V2, V3 = rct.PENALTY_LAW_V2, rct.PENALTY_LAW_V3
# sha256[:16] of the v2 kernel source at 0b33246 (the law of contact_v2_001); never update: a v2 change is a new law
V2_SOURCE = {"node_penalty_contact_v2": "a186c586994e3cdb"}


def _source_block(text, name):
    i = text.index(f"def {name}(")
    i = text.rfind("@wp.", 0, i)
    ends = [j for j in (text.find("\n@wp.", i+5), text.find("\n# ---", i+5)) if j > 0]
    return text[i:min(ends)]


def _material(mu_table=0.0, mu_pad=0.0, prony=()):
    return MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=mu_table, friction_pad=mu_pad,
                          prony=prony)


# ----------------------------------------------------------------------------- opt-in, frozen v1 / v2


def test_v3_is_opt_in_and_v1_v2_kernels_are_unchanged():
    from test_rubber_contact_v2 import V1_SOURCES
    text = inspect.getsource(rct)
    for name, pin in {**V1_SOURCES, **V2_SOURCE}.items():
        assert hashlib.sha256(_source_block(text, name).encode()).hexdigest()[:16] == pin, name
    assert rct.CONTACT_LAW == rct.PENALTY_LAW and ContactSpec().law == rct.PENALTY_LAW
    assert V3 == "grid_node_penalty_coulomb_v3" and V3 in rct.CONTACT_LAWS and V3 in rct.PENALTY_LAWS
    r2, r3 = ContactSpec(law=V2).receipt(), ContactSpec(law=V3).receipt()
    assert "slip_rate_damping_ratio" not in r2 and "slip_rate_damping_ratio" not in ContactSpec().receipt()
    assert r3["slip_rate_damping_ratio"] == 1.0 and "patch" in r3
    assert {k: v for k, v in r3.items() if k not in ("law", "slip_rate_damping_ratio", "slip_rate_rule", "patch")} == \
        {k: v for k, v in r2.items() if k != "law"}
    with pytest.raises(ValueError, match="slip_rate_damping_ratio"):
        ContactSpec(law=V3, slip_rate_damping_ratio=1.5)
    ContactSpec(law=rct.PENALTY_LAW, tangential_stiffness=1.0, damping_ratio=0.5)       # v1 specs unaffected
    spec = ContactSpec(law=V3, planes=((0, 0, 1, 0),),
                       frames=(rct.ContactFrame(kind="pad", boxes=(((0, 0, 0), (1, 1, 1)),) * 2),))
    assert spec.Q == 1+6*2


# ----------------------------------------------------------------------------- kernel level


def _kernel_case(zeta_d, vel_scale_env1=1.0):
    """Two envs, 6 nodes each, one table plane z = 0 per env, all nodes penetrating with tangential realised
    velocities that differ (so every node slips with mu 0.3); returns the outputs of the two v3 passes."""
    rng = np.random.default_rng(7)
    n_env, per = 2, 6
    n = n_env*per
    dt = 1e-5
    gm = np.full(n, 2e-6, np.float32)
    old = np.zeros((n, 3), np.float32)
    old[:, :2] = rng.normal(0.0, 0.05, (n, 2)).astype(np.float32)
    old[:, 0] += 0.2
    old[per:] *= np.float32(vel_scale_env1)
    X = np.zeros((n, 3))
    X[:, 0] = np.tile(np.arange(per)*1e-3, n_env)
    u = np.zeros((n, 3))
    u[:, 2] = -1e-7*(1+rng.random(n))
    env = np.repeat(np.arange(n_env), per).astype(np.int32)
    with wp.ScopedDevice("cpu"):
        A = lambda a, dt_=None: wp.array(a, dtype=dt_) if dt_ else wp.array(a)              # noqa: E731
        fric = A(np.array([[0.3, 0.0]] * n_env, np.float32), float)
        planes = A(np.array([[0, 0, 1, 0]] * n_env, np.float64), wp.vec4d)
        dummy_q = A(np.array([[0, 0, 0, 1]], np.float64), wp.quatd)
        dummy_v = A(np.zeros((1, 3)), wp.vec3d)
        new_v = A(old.copy(), wp.vec3)
        o = A(old, wp.vec3)
        xi = wp.zeros(n, dtype=wp.vec3d)
        obst = wp.full(n, -1, dtype=int)
        vobs = wp.zeros(n, dtype=wp.vec3d)
        jn, jt = wp.zeros(n, dtype=wp.vec3d), wp.zeros(n, dtype=wp.vec3d)
        prc = wp.zeros(n_env, dtype=wp.spatial_vectord)
        brc = wp.zeros(1, dtype=wp.spatial_vectord)
        work = wp.zeros((n_env, rct.NWORK), dtype=wp.float64)
        pen = wp.zeros(n_env, dtype=wp.float64)
        counts = wp.zeros((n_env, 3), dtype=int)
        Q = 1
        bbest, bprow, bflag = wp.full(n, -1, dtype=int), wp.full(n, -1, dtype=int), wp.zeros(n, dtype=int)
        bfn, bmu = wp.zeros(n, dtype=wp.float64), wp.zeros(n, dtype=wp.float64)
        bn, blev = wp.zeros(n, dtype=wp.vec3d), wp.zeros(n, dtype=wp.vec3d)
        pacc = wp.zeros((n_env*Q, 4), dtype=wp.float64)
        ints = lambda a: A(np.asarray(a, np.int32), int)                                     # noqa: E731
        wp.launch(rct.node_penalty_contact_v3_normal, n,
                  [A(gm, float), o, A(X, wp.vec3d), A(u, wp.vec3d), ints(np.full(n, -1)), ints([0]), ints(env), fric,
                   0, 1, 1, planes, 0, 0, ints([0]), ints([1]), dummy_q, dummy_q, dummy_v, dummy_v, dummy_v, dummy_v,
                   dummy_v, ints([0]), 1, wp.float64(dt), wp.float64(0.25), wp.float64(0.25), wp.float64(0.5),
                   wp.float64(0.5), xi, obst, vobs, jn, jt, prc, brc, work, pen, Q, bbest, bprow, bflag, bfn, bmu, bn,
                   blev, pacc])
        wp.launch(rct.node_penalty_contact_v3_tangential, n,
                  [A(gm, float), new_v, o, ints(env), 1, 0, 0, wp.float64(dt), wp.float64(0.25), wp.float64(0.5),
                   wp.float64(zeta_d), xi, obst, vobs, jn, jt, prc, brc, work, counts, Q, bbest, bprow, bflag, bfn,
                   bmu, bn, blev, pacc])
    return {"jt": jt.numpy(), "jn": jn.numpy(), "xi": xi.numpy(), "flag": bflag.numpy(), "prc": prc.numpy(),
            "work": work.numpy(), "pacc": pacc.numpy(), "new_v": new_v.numpy(), "per": per}


def test_v3_equalisation_is_net_zero_per_patch_and_env_indexed():
    a, b = _kernel_case(0.0), _kernel_case(1.0)
    per = a["per"]
    assert np.all(a["flag"] == rct.STATE_SLIP)                                   # every node slips (mu 0.3)
    assert not np.allclose(a["jt"], b["jt"])                                    # the equalisation acts per node
    for e in range(2):
        rows = slice(e*per, (e+1)*per)
        tot_a, tot_b = a["jt"][rows].sum(0), b["jt"][rows].sum(0)
        assert np.allclose(tot_a, tot_b, rtol=0, atol=1e-12*np.abs(tot_a).max())  # net zero over the patch
        fa, fb = a["prc"][e][:3], b["prc"][e][:3]                               # reaction force (the equalisation
        assert np.allclose(fa, fb, rtol=0, atol=1e-12*np.abs(fa).max())         # has a torque, not a net force)
        assert np.all(np.linalg.norm(b["jt"][rows].sum(0)) <= 0.3*np.abs(b["jn"][rows, 2]).sum()*(1+1e-12))
        assert np.array_equal(a["xi"][rows], b["xi"][rows])                     # the slider ignores the damper
    assert b["work"][0, 5] > a["work"][0, 5]                                    # booked as contact damping
    c = _kernel_case(1.0, vel_scale_env1=1.7)                                  # env 1 perturbed
    for key in ("jt", "jn", "xi", "new_v"):
        assert np.array_equal(b[key][:per], c[key][:per]), key
    assert np.array_equal(b["prc"][0], c["prc"][0]) and np.array_equal(b["work"][0], c["work"][0])
    assert np.array_equal(b["pacc"][0], c["pacc"][0]) and not np.array_equal(b["pacc"][1], c["pacc"][1])


# ----------------------------------------------------------------------------- solver level


def _incline(law, mu=0.8, ratio=1.2, zeta_d=None, steps=40, L=0.006, W=0.004, t=0.002):
    theta = math.atan(ratio*mu)
    spec = rs.RubberStripSpec(length=L, width=W, thickness=t, n_t=2, case="free")
    kw = {} if zeta_d is None else {"slip_rate_damping_ratio": zeta_d}
    live = LiveRubberMaterial(spec, material=_material(mu), control_dt=1e-3, gravity=(0, 0, 0),
                              stable_step_guard=1.0, state_precision="f64_shadow_v1",
                              contact=ContactSpec(planes=((0, 0, 1, 0),), law=law, **kw))
    s = live.solver
    X = s.node_X.numpy()
    xs, frac = [], []
    for k in range(steps):
        r = 0.5*(1-math.cos(math.pi*min((k+0.5)/20, 1.0)))
        s.gravity = wp.vec3(9.81*math.sin(theta)*r, 0.0, -9.81*math.cos(theta)*r)
        live.advance()
        bottom = (np.abs(X[:, 2]) < 1e-12) & (s.gm.numpy() > 0)
        xs.append(float(s.x64.numpy()[:, 0].mean()))
        frac.append(float((np.abs(s.ct_jn.numpy()[bottom, 2]) > 0).mean()))
    return live, np.array(xs), np.array(frac), theta


def test_v3_with_zero_equalisation_reproduces_v2_state():
    a, _, _, _ = _incline(V2, steps=12)
    b, _, _, _ = _incline(V3, zeta_d=0.0, steps=12)
    for name in ("x", "x64", "v", "F", "F64", "ct_node_u", "ct_xi", "ct_obst", "ct_vobs", "ct_jn", "ct_jt"):
        assert np.array_equal(getattr(a.solver, name).numpy(), getattr(b.solver, name).numpy()), name
    wa, wb = a.solver.ct_work.numpy()[0], b.solver.ct_work.numpy()[0]
    assert np.allclose(wa, wb, rtol=1e-10, atol=0)                              # atomic order differs (2 passes)


def test_v3_slides_at_the_coulomb_acceleration_without_flutter():
    """6 x 4 x 2 mm block, mu 0.8, tan theta = 1.2 mu: v3 slides at the rigid Coulomb acceleration (v2 on this
    block: +1.7 % of g cos theta, face mostly out of contact)."""
    live, xs, frac, theta = _incline(V3, steps=40)
    t = (np.arange(len(xs))+1)*1e-3
    sel = t >= 0.025
    a_fit = 2*np.polyfit(t[sel], xs[sel], 2)[0]
    g = 9.81
    err = (a_fit-g*(math.sin(theta)-0.8*math.cos(theta)))/(g*math.cos(theta))
    assert abs(err) < 0.01, err
    assert frac[sel].mean() > 0.5                                               # the face stays in contact
    c = live.last_contact
    pw = c["plane_wrench"][0]
    assert math.hypot(pw[0], pw[1]) <= 0.8*abs(pw[2])*(1+1e-6)                  # patch total inside the cone
    assert live.solver.solver_receipt()["options"]["contact_law"] == V3


# ----------------------------------------------------------------------------- batched (RUBBER-CONT-08 under v3)


@pytest.fixture
def batched_v3(monkeypatch):
    """The B1 test scenes (``test_rubber_contact_batched``) with every ContactSpec built as the v3 law."""
    import test_rubber_contact_batched as tb
    base = tb.ContactSpec

    class SpecV3(base):
        def __init__(self, *args, **kw):
            base.__init__(self, *args, **{**kw, "law": V3})
    monkeypatch.setattr(tb, "ContactSpec", SpecV3)
    return tb


@pytest.mark.parametrize("scene", ["inc", "pinch", "pinch_p2", "imp"])
def test_v3_batched_l0_single_and_heterogeneous(batched_v3, scene):
    tb = batched_v3
    _, kws = tb.SCENES[scene]
    for envs in ([dict()], kws):
        batch, singles = tb._run_l0(scene, envs)
        assert all(live.solver.contact.law == V3 and live.solver.ct_law == V3 for live in singles)
        assert batch.solver.ct_law == V3 and batch.solver.contact_signature["law"] == V3
        assert batch.solver.ct_Q == singles[0].solver.ct_Q
        assert all(live.last_contact["impulse_events"] > 0 for live in singles)
        audit = tb.rcb.contact_audit(batch.solver)
        assert not any(audit.values()), audit


def test_v3_batched_l0_exercises_the_equalisation(batched_v3):
    """The heterogeneous incline batch slips on its table patch (pass-1 sums nonzero) in the L0 run."""
    tb = batched_v3
    singles = tb._singles("inc", tb.INC_HETERO)
    batch = tb._build("inc", tb.INC_HETERO)
    used = 0
    for k in range(4):
        tb._advance("inc", batch, singles, k)
        for e, live in enumerate(singles):
            assert tb.compare_contact(batch, e, live) == [], (k, e)
        used += int(np.any(batch.solver.ct_pacc.numpy()[:, 3] > 0))
    assert used > 0


@pytest.mark.parametrize("what", ["pad_motion", "friction", "plane"])
def test_v3_batched_l1_isolation(batched_v3, what):
    batched_v3.test_l1_perturbing_one_env_leaves_the_others_byte_unchanged(what)


@pytest.mark.parametrize("scene", ["pinch_p2", "inc"])
def test_v3_batched_partial_reset_equals_fresh(batched_v3, scene):
    batched_v3.test_partial_reset_equals_fresh_env_and_keeps_others(scene)


def test_v3_batched_reset_classification_is_vars_complete(batched_v3):
    batched_v3.test_reset_classification_is_vars_complete_with_contact()


def test_v2_and_v3_in_one_batch_are_refused():
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    lives = [LiveRubberMaterial(spec, material=_material(0.5), control_dt=2e-4, gravity=(0, 0, -9.81),
                                contact=ContactSpec(planes=((0, 0, 1, 0),), law=law), strict=False)
             for law in (V2, V3)]
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)
