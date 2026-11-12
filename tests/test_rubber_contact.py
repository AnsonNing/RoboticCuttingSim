"""ADR 0023 §22 (P2-4): rubber contact law ``grid_node_penalty_coulomb_v1`` (``rubber_contact``,
``tlmpm_rubber_contact_v1``), single env, Warp CPU.

- spec validation and refusals; the frozen per-particle path is refused together with a ContactSpec;
- without a ContactSpec ``LiveRubberMaterial`` builds the unchanged elastic / visco classes;
- with an EMPTY ContactSpec every state array is byte-identical to ``TLMPMRubberVisco`` (elastic M = 0 and Prony M = 2)
  after every control step (the contact launches change nothing);
- float64 box SDF twin = host reference; Coulomb projection kernel = host reference;
- static penalty penetration of a loaded block = the law's analytic value k_n^-1 F_I;
- energy books of a frictionless rod impact close with the kick-paired contact work (layer N);
- reset clears the contact state, two episodes are byte-identical;
- pad frame: a moving pad pressing a block on the table; force balance of the pad / table wrenches.
"""
import math

import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim import rubber_contact as rct
from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_contact import ContactFrame, ContactSpec
from robotic_cutting_sim.rubber_material import MaterialValues
from robotic_cutting_sim.tlmpm_rubber_contact_v1 import CONTACT_RESET_ARRAYS, TLMPMRubberContact
from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber
from robotic_cutting_sim.tlmpm_rubber_visco_v1 import TLMPMRubberVisco

P2 = ((0.2, 0.005), (0.15, 0.05))


def _same_bytes(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _material(prony=(), mu_table=0.5, mu_pad=1.0):
    return MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=mu_table, friction_pad=mu_pad,
                          prony=prony)


def _block(n_t=2, **kw):
    return rs.compile_strip(rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=n_t, case="free",
                                               **kw))


# ----------------------------------------------------------------------------- spec


def test_spec_validation_and_refusals():
    ContactSpec(planes=((0, 0, 1, 0),))
    with pytest.raises(ValueError):
        ContactSpec(law="nope")
    with pytest.raises(ValueError):
        ContactSpec(planes=((0, 0, 2, 0),))                  # not a unit normal
    with pytest.raises(ValueError):
        ContactSpec(frames=(ContactFrame(boxes=(((0, 0, 0), (1, 1, 1)),)),
                            ContactFrame(boxes=(((0, 0, 0), (1, 1, 1)), ((0, 0, 0), (1, 1, 1))))))  # non-uniform B
    with pytest.raises(ValueError):
        ContactSpec(frames=(ContactFrame(kind="robot", boxes=(((0, 0, 0), (1, 1, 1)),)),))
    with pytest.raises(ValueError):
        ContactSpec(frames=(ContactFrame(boxes=(((0, 0, 0), (1, 0, 1)),)),))                     # zero half extent
    with pytest.raises(ValueError):
        ContactSpec(normal_stiffness=0.0)
    with pytest.raises(ValueError):
        ContactSpec(damping_ratio=1.5)                       # 2 zeta sqrt(kappa) > 1
    with pytest.raises(ValueError):
        ContactSpec(work="end_velocity_v1")
    spec = ContactSpec()
    assert spec.law == rct.PENALTY_LAW == "grid_node_penalty_coulomb_v1"
    assert (spec.normal_stiffness, spec.tangential_stiffness, spec.damping_ratio) == (0.25, 0.25, 0.5)
    r = spec.receipt()
    assert r["law"] == rct.PENALTY_LAW and r["work"] == "kick_average_v1" and r["damping_ratio"] == 0.5


def test_frozen_particle_contact_path_refused_with_a_spec():
    sc = _block()
    with pytest.raises(ValueError, match="frozen per-particle"):
        TLMPMRubberContact(sc, _material(), 1e-5, contact=ContactSpec(), table_plane=(0, 0, 1, 0))
    with pytest.raises(TypeError):
        TLMPMRubberContact(sc, _material(), 1e-5, contact={"planes": ()})


def test_live_without_contact_builds_the_unchanged_classes():
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    assert type(LiveRubberMaterial(spec, material=_material(), control_dt=1e-4).solver) is TLMPMRubber
    assert type(LiveRubberMaterial(spec, material=_material(P2), control_dt=1e-4).solver) is TLMPMRubberVisco
    live = LiveRubberMaterial(spec, material=_material(), control_dt=1e-4, contact=ContactSpec())
    assert type(live.solver) is TLMPMRubberContact


# ----------------------------------------------------------------------------- off = byte identity


@pytest.mark.parametrize("prony", [(), P2], ids=["elastic", "prony_p2"])
def test_empty_spec_is_byte_identical_to_the_visco_solver(prony):
    spec = rs.RubberStripSpec(length=0.008, width=0.002, thickness=0.002, n_t=2, case="A",
                              traction_regions=(("tip", (0.007, -1, -1), (1, 1, 1)),))
    kw = dict(material=_material(prony), control_dt=1e-4, gravity=(0, 0, -9.81), stable_step_guard=1.0,
              dirichlet="consistent_mass_projection_v1", constraint_work="kick_average_v1",
              traction_work="grid_displacement_v1", state_precision="f64_shadow_v1",
              visco_energy="branch_work_kick_average_v2" if prony else "off_v1")
    ref = LiveRubberMaterial(spec, **kw)            # elastic: TLMPMRubberCW (frozen launches); Prony: TLMPMRubberVisco
    con = LiveRubberMaterial(spec, contact=ContactSpec(), **kw)
    assert isinstance(con.solver, type(ref.solver))
    for k in range(4):
        motions = {0: {"translation": np.array([0.0, 0.0, 2e-5*(k+1)])}}
        w1 = ref.advance(motions, {0: np.array([0.0, 0.0, -1e-3])})
        w2 = con.advance(motions, {0: np.array([0.0, 0.0, -1e-3])})
        assert _same_bytes(w1, w2)
        a, b = ref.solver.device_arrays(), con.solver.device_arrays()
        for name, arr in a.items():
            assert _same_bytes(arr.numpy(), b[name].numpy()), (k, name)
    assert set(con.solver.device_arrays()) - set(ref.solver.device_arrays()) >= set(CONTACT_RESET_ARRAYS)


# ----------------------------------------------------------------------------- Warp functions vs host


@wp.kernel
def _sdf_probe(p: wp.array(dtype=wp.vec3d), h: wp.vec3d, n: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.float64)):
    i = wp.tid()
    nn, dd = rct.box_sdf_d(p[i], h)
    n[i] = nn
    d[i] = dd


@wp.kernel
def _proj_probe(v: wp.array(dtype=wp.vec3d), vo: wp.array(dtype=wp.vec3d), nrm: wp.array(dtype=wp.vec3d),
                gap: wp.array(dtype=wp.float64), dt: wp.float64, beta: wp.float64, mu: wp.float64,
                dvn: wp.array(dtype=wp.float64), dvt: wp.array(dtype=wp.vec3d), state: wp.array(dtype=int)):
    i = wp.tid()
    a, b, c = rct.coulomb_projection(v[i], vo[i], nrm[i], gap[i], dt, beta, mu)
    dvn[i] = a
    dvt[i] = b
    state[i] = c


def test_box_sdf_twin_matches_the_host_reference():
    rng = np.random.default_rng(3)
    pts = rng.uniform(-2, 2, (400, 3))
    pts[:50] = rng.uniform(-0.9, 0.9, (50, 3))           # inside
    h = np.array([1.0, 0.7, 0.4])
    n, d = wp.zeros(400, dtype=wp.vec3d, device="cpu"), wp.zeros(400, dtype=wp.float64, device="cpu")
    wp.launch(_sdf_probe, 400, [wp.array(pts, dtype=wp.vec3d, device="cpu"), wp.vec3d(*h), n, d], device="cpu")
    for p, nk, dk in zip(pts, n.numpy(), d.numpy()):
        hn, hd = rct.host_box_sdf(p, h)
        assert abs(hd-dk) <= 1e-14 and np.allclose(hn, nk, atol=1e-14)


def test_coulomb_projection_matches_the_host_reference():
    rng = np.random.default_rng(5)
    K = 300
    v, vo = rng.normal(0, 0.1, (K, 3)), rng.normal(0, 0.05, (K, 3))
    nrm = rng.normal(size=(K, 3))
    nrm /= np.linalg.norm(nrm, axis=1)[:, None]
    gap = rng.normal(0, 1e-6, K)
    dvn, dvt, st = (wp.zeros(K, dtype=t, device="cpu") for t in (wp.float64, wp.vec3d, int))
    wp.launch(_proj_probe, K, [wp.array(a, dtype=wp.vec3d, device="cpu") for a in (v, vo, nrm)]
              + [wp.array(gap, dtype=wp.float64, device="cpu"), wp.float64(1e-5), wp.float64(0.1), wp.float64(0.6),
                 dvn, dvt, st], device="cpu")
    states = set()
    for k in range(K):
        a, b, c = rct.host_coulomb_projection(v[k], vo[k], nrm[k], gap[k], 1e-5, 0.1, 0.6)
        assert c == st.numpy()[k]
        assert abs(a-dvn.numpy()[k]) <= 1e-12*max(1, abs(a)) and np.allclose(b, dvt.numpy()[k], atol=1e-12)
        states.add(c)
    assert states == {0, 1, 2}


# ----------------------------------------------------------------------------- physics (small, CPU)


def _settle_block(prony, p_top, steps=240, control_dt=5e-4, ramp=0.01, mu_table=0.0, zeta=0.5):
    """Block 6 x 4 x 2 mm (n_t 2) on the table z = 0, gravity -z, top traction pressure p_top ramped (cosine); a
    frictionless table by default (free lateral expansion: uniaxial stress, uniform base pressure). Returns the mean
    table force and the mean node z displacement over the last 40 control steps (20 ms)."""
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free",
                              traction_regions=(("top", (-1, -1, 0.0015), (1, 1, 1)),))
    live = LiveRubberMaterial(spec, material=_material(prony, mu_table=mu_table), control_dt=control_dt,
                              gravity=(0, 0, -9.81), stable_step_guard=1.0,
                              contact=ContactSpec(planes=((0, 0, 1, 0),), damping_ratio=zeta, normal_damping_ratio=zeta))
    F = p_top*0.006*0.004
    fz, uz = [], []
    for k in range(steps):
        t = (k+0.5)*control_dt
        r = 0.5*(1-math.cos(math.pi*t/ramp)) if t < ramp else 1.0
        live.advance(traction={0: np.array([0.0, 0.0, -F*r])})
        if k >= steps-40:
            fz.append(-live.last_contact["plane_wrench"][0, 2])
            uz.append(live.solver.ct_node_u.numpy()[:, 2].copy())
    return live, F, float(np.mean(fz)), np.mean(uz, axis=0)


def _face_spring_share(live, fz, uz):
    s = live.solver
    gm = s.gm.numpy().astype(np.float64)
    X = s.node_X.numpy()
    face = np.flatnonzero((np.abs(X[:, 2]) < 1e-12) & (gm > 0))
    k = 0.25*gm[face]/s.dt**2
    return float(np.sum(-uz[face]*k)/fz), face, k


def test_static_penalty_penetration_equals_the_law():
    """Loaded block on a frictionless table (uniform base pressure p). Undamped variant: every face node's
    penetration is p A_I / k_n (k_n = kappa_n m_I / dt^2) and the springs carry the whole reaction. Declared law
    (zeta_n = 0.5): the dashpot acts on the predicted (post-force) node velocity, which at a static fixed point is
    -dt F_I / m_I, so the springs carry the share 1 - 2 zeta_n sqrt(kappa_n) = 0.5 (no dissipation: the actual velocity
    is zero). Both table reactions balance weight + load."""
    for zeta, share in ((0.0, 1.0), (0.5, 0.5)):
        live, F, fz, uz = _settle_block(P2, p_top=2.0e4, zeta=zeta)
        s = live.solver
        W = s.mass_value*s.n*9.81
        assert abs(fz/(W+F)-1) < 2e-3
        spring, face, k = _face_spring_share(live, fz, uz)
        if zeta == 0.0:
            assert abs(spring-1) < 0.02
            X = s.node_X.numpy()
            A = (np.where((X[face, 0] > 1e-9) & (X[face, 0] < 0.006-1e-9), 1.0, 0.5)
                 * np.where((X[face, 1] > 1e-9) & (X[face, 1] < 0.004-1e-9), 1.0, 0.5)*s.dx**2)
            ratio = -uz[face]/(fz/0.006/0.004*A/k)
            assert abs(ratio.mean()-1) < 0.05 and ratio.min() > 0.8 and ratio.max() < 1.2
        else:
            assert 0.35 < spring < 0.6, spring
        assert live.last_contact["max_penetration_m"] < 0.25*s.spacing


def test_frictionless_rod_impact_books_close_with_the_contact_work():
    spec = rs.RubberStripSpec(length=0.012, width=0.002, thickness=0.002, n_t=2, case="free")
    live = LiveRubberMaterial(spec, material=_material(mu_table=0.0), control_dt=1e-4, gravity=(0, 0, 0),
                              stable_step_guard=1.0, contact=ContactSpec(planes=((1, 0, 0, 0),)))
    s = live.solver
    V = 0.2
    s.set_initial_velocity((-V, 0.0, 0.0))
    E0 = 0.5*s.mass_value*s.n*V*V
    worst = 0.0
    for _ in range(15):
        live.advance()
        d, c = live.last_diagnostics, live.last_contact
        books = d["kinetic_particle"]+d["elastic"]-c["contact_work_material"]-E0
        worst = max(worst, abs(books)/E0)
    assert worst < 0.03                     # layer N, O(dt) (n_t 2 here; the protocol runs finer)
    # the normal law dissipates only through its declared dashpot (zeta_n = 0.5): bounded, never energy-producing
    assert 0.0 < -c["normal_work_material"] < 0.15*E0
    assert c["max_penetration_m"] < 0.25*s.spacing
    assert s.v.numpy()[:, 0].mean() > 0.5*V                   # rebounded


def test_reset_clears_contact_state_and_episodes_repeat_bytewise():
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    live = LiveRubberMaterial(spec, material=_material(), control_dt=2e-4, gravity=(3.0, 0, -9.81),
                              stable_step_guard=1.0, contact=ContactSpec(planes=((0, 0, 1, 0),)))

    def episode():
        live.reset()
        live.solver.set_initial_velocity((0.0, 0.0, -0.05))
        for _ in range(4):
            live.advance()
        return {k: v.numpy().copy() for k, v in live.solver.device_arrays().items()}

    a = episode()
    assert np.abs(a["ct_node_u"]).max() > 0 and np.abs(a["ct_work"]).max() > 0
    live.reset()
    for name in CONTACT_RESET_ARRAYS:
        assert not np.any(getattr(live.solver, name).numpy()), name
    assert np.all(live.solver.ct_obst.numpy() == -1)
    b = episode()
    for name in a:
        assert _same_bytes(a[name], b[name]), name


def test_pad_frame_presses_block_force_balance():
    """A pad box (frame kind 'pad') lowered onto a block resting on the table: the table carries weight + pad force,
    the pad wrench is the material's push on the pad (+z), the pad frame torque is about its current origin."""
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    pad = ContactFrame(kind="pad", origin=(0.003, 0.002, 0.0025),
                       boxes=(((0.003, 0.002, 0.0025), (0.0015, 0.0015, 0.0005)),), name="pad")
    live = LiveRubberMaterial(spec, material=_material(P2), control_dt=5e-4, gravity=(0, 0, -9.81),
                              stable_step_guard=1.0, contact=ContactSpec(planes=((0, 0, 1, 0),), frames=(pad,)))
    s = live.solver
    depth = 2e-5
    for k in range(80):
        r = min(1.0, (k+1)/30)
        live.advance(frames={0: {"translation": np.array([0.0, 0.0, -depth*r])}})
    pw = live.last_contact["box_wrench"]
    tw = live.last_contact["plane_wrench"]
    W = s.mass_value*s.n*9.81
    assert pw.shape == (1, 1, 6) and pw[0, 0, 2] > 0
    assert abs((-tw[0, 2]) - (W+pw[0, 0, 2]))/(W+pw[0, 0, 2]) < 0.02
    assert abs(pw[0, 0, 3]) < 1e-2*pw[0, 0, 2]*0.0015 and abs(pw[0, 0, 4]) < 1e-2*pw[0, 0, 2]*0.0015


def test_undamped_variant_normal_law_is_conservative():
    """zeta_n = zeta_t = 0: frictionless rod impact returns its normal contact work (spring potential) on separation."""
    spec = rs.RubberStripSpec(length=0.012, width=0.002, thickness=0.002, n_t=2, case="free")
    live = LiveRubberMaterial(spec, material=_material(mu_table=0.0), control_dt=1e-4, gravity=(0, 0, 0),
                              contact=ContactSpec(planes=((1, 0, 0, 0),), damping_ratio=0.0, normal_damping_ratio=0.0))
    s = live.solver
    s.set_initial_velocity((-0.2, 0.0, 0.0))
    E0 = 0.5*s.mass_value*s.n*0.04
    for _ in range(15):
        live.advance()
    assert abs(live.last_contact["normal_work_material"]) < 2e-3*E0
    assert live.last_contact["contact_energy_normal"] == 0.0          # separated
