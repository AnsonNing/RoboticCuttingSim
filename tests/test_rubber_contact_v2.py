"""ADR 0023 §22.9 (P2-4 task X1): contact law ``grid_node_penalty_coulomb_v2`` (opt-in) and the frozen v1 law.

v2 = ``grid_node_penalty_coulomb_v1`` with both dashpots and the tangential trial on the node's REALISED velocity
(``old`` after the P2G grid update = B pi_n, the drift velocity of the previous substep) instead of the predicted lumped
velocity ``new_v``. Checked here (Warp CPU, no local artifacts):

- the v1 law's kernels are unchanged (source hashes of ``33f5fe3``), v1 stays the default, and ``ContactSpec`` /
  receipts of v1 are unchanged; v2 is selected only by ``law=``;
- the v1 defect and its correction: a frictionless elastic block at rest keeps a self-excited contact rattle under v1
  (bottom nodes intermittently out of contact, net POSITIVE normal contact work) and settles under v2;
- v2 static law: the dashpot carries no static load (realised velocity 0), so the springs carry the whole reaction
  (v1: half, ``test_rubber_contact.test_static_penalty_penetration_equals_the_law``);
- a frictionless rod impact closes its books with the kick-paired contact work under v2;
- batched (RUBBER-CONT-08 under v2): N = 3 heterogeneous batches of the incline and pinch scenes are byte-identical to
  their single-env v2 references after every control step; mixing v1 and v2 in one batch is refused.
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

V2 = rct.PENALTY_LAW_V2
# sha256[:16] of the source of every v1 kernel / function at 33f5fe3 (the law of contact_v1_001); never update: a v1
# change is a new law version
V1_SOURCES = {"box_sdf_d": "170bae15f53199be", "coulomb_projection": "105d853d713581f8",
              "frame_rotation": "fce5ebf5bbcea842", "node_contact": "8ac51079078e3346", "plane_hit": "f3c78685c6cdb33c",
              "node_penalty_contact": "fa38d7550354f798", "slider_update": "08e57c4d85e724b2",
              "contact_energy": "6b2fb98be44981a9", "contact_work_end": "ba8f34045dbe88e0",
              "node_displacement": "c65ad878bbe52af2"}


def _source_block(text, name):
    i = text.index(f"def {name}(")
    i = text.rfind("@wp.", 0, i)
    ends = [j for j in (text.find("\n@wp.", i+5), text.find("\n# ---", i+5)) if j > 0]
    return text[i:min(ends)]


def _material(mu_table=0.0, mu_pad=0.0, prony=()):
    return MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=mu_table, friction_pad=mu_pad,
                          prony=prony)


# ----------------------------------------------------------------------------- v1 frozen, v2 opt-in


def test_v1_law_sources_unchanged_and_v1_is_the_default():
    text = inspect.getsource(rct)
    for name, pin in V1_SOURCES.items():
        assert hashlib.sha256(_source_block(text, name).encode()).hexdigest()[:16] == pin, name
    assert rct.CONTACT_LAW == rct.PENALTY_LAW == "grid_node_penalty_coulomb_v1"
    assert ContactSpec().law == rct.PENALTY_LAW
    assert V2 == "grid_node_penalty_coulomb_v2" and V2 in rct.CONTACT_LAWS and V2 in rct.PENALTY_LAWS
    r1, r2 = ContactSpec().receipt(), ContactSpec(law=V2).receipt()
    assert r1["damping_rule"].startswith("c = 2 zeta sqrt(k m_I) on the node's predicted (post-force)")
    assert "realised relative velocity" in r2["damping_rule"]
    assert {k: v for k, v in r1.items() if k not in ("law", "damping_rule")} == \
        {k: v for k, v in r2.items() if k not in ("law", "damping_rule")}
    # the v2 kernel differs from v1 only in the dashpot / trial velocity
    v1 = _source_block(text, "node_penalty_contact")
    v2 = _source_block(text, "node_penalty_contact_v2")
    body = lambda s: s[s.index("i = wp.tid()"):]                                     # noqa: E731
    assert body(v1).replace("vstar = wp.vec3d(new_v[i])", "vstar = wp.vec3d(old[i])") == \
        body(v2).replace("vstar = wp.vec3d(old[i])                     # v2: the realised (drift) velocity, not the "
                         "predicted new_v", "vstar = wp.vec3d(old[i])")


def _rest_block(law, steps=60):
    """Block 6 x 4 x 2 mm (n_t 2) at rest on a FRICTIONLESS table z = 0, gravity ramped over 20 ms (1 ms steps)."""
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    live = LiveRubberMaterial(spec, material=_material(), control_dt=1e-3, gravity=(0, 0, 0), stable_step_guard=1.0,
                              contact=ContactSpec(planes=((0, 0, 1, 0),), law=law))
    s = live.solver
    X = s.node_X.numpy()
    frac = []
    for k in range(steps):
        r = 0.5*(1-math.cos(math.pi*min((k+0.5)/20, 1.0)))
        s.gravity = wp.vec3(0.0, 0.0, -9.81*r)
        live.advance()
        bottom = (np.abs(X[:, 2]) < 1e-12) & (s.gm.numpy() > 0)
        frac.append(float((np.abs(s.ct_jn.numpy()[bottom, 2]) > 0).mean()))
    return live, float(np.mean(frac[-10:]))


def test_v1_rattles_at_rest_and_v2_settles():
    """X1 root cause (ADR 0023 §22.9): the v1 dashpot on the predicted lumped velocity is not dissipative for the face's
    grid-scale modes under the TLMPM drift metric; a frictionless block at rest keeps bouncing (intermittent contact,
    net positive normal contact work). v2 (dashpot on the realised velocity) settles: every bottom node in contact,
    contact work dissipative, no residual vibration."""
    v1, f1 = _rest_block(rct.PENALTY_LAW)
    v2, f2 = _rest_block(V2)
    c1, c2 = v1.last_contact, v2.last_contact
    assert f1 < 0.9 and c1["normal_work_material"] > 0.0
    assert f2 == 1.0 and c2["normal_work_material"] < 0.0
    assert v2.last_diagnostics["kinetic_particle"] < 1e-2*v1.last_diagnostics["kinetic_particle"]
    assert v2.last_diagnostics["elastic"] < 0.1*v1.last_diagnostics["elastic"]


def test_v2_static_springs_carry_the_whole_reaction():
    """Loaded block on a frictionless table (as the v1 law test): under v2 the dashpot sees the realised velocity
    (0 at a static fixed point), so sum k_n delta_I = reaction (v1: 0.5) and the interior face nodes follow
    delta_I = p A_I / k_n."""
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free",
                              traction_regions=(("top", (-1, -1, 0.0015), (1, 1, 1)),))
    live = LiveRubberMaterial(spec, material=_material(prony=((0.2, 0.005), (0.15, 0.05))), control_dt=5e-4,
                              gravity=(0, 0, -9.81), stable_step_guard=1.0,
                              contact=ContactSpec(planes=((0, 0, 1, 0),), law=V2))
    F = 2.0e4*0.006*0.004
    fz, uz = [], []
    for k in range(240):
        t = (k+0.5)*5e-4
        r = 0.5*(1-math.cos(math.pi*t/0.01)) if t < 0.01 else 1.0
        live.advance(traction={0: np.array([0.0, 0.0, -F*r])})
        if k >= 200:
            fz.append(-live.last_contact["plane_wrench"][0, 2])
            uz.append(live.solver.ct_node_u.numpy()[:, 2].copy())
    s = live.solver
    fz, uz = float(np.mean(fz)), np.mean(uz, axis=0)
    assert abs(fz/(s.mass_value*s.n*9.81+F)-1) < 2e-3
    gm = s.gm.numpy().astype(np.float64)
    X = s.node_X.numpy()
    face = np.flatnonzero((np.abs(X[:, 2]) < 1e-12) & (gm > 0))
    k = 0.25*gm[face]/s.dt**2
    assert abs(float(np.sum(-uz[face]*k))/fz-1) < 0.02
    assert live.last_contact["max_penetration_m"] < 0.25*s.spacing


def test_v2_frictionless_rod_impact_books_close():
    spec = rs.RubberStripSpec(length=0.012, width=0.002, thickness=0.002, n_t=2, case="free")
    live = LiveRubberMaterial(spec, material=_material(), control_dt=1e-4, gravity=(0, 0, 0), stable_step_guard=1.0,
                              contact=ContactSpec(planes=((1, 0, 0, 0),), law=V2))
    s = live.solver
    V = 0.2
    s.set_initial_velocity((-V, 0.0, 0.0))
    E0 = 0.5*s.mass_value*s.n*V*V
    worst = 0.0
    for _ in range(15):
        live.advance()
        d, c = live.last_diagnostics, live.last_contact
        worst = max(worst, abs(d["kinetic_particle"]+d["elastic"]-c["contact_work_material"]-E0)/E0)
    assert worst < 0.03
    assert 0.0 < -c["normal_work_material"] < 0.15*E0
    assert s.v.numpy()[:, 0].mean() > 0.5*V
    assert s.solver_receipt()["options"]["contact_law"] == V2


# ----------------------------------------------------------------------------- batched (RUBBER-CONT-08 under v2)


@pytest.fixture
def batched_v2(monkeypatch):
    """The B1 test scenes (``test_rubber_contact_batched``) with every ContactSpec built as the v2 law."""
    import test_rubber_contact_batched as tb
    base = tb.ContactSpec

    class SpecV2(base):
        def __init__(self, *args, **kw):
            base.__init__(self, *args, **{**kw, "law": V2})
    monkeypatch.setattr(tb, "ContactSpec", SpecV2)
    return tb


@pytest.mark.parametrize("scene", ["inc", "pinch", "pinch_p2"])
def test_v2_batched_l0_heterogeneous(batched_v2, scene):
    tb = batched_v2
    _, kws = tb.SCENES[scene]
    batch, singles = tb._run_l0(scene, kws)
    assert all(live.solver.contact.law == V2 and live.solver.ct_law == V2 for live in singles)
    assert batch.solver.ct_law == V2 and batch.solver.contact_signature["law"] == V2
    assert all(live.last_contact["impulse_events"] > 0 for live in singles)


def test_v1_and_v2_in_one_batch_are_refused():
    from robotic_cutting_sim.batched import rubber_material_batch as rmb
    from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
    spec = rs.RubberStripSpec(length=0.006, width=0.004, thickness=0.002, n_t=2, case="free")
    lives = [LiveRubberMaterial(spec, material=_material(0.5), control_dt=2e-4, gravity=(0, 0, -9.81),
                                contact=ContactSpec(planes=((0, 0, 1, 0),), law=law), strict=False)
             for law in (rct.PENALTY_LAW, V2)]
    with pytest.raises(UnsupportedBatchConfiguration, match="UNSUPPORTED_NONUNIFORM_BATCH"):
        rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True)
