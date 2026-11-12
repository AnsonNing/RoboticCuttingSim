"""P2-6 device staging of ``BatchedRubberMaterial`` (plan §4.2): ``set_group_motions_device`` / ``set_traction_device``
/ ``advance_device`` are byte-identical to the host path on CPU (ADR 0023 §23.2 decision 7).

The host path computes the END quaternion from a rotation matrix with scipy and hemisphere-aligns it on the host; the
device path receives that quaternion (and the translation, traction) in device arrays and aligns it in a kernel. Every
state array, the motion table and the outputs must match byte for byte after every control step, including steps whose
quaternion lies in the opposite hemisphere (the alignment flips it). Warp CPU only, no Torch.
"""
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
from robotic_cutting_sim.rubber_contact import ContactSpec
from robotic_cutting_sim.rubber_material import MaterialValues

P26 = {"dirichlet": "consistent_mass_projection_v1", "constraint_work": "kick_average_v1",
       "traction_work": "grid_displacement_v1", "state_precision": "f64_shadow_v1"}
STEP = MaterialValues(mu0=8e5, kappa=1.6e7, density=1000.0)


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def _live(L, t, mu0=4e5, rho=1150.0, mu=0.5):
    spec = rs.RubberStripSpec(length=L, width=0.004, thickness=t, n_t=int(round(t/1e-3)), dx=1e-3, case="A",
                              traction_regions=(("tip", (L-0.002, -1, -1), (1, 1, 1)),),
                              resolution_profile="rubber_ref_v1")
    m = MaterialValues(mu0=mu0, kappa=20*mu0, density=rho, friction_table=mu, prony=((0.15, 0.01), (0.1, 0.1)))
    return LiveRubberMaterial(spec, material=m, control_dt=1e-3, step_material=STEP, strict=False,
                              stable_step_guard=1.0, contact=ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),),
                                                                         law="grid_node_penalty_coulomb_v3"),
                              self_penetration_check="surface_grid_v1", **P26)


KW = [dict(L=0.008, t=0.004), dict(L=0.006, t=0.005, mu0=6e5, rho=1250.0, mu=0.7)]


def _pose(k, e):
    """END pose of env e after control step k: a lift, a yaw that crosses the hemisphere boundary by sign choice."""
    yaw = 0.02*k*(1 if e == 0 else -1)
    R = Rotation.from_euler("xyz", [0.01*k, 0.0, yaw]).as_matrix()
    b = np.array([1e-4*k, -5e-5*k*e, 2e-4*k])
    return R, b


def _batch():
    return rmb.BatchedRubberMaterial([_live(**kw) for kw in KW], device="cpu", parity_mode=True, strict=False)


def _all_bytes(batch):
    out = {}
    for e in range(batch.num_envs):
        for k, v in batch.solver.snapshot(e).items():
            out[f"{e}.{k}"] = v
    for k, v in batch.device_arrays().items():
        out[f"dev.{k}"] = v.numpy()
    return out


def test_device_staging_is_bytewise_the_host_path():
    host, dev = _batch(), _batch()
    N = host.num_envs
    q_w = wp.zeros(N, dtype=wp.quatd, device="cpu")
    b_w = wp.zeros(N, dtype=wp.vec3d, device="cpu")
    f_w = wp.zeros(N, dtype=wp.vec3, device="cpu")
    flipped = 0
    for k in range(1, 6):
        R = np.zeros((N, 1, 3, 3))
        b = np.zeros((N, 1, 3))
        for e in range(N):
            R[e, 0], b[e, 0] = _pose(k, e)
        forces = np.array([[1e-3*k, 0.0, -2e-3], [0.0, 5e-4*k, 1e-3]], np.float32)
        q = Rotation.from_matrix(R.reshape(N, 3, 3)).as_quat()
        if k % 2 == 0:
            q = -q                                     # the device path must flip it back like the host path
        flipped += int(np.sum(np.einsum("ij,ij->i", dev.solver.group_q1.numpy(), q) < 0))
        host.advance(rotation=R, translation=b, traction=forces)
        q_w.assign(q)
        b_w.assign(b.reshape(N, 3))
        f_w.assign(forces)
        dev.advance_device(q_w, b_w, f_w)
        a, c = _all_bytes(host), _all_bytes(dev)
        assert set(a) == set(c)
        for name in a:
            assert _same(a[name], c[name]), (k, name)
    assert flipped > 0, "the hemisphere branch was exercised"
    assert host.invalid_reason.numpy().tolist() == [0, 0]


def test_device_staging_refusals():
    b = _batch()
    with pytest.raises(ValueError):
        b.set_group_motions_device(wp.zeros(1, dtype=wp.quatd, device="cpu"), wp.zeros(1, dtype=wp.vec3d, device="cpu"))
    with pytest.raises(ValueError):
        b.set_traction_device(wp.zeros(2, dtype=wp.vec3d, device="cpu"))
    with pytest.raises(ValueError):
        b.advance_device(q1=wp.zeros(2, dtype=wp.quatd, device="cpu"))


def test_advance_device_without_inputs_holds_like_advance():
    a, c = _batch(), _batch()
    for _ in range(2):
        a.advance()
        c.advance_device()
    x, y = _all_bytes(a), _all_bytes(c)
    for name in x:
        assert _same(x[name], y[name]), name
