"""P6.4: CPU tests of the Native rubber host bridge (no Kit): geometry, schedule, surface mesh, and the measured-pose
bridge loop with a fake PhysX adapter and a small CPU ``LiveRubberMaterial`` of the P2-6 profile shape (F8 + kick
average + F9 + F10 + guard + S1 + a penalty contact law, Prony M = 2). A replay of the recorded inputs reproduces the
material inputs bit for bit (and, on CPU, the statistics)."""
import math

import numpy as np
import pytest

from robotic_cutting_sim import native_rubber_host as H

wp = pytest.importorskip("warp")


def test_stat_names_are_the_46_declared_statistics():
    assert len(H.STAT_NAMES) == 46 == 27+6+6+7
    assert H.STAT_NAMES[:3] == ("marker0_x", "marker0_y", "marker0_z")
    assert H.STAT_NAMES[-7:] == ("psi_eq_j", "kinetic_j", "gravity_energy_j", "contact_work_j", "jbar_min",
                                 "jbar_max", "cfl_eff_max")
    assert len(set(H.STAT_NAMES)) == 46


def test_schedule_hold_ramp_hold():
    o = np.array([0.0, 0.008, 0.002])
    for i in (0, 10, 24):
        p, q = H.commanded_grip_pose(i, o)
        assert np.array_equal(p, o) and np.array_equal(q, [0, 0, 0, 1])
    p25, _ = H.commanded_grip_pose(25, o)
    assert 0 < p25[2]-o[2] < 1e-3
    zs = [H.commanded_grip_pose(i, o)[0][2] for i in range(24, 50)]
    assert all(b > a for a, b in zip(zs, zs[1:]))                # strictly increasing during the ramp
    for i in (49, 50, 99):
        p, q = H.commanded_grip_pose(i, o)
        assert p[2]-o[2] == pytest.approx(0.020, abs=1e-15)
        assert 2*math.atan2(q[2], q[3]) == pytest.approx(0.3, abs=1e-15)
    with pytest.raises(IndexError):
        H.commanded_grip_pose(100, o)


def test_grip_motion_maps_the_origin_to_p():
    from scipy.spatial.transform import Rotation
    o = np.array([0.0, 0.008, 0.002])
    q = Rotation.from_rotvec([0.1, -0.2, 0.3]).as_quat()
    p = np.array([0.01, 0.02, 0.03])
    m = H.grip_motion(p, q, o)
    assert np.allclose(m["rotation"], Rotation.from_quat(q).as_matrix(), atol=1e-15)
    assert np.allclose(m["rotation"]@o+m["translation"], p, atol=1e-16)
    assert m["active"] is True


def test_placement_round_trip_and_receipt():
    pl = H.StripPlacement((-0.0615, -0.13, -0.01))
    pose = pl.pose_to_world([0.0, 0.008, 0.002], [0, 0, 0, 1])
    p, q = pl.pose_from_world(pose)
    assert np.allclose(p, [0.0, 0.008, 0.002], atol=1e-17) and np.array_equal(q, [0, 0, 0, 1])
    assert pl.receipt()["kind"] == "translation_only"


def _scene(length=0.006, width=0.003, thickness=0.002, n_t=2):
    from robotic_cutting_sim import rubber_scene as rs
    return rs.compile_strip(rs.RubberStripSpec(length=length, width=width, thickness=thickness, n_t=n_t, case="A"))


def test_surface_mesh_is_a_closed_outward_shell_over_the_surface_particles():
    sc = _scene()
    ids, faces = H.surface_mesh(sc)
    assert np.array_equal(ids, np.flatnonzero(sc.surface))
    assert faces.dtype == np.int32 and faces.min() == 0 and faces.max() == len(ids)-1
    assert len(np.unique(faces)) == len(ids)                      # every surface particle is a vertex
    # closed 2-manifold: every undirected edge in exactly two triangles, every directed edge once
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    assert len({tuple(e) for e in directed.tolist()}) == len(directed)
    und = np.sort(directed, 1)
    _, counts = np.unique(und, axis=0, return_counts=True)
    assert (counts == 2).all()
    x = sc.X.astype(np.float64)[ids]
    a, b, c = x[faces[:, 0]], x[faces[:, 1]], x[faces[:, 2]]
    assert (np.linalg.norm(np.cross(b-a, c-a), axis=1) > 0).all()
    volume = np.einsum("ij,ij->i", a, np.cross(b, c)).sum()/6.0    # outward winding -> positive enclosed volume
    nx, ny, nz = sc.lattice
    s = sc.spacing
    assert volume == pytest.approx((nx-1)*(ny-1)*(nz-1)*s**3, rel=1e-5)


def test_bridge_integrity_and_depth_consistency_helpers():
    pts = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    r = H.bridge_integrity(pts.astype(np.float32), pts, [[0, 1, 1]], [[0, 1, 1]])
    assert r["max_abs_error_m"] < 1e-6 and r["topology_equal"]
    assert H.bridge_integrity(pts[:1], pts)["max_abs_error_m"] == math.inf
    K = np.array([[100.0, 0, 50], [0, 100, 50], [0, 0, 1]])
    T = np.eye(4)                                                   # camera at origin looking +z (OpenCV)
    depth = np.full((100, 100), 0.98, np.float32)
    c = H.marker_depth_consistency([[0.0, 0.0, 1.0], [10.0, 0.0, 1.0]], K, T, depth)
    assert c["visible"] == 1 and c["residual_min_m"] == pytest.approx(-0.02, abs=1e-6)
    assert c["markers"][1]["rendered_depth_m"] is None


# ----------------------------------------------------------------------------- bridge loop with a fake PhysX

CDT = 5e-4
EPISODE_SMALL = dict(H.EPISODE, steps=4, hold_until_step=1, ramp_steps=2, lift_m=2e-4, yaw_rad=0.05,
                     capture_every=2)


def _life(record=True):
    from robotic_cutting_sim import rubber_scene as rs
    from robotic_cutting_sim.live_rubber_material import LiveRubberMaterial
    from robotic_cutting_sim.rubber_contact import ContactSpec, PENALTY_LAW_V2
    from robotic_cutting_sim.rubber_material import MaterialValues
    spec = rs.RubberStripSpec(length=0.006, width=0.003, thickness=0.002, n_t=2, case="A")
    mat = MaterialValues(mu0=4e5, kappa=8e6, density=1150.0, friction_table=0.35, friction_pad=1.0,
                         friction_self=0.9, prony=((0.15, 0.01), (0.10, 0.1)))
    return LiveRubberMaterial(spec, material=mat, control_dt=CDT, device="cpu", record=record, strict=False,
                              stable_step_guard=1.0, dirichlet="consistent_mass_projection_v1",
                              traction_work="grid_displacement_v1", state_precision="f64_shadow_v1",
                              constraint_work="kick_average_v1",
                              contact=ContactSpec(planes=((0.0, 0.0, 1.0, 0.0),), law=PENALTY_LAW_V2,
                                                  normal_stiffness=0.25, tangential_stiffness=0.25,
                                                  damping_ratio=0.5, normal_damping_ratio=0.5),
                              self_penetration_check="surface_grid_v1", self_penetration_check_substeps=8)


class FakePhysx:
    """PhysX stand-in: the measured pose is the float32 target plus a small deterministic tracking offset, so the
    material input must be the MEASURED pose, not the commanded one."""

    def __init__(self):
        self.target, self.steps, self.calls = None, 0, []

    def set_clamp_target(self, pose7):
        self.target = np.asarray(pose7, np.float64)
        self.calls.append("target")

    def step(self):
        self.steps += 1
        self.calls.append("step")

    def measured_clamp_pose(self):
        self.calls.append("measure")
        m = self.target.astype(np.float32).astype(np.float64)
        m[:3] += np.array([1e-7, -2e-7, 3e-7])*self.steps
        m[3:] /= np.linalg.norm(m[3:])
        return m


class FakeSurface:
    def __init__(self):
        self.points, self.faces, self.times = None, None, []

    def update(self, vertices, faces, t):
        self.points, self.faces = np.asarray(vertices, np.float32), np.asarray(faces, np.int32)
        self.times.append(t)
        return {"surface_sha256": "x", "topology_sha256": "y"}

    def readback(self):
        return self.points.astype(np.float64), self.faces


@pytest.fixture(scope="module")
def bridged():
    live = _life()
    physx, surface = FakePhysx(), FakeSurface()
    pl = H.StripPlacement((-0.0031, -0.13, -0.01))
    origin = np.asarray(live.scene.groups[0]["origin"], np.float64)
    captures = []
    arrays, rows = H.run_bridge_episode(live, physx, surface, pl, origin, episode=EPISODE_SMALL,
                                        capture=captures.append)
    return live, physx, surface, pl, origin, arrays, rows, captures


def test_bridge_loop_order_and_shapes(bridged):
    live, physx, surface, pl, origin, arrays, rows, captures = bridged
    assert physx.calls == ["target", "step", "measure"]*4
    assert arrays.completed == 4 and live.step_count == 4
    assert captures == [0, 2, 4]
    assert len(rows) == 5 and surface.times == [0.0, CDT, 2*CDT, 3*CDT, 4*CDT]
    assert np.isfinite(arrays.statistics).all()                   # psi_eq, contact work, plane wrench present
    assert (arrays.invalid_reason == 0).all()
    assert len(live.records) == 5 and live.records[0]["kind"] == "header"


def test_material_receives_the_measured_pose_not_the_command(bridged):
    live, physx, surface, pl, origin, arrays, rows, captures = bridged
    p, q = pl.pose_from_world(arrays.measured_world[-1])
    assert np.array_equal(arrays.material_p[-1], p) and np.array_equal(arrays.material_q[-1], q)
    assert np.array_equal(arrays.material_p, arrays.measured_world[:, :3]-pl.translation)
    assert not np.array_equal(arrays.material_p, arrays.commanded_p)   # offset of the fake PhysX
    for i in range(4):
        m = H.grip_motion(arrays.material_p[i], arrays.material_q[i], origin)
        assert np.array_equal(m["rotation"], arrays.motion_rotation[i])
        assert np.array_equal(m["translation"], arrays.motion_translation[i])
        assert np.array_equal(arrays.staged_b1[i], arrays.motion_translation[i])   # translation staged as given
        qq = arrays.staged_q1[i]
        assert min(np.abs(qq-arrays.material_q[i]).max(), np.abs(qq+arrays.material_q[i]).max()) < 1e-12


def test_surface_vertices_are_surface_particles_under_the_strip_pose(bridged):
    live, physx, surface, pl, origin, arrays, rows, captures = bridged
    ids, faces = H.surface_mesh(live.scene)
    expected = pl.to_world(H.positions64(live)[ids])
    assert np.abs(surface.points.astype(np.float64)-expected).max() <= 1e-6
    assert all(r["max_abs_error_m"] <= 1e-6 and r["topology_equal"] for r in rows)
    assert np.array_equal(surface.faces, faces)


def test_replay_of_the_recorded_inputs_is_bitwise_on_cpu(bridged, tmp_path):
    from robotic_cutting_sim import rubber_native_compare as C
    live, physx, surface, pl, origin, arrays, rows, captures = bridged
    arrays.save(tmp_path/"native.npz")
    nat = H.EpisodeArrays.load(tmp_path/"native.npz")
    rep = H.replay_bridge_inputs(_life(), nat["material_p"], nat["material_q"], origin)
    rep.save(tmp_path/"replay.npz")
    r = H.EpisodeArrays.load(tmp_path/"replay.npz")
    check = C.check_inputs(nat, {"replay_00": r}, pl.translation)
    assert check["verdict"] == "PASS", check
    assert np.array_equal(r["statistics"], nat["statistics"])     # CPU Warp is deterministic
    # a one-ulp change of one recorded input is detected
    bad = dict(r, material_p=np.nextafter(r["material_p"], 1.0))
    assert C.check_inputs(nat, {"bad": bad}, pl.translation)["verdict"] == "FAIL"


def test_receipts_of_two_lives_are_identical_except_timing():
    from robotic_cutting_sim import rubber_native_compare as C
    a, b = _life(), _life()

    class Fam:
        def receipt(self):
            return {"id": "fake", "slots": ["x"]}
    ra, rb = H.strip_receipt(Fam(), a, "x"), H.strip_receipt(Fam(), b, "x")
    res = C.compare_receipts(ra, rb)
    assert res["identical"], res["differences"][:5]
    assert any(e["path"].endswith("build_wall_s") for e in res["exempt"])


def test_bridge_with_the_family_profile_on_the_cpu_parity_slot():
    """The host drives a ``RubberStripTaskFamily.build_life`` (declared profile, CPU parity test configuration)."""
    fam_mod = pytest.importorskip("robotic_cutting_sim.rl.rubber_family")
    fam = fam_mod.RubberStripTaskFamily(["P16_t4"], variant="cpu_parity_v1")
    live = fam.build_life("P16_t4", device="cpu", record=True)
    origin = np.asarray(fam.geometry("P16_t4")["origin"], np.float64)
    ep = dict(EPISODE_SMALL, steps=2, hold_until_step=1, ramp_steps=1, capture_every=1)
    arrays, rows = H.run_bridge_episode(live, FakePhysx(), FakeSurface(), H.StripPlacement((0.0, -0.1, -0.01)),
                                        origin, episode=ep)
    assert arrays.completed == 2 and np.isfinite(arrays.statistics).all() and (arrays.invalid_reason == 0).all()
    assert max(r["max_abs_error_m"] for r in rows) <= 1e-6
    r = H.strip_receipt(fam, live, "P16_t4")
    assert r["material"]["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED"
