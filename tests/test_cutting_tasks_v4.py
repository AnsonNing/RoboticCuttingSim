"""cutting_tasks_v4 (WP-11 Phase C4): v2-knife TRIM-02/CUT-03 tasks, chisel variant, tau/psi/rho/mu metrics."""
import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robotic_cutting_sim.skim_orientation import motion_elevation_deg, pose_orientation_metrics, skim_orientation_report
from robotic_cutting_sim.tasks import (PROFILE_TASK_IDS, PROTOCOL_PATH, PROTOCOL_V4_PATH, TASK_IDS, V4_TASK_IDS,
                                       ProfileTaskSpecV4, make_task, rake_edge_orientation_xyzw)
from robotic_cutting_sim.tool_profile import (PROFILES, RULER_KNIFE_V2_Y, RULER_KNIFE_V2_Y_CHISEL, handle_box_edge,
                                              make_profile_tool_spec, require_profile_consistency)

V1 = json.loads(PROTOCOL_PATH.read_text())
V4 = json.loads(PROTOCOL_V4_PATH.read_text())
RAKES = {"R00": 0., "R05": 5., "R10": 10., "R12P5": 12.5, "R15": 15., "R25": 25., "R35": 35.}


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def test_v4_protocol_declares_freeze_and_ids():
    assert V4["protocol_id"] == "cutting_tasks_v4_20261030"
    assert "before any v4" in V4["freeze_boundary"] and "unchanged" in V4["freeze_boundary"]
    assert tuple(V4["tasks"]) == V4_TASK_IDS == ProfileTaskSpecV4.TASK_IDS
    # Opt-in: live runtimes keep rejecting v4 ids until a live phase adds the physical-grasp guard.
    assert not set(V4_TASK_IDS) & set(TASK_IDS) and not set(V4_TASK_IDS) & set(PROFILE_TASK_IDS)


@pytest.mark.parametrize("task_id", V4_TASK_IDS)
def test_v4_tasks_inherit_v1_thresholds_and_geometry(task_id):
    task = make_task(task_id)
    entry = V4["tasks"][task_id]
    base = V1[entry["base"]]
    limits = task.protocol_document()[task_id]
    for key, value in base.items():
        if key.startswith(("minimum_", "maximum_", "required_", "surface_")) or key in (
                "geometry", "fixture_boxes_local", "edge_position_knots_local_txyz", "target_origin_local_m",
                "target_normal_local", "particle_spacing_m", "grid_dx_m", "goal_kind"):
            assert limits[key] == value, key
    assert task.edge_clearance_m == base.get("edge_clearance_m", 0.)
    assert task.tool_profile is PROFILES[entry["tool_profile"]]
    record = task.to_dict()
    assert record["protocol_sha256"] == _sha(PROTOCOL_V4_PATH.read_bytes())
    assert record["inherited_protocol_sha256"] == _sha(PROTOCOL_PATH.read_bytes())
    assert record["tool_profile"]["profile_id"] == entry["tool_profile"]


@pytest.mark.parametrize("tag", RAKES)
def test_trim02_rake_orientation_metrics(tag):
    task = make_task(f"TRIM-02-K2-{tag}")
    assert task.blade_rake_deg == RAKES[tag] and task.goal_kind == "flatten"
    tau, psi, rho = pose_orientation_metrics(task.edge_orientation_local_xyzw, [0, 0, 1], [1, 0, 0])
    assert np.allclose([tau[0], psi[0], rho[0]], [0, 0, RAKES[tag]], atol=1e-9)
    record = task.to_dict()["declared_orientation_metrics"]
    assert record["blade_rake_deg"] == RAKES[tag] and record["edge_tilt_deg"] == record["edge_skew_deg"] == 0


def test_r25_reproduces_trim01r_orientation_and_path():
    a, b = make_task("TRIM-01-R"), make_task("TRIM-02-K2-R25")
    assert np.allclose(a.edge_orientation_local_xyzw, b.edge_orientation_local_xyzw, atol=1e-15, rtol=0)
    assert a.edge_position_knots_local_txyz == b.edge_position_knots_local_txyz
    assert a.geometry == b.geometry and a.fixture_boxes_local == b.fixture_boxes_local
    assert np.array_equal(a.sample_particles()[0], b.sample_particles()[0])
    c = make_task("CUT-03-K2")
    assert c.edge_orientation_local_xyzw == make_task("CUT-01-R").edge_orientation_local_xyzw and c.blade_rake_deg is None
    assert c.goal_kind == "sever" and c.tool_profile_id == "ruler_knife_v2_y"


def test_rake_orientation_rejects_undefined_tilt_skew_and_mismatch():
    with pytest.raises(NotImplementedError):
        rake_edge_orientation_xyzw(5., tilt_deg=1.)
    with pytest.raises(NotImplementedError):
        rake_edge_orientation_xyzw(5., skew_deg=1.)
    from dataclasses import replace
    with pytest.raises(ValueError):
        replace(make_task("TRIM-02-K2-R05"), blade_rake_deg=10.)
    with pytest.raises(NotImplementedError):
        make_task("TRIM-02-K2-R05", variant="randomized")


def test_existing_profiles_unchanged_and_chisel_is_additive():
    # Digests of the profiles at 24e2bef (before the chisel entry was added).
    pins = {"ruler_knife_v1": ("d1c280791de17718b5755d1c1fed21c96d0d2a771a0dc78c302ab1dc6398eed6",
                               "808967427095beeced71278b05298f024ba9304540ac7800952e9648cf09c213",
                               "634f1104e007693e97a84514bd8c19a0516df43f7af6c238074cc98a3878cf19"),
            "ruler_knife_v2_y": ("37b71eddf6801e0286db0994d73dfafff9134f8b2fbd925187b5ba6c5302af04",
                                 "7987f0bf51c6dee0d9d26c879baddbc9d02d1a9d1c578470ef93c576e7df7789",
                                 "bf8afe3b2cba3c479dcda751e956033b04648e9c7e61d07cbe0a0881196d2dcf"),
            "ruler_knife_v2_x": ("303e01c159fd57b580dc95f8a8e15e91c1fa820f3ed13836c06886a417d5e530",
                                 "1948c1698d4583a6a675c42e8ec7b150eca51b3f547abe299c656966b805e5d1",
                                 "c490a883732e3f6eca8eb8c053f1a942c43b36d9d007c9c8945f06d5d315b780")}
    assert set(PROFILES) == {*pins, "ruler_knife_v2_y_chisel"}
    for profile_id, (record, arrays, handle) in pins.items():
        profile = PROFILES[profile_id]
        spec = make_profile_tool_spec(profile)
        blob = b"".join(np.ascontiguousarray(a).tobytes() for s in spec.shapes for a in (s.vertices_tool_m, s.triangles))
        center, half = handle_box_edge(profile)
        assert _sha(json.dumps(profile.to_dict(), sort_keys=True).encode()) == record, profile_id
        assert _sha(blob) == arrays and _sha(np.r_[center, half].tobytes()) == handle, profile_id


def test_chisel_profile_geometry():
    chisel, double = RULER_KNIFE_V2_Y_CHISEL, RULER_KNIFE_V2_Y
    for name in ("edge_length_m", "blade_depth_m", "thickness_m", "bevel_included_angle_deg", "blade_density_kg_m3",
                 "handle_half_extents_hand_m", "blade_normal_hand_axis"):
        assert getattr(chisel, name) == getattr(double, name)
    assert chisel.bevel == "single" and double.bevel == "double"
    section = chisel.cross_section_xz()
    assert np.allclose(section, [[0, 0], [-.0015/np.tan(np.deg2rad(25)), .0015], [-.04, .0015], [-.04, 0]])
    spec = make_profile_tool_spec(chisel)
    require_profile_consistency(spec, chisel)
    assert chisel.hand_from_edge() == double.hand_from_edge()

    def lowest(profile, rho):
        R = Rotation.from_euler("y", rho, degrees=True).as_matrix()
        s = profile.cross_section_xz()
        return (np.c_[s[:, 0], np.zeros(len(s)), s[:, 1]]@R.T)[:, 2].min()
    # Flat face toward the target: no chisel point below the edge for rho >= 0; the double bevel drops below
    # the edge for rho < 12.5 deg (0.75 mm at 0 deg).
    for rho in (0., 5., 10., 12.5, 25.):
        assert lowest(chisel, rho) >= -1e-15
    assert np.isclose(lowest(double, 0.), -.00075) and lowest(double, 10.) < 0 and abs(lowest(double, 12.5)) < 1e-12


def test_skim_orientation_metrics_signs_and_motion():
    # rho > 0 when the blade body (edge -x) rises behind the edge: R_y(+rho) tilts the advance downward.
    q = Rotation.from_euler("y", [[0.], [10.], [-10.]], degrees=True).as_quat()
    tau, psi, rho = pose_orientation_metrics(q, [0, 0, 1], [1, 0, 0])
    assert np.allclose(rho, [0, 10, -10]) and np.allclose(tau, 0) and np.allclose(psi, 0)
    body = Rotation.from_quat(q[1]).apply([-.04, 0, 0])
    assert body[2] > 0
    # Edge tilt about the cutting direction and skew about the target normal are separate from rake.
    tilt = Rotation.from_euler("x", 7., degrees=True).as_quat()
    skew = Rotation.from_euler("z", 9., degrees=True).as_quat()
    assert np.allclose(pose_orientation_metrics(tilt, [0, 0, 1], [1, 0, 0])[0], 7.)
    assert np.allclose(pose_orientation_metrics(skew, [0, 0, 1], [1, 0, 0])[1], 9.)
    mu = motion_elevation_deg([[0, 0, 0], [1e-3, 0, 0], [2e-3, 0, 1e-3], [2e-3, 0, 1e-3]], [0, 0, 1])
    assert np.allclose(mu[:2], [0, 45]) and np.isnan(mu[2])
    poses = np.c_[np.linspace(0, .01, 5), np.zeros((5, 2)), np.tile(q[1], (5, 1))]
    report = skim_orientation_report(poses, [0, 0, 1], [1, 0, 0])
    assert np.isclose(report["rho_blade_rake_deg"]["mean"], 10.) and report["mu_motion_elevation_deg"]["max_abs"] == 0.
