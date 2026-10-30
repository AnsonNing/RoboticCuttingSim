"""CPU tests of the pure helpers of scripts/run_wp12_board_throughcut.py (no GPU, no solver run)."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("wp12_board_throughcut", ROOT / "scripts/run_wp12_board_throughcut.py")
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def test_vertical_blade_lowest_point_is_the_edge_vertex():
    from robotic_cutting_sim.goal_planner import through_cut_rotation
    from robotic_cutting_sim.tool_profile import PROFILES
    profile = PROFILES["ruler_knife_v2_y"].tlmpm_profile()[0]
    assert np.allclose(profile, PROFILES["ruler_knife_v1"].tlmpm_profile()[0])  # identical TLMPM cross-section
    lowest, vertex, slope = study.lowest_blade_point(profile, through_cut_rotation([1., 0., 0.]))
    assert vertex == 0 and lowest == pytest.approx(0.) and slope == pytest.approx(0., abs=1e-12)
    # Tilting about the edge line keeps the edge lowest until a bevel face passes horizontal
    # (tilt > 90 - 12.5 deg); at 30 deg it is still the edge, at 80 deg a bevel corner is lower.
    tilt = lambda deg: Rotation.from_rotvec([0., np.deg2rad(deg), 0.])*through_cut_rotation([1., 0., 0.])
    assert study.lowest_blade_point(profile, tilt(30.))[1] == 0
    assert study.lowest_blade_point(profile, tilt(80.))[1] != 0


def test_box_sdf_and_contact_impulse_match_the_kernel_rules():
    normal, sdf = study.box_sdf(np.array([[0., 0., .001], [0., 0., -.002], [.1, 0., 0.]]), [0., 0., -.005], [.09, .025, .005])
    assert sdf == pytest.approx([.001, -.002, .01])
    assert np.allclose(normal[0], [0., 0., 1.]) and np.allclose(normal[1], [0., 0., 1.])
    # Penalty only for a (predicted) negative gap; friction capped by mu * jn.
    j = study.contact_impulse(np.array([-1e-4, 1e-4]), np.array([[0., 0., 1.]]*2), np.array([[1., 0., 0.]]*2),
                              np.array([1e-3, 1e-3]), 1e-4, .1, .3)
    jn = .1*1e-3/1e-4*1e-4
    assert j[0] == pytest.approx([-.3*jn, 0., jn]) and np.allclose(j[1], 0.)


def test_crossing_links_classify_conflicts_per_layer():
    lattice = np.array([[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
    digits = np.array([1, 2, 0, 2, 1, 1])
    rows = study.crossing_links(lattice, digits, 0)
    assert rows[0] == dict(links=1, conflicting=1, unconflicted=0, both_unassigned=0, one_unassigned=0, same_side=0)
    assert rows[1] == dict(links=2, conflicting=0, unconflicted=2, both_unassigned=0, one_unassigned=1, same_side=1)


def test_direct_bridges_need_both_sides_at_the_stencil():
    nodes = np.array([[0, 1], [1, 2], [2, 3], [5, 6]])
    valid = np.ones_like(nodes, bool)
    digits = np.array([1, 0, 2, 0])
    assert study.direct_bridges(nodes, valid, digits).tolist() == [1]
    assert study.direct_bridges(nodes, valid, np.array([1, 0, 1, 0])).tolist() == []


def test_rows_and_scenes_compile_without_hidden_supports():
    from robotic_cutting_sim.scene_config import raw_support_boxes, validate_scene
    assert study.ROWS["S_M025_DIAG"]["label"] == study.DIAGNOSTIC_LABEL
    assert all("label" not in spec for row, spec in study.ROWS.items() if row != "S_M025_DIAG")
    solid = raw_support_boxes(validate_scene(study.scene_for("S_000")))
    assert len(solid) == 1 and solid[0].center_m[2] + solid[0].half_extents_m[2] == pytest.approx(0.)
    assert solid[0].half_extents_m[0] >= .08 and solid[0].half_extents_m[1] >= .015
    slot = raw_support_boxes(validate_scene(study.scene_for("D8_000")))
    gaps = sorted(b.center_m[0] + s*b.half_extents_m[0] for b in slot for s in (-1, 1))
    assert gaps[1] == pytest.approx(-.004) and gaps[2] == pytest.approx(.004)
    assert study.two_part_check([.5, .495, .005]) == {"two_substantial_parts_each_ge_40pct": True,
                                                       "two_main_parts_together_ge_99pct": True}
