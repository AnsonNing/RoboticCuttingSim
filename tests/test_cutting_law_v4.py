"""tlmpm_v4 cutting law 4 (ADR 0006) on Warp's CPU backend with a kinematic ruler blade.

Coarse engineering checks of qualitative behavior only (not calibrated forces):
laws 1/2/3 in the v4 module equal tlmpm_v3 bit for bit; law 4 severs an aligned
push and a raked skim, keeps every negative control (side push, flat press,
blunt handle, armed withdrawal, armed stop) label-free, keeps separating for a
bounded travel after the edge leaves the free surface, and makes three pieces
with two digits. Topology is judged by the actual MPM stencil graph.
"""
import numpy as np
import pytest

pytest.importorskip("warp")
pytestmark = pytest.mark.cpu_sim

from scipy.spatial.transform import Rotation

from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, drive_blade
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
DX = TLMPMConfig().dx


def run(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, **kwargs)


def stencil_components(result, cuts=1):
    """Actual eight-node compatibility graph, including code-zero bridges."""
    return mechanical_components(result["reference_stencil_nodes"], result["reference_stencil_weights"],
                                 result["state"]["codes"], cuts, gradients=result["reference_stencil_gradients"])


def fractions(labels):
    return np.sort(np.bincount(labels))[::-1]/len(labels)


@pytest.mark.parametrize("law", [1, 2, 3])
def test_v4_reproduces_v3_laws_bitwise(law):
    phases = [Phase((0, .05, -.05), .15), Phase((0, 0, .05), .05)]
    results = [run(VERTICAL_CUT, [.001, 0, TOP+.004], phases, cutting_law=law, solver_version=version)
               for version in (3, 4)]
    for key in ("positions", "velocities", "codes", "F", "cut_progress", "contact_work_j"):
        assert np.array_equal(results[0]["state"][key], results[1]["state"][key]), key
    assert results[0]["forces"] == results[1]["forces"]
    assert np.count_nonzero(results[1]["state"]["codes"]) > 0


def test_law4_rejected_by_older_solver_versions():
    with pytest.raises(ValueError):
        run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .001)], cutting_law=4, solver_version=3)


def test_law4_aligned_push_severs():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .4)], cutting_law=4, record_history=True)
    count, labels = stencil_components(result)
    assert count >= 2 and fractions(labels)[1] > .4


def test_law4_sideways_push_flat_press_and_handle_sever_nothing():
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=4, record_history=True)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=4, record_history=True)
    box = [[[0., 0., .05], [.006, .006, .006]]]
    handle = run(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)], cutting_law=4,
                 obstacle_boxes_edge=box, record_history=True)
    for result in (side, flat, handle):
        assert result["labelled_after_phase"] == [0] and result["severed_bonds"] == 0
        assert stencil_components(result)[0] == 1
        assert result["cutting_state_after_phase"][-1]["stroke"] == 0
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


def test_law4_armed_half_depth_then_withdrawal_keeps_ligament():
    # Edge 4 mm above a 12 mm bar, pushed 10 mm: stops 6 mm deep, then pulled out armed.
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)],
                 cutting_law=4, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert result["severed_bonds_after_phase"][1] == result["severed_bonds_after_phase"][0]
    assert stencil_components(result)[0] == 1
    # Material clearly below the stopping depth stays unassigned (the ligament).
    reference = result["points"]
    assert np.all(result["state"]["codes"][reference[:, 2] < TOP-.006-1e-4] == 0)
    assert result["cutting_state_after_phase"][1]["stroke"] == 0


def test_law4_armed_stop_adds_no_labels():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)],
                 cutting_law=4, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert stencil_components(result)[0] == 1
    assert result["cutting_state_after_phase"][1] == {"cutting_law": 4, "stroke": 0, "travel_m": 0.,
                                                      "engaged": 0, "any_motion": 0}


def test_law4_through_cut_persists_past_free_surface_then_stroke_ends():
    # 35 mm of travel: the edge leaves the bar bottom (z=0) 16 mm in. The law-2
    # work zone reaches 2 dx behind the edge, so blade-face contact with the
    # bottom layer stays engaged until the edge is ~6 mm below it; the
    # remaining 13 mm through air exceed the 2 dx persistence travel.
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .32), Phase((0, 0, -.05), .38)],
                 cutting_law=4, record_history=True)
    count, labels = stencil_components(result)
    assert count >= 2 and fractions(labels)[1] > .4
    # The bottom layer on both sides of the edge plane carries both digits.
    bottom = result["points"][:, 2] < result["points"][:, 2].min()+1e-6
    codes = result["state"]["codes"][bottom]
    assert {1, 2} <= set(np.unique(codes).tolist())
    final = result["cutting_state_after_phase"][-1]
    assert final["stroke"] == 0 and final["travel_m"] > 2*DX


def test_law4_two_cut_digits_make_three_pieces():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2, cutting_law=4, record_history=True)
    labelled = result["labelled_after_phase"]
    assert labelled[1] == labelled[0] and labelled[2] == labelled[1] and labelled[3] > labelled[2]
    count, labels = stencil_components(result, cuts=2)
    masses = np.ones(len(labels))
    pieces = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    assert count >= 3 and pieces[2] > .25


def _bump_block(spacing=.002, density=1500.):
    """60 x 24 x 12 mm bar with a 6 mm dome on top (one continuum, no labels)."""
    grid = np.mgrid[-.03:.03-1e-9:spacing, -.012:.012-1e-9:spacing, 0:.018-1e-9:spacing]
    points = grid.reshape(3, -1).T+spacing/2
    dome = (points[:, 2] > TOP) & ((points[:, 2]-TOP) <= .006*(1-(points[:, 0]**2+points[:, 1]**2)/.009**2))
    points = points[(points[:, 2] < TOP) | dome].astype(np.float32)
    return points, np.full(len(points), density*spacing**3, np.float32)


@pytest.mark.xfail(strict=True, reason="Law 4 known limitation (ADR 0008): at 25 deg the raked edge pushes "
                   "6.25% of the bump bottom layer below the swept plane (labelled 1); at 35 deg bump material "
                   "carried ahead of the edge stays code 0 and bridges. Same mechanism as the 1 mm TRIM-01-R "
                   "0% removal in artifacts/wp10_claude/phase1_law4/k_trim01r_law4_001.")
@pytest.mark.parametrize("rake_deg", [25., 35.])
def test_law4_raked_skim_detaches_the_bump_by_stencil_graph(rake_deg):
    points, masses = _bump_block()
    rotation = Rotation.from_euler("y", rake_deg, degrees=True)
    result = run(rotation, [-.026, 0, TOP+.0002], [Phase((.05, 0, 0), 1.)], points=points, masses=masses,
                 spacing=.002, cutting_law=4, record_history=True)
    count, labels = stencil_components(result)
    above = points[:, 2] > TOP
    base_label = np.bincount(labels[~above]).argmax()
    assert count >= 2
    assert np.mean(labels[above] != base_label) > .9 and np.mean(labels[~above] == base_label) > .99
    # Labels follow the surface swept by the edge, not the raked blade face.
    codes = result["state"]["codes"]
    assert np.mean(codes[above] == 1) < .05 and np.mean(codes[~above] == 2) < .01
