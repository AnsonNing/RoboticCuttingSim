"""tlmpm_v3 cutting law 3 on Warp's CPU backend with a kinematic ruler blade.

Coarse engineering checks of qualitative behavior only (not calibrated forces):
law 1/2 in the v3 module equal tlmpm_v2 bit for bit; law 3 keeps every law-2
negative control (side push, flat press, armed withdrawal, blunt handle) and
separates a raked skim through a bump, judged by the actual MPM stencil graph.
"""
import numpy as np
import pytest

pytest.importorskip("warp")
pytestmark = pytest.mark.cpu_sim

from scipy.spatial.transform import Rotation

from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, drive_blade
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012


def run(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, **kwargs)


def stencil_components(result, cuts=1):
    """Actual eight-node compatibility graph, including code-zero bridges."""
    count, labels = mechanical_components(result["reference_stencil_nodes"], result["reference_stencil_weights"],
                                          result["state"]["codes"], cuts, gradients=result["reference_stencil_gradients"])
    return count, labels


@pytest.mark.parametrize("law", [1, 2])
def test_v3_reproduces_v2_laws_bitwise(law):
    phases = [Phase((0, .05, -.05), .15), Phase((0, 0, .05), .05)]
    results = [run(VERTICAL_CUT, [.001, 0, TOP+.004], phases, cutting_law=law, solver_version=version)
               for version in (2, 3)]
    for key in ("positions", "velocities", "codes", "F", "cut_progress", "contact_work_j"):
        assert np.array_equal(results[0]["state"][key], results[1]["state"][key]), key
    assert results[0]["forces"] == results[1]["forces"]
    assert np.count_nonzero(results[1]["state"]["codes"]) > 0


def test_law3_aligned_push_severs():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .4)], cutting_law=3, record_history=True)
    count, labels = stencil_components(result)
    fractions = np.sort(np.bincount(labels, weights=result["state"]["positions"][:, 0]*0+1))[::-1]/len(labels)
    assert count >= 2 and fractions[1] > .4


def test_law3_sideways_push_flat_press_and_handle_sever_nothing():
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=3)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=3)
    box = [[[0., 0., .05], [.006, .006, .006]]]
    handle = run(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)], cutting_law=3, obstacle_boxes_edge=box)
    for result in (side, flat, handle):
        assert result["severed_bonds"] == 0 and result["components"] == 1
    assert side["labelled_after_phase"] == [0]
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


def test_law3_armed_withdrawal_adds_no_separation():
    # Cutting stays armed (digit 0) while the edge is pulled straight back out.
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)],
                 cutting_law=3, record_history=True)
    first, second = result["labelled_after_phase"]
    bonds = result["severed_bonds_after_phase"]
    assert first > 0 and second == first and bonds[1] == bonds[0]


def test_law3_slicing_lowers_the_push_resistance():
    push = run(VERTICAL_CUT, [.001, 0, TOP+.001], [Phase((0, 0, -.01), .4)], cutting_law=3)
    slice_ = run(VERTICAL_CUT, [.001, -.03, TOP+.001], [Phase((0, .2, -.01), .4)], cutting_law=3)
    resist = lambda r: r["forces"][0]["mean_N"][2]
    assert resist(slice_) < .25*resist(push)


def test_law3_two_cut_digits_make_three_pieces():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, -1),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2, cutting_law=3)
    assert result["components"] >= 3 and min(result["mass_fractions"][:3]) > .25


def _bump_block(spacing=.002, density=1500.):
    """60 x 24 x 12 mm bar with a 6 mm dome on top (one continuum, no labels)."""
    grid = np.mgrid[-.03:.03-1e-9:spacing, -.012:.012-1e-9:spacing, 0:.018-1e-9:spacing]
    points = grid.reshape(3, -1).T+spacing/2
    dome = (points[:, 2] > TOP) & ((points[:, 2]-TOP) <= .006*(1-(points[:, 0]**2+points[:, 1]**2)/.009**2))
    points = points[(points[:, 2] < TOP) | dome].astype(np.float32)
    return points, np.full(len(points), density*spacing**3, np.float32)


@pytest.mark.parametrize("rake_deg", [25., 35.])
def test_law3_raked_skim_detaches_the_bump_by_stencil_graph(rake_deg):
    points, masses = _bump_block()
    rotation = Rotation.from_euler("y", rake_deg, degrees=True)
    result = run(rotation, [-.026, 0, TOP+.0002], [Phase((.05, 0, 0), 1.)], points=points, masses=masses,
                 spacing=.002, cutting_law=3, record_history=True)
    count, labels = stencil_components(result)
    above = points[:, 2] > TOP
    base_label = np.bincount(labels[~above]).argmax()
    assert count >= 2
    assert np.mean(labels[above] != base_label) > .9 and np.mean(labels[~above] == base_label) > .99
    # Labels follow the surface swept by the edge, not the raked blade face.
    # Only particles squeezed across the edge line at the cut may change side.
    codes = result["state"]["codes"]
    assert np.mean(codes[above] == 1) < .05 and np.mean(codes[~above] == 2) < .01
