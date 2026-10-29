"""tlmpm_v2 cutting law 2 on Warp's CPU backend with a kinematic ruler blade.

Coarse (4 mm) engineering checks of qualitative behavior only: which knife
motions separate material, not calibrated forces. Fracture energy 5 J/m^2.
"""
import numpy as np
import pytest

pytest.importorskip("warp")
pytestmark = pytest.mark.cpu_sim

from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, drive_blade
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012


def run(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, **kwargs)


@pytest.mark.parametrize("law", [1, 2])
def test_aligned_push_severs_under_both_laws(law):
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .4)], cutting_law=law)
    assert result["components"] >= 2 and min(result["mass_fractions"][:2]) > .4


@pytest.mark.parametrize("law", [1, 2])
def test_sideways_push_and_flat_press_sever_nothing(law):
    # Along the blade normal with the edge inside the block, and a flat blade
    # pressed down into the top: neither opens a crack under either law.
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=law)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=law)
    for result in (side, flat):
        assert result["severed_bonds"] == 0 and result["components"] == 1
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    if law == 2:
        # Law 1 still writes one-sided labels here, which a later cut with the
        # same digit would inherit; law 2 leaves the sideways push unlabelled.
        assert side["labelled_after_phase"] == [0]


def test_slicing_lowers_the_push_resistance():
    push = run(VERTICAL_CUT, [.001, 0, TOP+.001], [Phase((0, 0, -.01), .4)])
    slice_ = run(VERTICAL_CUT, [.001, -.03, TOP+.001], [Phase((0, .2, -.01), .4)])
    no_slice_work = run(VERTICAL_CUT, [.001, -.03, TOP+.001], [Phase((0, .2, -.01), .4)], slice_efficiency=0.)
    resist = lambda r: r["forces"][0]["mean_N"][2]
    assert slice_["labelled_after_phase"] == push["labelled_after_phase"]
    assert resist(slice_) < .25*resist(push) and resist(slice_) < resist(no_slice_work)


def test_pulling_back_adds_no_separation():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)])
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first


def test_blunt_obstacle_pushes_without_cutting():
    # A handle-sized box 50 mm beside the blade (edge-frame +z = world +x);
    # the blade itself stays outside the block.
    box = [[[0., 0., .05], [.006, .006, .006]]]
    start, phases = [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)]
    with_box = run(VERTICAL_CUT, start, phases, obstacle_boxes_edge=box)
    without = run(VERTICAL_CUT, start, phases)
    assert with_box["forces"][0]["mean_N"][2] > .5 and abs(without["forces"][0]["mean_N"][2]) < 1e-6
    assert with_box["severed_bonds"] == 0 and with_box["components"] == 1


def test_two_cut_digits_make_three_pieces():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, -1),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2)
    assert result["components"] >= 3 and min(result["mass_fractions"][:3]) > .25


def test_obstacles_are_not_confused_with_fixture_boxes():
    # Regression: the fixture setup once overwrote the obstacle array, so the
    # support floor was applied a second time in the blade frame.
    from robotic_cutting_sim import tlmpm_v2
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.material_config import TLMPMConfig
    points, masses = block()
    obstacle = [[[-.06, 0., 0.], [.02, .05, .011]]]
    bed = tlmpm_v2.TLMPM(points, masses, .004, TLMPMConfig(floor=False, cuts=1), "cpu",
                         fixture_boxes=[[[0., 0., -.005], [.09, .025, .005]]], table_plane=[0., 0., 1., .01],
                         obstacle_boxes_edge=obstacle)
    assert np.allclose(bed.box_centers.numpy(), [obstacle[0][0]]) and np.allclose(bed.box_halves.numpy(), [obstacle[0][1]])
