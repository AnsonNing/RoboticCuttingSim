"""tlmpm_v4 cutting law 8 (ADR 0012) on Warp's CPU backend with a kinematic ruler blade.

Coarse engineering checks of qualitative behavior only (not calibrated forces):
laws 1-7 in the v4 module are unchanged bit for bit by the law-8 addition
(against the git-loaded pre-law-8 module and law-7 digests recorded before law
8); law 8 equals law 4 bit for bit on in-plane cuts (no segment plows, so the
lead rule never runs on any particle) and equals law 5 bit for bit on
constant-rake skims at 25 and 35 degrees (every cutting segment plows, so the
lead mask equals seg_ok); the plowing mask follows the declared 15-degree
threshold per segment; and law 8 keeps every negative control (side push, flat
press, blunt handle, armed stop, armed half-depth withdrawal keeps its ligament)
and makes three pieces with two digits.
"""
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")
pytestmark = pytest.mark.cpu_sim

from scipy.spatial.transform import Rotation

import robotic_cutting_sim.tlmpm_v4 as tlmpm_v4
from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, drive_blade
from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
DX = TLMPMConfig().dx
ROOT = Path(__file__).resolve().parents[1]
PRE_LAW8_COMMIT = "b529929"  # last commit that changed tlmpm_v4.py before law 8 (law 7)
STATE_KEYS = ("positions", "velocities", "codes", "F", "cut_progress", "contact_work_j")
# SHA-256 of law-7 probe results produced by the committed (pre-law-8) tlmpm_v4
# (b529929) on Warp 1.17.0 CPU, recorded before law 8 was added.
LAW7_DIGESTS_WARP = "1.17.0"
LAW7_DIGESTS = {
    "push": {"positions": "0bcce4e4772d1d206dd819e1de48c761adf46ceee5047a770d37a79dab9c46ba",
             "velocities": "10a28ce66708a5c14d129614f71b54fdf3f18ff1ab371e483e5a1c9c28620b16",
             "codes": "d93c4505d743093a4b3ef1780df33c243d198bc15f5e29fda4552e26aa65a5b9",
             "F": "a92cb40e909b9d8f190715a2bd01cff5b9f289dcf2a33cba1841ecf38828a952",
             "cut_progress": "383fcf659a15dbf6c6f543b8a3307561c801ab920211df744308b9747c096949",
             "contact_work_j": "75e1b3c37a529bfcdb3f284edb3113d9fab6caec760b300575b33bb4372ae57e",
             "forces": "1fc4c735a19031981750105437c4c58208d667331d68e6b3baa49cfad05d4ecd"},
    "skim25": {"positions": "375472e82b5daae2fd248ae037859a64a1e13f02885c13777f21c2f861f011c5",
               "velocities": "5a560ce7e6e3e5a37b5154c0fd1891f830a0042eb24be2f740fe67e3626a1ac6",
               "codes": "6c89b0520a75314d962ffa24a48199a8e5cb44c612017ad89a9d3580758ade36",
               "F": "2ce8787fa3a25b1796489d3b41463a2c478c4a1d16dcc8893cf85031fb64beb0",
               "cut_progress": "045c755747acbfcaceead064e6a982f37aabdf3d0ec819e7909621993a6a6b79",
               "contact_work_j": "4d3c03be8934fede97bb64834808cd4747dca3500be740be3094bbae91996427",
               "forces": "0b5a1c2856a5197c66f1cc1934e623634d4bdc39e4340f6195a1a53a4d2b0b90"}}


def run(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, **kwargs)


def stencil_components(result, cuts=1):
    """Actual eight-node compatibility graph, including code-zero bridges."""
    return mechanical_components(result["reference_stencil_nodes"], result["reference_stencil_weights"],
                                 result["state"]["codes"], cuts, gradients=result["reference_stencil_gradients"])


def fractions(labels):
    return np.sort(np.bincount(labels))[::-1]/len(labels)


def _bump_block(spacing=.002, density=1500.):
    """60 x 24 x 12 mm bar with a 6 mm dome on top (same as tests/test_cutting_law_v4.py)."""
    grid = np.mgrid[-.03:.03-1e-9:spacing, -.012:.012-1e-9:spacing, 0:.018-1e-9:spacing]
    points = grid.reshape(3, -1).T+spacing/2
    dome = (points[:, 2] > TOP) & ((points[:, 2]-TOP) <= .006*(1-(points[:, 0]**2+points[:, 1]**2)/.009**2))
    points = points[(points[:, 2] < TOP) | dome].astype(np.float32)
    return points, np.full(len(points), density*spacing**3, np.float32)


def _probe(name, law, **kwargs):
    if name == "push":
        return run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .15), Phase((0, 0, .05), .05)],
                   cutting_law=law, **kwargs)
    points, masses = _bump_block()
    return run(Rotation.from_euler("y", 25., degrees=True), [-.026, 0, TOP+.0002], [Phase((.05, 0, 0), .5)],
               points=points, masses=masses, spacing=.002, cutting_law=law, **kwargs)


def _digest(result):
    digest = {k: hashlib.sha256(np.ascontiguousarray(result["state"][k]).tobytes()).hexdigest() for k in STATE_KEYS}
    digest["forces"] = hashlib.sha256(json.dumps(result["forces"]).encode()).hexdigest()
    return digest


def _without_law(states):
    return [{k: v for k, v in s.items() if k != "cutting_law"} for s in states]


def _assert_bitwise(a, b):
    for key in STATE_KEYS:
        assert np.array_equal(a["state"][key], b["state"][key]), key
    assert a["forces"] == b["forces"]
    assert _without_law(a["cutting_state_after_phase"]) == _without_law(b["cutting_state_after_phase"])


# ----------------------------------------------------------------------------- laws 1-7 unchanged

@pytest.mark.parametrize("probe", ["push", "skim25"])
def test_law7_matches_recorded_pre_law8_digests(probe):
    if wp.config.version != LAW7_DIGESTS_WARP:
        pytest.skip(f"digests recorded on Warp {LAW7_DIGESTS_WARP}, running {wp.config.version}")
    result = _probe(probe, 7)
    assert np.count_nonzero(result["state"]["codes"]) > 0
    assert _digest(result) == LAW7_DIGESTS[probe]


@pytest.fixture(scope="module")
def committed_v4(tmp_path_factory):
    """The tlmpm_v4 module as committed before law 8, loaded from git."""
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{PRE_LAW8_COMMIT}:src/robotic_cutting_sim/tlmpm_v4.py"],
                                capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"committed tlmpm_v4 unavailable: {error}")
    assert "_grid_update_eq8" in source and "_segment_plow" not in source
    path = tmp_path_factory.mktemp("committed_v4_pre_law8")/"tlmpm_v4_pre_law8.py"
    path.write_text(source)
    name = "robotic_cutting_sim._tlmpm_v4_pre_law8"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


def _assert_same_run(make, committed_v4, monkeypatch):
    current = make()
    with monkeypatch.context() as patch:
        patch.setattr(tlmpm_v4, "TLMPM", committed_v4.TLMPM)
        old = make()
    for key in STATE_KEYS:
        assert np.array_equal(current["state"][key], old["state"][key]), key
    assert current["forces"] == old["forces"]
    assert current.get("cutting_state_after_phase") == old.get("cutting_state_after_phase")
    return current


@pytest.mark.parametrize("law", [4, 5, 6, 7])
@pytest.mark.parametrize("probe", ["push", "skim25"])
def test_laws_4_to_7_bitwise_equal_committed_pre_law8_module(probe, law, committed_v4, monkeypatch):
    current = _assert_same_run(lambda: _probe(probe, law), committed_v4, monkeypatch)
    assert np.count_nonzero(current["state"]["codes"]) > 0


@pytest.mark.parametrize("law", [1, 2, 3])
def test_laws_1_2_3_in_v4_bitwise_equal_committed_pre_law8_module(law, committed_v4, monkeypatch):
    current = _assert_same_run(lambda: _probe("push", law, solver_version=4), committed_v4, monkeypatch)
    assert np.count_nonzero(current["state"]["codes"]) > 0


def test_law8_rejected_by_older_solver_versions_and_records_parameters():
    for version in (2, 3):
        with pytest.raises(ValueError):
            run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .001)], cutting_law=8, solver_version=version)
    with pytest.raises(ValueError):
        tlmpm_v4.TLMPM(*_bump_block(), .002, TLMPMConfig(), "cpu", cutting_law=9)
    points, masses = _bump_block()
    bed = tlmpm_v4.TLMPM(points, masses, .002, TLMPMConfig(), "cpu", cutting_law=8)
    assert bed.law8_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX, "theta_plow_deg": 15.}
    assert bed.law5_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX}
    assert bed.law4_parameters["band_h_m"] == 6*DX and bed.law4_parameters["theta_max_deg"] == 45.
    assert bed.cutting_state() == {"cutting_law": 8, "stroke": 0, "travel_m": 0., "engaged": 0, "any_motion": 0}


# ----------------------------------------------------------------------------- plowing mask

def _mask(rotation, velocity, cut=0):
    """seg_ok and seg_plow of one substep for a given edge rotation and linear velocity."""
    points, masses = _bump_block()
    bed = tlmpm_v4.TLMPM(points, masses, .002, TLMPMConfig(), "cpu", cutting_law=8,
                         blade_profile_xz=RULER_KNIFE_V1.tlmpm_profile()[0])
    pose = np.r_[[.2, 0, .1], rotation.as_quat()].astype(np.float32)[None]
    bed.set_trajectory(np.repeat(pose, 2, 0), np.repeat(np.r_[velocity, [0, 0, 0]].astype(np.float32)[None], 2, 0),
                       np.array([cut, cut], np.int32))
    bed.step()
    return bed.segment_ok.numpy().copy(), bed.segment_plow.numpy().copy()


@pytest.mark.parametrize("angle_deg,plowing", [(0., False), (10., False), (14., False), (16., True), (25., True),
                                                (35., True), (44., True)])
def test_law8_plowing_mask_follows_the_15_degree_threshold(angle_deg, plowing):
    # Edge frame = VERTICAL_CUT; motion along the advance direction tilted by angle_deg towards the blade normal.
    advance, normal = VERTICAL_CUT.apply([1, 0, 0]), VERTICAL_CUT.apply([0, 0, 1])
    a = np.deg2rad(angle_deg)
    ok, plow = _mask(VERTICAL_CUT, .05*(np.cos(a)*advance+np.sin(a)*normal))
    assert np.all(ok == 1)
    assert np.all(plow == int(plowing))


def test_law8_plowing_mask_requires_cutting_motion():
    normal = VERTICAL_CUT.apply([0, 0, 1])
    ok, plow = _mask(VERTICAL_CUT, .05*normal)                       # pure blade-normal motion: no M_k
    assert np.all(ok == 0) and np.all(plow == 0)
    advance = VERTICAL_CUT.apply([1, 0, 0])
    ok, plow = _mask(VERTICAL_CUT, .05*(np.cos(.5)*advance+np.sin(.5)*normal), cut=-1)  # disarmed
    assert np.all(ok == 0) and np.all(plow == 0)


# ----------------------------------------------------------------------------- law 8 vs laws 4 and 5

IN_PLANE = {
    "through_cut": [Phase((0, 0, -.05), .32), Phase((0, 0, -.05), .38), Phase((0, 0, 0), .01)],
    "push_withdraw": [Phase((0, 0, -.05), .15), Phase((0, 0, .05), .05)],
}


@pytest.mark.parametrize("name", sorted(IN_PLANE))
def test_law8_bitwise_equals_law4_on_in_plane_cuts(name):
    law4 = run(VERTICAL_CUT, [.001, 0, TOP+.004], IN_PLANE[name], cutting_law=4, record_history=True)
    law8 = run(VERTICAL_CUT, [.001, 0, TOP+.004], IN_PLANE[name], cutting_law=8, record_history=True)
    _assert_bitwise(law4, law8)
    assert np.count_nonzero(law8["state"]["codes"]) > 0
    if name == "through_cut":
        count, labels = stencil_components(law8)
        assert count >= 2 and fractions(labels)[1] > .4


@pytest.fixture(scope="module")
def raked_skims():
    """Laws 5 and 8 raked skims through the bump (law 5's skim), run once per rake."""
    cache = {}

    def get(rake_deg):
        if rake_deg not in cache:
            points, masses = _bump_block()
            runs = {law: run(Rotation.from_euler("y", rake_deg, degrees=True), [-.026, 0, TOP+.0002],
                             [Phase((.05, 0, 0), 1.)], points=points, masses=masses, spacing=.002,
                             cutting_law=law, record_history=True) for law in (5, 8)}
            count, labels = stencil_components(runs[8])
            above = points[:, 2] > TOP
            base_label = np.bincount(labels[~above]).argmax()
            codes = runs[8]["state"]["codes"]
            cache[rake_deg] = {"runs": runs, "count": count,
                               "off_base": float(np.mean(labels[above] != base_label)),
                               "in_base": float(np.mean(labels[~above] == base_label)),
                               "above_code0": int(np.count_nonzero(codes[above] == 0))}
        return cache[rake_deg]
    return get


@pytest.mark.parametrize("rake_deg", [25., 35.])
def test_law8_bitwise_equals_law5_on_constant_rake_skim(rake_deg, raked_skims):
    skim = raked_skims(rake_deg)
    _assert_bitwise(skim["runs"][5], skim["runs"][8])
    print(f"law8 rake {rake_deg}: components={skim['count']} above_off_base={skim['off_base']:.4f} "
          f"below_in_base={skim['in_base']:.4f} above_code0={skim['above_code0']} (bitwise law 5)")


# ----------------------------------------------------------------------------- negative controls

def test_law8_sideways_push_flat_press_and_handle_sever_nothing():
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=8, record_history=True)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=8, record_history=True)
    box = [[[0., 0., .05], [.006, .006, .006]]]
    handle = run(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)], cutting_law=8,
                 obstacle_boxes_edge=box, record_history=True)
    for result in (side, flat, handle):
        assert result["labelled_after_phase"] == [0] and result["severed_bonds"] == 0
        assert stencil_components(result)[0] == 1
        assert result["cutting_state_after_phase"][-1]["stroke"] == 0
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


def test_law8_armed_half_depth_then_withdrawal_keeps_ligament():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)],
                 cutting_law=8, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert result["severed_bonds_after_phase"][1] == result["severed_bonds_after_phase"][0]
    assert stencil_components(result)[0] == 1
    # As law 4: nothing below the stopping depth is labelled (core band included).
    reference, codes = result["points"], result["state"]["codes"]
    below = reference[:, 2] < TOP-.006-1e-4
    core = np.abs(reference[:, 0]-.001) <= DX
    assert np.any(below & core) and np.all(codes[below & core] == 0)
    assert result["cutting_state_after_phase"][1]["stroke"] == 0


def test_law8_armed_stop_adds_no_labels():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)],
                 cutting_law=8, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert stencil_components(result)[0] == 1
    assert result["cutting_state_after_phase"][1] == {"cutting_law": 8, "stroke": 0, "travel_m": 0.,
                                                      "engaged": 0, "any_motion": 0}


def test_law8_two_cut_digits_make_three_pieces():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2, cutting_law=8, record_history=True)
    labelled = result["labelled_after_phase"]
    assert labelled[1] == labelled[0] and labelled[2] == labelled[1] and labelled[3] > labelled[2]
    count, labels = stencil_components(result, cuts=2)
    masses = np.ones(len(labels))
    pieces = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    assert count >= 3 and pieces[2] > .25
