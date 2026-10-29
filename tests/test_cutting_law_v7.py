"""tlmpm_v4 cutting law 7 (ADR 0011) on Warp's CPU backend with a kinematic ruler blade.

Coarse engineering checks of qualitative behavior only (not calibrated forces):
laws 1-6 in the v4 module are unchanged bit for bit by the law-7 addition
(against the git-loaded pre-law-7 module and law-6 digests recorded before law
7); the law-7 grid-update kernel _grid_update_eq8 on a tiny synthetic node set
merges two compatible fields exactly like one field for both the FLIP reference
(old) and the new velocity, keeps an incompatible pair separate, gives the
pairwise Eq. (8) result for three fields, conserves momentum, and leaves new_v
bitwise equal to _grid_update's; a law-7 run equals law 5 bit for bit until the
first label exists. Law 7 keeps every law-5 negative control (side push, flat
press, blunt handle, armed stop, armed half-depth withdrawal keeps its code-0
core ligament), makes three pieces with two digits, and the raked skims on the
bump block are reported (25 degrees asserted to separate; 35 degrees does not
separate: strict xfail, as laws 5 and 6).
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
from robotic_cutting_sim.cut_topology import compatibility, mechanical_components
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
DX = TLMPMConfig().dx
ROOT = Path(__file__).resolve().parents[1]
PRE_LAW7_COMMIT = "9f57e41"  # last commit that changed tlmpm_v4.py before law 7 (law 6; unchanged at e5ef839)
STATE_KEYS = ("positions", "velocities", "codes", "F", "cut_progress", "contact_work_j")
# SHA-256 of law-6 probe results produced by the committed (pre-law-7) tlmpm_v4
# (9f57e41, unchanged at e5ef839) on Warp 1.17.0 CPU, recorded before law 7 was added.
# (skim25 has no stroke end, so it equals the law-4 skim25 digests of tests/test_cutting_law_v5.py.)
LAW6_DIGESTS_WARP = "1.17.0"
LAW6_DIGESTS = {
    "push": {"positions": "596b6745cb4db90946ae9790ff06364d91d68e7221db396a01df21ad7ac38bac",
             "velocities": "03a47463ab2d0df3b391dcd5d505483c7a41c3cc74260d18366fb5e6c2ca4b69",
             "codes": "d93c4505d743093a4b3ef1780df33c243d198bc15f5e29fda4552e26aa65a5b9",
             "F": "63555d2550ef3aa093852429ffbff68fda51c78735f378c854f6e3b4faa01c57",
             "cut_progress": "383fcf659a15dbf6c6f543b8a3307561c801ab920211df744308b9747c096949",
             "contact_work_j": "ca914824d65af237f98f31c555f80fb03bdfae04487f42959aa74cd77fd87b81",
             "forces": "e7a87b003798284d6d70f9e5be38f01bdc4414d3dd5885a2a747cea15518c226"},
    "skim25": {"positions": "74285038c9ab85c8045cf7c4e40b43c5bfa4e2d9a9efb59c86c48f3e1c2c6eb0",
               "velocities": "5d9006716450ce1175bcfc0917aabbb9da5bacbd7ad068be91f11f7b0ffc7123",
               "codes": "e40bd3f1ea3d9823f939766bcf20c562e7c8f15024d67583088d7a4d2c15bd98",
               "F": "8c415110535f4444b977a3e6b90a4ca099f4dd4e0ba999c28c3cc1f277c02470",
               "cut_progress": "68696496106c26bb92b3f7797acac98bdb46c1c3c10732f03ba32b5420f68a65",
               "contact_work_j": "5ebec0aa158b3d86ce624f7977ae89ee84cc17c1ef75f2edf52de07c24ce5e07",
               "forces": "c5ffca0e7d1f9827f006e5ef9df42b8bafc495df0456af2aa1891b57274d85ab"}}


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


# ----------------------------------------------------------------------------- laws 1-6 unchanged

@pytest.mark.parametrize("probe", ["push", "skim25"])
def test_law6_matches_recorded_pre_law7_digests(probe):
    if wp.config.version != LAW6_DIGESTS_WARP:
        pytest.skip(f"digests recorded on Warp {LAW6_DIGESTS_WARP}, running {wp.config.version}")
    result = _probe(probe, 6)
    assert np.count_nonzero(result["state"]["codes"]) > 0
    assert _digest(result) == LAW6_DIGESTS[probe]


@pytest.fixture(scope="module")
def committed_v4(tmp_path_factory):
    """The tlmpm_v4 module as committed before law 7, loaded from git."""
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{PRE_LAW7_COMMIT}:src/robotic_cutting_sim/tlmpm_v4.py"],
                                capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"committed tlmpm_v4 unavailable: {error}")
    assert "_commit_pending" in source and "_grid_update_eq8" not in source
    path = tmp_path_factory.mktemp("committed_v4_pre_law7")/"tlmpm_v4_pre_law7.py"
    path.write_text(source)
    name = "robotic_cutting_sim._tlmpm_v4_pre_law7"
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


@pytest.mark.parametrize("law", [4, 5, 6])
@pytest.mark.parametrize("probe", ["push", "skim25"])
def test_laws_4_5_6_bitwise_equal_committed_pre_law7_module(probe, law, committed_v4, monkeypatch):
    current = _assert_same_run(lambda: _probe(probe, law), committed_v4, monkeypatch)
    assert np.count_nonzero(current["state"]["codes"]) > 0


@pytest.mark.parametrize("law", [1, 2, 3])
def test_laws_1_2_3_in_v4_bitwise_equal_committed_pre_law7_module(law, committed_v4, monkeypatch):
    current = _assert_same_run(lambda: _probe("push", law, solver_version=4), committed_v4, monkeypatch)
    assert np.count_nonzero(current["state"]["codes"]) > 0


def test_law7_rejected_by_older_solver_versions_and_records_parameters():
    for version in (2, 3):
        with pytest.raises(ValueError):
            run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .001)], cutting_law=7, solver_version=version)
    with pytest.raises(ValueError):
        tlmpm_v4.TLMPM(*_bump_block(), .002, TLMPMConfig(), "cpu", cutting_law=9)
    points, masses = _bump_block()
    bed = tlmpm_v4.TLMPM(points, masses, .002, TLMPMConfig(), "cpu", cutting_law=7)
    assert bed.law7_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX, "flip_reference": "eq8_coupled_pn"}
    assert bed.law5_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX}
    assert bed.law4_parameters["band_h_m"] == 6*DX
    assert bed.cutting_state() == {"cutting_law": 7, "stroke": 0, "travel_m": 0., "engaged": 0, "any_motion": 0}


# ----------------------------------------------------------------------------- grid-update kernel

def _grid(cuts, fields, seed=0):
    """Synthetic per-node fields {node: {code: mass}} with random p^n and dp (float32)."""
    rng = np.random.default_rng(seed)
    codes, nodes = 3**cuts, len(fields)
    gm = np.zeros(codes*nodes, np.float32)
    gp = np.zeros((codes*nodes, 3), np.float32)
    gj = np.zeros((codes*nodes, 3), np.float32)
    for node, entries in enumerate(fields):
        for code, mass in entries.items():
            i = code*nodes+node
            gm[i] = mass
            gp[i] = mass*rng.normal(size=3)
            gj[i] = mass*.2*rng.normal(size=3)
    return gm, gp, gj


def _launch(kernel, cuts, gm, gp, gj):
    codes, nodes = 3**cuts, len(gm)//3**cuts
    with wp.ScopedDevice("cpu"):
        old, new_v = wp.zeros(len(gm), dtype=wp.vec3), wp.zeros(len(gm), dtype=wp.vec3)
        wp.launch(kernel, (codes, nodes), [wp.array(gm, dtype=float), wp.array(gp, dtype=wp.vec3),
                  wp.array(gj, dtype=wp.vec3), old, new_v,
                  wp.array(compatibility(cuts).astype(np.int32), dtype=int), nodes, codes])
        return old.numpy().astype(float), new_v.numpy().astype(float)


def _eq8(m, s, compatible):
    """Reference Eq. (8) exchange in float64: S_a + sum_b compatible (m_a S_b - m_b S_a) / M."""
    total = m.sum()
    out = s.copy()
    for a in range(len(m)):
        if m[a] > 0:
            for b in range(len(m)):
                if b != a and compatible[a, b]:
                    out[a] += (m[a]*s[b]-m[b]*s[a])/total
    return out


CUTS1_FIELDS = [{0: .3, 1: .7},          # compatible pair (uncut + side 1)
                {1: .4, 2: .5},          # incompatible pair (an actual cut)
                {0: .2, 1: .3, 2: .5},   # three fields {0, 1, 2}
                {2: .6},                 # single field
                {}]                      # empty node
CUTS2_FIELDS = [{1: .25, 3: .45},        # digit-0 side 1 + digit-1 side 1: compatible
                {1: .3, 2: .6},          # digit-0 sides 1 and 2: incompatible
                {0: .1, 4: .2, 8: .3}]   # 0 couples with both; (1,1) vs (2,2) conflict


@pytest.mark.parametrize("cuts,fields", [(1, CUTS1_FIELDS), (2, CUTS2_FIELDS)], ids=["one_cut", "two_cuts"])
def test_grid_update_eq8_matches_pairwise_eq8_and_conserves_momentum(cuts, fields):
    gm, gp, gj = _grid(cuts, fields)
    old, new_v = _launch(tlmpm_v4._grid_update_eq8, cuts, gm, gp, gj)
    old_ref, new_ref = _launch(tlmpm_v4._grid_update, cuts, gm, gp, gj)
    # The coupled p + dp (new_v) is computed exactly as by _grid_update; only old changes.
    assert np.array_equal(new_v, new_ref)
    compatible = compatibility(cuts)
    nodes = len(fields)
    for node, entries in enumerate(fields):
        index = np.arange(3**cuts)*nodes+node
        m = gm[index].astype(float)
        present = m > 0
        assert np.all(old[index[~present]] == 0) and np.all(new_v[index[~present]] == 0)
        if not present.any():
            continue
        coupled_p = _eq8(m, gp[index].astype(float), compatible)
        coupled_raw = _eq8(m, (gp[index]+gj[index]).astype(float), compatible)
        np.testing.assert_allclose(old[index[present]], coupled_p[present]/m[present, None], rtol=2e-5, atol=1e-6)
        np.testing.assert_allclose(new_v[index[present]], coupled_raw[present]/m[present, None], rtol=2e-5, atol=1e-6)
        # Momentum conservation: the exchange is antisymmetric.
        np.testing.assert_allclose((m[:, None]*old[index]).sum(0), gp[index].astype(float).sum(0), rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose((m[:, None]*new_v[index]).sum(0), (gp[index]+gj[index]).astype(float).sum(0),
                                   rtol=1e-5, atol=1e-6)
        # The pre-law-7 FLIP reference is the uncoupled p_c/m_c.
        np.testing.assert_allclose(old_ref[index[present]], gp[index[present]]/m[present, None], rtol=1e-6)


def test_grid_update_eq8_two_compatible_fields_equal_one_merged_field():
    for cuts, fields, pair in ((1, CUTS1_FIELDS, (0, 1)), (2, CUTS2_FIELDS, (1, 3))):
        gm, gp, gj = _grid(cuts, fields, seed=3)
        old, new_v = _launch(tlmpm_v4._grid_update_eq8, cuts, gm, gp, gj)
        nodes = len(fields)
        a, b = (c*nodes for c in pair)  # node 0
        merged_mass = float(gm[a])+float(gm[b])
        merged_old = (gp[a].astype(float)+gp[b])/merged_mass
        merged_new = (gp[a].astype(float)+gj[a]+gp[b]+gj[b])/merged_mass
        for i in (a, b):
            np.testing.assert_allclose(old[i], merged_old, rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(new_v[i], merged_new, rtol=1e-5, atol=1e-7)
        # Merged single-field run through the same kernel (both masses as one uncut field).
        single_gm, single_gp, single_gj = (np.zeros_like(x) for x in (gm, gp, gj))
        single_gm[0], single_gp[0], single_gj[0] = gm[a]+gm[b], gp[a]+gp[b], gj[a]+gj[b]
        single_old, single_new = _launch(tlmpm_v4._grid_update_eq8, cuts, single_gm, single_gp, single_gj)
        for i in (a, b):
            np.testing.assert_allclose(old[i], single_old[0], rtol=1e-5, atol=1e-7)
            np.testing.assert_allclose(new_v[i], single_new[0], rtol=1e-5, atol=1e-7)
        # Node 1: an incompatible pair stays separate (each its own p/m).
        for code in (1, 2):
            i = code*nodes+1
            np.testing.assert_allclose(old[i], gp[i]/gm[i], rtol=1e-6)
            np.testing.assert_allclose(new_v[i], (gp[i]+gj[i])/gm[i], rtol=1e-6)


def test_grid_update_eq8_musl_pass_and_common_velocity():
    # MUSL launch: gj is zero and gp is the remapped momentum, so new_v = old = coupled p / m,
    # and new_v is bitwise _grid_update's.
    gm, gp, _ = _grid(1, CUTS1_FIELDS, seed=5)
    zero = np.zeros_like(gp)
    old, new_v = _launch(tlmpm_v4._grid_update_eq8, 1, gm, gp, zero)
    _, new_ref = _launch(tlmpm_v4._grid_update, 1, gm, gp, zero)
    assert np.array_equal(new_v, new_ref) and np.array_equal(old, new_v)
    # A common velocity in all fields at a node is preserved exactly by the exchange (to rounding).
    velocity = np.array([.3, -.1, .05], np.float32)
    common = (gm[:, None]*velocity).astype(np.float32)
    old, new_v = _launch(tlmpm_v4._grid_update_eq8, 1, gm, common, zero)
    present = gm > 0
    np.testing.assert_allclose(old[present], np.broadcast_to(velocity, old[present].shape), rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(new_v[present], old[present], rtol=0, atol=0)


# ----------------------------------------------------------------------------- whole-solver behaviour

def _hashing_run(monkeypatch, law, phases):
    hashes = []

    class Recording(tlmpm_v4.TLMPM):
        def step(self, advance=True):
            super().step(advance)
            hashes.append((hashlib.sha256(self.x.numpy().tobytes()).hexdigest(),
                           hashlib.sha256(self.code.numpy().tobytes()).hexdigest(),
                           int(np.count_nonzero(self.code.numpy()))))
    with monkeypatch.context() as patch:
        patch.setattr(tlmpm_v4, "TLMPM", Recording)
        result = run(VERTICAL_CUT, [.001, 0, TOP+.004], phases, cutting_law=law, record_history=True)
    return hashes, result


def test_law7_equals_law5_until_first_label_then_differs(monkeypatch):
    phases = [Phase((0, 0, -.05), .15)]
    law5, result5 = _hashing_run(monkeypatch, 5, phases)
    law7, result7 = _hashing_run(monkeypatch, 7, phases)
    assert len(law5) == len(law7)
    first = next(k for k, (_, _, count) in enumerate(law5) if count > 0)
    # Labels act from the next substep: until then every node holds one field, and
    # _grid_update_eq8 reduces to _grid_update exactly.
    for k in range(first+1):
        assert law5[k] == law7[k], k
    assert not np.array_equal(result5["state"]["positions"], result7["state"]["positions"])
    print(f"law7 vs law5: identical through substep {first} (first label), of {len(law5)}; final labels "
          f"law5={law5[-1][2]} law7={law7[-1][2]}")


def test_law7_aligned_push_severs():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .4)], cutting_law=7, record_history=True)
    count, labels = stencil_components(result)
    assert count >= 2 and fractions(labels)[1] > .4


def test_law7_sideways_push_flat_press_and_handle_sever_nothing():
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=7, record_history=True)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=7, record_history=True)
    box = [[[0., 0., .05], [.006, .006, .006]]]
    handle = run(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)], cutting_law=7,
                 obstacle_boxes_edge=box, record_history=True)
    for result in (side, flat, handle):
        assert result["labelled_after_phase"] == [0] and result["severed_bonds"] == 0
        assert stencil_components(result)[0] == 1
        assert result["cutting_state_after_phase"][-1]["stroke"] == 0
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


def test_law7_armed_half_depth_then_withdrawal_keeps_ligament():
    # Edge 4 mm above a 12 mm bar, pushed 10 mm: stops 6 mm deep, then pulled out armed.
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)],
                 cutting_law=7, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert result["severed_bonds_after_phase"][1] == result["severed_bonds_after_phase"][0]
    assert stencil_components(result)[0] == 1
    # The core band (|s| <= dx about the edge plane x = 1 mm) below the stop stays code 0.
    reference, codes = result["points"], result["state"]["codes"]
    below = reference[:, 2] < TOP-.006-1e-4
    core = np.abs(reference[:, 0]-.001) <= DX
    assert np.any(below & core) and np.all(codes[below & core] == 0)
    assert result["cutting_state_after_phase"][1]["stroke"] == 0


def test_law7_armed_stop_adds_no_labels():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)],
                 cutting_law=7, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert stencil_components(result)[0] == 1
    assert result["cutting_state_after_phase"][1] == {"cutting_law": 7, "stroke": 0, "travel_m": 0.,
                                                      "engaged": 0, "any_motion": 0}


def test_law7_two_cut_digits_make_three_pieces():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2, cutting_law=7, record_history=True)
    labelled = result["labelled_after_phase"]
    assert labelled[1] == labelled[0] and labelled[2] == labelled[1] and labelled[3] > labelled[2]
    count, labels = stencil_components(result, cuts=2)
    masses = np.ones(len(labels))
    pieces = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    assert count >= 3 and pieces[2] > .25


# ----------------------------------------------------------------------------- raked skims

@pytest.fixture(scope="module")
def raked_skims():
    """Law-7 raked skims through the bump (the law-5 skim), run once per rake."""
    cache = {}

    def get(rake_deg):
        if rake_deg not in cache:
            points, masses = _bump_block()
            result = run(Rotation.from_euler("y", rake_deg, degrees=True), [-.026, 0, TOP+.0002],
                         [Phase((.05, 0, 0), 1.)], points=points, masses=masses, spacing=.002,
                         cutting_law=7, record_history=True)
            count, labels = stencil_components(result)
            above = points[:, 2] > TOP
            base_label = np.bincount(labels[~above]).argmax()
            codes = result["state"]["codes"]
            cache[rake_deg] = {"count": count,
                               "off_base": float(np.mean(labels[above] != base_label)),
                               "in_base": float(np.mean(labels[~above] == base_label)),
                               "above_side1": float(np.mean(codes[above] == 1)),
                               "above_code0": int(np.count_nonzero(codes[above] == 0)),
                               "below_side2": float(np.mean(codes[~above] == 2))}
        return cache[rake_deg]
    return get


@pytest.mark.parametrize("rake_deg", [25., pytest.param(35., marks=pytest.mark.xfail(strict=True, reason=(
    "Law 7 does not separate the 35-degree bump on this coarse CPU block (as laws 5 and 6): the stencil graph "
    "is one component and 62 above-plane particles stay code 0 (law 5 leaves 44, law 6 32, law 4 76); "
    "25 degrees separates (2 components, 95.8% above-plane off base). Recorded 2026-10-29 on Warp 1.17.0 CPU.")))])
def test_law7_raked_skim_bump_separation_by_stencil_graph(rake_deg, raked_skims):
    skim = raked_skims(rake_deg)
    print(f"law7 rake {rake_deg}: components={skim['count']} above_off_base={skim['off_base']:.4f} "
          f"below_in_base={skim['in_base']:.4f} above_side1={skim['above_side1']:.4f} "
          f"above_code0={skim['above_code0']} below_side2={skim['below_side2']:.4f}")
    assert skim["count"] >= 2
    assert skim["off_base"] >= .9 and skim["in_base"] >= .99
