"""tlmpm_v4 cutting law 5 (ADR 0008) on Warp's CPU backend with a kinematic ruler blade.

Coarse engineering checks of qualitative behavior only (not calibrated forces):
law 4 in the v4 module is unchanged bit for bit by the law-5 addition; law 5
keeps every law-4 negative control (side push, flat press, blunt handle, armed
stop, armed half-depth withdrawal keeps its ligament), makes three pieces with
two digits, detaches the bump of a raked skim at 25 degrees by the actual MPM
stencil graph, and pre-assigns ahead of the edge only particles more than one
reference cell off the swept surface. The 35-degree skim does not detach (strict
xfail, ADR 0008 acceptance failure preserved as evidence).
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
LAW4_COMMIT = "3c1f4b4"  # last commit that changed tlmpm_v4.py before law 5
STATE_KEYS = ("positions", "velocities", "codes", "F", "cut_progress", "contact_work_j")
# SHA-256 of law-4 probe results produced by the committed (pre-law-5) tlmpm_v4
# on Warp 1.17.0 CPU, recorded before law 5 was added.
LAW4_DIGESTS_WARP = "1.17.0"
LAW4_DIGESTS = {
    "push": {"positions": "4e19a0a879a487f108a1d8e2a9eac18f9ce78bf4f1081920ad91143d3c3f943c",
             "velocities": "c7c413183c7e5e423398f914deef6b3bd85d4f8f7806dff2205fd44ac1602bbb",
             "codes": "5e30f03caa89d3deae13e73cab0eaa05b886b98c0929acc741eb6c360d5dfe01",
             "F": "6304ea690222b32d5ef73b9cd54c9aaec8750926bd972ad7df830998c21c65a9",
             "cut_progress": "383fcf659a15dbf6c6f543b8a3307561c801ab920211df744308b9747c096949",
             "contact_work_j": "ca914824d65af237f98f31c555f80fb03bdfae04487f42959aa74cd77fd87b81",
             "forces": "b2c13b4b132c9f02187c22ce550e7d8f725232ad8ef87b0d15b1928ac9995af1"},
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


LAW4_PROBES = {
    "push": lambda: run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .15), Phase((0, 0, .05), .05)],
                        cutting_law=4),
    "skim25": lambda: run(Rotation.from_euler("y", 25., degrees=True), [-.026, 0, TOP+.0002],
                          [Phase((.05, 0, 0), .5)], points=_bump_block()[0], masses=_bump_block()[1],
                          spacing=.002, cutting_law=4),
}


def _digest(result):
    digest = {k: hashlib.sha256(np.ascontiguousarray(result["state"][k]).tobytes()).hexdigest() for k in STATE_KEYS}
    digest["forces"] = hashlib.sha256(json.dumps(result["forces"]).encode()).hexdigest()
    return digest


@pytest.mark.parametrize("probe", sorted(LAW4_PROBES))
def test_law4_matches_recorded_pre_law5_digests(probe):
    if wp.config.version != LAW4_DIGESTS_WARP:
        pytest.skip(f"digests recorded on Warp {LAW4_DIGESTS_WARP}, running {wp.config.version}")
    result = LAW4_PROBES[probe]()
    assert np.count_nonzero(result["state"]["codes"]) > 0
    assert _digest(result) == LAW4_DIGESTS[probe]


@pytest.fixture(scope="module")
def committed_v4(tmp_path_factory):
    """The tlmpm_v4 module as committed before law 5, loaded from git."""
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{LAW4_COMMIT}:src/robotic_cutting_sim/tlmpm_v4.py"],
                                capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"committed tlmpm_v4 unavailable: {error}")
    assert "_assign_lead" not in source
    path = tmp_path_factory.mktemp("committed_v4")/"tlmpm_v4_committed.py"
    path.write_text(source)
    name = "robotic_cutting_sim._tlmpm_v4_pre_law5"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.mark.parametrize("probe", sorted(LAW4_PROBES))
def test_law4_bitwise_equals_committed_pre_law5_module(probe, committed_v4, monkeypatch):
    current = LAW4_PROBES[probe]()
    monkeypatch.setattr(tlmpm_v4, "TLMPM", committed_v4.TLMPM)
    old = LAW4_PROBES[probe]()
    for key in STATE_KEYS:
        assert np.array_equal(current["state"][key], old["state"][key]), key
    assert current["forces"] == old["forces"]
    assert current["cutting_state_after_phase"] == old["cutting_state_after_phase"]
    assert np.count_nonzero(current["state"]["codes"]) > 0


def test_law5_rejected_by_older_solver_versions_and_records_parameters():
    with pytest.raises(ValueError):
        run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .001)], cutting_law=5, solver_version=3)
    points, masses = _bump_block()
    bed = tlmpm_v4.TLMPM(points, masses, .002, TLMPMConfig(), "cpu", cutting_law=5)
    assert bed.law5_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX}
    assert bed.law4_parameters["band_h_m"] == 6*DX
    assert bed.cutting_state()["cutting_law"] == 5 and bed.cutting_state()["stroke"] == 0


class _Recorder:
    """Wraps TLMPM.step (test-only) and records every new label with the substep's
    edge pose/twist and the particle positions the assignment kernels saw."""
    def __init__(self, monkeypatch):
        self.records = []
        base = tlmpm_v4.TLMPM
        records = self.records

        class Recording(base):
            def step(self, advance=True):
                before = self.code.numpy().copy()
                i = int(self.index.numpy()[0])
                super().step(advance)
                after = self.code.numpy()
                new = np.flatnonzero(after != before)
                if new.size:
                    records.append({"pose": self.poses.numpy()[i].copy(), "twist": self.twists.numpy()[i].copy(),
                                    "cut": int(self.cuts.numpy()[i]), "particles": new,
                                    "x": self.x.numpy()[new].copy(), "before": before[new], "after": after[new].copy()})
        monkeypatch.setattr(tlmpm_v4, "TLMPM", Recording)

    def lead_assignments(self, half_width):
        """Recompute local frame and swept-surface side (numpy replica of _swept_side)."""
        local_x, side, digit = [], [], []
        for r in self.records:
            rotation = Rotation.from_quat(r["pose"][3:])  # warp transform: p, q (XYZW)
            local = rotation.inv().apply(r["x"]-r["pose"][:3])
            along, normal = rotation.apply([0, 1, 0]), rotation.apply([0, 0, 1])
            linear, angular = r["twist"][:3].astype(float), r["twist"][3:].astype(float)
            motion = linear[None]+np.cross(angular[None], along[None]*local[:, 1:2])
            swept = np.cross(motion, along[None])
            s = local[:, 2].copy()
            norm = np.linalg.norm(swept, axis=1)
            ok = norm**2 > 1e-20
            swept[ok] /= norm[ok, None]
            swept[ok] *= np.where(swept[ok]@normal < 0, -1., 1.)[:, None]
            s[ok] = np.einsum("ij,ij->i", r["x"][ok]-r["pose"][:3], swept[ok])
            factor = 3 if r["cut"] == 1 else 1
            assert np.all((r["before"]//factor) % 3 == 0)
            local_x.append(local[:, 0]); side.append(s); digit.append((r["after"]//factor) % 3)
            assert np.all(np.abs(local[:, 1]) <= half_width+1e-6)
        if not local_x:
            return np.zeros(0), np.zeros(0), np.zeros(0, int)
        local_x, side, digit = map(np.concatenate, (local_x, side, digit))
        lead = local_x > 1e-7  # band assignment requires f <= 0, lead f > 0
        return local_x[lead], side[lead], digit[lead]


def check_lead_preassignment(recorder):
    half = RULER_KNIFE_V1.tlmpm_profile()[1]
    f, s, digit = recorder.lead_assignments(half)
    if len(f):
        assert f.max() <= 2*DX+1e-6
        assert np.abs(s).min() > DX*(1-1e-4), np.abs(s).min()
        assert np.abs(s).max() <= 6*DX*(1+1e-4)
        assert np.all(digit == np.where(s < 0, 1, 2))
    return len(f), (float(np.abs(s).min()) if len(f) else None)


def test_law5_aligned_push_severs_and_preassigns_outside_core(monkeypatch):
    recorder = _Recorder(monkeypatch)
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .4)], cutting_law=5, record_history=True)
    count, labels = stencil_components(result)
    assert count >= 2 and fractions(labels)[1] > .4
    lead_count, _ = check_lead_preassignment(recorder)
    assert lead_count > 0


def test_law5_sideways_push_flat_press_and_handle_sever_nothing():
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=5, record_history=True)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=5, record_history=True)
    box = [[[0., 0., .05], [.006, .006, .006]]]
    handle = run(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)], cutting_law=5,
                 obstacle_boxes_edge=box, record_history=True)
    for result in (side, flat, handle):
        assert result["labelled_after_phase"] == [0] and result["severed_bonds"] == 0
        assert stencil_components(result)[0] == 1
        assert result["cutting_state_after_phase"][-1]["stroke"] == 0
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


def test_law5_armed_half_depth_then_withdrawal_keeps_ligament(monkeypatch):
    # Edge 4 mm above a 12 mm bar, pushed 10 mm: stops 6 mm deep, then pulled out armed.
    recorder = _Recorder(monkeypatch)
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)],
                 cutting_law=5, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert result["severed_bonds_after_phase"][1] == result["severed_bonds_after_phase"][0]
    assert stencil_components(result)[0] == 1
    # Lead pre-assignment may label off-core material below the stopping depth;
    # the core band (|s| <= dx about the edge plane x = 1 mm) below it stays
    # unassigned and keeps the ligament connected.
    reference, codes = result["points"], result["state"]["codes"]
    below = reference[:, 2] < TOP-.006-1e-4
    core = np.abs(reference[:, 0]-.001) <= DX
    assert np.any(below & core) and np.all(codes[below & core] == 0)
    assert result["cutting_state_after_phase"][1]["stroke"] == 0
    check_lead_preassignment(recorder)


def test_law5_armed_stop_adds_no_labels():
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)],
                 cutting_law=5, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first
    assert stencil_components(result)[0] == 1
    assert result["cutting_state_after_phase"][1] == {"cutting_law": 5, "stroke": 0, "travel_m": 0.,
                                                      "engaged": 0, "any_motion": 0}


def test_law5_two_cut_digits_make_three_pieces():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2, cutting_law=5, record_history=True)
    labelled = result["labelled_after_phase"]
    assert labelled[1] == labelled[0] and labelled[2] == labelled[1] and labelled[3] > labelled[2]
    count, labels = stencil_components(result, cuts=2)
    masses = np.ones(len(labels))
    pieces = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    assert count >= 3 and pieces[2] > .25


@pytest.fixture(scope="module")
def raked_skims():
    """Law-5 raked skims through the bump, run once per rake with label recording."""
    cache = {}

    def get(rake_deg):
        if rake_deg not in cache:
            points, masses = _bump_block()
            with pytest.MonkeyPatch.context() as patch:
                recorder = _Recorder(patch)
                result = run(Rotation.from_euler("y", rake_deg, degrees=True), [-.026, 0, TOP+.0002],
                             [Phase((.05, 0, 0), 1.)], points=points, masses=masses, spacing=.002,
                             cutting_law=5, record_history=True)
            count, labels = stencil_components(result)
            above = points[:, 2] > TOP
            base_label = np.bincount(labels[~above]).argmax()
            codes = result["state"]["codes"]
            cache[rake_deg] = {"recorder": recorder, "count": count,
                               "off_base": float(np.mean(labels[above] != base_label)),
                               "in_base": float(np.mean(labels[~above] == base_label)),
                               "above_side1": float(np.mean(codes[above] == 1)),
                               "above_code0": int(np.count_nonzero(codes[above] == 0)),
                               "below_side2": float(np.mean(codes[~above] == 2))}
        return cache[rake_deg]
    return get


@pytest.mark.parametrize("rake_deg", [25., 35.])
def test_law5_raked_skim_preassigns_only_outside_the_core(rake_deg, raked_skims):
    skim = raked_skims(rake_deg)
    lead_count, min_side = check_lead_preassignment(skim["recorder"])
    print(f"law5 rake {rake_deg}: lead_preassigned={lead_count} min_abs_s_m={min_side}")
    assert lead_count > 0


@pytest.mark.parametrize("rake_deg", [25., pytest.param(35., marks=pytest.mark.xfail(strict=True, reason=(
    "Law 5 known limitation (ADR 0008 CPU acceptance FAIL at 35 deg): the raked edge compresses the "
    "bump bottom ahead of itself into the core band |s| <= dx (final z 12.4-15.7 mm vs edge z 12.2 mm), "
    "where lead pre-assignment does not apply; 44 above-plane particles carried 1-9 mm ahead of the edge "
    "stay code 0 and keep one stencil component (law 4 leaves 76).")))])
def test_law5_raked_skim_detaches_the_bump_by_stencil_graph(rake_deg, raked_skims):
    skim = raked_skims(rake_deg)
    # Side-1 mislabel fraction is reported, not asserted (ADR 0008 bounds only the graph split).
    print(f"law5 rake {rake_deg}: components={skim['count']} above_off_base={skim['off_base']:.4f} "
          f"below_in_base={skim['in_base']:.4f} above_side1={skim['above_side1']:.4f} "
          f"above_code0={skim['above_code0']} below_side2={skim['below_side2']:.4f}")
    assert skim["count"] >= 2
    assert skim["off_base"] >= .9 and skim["in_base"] >= .99
