"""tlmpm_v4 cutting law 6 (ADR 0010) on Warp's CPU backend with a kinematic ruler blade.

Coarse engineering checks of qualitative behavior only (not calibrated forces):
laws 4 and 5 in the v4 module are unchanged bit for bit by the law-6 addition
(against the git-loaded pre-law-6 module and digests recorded before law 6);
instrumented per substep, law 6 never labels a particle ahead of the edge while
the stroke is active, records pending decisions only under law 5's lead
condition, keeps pending decisions on the active digit only, and commits exactly
the still-unassigned pending particles when the stroke ends. A through cut is
bitwise equal to law 4 until the first commit that labels anything. Law 6 keeps
every law-4/5 negative control (side push, flat press, blunt handle, armed stop,
armed half-depth withdrawal keeps its ligament), makes three pieces with two
digits, and the raked skims on the bump block are reported (25 degrees asserted
to separate as under law 5; 35 degrees does not separate: strict xfail, as law 5).
"""
import hashlib
import importlib.util
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
PRE_LAW6_COMMIT = "ebd7395"  # last commit that changed tlmpm_v4.py before law 6 (law 5)
STATE_KEYS = ("positions", "velocities", "codes", "F", "cut_progress", "contact_work_j")
HALF = RULER_KNIFE_V1.tlmpm_profile()[1]
# SHA-256 of law-5 probe results produced by the committed (pre-law-6) tlmpm_v4
# (ebd7395, unchanged at 51c7e7b) on Warp 1.17.0 CPU, recorded before law 6 was added.
LAW5_DIGESTS_WARP = "1.17.0"
LAW5_DIGESTS = {
    "push": {"positions": "917ac889f80ba05c498381e5ef80d0497831b3ab99f9d1955e74beafc29c6249",
             "velocities": "e96672382669a7ee4a4bef72a0aea65b3130c3acca3dc4aeb7e451dfec8f48cb",
             "codes": "d93c4505d743093a4b3ef1780df33c243d198bc15f5e29fda4552e26aa65a5b9",
             "F": "dd93c211d4f0b72ebe9fabceac5e4c52a7fab2365aea4c46229bf8e135ab3e28",
             "cut_progress": "383fcf659a15dbf6c6f543b8a3307561c801ab920211df744308b9747c096949",
             "contact_work_j": "c88847d96253308488dea49b940d45609f715840e53357b834eb359d0c1b3268",
             "forces": "758dbc26cbfb953efecfa8697b44dfee085fc7e4c331cc4fe5fb111794a872c4"},
    "skim25": {"positions": "b0b8126813cada445de364e949e565b54a23e040c211a16b6017d606d7b43311",
               "velocities": "f9f633a30b2670c20d741b6dd5c8fd2af2eeeec9c3eb89cd90a4dd7d0e9474f6",
               "codes": "bbe6bc66fb0c84a0d74560b4bbb4802f5a5d6a341337c29dffafafcc176f4cd8",
               "F": "fb9c661e6dd2e951ad38a715c10a4c6b7c868dbcf967d02368ebfe6f9d420604",
               "cut_progress": "6eca28832d47aa9bb6908ed21f3a506d0a49f80345c9b4a6c835dfe9558712de",
               "contact_work_j": "bc9fcc509508254d8cdbb489ea6606caa0ec5976252511ee596a6107e67d4336",
               "forces": "b239fbc96aa865c501d82c628cef70a9fe99516a3ebbd79302220158e4f71245"}}


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


def _probe(name, law):
    if name == "push":
        return run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .15), Phase((0, 0, .05), .05)],
                   cutting_law=law)
    points, masses = _bump_block()
    return run(Rotation.from_euler("y", 25., degrees=True), [-.026, 0, TOP+.0002], [Phase((.05, 0, 0), .5)],
               points=points, masses=masses, spacing=.002, cutting_law=law)


def _digest(result):
    import json
    digest = {k: hashlib.sha256(np.ascontiguousarray(result["state"][k]).tobytes()).hexdigest() for k in STATE_KEYS}
    digest["forces"] = hashlib.sha256(json.dumps(result["forces"]).encode()).hexdigest()
    return digest


# ----------------------------------------------------------------------------- laws 4/5 unchanged

@pytest.mark.parametrize("probe", ["push", "skim25"])
def test_law5_matches_recorded_pre_law6_digests(probe):
    if wp.config.version != LAW5_DIGESTS_WARP:
        pytest.skip(f"digests recorded on Warp {LAW5_DIGESTS_WARP}, running {wp.config.version}")
    result = _probe(probe, 5)
    assert np.count_nonzero(result["state"]["codes"]) > 0
    assert _digest(result) == LAW5_DIGESTS[probe]


@pytest.fixture(scope="module")
def committed_v4(tmp_path_factory):
    """The tlmpm_v4 module as committed before law 6, loaded from git."""
    try:
        source = subprocess.run(["git", "-C", str(ROOT), "show", f"{PRE_LAW6_COMMIT}:src/robotic_cutting_sim/tlmpm_v4.py"],
                                capture_output=True, check=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        pytest.skip(f"committed tlmpm_v4 unavailable: {error}")
    assert "_assign_lead" in source and "_commit_pending" not in source
    path = tmp_path_factory.mktemp("committed_v4_pre_law6")/"tlmpm_v4_pre_law6.py"
    path.write_text(source)
    name = "robotic_cutting_sim._tlmpm_v4_pre_law6"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(name, None)


@pytest.mark.parametrize("law", [4, 5])
@pytest.mark.parametrize("probe", ["push", "skim25"])
def test_laws_4_5_bitwise_equal_committed_pre_law6_module(probe, law, committed_v4, monkeypatch):
    current = _probe(probe, law)
    monkeypatch.setattr(tlmpm_v4, "TLMPM", committed_v4.TLMPM)
    old = _probe(probe, law)
    for key in STATE_KEYS:
        assert np.array_equal(current["state"][key], old["state"][key]), key
    assert current["forces"] == old["forces"]
    assert current["cutting_state_after_phase"] == old["cutting_state_after_phase"]
    assert np.count_nonzero(current["state"]["codes"]) > 0


def test_law6_rejected_by_older_solver_versions_and_records_parameters():
    for version in (2, 3):
        with pytest.raises(ValueError):
            run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .001)], cutting_law=6, solver_version=version)
    with pytest.raises(ValueError):
        tlmpm_v4.TLMPM(*_bump_block(), .002, TLMPMConfig(), "cpu", cutting_law=8)
    points, masses = _bump_block()
    bed = tlmpm_v4.TLMPM(points, masses, .002, TLMPMConfig(), "cpu", cutting_law=6)
    assert bed.law6_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX, "lead_commit": "stroke_end"}
    assert bed.law5_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX}
    assert bed.law4_parameters["band_h_m"] == 6*DX
    assert bed.cutting_state() == {"cutting_law": 6, "stroke": 0, "travel_m": 0., "engaged": 0, "any_motion": 0,
                                   "pending_count": 0, "pending_unassigned_count": 0, "committed_count": 0,
                                   "ended_cut": -1}
    assert np.all(bed.pending_cut.numpy() == -1) and np.all(bed.pending_side.numpy() == 0)


# ----------------------------------------------------------------------------- per-substep instrumentation

def _local_and_side(pose, twist, x):
    """Edge-frame coordinates and swept-surface side (numpy replica of _swept_side)."""
    rotation = Rotation.from_quat(pose[3:])  # warp transform: p, q (XYZW)
    local = rotation.inv().apply(x-pose[:3])
    along, normal = rotation.apply([0, 1, 0]), rotation.apply([0, 0, 1])
    linear, angular = twist[:3].astype(float), twist[3:].astype(float)
    motion = linear[None]+np.cross(angular[None], along[None]*local[:, 1:2])
    swept = np.cross(motion, along[None])
    s = local[:, 2].copy()
    norm = np.linalg.norm(swept, axis=1)
    ok = norm**2 > 1e-20
    swept[ok] /= norm[ok, None]
    swept[ok] *= np.where(swept[ok]@normal < 0, -1., 1.)[:, None]
    s[ok] = np.einsum("ij,ij->i", x[ok]-pose[:3], swept[ok])
    return local, s


class _Law6Recorder:
    """Wraps TLMPM.step (test-only) and checks law 6 at every substep.

    Per substep: new labels are classified as commit labels (the digit whose
    stroke ended in this substep) or band labels; band labels must lie behind
    the edge (f <= 0); commit labels must be exactly the particles that held a
    pending decision for that digit with the digit still 0, with the pending
    side; pending writes must satisfy law 5's lead condition; and afterwards
    every pending decision belongs to the active stroke's digit.
    """
    def __init__(self, monkeypatch, hashes=False):
        self.substeps = 0
        self.ahead_band_labels = 0      # labels with f > 0 outside a commit (must stay 0)
        self.band_labels = 0
        self.commit_substeps = []       # (substep, ended cut, labels committed, pending cleared)
        self.pending_writes = 0
        self.pending_min_abs_s = None
        self.max_pending = 0
        self.hashes = [] if hashes else None
        recorder = self

        class Recording(tlmpm_v4.TLMPM):
            def step(self, advance=True):
                recorder.before(self)
                super().step(advance)
                recorder.after(self)
        monkeypatch.setattr(tlmpm_v4, "TLMPM", Recording)

    def before(self, bed):
        self._i = int(bed.index.numpy()[0])
        self._code = bed.code.numpy().copy()
        self._stroke = int(bed.stroke.numpy()[0])
        if bed.cutting_law == 6:
            self._side = bed.pending_side.numpy().copy()
            self._cut = bed.pending_cut.numpy().copy()

    def after(self, bed):
        k = self.substeps
        self.substeps += 1
        code = bed.code.numpy()
        if self.hashes is not None:
            self.hashes.append((hashlib.sha256(bed.x.numpy().tobytes()).hexdigest(),
                                hashlib.sha256(code.tobytes()).hexdigest(),
                                int(bed.pending_committed.numpy()[0]) if bed.cutting_law == 6 else 0))
        if bed.cutting_law != 6:
            return
        i = self._i
        pose, twist = bed.poses.numpy()[i].astype(float), bed.twists.numpy()[i].astype(float)
        cut = int(bed.cuts.numpy()[i])
        stroke = int(bed.stroke.numpy()[0])
        ended = int(bed.stroke_ended_cut.numpy()[0])
        commit = self._stroke > 0 and stroke != self._stroke
        assert ended == (self._stroke-1 if commit else -1), (k, self._stroke, stroke, ended)
        x = bed.x.numpy().astype(float)
        changed = np.flatnonzero(code != self._code)
        committed_here = 0
        if commit:
            factor = 3 if ended == 1 else 1
            old_digit = (self._code//factor) % 3
            new_digit = (code//factor) % 3
            expected = (self._cut == ended) & (old_digit == 0)
            got = old_digit != new_digit
            assert np.array_equal(got, expected), (k, np.flatnonzero(got ^ expected))
            assert np.array_equal(new_digit[expected], self._side[expected])
            assert np.all(np.isin(self._side[expected], (1, 2)))
            committed_here = int(expected.sum())
            self.commit_substeps.append((k, ended, committed_here, int(np.count_nonzero(self._cut == ended))))
            band = changed[~got[changed]]
        else:
            band = changed
        if band.size:
            # Band labels: only the active stroke's digit, only behind the edge.
            assert stroke > 0 and cut == stroke-1
            factor = 3 if cut == 1 else 1
            assert np.all((self._code[band]//factor) % 3 == 0)
            local, s = _local_and_side(pose, twist, x[band])
            assert np.all(np.abs(local[:, 1]) <= HALF+1e-6)
            self.ahead_band_labels += int(np.count_nonzero(local[:, 0] > 1e-7))
            self.band_labels += int(band.size)
        side, pcut = bed.pending_side.numpy(), bed.pending_cut.numpy()
        # Invariant: pending decisions only for the active stroke's digit, cleared otherwise.
        assert np.all((pcut == -1) | (pcut == stroke-1)), (k, stroke, np.unique(pcut))
        assert np.all((pcut >= 0) == (side > 0))
        written = np.flatnonzero((pcut != self._cut) | (side != self._side))
        written = written[pcut[written] >= 0]
        if written.size:
            # Pending writes: exactly law 5's lead condition (digit 0, 0 < f <= 2 dx, dx < |s| <= 6 dx, side).
            assert not commit or ended != pcut[written[0]]
            factor = 3 if cut == 1 else 1
            assert stroke == cut+1 and np.all(pcut[written] == cut)
            assert np.all((code[written]//factor) % 3 == 0)
            local, s = _local_and_side(pose, twist, x[written])
            assert np.all(np.abs(local[:, 1]) <= HALF+1e-6)
            assert np.all(local[:, 0] > -1e-7) and np.all(local[:, 0] <= 2*DX+1e-6)
            assert np.abs(s).min() > DX*(1-1e-4) and np.abs(s).max() <= 6*DX*(1+1e-4)
            assert np.all(side[written] == np.where(s < 0, 1, 2))
            self.pending_writes += int(written.size)
            low = float(np.abs(s).min())
            self.pending_min_abs_s = low if self.pending_min_abs_s is None else min(self.pending_min_abs_s, low)
        self.max_pending = max(self.max_pending, int(np.count_nonzero(pcut >= 0)))

    @property
    def committed(self):
        return sum(c[2] for c in self.commit_substeps)

    def summary(self):
        return (f"substeps={self.substeps} band_labels={self.band_labels} ahead_band_labels={self.ahead_band_labels} "
                f"pending_writes={self.pending_writes} max_pending={self.max_pending} "
                f"min_abs_s_m={self.pending_min_abs_s} commits={self.commit_substeps}")


def test_law6_aligned_push_defers_lead_labels_then_commits_at_stroke_end(monkeypatch):
    recorder = _Law6Recorder(monkeypatch)
    # Push 20 mm (through the 12 mm bar onto the table), then stop: the stop ends the stroke.
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .4), Phase((0, 0, 0), .01)],
                 cutting_law=6, record_history=True)
    print("law6 aligned push:", recorder.summary())
    assert recorder.ahead_band_labels == 0 and recorder.band_labels > 0
    assert recorder.pending_writes > 0 and recorder.max_pending > 0
    assert recorder.pending_min_abs_s > DX*(1-1e-4)
    # The stroke ends (at the latest at the stop) and the final state holds no pending decision.
    assert recorder.commit_substeps
    final = result["cutting_state_after_phase"][-1]
    assert final["stroke"] == 0 and final["pending_count"] == 0 and final["pending_unassigned_count"] == 0
    assert final["committed_count"] == recorder.committed
    count, labels = stencil_components(result)
    assert count >= 2 and fractions(labels)[1] > .4


def _hashing_run(monkeypatch, law, phases):
    with monkeypatch.context() as patch:
        recorder = _Law6Recorder(patch, hashes=True)
        result = run(VERTICAL_CUT, [.001, 0, TOP+.004], phases, cutting_law=law, record_history=True)
    return recorder, result


def test_law6_through_cut_bitwise_equals_law4_until_first_labelling_commit(monkeypatch):
    # The law-4 through-cut (tests/test_cutting_law_v4.py): the edge leaves the bar bottom
    # and the stroke ends in air; then a short stop.
    phases = [Phase((0, 0, -.05), .32), Phase((0, 0, -.05), .38), Phase((0, 0, 0), .01)]
    law4, result4 = _hashing_run(monkeypatch, 4, phases)
    law6, result6 = _hashing_run(monkeypatch, 6, phases)
    assert law4.substeps == law6.substeps
    labelling = [k for k, (_, _, committed) in enumerate(law6.hashes) if committed > 0]
    first = labelling[0] if labelling else None
    print(f"law6 through cut: first labelling commit substep={first} of {law6.substeps}; {law6.summary()}")
    assert law6.ahead_band_labels == 0 and law6.pending_writes > 0 and law6.commit_substeps
    end = law6.substeps if first is None else first
    for k in range(end):
        assert law4.hashes[k][:2] == law6.hashes[k][:2], k
    if first is None:
        # Nothing was pending at stroke end: the whole run equals law 4 bit for bit.
        for key in STATE_KEYS:
            assert np.array_equal(result4["state"][key], result6["state"][key]), key
        assert result4["forces"] == result6["forces"]
    else:
        # The commit substep itself still moves particles as law 4 (labels act from the next substep).
        assert law4.hashes[first][0] == law6.hashes[first][0]
    final = result6["cutting_state_after_phase"][-1]
    assert final["stroke"] == 0 and final["pending_count"] == 0
    count, labels = stencil_components(result6)
    assert count >= 2 and fractions(labels)[1] > .4


# ----------------------------------------------------------------------------- negative controls

def test_law6_sideways_push_flat_press_and_handle_sever_nothing():
    side = run(VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], cutting_law=6, record_history=True)
    flat = run(FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], cutting_law=6, record_history=True)
    box = [[[0., 0., .05], [.006, .006, .006]]]
    handle = run(VERTICAL_CUT, [-.05, 0, TOP+.01], [Phase((0, 0, -.05), .2)], cutting_law=6,
                 obstacle_boxes_edge=box, record_history=True)
    for result in (side, flat, handle):
        assert result["labelled_after_phase"] == [0] and result["severed_bonds"] == 0
        assert stencil_components(result)[0] == 1
        state = result["cutting_state_after_phase"][-1]
        assert state["stroke"] == 0 and state["pending_count"] == 0 and state["committed_count"] == 0
    assert abs(side["forces"][0]["mean_N"][0]) > .1 and flat["forces"][0]["mean_N"][2] > 1.
    assert handle["forces"][0]["mean_N"][2] > .5


def test_law6_armed_half_depth_then_withdrawal_keeps_ligament(monkeypatch):
    # Edge 4 mm above a 12 mm bar, pushed 10 mm: stops 6 mm deep, then pulled out armed.
    recorder = _Law6Recorder(monkeypatch)
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)],
                 cutting_law=6, record_history=True)
    print("law6 half depth + withdrawal:", recorder.summary())
    first, second = result["labelled_after_phase"]
    assert first > 0 and recorder.ahead_band_labels == 0
    # Withdrawal ends the stroke in its first substep; the only labels it adds are the commit.
    assert second == first+recorder.committed
    assert len(recorder.commit_substeps) == 1 and recorder.commit_substeps[0][0] == 2000
    assert stencil_components(result)[0] == 1
    # Pending off-core labels below the stopping depth are committed at withdrawal; the
    # core band (|s| <= dx about the edge plane x = 1 mm) below it stays code 0.
    reference, codes = result["points"], result["state"]["codes"]
    below = reference[:, 2] < TOP-.006-1e-4
    core = np.abs(reference[:, 0]-.001) <= DX
    assert np.any(below & core) and np.all(codes[below & core] == 0)
    state = result["cutting_state_after_phase"][1]
    assert state["stroke"] == 0 and state["pending_count"] == 0


def test_law6_armed_stop_commits_only_pending_and_keeps_one_component(monkeypatch):
    recorder = _Law6Recorder(monkeypatch)
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)],
                 cutting_law=6, record_history=True)
    first, second = result["labelled_after_phase"]
    assert first > 0 and second == first+recorder.committed
    assert len(recorder.commit_substeps) == 1 and recorder.commit_substeps[0][0] == 2000
    assert stencil_components(result)[0] == 1
    assert result["cutting_state_after_phase"][1] == {
        "cutting_law": 6, "stroke": 0, "travel_m": 0., "engaged": 0, "any_motion": 0, "pending_count": 0,
        "pending_unassigned_count": 0, "committed_count": recorder.committed, "ended_cut": -1}


def test_law6_two_cut_digits_make_three_pieces(monkeypatch):
    recorder = _Law6Recorder(monkeypatch)
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0),
              Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]
    result = run(VERTICAL_CUT, [-.012, 0, TOP+.004], phases, cuts=2, cutting_law=6, record_history=True)
    print("law6 two digits:", recorder.summary())
    labelled = result["labelled_after_phase"]
    assert labelled[1] >= labelled[0] and labelled[2] == labelled[1] and labelled[3] > labelled[2]
    assert recorder.ahead_band_labels == 0
    assert {cut for _, cut, _, _ in recorder.commit_substeps} >= {0}
    count, labels = stencil_components(result, cuts=2)
    masses = np.ones(len(labels))
    pieces = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    assert count >= 3 and pieces[2] > .25


def test_law6_digit_change_and_disarm_in_one_substep_commit_the_old_digit(monkeypatch):
    """Edge cases: an engaged stroke of digit 0 switches to digit 1 in one substep
    (stroke 1 -> 2 directly), then disarms in one substep (2 -> 0). Each ending
    commits only its own digit, and no decision of the old digit survives."""
    recorder = _Law6Recorder(monkeypatch)
    phases = [Phase((0, 0, -.05), .1, 0), Phase((0, 0, -.05), .1, 1), Phase((0, 0, -.05), .05, -1)]
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], phases, cuts=2, cutting_law=6, record_history=True)
    print("law6 digit change / disarm:", recorder.summary())
    ends = [(k, cut) for k, cut, _, _ in recorder.commit_substeps]
    assert (1000, 0) in ends and (2000, 1) in ends
    assert recorder.ahead_band_labels == 0
    states = result["cutting_state_after_phase"]
    assert states[0]["stroke"] == 1 and states[1]["stroke"] == 2 and states[2]["stroke"] == 0
    assert states[2]["pending_count"] == 0


# ----------------------------------------------------------------------------- raked skims

@pytest.fixture(scope="module")
def raked_skims():
    """Law-6 raked skims through the bump (law 5's skim plus a 10 ms stop), run once per rake."""
    cache = {}

    def get(rake_deg):
        if rake_deg not in cache:
            points, masses = _bump_block()
            with pytest.MonkeyPatch.context() as patch:
                recorder = _Law6Recorder(patch)
                result = run(Rotation.from_euler("y", rake_deg, degrees=True), [-.026, 0, TOP+.0002],
                             [Phase((.05, 0, 0), 1.), Phase((0, 0, 0), .01)], points=points, masses=masses,
                             spacing=.002, cutting_law=6, record_history=True)
            count, labels = stencil_components(result)
            above = points[:, 2] > TOP
            base_label = np.bincount(labels[~above]).argmax()
            codes = result["state"]["codes"]
            cache[rake_deg] = {"recorder": recorder, "count": count, "result": result,
                               "off_base": float(np.mean(labels[above] != base_label)),
                               "in_base": float(np.mean(labels[~above] == base_label)),
                               "above_side1": float(np.mean(codes[above] == 1)),
                               "above_code0": int(np.count_nonzero(codes[above] == 0)),
                               "below_side2": float(np.mean(codes[~above] == 2))}
        return cache[rake_deg]
    return get


@pytest.mark.parametrize("rake_deg", [25., 35.])
def test_law6_raked_skim_defers_lead_labels(rake_deg, raked_skims):
    skim = raked_skims(rake_deg)
    recorder = skim["recorder"]
    states = skim["result"]["cutting_state_after_phase"]
    print(f"law6 rake {rake_deg}: {recorder.summary()} state_after_skim={states[0]} final={states[-1]}")
    assert recorder.ahead_band_labels == 0 and recorder.pending_writes > 0 and recorder.commit_substeps
    assert states[-1]["stroke"] == 0 and states[-1]["pending_count"] == 0


@pytest.mark.parametrize("rake_deg", [25., pytest.param(35., marks=pytest.mark.xfail(strict=True, reason=(
    "Law 6 does not separate the 35-degree bump (as law 5, ADR 0008 CPU acceptance FAIL preserved): after the "
    "stroke-end commit (296 labels) 32 above-plane particles remain code 0 and the stencil graph is one "
    "component (law 5 leaves 44, law 4 76). Recorded 2026-10-29 on Warp 1.17.0 CPU.")))])
def test_law6_raked_skim_bump_separation_by_stencil_graph(rake_deg, raked_skims):
    skim = raked_skims(rake_deg)
    print(f"law6 rake {rake_deg}: components={skim['count']} above_off_base={skim['off_base']:.4f} "
          f"below_in_base={skim['in_base']:.4f} above_side1={skim['above_side1']:.4f} "
          f"above_code0={skim['above_code0']} below_side2={skim['below_side2']:.4f} "
          f"committed={skim['recorder'].committed}")
    assert skim["count"] >= 2
    assert skim["off_base"] >= .9 and skim["in_base"] >= .99
