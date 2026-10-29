"""tlmpm_v5 (ADR 0014): packed 2-bit cut digits and sparse per-node logical-field slots.

Coarse engineering checks on Warp's CPU backend (not calibrated forces):
- packed encode/decode and the bitmask conflict test equal the ternary
  ``conflicting`` truth table for K <= 3, and the packed stencil graph equals the
  ternary one;
- the slot-claim kernel puts exactly each node's distinct codes into a
  contiguous prefix of its slots, maps every (particle, corner) to the slot of
  its own code, and counts (never merges) entries beyond S slots;
- law 8 (and laws 4/5) on tlmpm_v5 equal tlmpm_v4 on the law-8 CPU scenarios.
  Tolerance: codes, decoded digits and stencil components must be identical;
  floats must agree within FLOAT_RTOL/FLOAT_ATOL. v5's grid update visits fields
  in v4's (ascending code) order, so on CPU the runs are in fact bitwise equal,
  which test_v5_bitwise_equals_v4_on_aligned_push records separately;
- K = 3 and K = 5 sequential slices give 4 and 6 pieces in the packed stencil
  graph with persistent per-digit history; disarmed lift/move adds no labels;
- three bumps removed by three raked skims (three digits): each bump separates,
  the substrate stays one body;
- every negative control holds under v5 law 8.
"""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from scipy.spatial.transform import Rotation

from robotic_cutting_sim import tlmpm_v5
from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block, drive_blade
from robotic_cutting_sim.cut_topology import (conflicting, conflicting_packed, decode_digits, mechanical_components,
                                              mechanical_components_packed, pack_digits, packed_to_ternary,
                                              ternary_to_packed, validate_packed)
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
DX = TLMPMConfig().dx
FLOAT_KEYS = ("positions", "velocities", "F", "cut_progress", "contact_work_j")
FLOAT_RTOL, FLOAT_ATOL = 1e-5, 1e-7   # stated v4/v5 tolerance (CPU runs are observed bitwise equal)


def run(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, **kwargs)


def same_partition(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return len(np.unique(a)) == len(np.unique(b)) == len(set(zip(a.tolist(), b.tolist())))


def packed_components(result, cuts):
    return mechanical_components_packed(result["reference_stencil_nodes"], result["reference_stencil_weights"],
                                        result["state"]["codes"], cuts, gradients=result["reference_stencil_gradients"])


def ternary_components(result, cuts):
    return mechanical_components(result["reference_stencil_nodes"], result["reference_stencil_weights"],
                                 result["state"]["codes"], cuts, gradients=result["reference_stencil_gradients"])


# ----------------------------------------------------------------------------- encoding (pure NumPy)

@pytest.mark.parametrize("cuts", [1, 2, 3])
def test_packed_conflict_truth_table_equals_ternary(cuts):
    ternary = np.arange(3**cuts)
    packed = ternary_to_packed(ternary, cuts)
    assert np.array_equal(packed_to_ternary(packed, cuts), ternary)
    assert np.array_equal(decode_digits(packed, cuts), (ternary[:, None]//3**np.arange(cuts)) % 3)
    expected = conflicting(ternary[:, None], ternary[None, :], cuts)
    assert np.array_equal(conflicting_packed(packed[:, None], packed[None, :]), expected)
    assert validate_packed(packed, cuts)


def test_packed_digits_round_trip_up_to_15_cuts():
    rng = np.random.default_rng(0)
    digits = rng.integers(0, 3, (500, 15))
    codes = pack_digits(digits)
    assert codes.max() < 2**30 and validate_packed(codes, 15)
    assert np.array_equal(decode_digits(codes, 15), digits)
    assert np.array_equal(ternary_to_packed(packed_to_ternary(codes, 15), 15), codes)
    assert not validate_packed([3], 1) and not validate_packed([4], 1) and not validate_packed([-1], 2)
    with pytest.raises(ValueError):
        pack_digits(np.full((1, 16), 1))
    a, b = pack_digits([[1, 0, 2]]), pack_digits([[0, 0, 1]])
    assert conflicting_packed(a, b)[0] and not conflicting_packed(a, pack_digits([[1, 2, 2]]))[0]


def test_packed_stencil_graph_equals_ternary_graph():
    rng = np.random.default_rng(3)
    nodes = rng.integers(0, 60, (400, 8))
    weights = rng.random((400, 8))*(rng.random((400, 8)) > .3)
    ternary = rng.integers(0, 9, 400)
    count_t, labels_t = mechanical_components(nodes, weights, ternary, 2)
    count_p, labels_p = mechanical_components_packed(nodes, weights, ternary_to_packed(ternary, 2), 2)
    assert count_t == count_p and same_partition(labels_t, labels_p)
    sparse = np.where(rng.random((400, 8)) > .9, weights, 0.)
    count_t, labels_t = mechanical_components(nodes, sparse, ternary, 2)
    count_p, labels_p = mechanical_components_packed(nodes, sparse, ternary_to_packed(ternary, 2), 2)
    assert count_t == count_p > 1 and same_partition(labels_t, labels_p)


# ----------------------------------------------------------------------------- slot claim (Warp CPU)

def _claim(nodes, codes, slots, node_count):
    device = "cpu"
    slot_code = wp.full(node_count*slots, -1, dtype=int, device=device)
    slot = wp.zeros(nodes.shape, dtype=int, device=device)
    overflow, occupied = wp.zeros(1, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)
    wp.launch(tlmpm_v5._claim_slots, nodes.shape, [wp.array(codes, dtype=int, device=device),
              wp.array(nodes, dtype=int, device=device), slot_code, slot, slots, overflow, occupied], device=device)
    return (slot_code.numpy().reshape(node_count, slots), slot.numpy(), int(overflow.numpy()[0]),
            int(occupied.numpy()[0]))


@pytest.mark.cpu_sim
def test_slot_claim_holds_exactly_each_nodes_distinct_codes():
    rng = np.random.default_rng(1)
    node_count = 12
    nodes = rng.integers(0, node_count, (300, 8)).astype(np.int32)
    codes = pack_digits(rng.integers(0, 3, (300, 2))*(rng.random((300, 1)) > .2)).astype(np.int32)  # <= 9 codes
    distinct = [set(codes[np.any(nodes == k, axis=1)].tolist()) for k in range(node_count)]
    slot_code, slot, overflow, occupied = _claim(nodes, codes, 9, node_count)
    assert overflow == 0 and occupied == max(len(d) for d in distinct)
    for k in range(node_count):
        held = slot_code[k][slot_code[k] >= 0]
        assert set(held.tolist()) == distinct[k] and len(held) == len(distinct[k])
        assert np.all(slot_code[k][:len(held)] >= 0) and np.all(slot_code[k][len(held):] == -1)  # contiguous prefix
    rows = np.repeat(np.arange(300), 8).reshape(300, 8)
    assert np.all(slot >= 0) and np.array_equal(slot_code[nodes, slot], codes[rows])


@pytest.mark.cpu_sim
def test_slot_claim_overflow_is_counted_never_merged():
    # One node, five distinct codes, three slots: entries of the 4th/5th codes are dropped and counted.
    codes = np.array([0, 1, 2, 4, 8, 0, 1], np.int32)
    nodes = np.zeros((7, 8), np.int32)
    slot_code, slot, overflow, occupied = _claim(nodes, codes, 3, 1)
    assert occupied == 3 and sorted(slot_code[0].tolist()) == [0, 1, 2]
    assert overflow == 2*8 and np.all(slot[3:5] == -1) and np.all(slot[[0, 1, 2, 5, 6]] >= 0)


@pytest.mark.cpu_sim
def test_solver_surfaces_overflow_as_an_error():
    with pytest.raises(tlmpm_v5.SlotOverflowError):
        run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .15)], cutting_law=8, solver_version=5,
            slots_per_node=1)
    with pytest.raises(tlmpm_v5.SlotOverflowError):  # codes 0, 1 and 2 meet at nodes on the cut plane
        run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .15)], cutting_law=8, solver_version=5,
            slots_per_node=2)
    result = run(VERTICAL_CUT, [.001, 0, TOP+.004], [Phase((0, 0, -.05), .15)], cutting_law=8, solver_version=5,
                 slots_per_node=3)
    assert result["slot_diagnostics"]["max_occupied_slots"] == 3 and result["slot_diagnostics"]["overflow_entries"] == 0


@pytest.mark.cpu_sim
def test_v5_constructor_limits_and_parameters():
    points, masses = block()
    for cuts in (0, 16):
        with pytest.raises(ValueError):
            tlmpm_v5.TLMPM(points, masses, .004, TLMPMConfig(cuts=cuts), "cpu")
    for law in (1, 2, 3, 6, 7):
        with pytest.raises(ValueError):
            tlmpm_v5.TLMPM(points, masses, .004, TLMPMConfig(), "cpu", cutting_law=law)
    bed = tlmpm_v5.TLMPM(points, masses, .004, TLMPMConfig(cuts=15), "cpu", cutting_law=8)
    assert bed.max_cuts == 15 and bed.slots_per_node == 8 and bed.code_encoding == "packed2"
    assert bed.law8_parameters == {"lead_b_lead_m": 2*DX, "core_s_core_m": DX, "theta_plow_deg": 15.}
    assert bed.law4_parameters["band_h_m"] == 6*DX and bed.law4_parameters["theta_max_deg"] == 45.
    with pytest.raises(ValueError):
        bed.set_trajectory(np.zeros((1, 7), np.float32) + [0, 0, 0, 0, 0, 0, 1], np.zeros((1, 6), np.float32), [15])
    assert bed.gm.shape == (bed.node_count*8,)


# ----------------------------------------------------------------------------- v5 vs v4 on the law-8 CPU scenarios

def _bump_block(spacing=.002, density=1500.):
    """60 x 24 x 12 mm bar with a 6 mm dome on top (tests/test_cutting_law_v8.py)."""
    grid = np.mgrid[-.03:.03-1e-9:spacing, -.012:.012-1e-9:spacing, 0:.018-1e-9:spacing]
    points = grid.reshape(3, -1).T+spacing/2
    dome = (points[:, 2] > TOP) & ((points[:, 2]-TOP) <= .006*(1-(points[:, 0]**2+points[:, 1]**2)/.009**2))
    points = points[(points[:, 2] < TOP) | dome].astype(np.float32)
    return points, np.full(len(points), density*spacing**3, np.float32)


BOX = [[[0., 0., .05], [.006, .006, .006]]]
SCENARIOS = {
    "aligned_push": dict(rotation=VERTICAL_CUT, start=[.001, 0, TOP+.004],
                         phases=[Phase((0, 0, -.05), .32), Phase((0, 0, -.05), .38), Phase((0, 0, 0), .01)]),
    "two_digits_three_pieces": dict(rotation=VERTICAL_CUT, start=[-.012, 0, TOP+.004], cuts=2,
                                    phases=[Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0),
                                            Phase((.24, 0, 0), .1, -1), Phase((0, 0, -.1), .2, 1)]),
    "skim25": dict(rotation=Rotation.from_euler("y", 25., degrees=True), start=[-.026, 0, TOP+.0002],
                   phases=[Phase((.05, 0, 0), 1.)], bump=True),
    "side_push": dict(rotation=VERTICAL_CUT, start=[-.034, 0, .002], phases=[Phase((.05, 0, 0), .3)]),
    "flat_press": dict(rotation=FLAT_SKIM, start=[.010, 0, TOP+.002], phases=[Phase((0, 0, -.02), .3)]),
    "blunt_handle": dict(rotation=VERTICAL_CUT, start=[-.05, 0, TOP+.01], phases=[Phase((0, 0, -.05), .2)],
                         obstacle_boxes_edge=BOX),
    "no_contact": dict(rotation=VERTICAL_CUT, start=[.001, 0, TOP+.03], phases=[Phase((.05, 0, 0), .1),
                                                                                Phase((0, 0, -.05), .1)]),
    "armed_stop": dict(rotation=VERTICAL_CUT, start=[.001, 0, TOP+.004],
                       phases=[Phase((0, 0, -.05), .2), Phase((0, 0, 0), .2)]),
    "half_depth_withdrawal": dict(rotation=VERTICAL_CUT, start=[.001, 0, TOP+.004],
                                  phases=[Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)]),
}
_CACHE = {}


def scenario(name, solver, law=8):
    key = (name, solver, law)
    if key not in _CACHE:
        spec = dict(SCENARIOS[name])
        rotation, start, phases = spec.pop("rotation"), spec.pop("start"), spec.pop("phases")
        if spec.pop("bump", False):
            points, masses = _bump_block()
            spec.update(points=points, masses=masses, spacing=.002)
        _CACHE[key] = run(rotation, start, phases, cutting_law=law, record_history=True,
                          solver_version=5 if solver == "v5" else None, **spec)
    return _CACHE[key]


def _without_law(states):
    return [{k: v for k, v in s.items() if k != "cutting_law"} for s in states]


def assert_equivalent(v4, v5, cuts):
    assert np.array_equal(packed_to_ternary(v5["state"]["codes"], cuts), v4["state"]["codes"])
    for key in FLOAT_KEYS:
        np.testing.assert_allclose(v5["state"][key], v4["state"][key], rtol=FLOAT_RTOL, atol=FLOAT_ATOL, err_msg=key)
    np.testing.assert_allclose([f["mean_N"] for f in v5["forces"]], [f["mean_N"] for f in v4["forces"]],
                               rtol=FLOAT_RTOL, atol=1e-6)
    assert v5["labelled_after_phase"] == v4["labelled_after_phase"]
    assert v5["severed_bonds_after_phase"] == v4["severed_bonds_after_phase"]
    assert _without_law(v5["cutting_state_after_phase"]) == _without_law(v4["cutting_state_after_phase"])
    count_4, labels_4 = ternary_components(v4, cuts)
    count_5, labels_5 = packed_components(v5, cuts)
    assert count_4 == count_5 and same_partition(labels_4, labels_5)
    assert v5["slot_diagnostics"]["overflow_entries"] == 0
    return count_5, labels_5


@pytest.mark.cpu_sim
@pytest.mark.parametrize("name", sorted(SCENARIOS))
def test_v5_law8_equals_v4_law8(name):
    cuts = SCENARIOS[name].get("cuts", 1)
    v4, v5 = scenario(name, "v4"), scenario(name, "v5")
    count, labels = assert_equivalent(v4, v5, cuts)
    if name == "aligned_push":
        assert count >= 2 and np.sort(np.bincount(labels))[::-1][1]/len(labels) > .4
    if name == "two_digits_three_pieces":
        assert count >= 3 and np.sort(np.bincount(labels))[::-1][2]/len(labels) > .25
    if name == "skim25":
        points = v5["points"]
        above = points[:, 2] > TOP
        base = np.bincount(labels[~above]).argmax()
        assert np.mean(labels[above] != base) > .9 and np.all(labels[~above] == base)
    print(f"{name}: components={count} max_occupied_slots={v5['slot_diagnostics']['max_occupied_slots']}")


@pytest.mark.cpu_sim
def test_v5_bitwise_equals_v4_on_aligned_push():
    """Stronger observed property (not the ADR requirement): CPU runs are bitwise equal."""
    v4, v5 = scenario("aligned_push", "v4"), scenario("aligned_push", "v5")
    for key in FLOAT_KEYS:
        assert np.array_equal(v4["state"][key], v5["state"][key]), key
    assert v4["forces"] == v5["forces"]


@pytest.mark.cpu_sim
@pytest.mark.parametrize("law", [4, 5])
def test_v5_reference_laws_equal_v4_on_skim(law):
    points, masses = _bump_block()
    kwargs = dict(points=points, masses=masses, spacing=.002, cutting_law=law, record_history=True)
    phases = [Phase((.05, 0, 0), .5)]   # edge reaches and enters the dome (starts 17 mm before it)
    rotation = Rotation.from_euler("y", 25., degrees=True)
    v4 = run(rotation, [-.026, 0, TOP+.0002], phases, **kwargs)
    v5 = run(rotation, [-.026, 0, TOP+.0002], phases, solver_version=5, **kwargs)
    assert np.count_nonzero(v5["state"]["codes"]) > 0
    assert_equivalent(v4, v5, 1)


# ----------------------------------------------------------------------------- negative controls under v5 law 8

@pytest.mark.cpu_sim
def test_v5_law8_negative_controls():
    for name in ("side_push", "flat_press", "blunt_handle", "no_contact"):
        result = scenario(name, "v5")
        assert set(result["labelled_after_phase"]) == {0} and result["severed_bonds"] == 0, name
        assert packed_components(result, 1)[0] == 1, name
        assert result["cutting_state_after_phase"][-1]["stroke"] == 0, name
    assert abs(scenario("side_push", "v5")["forces"][0]["mean_N"][0]) > .1
    assert scenario("flat_press", "v5")["forces"][0]["mean_N"][2] > 1.
    assert scenario("blunt_handle", "v5")["forces"][0]["mean_N"][2] > .5
    assert np.allclose(scenario("no_contact", "v5")["forces"][0]["mean_N"], 0.)

    stop = scenario("armed_stop", "v5")
    first, second = stop["labelled_after_phase"]
    assert first > 0 and second == first and packed_components(stop, 1)[0] == 1
    assert stop["cutting_state_after_phase"][1] == {"cutting_law": 8, "stroke": 0, "travel_m": 0.,
                                                    "engaged": 0, "any_motion": 0}

    withdrawal = scenario("half_depth_withdrawal", "v5")
    first, second = withdrawal["labelled_after_phase"]
    assert first > 0 and second == first
    assert withdrawal["severed_bonds_after_phase"][1] == withdrawal["severed_bonds_after_phase"][0]
    assert packed_components(withdrawal, 1)[0] == 1          # half-depth ligament holds the bar together
    reference, codes = withdrawal["points"], withdrawal["state"]["codes"]
    below = reference[:, 2] < TOP-.006-1e-4
    core = np.abs(reference[:, 0]-.001) <= DX
    assert np.any(below & core) and np.all(codes[below & core] == 0)
    assert withdrawal["cutting_state_after_phase"][1]["stroke"] == 0


# ----------------------------------------------------------------------------- multi-cut (K > 2)

def sequential_slices(xs, length):
    """Armed vertical slice k at x = xs[k] with cut digit k; disarmed lift and move between slices."""
    phases = []
    for k, x in enumerate(xs):
        phases += [Phase((0, 0, -.1), .2, k), Phase((0, 0, .1), .2, -1)]
        if k+1 < len(xs):
            phases.append(Phase(((xs[k+1]-x)/.1, 0, 0), .1, -1))
    points, masses = block(size_m=(length, .024, .012))
    return points, masses, phases


@pytest.mark.cpu_sim
@pytest.mark.parametrize("cuts,xs,length", [(3, (-.016, 0., .016), .072), (5, (-.032, -.016, 0., .016, .032), .096)])
def test_v5_sequential_slices_make_k_plus_one_pieces(cuts, xs, length):
    points, masses, phases = sequential_slices(xs, length)
    result = run(VERTICAL_CUT, [xs[0], 0, TOP+.004], phases, points=points, masses=masses, cuts=cuts, cutting_law=8,
                 solver_version=5, record_history=True)
    count, labels = packed_components(result, cuts)
    pieces = np.sort(np.bincount(labels, weights=masses))[::-1]/masses.sum()
    assert count == cuts+1 and pieces[cuts] > .5/(cuts+1), pieces
    # Each piece is bounded by the declared slices: piece of particle p = number of slices left of it.
    expected = np.searchsorted(np.asarray(xs), points[:, 0])
    assert same_partition(labels, expected)
    digit_counts = np.asarray(result["digit_labelled_after_phase"])
    labelled = result["labelled_after_phase"]
    for k in range(cuts):
        cut_phase = 3*k
        assert digit_counts[cut_phase, k] > 0 and np.all(digit_counts[:cut_phase, k] == 0)
        assert np.all(digit_counts[cut_phase:, k] == digit_counts[cut_phase, k])   # history persists
        for disarmed in range(cut_phase+1, min(cut_phase+3, len(phases))):
            assert labelled[disarmed] == labelled[cut_phase]                      # lift/move adds no labels
    digits = decode_digits(result["state"]["codes"], cuts)
    for k, x in enumerate(xs):
        near = np.abs(points[:, 0]-x) <= 1.5*.004
        assert np.all(digits[near & (points[:, 0] < x), k] == 1) and np.all(digits[near & (points[:, 0] > x), k] == 2)
    diagnostics = result["slot_diagnostics"]
    assert diagnostics["overflow_entries"] == 0 and diagnostics["max_occupied_slots"] <= 8
    print(f"K={cuts}: pieces={np.round(pieces, 3).tolist()} max_occupied_slots={diagnostics['max_occupied_slots']}")


def _three_bump_block(centers, spacing=.002, density=1500.):
    """100 x 24 x 12 mm bar with three 6 mm domes (radius 9 mm) at x = centers."""
    grid = np.mgrid[-.05:.05-1e-9:spacing, -.012:.012-1e-9:spacing, 0:.018-1e-9:spacing]
    points = grid.reshape(3, -1).T+spacing/2
    owner = np.full(len(points), -1)
    for k, cx in enumerate(centers):
        dome = (points[:, 2] > TOP) & ((points[:, 2]-TOP) <= .006*(1-((points[:, 0]-cx)**2+points[:, 1]**2)/.009**2))
        owner[dome] = k
    keep = (points[:, 2] < TOP) | (owner >= 0)
    return points[keep].astype(np.float32), np.full(int(keep.sum()), density*spacing**3, np.float32), owner[keep]


@pytest.mark.cpu_sim
def test_v5_three_bumps_removed_by_three_skims():
    centers = (-.032, 0., .032)
    points, masses, owner = _three_bump_block(centers)
    rotation = Rotation.from_euler("y", 25., degrees=True)
    lift = .02
    phases = []
    for k, cx in enumerate(centers):
        phases.append(Phase((.05, 0, 0), .8, k))                  # armed 25-degree skim over bump k (40 mm)
        if k+1 < len(centers):
            travel = (centers[k+1]-.020)-(cx+.020)
            phases += [Phase((0, 0, lift/.1), .1, -1), Phase((travel/.1, 0, 0), .1, -1),
                       Phase((0, 0, -lift/.1), .1, -1)]
    phases.append(Phase((0, 0, lift/.1), .1, -1))
    result = run(rotation, [centers[0]-.020, 0, TOP+.0002], phases, points=points, masses=masses, spacing=.002,
                 cuts=3, cutting_law=8, solver_version=5, record_history=True)
    count, labels = packed_components(result, 3)
    substrate = points[:, 2] < TOP
    base = np.bincount(labels[substrate]).argmax()
    assert np.all(labels[substrate] == base)                                   # one substrate body
    digits = decode_digits(result["state"]["codes"], 3)
    removed = []
    for k in range(3):
        bump = owner == k
        removed.append(float(np.sum(masses[bump & (labels != base)])/np.sum(masses[bump])))
        assert np.mean(digits[bump, k] == 2) > .9
    assert min(removed) > .9, removed
    labelled = result["labelled_after_phase"]
    for disarmed in (1, 2, 3, 5, 6, 7, 9):
        assert labelled[disarmed] == labelled[disarmed-1]
    assert result["slot_diagnostics"]["overflow_entries"] == 0
    print(f"three bumps: removed={np.round(removed, 4).tolist()} components={count} "
          f"max_occupied_slots={result['slot_diagnostics']['max_occupied_slots']}")


# ----------------------------------------------------------------------------- runtime wiring

def test_live_material_and_kinematic_config_accept_v5_only_explicitly():
    from robotic_cutting_sim.kinematic_env import KinematicConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1 as profile
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=profile, cutting_law=8, max_cuts=3)               # v4 limit unchanged
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=profile, cutting_law=7, max_cuts=3, solver="tlmpm_v5")
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=profile, cutting_law=8, max_cuts=16, solver="tlmpm_v5")
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=profile, cutting_law=8, slots_per_node=4)
    material = LiveCuttingMaterial(tool_profile=profile, cutting_law=8, max_cuts=15, solver="tlmpm_v5")
    assert material.code_encoding == "packed2" and material.slots_per_node == 8
    assert LiveCuttingMaterial(tool_profile=profile, cutting_law=8).code_encoding == "ternary"
    with pytest.raises(ValueError):
        KinematicConfig(max_cuts=3)
    assert KinematicConfig(max_cuts=15, solver="tlmpm_v5").max_cuts == 15


@pytest.mark.cpu_sim
def test_kinematic_env_v5_receipt_and_packed_history():
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    env = make_kinematic_env("CUT-01-R", particle_spacing_m=.004, material_dt=.0005, cutting_law=8, max_cuts=3,
                             solver="tlmpm_v5", slots_per_node=6)
    env.reset(seed=42)
    material = env.backend.material
    receipt = material.parameter_receipt
    assert receipt["solver_module"] == "robotic_cutting_sim.tlmpm_v5" and receipt["code_encoding"] == "packed2"
    assert receipt["max_cuts"] == 3 and receipt["slots_per_node"] == 6 and material.bed.max_cuts == 3
    for _ in range(3):
        env.step()
    assert material.bed.check_slots()["overflow_entries"] == 0
    assert env.backend.receipt()["config"]["solver"] == "tlmpm_v5"
    env.close()
