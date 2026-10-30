"""cut_events_v1 (WP-12 Phase 3, tech review §4 R1): read-only Hermite cut-event recorder for tlmpm_v5.

Coarse CPU checks (Warp device "cpu", 4 mm blocks of 270 particles):
- physics neutrality: a kinematic law-8 cut with the recorder on equals the run with it off bit for bit
  (every snapshot array, forces, label counts, stroke state), through blade_probe and LiveCuttingMaterial;
- the Warp recorder equals the pure-NumPy reference substep by substep (same link set and flags; t, s,
  points and normals within float32 tolerance) on a planar and an oblique/drawing cut;
- every logged link conflicts in its cut in the final codes and every conflicting link is logged once;
- planar vertical cut with the swept plane midway between lattice columns: reference Hermite points lie
  within 0.05 particle spacings (P95) of the plane x = x0 (measured P95 0.034 s at 4 mm). With the plane
  1 mm / 3 mm from the neighbouring columns the same statistic is 0.21 s (measured, not asserted): the
  material is displaced by the blade before the band assigns it, and P is the material point that is on
  the swept plane at the assignment instant, not the undeformed plane;
- an exact geometric oracle with undeformed material (x = X, F = I) on an oblique swept plane;
- negative controls (disarmed, pure normal press, side push, in-plane slide without contact): no events;
- K = 2 sequential cuts write cut ids 0 and 1; overflow raises; the default-off path launches nothing new.
"""
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from robotic_cutting_sim import cut_events as ce
from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase, block, drive_blade
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
SPACING = .004
X0 = .002          # midway between the lattice columns x = 0 and x = 4 mm of block()
T_ATOL = 2e-5      # float32 recorder vs float64 reference
LENGTH_ATOL = 2e-8  # m
NORMAL_ATOL = 2e-5


def probe(rotation, start, phases, **kwargs):
    return drive_blade(RULER_KNIFE_V1, rotation, start, phases, cutting_law=8, solver_version=5, **kwargs)


def make_bed(points, masses, cuts=1, spacing=SPACING):
    from robotic_cutting_sim.tlmpm_v5 import TLMPM
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    config = TLMPMConfig(cuts=cuts, blade_half_width=float(half), cutting_energy=5.)
    return TLMPM(points, masses, spacing, config, "cpu", blade_profile_xz=polygon, cutting_law=8), config


def lattice_of(points, spacing=SPACING):
    return np.round((points-points.min(0))/spacing).astype(int)


def manual_run(rotation, start, phases, *, cuts=1):
    """Step tlmpm_v5 like drive_blade (200-substep chunks) with the recorder and the NumPy reference."""
    points, masses = block()
    bed, config = make_bed(points, masses, cuts)
    recorder = ce.CutEventRecorder(bed, lattice_of(points))
    quaternion, position = rotation.as_quat(), np.asarray(start, float)
    expected, substep, new_counts, extra = [], 0, [], {"F": []}
    for phase in phases:
        velocity = np.asarray(phase.velocity_m_s, float)
        steps = int(round(phase.duration_s/config.dt))
        for first in range(0, steps, 200):
            count = min(200, steps-first)
            poses = np.array([np.r_[position+velocity*config.dt*i, quaternion] for i in range(count+1)], np.float32)
            twists = np.tile(np.r_[velocity, 0., 0., 0.], (count+1, 1)).astype(np.float32)
            bed.set_trajectory(poses, twists, np.full(count+1, phase.cut, np.int32))
            for i in range(count):
                previous = bed.code.numpy().copy()
                recorder.before_step()
                bed.step()
                recorder.after_step()
                codes = bed.code.numpy()
                if not np.array_equal(codes, previous):
                    events, new = ce.reference_substep_events(previous, codes, bed.x.numpy(), bed.F.numpy(),
                        recorder.reference_host, recorder.neighbours_host, poses[i], twists[i], phase.cut,
                        substep, cuts)
                    expected.append(events)
                    new_counts.append((substep, new))
                    for event in events:   # F-mapped plane residual needs x and F at the assignment instant
                        p, q = event["p"], event["q"]
                        fbar = .5*(bed.F.numpy()[p].astype(float)+bed.F.numpy()[q].astype(float))
                        n = event["normal_current"].astype(float)
                        s_p = (bed.x.numpy()[p]-event["edge_origin"])@n
                        X = recorder.reference_host
                        extra["F"].append(abs(s_p+n@fbar@(event["hermite_reference"]-X[p]))
                                          / np.linalg.norm(fbar.T@n)/SPACING)
                substep += 1
            position = position+velocity*config.dt*count
    recorder.check()
    reference = ce.sort_log(np.concatenate(expected)) if expected else np.zeros(0, ce.LOG_DTYPE)
    return bed, recorder, reference, new_counts, np.asarray(extra["F"]), points


def assert_matches_reference(log, reference):
    assert len(log) == len(reference) > 0
    for key in ("p", "q", "cut", "substep", "flags"):
        assert np.array_equal(log[key], reference[key]), key
    ok = (log["flags"] & ce.FLAG_SIGN_INCONSISTENT) == 0
    assert np.array_equal(np.isnan(log["t"]), ~ok)
    np.testing.assert_allclose(log["t"][ok], reference["t"][ok], rtol=0, atol=T_ATOL)
    for key in ("s_p", "s_q", "hermite_reference", "edge_origin"):
        np.testing.assert_allclose(log[key], reference[key], rtol=0, atol=LENGTH_ATOL, err_msg=key)
    for key in ("normal_current", "normal_reference"):
        np.testing.assert_allclose(log[key], reference[key], rtol=0, atol=NORMAL_ATOL, err_msg=key)


def assert_final_conflicts(log, codes, neighbours, cuts):
    for k in range(cuts):
        assert ce.log_links(log, k) == ce.links_conflicting_in_cut(codes, neighbours, k), k
    keys = np.minimum(log["p"], log["q"]).astype(np.int64)*len(codes)+np.maximum(log["p"], log["q"])
    assert len(np.unique(np.c_[keys, log["cut"]], axis=0)) == len(log)   # each link once per cut


# ----------------------------------------------------------------------------- lattice table

def test_lattice_neighbours_match_lattice_bonds():
    from robotic_cutting_sim.law8_validation import lattice_bonds
    points, _ = block()
    lattice = lattice_of(points)
    rng = np.random.default_rng(3)
    keep = np.sort(rng.choice(len(points), 200, replace=False))      # holes: absent neighbours are -1
    table = ce.lattice_neighbours(lattice[keep])
    a, b = lattice_bonds(lattice[keep])
    ours = {(min(p, q), max(p, q)) for p in range(len(keep)) for q in table[p] if q >= 0}
    assert ours == {(min(x, y), max(x, y)) for x, y in zip(np.asarray(a).tolist(), np.asarray(b).tolist())}
    for p in range(len(keep)):
        for j, q in enumerate(table[p]):
            if q >= 0:
                assert np.array_equal(lattice[keep][q]-lattice[keep][p], ce.NEIGHBOUR_OFFSETS[j])
    with pytest.raises(ValueError):
        ce.lattice_neighbours(np.zeros((2, 3), int))


# ----------------------------------------------------------------------------- neutrality

ALIGNED = dict(rotation=VERTICAL_CUT, start=[X0, 0, TOP+.004],
               phases=[Phase((0, 0, -.05), .32), Phase((0, 0, -.05), .38), Phase((0, 0, 0), .01)])


@pytest.mark.cpu_sim
def test_recorder_is_physics_neutral_bitwise_on_cpu():
    off = probe(**ALIGNED, record_history=True)
    on = probe(**ALIGNED, record_history=True, cut_events="cut_events_v1")
    assert "cut_events" not in off and "cut_events" in on
    assert set(off["state"]) == set(on["state"])
    for key in off["state"]:
        assert off["state"][key].tobytes() == on["state"][key].tobytes(), key
    assert off["forces"] == on["forces"] and off["force_history_N"][0].tobytes() == on["force_history_N"][0].tobytes()
    for key in ("labelled_after_phase", "severed_bonds_after_phase", "cutting_state_after_phase", "components",
                "digit_labelled_after_phase", "slot_diagnostics", "mass_fractions"):
        assert off[key] == on[key], key
    metadata, log = on["cut_events"]["metadata"], on["cut_events"]["log"]
    assert metadata["version"] == "cut_events_v1" and metadata["enabled"] and metadata["event_overflow"] == 0
    assert metadata["substeps"] == 7100 and len(on["cut_events"]["activation"]) == 7100
    assert metadata["events_logged"] == len(log) > 0 and metadata["unexpected_code_changes"] == 0
    assert metadata["log_sha256"] == ce._sha(log)
    assert off["components"] >= 2   # a through-cut (so the neutrality covers labelling and separation)
    assert_final_conflicts(log, on["state"]["codes"], on["cut_events"]["neighbours"], 1)


@pytest.mark.cpu_sim
def test_live_material_path_is_neutral_and_launches_nothing_new_when_off(monkeypatch):
    """LiveCuttingMaterial on CUT-01-R (4 mm, material_dt 0.5 ms): the edge descends through the bar top."""
    from robotic_cutting_sim.kinematic_env import KinematicConfig, KinematicCuttingBackend
    from robotic_cutting_sim import tlmpm_v5
    counts = {"inside": 0, "outside": 0, "copy": 0}
    state = {"in_step": False}
    launch, copy, step = wp.launch, wp.copy, tlmpm_v5.TLMPM.step

    def counting_launch(*args, **kwargs):
        counts["inside" if state["in_step"] else "outside"] += 1
        return launch(*args, **kwargs)

    def counting_copy(*args, **kwargs):
        counts["copy"] += 1
        return copy(*args, **kwargs)

    def counting_step(self, *args, **kwargs):
        state["in_step"] = True
        try:
            return step(self, *args, **kwargs)
        finally:
            state["in_step"] = False

    monkeypatch.setattr(wp, "launch", counting_launch)
    monkeypatch.setattr(wp, "copy", counting_copy)
    monkeypatch.setattr(tlmpm_v5.TLMPM, "step", counting_step)
    results = {}
    for option in (None, "cut_events_v1"):
        backend = KinematicCuttingBackend(KinematicConfig("CUT-01-R", particle_spacing_m=.004, material_dt=.0005,
                                                          cutting_law=8, solver="tlmpm_v5", cut_events=option))
        backend.reset(seed=42)
        material, task = backend.material, backend.task
        start = 3.2   # the scripted edge is ~2 mm into the 20 mm bar; labels start within 0.2 s
        material.reset(task.reference_edge_pose_world(start), seed=42)
        for key in counts:
            counts[key] = 0
        diagnostics = []
        for k in range(1, 21):
            diagnostics.append(material.advance(task.reference_edge_pose_world(start+k*task.control_dt_s),
                                                round(k*task.control_dt_s, 9), cutting=True, cut_index=0))
        results[option] = (material, dict(counts), diagnostics, backend.receipt())
    off, on = results[None], results["cut_events_v1"]
    substeps = off[0].substeps*20
    # Off: every kernel launch of advance() is inside TLMPM.step; the copies are advance()'s own pose assigns.
    assert off[1]["outside"] == 0 and off[0].cut_event_recorder is None
    assert on[1]["inside"] == off[1]["inside"] and on[1]["outside"] == 2*substeps
    assert on[1]["copy"] == off[1]["copy"]+substeps
    assert all("cut_events" not in d for d in off[2]) and all("cut_events" in d for d in on[2])
    assert "cut_events" not in off[3]["config"] and on[3]["config"]["cut_events"] == "cut_events_v1"
    assert off[0].cut_events_metadata() == {"version": "cut_events_v1", "enabled": False}
    a, b = off[0].debug_state(), on[0].debug_state()
    for key in a:
        assert a[key].tobytes() == b[key].tobytes(), key
    for x, y in zip(off[2], on[2]):
        assert np.array_equal(x["current_material_reaction_impulse_world"], y["current_material_reaction_impulse_world"])
    metadata = on[0].cut_events_metadata()
    assert metadata["substeps"] == substeps and metadata["events_logged"] > 0 and metadata["event_overflow"] == 0
    log = on[0].cut_event_recorder.log()
    assert_final_conflicts(log, b["codes"], on[0].cut_event_recorder.neighbours_host, 1)
    print(f"live path: {metadata['events_logged']} events in {substeps} substeps, labelled "
          f"{np.count_nonzero(b['codes'])}")


# ----------------------------------------------------------------------------- recorder vs NumPy reference

@pytest.mark.cpu_sim
def test_planar_cut_matches_reference_and_lies_on_the_plane():
    bed, recorder, reference, new_counts, residual, points = manual_run(**ALIGNED)
    log, activation = recorder.log(), recorder.activation()
    assert_matches_reference(log, reference)
    assert_final_conflicts(log, bed.code.numpy(), recorder.neighbours_host, 1)
    for substep, new in new_counts:
        assert activation["newly_assigned"][substep] == new
        assert activation["stroke"][substep] == 1 and activation["cut"][substep] == 0
    assert np.array_equal(activation["substep"], np.arange(len(activation)))
    assert activation["links_logged"].sum() == len(log)
    assert np.count_nonzero(activation["newly_assigned"]) == len(new_counts)
    assert recorder.counters()["unexpected_code_changes"] == 0
    assert np.all((log["flags"] & ce.FLAG_SIGN_INCONSISTENT) == 0) and np.all((log["t"] >= 0) & (log["t"] < 1))
    assert np.allclose(log["normal_current"], [1, 0, 0], atol=1e-6)   # cross(-z, y) = +x
    # Hermite points against the swept plane x = X0 in the reference configuration.
    offset = np.abs(log["hermite_reference"][:, 0]-X0)/SPACING
    p95 = float(np.percentile(offset, 95))
    f95 = float(np.percentile(residual, 95))
    angle = np.degrees(np.arccos(np.clip(log["normal_reference"]@[1., 0, 0], -1, 1)))
    print(f"planar: {len(log)} links, reference-plane offset P50/P95/max = "
          f"{np.percentile(offset, 50):.4f}/{p95:.4f}/{offset.max():.4f} spacings; "
          f"F-mapped plane residual P95 {f95:.4f} spacings; N_ref angle P95 {np.percentile(angle, 95):.3f} deg")
    assert p95 <= .05
    assert f95 <= .05
    # Reference Hermite points stay inside the material's reference box and on their link.
    P, X = log["hermite_reference"], recorder.reference_host
    lo, hi = np.minimum(X[log["p"]], X[log["q"]]), np.maximum(X[log["p"]], X[log["q"]])
    assert np.all(P >= lo-1e-7) and np.all(P <= hi+1e-7)


@pytest.mark.cpu_sim
def test_oblique_and_drawing_cut_matches_reference():
    """Off-plane drift (21.8 deg, plowing: law-8 lead rule) then a pure draw (degenerate swept normal)."""
    phases = [Phase((.02, 0, -.05), .3), Phase((0, .05, 0), .15), Phase((.02, 0, -.05), .25)]
    bed, recorder, reference, _, _, _ = manual_run(VERTICAL_CUT, [X0-.004, 0, TOP+.004], phases)
    log = recorder.log()
    assert_matches_reference(log, reference)
    assert_final_conflicts(log, bed.code.numpy(), recorder.neighbours_host, 1)
    tilted = (log["flags"] & ce.FLAG_BLADE_Z_FALLBACK) == 0
    expected = np.array([.05, 0, .02])/np.linalg.norm([.05, 0, .02])
    assert np.allclose(log["normal_current"][tilted], expected, atol=1e-6)
    metadata = recorder.metadata()
    print({k: metadata[k] for k in ("events_logged", "lead_links", "both_new_links", "blade_z_fallback_links",
                                    "sign_inconsistent_links")})
    # At 4 mm spacing lead-assigned particles (|s| > s_core = 4 mm) have no opposite-digit 6-neighbour when
    # assigned, so their links are logged later from the band endpoint (lead_links is 0 here, not asserted).
    assert metadata["unexpected_code_changes"] == 0 and metadata["sign_inconsistent_links"] == 0


@pytest.mark.parametrize("motion", ["oblique", "stopped"])
def test_undeformed_plane_oracle_is_exact(motion):
    """Codes written by the exact swept-side rule on undeformed material (x = X, F = I): P on the plane.

    ``stopped`` (zero twist) exercises the blade-z fallback of _swept_side. (A pure draw along the edge is not
    a fallback in float32 at generic orientations: m x e is rounding noise above the 1e-20 threshold.)"""
    from scipy.spatial.transform import Rotation
    points, masses = block()
    bed, _ = make_bed(points, masses)
    recorder = ce.CutEventRecorder(bed, lattice_of(points))
    rotation = Rotation.from_euler("zy", [20., 90.], degrees=True)
    origin = np.array([X0+.0013, -.001, .006])
    velocity = rotation.apply([1., 0, 0])*.05 + rotation.apply([0, 0, 1.])*.01   # in-plane and off-plane motion
    if motion == "stopped":
        velocity = np.zeros(3)
    pose = np.r_[origin, rotation.as_quat()].astype(np.float32)
    twist = np.r_[velocity, 0, 0, 0].astype(np.float32)
    bed.set_trajectory(np.stack([pose, pose]), np.stack([twist, twist]), np.zeros(2, np.int32))
    along = rotation.apply([0, 1., 0])
    n = np.cross(velocity, along) if motion == "oblique" else rotation.apply([0, 0, 1.])
    n /= np.linalg.norm(n)
    n = n if n@rotation.apply([0, 0, 1.]) >= 0 else -n
    side = (points-origin)@n
    recorder.before_step()
    bed.code.assign(np.where(np.abs(side) <= .01, np.where(side < 0, 1, 2), 0).astype(np.int32))
    bed.index.fill_(1)   # as after step(advance=True): the assignment used index 0
    recorder.after_step()
    log = recorder.log()
    assert len(log) >= 18 and np.all((log["flags"] & ce.FLAG_SIGN_INCONSISTENT) == 0)
    assert np.all(((log["flags"] & ce.FLAG_BLADE_Z_FALLBACK) != 0) == (motion == "stopped"))
    distance = np.abs((log["hermite_reference"]-origin)@n)/SPACING
    assert distance.max() <= 1e-5, distance.max()
    assert np.allclose(log["normal_reference"], n, atol=1e-6) and np.allclose(log["normal_current"], n, atol=1e-6)
    assert ce.log_links(log) == ce.links_conflicting_in_cut(bed.code.numpy(), recorder.neighbours_host, 0)
    reference, new = ce.reference_substep_events(np.zeros(bed.n, int), bed.code.numpy(), points, bed.F.numpy(),
                                                 points, recorder.neighbours_host, pose, twist, 0, 0, 1)
    assert_matches_reference(log, reference)
    assert recorder.activation()["newly_assigned"][0] == new
    recorder.reset()
    assert recorder.counters()["events_logged"] == 0 and len(recorder.activation()) == 0


# ----------------------------------------------------------------------------- negative controls

@pytest.mark.cpu_sim
def test_no_events_without_a_stroke():
    controls = {
        "disarmed": (VERTICAL_CUT, [X0, 0, TOP+.004], [Phase((0, 0, -.05), .4, -1)], {}),
        "flat_press": (FLAT_SKIM, [.010, 0, TOP+.002], [Phase((0, 0, -.02), .3)], {}),
        "side_push": (VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)], {}),
        "slide_no_contact": (VERTICAL_CUT, [X0, 0, TOP+.03], [Phase((0, 0, -.05), .1), Phase((0, .05, 0), .1)], {}),
    }
    for name, (rotation, start, phases, kwargs) in controls.items():
        result = probe(rotation, start, phases, cut_events="cut_events_v1", **kwargs)
        metadata, activation = result["cut_events"]["metadata"], result["cut_events"]["activation"]
        assert metadata["events_logged"] == 0 and len(result["cut_events"]["log"]) == 0, name
        assert np.all(activation["newly_assigned"] == 0) and np.all(activation["stroke"] == 0), name
        assert np.all(activation["stroke_engaged"] == 0), name
        assert set(result["labelled_after_phase"]) == {0}, name
        if name == "disarmed":
            assert np.all(activation["cut"] == -1) and np.all(activation["segment_ok_count"] == 0)
        if name == "slide_no_contact":   # armed in-plane motion: the motion gate opens, nothing is engaged
            assert np.all(activation["stroke_any_motion"] == 1) and np.all(activation["segment_ok_count"] > 0)
        if name in ("flat_press", "side_push"):   # pure blade-normal motion fails M_k
            assert np.all(activation["stroke_any_motion"] == 0) and np.all(activation["segment_plow_count"] == 0)


# ----------------------------------------------------------------------------- multi-cut, overflow, options

@pytest.mark.cpu_sim
def test_two_sequential_cuts_write_cut_ids_0_and_1():
    phases = [Phase((0, 0, -.1), .2, 0), Phase((0, 0, .1), .2, 0), Phase((.24, 0, 0), .1, -1),
              Phase((0, 0, -.1), .2, 1)]
    result = probe(VERTICAL_CUT, [-.012+X0, 0, TOP+.004], phases, cuts=2, cut_events="cut_events_v1")
    log, activation = result["cut_events"]["log"], result["cut_events"]["activation"]
    assert set(np.unique(log["cut"]).tolist()) == {0, 1}
    assert np.all(activation["cut"][log["substep"]] == log["cut"])
    assert log["substep"][log["cut"] == 0].max() < log["substep"][log["cut"] == 1].min()
    assert_final_conflicts(log, result["state"]["codes"], result["cut_events"]["neighbours"], 2)
    x = result["points"][:, 0]
    for k, plane in enumerate((-.012+X0, .012+X0)):
        rows = log[log["cut"] == k]
        assert np.percentile(np.abs(rows["hermite_reference"][:, 0]-plane), 95) <= .05*SPACING, k
        assert np.all(np.abs(x[rows["p"]]-plane) < SPACING)
    assert result["cut_events"]["metadata"]["events_per_cut"] == {"0": int(np.sum(log["cut"] == 0)),
                                                                  "1": int(np.sum(log["cut"] == 1))}


@pytest.mark.cpu_sim
def test_overflow_raises():
    with pytest.raises(ce.CutEventOverflowError):
        probe(VERTICAL_CUT, [X0, 0, TOP+.004], [Phase((0, 0, -.05), .6)], cut_events="cut_events_v1",
              cut_events_capacity=4)
    points, masses = block()
    bed, _ = make_bed(points, masses)
    recorder = ce.CutEventRecorder(bed, lattice_of(points), summary_capacity=3)
    for _ in range(4):
        recorder.before_step()
        bed.step()
        recorder.after_step()
    with pytest.raises(ce.CutEventOverflowError):
        recorder.check()


def test_option_validation():
    from robotic_cutting_sim.kinematic_env import KinematicConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    with pytest.raises(ValueError):
        drive_blade(RULER_KNIFE_V1, VERTICAL_CUT, [0, 0, .02], [Phase((0, 0, 0), .001)], cutting_law=8,
                    cut_events="cut_events_v1")                         # tlmpm_v4: not recorded
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=RULER_KNIFE_V1, cutting_law=8, cut_events="cut_events_v1")
    with pytest.raises(ValueError):
        LiveCuttingMaterial(tool_profile=RULER_KNIFE_V1, cutting_law=8, solver="tlmpm_v5", cut_events="v2")
    with pytest.raises(ValueError):
        KinematicConfig(cut_events="cut_events_v1")
    assert KinematicConfig(solver="tlmpm_v5", cut_events="cut_events_v1").cut_events == "cut_events_v1"
    points, masses = block()
    with pytest.raises(ValueError):
        from robotic_cutting_sim.tlmpm_v4 import TLMPM
        polygon, half = RULER_KNIFE_V1.tlmpm_profile()
        ce.CutEventRecorder(TLMPM(points, masses, SPACING, TLMPMConfig(blade_half_width=float(half)), "cpu",
                                  blade_profile_xz=polygon, cutting_law=8), lattice_of(points))
