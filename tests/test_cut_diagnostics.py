"""cut_diagnostics_v1 (WP-12 Phase 4a, spec §4.2): read-only per-segment / per-particle cutting diagnostics.

Coarse CPU checks (Warp device "cpu", 4 mm blocks of 270 particles, ruler_knife_v1):
- physics neutrality: a law-8 through-cut with the recorder on equals the run with it off bit for bit
  (every snapshot array, reaction, stroke state) on tlmpm_v5 (packed codes) and tlmpm_v4 (ternary codes);
- every new digit is recorded once (assign_substep set exactly for the labelled particles, per-segment band/lead
  counts sum to the labels, substep rows sum to the labels);
- a half-depth cut leaves the ligament unassigned with decisive class ``edge_never_passed_while_stroke_on``;
- a 46 deg raked skim (motion 1 deg beyond theta_max = 45 deg) never opens the law-4 gate: 0 seg_ok, stroke never on,
  0 labels, while contact and cutting work are recorded with the gate closed;
- a pure draw at a generic orientation shows the swept-normal conditioning flags (``ill_float``) on its new
  digits, and the recorded flipped count equals a NumPy recomputation.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

wp = pytest.importorskip("warp")

from robotic_cutting_sim import cut_diagnostics as cd
from robotic_cutting_sim.blade_probe import VERTICAL_CUT, block
from robotic_cutting_sim.material_config import TLMPMConfig
from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

TOP = .012
X0 = .002


def make_bed(module, cuts=1, half_width=None):
    points, masses = block()
    polygon, half = RULER_KNIFE_V1.tlmpm_profile()
    config = TLMPMConfig(cuts=cuts, blade_half_width=float(half if half_width is None else half_width),
                         cutting_energy=5.)
    if module == 5:
        from robotic_cutting_sim.tlmpm_v5 import TLMPM
    else:
        from robotic_cutting_sim.tlmpm_v4 import TLMPM
    return TLMPM(points, masses, .004, config, "cpu", blade_profile_xz=polygon, cutting_law=8), config, points


def run(module, rotation, start, phases, *, diagnostics, half_width=None):
    bed, config, points = make_bed(module, half_width=half_width)
    recorder = cd.CutDiagnosticsRecorder(bed) if diagnostics else None
    quaternion, position = rotation.as_quat(), np.asarray(start, float)
    reactions = []
    for velocity, duration, cut in phases:
        velocity = np.asarray(velocity, float)
        steps = int(round(duration/config.dt))
        for first in range(0, steps, 200):
            count = min(200, steps-first)
            poses = np.array([np.r_[position+velocity*config.dt*i, quaternion] for i in range(count+1)], np.float32)
            twists = np.tile(np.r_[velocity, 0., 0., 0.], (count+1, 1)).astype(np.float32)
            bed.set_trajectory(poses, twists, np.full(count+1, cut, np.int32))
            for _ in range(count):
                bed.reaction.zero_()
                if recorder is not None:
                    recorder.before_step()
                bed.step()
                if recorder is not None:
                    recorder.after_step()
                reactions.append(bed.reaction.numpy()[0].copy())
            if recorder is not None:
                recorder.end_control_step()
            position = position+velocity*config.dt*count
    return bed, recorder, np.asarray(reactions), points


THROUGH = dict(rotation=VERTICAL_CUT, start=[X0, 0, TOP+.004],
               phases=[((0, 0, -.05), .32, 0), ((0, 0, -.05), .38, 0), ((0, 0, 0), .01, 0)])


@pytest.mark.cpu_sim
@pytest.mark.parametrize("module", [5, 4])
def test_recorder_is_physics_neutral_bitwise_on_cpu(module):
    off, _, r_off, _ = run(module, **THROUGH, diagnostics=False)
    on, recorder, r_on, _ = run(module, **THROUGH, diagnostics=True)
    a, b = off.snapshot(), on.snapshot()
    assert set(a) == set(b)
    for key in a:
        assert a[key].tobytes() == b[key].tobytes(), key
    assert r_off.tobytes() == r_on.tobytes()
    assert off.cutting_state(count_labels=True) == on.cutting_state(count_labels=True)
    codes = b["codes"]
    labelled = np.count_nonzero(codes)
    assert labelled > 0
    table = recorder.particle_table()
    assigned = table["assign_substep"][:, 0] >= 0
    assert np.array_equal(assigned, codes != 0)
    segments = recorder.segment_table()
    assert segments.shape[1:] == (on.edge_segments, len(cd.SEGMENT_COLUMNS))
    new_columns = [cd.SEGMENT_COLUMNS.index(c) for c in ("band_new_digit1", "band_new_digit2", "lead_new_digit1",
                                                          "lead_new_digit2")]
    assert int(segments[:, :, new_columns].sum()) == labelled
    rows = recorder.substep_table()
    assert len(rows) == 7100 and int(rows[:, cd.SUBSTEP_COLUMNS.index("band_new")].sum()
                                     + rows[:, cd.SUBSTEP_COLUMNS.index("lead_new")].sum()) == labelled
    work = segments[:, :, cd.SEGMENT_COLUMNS.index("cutting_work_sum_J")].sum()
    assert np.isclose(work, b["contact_work_j"].astype(np.float64).sum(), rtol=1e-4)
    metadata = recorder.metadata()
    assert metadata["version"] == "cut_diagnostics_v1" and metadata["new_digits"] == labelled
    assert metadata["code_encoding"] == ("packed2" if module == 5 else "ternary")
    # A vertical in-plane cut: the motion gate holds, the plow mask never does, and the swept normal is the plane.
    assert segments[:, :, cd.SEGMENT_COLUMNS.index("seg_plow_substeps")].sum() == 0
    assert metadata["new_digits_by_flag"]["flipped_by_float_fallback"] == 0


@pytest.mark.cpu_sim
def test_half_depth_ligament_is_classified_edge_never_passed():
    # Edge from 4 mm above the 12 mm block top down to 6 mm depth (z = 6 mm), then stop.
    bed, recorder, _, points = run(5, VERTICAL_CUT, [X0, 0, TOP+.004], [((0, 0, -.05), .2, 0), ((0, 0, 0), .05, 0)],
                                   diagnostics=True)
    codes = bed.code.numpy()
    table = recorder.particle_table()
    near = np.abs(points[:, 0]-X0) <= .004
    ligament = near & (points[:, 2] < .006-.002) & (codes == 0)
    assert ligament.sum() > 0
    idx, classes = cd.classify_unassigned(table, 0, ligament)
    assert np.all(classes == cd.UNASSIGNED_CLASSES.index("edge_never_passed_while_stroke_on"))
    assert np.all(table["min_f_stroke_m"][idx, 0] > 0)
    labelled = codes != 0
    assert labelled.sum() > 0 and np.all(table["assign_f_m"][labelled, 0] <= 0)   # band rule only (no plowing)


@pytest.mark.cpu_sim
def test_rake_46_skim_never_opens_the_motion_gate():
    # Edge frame: advance d, normal n; rake 46 deg: the horizontal +x motion is 46 deg out of the blade plane.
    rotation = Rotation.from_euler("y", 46, degrees=True)   # d = (cos, 0, -sin), n = (sin, 0, cos)
    bed, recorder, _, points = run(5, rotation, [-.040, 0, .009], [((.025, 0, 0), .6, 0)], diagnostics=True)
    segments = recorder.segment_table()
    col = cd.SEGMENT_COLUMNS.index
    assert segments[:, :, col("armed_substeps")].sum() > 0
    assert segments[:, :, col("seg_ok_substeps")].sum() == 0
    assert np.count_nonzero(bed.code.numpy()) == 0
    rows = recorder.substep_table()
    assert np.all(rows[:, cd.SUBSTEP_COLUMNS.index("stroke")] == 0)
    assert rows[:, cd.SUBSTEP_COLUMNS.index("contact_particles")].sum() > 0
    assert segments[:, :, col("work_with_gate_closed_particle_substeps")].sum() > 0
    table = recorder.particle_table()
    first = table["first_fail"][:, 0]
    assert first[:, cd.REASONS.index("stroke_off")].sum() > 0 and first[:, 1:].sum() == 0


@pytest.mark.cpu_sim
def test_pure_draw_flags_swept_normal_conditioning():
    # Edge along ~x (a generic 97 deg yaw of VERTICAL_CUT) shortened to 24 mm on the 60 mm block: descend 6 mm into
    # the top, then draw along the edge so its span moves into uncut material (new digits during a pure draw).
    rotation = Rotation.from_euler("yz", [90, 97], degrees=True)
    along, advance, normal = (rotation.apply(v) for v in ([0, 1, 0], [1, 0, 0], [0, 0, 1]))
    start = np.array([.012, 0, TOP+.004])
    bed, recorder, _, _ = run(5, rotation, start, [(.05*advance, .2, 0), (.05*along, .3, 0)], diagnostics=True,
                              half_width=.012)
    table = recorder.particle_table()
    flags = table["assign_flags"][:, 0]
    drawn = table["assign_substep"][:, 0] >= 2000
    assert drawn.sum() > 0
    assert np.all(flags[drawn] & cd.ASSIGN_FLAGS["ill_float"])
    # NumPy recomputation: the corrected fallback side is the blade-frame z coordinate at the assignment pose.
    x = table["assign_x_m"][drawn, 0]
    digit = cd.digits_of(bed.code.numpy()[drawn], 0, "packed2")
    steps = table["assign_substep"][drawn, 0]-2000
    centre = (start+.05*advance*.2)[None]+.05*along[None]*(steps[:, None]*1e-4)
    z = (x-centre)@normal
    robust = np.where(z < 0, 1, 2)
    clear = np.abs(z) > 1e-6          # float64 vs float32 pose: compare only particles clear of the plane
    flipped = (flags[drawn] & cd.ASSIGN_FLAGS["flipped_by_float_fallback"]) != 0
    assert np.array_equal(flipped[clear], (robust != digit)[clear])
    rows = recorder.substep_table()
    assert int(rows[:, cd.SUBSTEP_COLUMNS.index("new_digits_flipped_by_float_fallback")].sum()) == int(flipped.sum())
    print(f"pure draw: {drawn.sum()} digits during the draw, {int(flipped.sum())} would flip under the fallback")


def test_option_validation():
    class Fake:
        cutting_law = 2
    with pytest.raises(ValueError):
        cd.CutDiagnosticsRecorder(Fake())
