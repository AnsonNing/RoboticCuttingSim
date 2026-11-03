"""Direct CPU coverage of the zero-contribution blade reaction optimization."""
import numpy as np
import pytest
import warp as wp
from scipy.spatial.transform import Rotation

from robotic_cutting_sim import tlmpm_v7_2_contacts as V


def run_rigid(kernel=V._rigid_contacts, *, local_x=(.0002, 0., 0.),
              local_v=(-.2, .12, -.05), mass=1.5e-6, rotated=False,
              armed=False, floor=False):
    """Launch the real contact kernel without grid transfers or a GPU."""
    rotation = Rotation.from_rotvec([.4, -.3, .2]) if rotated else Rotation.identity()
    origin = np.array([.03, -.02, .05]) if rotated else np.zeros(3)
    position = rotation.apply(local_x) + origin
    velocity = rotation.apply(local_v)
    linear = rotation.apply([.02, 0., 0.]) if armed else np.zeros(3)
    angular = np.array([.2, -.1, .3]) if rotated else np.zeros(3)
    array = lambda values, dtype: wp.array(values, dtype=dtype, device='cpu')
    zero = lambda count, dtype: wp.zeros(count, dtype=dtype, device='cpu')
    x, v = array([position], wp.vec3), array([velocity], wp.vec3)
    impulse = zero(1, wp.vec3)
    progress, work = array([.2], float), array([1.e-6], float)
    reaction, stats = zero(1, wp.spatial_vector), zero(7, float)
    flags, engaged, totals = zero(2, int), zero(1, int), zero(12, wp.float64)
    pose = array([[*origin, *rotation.as_quat()]], wp.transform)
    twist = array([[*linear, *angular]], wp.spatial_vector)
    profile = array([[-.01, -.001], [0., -.001], [0., .001], [-.01, .001]], wp.vec2)
    wp.launch(kernel, 1, [x, v, array([mass], float), impulse, progress, work,
        pose, twist, array([0 if armed else -1], int), zero(1, int), reaction, stats,
        .00045, 1.e-9, .0001, .1, .3, 1., .5, profile, 4, .025,
        int(floor), 0, 1, 4, 1., .001, .01, flags, 2, array([1, 1], int), engaged, totals],
        device='cpu')
    return {name: value.numpy() for name, value in {
        'impulse': impulse, 'reaction': reaction, 'progress': progress, 'work': work,
        'stats': stats, 'flags': flags, 'engaged': engaged, 'totals': totals}.items()}


def test_air_contact_keeps_all_physical_outputs_and_work_unchanged():
    state = run_rigid(local_x=(.2, 0., 0.), armed=True)
    for name in ('impulse', 'reaction', 'stats', 'flags', 'engaged', 'totals'):
        assert not state[name].any(), name
    np.testing.assert_array_equal(state['progress'], np.array([.2], np.float32))
    np.testing.assert_array_equal(state['work'], np.array([1.e-6], np.float32))


@pytest.mark.parametrize('rotated', [False, True])
def test_nonzero_blade_impulse_reaction_and_cutting_engagement_are_retained(rotated):
    state = run_rigid(rotated=rotated, armed=True)
    impulse = state['impulse'][0]
    assert np.linalg.norm(impulse) > 0.
    np.testing.assert_array_equal(state['reaction'][0, :3], -impulse)
    assert state['stats'][1] == 1.
    assert state['engaged'][0] == 1
    assert state['progress'][0] > .2
    assert state['work'][0] > 1.e-6
    assert state['totals'][0] > 0.  # normal impulse
    assert state['totals'][1] > 0.  # friction impulse
    assert state['totals'][5] > 0.  # consumed cutting work
    assert np.linalg.norm(state['reaction'][0, 3:]) > 0.


def test_positive_gap_with_predicted_impact_still_transfers_reaction():
    state = run_rigid(local_x=(.00055, 0., 0.), local_v=(-2., .1, 0.))
    assert state['impulse'][0, 0] > 0.
    np.testing.assert_array_equal(state['reaction'][0, :3], -state['impulse'][0])
    assert state['stats'][1] == 1.
    assert state['stats'][3] == 0.  # no current penetration


def test_tiny_nonzero_impulse_survives_squared_norm_underflow():
    state = run_rigid(local_x=(.00045, 0., 0.), local_v=(-.2, 0., 0.), mass=1.5e-30)
    impulse = state['impulse'][0]
    assert impulse[0] > 0.
    assert np.dot(impulse, impulse) == 0.
    np.testing.assert_array_equal(state['reaction'][0, :3], -impulse)


def test_clear_blade_does_not_remove_existing_floor_contact():
    # Below the blade polygon but on the retained support table at z=-10 mm.
    state = run_rigid(local_x=(.2, 0., -.0097), local_v=(0., 0., -.1), floor=True)
    assert state['impulse'][0, 2] > 0.
    assert not state['reaction'].any()
    assert state['stats'][0] > 0.


def test_v7_2_contact_execution_has_an_explicit_original_path():
    from robotic_cutting_sim import tlmpm_v7_2 as C
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.material_config import TLMPMConfig
    points, masses = block(size_m=(.016, .008, .008), spacing=.002)
    config = TLMPMConfig(cuts=1, floor=False, tool=False, material_contact=False, gravity=(0., 0., 0.))
    beds = [C.TLMPM(points, masses, .002, config, 'cpu', optimize_contacts=enabled)
            for enabled in (False, True)]
    for bed in beds:
        bed.step(advance=False)
    assert beds[0].solver_receipt()['optimize_contacts'] is False
    assert beds[1].solver_receipt()['optimize_contacts'] is True
    assert len(beds[1].solver_receipt()['contact_source_sha256']) == 64
    for key in ('positions', 'velocities', 'F', 'codes', 'contact_work_j'):
        np.testing.assert_array_equal(beds[0].snapshot()[key], beds[1].snapshot()[key])


def test_original_v7_cut_profile_and_constructor_keep_their_source_binding():
    from pathlib import Path
    import hashlib
    from robotic_cutting_sim import research_profiles as P, tlmpm_v7 as original
    from robotic_cutting_sim.blade_probe import block
    from robotic_cutting_sim.material_config import TLMPMConfig
    task = P.compile_research_task('v7_cut_research_v1',
                                   'configs/wp12/scenes/bar_flat_board.json', 'sever_mid')
    P.validate_runtime(task, 'tlmpm_v7', 8, .0001, 4, 8, 1.)
    expected = P.document('v7_cut_research_v1')['source_sha256']['tlmpm_v7.py']
    assert hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest() == expected
    points, masses = block(size_m=(.016, .008, .008), spacing=.002)
    bed = original.TLMPM(points, masses, .002, TLMPMConfig(), 'cpu')
    assert bed.solver_receipt()['source_sha256']['tlmpm_v7.py'] == expected
    assert not hasattr(bed, 'optimize_contacts')
