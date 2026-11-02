"""Small synthetic CPU fixtures; these are never Native cutting evidence."""
import copy
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.agentic_trim_media import export_success
from robotic_cutting_sim.agentic_trim_env import AgenticTrimEnv
from robotic_cutting_sim.policy_env import load_episode_recorder
from robotic_cutting_sim.goals import protrusion_ownership
from robotic_cutting_sim.scene_config import compile_task, configured_task_payload
from robotic_cutting_sim.seeded_trim import evaluate_recording, generate_scene, goal_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def task():
    # Coarse particles keep graph tests quick. The common support below is an
    # intentionally synthetic graph, not a simulated blade/material outcome.
    return compile_task(generate_scene(0), goal_config(), particle_spacing_m=.004)


def synthetic(task):
    local, masses, _ = task.sample_particles()
    world = task.points_to_world(local)
    owner = protrusion_ownership(task, local, task.geometry.base_height_m)
    first = np.where(owner == 0, 2, 1).astype(np.int64)
    final = first | (np.where(owner == 1, 2, 1).astype(np.int64) << 2)
    return dict(time_s=np.array([0., 12., 24.]), positions=np.stack([world] * 3),
        codes=np.stack([np.zeros(len(local), np.int64), first, final]), mass_kg=masses,
        reference_positions=world, reference_stencil_nodes=np.zeros((len(local), 1), np.int64),
        reference_stencil_weights=np.ones((len(local), 1)),
        reference_stencil_gradients=np.zeros((len(local), 1, 3)),
        particle_spacing_m=task.particle_spacing_m, code_encoding='packed2', max_cuts=2), owner


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def recording(path, task, *, with_cameras=False):
    path.mkdir()
    data, _ = synthetic(task)
    np.savez_compressed(path / 'physics_state.npz', **data)
    (path / 'compiled_task.json').write_text(configured_task_payload(task))
    save_json(path / 'episode_receipt.json', dict(home_status='NOT_RUN', simulation_duration_s=24., wall_time_s=.1))
    save_json(path / 'provenance.json', dict(controller_provenance='synthetic_unit_test', split='unit_test',
        scene_seed=0, policy_seed=0, sim_seed=42, candidate_id='synthetic_unit_test',
        controller_sha256='0' * 64, solver_version='tlmpm_v7_2'))
    if with_cameras:
        (path / 'observations').mkdir()
        rows = [dict(frame=i, time_s=float(t)) for i, t in enumerate(data['time_s'])]
        save_json(path / 'frame_manifest.json', rows)
        for i, t in enumerate(data['time_s']):
            rgb = np.full((480, 640, 3), 30 + i * 50, np.uint8)
            np.savez_compressed(path / 'observations' / f'global_d455_{i:04d}.npz',
                rgb=rgb, physics_timestamp_s=t)
            np.savez_compressed(path / 'observations' / f'wrist_d405_0_{i:04d}.npz',
                rgb=rgb, depth_m=np.full((480, 640), .2, np.float32),
                valid_mask=np.ones((480, 640), bool), physics_timestamp_s=t)
        action = dict(info=dict(action_requested={'cut_enabled': False}, action_executed={'cut_enabled': False}))
        (path / 'actions.jsonl').write_text(json.dumps(action) + '\n')
    return data


def test_packed_two_cut_recording_uses_original_evaluator(task, tmp_path):
    rec = tmp_path / 'attempt_000'
    data = recording(rec, task)
    expected = task.evaluate_goal(data)
    before = {p.name: p.read_bytes() for p in rec.iterdir()}
    actual = evaluate_recording(rec)
    assert actual['status'] == actual['verdict'] == 'PASS'
    assert actual['success'] and actual['cuts'] == 2
    assert actual['outcome'] == expected['outcome']
    assert actual['outcome']['metrics']['mechanical_components'] == 3
    assert all(row['removed_mass_fraction'] == 1. for row in actual['outcome']['metrics']['items'])
    assert actual['surface_quality_status'] == 'NOT_RUN_MESH_NOT_EXPORTED'
    assert actual['home_status'] == 'NOT_RUN'
    assert before == {p.name: p.read_bytes() for p in rec.iterdir()}


def test_height_reduction_without_mechanical_detachment_fails(task):
    data, _ = synthetic(task)
    data['codes'][:] = 0
    data['positions'][-1, :, 2] = task.points_to_world(np.array([[0., 0., .015]]))[0, 2]
    result = task.evaluate_goal(data)
    assert result['integrity']['status'] == 'PASS'
    assert result['outcome']['status'] == 'FAIL' and not result['success']
    assert all(row['removed_mass_fraction'] == 0. for row in result['outcome']['metrics']['items'])


def test_gradient_only_code_zero_bridge_prevents_false_separation(task):
    data, owner = synthetic(task)
    bridge = np.flatnonzero(owner == 1)[0]
    data['codes'][1:, bridge] = 0
    data['reference_stencil_nodes'] = np.c_[np.zeros(len(owner), np.int64), np.arange(len(owner)) + 1]
    data['reference_stencil_weights'] = np.c_[np.ones(len(owner)), np.zeros(len(owner))]
    data['reference_stencil_weights'][bridge] = [0., 1.]
    gradients = np.zeros((len(owner), 2, 3))
    gradients[bridge, 0, 0] = 1.
    data['reference_stencil_gradients'] = gradients
    result = task.evaluate_goal(data)
    assert result['integrity']['status'] == 'PASS'
    assert result['outcome']['metrics']['mechanical_components'] == 1
    assert not result['success']
    # Without that zero-weight, nonzero-gradient support the apparent outcome
    # passes. It is the original evaluator's gradient graph that rejects it.
    without_gradient = copy.deepcopy(data)
    without_gradient['reference_stencil_gradients'][:] = 0
    assert task.evaluate_goal(without_gradient)['outcome']['status'] == 'PASS'


def test_each_bump_threshold_and_substrate_threshold_remain_strict(task):
    data, owner = synthetic(task)
    # One retained particle in a seven-particle target means <90% removed.
    data['codes'][-1, np.flatnonzero(owner == 0)[0]] = 5
    data['codes'][1] = 0
    result = task.evaluate_goal(data)
    assert result['integrity']['status'] == 'PASS'
    assert not result['success'] and not result['outcome']['checks']['selected_protrusions_removed']
    data, owner = synthetic(task)
    local = task.points_to_local(data['reference_positions'])
    bottom = local[:, 2].min()
    substrate = np.flatnonzero((owner == -1) & (local[:, 2] > bottom + 1e-7))
    # Detach >1% of the substrate while all targets remain detached.
    data['codes'][-1, substrate[:18]] = 6
    data['codes'][1] = 0
    result = task.evaluate_goal(data)
    assert result['integrity']['status'] == 'PASS'
    assert result['outcome']['checks']['selected_protrusions_removed']
    assert not result['outcome']['checks']['substrate_holds_below_plane_mass']
    assert not result['outcome']['checks']['detached_below_plane_mass_bounded']
    assert not result['success']


def test_history_overwrite_and_early_close_fail_integrity_independently(task):
    data, _ = synthetic(task)
    data['codes'][1] = data['codes'][-1]
    data['codes'][-1] &= 12  # drop the first packed digit after it was assigned
    result = task.evaluate_goal(data)
    assert result['verdict'] == 'FAIL_INTEGRITY'
    assert not result['integrity']['checks']['persistent_history_no_digit_rewrites']
    data, _ = synthetic(task)
    data['time_s'][-1] = 23.98
    result = task.evaluate_goal(data)
    assert result['verdict'] == 'FAIL_INTEGRITY'
    assert result['outcome']['status'] == 'PASS'
    assert not result['integrity']['checks']['complete_episode']


def test_reset_attempts_are_all_scored_and_preserved(task, tmp_path):
    rec = tmp_path / 'recording'
    rec.mkdir()
    for i in range(2):
        data = recording(rec / f'attempt_{i:03d}', task)
        if i == 0:
            data['codes'][:] = 0
            np.savez_compressed(rec / 'attempt_000' / 'physics_state.npz', **data)
    save_json(rec / 'attempts.json', [dict(path=f'attempt_{i:03d}') for i in range(2)])
    result = evaluate_recording(rec)
    assert result['attempt_count'] == 2
    assert result['attempt_outcomes'] == [dict(status='FAIL', success=False), dict(status='PASS', success=True)]
    assert result['success'] and (rec / 'attempt_000' / 'physics_state.npz').is_file()


def test_fresh_reset_uses_actual_recorder_and_persists_final_raw_state(task, tmp_path, monkeypatch):
    recorder_type, converter = load_episode_recorder()
    monkeypatch.setitem(recorder_type.__init__.__globals__, 'manifest',
                        lambda config, backend: dict(config=config, backend=backend))
    monkeypatch.setattr('robotic_cutting_sim.agentic_trim_env.load_episode_recorder',
                        lambda: (recorder_type, converter))
    data, _ = synthetic(task)
    points = task.sample_particles()[0]
    array = lambda values: SimpleNamespace(numpy=lambda: values)
    bed = SimpleNamespace(nodes=array(data['reference_stencil_nodes']),
        weights=array(data['reference_stencil_weights']), gradients=array(data['reference_stencil_gradients']))
    material = SimpleNamespace(bed=bed, masses=data['mass_kg'], world_points=data['reference_positions'],
        points=points, lattice=np.arange(len(points))[:, None], parameter_receipt={'scope': 'synthetic_unit_test'},
        history_state=lambda: dict(positions=data['positions'][0], velocities=np.zeros_like(points), codes=data['codes'][0]))

    @dataclass
    class Options:
        camera: bool = True

    backend = SimpleNamespace(config=Options(), material=material, calibration={}, rigid_link_paths=['test_body'],
        body_poses=lambda: np.array([[0., 0., 0., 0., 0., 0., 1.]]),
        measured=lambda: {'tool_pose': np.array([0., 0., 0., 0., 0., 0., 1.])})
    rgb = np.zeros((2, 3, 3), np.uint8)
    inner = SimpleNamespace(backend=backend, task=task, time_s=0., runtime_source='synthetic_unit_test',
        _frames={'global_d455': dict(rgb=rgb, physics_timestamp_s=0.)})
    env = SimpleNamespace(inner=inner, _step_index=0, time_s=0., reset=lambda **kwargs: None,
                          component_labels=lambda: {'count': 1})
    output = tmp_path / 'episode'
    output.mkdir()
    adapter = AgenticTrimEnv(env, output, task,
        dict(policy_seed=0, sim_seed=42, scene_hash='test_scene'), seed=0)
    monkeypatch.setattr(adapter, 'observe', lambda: {})
    adapter.reset()
    recorder = env._recorder
    assert isinstance(recorder, recorder_type)
    assert recorder.output == output / 'recording' / 'attempt_000'
    inner.time_s = .02
    inner._frames['global_d455']['physics_timestamp_s'] = .02
    recorder.finish({'status': 'SYNTHETIC_UNIT_TEST'})
    with np.load(recorder.output / 'physics_state.npz', allow_pickle=False) as saved:
        assert saved['time_s'].tolist() == [0., .02]
        np.testing.assert_array_equal(saved['reference_stencil_gradients'], data['reference_stencil_gradients'])
        np.testing.assert_array_equal(saved['codes'][0], 0)
    assert (recorder.output / 'observations/global_d455_0001.npz').is_file()
    assert json.loads((output / 'recording/attempts.json').read_text())[0]['path'] == 'attempt_000'
    save_json(recorder.output / 'episode_receipt.json', dict(home_status='NOT_RUN', simulation_duration_s=.02))
    assert evaluate_recording(recorder.output)['status'] == 'FAIL_INTEGRITY'


def test_actual_success_encoding_and_audit_from_same_original_frames(task, tmp_path):
    if not shutil.which('ffmpeg') or not shutil.which('ffprobe'):
        pytest.skip('ffmpeg/ffprobe unavailable: media verification NOT_RUN')
    rec = tmp_path / 'attempt_000'
    recording(rec, task, with_cameras=True)
    save_json(rec / 'independent_evaluation.json', evaluate_recording(rec))
    before = (rec / 'physics_state.npz').read_bytes()
    manifest = export_success(rec)
    assert manifest['media_decode_status'] == 'PASS'
    assert manifest['original_frame_count'] == 3
    assert manifest['capture_time_s'] == [0., 12., 24.]
    assert export_success(rec) == manifest  # resumable idempotent export
    spec = importlib.util.spec_from_file_location('wp15_audit', ROOT / 'scripts/audit_wp15_evidence.py')
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    result = audit.audit([rec], public=True)
    assert result['status'] == result['success_media_status'] == 'PASS'
    assert result['media_complete_count'] == result['successful_attempt_count'] == 1
    assert str(tmp_path) not in json.dumps(result)
    assert before == (rec / 'physics_state.npz').read_bytes()
    # Changed source imagery must invalidate an existing successful movie.
    (rec / 'observations/global_d455_0001.npz').write_bytes(b'changed unit-test source')
    damaged = audit.audit([rec], public=True)
    assert damaged['status'] == 'FAIL'
