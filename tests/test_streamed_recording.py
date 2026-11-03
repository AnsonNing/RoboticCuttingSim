"""Synthetic CPU recording checks; never simulated cutting evidence."""
import json
from pathlib import Path
import select
import subprocess
import sys

import numpy as np

from robotic_cutting_sim.goals import protrusion_ownership
from robotic_cutting_sim.recording import PhysicsStreamWriter, load_physics_recording
from robotic_cutting_sim.scene_config import compile_task, configured_task_payload
from robotic_cutting_sim.seeded_trim import evaluate_recording, generate_scene, goal_config


def test_killed_writer_recovers_exact_committed_prefix_and_cannot_pass(tmp_path):
    task = compile_task(generate_scene(0), goal_config(), particle_spacing_m=.004)
    local, masses, _ = task.sample_particles()
    world = task.points_to_world(local)
    owner = protrusion_ownership(task, local, task.geometry.base_height_m)
    first = np.where(owner == 0, 2, 1).astype(np.int64)
    final = first | (np.where(owner == 1, 2, 1).astype(np.int64) << 2)
    # The synthetic support graph passes the original mechanical evaluator. A
    # missing finalization receipt must still prevent an interrupted PASS.
    reference = dict(mass_kg=masses, reference_positions=world,
        reference_stencil_nodes=np.zeros((len(local), 1), np.int64),
        reference_stencil_weights=np.ones((len(local), 1)),
        reference_stencil_gradients=np.zeros((len(local), 1, 3)))
    states = dict(positions=np.stack([world] * 3), velocities=np.zeros((3, len(local), 3)),
                  codes=np.stack([np.zeros(len(local), np.int64), first, final]))
    times = np.array([0., 12., 24.])
    fixture = tmp_path / 'synthetic_input.npz'
    np.savez_compressed(fixture, **reference, **states, time_s=times)
    interrupted = tmp_path / 'interrupted'
    interrupted.mkdir()
    (interrupted / 'compiled_task.json').write_text(configured_task_payload(task))
    program = '''
import os, signal, sys
import numpy as np
sys.path.insert(0, sys.argv[1])
from robotic_cutting_sim.recording import PhysicsStreamWriter
with np.load(sys.argv[2], allow_pickle=False) as source:
    data = dict(source)
keys = ('positions', 'velocities', 'codes')
writer = PhysicsStreamWriter(sys.argv[3], {k: v for k, v in data.items()
    if k not in (*keys, 'time_s')}, expected_duration_s=24.)
for i, time in enumerate(data['time_s']):
    writer.append(time, {k: data[k][i] for k in keys})
# Simulate a hard kill between an array write and its atomic checkpoint.
stream = writer.streams['positions']
stream.write(b'uncommitted-tail')
stream.flush()
os.fsync(stream.fileno())
print('committed', flush=True)
signal.pause()
'''
    source = Path(__file__).resolve().parents[1] / 'src'
    process = subprocess.Popen([sys.executable, '-I', '-c', program, str(source), str(fixture), str(interrupted)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert select.select([process.stdout], [], [], 20)[0], 'Writer did not reach a committed prefix'
        assert process.stdout.readline().strip() == 'committed'
    finally:
        process.kill()
        stdout, stderr = process.communicate(timeout=10)
    assert process.returncode == -9, (stdout, stderr)
    before = {p.name: p.read_bytes() for p in interrupted.iterdir()}
    recovered, checkpoint = load_physics_recording(interrupted)
    assert checkpoint['sample_count'] == 3 and not checkpoint['finalized'] and not checkpoint['complete']
    for key, expected in {**reference, **states, 'time_s': times}.items():
        assert recovered[key].dtype == expected.dtype
        np.testing.assert_array_equal(recovered[key], expected)
    assert isinstance(recovered['positions'], np.memmap) and not recovered['positions'].flags.writeable
    partial = evaluate_recording(interrupted)
    assert partial['outcome']['status'] == 'PASS'  # synthetic graph only
    assert partial['status'] == 'FAIL_INTEGRITY' and not partial['success']
    assert not partial['integrity']['checks']['stream_finalized']
    assert before == {p.name: p.read_bytes() for p in interrupted.iterdir()}

    finalized = tmp_path / 'finalized'
    finalized.mkdir()
    (finalized / 'compiled_task.json').write_text(configured_task_payload(task))
    writer = PhysicsStreamWriter(finalized, reference, expected_duration_s=24.)
    for i, time in enumerate(times):
        writer.append(time, {key: value[i] for key, value in states.items()})
    writer.finalize()
    complete = evaluate_recording(finalized)
    expected = task.evaluate_goal(dict(**reference, **states, time_s=times,
        particle_spacing_m=task.particle_spacing_m, code_encoding='packed2', max_cuts=2))
    assert complete['success'] and complete['integrity']['status'] == 'PASS'
    assert complete['outcome'] == expected['outcome'] == partial['outcome']
    assert complete['recording_completion']['complete']
    assert json.loads((finalized / 'physics_stream.json').read_text())['finalized']


def test_streamed_recorder_reuses_single_sample_and_legacy_default_stays_full(tmp_path, monkeypatch):
    from test_live_env import backend
    from robotic_cutting_sim.live_env import CuttingTaskEnv
    from robotic_cutting_sim.policy_env import load_episode_recorder

    native = backend('lab')
    native.material.points = native.material.world_points.copy()
    native.material.lattice = np.zeros((2, 3), int)
    native.material.parameter_receipt = {'scope': 'synthetic_unit_test'}
    native.material.history_state = native.material.world_state
    native.body_poses = lambda: np.array([[0., 0., 0., 0., 0., 0., 1.]])
    native.rigid_link_paths = ['synthetic_body']
    from dataclasses import dataclass

    @dataclass
    class Options:
        camera: bool = False

    native.config = Options()
    native.task.seed = 42
    native.task.control_dt_s = .02
    env = CuttingTaskEnv(native, history_mode='latest', observation_profile='state')
    env.reset()
    env._frames = {}
    recorder_type, _ = load_episode_recorder()
    monkeypatch.setitem(recorder_type.__init__.__globals__, 'manifest', lambda config, backend: {})
    recorder = recorder_type(tmp_path / 'streamed', env, 'synthetic_unit_test', every=1, stream_physics=True)
    # A second transfer would duplicate the matching material sample.
    native.material.history_state = lambda: (_ for _ in ()).throw(AssertionError('Unexpected duplicate transfer'))
    for timestamp in (.02, .04):
        native.time_s = timestamp
        native.material.history_state = native.material.world_state
        env._record_oracle(force=True)
        native.material.history_state = lambda: (_ for _ in ()).throw(AssertionError('Unexpected duplicate transfer'))
        recorder.capture()
        assert len(env._history) == 1 and recorder.samples == []
    recorder.finish({'status': 'SYNTHETIC_UNIT_TEST'})
    saved, checkpoint = load_physics_recording(recorder.output)
    assert saved['time_s'].tolist() == [0., .02, .04]
    assert checkpoint['finalized'] and checkpoint['complete']
    assert not (recorder.output / 'physics_state.npz').exists()
    assert env.evaluate()['status'] == 'NOT_RUN'
    assert CuttingTaskEnv(backend('lab')).history_mode == 'full'
