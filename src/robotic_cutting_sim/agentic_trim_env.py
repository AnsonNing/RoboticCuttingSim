"""Deployable single-arm boundary and durable recordings for agentic control robot research."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .policy_env import jsonable, load_episode_recorder
from .scene_config import SceneInfeasible, configured_task_payload
from .seeded_trim import GOAL_TEXT, VERSION, compile_seed
from .runtime_memory import record_memory

ROOT = Path(__file__).resolve().parents[2]
CAMERAS = ('global_d455', 'wrist_d405_0')


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def independent_evaluate(recording, python=None):
    """A fresh CPU-only process imports only the frozen evaluator source and data."""
    target = Path(recording)/'independent_evaluation.json'
    program = ('import sys,json; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
        'from robotic_cutting_sim.seeded_trim import evaluate_recording; '
        'from robotic_cutting_sim.policy_env import jsonable; '
        'Path(sys.argv[3]).write_text(json.dumps(jsonable(evaluate_recording(sys.argv[2])),indent=2)+"\\n")')
    subprocess.run([python or sys.executable, '-I', '-c', program, str(ROOT/'src'),
                    str(recording), str(target)], check=True, timeout=300,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
    return json.loads(target.read_text())


class AgenticTrimEnv:
    """The external proxy sees reset, observe, explicit step and deferred close only.

    Scene compilation precedes Native startup. Mid-episode reset reconstructs
    the same scene and preserves the abandoned attempt as independent evidence.
    A reset before the first action is idempotent (the external worker resets too).
    """
    def __init__(self, env, output, task, manifest, *, seed, evaluator_python=None,
                 stream_physics=False, memory_log=None):
        self.env, self.output, self.task = env, Path(output), task
        self.manifest, self.seed = manifest, int(seed)
        self.evaluator_python = evaluator_python
        self.stream_physics, self.memory_log = bool(stream_physics), memory_log
        self.started, self.finished, self.attempts = False, True, []
        self.total_steps = 0
        self.wall_start = time.perf_counter()
        spec_path = self.output.parent/'spec.json'
        spec = json.loads(spec_path.read_text()) if spec_path.exists() else {}
        self.provenance = dict(version=VERSION, scene_seed=self.seed,
            policy_seed=manifest['policy_seed'], sim_seed=manifest['sim_seed'],
            scene_hash=manifest['scene_hash'], controller_sha256=spec.get('control', {}).get('sha256'),
            candidate_id=spec.get('ctx', {}).get('candidate_id') or
                         Path(spec.get('control', {}).get('path', 'diagnostic')).stem,
            agent_commit=spec.get('ctx', {}).get('agent_commit'),
            controller_provenance='agent_candidate' if spec else 'environment_diagnostic',
            split=spec.get('ctx', {}).get('split', 'diagnostic'), solver_version='tlmpm_v7_2',
            physical_recording='streamed_physics_v1' if self.stream_physics else 'legacy_npz')
        try:
            self.provenance['solver_commit'] = subprocess.check_output(
                ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True,
                stderr=subprocess.PIPE, timeout=10).strip()
        except (OSError, subprocess.SubprocessError) as error:
            self.provenance['solver_commit'] = None
            self.provenance['solver_commit_error'] = f'{type(error).__name__}: {error}'

    @property
    def time_s(self):
        return self.env.time_s

    def reset(self, seed=None):
        if seed is not None and int(seed) != self.seed:
            raise ValueError('An episode retains its compiled scene; open a new episode to change seed')
        if self.started and self.env._step_index == 0:
            return self.observe(), {'status': 'READY', 'timestamp_s': self.time_s}
        if self.started:
            self.finalize('RESET')
        record_memory(self.memory_log, 'before_reset', attempt=len(self.attempts))
        self.env.reset(seed=self.manifest['sim_seed'], controller='agentic control robot')
        record_memory(self.memory_log, 'after_reset', attempt=len(self.attempts))
        index = len(self.attempts)
        rec = self.output/'recording'/f'attempt_{index:03d}'
        rec.parent.mkdir(exist_ok=True)
        recorder, _ = load_episode_recorder()
        options = {'stream_physics': True} if self.stream_physics else {}
        self.env._recorder = recorder(rec, self.env.inner, 'agentic control robot', every=10, **options)
        self.env._recorder_finished = False
        self.current_recording = rec
        write_json(rec/'compiled_task.json', json.loads(configured_task_payload(self.task)))
        write_json(rec/'scene_manifest.json', self.manifest)
        write_json(rec/'provenance.json', self.provenance)
        initial = self.env.component_labels()
        if initial['count'] != 1:
            raise SceneInfeasible('initial_mechanical_connectivity', str(initial['count']))
        self.attempts.append(dict(path=rec.name, initial_mechanical_components=initial['count'], status='RUNNING'))
        write_json(self.output/'recording'/'attempts.json', self.attempts)
        self.started, self.finished = True, False
        return self.observe(), {'status': 'READY', 'timestamp_s': self.time_s}

    def task_spec(self):
        return dict(task=VERSION, goal_text=GOAL_TEXT, action_types=['edge_delta', 'edge_pose', 'joint_position'],
                    control_dt_s=.02, duration_s=24., max_cuts=2, controlled_arm='arm_0',
                    gripper='hold', cameras=list(CAMERAS),
                    action_limits=self.env.action_limits.to_dict())

    def observe(self):
        observation = self.env.observe()
        cameras = observation['cameras']
        if set(cameras.get('rgb', {})) != set(CAMERAS):
            raise RuntimeError('Required independent live camera streams unavailable')
        if set(cameras.get('depth_m', {})) != {'wrist_d405_0'}:
            raise RuntimeError('Challenge depth must come only from wrist_d405_0')
        proprio = {}
        for key, value in observation['proprio'].items():
            array = np.asarray(value)
            if key in ('q', 'qdot', 'finger_q', 'finger_qdot'):
                array = array[0]
            if key in ('q', 'qdot', 'hand_pose', 'finger_q', 'finger_qdot'):
                proprio[key] = array.copy()
        return dict(timestamp_s=observation['timestamp_s'], cameras=copy.deepcopy(cameras), proprio=proprio,
            ee_pose_world_xyzw=np.asarray(observation['ee_pose_world_xyzw']).copy(),
            edge_command_world_xyzw=np.asarray(observation['edge_command_world_xyzw']).copy(),
            gripper=copy.deepcopy(observation['gripper']), camera_timestamp_s=observation['camera_timestamp_s'],
            task=self.task_spec())

    def step(self, action):
        if action is None:
            raise ValueError('Explicit candidate action required; reference playback is unavailable')
        if self.finished:
            raise RuntimeError('Episode ended')
        if self.total_steps >= 1200:
            raise RuntimeError('The total episode control-step budget is exhausted')
        _, _, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        if self.total_steps == 1 or self.total_steps % 100 == 0:
            record_memory(self.memory_log, 'control_step', control_steps=self.total_steps,
                          simulation_time_s=self.time_s)
        allowed = ('action_status', 'action_requested', 'action_executed', 'action_report',
                   'action_timestamp_s', 'timestamp_s', 'control_step', 'ik_status', 'termination_reason')
        public_info = {k: copy.deepcopy(info[k]) for k in allowed if k in info}
        if info.get('action_rejection'):
            public_info['action_rejection'] = {'reason': info['action_rejection']['reason']}
        return self.observe(), 0., bool(terminated), bool(truncated or self.total_steps >= 1200), public_info

    def close(self):
        # The worker persists its result before Native teardown. Candidate close
        # finalizes data but Native shutdown is deferred to the context manager.
        self.finalize('CONTROLLER_CLOSE')

    def finalize(self, reason='CONTROLLER_DONE'):
        if not self.started or self.finished:
            return
        rec = self.current_recording
        record_memory(self.memory_log, 'before_finalize', control_steps=self.total_steps,
                      simulation_time_s=self.time_s)
        write_json(rec/'episode_receipt.json', {**self.provenance, 'reason': reason, 'home_status': 'NOT_RUN',
                   'wall_time_s': time.perf_counter()-self.wall_start, 'simulation_duration_s': self.time_s})
        try:
            backend = self.env.backend
            backend.refresh_surface_for_evaluation()
            vertices, triangles = backend.surface_arrays
            face_ids = np.full(len(triangles), -1, np.int32)
            for row in backend.surface_info['component_ranges']:
                face_ids[row['first_triangle']:row['first_triangle']+row['triangles']] = row['mechanical_component']
            np.savez_compressed(rec/'surface_state.npz', surface_vertices_world=vertices,
                surface_triangles=triangles, surface_triangle_component_ids=face_ids)
        except Exception as error:
            write_json(rec/'surface_quality_error.json', dict(status='NOT_RUN', error=repr(error)))
        if not self.env._recorder_finished:
            self.env._refresh_cameras(force=True)
            self.env.finish_recording(dict(status='RECORDED', reason=reason, **self.provenance))
        record_memory(self.memory_log, 'recording_finalized', simulation_time_s=self.time_s)
        # Evaluate and preserve results BEFORE SimulationApp.close may exit.
        result = independent_evaluate(rec, self.evaluator_python)
        record_memory(self.memory_log, 'independent_evaluation_finished',
                      evaluation_status=result.get('status'), success=bool(result.get('success')))
        write_json(rec/'evaluation.json', result)
        self.attempts[-1].update(status=result['status'], task_success=bool(result.get('success')))
        write_json(self.output/'recording'/'attempts.json', self.attempts)
        self.finished = True
        if result.get('success'):
            write_json(rec/'success_media_pending.json', dict(status='SUCCESS_MEDIA_PENDING', **self.provenance))
            from .agentic_trim_media import export_success
            try:
                export_success(rec)
            except Exception as error:
                write_json(rec/'success_media_error.json', dict(status='SUCCESS_MEDIA_PENDING', error=repr(error)))


@contextmanager
def open_env(*, output_dir, seed, policy_seed=0, sim_seed=42, distribution='id', evaluator_python=None,
             stream_physics=False, memory_receipts=False, native_sensor_runtime=None):
    """Entry used by a GPU1-only launcher; never selects a GPU by itself."""
    from .live_env import open_cutting_env
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    memory_log = output/'runtime_memory.jsonl' if memory_receipts else None
    record_memory(memory_log, 'before_scene_compile')
    try:
        task, manifest = compile_seed(seed, policy_seed=policy_seed, sim_seed=sim_seed,
                                     distribution=distribution, output_dir=output/'scene')
    except SceneInfeasible as error:
        write_json(output/'scene_infeasible.json', dict(status='SCENE_INFEASIBLE', scene_seed=seed, reason=str(error)))
        raise
    record_memory(memory_log, 'scene_compiled', particle_count=manifest['particle_count'])
    options = {'history_mode': 'latest'} if stream_physics else {}
    if native_sensor_runtime is not None:
        options['native_sensor_runtime'] = native_sensor_runtime
    previous_log = os.environ.get('RCS_RUNTIME_MEMORY_LOG')
    if memory_log:
        os.environ['RCS_RUNTIME_MEMORY_LOG'] = str(memory_log)
    try:
        with open_cutting_env(runtime='isaac_sim', output_dir=output/'native', task=task, seed=sim_seed,
                preset='cutting_research_v1', camera_rig_profile='wrist_camera_rig_v14', global_rgb_only=True,
                active_camera_ids=CAMERAS, device='cuda:0', dx=.004, material_dt=.0001,
                surface_update='on_capture', history_stride=10, **options,
                preset_overrides=dict(camera_every=10, gripper_control_mode='hold')) as env:
            adapter = AgenticTrimEnv(env, output, task, manifest, seed=seed,
                evaluator_python=evaluator_python, stream_physics=stream_physics, memory_log=memory_log)
            try:
                yield adapter
            except BaseException as error:
                import traceback
                write_json(output/'controller_error.json', dict(error=f'{type(error).__name__}: {error}',
                                                               traceback=traceback.format_exc()))
                raise
            finally:
                adapter.finalize('WORKER_EXIT')
    finally:
        if memory_log:
            if previous_log is None:
                os.environ.pop('RCS_RUNTIME_MEMORY_LOG', None)
            else:
                os.environ['RCS_RUNTIME_MEMORY_LOG'] = previous_log
