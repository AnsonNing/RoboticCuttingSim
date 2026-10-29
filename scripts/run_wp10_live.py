#!/usr/bin/env python3
"""Recorded WP-10 validation on a real Isaac runtime (never a replay backend)."""
import argparse
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.control import ActionLimits, DELTA_MODE
from robotic_cutting_sim.live_env import open_cutting_env
from robotic_cutting_sim.policy_config import policy_preset
from robotic_cutting_sim.recording import manifest, new_output, write_json


def jsonable(value):
    return json.loads(json.dumps(value, default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                                else v.item() if isinstance(v, np.generic) else str(v)))


@contextmanager
def failure_receipt(output, env=None):
    # Native Kit may exit Python during app.close(); write failures before closing it.
    try:
        yield
    except BaseException as error:
        write_json(output / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        write_json(output / 'result.json', dict(status='FAIL_EXECUTION', error=repr(error)))
        if env is not None:
            backend = env.backend
            try:
                for name in ('grasp_initialization', 'grasp_settling_trace', 'grasp_reset_receipt'):
                    value = getattr(backend, name, None)
                    if value is not None:
                        write_json(output / ('failure_' + name + '.json'), jsonable(value))
                tool = getattr(backend, 'dynamic_tool', None)
                if tool is not None:
                    tool.save_raw_trace(output / 'failure_tool_substeps.npz')
            except Exception as capture_error:
                (output / 'failure_trace_capture_error.txt').write_text(repr(capture_error))
        traceback.print_exc()
        raise


class EpisodeRecorder:
    """Raw RGB-D, poses, commands and actual material history, with fresh names."""
    def __init__(self, output, env, controller, every=10):
        self.output, self.env, self.controller, self.every = Path(output), env, controller, every
        self.output.mkdir(exist_ok=False)
        (self.output / 'observations').mkdir()
        self.rows, self.samples, self.states, self.errors, self.rotation_errors = [], [], [], [], []
        self.body_states = [self.body_poses()]
        self.events = dict(rejected=0, clamped=0, cutting_steps=0)
        self.steps = 0
        self.phases = []  # (input time_s, scripted policy phase label) per control step
        self.stream = (self.output / 'actions.jsonl').open('w')
        config = getattr(env.backend, 'options', getattr(env.backend, 'config', None))
        write_json(self.output / 'manifest.json', manifest(jsonable(dict(
            controller=controller, runtime_source=env.runtime_source, config=asdict(config),
            task_id=env.task.task_id, seed=env.task.seed)), 'wp10_actual_live_episode'))
        self.capture()

    def measured(self):
        backend = self.env.backend
        return backend.measured() if hasattr(backend, 'measured') else backend.proprioception()

    def body_poses(self):
        backend = self.env.backend
        return backend.body_poses().copy() if hasattr(backend, 'body_poses') else backend.measured()['body_pose'].copy()

    def capture(self):
        env = self.env
        backend = env.backend
        index = len(self.rows)
        frames = env._frames
        if not frames and getattr(getattr(backend, 'config', None), 'camera', False):
            frames, _ = backend.capture()
        # State episodes may record no cameras; this is explicit in every receipt.
        for name, frame in frames.items():
            public = {key: value for key, value in frame.items() if not key.startswith('rtx_reference_')
                      and not key.startswith('depth_sensor_') and key != 'native_source_index_debug'}
            timestamp = float(frame.get('physics_timestamp_s', frame.get('timestamp_s', float('nan'))))
            if abs(timestamp - env.time_s) > 1e-9 or not np.isfinite(timestamp):
                raise ValueError('Only current, measured camera frames may be recorded')
            public['physics_timestamp_s'] = timestamp
            np.savez_compressed(self.output / 'observations' / f'{name}_{index:04d}.npz', **public)
        overview = getattr(backend, 'overview_frame', None) if frames else None
        if overview is not None:
            # Presentation-only RGB (never an observation); same measured-time rule.
            timestamp = float(overview['physics_timestamp_s'])
            if abs(timestamp - env.time_s) > 1e-9 or not np.isfinite(timestamp):
                raise ValueError('Only current overview frames may be recorded')
            np.savez_compressed(self.output / 'observations' / f'overview_rgb_{index:04d}.npz',
                                rgb=overview['rgb'], physics_timestamp_s=timestamp)
        closeup = getattr(backend, 'closeup_frame', None) if frames else None
        if closeup is not None:
            # Presentation-only close-up RGB (never an observation); same measured-time rule.
            timestamp = float(closeup['physics_timestamp_s'])
            if abs(timestamp - env.time_s) > 1e-9 or not np.isfinite(timestamp):
                raise ValueError('Only current close-up frames may be recorded')
            np.savez_compressed(self.output / 'observations' / f'closeup_rgb_{index:04d}.npz',
                                rgb=closeup['rgb'], physics_timestamp_s=timestamp)
        self.rows.append(dict(frame=index, time_s=env.time_s, controller=self.controller,
                              camera_calibration=jsonable(backend.calibration),
                              cut_enabled=getattr(self, 'last_cut', False)))
        state = backend.material.history_state()
        self.samples.append({key: state[key].copy() for key in ('positions', 'velocities', 'codes')})
        self.states.append(jsonable(self.measured()))
        write_json(self.output / 'frame_manifest.json', self.rows)

    def step(self, action, result, before_observation=None, step_wall_s=None, policy_phase=None,
             controller_profile=None, policy_tracking=None, policy_vision=None):
        self.steps += 1
        if policy_phase is not None:
            self.phases.append(((before_observation or {}).get('timestamp_s', self.env.time_s - self.env.task.control_dt_s),
                                policy_phase))
        self.body_states.append(self.body_poses())
        observation, reward, terminated, truncated, info = result
        report = info.get('action_report') or {}
        target = report.get('resolved_target_world_xyz_xyzw', info.get('requested_reference_target_world_xyz_xyzw'))
        if target is None:
            target = self.env._commanded
        measured = self.measured()
        target = np.asarray(target, float)
        actual = np.asarray(measured['tool_pose'], float)
        error = float(np.linalg.norm(actual[:3] - target[:3]))
        rotation_error = float((Rotation.from_quat(actual[3:]) * Rotation.from_quat(target[3:]).inv()).magnitude())
        self.errors.append(error)
        self.rotation_errors.append(rotation_error)
        self.last_cut = action is None or bool(action.get('cut_enabled', True))
        self.events['cutting_steps'] += self.last_cut
        self.events['rejected'] += info.get('ik_status') == 'rejected'
        self.events['clamped'] += any(report.get(key, False) for key in
                                      ('translation_clamped', 'rotation_clamped', 'workspace_clamped'))
        diagnostics = self.env.backend.last_diagnostics
        row = dict(step=self.steps, time_s=self.env.time_s, controller=self.controller,
                   submitted=action, info=info, measured=measured, reward=reward,
                   terminated=terminated, truncated=truncated, tracking_error_m=error,
                   rotation_error_rad=rotation_error,
                   material_diagnostics=diagnostics.get('material', diagnostics.get('material_diagnostics', {})),
                   privileged_diagnostics=diagnostics,
                   observation_proprio=self.env.observe().get('proprio', {}),
                   input_observation_proprio=(before_observation or {}).get('proprio', {}),
                   input_timestamp_s=(before_observation or {}).get('timestamp_s'), step_wall_s=step_wall_s,
                   grasp_status=diagnostics.get('grasp_status'),
                   resolved_target_world_xyz_xyzw=target,
                   accepted_target_world_xyz_xyzw=self.env._commanded)
        if policy_phase is not None:
            row['policy_phase'] = policy_phase
        if controller_profile is not None:
            row['controller_profile'] = controller_profile
        if policy_tracking is not None:
            row['policy_tracking'] = policy_tracking
        if policy_vision is not None:
            row['policy_vision'] = policy_vision
        self.stream.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
        self.stream.flush()
        if self.steps % self.every == 0 or terminated or truncated:
            self.capture()

    def push_interval(self, label='push'):
        """Actual [start, end] time of the scripted ``label`` phase, its completion and source."""
        times = [time for time, phase in self.phases if phase == label]
        if not times:
            return None
        start = times[0]
        later = [time for time, phase in self.phases if time > start and phase != label]
        end = later[0] if later else float(self.env.time_s)
        return dict(interval_s=[float(start), float(end)], completed=bool(later),
                    source='scripted policy phase labels (input time of the first push action to the first later non-push step)')

    def phase_arrays(self):
        if not self.phases:
            return {}
        arrays = dict(policy_phase_time_s=np.asarray([time for time, _ in self.phases], float),
                      policy_phases=np.asarray([phase for _, phase in self.phases], str))
        push = self.push_interval()
        if push is not None:
            arrays.update(push_interval_s=np.asarray(push['interval_s'], float), push_completed=push['completed'])
        return arrays

    def finish(self, result):
        self.stream.close()
        env, material = self.env, self.env.backend.material
        if self.rows[-1]['time_s'] != env.time_s:
            self.capture()
        samples = self.samples
        np.savez_compressed(self.output / 'physics_state.npz',
            time_s=[row['time_s'] for row in self.rows],
            **{key: np.asarray([s[key] for s in samples]) for key in ('positions', 'velocities', 'codes')},
            mass_kg=material.masses, reference_positions=material.world_points,
            reference_positions_local=material.points, reference_indices=material.lattice,
            reference_stencil_nodes=material.bed.nodes.numpy(),
            reference_stencil_weights=material.bed.weights.numpy(),
            reference_stencil_gradients=material.bed.gradients.numpy(), **self.phase_arrays())
        write_json(self.output / 'robot_state.json', self.states)
        backend = env.backend
        body_paths = getattr(backend, 'rigid_link_paths', None)
        if body_paths is None:
            body_paths = list(backend.model.body_label)
        np.savez_compressed(self.output / 'robot_state.npz',
            time_s=np.arange(len(self.body_states)) * env.task.control_dt_s,
            body_pose=self.body_states, body_prim_paths=body_paths)
        trace = getattr(backend, 'camera_pose_trace', [])
        if trace:
            write_json(self.output / 'camera_pose_trace.json', dict(samples=trace))
            rig = getattr(backend, 'camera_specs', None)
            if rig is None:
                rig = backend.camera_rig.specs
            write_json(self.output / 'camera_rig.json', dict(specs=rig))
            stage = getattr(backend, 'stage', None)
            if stage is None:
                stage = backend.sim.stage
            stage.Export(str((self.output / 'final_live_scene.usda').resolve()))
            write_json(self.output / 'setup.json', dict(body_labels=body_paths))
        write_json(self.output / 'task.json', env.task.to_dict())
        write_json(self.output / 'material.json', material.parameter_receipt)
        metrics = dict(steps=self.steps, events=self.events,
                       tracking_error_m={key: float(value) for key, value in zip(
                           ('p50', 'p95', 'max'), np.percentile(self.errors or [0.], [50, 95, 100]))},
                       rotation_error_rad={key: float(value) for key, value in zip(
                           ('p50', 'p95', 'max'), np.percentile(self.rotation_errors or [0.], [50, 95, 100]))})
        write_json(self.output / 'result.json', {**result, **metrics})
        return metrics


def fuzz_action(step, rng, pattern='balanced'):
    # Fixed requests independent of runtime responses, including unreachable requests,
    # saturated translations and flips. ActionLimits must bound all executed targets.
    if step % 100 < 10:
        delta = np.r_[rng.normal(0, 2., 3), [0., np.pi * (-1 if pattern == 'balanced' and step % 2 else 1), 0.]]
    elif step % 100 < 70:
        angle = .003 * (-1 if step // 10 % 2 else 1) if pattern == 'balanced' else .2
        delta = np.r_[[0., 0., .0001], [0., angle, 0.]]
    else:
        delta = np.r_[rng.normal(0, .0002, 3), rng.normal(0, .01, 3)]
    return dict(mode=DELTA_MODE, target=delta, frame='world', cut_enabled=False, cut_index=0)


def episode_step_limit(task_id):
    """Control steps in one task episode (v1/v2: 8 s -> 400; SLICE-PUSH-01: 16 s -> 800)."""
    from robotic_cutting_sim.tasks import make_task
    task = make_task(task_id)
    return int(round(task.duration_s / task.control_dt_s))


def validate_arguments(p, a):
    try:
        limit = episode_step_limit(a.task)
    except (ValueError, KeyError) as error:
        p.error(f'Unknown task {a.task!r}: {error}')
    if not 1 <= a.steps <= limit or a.episodes < 1:
        p.error(f'A {a.task} episode has 1..{limit} steps')
    if a.overview_camera and (a.runtime != 'isaac_sim' or not (a.record_cameras or a.observation != 'state')):
        p.error('--overview-camera is Native-only and needs recorded cameras (--record-cameras or a vision observation)')
    if (a.edge_feedback != 'nominal' or a.position_tolerance is not None) and a.controller != 'scripted':
        p.error('--edge-feedback/--position-tolerance apply to the scripted controller only')
    if a.controller_profile != 'v1' and a.controller != 'scripted':
        p.error('--controller-profile applies to the scripted controller only')
    from robotic_cutting_sim.policies import vision_survey_options
    survey = vision_survey_options(a.controller_profile)
    if survey is not None and (a.task not in survey['task_ids'] or not (a.record_cameras or a.observation != 'state')):
        p.error(f'--controller-profile {a.controller_profile} (ADR 0009 wrist-depth survey) is declared for '
                f'{list(survey["task_ids"])} only and needs rendered cameras (--record-cameras or a vision observation)')
    if a.position_tolerance is not None and not 0 < a.position_tolerance < .01:
        p.error('--position-tolerance must be in (0, 0.01) m')
    from robotic_cutting_sim.tasks import DemoTaskSpec, make_task
    if a.offset and (a.controller != 'scripted' or a.task in DemoTaskSpec.TASK_IDS
                     or make_task(a.task).goal_kind == 'flatten' or not abs(a.offset) <= .05):
        p.error('--offset (|offset| <= 0.05 m) shifts the scripted through_cut line; not for slice_push or skim')
    if a.task in DemoTaskSpec.TASK_IDS and a.controller == 'scripted' and (a.motion != 'push' or a.angle):
        p.error(f'{a.task} uses the slice_push plan; --motion/--angle variants are not implemented for it')


def is_demo_task(task):
    from robotic_cutting_sim.tasks import DemoTaskSpec
    return isinstance(task, DemoTaskSpec)


def policy_plan_name(task, policy=None):
    if getattr(policy, 'vision', None):
        return 'slice_push_vision'
    return 'slice_push' if is_demo_task(task) else 'skim' if task.goal_kind == 'flatten' else 'through_cut'


def edge_feedback_receipt(a):
    if a.edge_feedback == 'nominal':
        return dict(mode='nominal', source='nominal_hand_mount_edge_from_proprioception')
    return dict(mode='measured', source='measured_tool_edge_feedback',
                scope='Simulator tool-edge pose of the physically held knife (privileged tool proprioception); '
                      'not material oracle state and not part of the policy observation')


def measured_edge_estimator(env):
    """WORLD edge XYZ+XYZW of the actually held tool at the observation time."""
    backend = env.backend

    def estimate(observation):
        measured = backend.measured() if hasattr(backend, 'measured') else backend.proprioception()
        stamp = float(measured.get('timestamp_s', env.time_s))
        if abs(stamp - float(observation['timestamp_s'])) > 1e-9:
            raise ValueError('Measured tool-edge feedback must match the observation time')
        return np.asarray(measured['tool_pose'], float).copy()
    return estimate


def wrist_depth_source(env):
    """Latest rendered depth observation of one camera at the current physics time (ADR 0009 survey input).

    Only depth, its valid mask, K and T_world_camera (robot kinematics) are returned; with a state
    observation profile the frame is rendered on demand (rendering holds physics). The timestamp
    is the frame's own measured physics time, checked by the policy against its observation.
    """
    backend = env.backend

    def latest(camera_id):
        frames = env._frames
        if camera_id not in frames:
            frames = backend.frames if hasattr(backend, 'measured') else backend.capture()[0]
        frame = frames[camera_id]
        calibration = backend.calibration[camera_id]
        timestamp = float(frame.get('physics_timestamp_s', frame.get('timestamp_s', float('nan'))))
        if abs(timestamp - env.time_s) > 1e-9:
            raise ValueError(f'{camera_id} frame at {timestamp} s is not current ({env.time_s} s)')
        return dict(camera_id=camera_id, depth_m=np.asarray(frame['depth_m']).copy(),
                    valid_mask=np.asarray(frame['valid_mask']).copy(),
                    K=np.asarray(frame.get('K', calibration['K']), float).copy(),
                    T_world_camera=np.asarray(frame.get('T_world_camera', calibration['T_world_camera']), float).copy(),
                    physics_timestamp_s=timestamp)
    return latest


def controller_profile_receipt(a):
    """Scripted controller profile name and parameters (v1 = the frozen point-to-point controller)."""
    from robotic_cutting_sim.policies import CONTROLLER_PROFILES
    receipt = dict(CONTROLLER_PROFILES[a.controller_profile])
    if a.position_tolerance is not None:
        receipt['position_tolerance_m'] = a.position_tolerance  # explicit --position-tolerance wins
    return receipt


def scripted_policy(env, a):
    from robotic_cutting_sim.policies import (WaypointPolicy, controller_profile, skim, slice_push, through_cut,
                                              vision_policy, vision_survey_options)
    profile_policy, profile_plan = controller_profile(getattr(a, 'controller_profile', 'v1'))
    if vision_survey_options(getattr(a, 'controller_profile', 'v1')) is not None:
        options = {}
        if a.edge_feedback == 'measured':
            options['edge_estimator'] = measured_edge_estimator(env)
        if a.position_tolerance is not None:
            options['position_tolerance_m'] = a.position_tolerance
        return vision_policy(env.task, a.controller_profile, depth_source=wrist_depth_source(env),
                             limits=env.action_limits or ActionLimits(), cut_speed_m_s=a.speed, **options)
    saw = .006 if a.motion == 'saw' else 0.
    if is_demo_task(env.task):
        plan = slice_push(env.task, cut_speed_m_s=a.speed, **profile_plan)
    elif env.task.goal_kind == 'flatten':
        plan = skim(env.task, saw_amplitude_m=saw, **profile_plan)
    else:
        motion_args = {'saw_amplitude_m': saw, 'cut_speed_m_s': a.speed, **profile_plan}
        if getattr(a, 'offset', 0.):
            motion_args['offset_m'] = a.offset
        if a.motion in ('oblique', 'yaw', 'roll'):
            motion_args[{'oblique':'tilt_deg','yaw':'yaw_deg','roll':'roll_deg'}[a.motion]] = a.angle
        plan = through_cut(env.task, **motion_args)
    options = dict(profile_policy)
    if a.edge_feedback == 'measured':
        options['edge_estimator'] = measured_edge_estimator(env)
    if a.position_tolerance is not None:
        options['position_tolerance_m'] = a.position_tolerance
    return WaypointPolicy(env.task, plan, env.action_limits or ActionLimits(), **options)


def evaluate_episode(env, push=None):
    """(strict, research, demo). Demo tasks: frozen oracle NOT_APPLICABLE, multi-cut research, demo report."""
    if not is_demo_task(env.task):
        return env.evaluate(), env.evaluate(profile='research'), None
    strict = {**env.task.evaluate_oracle({}), 'success': False, 'runtime_source': env.runtime_source}
    research = env.evaluate(profile='research', goal=env.task.multi_cut_goal())
    demo = env.evaluate(push_interval_s=None if push is None else push['interval_s'])
    return strict, research, demo


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', choices=('isaac_sim', 'isaac_lab'), required=True)
    p.add_argument('--task', default='CUT-01')
    p.add_argument('--preset', choices=('legacy', 'policy'), default='legacy')
    p.add_argument('--tool-attachment-mode', choices=('legacy_fixed_tool', 'physical_grasp'))
    p.add_argument('--controller', choices=('reference', 'scripted', 'fuzz', 'gym_hold', 'reject_probe'), default='reference')
    p.add_argument('--fuzz-pattern', choices=('balanced', 'stress'), default='balanced')
    p.add_argument('--observation', choices=('state', 'vision_proprio'), default='state')
    p.add_argument('--gym', action='store_true')
    p.add_argument('--record-cameras', action='store_true', help='Record RGB-D with a state-only controller')
    p.add_argument('--scenario', choices=('raised_workpiece_50mm_v1', 'ruler_workcell_v2', 'blademaster_workcell_v1'))
    p.add_argument('--overview-camera', action='store_true',
                   help='Native only: author and record the presentation-only overview and close-up RGB cameras '
                        '(never observations)')
    p.add_argument('--edge-feedback', choices=('nominal', 'measured'), default='nominal',
                   help='Scripted controller edge source: nominal hand-mount edge (default) or the measured '
                        'simulator tool-edge pose (measured_tool_edge_feedback; not material oracle state)')
    p.add_argument('--position-tolerance', type=float,
                   help='Scripted waypoint position tolerance in m (default: the controller profile; v1 0.5 mm)')
    p.add_argument('--controller-profile', choices=('v1', 'path_v2', 'path_v2_vision'), default='v1',
                   help='Scripted controller profile: v1 (default, frozen point-to-point waypoints), path_v2 '
                        '(opt-in: cut segments tracked from the planned start with cross-track feedback 0.5, '
                        'final 10 mm approach at 0.02 m/s, 0.3 s settle, 0.3 mm position tolerance) or '
                        'path_v2_vision (SLICE-PUSH-01 only, needs cameras: path_v2 plus the ADR 0009 wrist D405 '
                        'depth survey that shifts cut 1 and later waypoints by the measured part shift)')
    p.add_argument('--material-profile')
    p.add_argument('--material-dt', type=float, default=.0001)
    p.add_argument('--max-cuts', type=int, default=1)
    p.add_argument('--cutting-law', type=int, choices=(1, 2, 3, 4, 5), help='None keeps the task default (v2 tasks: law 2)')
    p.add_argument('--ik-iterations', type=int, default=48)
    p.add_argument('--oracle-stride', type=int, default=1)
    p.add_argument('--motion', choices=('push', 'saw', 'oblique', 'yaw', 'roll'), default='push')
    p.add_argument('--angle', type=float, default=0.)
    p.add_argument('--speed', type=float, default=.01)
    p.add_argument('--offset', type=float, default=0.,
                   help='Scripted through_cut only: shift the cut line along the local target normal (m; default 0)')
    p.add_argument('--reward', action='store_true')
    p.add_argument('--grip-force', type=float)
    p.add_argument('--settle-seconds', type=float)
    p.add_argument('--grip-friction', type=float)
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--steps', type=int, default=400)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--spacing', type=float)
    p.add_argument('--record-every', type=int, default=10)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    validate_arguments(p, a)
    out = new_output(a.out)
    write_json(out / 'manifest.json', manifest(jsonable(vars(a)), 'wp10_actual_live_validation'))
    results, start = [], time.monotonic()
    options = policy_preset(a.runtime) if a.preset == 'policy' else {}
    if a.tool_attachment_mode is not None:
        options['tool_attachment_mode'] = a.tool_attachment_mode
        if a.runtime == 'isaac_lab' and a.tool_attachment_mode == 'legacy_fixed_tool':
            options.pop('grip_contact_profile', None)
    options.update(particle_spacing_m=a.spacing, camera=a.observation != 'state' or a.record_cameras)
    if a.overview_camera:
        options['overview_camera'] = True
    options.update(scenario=a.scenario, material_profile_id=a.material_profile, material_dt=a.material_dt,
                   max_cuts=a.max_cuts, ik_iterations=a.ik_iterations, history_stride=a.oracle_stride,
                   cutting_law=a.cutting_law)
    grip_overrides = {key: value for key, value in dict(max_finger_force_n=a.grip_force,
        settle_s=a.settle_seconds, friction=a.grip_friction).items() if value is not None}
    if grip_overrides:
        from robotic_cutting_sim.physical_tool import ToolGraspConfig
        options['grasp'] = replace(options.get('grasp') or ToolGraspConfig(), **grip_overrides)
    if a.runtime == 'isaac_sim':
        options.update(gpu=a.gpu, device=f'cuda:{a.gpu}')
    else:
        options.update(device=f'cuda:{a.gpu}', material_device=f'cuda:{a.gpu}')
    try:
        with open_cutting_env(a.task, runtime=a.runtime, output_dir=out / 'runtime', seed=a.seed,
                              observation_profile=a.observation, **options) as env, failure_receipt(out, env):
            if a.reward:
                from robotic_cutting_sim.reward import CuttingReward
                env.reward = CuttingReward(env.task)
            for episode in range(a.episodes):
                seed, episode_start = a.seed + episode, time.monotonic()
                if a.gym or a.controller == 'gym_hold':
                    from robotic_cutting_sim.gym_env import CuttingGymEnv
                    gym = CuttingGymEnv(env, seed=seed)
                    observation = env.observe()
                else:
                    gym = None
                    observation, _ = env.reset(seed=seed)
                recorder = EpisodeRecorder(out / f'episode_{episode:03d}', env, a.controller, a.record_every)
                before = env.time_s
                try:
                    env.step(dict(mode=DELTA_MODE, target=np.zeros(6), cut_index=env.backend.material.max_cuts))
                except ValueError as error:
                    cut_index_check = dict(status='PASS' if env.time_s == before else 'FAIL', error=str(error))
                else:
                    cut_index_check = dict(status='FAIL', reason='Unavailable digit was accepted')
                policy, profile_receipt = None, None
                if a.controller == 'scripted':
                    policy = scripted_policy(env, a)
                    profile_receipt = controller_profile_receipt(a)
                rng = np.random.default_rng(seed)
                for step in range(a.steps):
                    action = (fuzz_action(step, rng, a.fuzz_pattern) if a.controller == 'fuzz' else
                              policy(observation) if policy else
                              dict(mode=DELTA_MODE, target=np.zeros(6), cut_enabled=False) if gym else None)
                    saved_limits = env.action_limits
                    before_observation = env.observe()
                    step_start = time.monotonic()
                    if a.controller == 'reject_probe':
                        # Separate IK-layer probe, explicitly disabling absolute clamping
                        # for an impossible request; the default fuzz retains ActionLimits().
                        env.action_limits = replace(env.action_limits or ActionLimits(), limit_absolute=False)
                        action = dict(mode='cartesian_edge_pose_world',
                                      target=np.r_[[1000., 1000., 1000.], env._commanded[3:]], cut_enabled=False)
                    if gym:
                        limits = env.action_limits
                        gym.cut_enabled = action['cut_enabled']
                        vector = np.r_[action['target'][:3] / limits.max_translation_m,
                                       action['target'][3:] / limits.max_rotation_rad]
                        result = gym.step(vector)
                        observation = env.observe()
                    else:
                        result = env.step(action)
                        observation = result[0]
                    env.action_limits = saved_limits
                    recorder.step(action, result, before_observation, time.monotonic()-step_start,
                                  policy_phase=policy.phase if policy is not None else None,
                                  controller_profile=profile_receipt,
                                  policy_tracking=policy.last_tracking if policy is not None else None,
                                  policy_vision=getattr(policy, 'last_vision', None))
                    if (step + 1) % 25 == 0:
                        print(f'{a.runtime} {a.task} {a.controller} episode={episode} step={step+1} t={env.time_s:.2f}', flush=True)
                    if result[2] or result[3]:
                        break
                push = recorder.push_interval()
                strict, research, demo = evaluate_episode(env, push)
                row = dict(runtime=a.runtime, task_id=a.task, controller=a.controller, seed=seed,
                           status='PASS_EXECUTION', task_status=(demo or strict).get('status'), strict=strict, research=research,
                           cut_index_check=cut_index_check, runtime_source=env.runtime_source,
                           observation_profile=a.observation, wall_s=time.monotonic() - episode_start)
                if demo is not None:
                    row.update(demo=demo, push_phase=push)
                if policy is not None:
                    if getattr(policy, 'vision', None):
                        from robotic_cutting_sim.policies import ideal_plan_end_s
                        row.update(policy_vision=policy.vision, policy_plan_ideal_end_s=ideal_plan_end_s(
                            env.task, list(policy.waypoints), path_tracking=policy.path_tracking,
                            cross_track_gain=policy.cross_track_gain, position_tolerance_m=policy.position_tolerance_m))
                    row.update(policy_plan=policy_plan_name(env.task, policy), edge_feedback=edge_feedback_receipt(a),
                               policy_position_tolerance_m=policy.position_tolerance_m,
                               policy_final_phase=policy.phase, controller_profile=profile_receipt,
                               policy_motion=dict(motion=a.motion, angle_deg=a.angle, offset_m=a.offset, speed_m_s=a.speed))
                row.update(recorder.finish(row))
                results.append(row)
                write_json(out / 'results.json', results)
            write_json(out / 'result.json', dict(status='PASS_EXECUTION', rows=results,
                                                wall_s=time.monotonic() - start))
        status = 'PASS_EXECUTION'
    except BaseException as error:
        status = 'FAIL_EXECUTION'
        write_json(out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
        traceback.print_exc()
    write_json(out / 'result.json', dict(status=status, rows=results, wall_s=time.monotonic() - start))
    return 0 if status == 'PASS_EXECUTION' else 1


if __name__ == '__main__':
    sys.exit(main())
