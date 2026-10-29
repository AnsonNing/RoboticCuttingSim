#!/usr/bin/env python3
"""Reproducible runner for the BladeMaster demos (ADR 0007 section 6, demo plan sections 4-5).

Each demo is one live Native episode of ``scripts/run_wp10_live.py``
(Isaac Sim PhysX/Kit/RTX, physical two-finger grasp, scripted waypoint
controller, scenario ``blademaster_workcell_v1``). The runner then evaluates the
episode read-only, renders it with ``scripts/render_demo_episode.py`` and can
publish small media. The demos are:

  1a  CUT-01-R vertical push cut, 400 steps        (Demo 1, complete cutting)
  1b  CUT-01-R 20 deg oblique cut, 400 steps       (Demo 1, oblique variant)
  2   TRIM-01-R skim at the scenario 35 deg rake    (Demo 2, surface trimming)
  3   SLICE-PUSH-01, two cuts + blade-face push     (Demo 3, 800 steps)

Layout (never overwritten; a repeated evaluate/render/publish stage writes a
new ``_rNNN`` revision next to the earlier one)::

  <root>/<run_id>/index.json                  summary of every demo of the run
  <root>/<run_id>/logs/<demo>_run.log|.json   physics stdout and run receipt
  <root>/<run_id>/<demo>/                     run_wp10_live --out (manifest, result.json, episode_000/)
  <root>/<run_id>/<demo>/run_receipt.json     exit code, wall time, exact command
  <root>/<run_id>/<demo>/demo_summary.json    execution / outcome / quality layers
  <root>/<run_id>/<demo>/render/              render_demo_episode output

Usage (recommended: reproduce the final demo set with one command)::

  .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v1 --dry-run
  .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v1 --gpu 1 --run-id <id>
  .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v1 --run-id <id> --stage evaluate --stage render
  .venv/bin/python scripts/run_blademaster_demos.py --demo-set final_v1 --run-id <id> --stage publish

``--demo-set final_v1`` is the final BladeMaster demo set: demos 1a, 1b and 2
with controller profile ``path_v2`` and demo 3 with ``path_v2_vision``, all
with cutting law 5 and seed 42 -- the configurations of the recorded final
runs ``live_20261029_pathv2_d`` (1a, 1b), ``live_20261029_pathv2_c`` (2) and
``live_20261029_vision_e`` (3). The profile is chosen per demo and recorded
per demo in the run receipt, the demo summary and the index. ``--demo-set``
is mutually exclusive with ``--demo`` and ``--controller-profile``; pass it to
every stage of the run (evaluate checks each demo against its set profile).

First-round / single-demo usage (``--demo``, frozen v1 definitions by default)::

  .venv/bin/python scripts/run_blademaster_demos.py --demo all --dry-run
  .venv/bin/python scripts/run_blademaster_demos.py --demo all --gpu 1 --run-id <id>
  .venv/bin/python scripts/run_blademaster_demos.py --demo 2 --run-id <id> --stage evaluate --stage render
  .venv/bin/python scripts/run_blademaster_demos.py --demo all --run-id <id> --stage publish
  .venv/bin/python scripts/run_blademaster_demos.py --demo 2 --gpu 1 --run-id <id> --controller-profile path_v2

``--demo all`` with the default ``--controller-profile v1`` reproduces the
first-round v1 configuration (kept for backward compatibility), not the final
set. ``--controller-profile`` (default ``v1``, the frozen demo definitions)
selects one scripted controller profile of run_wp10_live for every selected
demo; ``path_v2`` is the opt-in path-tracking controller; ``path_v2_vision``
(Demo 3 only, ADR 0009) adds the wrist-depth survey that retargets cut 1 (its
measurements and applied shift are summarised under
``execution.policy_vision``). Pass the same value to every stage of a run: the
evaluate stage checks the recorded profile against it.

``--stage all`` (the default) is run -> evaluate -> render, plus publish with
``--publish``. Evaluation never modifies the episode (its files are hashed
before and after). The demo outcome is reported separately from the frozen v1
oracle and from the precision score; nothing here changes physics or thresholds.
"""
import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.policies import CONTROLLER_PROFILES, vision_survey_options  # noqa: E402

SCRIPTS = ROOT / 'scripts'
DEFAULT_ROOT = ROOT / 'artifacts' / 'wp10_claude' / 'demos'
DEFAULT_MEDIA_ROOT = ROOT / 'docs' / 'media' / 'demos'
DEFAULT_NATIVE_PYTHON = ROOT / '.deps' / 'isaacsim-venv' / 'bin' / 'python'
DEFAULT_RENDER_PYTHON = ROOT / '.venv' / 'bin' / 'python'
DEFAULT_TMPDIR = '/mnt/data/ningan/tmp'
SCENARIO = 'blademaster_workcell_v1'
CONTROL_MODE = 'scripted waypoint controller, Native PhysX/Kit/RTX, physical two-finger grasp'
LIVE_RUNTIME = 'live_isaac_sim'
SUMMARY_SCHEMA = 'blademaster_demo_summary_v1'
COMMON_FLAGS = ('--preset', 'policy', '--tool-attachment-mode', 'physical_grasp', '--controller', 'scripted',
                '--scenario', SCENARIO, '--record-cameras', '--overview-camera', '--record-every', '5',
                '--edge-feedback', 'measured')
# run_wp10_live manifest config keys every demo episode must carry (cutting law/gpu/seed reported, not required).
COMMON_CONFIG = dict(runtime='isaac_sim', preset='policy', tool_attachment_mode='physical_grasp', controller='scripted',
                     scenario=SCENARIO, record_cameras=True, overview_camera=True, record_every=5,
                     edge_feedback='measured', observation='state')
SURFACE_CHECKS = ('maximum_substrate_surface_residual_height_m', 'surface_p95_absolute_plane_error_m',
                  'surface_rms_plane_error_m', 'maximum_gouge_depth_m', 'overcut_volume_fraction_of_initial_base',
                  'missing_surface_fraction', 'surface_has_explicit_mechanical_provenance',
                  'surface_timestamp_matches_final_physics', 'actual_substrate_surface_present')


@dataclass(frozen=True)
class DemoSpec:
    key: str
    dirname: str
    kind: str          # complete_cut | oblique_cut | trim | slice_push
    task_id: str
    steps: int
    extra: tuple       # run_wp10_live arguments specific to the demo
    config: tuple      # expected manifest config (key, value) pairs specific to the demo
    title: str


DEMOS = {
    '1a': DemoSpec('1a', 'demo1a', 'complete_cut', 'CUT-01-R', 400, ('--motion', 'push'),
                   (('task', 'CUT-01-R'), ('motion', 'push'), ('angle', 0.), ('max_cuts', 1), ('steps', 400)),
                   'Demo 1 — Complete Cutting (CUT-01-R, vertical push cut)'),
    '1b': DemoSpec('1b', 'demo1b', 'oblique_cut', 'CUT-01-R', 400, ('--motion', 'oblique', '--angle', '20'),
                   (('task', 'CUT-01-R'), ('motion', 'oblique'), ('angle', 20.), ('max_cuts', 1), ('steps', 400)),
                   'Demo 1 — Complete Cutting, 20° oblique variant (CUT-01-R)'),
    '2': DemoSpec('2', 'demo2', 'trim', 'TRIM-01-R', 400, (),
                  (('task', 'TRIM-01-R'), ('motion', 'push'), ('angle', 0.), ('max_cuts', 1), ('steps', 400)),
                  'Demo 2 — Surface Trimming (TRIM-01-R, 35° rake skim)'),
    '3': DemoSpec('3', 'demo3', 'slice_push', 'SLICE-PUSH-01', 800, ('--max-cuts', '2'),
                  (('task', 'SLICE-PUSH-01'), ('motion', 'push'), ('angle', 0.), ('max_cuts', 2), ('steps', 800)),
                  'Demo 3 — Sequential Slicing & Blade-Face Push (SLICE-PUSH-01)'),
}
ORDER = ('1a', '1b', '2', '3')
STAGES = ('run', 'evaluate', 'render', 'publish')


@dataclass(frozen=True)
class DemoSet:
    """A versioned, fixed demo set: per-demo controller profiles plus the pinned cutting law and seed."""
    name: str
    profiles: tuple    # ((demo key, controller profile), ...) in ORDER
    cutting_law: int
    seed: int
    reference_runs: tuple  # ((demo key, recorded run id), ...) this set reproduces
    description: str

    @property
    def demos(self):
        return tuple(key for key, _ in self.profiles)

    def profile(self, key):
        return dict(self.profiles)[key]

    def as_dict(self):
        return {'name': self.name, 'demos': list(self.demos), 'controller_profiles': dict(self.profiles),
                'cutting_law': self.cutting_law, 'seed': self.seed, 'reference_runs': dict(self.reference_runs),
                'description': self.description}


DEMO_SETS = {
    'final_v1': DemoSet('final_v1', (('1a', 'path_v2'), ('1b', 'path_v2'), ('2', 'path_v2'), ('3', 'path_v2_vision')),
                        5, 42, (('1a', 'live_20261029_pathv2_d'), ('1b', 'live_20261029_pathv2_d'),
                                ('2', 'live_20261029_pathv2_c'), ('3', 'live_20261029_vision_e')),
                        'Final BladeMaster demo set: path_v2 for demos 1a/1b/2, path_v2_vision (ADR 0009) for '
                        'demo 3, cutting law 5, seed 42'),
}
V1_ALL_NOTICE = ('NOTICE: --demo all with --controller-profile v1 (the default): this reproduces the first-round v1 '
                 'configuration; use --demo-set final_v1 for the final set')


class StageError(RuntimeError):
    """A stage could not be carried out (missing input, refusal, failed subprocess)."""


# ----------------------------------------------------------------------------- small helpers

def clean(value):
    """JSON-safe copy: numpy -> Python, NaN/inf -> None, Path -> str."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_new_json(path, data):
    """Exclusive create: an existing file is never overwritten."""
    path = Path(path)
    with path.open('x') as stream:
        json.dump(clean(data), stream, indent=2, allow_nan=False)
        stream.write('\n')
    return path


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def revision_path(path, revision):
    path = Path(path)
    if revision == 1:
        return path
    suffix = f'_r{revision:03d}'
    return path.with_name(path.stem + suffix + path.suffix) if path.suffix else path.with_name(path.name + suffix)


def next_free(path):
    """``path`` when unused, else the first unused ``<stem>_rNNN<suffix>`` revision."""
    revision = 1
    while revision_path(path, revision).exists():
        revision += 1
    return revision_path(path, revision)


def latest_existing(path):
    """Highest existing revision of ``path`` (``path`` itself is revision 1) or None."""
    revision, found = 1, None
    while revision_path(path, revision).exists():
        found, revision = revision_path(path, revision), revision + 1
    return found


def revision_suffix(path, base):
    name, stem = Path(path).name, Path(base).stem if Path(base).suffix else Path(base).name
    tail = name[len(stem):]
    tail = tail[:-len(Path(base).suffix)] if Path(base).suffix and tail.endswith(Path(base).suffix) else tail
    return tail  # '' for revision 1, '_r002' ...


def git_info():
    def run(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip()
    return dict(sha=run('rev-parse', 'HEAD'), dirty=bool(run('status', '--porcelain')),
                branch=run('rev-parse', '--abbrev-ref', 'HEAD'))


def load_script(name):
    spec = importlib.util.spec_from_file_location(f'_bm_{name}', SCRIPTS / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def deep(mapping, *keys, default=None):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return default
        mapping = mapping[key]
    return mapping


# ----------------------------------------------------------------------------- commands

def profile_flags(controller_profile):
    """run_wp10_live flags of a scripted controller profile (v1, the default, adds none)."""
    if controller_profile not in CONTROLLER_PROFILES:
        raise ValueError(f'unknown controller profile {controller_profile!r}')
    return () if controller_profile == 'v1' else ('--controller-profile', controller_profile)


def run_argv(spec, demo_dir, *, python=DEFAULT_NATIVE_PYTHON, gpu=1, cutting_law=5, seed=42, controller_profile='v1'):
    """Exact run_wp10_live command line of one demo."""
    return [str(python), str(SCRIPTS / 'run_wp10_live.py'), '--runtime', 'isaac_sim', '--task', spec.task_id,
            *COMMON_FLAGS, '--cutting-law', str(cutting_law), *spec.extra, *profile_flags(controller_profile),
            '--steps', str(spec.steps), '--seed', str(seed), '--gpu', str(gpu), '--out', str(demo_dir)]


def render_argv(spec, episode_dir, out, label, *, python=DEFAULT_RENDER_PYTHON):
    return [str(python), str(SCRIPTS / 'render_demo_episode.py'), str(episode_dir), '--out', str(out),
            '--title', spec.title, '--label', label]


def expected_config(spec, controller_profile='v1', cutting_law=None):
    """Manifest config a demo must carry; ``cutting_law`` is only checked when a demo set pins it."""
    config = {**COMMON_CONFIG, **dict(spec.config), 'controller_profile': controller_profile}
    if cutting_law is not None:
        config['cutting_law'] = cutting_law
    return config


def demo_set_of(options):
    name = getattr(options, 'demo_set', None)
    return None if name is None else DEMO_SETS[name]


def profile_of(options, key):
    """Controller profile of one demo: the demo set's per-demo profile, else the single --controller-profile."""
    demo_set = demo_set_of(options)
    return options.controller_profile if demo_set is None else demo_set.profile(key)


def cutting_law_of(options):
    demo_set = demo_set_of(options)
    return options.cutting_law if demo_set is None else demo_set.cutting_law


def seed_of(options):
    demo_set = demo_set_of(options)
    return options.seed if demo_set is None else demo_set.seed


def recorded_config(config, key):
    """Recorded manifest value; recordings made before --controller-profile existed ran the v1 controller."""
    value = config.get(key)
    return 'v1' if key == 'controller_profile' and value is None else value


def shell_line(argv, tmpdir):
    return f'cd {shlex.quote(str(ROOT))} && TMPDIR={shlex.quote(str(tmpdir))} {shlex.join(argv)}'


# ----------------------------------------------------------------------------- episode loading

EPISODE_FILES = ('result.json', 'physics_state.npz', 'actions.jsonl', 'frame_manifest.json', 'task.json',
                 'manifest.json', 'robot_state.json', 'robot_state.npz')


def episode_hashes(episode):
    return {name: sha256_file(episode / name) for name in EPISODE_FILES if (episode / name).is_file()}


def load_actions(path):
    if not Path(path).is_file():
        return []
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_physics(path):
    with np.load(path) as data:
        return {key: data[key] for key in data.files if key != 'velocities'}


def reconstruct_task(task_json, config):
    """The scenario-placed task from its id/seed/spacing; checked against the recorded task.json."""
    from robotic_cutting_sim.task_scenarios import configure_task
    from robotic_cutting_sim.tasks import make_task
    task = configure_task(make_task(task_json['task_id'], seed=int(task_json['seed']),
                                    particle_spacing_m=float(task_json['particle_spacing_m'])),
                          scenario=config.get('scenario'), material_profile_id=config.get('material_profile'))
    checks = {
        'world_from_workpiece_matches': bool(np.allclose(task.world_from_workpiece_xyzw,
                                                         task_json['world_from_workpiece_xyzw'], atol=1e-12)),
        'edge_orientation_matches': bool(np.allclose(task.edge_orientation_local_xyzw,
                                                     task_json['edge_orientation_local_xyzw'], atol=1e-12)),
        'fixture_box_count_matches': len(task.fixture_boxes_local) == len(task_json.get('fixture_boxes_local') or ()),
        'target_matches': bool(np.allclose(task.target_origin_local_m, task_json['target_origin_local_m'], atol=1e-12)
                               and np.allclose(task.target_normal_local, task_json['target_normal_local'], atol=1e-12))}
    return task, checks


def measured_capture_rate(observations, n_frames):
    from robotic_cutting_sim import demo_media as dm
    streams = dm.discover_streams(p.name for p in observations.glob('*.npz')) if observations.is_dir() else {}
    if not streams:
        return dict(frames=0, rate_hz=None, interval_s=None, source='no recorded camera streams', streams={})
    panels = dm.choose_panels(streams)
    stream = panels['large'] or next(iter(streams))
    stamps = []
    for index in range(n_frames):
        path = observations / f'{stream}_{index:04d}.npz'
        if path.is_file():
            with np.load(path) as data:
                stamps.append(float(data['physics_timestamp_s']) if 'physics_timestamp_s' in data.files else np.nan)
    rate = dm.measured_rate(stamps)
    rate.update(source=f'observations/{stream} physics_timestamp_s', streams={k: len(v) for k, v in streams.items()},
                overview_recorded=panels['overview'] is not None)
    return rate


def row_diagnostics(row):
    return row.get('privileged_diagnostics') or {}


def grasp_summary(rows):
    lost_steps, statuses, slips = [], {}, []
    for row in rows:
        pd = row_diagnostics(row)
        status = row.get('grasp_status') or pd.get('grasp_status') or {}
        if isinstance(status, str):
            status = {'status': status}
        lost = bool(pd.get('grasp_lost', False) or status.get('grasp_lost', False))
        if lost:
            lost_steps.append(int(row.get('step', -1)))
        name = str(status.get('status', 'n/a'))
        statuses[name] = statuses.get(name, 0) + 1
        if status.get('slip_since_settling_m') is not None:
            slips.append(float(status['slip_since_settling_m']))
    final = (rows[-1].get('grasp_status') or row_diagnostics(rows[-1]).get('grasp_status') or {}) if rows else {}
    return {'grasp_lost': bool(lost_steps), 'first_lost_step': lost_steps[0] if lost_steps else None,
            'lost_steps': len(lost_steps), 'status_counts': statuses,
            'final_status': final.get('status') if isinstance(final, dict) else final,
            'max_slip_since_settling_m': max(slips) if slips else None,
            'source': 'actions.jsonl grasp_status / privileged_diagnostics.grasp_lost (operational grasp monitor)'}


# ----------------------------------------------------------------------------- layers

def execution_layer(spec, demo_dir, episode, loaded, controller_profile='v1', demo_set=None):
    manifest, top, row, receipt = loaded['manifest'], loaded['top_result'], loaded['row'], loaded['run_receipt']
    config = (manifest or {}).get('config') or {}
    expected = expected_config(spec, controller_profile, None if demo_set is None else demo_set.cutting_law)
    recorded = {key: recorded_config(config, key) for key in expected}
    mismatched = {key: {'expected': value, 'recorded': recorded[key]} for key, value in expected.items()
                  if not (recorded[key] == value or (isinstance(value, float)
                          and isinstance(recorded[key], (int, float)) and abs(recorded[key] - value) < 1e-12))}
    rows, physics, frames = loaded['actions'], loaded['physics'], loaded['frame_manifest']
    grasp = grasp_summary(rows)
    events = row.get('events') or {}
    missing = [] if receipt is not None else ['run_receipt.json (exit code/wall time of the physics process)']
    checks = {'run_wp10_live_status_pass': (top or {}).get('status') == 'PASS_EXECUTION',
              'episode_status_pass': row.get('status') == 'PASS_EXECUTION',
              'all_requested_steps_executed': row.get('steps') == spec.steps,
              'live_native_runtime': row.get('runtime_source') == LIVE_RUNTIME,
              'command_matches_demo_definition': not mismatched,
              'task_reconstruction_matches_task_json': all(loaded['task_checks'].values()),
              'physics_frames_match_frame_manifest': physics is not None and frames is not None
                  and len(physics['time_s']) == len(frames),
              'action_rows_match_steps': len(rows) == row.get('steps'),
              'grasp_never_lost': bool(rows) and not grasp['grasp_lost']}
    if receipt is not None:
        checks['physics_process_exit_code_zero'] = receipt.get('exit_code') == 0
    status = 'PASS_EXECUTION' if all(checks.values()) else 'FAIL_EXECUTION'
    return {'status': status, 'checks': checks, 'missing_evidence': missing,
            'exit_code': None if receipt is None else receipt.get('exit_code'),
            'timed_out': None if receipt is None else receipt.get('timed_out'),
            'steps_executed': row.get('steps'), 'steps_requested': spec.steps,
            'grasp': grasp, 'ik_rejections': events.get('rejected'), 'action_clamps': events.get('clamped'),
            'cut_enabled_steps': events.get('cutting_steps'),
            'tracking_error_m': row.get('tracking_error_m'), 'rotation_error_rad': row.get('rotation_error_rad'),
            'wall_time_s': {'runner_process': None if receipt is None else receipt.get('wall_s'),
                            'run_wp10_live_total': (top or {}).get('wall_s'), 'episode': row.get('wall_s')},
            'runtime_source': row.get('runtime_source'), 'policy_plan': row.get('policy_plan'),
            'policy_final_phase': row.get('policy_final_phase'), 'edge_feedback': row.get('edge_feedback'),
            'policy_motion': row.get('policy_motion'), 'cut_index_check': row.get('cut_index_check'),
            'controller_profile': {'expected': controller_profile, 'recorded': recorded['controller_profile'],
                                   'receipt': row.get('controller_profile'),
                                   'position_tolerance_m': row.get('policy_position_tolerance_m')},
            'policy_vision': vision_summary(row),
            'manifest_mismatches': mismatched,
            'task_reconstruction_checks': loaded['task_checks']}


def vision_summary(row):
    """ADR 0009 survey receipt of a path_v2_vision episode (None for profiles without a survey)."""
    vision = row.get('policy_vision')
    if not vision:
        return None
    estimate = vision.get('estimate') or {}
    stamps = {kind: {key: (vision.get(kind) or {}).get(key) for key in ('observation_timestamp_s', 'frame_timestamp_s',
                                                                      'valid', 'reason')}
              for kind in ('calibration', 'survey')}
    return {'method': vision.get('method'), 'camera_id': vision.get('camera_id'), 'valid': estimate.get('valid'),
            'reasons': estimate.get('reasons'), 'shift_estimate_m': estimate.get('shift_m'),
            'applied_shift_m': vision.get('applied_shift_m'), 'applied_shift_world_m': vision.get('applied_shift_world_m'),
            'shifted_labels': vision.get('shifted_labels'), 'calibration_face_x_m': estimate.get('calibration_face_x_m'),
            'survey_face_x_m': estimate.get('survey_face_x_m'), 'frames': stamps,
            'diagnostics': {name: {key: record.get(key) for key in ('valid', 'reasons', 'shift_m')}
                            for name, record in (vision.get('diagnostics') or {}).items()},
            'plan_ideal_end_s': row.get('policy_plan_ideal_end_s'), 'adr': vision.get('adr')}


class Components:
    """mechanical_components memoised per code vector of one recording."""

    def __init__(self, physics, cuts):
        self.physics, self.cuts, self.cache = physics, cuts, {}

    def __call__(self, codes):
        from robotic_cutting_sim.cut_topology import mechanical_components
        codes = np.ascontiguousarray(codes)
        key = hashlib.sha1(codes.tobytes()).hexdigest()
        if key not in self.cache:
            p = self.physics
            self.cache[key] = mechanical_components(p['reference_stencil_nodes'], p['reference_stencil_weights'],
                                                    codes, self.cuts, gradients=p['reference_stencil_gradients'])
        return self.cache[key]


def strict_summary(strict):
    strict = strict or {}
    keys = ('final_components', 'substantial_parts', 'two_main_parts_mass_fraction', 'minor_parts_mass_fraction',
            'maximum_reference_midpoint_plane_distance_m', 'postcut_persistence_time_s', 'postcut_graph_count_minimum',
            'removed_reference_above_plane_mass_fraction', 'surface')
    metrics = strict.get('metrics') or {}
    return {'status': strict.get('status'), 'success': strict.get('success'), 'checks': strict.get('checks'),
            'metrics': {k: metrics[k] for k in keys if k in metrics}, 'missing_evidence': strict.get('missing_evidence'),
            'source': 'episode_000/result.json strict (frozen v1 oracle on the live in-memory history, as recorded)'}


def research_summary(research):
    research = research or {}
    return {'status': research.get('status'), 'success': research.get('success'),
            'cutting_outcome_status': deep(research, 'cutting_outcome', 'status'),
            'profile_assessment': research.get('profile_assessment'),
            'source': 'episode_000/result.json research (Evaluation v2 research profile, as recorded)'}


def withdrawal_and_persistence(rows, physics, components, minimum_persistence_s):
    """Edge withdrawn above the initial material top after the last cut-enabled step; separation afterwards."""
    active = [row for row in rows if row_diagnostics(row).get('active_cut')]
    top_z = float(np.asarray(physics['reference_positions'], float)[:, 2].max())
    result = {'active_cut_steps': len(active), 'material_top_world_z_m': top_z,
              'last_active_cut_time_s': float(active[-1]['time_s']) if active else None,
              'withdrawal_time_s': None, 'frames_after_withdrawal': 0, 'components_after_withdrawal': [],
              'persistence_after_withdrawal_s': 0., 'minimum_persistence_s': minimum_persistence_s,
              'definition': 'withdrawal = first control step after the last active cut whose measured edge '
                            'centre is above the highest initial reference particle'}
    if not active:
        return result
    last = result['last_active_cut_time_s']
    for row in rows:
        pose = row_diagnostics(row).get('measured_tool_pose') or deep(row, 'measured', 'tool_pose')
        if row['time_s'] > last + 1e-9 and pose is not None and float(pose[2]) >= top_z:
            result['withdrawal_time_s'] = float(row['time_s'])
            break
    if result['withdrawal_time_s'] is None:
        return result
    time_s = np.asarray(physics['time_s'], float)
    after = np.flatnonzero(time_s >= result['withdrawal_time_s'] - 1e-9)
    counts = [int(components(physics['codes'][i])[0]) for i in after]
    result.update(frames_after_withdrawal=int(len(after)), components_after_withdrawal=counts,
                  persistence_after_withdrawal_s=float(time_s[after[-1]] - result['withdrawal_time_s']) if len(after) else 0.)
    return result


def cut_outcome(spec, task, loaded):
    """Demo 1 outcome (two parts, mass, persistent separation after withdrawal, no tool drop) + quality."""
    from robotic_cutting_sim.demo_evaluation import label_distance_distribution, swept_plane_interface
    physics, rows, row = loaded['physics'], loaded['actions'], loaded['row']
    protocol = task.protocol_document()
    limits, common = protocol[task.task_id], protocol['common']
    components = Components(physics, 1)
    count, labels = components(physics['codes'][-1])
    masses = np.asarray(physics['mass_kg'], float)
    fractions = np.sort(np.bincount(labels, weights=masses, minlength=count) / masses.sum())[::-1]
    _, expected_mass, _ = task.sample_particles()
    expected_total = float(np.sum(expected_mass))
    mass_error = abs(float(masses.sum()) - expected_total)
    persistence = withdrawal_and_persistence(rows, physics, components, common['minimum_postcut_topology_persistence_s'])
    grasp = grasp_summary(rows)
    substantial = int(np.count_nonzero(fractions >= limits['minimum_substantial_part_mass_fraction_each']))
    checks = {'two_substantial_parts': substantial == limits['required_substantial_parts'],
              'two_main_parts_retain_mass': float(fractions[:2].sum()) >= limits['minimum_two_main_parts_total_mass_fraction'],
              'reference_mass_budget': mass_error <= max(common['mass_absolute_tolerance_kg'],
                                                         common['mass_relative_tolerance'] * expected_total),
              'withdrawn_after_cut': persistence['withdrawal_time_s'] is not None,
              'separation_persists_after_withdrawal': persistence['frames_after_withdrawal'] >= 1
                  and min(persistence['components_after_withdrawal'], default=0) >= 2
                  and persistence['persistence_after_withdrawal_s'] >= persistence['minimum_persistence_s'] - 1e-9,
              'no_tool_drop': bool(rows) and not grasp['grasp_lost']}
    R = task.T_world_workpiece[:3, :3]
    target_origin = task.points_to_world(np.asarray(task.target_origin_local_m, float))
    target_normal = R @ np.asarray(task.target_normal_local, float)
    reference = np.asarray(physics['reference_positions'], float)
    location = swept_plane_interface(physics['reference_stencil_nodes'], physics['reference_stencil_weights'],
                                     physics['reference_stencil_gradients'], labels, reference, target_origin, target_normal)
    strict = row.get('strict') or {}
    frozen_location = {'check_cut_at_workpiece_local_target_plane': deep(strict, 'checks', 'cut_at_workpiece_local_target_plane'),
                       'recorded_maximum_reference_midpoint_plane_distance_m':
                           deep(strict, 'metrics', 'maximum_reference_midpoint_plane_distance_m'),
                       'threshold_m': limits['maximum_severed_reference_stencil_midpoint_plane_distance_m'],
                       'recomputed_from_physics_state_m': location['maximum_reference_midpoint_plane_distance_m'],
                       'scope': 'Frozen v1 location statistic to the task target plane; reported, not part of the outcome'}
    metrics = {'final_mechanical_components': int(count), 'final_component_mass_fractions': fractions[:6].tolist(),
               'substantial_parts': substantial, 'two_main_parts_mass_fraction': float(fractions[:2].sum()),
               'recorded_mass_kg': float(masses.sum()), 'expected_task_sampler_mass_kg': expected_total,
               'reference_mass_absolute_error_kg': mass_error, 'withdrawal_and_persistence': persistence,
               'grasp_lost': grasp['grasp_lost']}
    quality = {'location': frozen_location, 'frozen_strict': strict_summary(strict),
               'research_profile': research_summary(row.get('research')),
               'label_distance_to_target_plane': label_distance_distribution(reference, physics['codes'], target_origin,
                                                                             target_normal, digit=0)}
    if spec.kind == 'oblique_cut':
        swept = swept_plane_outcome(task, physics, rows)
        metrics['swept_plane'] = swept
        checks['swept_plane_interface_le_ideal_plus_one_spacing'] = deep(
            swept, 'rescore', 'checks', 'interface_le_ideal_plus_one_spacing') is True
        checks['no_far_particle_in_opposite_side_component'] = deep(
            swept, 'rescore', 'checks', 'no_far_particle_in_opposite_side_component') is True
        quality['location']['scope'] = ('Frozen v1 location check against the vertical task target plane, reported '
                                        'as is; the oblique outcome uses the measured swept plane instead')
        if swept.get('fit') is not None:
            angle = math.degrees(math.acos(min(1., abs(float(np.dot(swept['fit']['normal'], target_normal))))))
            quality['swept_plane_tilt_from_target_plane_deg'] = angle
        quality['swept_plane_original_2mm'] = deep(swept, 'rescore', 'original_2mm')
    status = 'PASS' if all(checks.values()) else 'FAIL'
    fr = fractions[:2] * 100
    parts = f"{count} component{'' if count == 1 else 's'}" + (f' {fr[0]:.0f}/{fr[1]:.0f}%' if len(fr) == 2 else '')
    label = (f'outcome {status} ({parts}, '
             f"{'separated after withdrawal' if checks['separation_persists_after_withdrawal'] else 'NOT separated after withdrawal'}, "
             f"{'grasp held' if checks['no_tool_drop'] else 'TOOL DROPPED'}"
             + (f", swept-plane {'ok' if checks['swept_plane_interface_le_ideal_plus_one_spacing'] and checks['no_far_particle_in_opposite_side_component'] else 'FAIL'}"
                if spec.kind == 'oblique_cut' else '')
             + f") | frozen strict {strict.get('status')}")
    protocol_name = ('demo1_complete_cut_v1' if spec.kind == 'complete_cut' else
                     'demo1_oblique_cut_adr0006_amendment1_v1')
    return {'protocol': protocol_name, 'status': status, 'checks': {k: bool(v) for k, v in checks.items()},
            'metrics': metrics, 'thresholds': {**{k: limits[k] for k in limits if k.startswith(('minimum', 'required'))},
                                               'minimum_postcut_topology_persistence_s': common['minimum_postcut_topology_persistence_s']},
            'scope': 'Mechanical components of the actual stencil graph (weights + stress gradients) recomputed from '
                     'physics_state.npz; oracle evaluation only, never an observation'}, quality, label


def swept_plane_outcome(task, physics, rows):
    """ADR 0006 Amendment 1 rule against the plane fitted to MEASURED edge poses during the active cut."""
    from scipy.spatial.transform import Rotation
    from robotic_cutting_sim.demo_evaluation import fit_swept_plane
    active = [row for row in rows if row_diagnostics(row).get('active_cut')]
    poses = [row_diagnostics(row).get('measured_tool_pose') for row in active]
    poses = np.asarray([p for p in poses if p is not None], float)
    base = {'source': 'actions.jsonl privileged_diagnostics.measured_tool_pose at steps with active_cut',
            'active_cut_steps': len(active), 'pose_samples': int(len(poses)),
            'material_contact_steps': int(sum(1 for row in active
                                              if np.size(row_diagnostics(row).get('material_contact_statistics', [])) > 1
                                              and np.asarray(row_diagnostics(row)['material_contact_statistics']).ravel()[1] > 0)),
            'rule': 'ADR 0006 Amendment 1: actual interface <= ideal-geometric-label interface + one particle spacing; '
                    'no particle beyond two spacings in the opposite-side substantial component'}
    if len(poses) < 2:
        return {**base, 'status': 'NOT_RUN', 'fit': None, 'rescore': None,
                'reason': 'Fewer than two measured edge poses during the active cut'}
    rotations = Rotation.from_quat(poses[:, 3:])
    lines, normals = rotations.apply([0., 1., 0.]), rotations.apply([0., 0., 1.])
    try:
        fit = fit_swept_plane(poses[:, :3], lines, edge_half_length_m=task.blade_half_width_m,
                              reference_normal=normals.mean(0))
    except ValueError as error:
        return {**base, 'status': 'NOT_RUN', 'fit': None, 'rescore': None, 'reason': str(error)}
    rescore = load_script('rescore_phase1_oblique').rescore_m2(
        physics, np.asarray(fit['origin'], float), np.asarray(fit['normal'], float), task.particle_spacing_m, cuts=1)
    return {**base, 'status': 'FIT', 'fit': fit, 'rescore': rescore,
            'rescore_source': 'scripts/rescore_phase1_oblique.py rescore_m2 (imported unchanged)'}


def trim_outcome(spec, task, loaded):
    from robotic_cutting_sim.demo_evaluation import trim_separation_for_task
    physics, row = loaded['physics'], loaded['row']
    separation = trim_separation_for_task(task, physics)
    strict = row.get('strict') or {}
    strict_checks = strict.get('checks') or {}
    surface = {name: bool(strict_checks[name]) for name in SURFACE_CHECKS if name in strict_checks}
    passed = sum(surface.values())
    precision = {'status': 'NOT_RUN' if not surface else 'PASS_PRECISION' if passed == len(surface) else 'FAIL_PRECISION',
                 'checks': surface, 'passed': passed, 'total': len(surface),
                 'score': None if not surface else passed / len(surface),
                 'surface_metrics': deep(strict, 'metrics', 'surface'),
                 'frozen_strict_status': strict.get('status'),
                 'scope': 'Frozen v1 surface precision checks (0.5 mm profile) reported as is; never part of the '
                          'Demo 2 outcome verdict'}
    status = 'PASS' if separation['status'] == 'PASS_TRIM_SEPARATION' else 'FAIL'
    metrics = separation['metrics']
    removed, share = metrics['removed_reference_above_plane_mass_fraction'], metrics['largest_component_below_plane_mass_share']
    label = (f'outcome {status} (above-plane removed '
             f"{'n/a' if removed is None else f'{100 * removed:.1f}%'}, substrate "
             f"{'n/a' if share is None else f'{100 * share:.1f}%'}) | precision "
             f"{precision['passed']}/{precision['total']} ({precision['status']})")
    quality = {'precision_score': precision, 'frozen_strict': strict_summary(strict),
               'research_profile': research_summary(row.get('research'))}
    outcome = {'protocol': 'demo2_trim_separation_v1', 'status': status, 'checks': separation['checks'],
               'metrics': metrics, 'thresholds': separation['thresholds'], 'trim_separation': separation,
               'scope': 'demo_evaluation.trim_separation_for_task on physics_state.npz (final frame)'}
    return outcome, quality, label


def slice_push_outcome(spec, task, loaded):
    physics, row = loaded['physics'], loaded['row']
    report = task.evaluate_demo(physics)
    push_completed = bool(physics['push_completed']) if 'push_completed' in physics else \
        bool(deep(row, 'push_phase', 'completed', default=False))
    checks = {'slice_push_report_pass': report.get('status') == 'PASS_SLICE_PUSH',
              'push_phase_completed': push_completed}
    status = 'PASS' if all(checks.values()) else 'FAIL'
    manipulation = report.get('post_cut_manipulation') or {}
    pushed = deep(manipulation, 'metrics', 'pushed_displacement_along_push_m')
    label = (f"outcome {status} ({report.get('status')}, components "
             f"{deep(manipulation, 'metrics', 'components_at_push_start', default='n/a')}, pushed "
             f"{'n/a' if pushed is None else f'{1e3 * pushed:.1f} mm'}, push "
             f"{'completed' if push_completed else 'NOT completed'})")
    recorded_live = row.get('demo') or {}
    quality = {'label_distance_to_declared_planes': report.get('label_distance_to_declared_planes'),
               'multi_cut_plane_errors': deep(report, 'multi_cut', 'metrics'),
               'recorded_live_demo_status': recorded_live.get('status'),
               'recorded_live_demo_agrees': recorded_live.get('status') == report.get('status'),
               'research_profile': research_summary(row.get('research')),
               'frozen_strict': {'status': deep(row, 'strict', 'status'),
                                 'scope': 'Frozen two-part v1 oracle is NOT_APPLICABLE to a three-part task'}}
    outcome = {'protocol': 'demo3_slice_push_v1', 'status': status, 'checks': checks, 'slice_push_report': report,
               'push_phase': {'completed': push_completed,
                              'interval_s': physics['push_interval_s'].tolist() if 'push_interval_s' in physics
                              else deep(row, 'push_phase', 'interval_s'),
                              'source': 'physics_state.npz push_interval_s/push_completed (scripted phase labels)'},
               'scope': 'task.evaluate_demo (evaluate_multi_cut + post_cut_manipulation) on physics_state.npz'}
    return outcome, quality, label


OUTCOMES = {'complete_cut': cut_outcome, 'oblique_cut': cut_outcome, 'trim': trim_outcome, 'slice_push': slice_push_outcome}


def evaluate_demo_dir(spec, demo_dir, controller_profile='v1', demo_set=None):
    """Read-only evaluation of one demo directory -> summary dict (nothing is written here).

    ``controller_profile`` is the profile the demo definition expects (the
    recorded manifest must match it; recordings without the key are v1).
    ``demo_set`` (a DemoSet or its name) additionally requires the set's
    per-demo profile and pinned cutting law.
    """
    if isinstance(demo_set, str):
        demo_set = DEMO_SETS[demo_set]
    if demo_set is not None and demo_set.profile(spec.key) != controller_profile:
        raise ValueError(f'demo set {demo_set.name} declares {demo_set.profile(spec.key)!r} for demo {spec.key}, '
                         f'not {controller_profile!r}')
    demo_dir = Path(demo_dir)
    episode = demo_dir / 'episode_000'
    start = time.perf_counter()
    base = {'schema': SUMMARY_SCHEMA, 'demo': spec.key, 'title': spec.title, 'task_id': spec.task_id,
            'demo_kind': spec.kind, 'control_mode': CONTROL_MODE, 'demo_dir': str(demo_dir.resolve()),
            'episode_dir': str(episode.resolve()), 'evaluator': {'script': str(Path(__file__).resolve()),
            'script_sha256': sha256_file(__file__), 'git': git_info(), 'command': sys.argv},
            'read_only': True}
    manifest = read_json(demo_dir / 'manifest.json')
    receipt = read_json(demo_dir / 'run_receipt.json')
    config = (manifest or {}).get('config') or {}
    base['cutting_law'] = config.get('cutting_law')
    base['controller_profile'] = {'expected': controller_profile,
                                  'recorded': recorded_config(config, 'controller_profile') if manifest else None,
                                  'parameters': CONTROLLER_PROFILES.get(controller_profile)}
    base['demo_set'] = None if demo_set is None else demo_set.as_dict()
    base['run_receipt'] = receipt
    if not (episode / 'result.json').is_file() or not (episode / 'physics_state.npz').is_file():
        failure = read_json(demo_dir / 'failure.json')
        execution = {'status': 'FAIL_EXECUTION' if (demo_dir.is_dir() or receipt) else 'NOT_RUN',
                     'exit_code': None if receipt is None else receipt.get('exit_code'),
                     'missing_evidence': [str(episode / 'result.json'), str(episode / 'physics_state.npz')],
                     'failure': failure, 'top_level_result': read_json(demo_dir / 'result.json')}
        return {**base, 'verdict': 'NOT_RUN' if execution['status'] == 'NOT_RUN' else 'FAIL',
                'execution': execution, 'outcome': {'status': 'NOT_RUN', 'reason': 'No complete episode recording'},
                'quality': {}, 'label': f"{execution['status']} (no episode recording)",
                'evaluation_wall_s': time.perf_counter() - start}
    before = episode_hashes(episode)
    row = read_json(episode / 'result.json')
    task_json = read_json(episode / 'task.json')
    loaded = {'manifest': manifest, 'top_result': read_json(demo_dir / 'result.json'), 'row': row,
              'run_receipt': receipt, 'actions': load_actions(episode / 'actions.jsonl'),
              'physics': load_physics(episode / 'physics_state.npz'),
              'frame_manifest': read_json(episode / 'frame_manifest.json')}
    task, task_checks = reconstruct_task(task_json, config)
    loaded['task_checks'] = task_checks
    execution = execution_layer(spec, demo_dir, episode, loaded, controller_profile, demo_set)
    if all(task_checks.values()):
        outcome, quality, label = OUTCOMES[spec.kind](spec, task, loaded)
    else:
        outcome, quality = {'status': 'NOT_RUN', 'reason': 'Recorded task.json does not match the scenario task'}, {}
        label = 'outcome NOT_RUN (task mismatch)'
    after = episode_hashes(episode)
    if after != before:
        raise AssertionError('The episode changed during evaluation')
    n_frames = len(loaded['frame_manifest'] or [])
    verdict = ('PASS' if execution['status'] == 'PASS_EXECUTION' and outcome['status'] == 'PASS' else
               'NOT_RUN' if outcome['status'] == 'NOT_RUN' and execution['status'] == 'PASS_EXECUTION' else 'FAIL')
    if execution['status'] != 'PASS_EXECUTION':
        label = f"{label} | execution {execution['status']}"
    if controller_profile != 'v1':
        label = f"{label} | controller {controller_profile}"
    return {**base, 'verdict': verdict,
            'verdict_rule': 'PASS only when execution is PASS_EXECUTION and the demo outcome is PASS; '
                            'quality (frozen strict oracle, precision, research profile) never gates the verdict',
            'runtime': {'runtime_source': row.get('runtime_source'), 'native_live': row.get('runtime_source') == LIVE_RUNTIME,
                        'observation_profile': row.get('observation_profile')},
            'execution': execution, 'outcome': outcome, 'quality': quality, 'label': label,
            'measured_capture_rate': measured_capture_rate(episode / 'observations', n_frames),
            'artifacts': {'demo_dir': str(demo_dir.resolve()), 'episode_dir': str(episode.resolve()),
                          'files': {name: {'path': str((episode / name).resolve()), 'sha256': digest}
                                    for name, digest in before.items()},
                          'top_level_result_sha256': sha256_file(demo_dir / 'result.json')
                          if (demo_dir / 'result.json').is_file() else None,
                          'run_log': None if receipt is None else receipt.get('log')},
            'episode_unchanged_by_evaluation': True, 'evaluation_wall_s': time.perf_counter() - start,
            'material_note': 'Material parameters are provisional and uncalibrated'}


# ----------------------------------------------------------------------------- stages

def demo_dir_of(options, spec):
    return Path(options.root) / options.run_id / spec.dirname


def gpu_snapshot():
    query = ['nvidia-smi', '--query-gpu=index,name,memory.used,memory.total,utilization.gpu', '--format=csv,noheader']
    try:
        result = subprocess.run(query, capture_output=True, text=True, timeout=20)
        return {'command': query, 'returncode': result.returncode, 'stdout': result.stdout.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': query, 'error': repr(error)}


def stage_run(options, spec):
    demo_dir = demo_dir_of(options, spec)
    if demo_dir.exists():
        raise StageError(f'refusing to overwrite existing episode directory {demo_dir}')
    logs = Path(options.root) / options.run_id / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    log, receipt_path = logs / f'{spec.dirname}_run.log', logs / f'{spec.dirname}_run.json'
    if log.exists() or receipt_path.exists():
        raise StageError(f'refusing to overwrite existing run log {log}')
    profile = profile_of(options, spec.key)
    argv = run_argv(spec, demo_dir, python=options.native_python, gpu=options.gpu,
                    cutting_law=cutting_law_of(options), seed=seed_of(options), controller_profile=profile)
    env = {**os.environ, 'TMPDIR': str(options.tmpdir)}
    receipt = {'demo': spec.key, 'command': argv, 'shell': shell_line(argv, options.tmpdir), 'cwd': str(ROOT),
               'controller_profile': profile, 'demo_set': getattr(options, 'demo_set', None),
               'tmpdir': str(options.tmpdir), 'log': str(log), 'git': git_info(), 'gpu_before': gpu_snapshot(),
               'started_unix_s': time.time(), 'timeout_s': options.timeout}
    print(f'[{spec.key}] run: {receipt["shell"]}', flush=True)
    start = time.monotonic()
    with log.open('x') as stream:
        try:
            process = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        except OSError as error:
            receipt.update(exit_code=None, timed_out=False, wall_s=time.monotonic() - start, launch_error=repr(error))
            write_new_json(receipt_path, receipt)
            raise StageError(f'could not launch the physics process: {error!r}') from error
        try:
            code, timed_out = process.wait(timeout=options.timeout), False
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)  # our own process group only
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
    receipt.update(exit_code=code, timed_out=timed_out, wall_s=time.monotonic() - start, finished_unix_s=time.time())
    write_new_json(receipt_path, receipt)
    if demo_dir.is_dir():
        write_new_json(demo_dir / 'run_receipt.json', receipt)
    print(f'[{spec.key}] run finished: exit={code} timed_out={timed_out} wall={receipt["wall_s"]:.0f}s', flush=True)
    if code != 0:
        raise StageError(f'physics process exited with {code} (output preserved in {demo_dir}, log {log})')
    return receipt


def stage_evaluate(options, spec):
    demo_dir = demo_dir_of(options, spec)
    if not demo_dir.is_dir():
        raise StageError(f'no demo directory {demo_dir}; run the physics stage first')
    try:
        summary = evaluate_demo_dir(spec, demo_dir, controller_profile=profile_of(options, spec.key),
                                    demo_set=demo_set_of(options))
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise StageError(f'evaluation failed: {error!r}') from error
    path = next_free(demo_dir / 'demo_summary.json')
    summary['summary_path'] = str(path)
    write_new_json(path, summary)
    print(f'[{spec.key}] evaluate: {summary["verdict"]} | {summary["label"]} -> {path}', flush=True)
    return summary


def stage_render(options, spec):
    demo_dir = demo_dir_of(options, spec)
    summary_path = latest_existing(demo_dir / 'demo_summary.json')
    if summary_path is None:
        raise StageError(f'no demo_summary.json in {demo_dir}; run the evaluate stage first (the label comes from it)')
    episode = demo_dir / 'episode_000'
    if not (episode / 'frame_manifest.json').is_file():
        raise StageError(f'no recorded episode frames in {episode}')
    summary = read_json(summary_path)
    out = next_free(demo_dir / 'render')
    argv = render_argv(spec, episode, out, summary['label'], python=options.render_python)
    print(f'[{spec.key}] render: {shell_line(argv, options.tmpdir)}', flush=True)
    start = time.monotonic()
    result = subprocess.run(argv, cwd=ROOT, env={**os.environ, 'TMPDIR': str(options.tmpdir)})
    if result.returncode != 0:
        raise StageError(f'render exited with {result.returncode} (see {out})')
    print(f'[{spec.key}] render finished in {time.monotonic() - start:.0f}s -> {out}', flush=True)
    return out


def publish_names(spec, run_id, render_suffix=''):
    stem = f'{run_id}{render_suffix}_{spec.dirname}'
    return {'demo.gif': f'{stem}.gif', 'poster.png': f'{stem}_poster.png', 'demo.mp4': f'{stem}.mp4',
            'provenance': f'{stem}.json'}


def stage_publish(options, spec):
    demo_dir = demo_dir_of(options, spec)
    render = latest_existing(demo_dir / 'render')
    summary_path = latest_existing(demo_dir / 'demo_summary.json')
    if render is None or not (render / 'render_receipt.json').is_file() or summary_path is None:
        raise StageError(f'no finished render/summary in {demo_dir}; run evaluate and render first')
    summary = read_json(summary_path)
    names = publish_names(spec, options.run_id, revision_suffix(render, demo_dir / 'render'))
    target_dir = Path(options.media_root) / spec.dirname
    planned = {source: target_dir / names[source] for source in ('demo.gif', 'poster.png', 'demo.mp4')
               if (render / source).is_file()}
    limit = options.publish_max_mb * 1e6
    published, skipped = {}, {}
    existing = [str(p) for p in [*planned.values(), target_dir / names['provenance']] if p.exists()]
    if existing:
        raise StageError(f'refusing to overwrite published media: {existing}')
    target_dir.mkdir(parents=True, exist_ok=True)
    for source, target in planned.items():
        size = (render / source).stat().st_size
        if size > limit:
            skipped[source] = f'{size} bytes exceeds --publish-max-mb {options.publish_max_mb}'
            continue
        with open(render / source, 'rb') as src, open(target, 'xb') as dst:
            shutil.copyfileobj(src, dst)
        published[source] = {'path': str(target), 'bytes': size, 'sha256': sha256_file(target)}
    provenance = {'demo': spec.key, 'title': spec.title, 'run_id': options.run_id, 'verdict': summary['verdict'],
                  'label': summary['label'], 'control_mode': CONTROL_MODE, 'cutting_law': summary.get('cutting_law'),
                  'runtime_source': deep(summary, 'runtime', 'runtime_source'), 'episode_dir': summary['episode_dir'],
                  'summary': str(summary_path), 'render_dir': str(render), 'published': published, 'skipped': skipped,
                  'episode_sha256': {k: v['sha256'] for k, v in deep(summary, 'artifacts', 'files', default={}).items()}}
    provenance_path = write_new_json(target_dir / names['provenance'], provenance)
    receipt_path = write_new_json(next_free(demo_dir / 'publish_receipt.json'),
                                  {**provenance, 'provenance_path': str(provenance_path)})
    print(f'[{spec.key}] publish: {sorted(p["path"] for p in published.values())} skipped={skipped} -> {receipt_path}',
          flush=True)
    return published


def write_index(options):
    run_root = Path(options.root) / options.run_id
    if not run_root.is_dir():
        return None
    demos = {}
    for key in ORDER:
        spec = DEMOS[key]
        demo_dir = run_root / spec.dirname
        if not demo_dir.is_dir():
            continue
        summary_path = latest_existing(demo_dir / 'demo_summary.json')
        summary = read_json(summary_path) if summary_path else None
        render = latest_existing(demo_dir / 'render')
        receipt = read_json(render / 'render_receipt.json') if render else None
        publish = latest_existing(demo_dir / 'publish_receipt.json')
        demos[key] = {'title': spec.title, 'task_id': spec.task_id, 'demo_dir': str(demo_dir),
                      'summary': None if summary_path is None else str(summary_path),
                      'verdict': None if summary is None else summary['verdict'],
                      'execution': deep(summary, 'execution', 'status'), 'outcome': deep(summary, 'outcome', 'status'),
                      'label': None if summary is None else summary['label'],
                      'cutting_law': None if summary is None else summary.get('cutting_law'),
                      'controller_profile': deep(summary, 'controller_profile', 'recorded'),
                      'expected_controller_profile': profile_of(options, key),
                      'vision_applied_shift_m': deep(summary, 'execution', 'policy_vision', 'applied_shift_m'),
                      'runtime_source': deep(summary, 'runtime', 'runtime_source'),
                      'measured_capture_rate_hz': deep(summary, 'measured_capture_rate', 'rate_hz'),
                      'wall_time_s': deep(summary, 'execution', 'wall_time_s'),
                      'render_dir': None if render is None else str(render),
                      'render_outputs': None if receipt is None else receipt.get('outputs'),
                      'publish_receipt': None if publish is None else str(publish)}
    index = {'schema': 'blademaster_demo_index_v1', 'run_id': options.run_id, 'run_root': str(run_root),
             'control_mode': CONTROL_MODE, 'scenario': SCENARIO, 'git': git_info(), 'command': sys.argv,
             'controller_profile': options.controller_profile if demo_set_of(options) is None else 'per_demo',
             'controller_profiles': {key: profile_of(options, key) for key in demos},
             'demo_set': None if demo_set_of(options) is None else demo_set_of(options).as_dict(),
             'written_unix_s': time.time(), 'demos': demos,
             'demo_definitions': {key: {'task_id': DEMOS[key].task_id, 'steps': DEMOS[key].steps,
                                        'extra_arguments': list(DEMOS[key].extra), 'title': DEMOS[key].title}
                                  for key in ORDER}}
    path = write_new_json(next_free(run_root / 'index.json'), index)
    print(f'index -> {path}', flush=True)
    return path


# ----------------------------------------------------------------------------- CLI

def selected_demos(values):
    keys = []
    for value in values or ['all']:
        for key in (ORDER if value == 'all' else (value,)):
            if key not in keys:
                keys.append(key)
    return [key for key in ORDER if key in keys]


def selected_stages(values, publish):
    chosen = set()
    for value in values or ['all']:
        chosen |= ({'run', 'evaluate', 'render'} | ({'publish'} if publish else set())) if value == 'all' else {value}
    return [stage for stage in STAGES if stage in chosen]


def dry_run(options, demos, stages):
    demo_set = demo_set_of(options)
    if demo_set is not None:
        print(f'# demo set {demo_set.name}: {demo_set.description}')
    for key in demos:
        spec = DEMOS[key]
        demo_dir = demo_dir_of(options, spec)
        episode = demo_dir / 'episode_000'
        print(f'# demo {key}: {spec.title}' + ('' if demo_set is None else f' [controller {profile_of(options, key)}]'))
        if 'run' in stages:
            if demo_dir.exists():
                print(f'# REFUSED run: {demo_dir} exists (never overwritten)')
            print(shell_line(run_argv(spec, demo_dir, python=options.native_python, gpu=options.gpu,
                                      cutting_law=cutting_law_of(options), seed=seed_of(options),
                                      controller_profile=profile_of(options, key)), options.tmpdir))
        if 'evaluate' in stages:
            print(f'# evaluate (in-process, read-only): {episode} -> {next_free(demo_dir / "demo_summary.json")}')
        if 'render' in stages:
            summary_path = latest_existing(demo_dir / 'demo_summary.json')
            label = read_json(summary_path)['label'] if summary_path and 'evaluate' not in stages \
                else '<label from demo_summary.json>'
            print(shell_line(render_argv(spec, episode, next_free(demo_dir / 'render'), label,
                                         python=options.render_python), options.tmpdir))
        if 'publish' in stages:
            render = latest_existing(demo_dir / 'render')
            suffix = revision_suffix(render, demo_dir / 'render') if render and 'render' not in stages else (
                revision_suffix(next_free(demo_dir / 'render'), demo_dir / 'render'))
            names = publish_names(spec, options.run_id, suffix)
            print(f'# publish: {Path(options.media_root) / spec.dirname}/'
                  f'{{{names["demo.gif"]},{names["poster.png"]},{names["demo.mp4"]},{names["provenance"]}}}')
    print(f'# index: {next_free(Path(options.root) / options.run_id / "index.json")}')


class _ExplicitStore(argparse.Action):
    """``store`` that also records that the option was given explicitly (for --demo-set exclusion)."""

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        setattr(namespace, f'{self.dest}_explicit', True)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--demo-set', choices=tuple(DEMO_SETS), default=None,
                   help='Versioned demo set with per-demo controller profiles (recommended: final_v1 = 1a/1b/2 '
                        'path_v2 + 3 path_v2_vision, cutting law 5, seed 42). Mutually exclusive with --demo and '
                        '--controller-profile')
    p.add_argument('--demo', action='append', choices=(*ORDER, 'all'), help='Repeatable; default all')
    p.add_argument('--stage', action='append', choices=(*STAGES, 'all'),
                   help='Repeatable; default all = run, evaluate, render (+ publish with --publish)')
    p.add_argument('--run-id', default=None, help='Run directory name (default: local timestamp)')
    p.add_argument('--gpu', type=int, default=1)
    p.add_argument('--cutting-law', type=int, choices=(1, 2, 3, 4, 5), default=5,
                   action=_ExplicitStore)
    p.add_argument('--seed', type=int, default=42, action=_ExplicitStore)
    p.add_argument('--controller-profile', choices=tuple(CONTROLLER_PROFILES), default='v1', action=_ExplicitStore,
                   help='Scripted controller profile passed to run_wp10_live (default v1 = the frozen demo '
                        'definitions; path_v2 = opt-in path tracking + slow approach/settle; path_v2_vision = '
                        'path_v2 + ADR 0009 wrist-depth cut-1 retarget, Demo 3 only). The evaluate stage '
                        'checks the recorded profile against this value')
    p.add_argument('--dry-run', action='store_true', help='Print the exact commands; write nothing')
    p.add_argument('--publish', action='store_true', help='With --stage all: also copy small media to docs/media/demos')
    p.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    p.add_argument('--media-root', type=Path, default=DEFAULT_MEDIA_ROOT)
    p.add_argument('--native-python', type=Path, default=DEFAULT_NATIVE_PYTHON)
    p.add_argument('--render-python', type=Path, default=DEFAULT_RENDER_PYTHON)
    p.add_argument('--tmpdir', default=DEFAULT_TMPDIR)
    p.add_argument('--timeout', type=float, default=None, help='Seconds per physics run (default: none)')
    p.add_argument('--publish-max-mb', type=float, default=10.)
    p.set_defaults(controller_profile_explicit=False, cutting_law_explicit=False, seed_explicit=False)
    return p


def main(argv=None):
    parser = build_parser()
    options = parser.parse_args(argv)
    options.run_id = options.run_id or time.strftime('%Y%m%d_%H%M%S')
    if not options.run_id or '/' in options.run_id or options.run_id in ('.', '..'):
        parser.error('--run-id must be a plain directory name')
    demo_set = demo_set_of(options)
    if demo_set is not None:
        if options.demo is not None:
            parser.error('--demo-set is mutually exclusive with --demo (the set fixes its demos)')
        if options.controller_profile_explicit:
            parser.error('--demo-set is mutually exclusive with --controller-profile (the set fixes a profile '
                         'per demo)')
        for name, pinned in (('cutting_law', demo_set.cutting_law), ('seed', demo_set.seed)):
            if getattr(options, f'{name}_explicit') and getattr(options, name) != pinned:
                parser.error(f'--demo-set {demo_set.name} pins --{name.replace("_", "-")} {pinned}')
        demos = list(demo_set.demos)
    else:
        demos = selected_demos(options.demo)
        if options.controller_profile == 'v1' and (options.demo is None or 'all' in options.demo):
            print(V1_ALL_NOTICE, file=sys.stderr, flush=True)
    stages = selected_stages(options.stage, options.publish)
    for key in demos:
        profile = profile_of(options, key)
        survey = vision_survey_options(profile)
        if survey is not None and DEMOS[key].task_id not in survey['task_ids']:
            parser.error(f'--controller-profile {profile} is declared for '
                         f'{list(survey["task_ids"])} only (Demo 3); select --demo 3')
    if options.dry_run:
        dry_run(options, demos, stages)
        return 0
    if 'run' in stages:
        existing = [str(demo_dir_of(options, DEMOS[key])) for key in demos if demo_dir_of(options, DEMOS[key]).exists()]
        if existing:
            parser.error(f'refusing to overwrite existing episode directories: {existing}')
    errors = {}
    for key in demos:
        spec = DEMOS[key]
        for stage in stages:
            try:
                {'run': stage_run, 'evaluate': stage_evaluate, 'render': stage_render, 'publish': stage_publish}[stage](
                    options, spec)
            except StageError as error:
                errors.setdefault(key, []).append(f'{stage}: {error}')
                print(f'[{key}] {stage} ERROR: {error}', flush=True)
                if stage == 'run' and not demo_dir_of(options, spec).is_dir():
                    break  # nothing recorded to evaluate or render
    write_index(options)
    if errors:
        print(json.dumps(errors, indent=2), flush=True)
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
