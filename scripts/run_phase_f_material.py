#!/usr/bin/env python3
"""WP-11 Phase F2/F3: material-level law-8 validation and resolution study (kinematic, one row per invocation).

Implements the frozen protocol ``configs/wp11/phaseF_protocol_v1.json`` (rows, setups and
pass criteria are not changed here). Each invocation runs one kinematic ideal-tracking
episode (no robot, IK, grasp, slip or cameras; ``runtime_source = kinematic_ideal_tracking``)
with cutting law 8 on tlmpm_v4 (tlmpm_v5 for F2.11), seed 42, 1 mm particles, 4 mm grid,
0.1 ms material dt and 20 ms control dt unless an F3 variant is selected, and writes
``manifest.json``, ``actions.jsonl``, ``physics_state.npz`` (history every
``--history-stride`` control steps plus per-step label-assignment records) and
``result.json`` with the row verdict, the protocol's always_report metrics, the mass check
and FAIL diagnostics. ``ROW_INTERPRETATION`` states exactly how each criterion is computed.
Law parameters and thresholds are never changed; oracle state is evaluation-only.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_phase_f_material.py --row F2.3_inplane_v2 \
      --device cuda:0 --out artifacts/wp11/phaseF/f2_3_inplane_v2_law8_001
  ... --row F3_cut --variant coarse --out artifacts/wp11/phaseF/f3_cut_coarse_law8_001
"""
import argparse
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import robotic_cutting_sim  # noqa: E402
from robotic_cutting_sim import law8_validation as L  # noqa: E402
from robotic_cutting_sim.control import ActionLimits  # noqa: E402
from robotic_cutting_sim.cut_topology import (mechanical_components, mechanical_components_packed,  # noqa: E402
                                              packed_to_ternary)
from robotic_cutting_sim.demo_evaluation import trim_separation_for_task  # noqa: E402
from robotic_cutting_sim.policies import Waypoint, WaypointPolicy, oriented, through_cut  # noqa: E402
from robotic_cutting_sim.recording import manifest, new_output, write_json  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


matrix = _load_script('run_phase1_matrix')            # read-only reuse: plans, swept planes, label distances, forces
c4 = _load_script('run_phase_c4_trim_material')       # read-only reuse: TRIM plans, chip and code-0 diagnostics
oblique = _load_script('rescore_phase1_oblique')      # read-only reuse: swept-plane interface / opposite-side check

SEED = 42
LAW = 8
PROTOCOL = json.loads(L.PHASE_F_PROTOCOL_PATH.read_text())
FORMAL = (.001, .004, 1e-4)
F3_VARIANTS = {'formal': FORMAL, 'coarse': (.002, .008, 2e-4), 'dt_half': (.001, .004, 5e-5),
               'medium': (.0015, .006, 1.5e-4),
               # Not a protocol configuration: the closest integer substep count to the declared medium dt
               # (0.15 ms gives 133.3 substeps per 20 ms control step, which the coupling rejects).
               'medium_dt133_supplementary': (.0015, .006, .02 / 133)}
SIDE_FORCE_N, FLAT_FORCE_N, HANDLE_FORCE_N = .1, 1., .1   # protocol F2.9 levels (= run_phase1_matrix M5 / C4)

ROWS = {
    'F2.1_gate_14': dict(task='TRIM-02-K2-R14', kind='trim_skim'),
    'F2.1_gate_15': dict(task='TRIM-02-K2-R15', kind='trim_skim'),
    'F2.1_gate_16': dict(task='TRIM-02-K2-R16', kind='trim_skim'),
    'F2.3_inplane_v2': dict(task='CUT-03-K2', kind='through_cut'),
    'F2.4_rocking': dict(task='CUT-03-K2', kind='rocking'),
    'F2.5_pure_rotation': dict(task='CUT-03-K2', kind='pure_rotation'),
    'F2.6_saw_cut': dict(task='CUT-03-K2', kind='saw_cut'),
    'F2.6_pure_draw': dict(task='CUT-03-K2', kind='pure_draw'),
    'F2.7_withdrawal_v2': dict(task='CUT-03-K2', kind='withdrawal'),
    'F2.8_finite_edge': dict(task='CUT-04-K2-W60', kind='finite_edge', stride=12),
    'F2.9_flat_press': dict(task='TRIM-02-K2-R00', kind='flat_press'),
    'F2.9_blunt_handle': dict(task='CUT-03-K2', kind='blunt_handle'),
    'F2.9_side_push': dict(task='CUT-03-K2', kind='side_push'),
    'F2.9_no_contact': dict(task='TRIM-02-K2-R10', kind='no_contact'),
    'F2.10_recontact': dict(task='CUT-03-K2', kind='recontact'),
    'F2.11_multicut_history': dict(configured='F2.11_slotted_bar_three_cuts_push', kind='multicut', stride=20),
    'F3_cut': dict(task='CUT-03-K2', kind='through_cut', f3=True),
    'F3_trim': dict(task='TRIM-02-K2-R10', kind='trim_skim', f3=True),
}
PROTOCOL_ROW = {name: ('F3_resolution' if spec.get('f3') else name) for name, spec in ROWS.items()}

ROW_INTERPRETATION = {
    'trim_skim': 'trim_separation_for_task: removed_reference_above_plane_mass_fraction >= 0.9 AND '
                 'largest_component_below_plane_mass_share >= 0.99 AND mass check; trim_separation status (all four '
                 'of its checks) and the frozen precision are reported; plan = C4 skim_plan, WaypointPolicy 0.5 mm',
    'through_cut': 'frozen strict oracle env.evaluate() status PASS_TASK_METRICS AND mass check; plan = '
                   'policies.through_cut (as matrix M1), WaypointPolicy 0.5 mm',
    'rocking': 'through_cut geometry; armed descent 10 mm/s from top + 8 mm to bottom - 4 mm (3.2 s = 4 periods) with '
               'the edge rolled theta = 10 deg sin(2 pi t / 0.8 s) about the blade normal; checks: two largest '
               'stencil-graph components >= 40% each and >= 99% together; no particle with |signed distance| > 2 '
               'particle spacings to the fitted swept plane (digit 0) in the main component of the opposite side '
               '(rescore_phase1_oblique.rescore_m2); mass check; interface statistic reported',
    'pure_rotation': 'armed half-depth descent (matrix half_depth_plan geometry, 10 mm/s to local z = 10 mm), then '
                     '0.5 s armed rotation about the edge centre, theta = triangle wave 0 -> +8 -> -8 -> 0 deg about '
                     'the blade normal (64 deg/s), no centre translation, then armed withdrawal 10 mm/s; checks: one '
                     'component; every final label (reference position) AND every label at its assignment step '
                     '(position after that control step) projects onto the blade plane within 1 reference cell '
                     '(grid dx) of the region the edge line swept while armed (rasterized 0.25 mm); mass check',
    'saw_cut': 'policies.through_cut with saw_amplitude 6 mm, saw_period 0.4 s; frozen strict oracle PASS AND mass check',
    'pure_draw': 'armed half-depth descent as pure_rotation, then 0.8 s armed draw in place: edge centre at the stop '
                 '+ 6 mm sin(2 pi t / 0.4 s) along the edge line, no advance, then armed withdrawal; checks: one '
                 'component; depth along the advance of every final label (reference) and every label at assignment '
                 '<= deepest armed edge-line depth + 1 reference cell; mass check',
    'withdrawal': 'matrix half_depth_plan(after=withdraw) (WaypointPolicy 0.01 mm); checks: one component; core band '
                  '(reference |s| <= grid dx about the fitted swept plane) deeper than the deepest armed edge '
                  'position is 100% code 0 (ADR 0008 ligament); labelled count unchanged during withdraw; mass check',
    'finite_edge': 'CUT-04-K2-W60 (CUT-02 bar, 60 mm wide) policies.through_cut shifted by -60 mm along the edge '
                   'line (edge spans world y in [-120, 0] mm; bar y in [-30, 30] mm; grip beyond -120 mm); checks: one '
                   'component; every label at its assignment step has |l| <= L (edge half-length) in the edge frame '
                   'of that step; every severed reference bond (conflicting codes) has its reference midpoint at '
                   '|l| <= L in the edge frame of the cut pose; mass check',
    'flat_press': 'C4 flat_press_plan on TRIM-02-K2-R00 (lowest blade face from 3 mm above to 5 mm into the bar top); '
                  'checks: zero labels at every step; one component; mean |total force| over press steps with blade '
                  'contacts > 1 N; mass check',
    'blunt_handle': 'CUT-03-K2 edge orientation turned 90 deg about world z (edge line -x, blade normal +y); grip '
                    '(-n face) from 3 mm outside the bar +y side face pressed 6 mm along -y (3 mm into the side) at '
                    '10 mm/s, armed, then held 0.3 s armed; blade 8+ mm outside the side face; checks: zero labels; one '
                    'component; mean |grip obstacle force| over press/hold steps with grip contact > 0.1 N; mass check',
    'side_push': 'matrix side_push_plan on CUT-03-K2: armed blade-face (normal) push +x from x = -88 to -68 mm into '
                 'the -x bar end face, edge at local z = 10 mm; checks: zero labels; one component; mean |total '
                 'force| over push steps with blade contacts > 0.1 N; mass check',
    'no_contact': 'C4 no_contact_plan on TRIM-02-K2-R10 (lowest blade point 3 mm above the dome top); checks: zero '
                  'blade contacts and zero grip force on every armed step; zero labels; mass check',
    'recontact': 'CUT-03-K2 through_cut, disarmed hold 0.2 s and retract; disarmed move to 1 mm outside the +x end '
                 'face, lower the edge to local z = 5 mm, push -x 4 mm in 0.5 s (8 mm/s), hold 0.2 s, release, lift; '
                 'checks: component count 2 at every history frame from the first post-cut frame to the end; '
                 'particle-particle contact events (stats[4]) > 0 summed over push steps AND cross-component '
                 'particle pairs within contact distance at a push/hold history frame; zero label changes on every '
                 'disarmed step; mass check',
    'multicut': 'tlmpm_v5 configured task (cutting_tasks_v4_phaseF.json F2.11): bar_slotted_board_4slots with '
                'ruler_knife_v2_y, goal-planner plan (WaypointPolicy), cuts at x = -48/-16/+16 mm (digits 0/1/2) then '
                'a cut-disabled 10 mm push of the +x piece; checks: final packed stencil graph has exactly 4 '
                'components; for each digit k no change of digit k after the last WITHDRAW:cut<k> step (per-step '
                'records), zero nonzero-digit rewrites; zero slot overflow; mass check',
}


def jsonable(value):
    return matrix.jsonable(value)


def _pose(position, rotation):
    return tuple(np.r_[position, rotation.as_quat()])


def _local(task, xyz):
    return matrix._local(task, xyz)


def _rotation(task):
    return Rotation.from_quat(task.reference_edge_pose_world(0.)[3:])


# ----------------------------------------------------------------------------- plans

def through_cut_plan(task, **kwargs):
    return matrix.labelled(through_cut(task, **kwargs), ('approach', 'start', 'cut', 'end', 'retract'))


def finite_edge_plan(task):
    shift = _rotation(task).apply([0., -task.tool_profile.half_width_m, 0.])  # -L along the edge line
    return [replace(w, pose=tuple(np.r_[np.asarray(w.pose[:3]) + shift, w.pose[3:]])) for w in through_cut_plan(task)]


def blunt_handle_plan(task):
    R = Rotation.from_rotvec([0., 0., np.pi / 2]) * _rotation(task)   # edge line -> -x, blade normal -> +y
    corners = c4.handle_box_edge(task.tool_profile)
    center, half = corners
    signs = np.stack(np.meshgrid(*[[-1., 1.]] * 3, indexing='ij'), -1).reshape(-1, 3)
    offsets = R.apply(center + signs * half)          # world offsets of the grip corners from the edge
    side = task.geometry.width_m / 2
    y0 = side + .003 - offsets[:, 1].min()            # grip -y face 3 mm outside the +y side face
    z0 = -.006                                        # grip spans local z = 10..34 mm (bar 0..20 mm)
    begin, end = _local(task, [0., y0, z0]), _local(task, [0., y0 - .006, z0])
    blade_y = y0 - .006 - task.tool_profile.thickness_m / 2
    if blade_y <= side + .005:
        raise ValueError('The blade would come within 5 mm of the bar side face')
    return [Waypoint(_pose(begin + [0., 0., .045], R), label='approach'), Waypoint(_pose(begin, R), label='lower_outside'),
            Waypoint(_pose(end, R), cut=True, speed_m_s=.01, label='handle_press'),
            Waypoint(_pose(end, R), cut=True, hold_s=.3, label='handle_hold'),
            Waypoint(_pose(end + [0., 0., .045], R), label='retract')]


def recontact_plan(task):
    plan = through_cut_plan(task)
    plan[3] = replace(plan[3], hold_s=.2)
    R = _rotation(task)
    face = task.geometry.length_m / 2 + task.tool_profile.thickness_m / 2 + .001   # blade face 1 mm off the +x end
    lift_z = plan[4].pose[2]
    over = np.r_[_local(task, [face, 0., 0.])[:2], lift_z]
    low = _local(task, [face, 0., .005])
    pushed = low - task.T_world_workpiece[:3, 0] * .004
    released = pushed + task.T_world_workpiece[:3, 0] * .002
    return plan + [Waypoint(_pose(over, R), label='move_over'), Waypoint(_pose(low, R), label='lower'),
                   Waypoint(_pose(pushed, R), speed_m_s=.008, label='push'),
                   Waypoint(_pose(pushed, R), hold_s=.2, label='push_hold'),
                   Waypoint(_pose(released, R), speed_m_s=.01, label='release'),
                   Waypoint(_pose(released + [0., 0., .03], R), label='lift')]


def _half_depth_segments(task, dt):
    R = _rotation(task)
    top = task.geometry.base_height_m
    start, stop = _local(task, [0., 0., top + .008]), _local(task, [0., 0., .010])
    initial = np.asarray(task.reference_edge_pose_world(0.)[:3], float)
    return R, start, stop, [L.line('approach', initial, start + [0., 0., .03], R, .1, dt),
                            L.line('start', start + [0., 0., .03], start, R, .1, dt),
                            L.line('descend', start, stop, R, .01, dt, cut=True)]


def rocking_segments(task):
    dt = task.control_dt_s
    R = _rotation(task)
    normal = R.apply([0., 0., 1.])
    top = task.geometry.base_height_m
    start, end = _local(task, [0., 0., top + .008]), _local(task, [0., 0., -.004])
    initial = np.asarray(task.reference_edge_pose_world(0.)[:3], float)
    duration = float(np.linalg.norm(end - start) / .01)
    roll = lambda s: Rotation.from_rotvec(np.deg2rad(10.) * np.sin(2 * np.pi * s / .8) * normal) * R  # noqa: E731
    return [L.line('approach', initial, start + [0., 0., .03], R, .1, dt), L.line('start', start + [0., 0., .03], start, R, .1, dt),
            L.Segment('cut', duration, lambda s: np.r_[start + (end - start) * min(s / duration, 1.), roll(s).as_quat()], True),
            L.hold('end', end, R, .2), L.line('retract', end, start + [0., 0., .03], R, .1, dt)]


def pure_rotation_segments(task):
    dt = task.control_dt_s
    R, start, stop, segments = _half_depth_segments(task, dt)
    normal = R.apply([0., 0., 1.])
    turn = lambda s: Rotation.from_rotvec(np.deg2rad(L.triangle_wave(s, 8., .5)) * normal) * R  # noqa: E731
    return segments + [L.Segment('rotate', .5, lambda s: np.r_[stop, turn(s).as_quat()], True),
                       L.line('withdraw', stop, start, R, .01, dt, cut=True)]


def pure_draw_segments(task):
    dt = task.control_dt_s
    R, start, stop, segments = _half_depth_segments(task, dt)
    along = R.apply([0., 1., 0.])
    return segments + [L.Segment('draw', .8, lambda s: np.r_[stop + along * .006 * np.sin(2 * np.pi * s / .4), R.as_quat()], True),
                       L.line('withdraw', stop, start, R, .01, dt, cut=True)]


WAYPOINT_PLANS = {
    'trim_skim': (c4.skim_plan, None), 'through_cut': (through_cut_plan, None),
    'saw_cut': (lambda t: through_cut_plan(t, saw_amplitude_m=.006, saw_period_s=.4), None),
    'withdrawal': (lambda t: matrix.half_depth_plan(t, after='withdraw'), 1e-5),
    'finite_edge': (finite_edge_plan, None), 'flat_press': (c4.flat_press_plan, 1e-5),
    'blunt_handle': (blunt_handle_plan, 1e-5), 'side_push': (matrix.side_push_plan, 1e-5),
    'no_contact': (c4.no_contact_plan, None), 'recontact': (recontact_plan, 1e-5)}
SCHEDULES = {'rocking': rocking_segments, 'pure_rotation': pure_rotation_segments, 'pure_draw': pure_draw_segments}


# ----------------------------------------------------------------------------- episode

def make_env(spec, variant, args):
    from robotic_cutting_sim.kinematic_env import make_kinematic_env
    spacing, grid, dt = F3_VARIANTS[variant]
    common = dict(seed=SEED, device=args.device, material_dt=dt, delta_frame='world', ik_failure='reject',
                  history_stride=args.history_stride)
    if 'configured' in spec:
        from robotic_cutting_sim import goal_planner
        from robotic_cutting_sim.scene_config import compile_task, make_configured_kinematic_env
        scene, goal = L.configured_phase_f_scene_goal(spec['configured'])
        task = compile_task(scene, goal, seed=SEED, **({} if spacing == FORMAL[0] else {'particle_spacing_m': spacing}))
        options = goal_planner.planner_options(task.goal, None)
        limits = goal_planner.action_limits(options, scene)  # as compile_task's reference schedule
        env = make_configured_kinematic_env(task, action_limits=limits, **common)
        return env, {'scene': scene, 'goal': goal, 'planner_options': options, 'limits': limits}
    task_id = spec['task']
    if task_id in L.ProfileTaskSpecV4F.TASK_IDS or grid != FORMAL[1]:
        task = L.make_phase_f_task(task_id, SEED, particle_spacing_m=spacing, grid_dx_m=grid)
        env = make_kinematic_env(task=task, cutting_law=LAW, max_cuts=1, action_limits=ActionLimits(), **common)
    else:
        env = make_kinematic_env(task_id, particle_spacing_m=spacing, cutting_law=LAW, max_cuts=1,
                                 action_limits=ActionLimits(), **common)
    return env, {'limits': ActionLimits()}


def make_policy(spec, env, extra):
    kind, task = spec['kind'], env.task
    if kind == 'multicut':
        from robotic_cutting_sim import goal_planner
        options = extra['planner_options']
        plan = goal_planner.plan(task.scene, task.goal, task.tool_profile, options)
        policy = WaypointPolicy(task, plan, extra['limits'], **goal_planner.policy_options(options))
        return policy, [dict(label=w.label, pose=list(w.pose), cut=w.cut, cut_index=w.cut_index, speed_m_s=w.speed_m_s,
                             hold_s=w.hold_s) for w in plan]
    if kind in SCHEDULES:
        policy = L.ScheduledPolicy(task, SCHEDULES[kind](task), extra['limits'])
        if policy.duration_s > task.duration_s - task.control_dt_s:
            raise ValueError(f'Schedule ({policy.duration_s:.2f} s) does not fit the episode ({task.duration_s} s)')
        return policy, policy.describe()
    builder, tolerance = WAYPOINT_PLANS[kind]
    plan = builder(task)
    policy = WaypointPolicy(task, plan, extra['limits'], **({} if tolerance is None else {'position_tolerance_m': tolerance}))
    return policy, [dict(label=w.label, pose=list(w.pose), cut=w.cut, cut_index=w.cut_index, speed_m_s=w.speed_m_s,
                         hold_s=w.hold_s, saw_amplitude_m=w.saw_amplitude_m, saw_period_s=w.saw_period_s) for w in plan]


def run_episode(spec, variant, args, out):
    env, extra = make_env(spec, variant, args)
    observation, _ = env.reset(seed=SEED)
    policy, plan = make_policy(spec, env, extra)
    task = env.task
    dt = task.control_dt_s
    material = env.backend.material
    bed = material.bed
    T = np.asarray(material.T_world_local, float)
    packed = getattr(material, 'code_encoding', 'ternary') == 'packed2'
    cuts = int(getattr(material, 'max_cuts', 1))
    segments = int(getattr(bed, 'edge_segments', 0))
    prev_codes = bed.code.numpy().copy()
    records = {k: [] for k in ('step', 'index', 'old', 'new', 'position')}
    per_step = {k: [] for k in ('min_det_F', 'max_speed', 'finite')}
    steps = []
    limit = int(round(task.duration_s / dt)) if args.steps is None else args.steps
    start = time.monotonic()
    with (out / 'actions.jsonl').open('w') as stream:
        for step in range(limit):
            action = policy(observation)
            phase = policy.phase
            observation, _, terminated, truncated, info = env.step(action)
            diag = env.backend.last_diagnostics['material']
            stroke = env.backend.stroke_history[-1]
            blade = np.asarray(diag['current_material_reaction_impulse_world'], float) / dt
            total = np.asarray(diag['current_total_material_reaction_impulse_world'], float) / dt
            boxes = bed.box_reactions.numpy().astype(float) / dt if hasattr(bed, 'box_reactions') else np.zeros((0, 6))
            codes = bed.code.numpy()
            x = bed.x.numpy()
            v = bed.v.numpy()
            changed = np.flatnonzero(codes != prev_codes)
            if changed.size:
                records['step'].append(np.full(changed.size, step + 1, np.int32))
                records['index'].append(changed.astype(np.int64))
                records['old'].append(prev_codes[changed].copy())
                records['new'].append(codes[changed].copy())
                records['position'].append(x[changed].astype(float) @ T[:3, :3].T + T[:3, 3])
            prev_codes = codes.copy()
            det = np.linalg.det(bed.F.numpy().astype(float))
            finite = bool(np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(det).all())
            per_step['min_det_F'].append(float(np.nanmin(det)) if np.isfinite(det).any() else float('nan'))
            per_step['max_speed'].append(float(np.linalg.norm(v, axis=1).max()))
            per_step['finite'].append(finite)
            digit_counts = None
            if cuts > 1:
                digits = L.decode_digits(codes, cuts) if packed else np.stack([(codes // 3 ** k) % 3 for k in range(cuts)], 1)
                digit_counts = np.count_nonzero(digits, axis=0).tolist()
            row = dict(step=step + 1, time_s=env.time_s, phase=phase, cut_enabled=bool(action['cut_enabled']),
                       cut_index=int(diag['cut_index']), edge_pose_world=env.backend.edge.tolist(),
                       blade_force_world_N=blade[:3].tolist(), total_force_world_N=total[:3].tolist(),
                       blade_torque_world_Nm=blade[3:].tolist(),
                       handle_force_local_N=boxes[0, :3].tolist() if len(boxes) else None,
                       contact_statistics=np.asarray(diag['material_contact_statistics']).tolist(),
                       stroke=stroke['stroke'], travel_m=stroke['travel_m'], engaged=stroke['engaged'],
                       any_motion=stroke['any_motion'], labelled_count=stroke['labelled_count'],
                       digit_labelled_counts=digit_counts, new_labels=int(changed.size),
                       plowing_segments_last_substep=int(bed.segment_plow.numpy().sum()) if hasattr(bed, 'segment_plow') else None,
                       cutting_segments_last_substep=int(bed.segment_ok.numpy().sum()) if hasattr(bed, 'segment_ok') else None,
                       edge_segments=segments, min_det_F=per_step['min_det_F'][-1], finite=finite,
                       action={**action, 'target': np.asarray(action['target']).tolist()}, ik_status=info['ik_status'],
                       workspace_clamped=bool((info.get('action_report') or {}).get('workspace_clamped', False)))
            stream.write(json.dumps(jsonable(row)) + '\n')
            steps.append(row)
            if terminated or truncated:
                break
    episode_wall = time.monotonic() - start
    recording = {'time_s': np.asarray([s['time_s'] for s in env._history], float),
                 **{k: np.asarray([s[k] for s in env._history]) for k in ('positions', 'velocities', 'codes')},
                 'mass_kg': material.masses, 'reference_positions': material.world_points,
                 'reference_indices': material.lattice, 'reference_stencil_nodes': bed.nodes.numpy(),
                 'reference_stencil_weights': bed.weights.numpy(), 'reference_stencil_gradients': bed.gradients.numpy(),
                 **env.backend.law4_stroke_arrays()}
    label_records = {f'label_record_{k}': (np.concatenate(v) if v else np.zeros((0, 3) if k == 'position' else 0))
                     for k, v in records.items()}
    solver_mass = bed.mass.numpy().astype(float)
    slots = bed.check_slots() if hasattr(bed, 'check_slots') else None
    eval_start = time.monotonic()
    evaluation = env.evaluate()
    evaluation_wall = time.monotonic() - eval_start
    extras = dict(step_time_s=np.asarray([s['time_s'] for s in steps], float),
                  step_edge_pose_world=np.asarray([s['edge_pose_world'] for s in steps], float),
                  step_min_det_F=np.asarray(per_step['min_det_F'], float), step_max_speed=np.asarray(per_step['max_speed'], float),
                  solver_mass_kg=solver_mass, **label_records)
    if packed:
        extras.update(code_encoding='packed2', max_cuts=cuts, slot_max_occupied=slots['max_occupied_slots'],
                      slot_overflow_entries=slots['overflow_entries'], slots_per_node=slots['slots_per_node'])
    np.savez_compressed(out / 'physics_state.npz', **recording, **extras)
    context = {'task': task, 'packed': packed, 'max_cuts': cuts, 'policy_done': bool(policy.done), 'plan': plan,
               'law4_parameters': bed.law4_parameters, 'law8_parameters': getattr(bed, 'law8_parameters', None),
               'material_receipt': material.parameter_receipt, 'runtime_source': env.runtime_source,
               'blade_half_width_m': float(bed.config.blade_half_width), 'grid_dx_m': float(task.grid_dx_m),
               'particle_spacing_m': float(task.particle_spacing_m), 'material_dt_s': float(material.config.material_dt),
               'substeps_per_control_step': int(material.substeps), 'edge_segments': segments,
               'solver_module': type(bed).__module__, 'solver_particles': int(bed.n), 'slots': slots,
               'position_tolerance_m': getattr(policy, 'position_tolerance_m', None),
               'schedule_max_translation_m': getattr(policy, 'max_translation_m', None),
               'schedule_max_rotation_deg': (float(np.rad2deg(policy.max_rotation_rad))
                                             if hasattr(policy, 'max_rotation_rad') else None),
               'episode_wall_s': episode_wall, 'evaluation_wall_s': evaluation_wall,
               'stability': L.stability_summary(per_step['min_det_F'], per_step['max_speed'], per_step['finite'])}
    if 'scene' in extra:
        context.update(scene=extra['scene'], goal=extra['goal'], planner_options=extra['planner_options'],
                       configured_task_id=task.task_id)
    env.close()
    return steps, recording, {**label_records, 'solver_mass_kg': solver_mass}, evaluation, context


# ----------------------------------------------------------------------------- analysis

def ternary_view(data, context):
    """Recording with ternary codes (exact re-encoding of packed codes) for the ternary evaluators."""
    if not context['packed']:
        return data
    return {**data, 'codes': packed_to_ternary(np.asarray(data['codes']), context['max_cuts']).astype(np.int64)}


def final_components(data, context):
    stencil = [np.asarray(data[k]) for k in ('reference_stencil_nodes', 'reference_stencil_weights')]
    gradients = np.asarray(data['reference_stencil_gradients'])
    codes = np.asarray(data['codes'])[-1]
    if context['packed']:
        return mechanical_components_packed(*stencil, codes, context['max_cuts'], gradients=gradients)
    return mechanical_components(*stencil, codes, context['max_cuts'], gradients=gradients)


def component_history(data, context):
    stencil = [np.asarray(data[k]) for k in ('reference_stencil_nodes', 'reference_stencil_weights')]
    gradients = np.asarray(data['reference_stencil_gradients'])
    graph = mechanical_components_packed if context['packed'] else mechanical_components
    cache, counts = {}, []
    for codes in np.asarray(data['codes']):
        key = codes.tobytes()
        if key not in cache:
            cache[key] = int(graph(*stencil, codes, context['max_cuts'], gradients=gradients)[0])
        counts.append(cache[key])
    return counts


def common_metrics(task, steps, data, records, evaluation, context):
    count, labels = final_components(data, context)
    view = ternary_view(data, context)
    planes = matrix.swept_planes(steps, context['max_cuts'], context['blade_half_width_m'])
    armed_phases = list(dict.fromkeys(s['phase'] for s in steps if s['cut_enabled']))
    all_steps = np.linalg.norm([s['blade_force_world_N'] for s in steps], axis=1)
    contact = np.asarray([s['contact_statistics'][1] > 0 for s in steps])
    codes_final = np.asarray(data['codes'])[-1]
    metrics = {
        'final_mechanical_components': int(count),
        'components': L.component_masses(data['mass_kg'], labels, count),
        'mass_check': L.mass_check(task, records['solver_mass_kg'], context['solver_particles']),
        'recorded_mass_kg_vs_solver': float(np.abs(np.asarray(data['mass_kg'], float) - records['solver_mass_kg']).max()),
        'final_labelled_count': int(np.count_nonzero(codes_final)),
        'labelled_count_history': [int(s['labelled_count']) for s in steps],
        'max_labelled_count': int(max(s['labelled_count'] for s in steps)),
        'component_count_history': component_history(data, context),
        'history_time_s': np.asarray(data['time_s']).tolist(),
        'swept_planes': planes,
        'label_to_swept_surface': matrix.plane_reports(view, planes, context['max_cuts']),
        'forces': {phase: matrix.force_summary(steps, phase) for phase in armed_phases},
        'blade_reaction_episode': {'mean_N_all_steps': float(all_steps.mean()), 'max_N': float(all_steps.max()),
                                   'mean_N_contact_steps': float(all_steps[contact].mean()) if contact.any() else 0.,
                                   'contact_steps': int(contact.sum()), 'steps': len(steps)},
        'plowing': L.plowing_summary(steps),
        'stroke': {k: v for k, v in L.stroke_summary(steps).items() if k not in ('cutting_segments_last_substep', 'stroke_per_step')},
        'stability': context['stability'],
        'workspace_clamped_steps': sum(s['workspace_clamped'] for s in steps),
        'ik_not_ok_steps': sum(s['ik_status'] != 'ok' for s in steps),
        'label_records': int(len(records['label_record_step'])),
        'bridges': [L.bridge_report(data['reference_stencil_nodes'], data['reference_stencil_weights'],
                                    data['reference_stencil_gradients'], codes_final, digit_index=k, cuts=context['max_cuts'],
                                    packed=context['packed'], reference_positions=data['reference_positions'])
                    for k in range(context['max_cuts'])]}
    return metrics, labels


def _phase_steps(steps, phases):
    return [i for i, s in enumerate(steps) if s['phase'] in phases]


def _armed_poses(steps, initial_pose):
    """Consecutive edge poses (before/after) of every armed control step, as chains for the swept region."""
    chains, current, previous = [], [], np.asarray(initial_pose, float)
    for s in steps:
        pose = np.asarray(s['edge_pose_world'], float)
        if s['cut_enabled']:
            if not current:
                current.append(previous)
            current.append(pose)
        elif current:
            chains.append(np.asarray(current))
            current = []
        previous = pose
    if current:
        chains.append(np.asarray(current))
    return chains


def armed_region(task, steps, half_length):
    """Swept region of the edge line over every armed control step (chains of consecutive armed poses)."""
    R = _rotation(task)
    chains = _armed_poses(steps, task.reference_edge_pose_world(0.))
    origin = np.concatenate(chains)[:, :3].mean(0)
    return L.swept_region(chains, half_length, origin=origin, edge_line=R.apply([0., 1., 0.]),
                          advance=R.apply([1., 0., 0.])), chains


def assignment_frame_coordinates(steps, records):
    """(f, l, s) of every label record in the edge frame of its control step's end pose."""
    poses = np.asarray([s['edge_pose_world'] for s in steps], float)
    index = np.asarray(records['label_record_step'], np.int64) - 1
    if not len(index):
        return np.zeros((0, 3))
    R = Rotation.from_quat(poses[index, 3:])
    rel = np.asarray(records['label_record_position'], float) - poses[index, :3]
    return R.inv().apply(rel)


def label_change_on(steps, predicate):
    rows = [s for s in steps if predicate(s)]
    return {'steps': len(rows), 'new_label_events': int(sum(s['new_labels'] for s in rows)),
            'steps_with_new_labels': int(sum(s['new_labels'] > 0 for s in rows))}


def evaluate(spec, variant, task, steps, data, records, evaluation, context):
    kind = spec['kind']
    metrics, labels = common_metrics(task, steps, data, records, evaluation, context)
    mass_ok = metrics['mass_check']['pass']
    count = metrics['final_mechanical_components']
    dx = context['grid_dx_m']
    checks = {}
    if kind == 'trim_skim':
        separation = trim_separation_for_task(task, data)
        sm = separation['metrics']
        metrics.update(trim_separation=separation, precision=c4.precision(evaluation),
                       orientation_metrics=c4.orientation_reports(task, steps),
                       lowest_blade_point_relative_to_edge_m=c4.lowest_blade_offset_m(task),
                       chip=c4.chip_analysis(task, steps, data, labels, sm['reference_substrate_component_ids']),
                       code0_diagnosis=c4.code0_diagnosis(task, steps, data, context['law4_parameters']))
        checks = {'above_plane_removal_ge_0_9': (sm['removed_reference_above_plane_mass_fraction'] or 0.) >= .9,
                  'substrate_share_ge_0_99': (sm['largest_component_below_plane_mass_share'] or 0.) >= .99}
    elif kind in ('through_cut', 'saw_cut'):
        metrics['frozen_strict'] = matrix.strict_summary(evaluation)
        checks = {'frozen_strict_pass': evaluation.get('status') == 'PASS_TASK_METRICS'}
        if kind == 'saw_cut':
            metrics['swept_normal'] = L.swept_normal_report(steps, ('cut',))
            metrics['stroke_cut_phase'] = L.stroke_summary(steps, ('cut',))
    elif kind == 'rocking':
        plane = metrics['swept_planes'].get(0, {})
        if plane.get('status') == 'FIT':
            scored = oblique.rescore_m2(data, np.asarray(plane['fit']['origin']), np.asarray(plane['fit']['normal']),
                                        context['particle_spacing_m'])
            metrics['rocking_interface'] = scored
            sc = scored['checks']
            checks = {'two_substantial_parts_each_ge_40pct': sc['two_substantial_parts_each_ge_40pct'],
                      'two_parts_together_ge_99pct': sc['two_parts_together_ge_99pct'],
                      'no_far_particle_in_opposite_side_component': sc['no_far_particle_in_opposite_side_component']}
        else:
            checks = {'swept_plane_fit': False}
        metrics['frozen_strict_report_only'] = matrix.strict_summary(evaluation)
        metrics['roll_deg_range'] = _roll_range(task, steps)
        metrics['swept_normal'] = L.swept_normal_report(steps, ('cut',))
        metrics['stroke_cut_phase'] = L.stroke_summary(steps, ('cut',))
    elif kind in ('pure_rotation', 'pure_draw'):
        half = context['blade_half_width_m']
        region, chains = armed_region(task, steps, half)
        codes_final = np.asarray(data['codes'])[-1]
        final_ref = np.asarray(data['reference_positions'], float)[codes_final != 0]
        at_assignment = np.asarray(records['label_record_position'], float)
        newly = np.asarray(records['label_record_old']) == 0
        metrics['swept_region'] = {**region.extent(), 'chains': len(chains),
                                   'out_of_plane_max_m': region.out_of_plane_max_m}
        metrics['labels_vs_swept_region'] = {
            'final_reference': L.label_region_report(region, final_ref, tolerance_m=dx, what='final labels, reference positions'),
            'at_assignment': L.label_region_report(region, at_assignment[newly], tolerance_m=dx,
                                                   what='labels at their assignment control step')}
        phase = 'rotate' if kind == 'pure_rotation' else 'draw'
        idx = _phase_steps(steps, (phase,))
        metrics[f'stroke_during_{phase}'] = L.stroke_summary(steps, (phase,))
        metrics[f'swept_normal_during_{phase}'] = L.swept_normal_report(steps, (phase,))
        metrics[f'labels_added_during_{phase}'] = label_change_on(steps, lambda s: s['phase'] == phase)
        metrics['labels_added_during_withdraw'] = label_change_on(steps, lambda s: s['phase'] == 'withdraw')
        metrics['forces_during_phase'] = matrix.force_summary(steps, phase)
        metrics['ligament'] = matrix.ligament(data, task.points_to_local(np.asarray([steps[idx[-1]]['edge_pose_world'][:3]]))[0, 2], task) if idx else None
        checks = {'one_component': count == 1}
        if kind == 'pure_rotation':
            checks['final_labels_within_one_cell_of_swept_region'] = metrics['labels_vs_swept_region']['final_reference']['outside_tolerance'] == 0
            checks['assigned_labels_within_one_cell_of_swept_region'] = metrics['labels_vs_swept_region']['at_assignment']['outside_tolerance'] == 0
        else:
            advance = _rotation(task).apply([1., 0., 0.])
            sweep = np.concatenate([L.edge_sweep_points(c, half, cell_m=2.5e-4) for c in chains])
            deepest = float((sweep @ advance).max())
            depth_final = final_ref @ advance - deepest if len(final_ref) else np.zeros(0)
            depth_assigned = at_assignment[newly] @ advance - deepest if newly.any() else np.zeros(0)
            metrics['depth_below_deepest_edge'] = {
                'deepest_edge_depth_along_advance_m': deepest, 'tolerance_m': dx,
                'final_reference_max_excess_m': float(depth_final.max()) if len(depth_final) else None,
                'at_assignment_max_excess_m': float(depth_assigned.max()) if len(depth_assigned) else None,
                'final_reference_beyond_tolerance': int(np.count_nonzero(depth_final > dx)),
                'at_assignment_beyond_tolerance': int(np.count_nonzero(depth_assigned > dx))}
            checks['no_final_label_more_than_one_cell_below_deepest_edge'] = metrics['depth_below_deepest_edge']['final_reference_beyond_tolerance'] == 0
            checks['no_assigned_label_more_than_one_cell_below_deepest_edge'] = metrics['depth_below_deepest_edge']['at_assignment_beyond_tolerance'] == 0
    elif kind == 'withdrawal':
        descent = [s for s in steps if s['phase'] == 'descend']
        stop = np.asarray(descent[-1]['edge_pose_world'], float)
        advance = Rotation.from_quat(stop[3:]).apply([1., 0., 0.])
        plane = metrics['swept_planes'].get(0, {})
        origin, normal = ((np.asarray(plane['fit']['origin']), np.asarray(plane['fit']['normal'])) if plane.get('status') == 'FIT'
                          else (stop[:3], Rotation.from_quat(stop[3:]).apply([0., 0., 1.])))
        reference = np.asarray(data['reference_positions'], float)
        codes_final = np.asarray(data['codes'])[-1]
        armed = np.asarray([s['edge_pose_world'][:3] for s in steps if s['cut_enabled']], float)
        deepest = float((armed @ advance).max())
        core = (np.abs((reference - origin) @ normal) <= dx) & (reference @ advance > deepest)
        metrics['core_band_ligament'] = {
            'core_half_width_m': dx, 'plane_source': 'swept_plane_fit' if plane.get('status') == 'FIT' else 'stop_pose',
            'deepest_armed_edge_depth_m': deepest, 'core_particles_below_stop': int(core.sum()),
            'core_code0_fraction': float(np.mean(codes_final[core] == 0)) if core.any() else None,
            'core_labelled': int(np.count_nonzero(codes_final[core] != 0))}
        metrics['ligament_matrix_definition'] = matrix.ligament(data, task.points_to_local(stop[None, :3])[0, 2], task)
        metrics['descent_label_change'] = matrix.phase_label_changes(steps, ('descend',))
        metrics['withdraw_label_change'] = matrix.phase_label_changes(steps, ('withdraw',))
        metrics['stroke_during_withdraw'] = L.stroke_summary(steps, ('withdraw',))
        checks = {'one_component': count == 1,
                  'core_band_below_stop_all_code0': metrics['core_band_ligament']['core_code0_fraction'] == 1.,
                  'labels_unchanged_during_withdrawal': metrics['withdraw_label_change'].get('change') == 0}
    elif kind == 'finite_edge':
        half = context['blade_half_width_m']
        coords = assignment_frame_coordinates(steps, records)
        newly = np.asarray(records['label_record_old']) == 0
        cut_pose = np.asarray([s['edge_pose_world'] for s in steps if s['phase'] == 'cut'][0], float)
        R = Rotation.from_quat(cut_pose[3:])
        a, b = L.severed_bonds(data['reference_indices'], np.asarray(data['codes'])[-1], 1)
        reference = np.asarray(data['reference_positions'], float)
        mid_l = R.inv().apply((reference[a] + reference[b]) / 2 - cut_pose[:3])[:, 1] if len(a) else np.zeros(0)
        ref_l = R.inv().apply(reference - cut_pose[:3])[:, 1]
        codes_final = np.asarray(data['codes'])[-1]
        metrics['finite_edge'] = {
            'edge_half_length_m': half, 'edge_span_world_y_m': [float(cut_pose[1] - half), float(cut_pose[1] + half)],
            'labels_at_assignment': int(newly.sum()),
            'max_abs_l_at_assignment_m': float(np.abs(coords[newly, 1]).max()) if newly.any() else None,
            'labels_with_abs_l_gt_L_at_assignment': int(np.count_nonzero(np.abs(coords[newly, 1]) > half)) if newly.any() else 0,
            'final_labels_reference_max_l_m': float(ref_l[codes_final != 0].max()) if (codes_final != 0).any() else None,
            'severed_bonds': int(len(a)),
            'severed_bond_midpoint_l_range_m': [float(mid_l.min()), float(mid_l.max())] if len(a) else None,
            'severed_bonds_outside_span': int(np.count_nonzero(np.abs(mid_l) > half)),
            'bar_width_under_edge_fraction': float(np.mean(np.abs(ref_l) <= half)),
            'scope': 'l = coordinate along the edge line; assignment positions are after the labelling control step'}
        metrics['frozen_strict_report_only'] = matrix.strict_summary(evaluation)
        checks = {'one_component': count == 1,
                  'no_label_beyond_edge_half_length_at_assignment': metrics['finite_edge']['labels_with_abs_l_gt_L_at_assignment'] == 0,
                  'severed_bonds_only_within_edge_span': len(a) > 0 and metrics['finite_edge']['severed_bonds_outside_span'] == 0}
    elif kind in ('flat_press', 'side_push', 'blunt_handle', 'no_contact'):
        zero = metrics['final_labelled_count'] == 0 and metrics['max_labelled_count'] == 0
        if kind == 'flat_press':
            metrics['contact_window'] = L.contact_window_force(steps, ('press',), use='total', contact='blade')
            metrics['c4_all_steps_force'] = matrix.force_summary(steps, 'press')
            checks = {'zero_new_labels': zero, 'one_component': count == 1,
                      'contact_window_mean_force_gt_1N': metrics['contact_window']['mean_N'] > FLAT_FORCE_N}
        elif kind == 'side_push':
            metrics['contact_window'] = L.contact_window_force(steps, ('side_push',), use='total', contact='blade')
            metrics['c4_all_steps_force'] = matrix.force_summary(steps, 'side_push')
            checks = {'zero_new_labels': zero, 'one_component': count == 1,
                      'contact_window_mean_force_gt_0_1N': metrics['contact_window']['mean_N'] > SIDE_FORCE_N}
        elif kind == 'blunt_handle':
            metrics['contact_window'] = L.contact_window_force(steps, ('handle_press', 'handle_hold'), use='grip', contact='grip')
            metrics['blade_contacts_during_press'] = int(sum(s['contact_statistics'][1] > 0 for s in steps
                                                             if s['phase'] in ('handle_press', 'handle_hold')))
            checks = {'zero_new_labels': zero, 'one_component': count == 1,
                      'grip_contact_window_mean_force_gt_0_1N': metrics['contact_window']['mean_N'] > HANDLE_FORCE_N}
        else:
            armed = [s for s in steps if s['cut_enabled']]
            blade_contacts = sum(s['contact_statistics'][1] > 0 for s in armed)
            grip = sum(s['handle_force_local_N'] is not None and np.linalg.norm(s['handle_force_local_N']) > 0 for s in armed)
            metrics['armed_contacts'] = {'armed_steps': len(armed), 'steps_with_blade_contacts': int(blade_contacts),
                                         'steps_with_grip_force': int(grip),
                                         'episode_steps_with_blade_contacts': int(sum(s['contact_statistics'][1] > 0 for s in steps))}
            checks = {'zero_contacts': blade_contacts == 0 and grip == 0, 'zero_labels': zero}
            metrics['one_component_report'] = count == 1
        metrics['armed_stroke'] = L.stroke_summary(steps, tuple(dict.fromkeys(s['phase'] for s in steps if s['cut_enabled'])))
        metrics['armed_stroke'].pop('stroke_per_step', None)
    elif kind == 'recontact':
        counts = metrics['component_count_history']
        times = np.asarray(data['time_s'], float)
        cut_end = max(s['time_s'] for s in steps if s['phase'] == 'cut')
        after = [c for c, t in zip(counts, times) if t > cut_end + 1e-9]
        push_rows = [s for s in steps if s['phase'] in ('push', 'push_hold')]
        pre_rows = [s for s in steps if s['phase'] in ('end', 'retract', 'move_over', 'lower')]
        contact_events = float(sum(s['contact_statistics'][4] for s in push_rows))
        contact_impulse = float(sum(s['contact_statistics'][5] for s in push_rows))
        push_window = (push_rows[0]['time_s'] - task.control_dt_s, push_rows[-1]['time_s']) if push_rows else (0., 0.)
        proximity = []
        main = metrics['components']['component_ids_by_mass'][:2]
        if len(main) == 2:
            radius = .9 * context['particle_spacing_m'] * .5   # TLMPMConfig contact_radius_scale * spacing / 2
            for i, t in enumerate(times):
                if push_window[0] - 1e-9 <= t <= push_window[1] + 1e-9:
                    proximity.append({'time_s': float(t), **L.cross_component_proximity(
                        np.asarray(data['positions'][i]), labels, main[0], main[1], 2.4 * radius)})
        disarmed = label_change_on(steps, lambda s: not s['cut_enabled'] and s['time_s'] > cut_end + 1e-9)
        pushed_x = _component_mean_x(task, data, labels, main)
        metrics['recontact'] = {'post_cut_component_counts': after, 'cut_end_time_s': cut_end,
                                'push_steps': len(push_rows), 'push_particle_contact_events': contact_events,
                                'push_particle_contact_impulse_sum_Ns': contact_impulse,
                                'pre_push_particle_contact_events': float(sum(s['contact_statistics'][4] for s in pre_rows)),
                                'push_blade_force': matrix.force_summary(steps, 'push'),
                                'cross_component_proximity_push_frames': proximity,
                                'disarmed_label_changes': disarmed, 'main_component_mean_local_x_m': pushed_x,
                                'scope': 'stats[4]/[5] count particle-pair impulses between conflicting codes or '
                                         'reference-far pairs (tlmpm_v4 _particle_contacts); proximity uses the '
                                         'contact query radius 2.4 r (r = 0.45 spacing)'}
        checks = {'components_stay_2': bool(after) and all(c == 2 for c in after) and count == 2,
                  'particle_contact_impulses_between_pieces_nonzero': contact_events > 0 and contact_impulse > 0
                  and any(p['pairs_within'] > 0 for p in proximity),
                  'zero_label_changes_while_disarmed': disarmed['new_label_events'] == 0}
    elif kind == 'multicut':
        cuts = context['max_cuts']
        ends = {}
        for k in range(cuts):
            idx = _phase_steps(steps, (f'WITHDRAW:cut{k}',))
            ends[k] = steps[idx[-1]]['step'] if idx else -1
        persistence = L.digit_persistence(records['label_record_step'], records['label_record_old'],
                                          records['label_record_new'], cuts, ends, packed=context['packed'])
        view = ternary_view(data, context)
        stencil = [np.asarray(view[k]) for k in ('reference_stencil_nodes', 'reference_stencil_weights')]
        count_t, _ = mechanical_components(*stencil, np.asarray(view['codes'])[-1], cuts,
                                           gradients=np.asarray(view['reference_stencil_gradients']))
        push = [s for s in steps if s['phase'].startswith(('PUSH', 'APPROACH_PUSH', 'LOWER', 'RELEASE', 'RETRACT'))]
        metrics['multicut'] = {'digit_persistence': persistence, 'digit_end_steps': ends,
                               'final_components_packed_graph': int(count), 'final_components_ternary_graph': int(count_t),
                               'slots': context['slots'], 'push_label_change': label_change_on(
                                   steps, lambda s: s['phase'].startswith(('PUSH', 'LOWER', 'RELEASE', 'RETRACT', 'APPROACH_PUSH'))),
                               'push_steps': len(push),
                               'digit_labelled_counts_final': steps[-1]['digit_labelled_counts'],
                               'goal_evaluation_report_only': _goal_summary(evaluation)}
        checks = {'four_mechanical_parts': count == 4,
                  'earlier_digits_unchanged': all(v['changes_after_end'] == 0 and v['nonzero_digit_rewrites'] == 0
                                                  for v in persistence.values()),
                  'zero_slot_overflow': bool(context['slots']) and context['slots']['overflow_entries'] == 0}
    checks['mass_conserved'] = bool(mass_ok)
    if not context['stability']['stable']:
        metrics['stability_warning'] = 'NaN/inf or det(F) <= 0 occurred; see stability'
    return {k: bool(v) for k, v in checks.items()}, metrics


def _roll_range(task, steps):
    R0 = _rotation(task)
    rolls = [np.rad2deg((Rotation.from_quat(s['edge_pose_world'][3:]) * R0.inv()).as_rotvec() @ R0.apply([0., 0., 1.]))
             for s in steps if s['phase'] == 'cut']
    return [float(min(rolls)), float(max(rolls))] if rolls else None


def _component_mean_x(task, data, labels, main):
    masses = np.asarray(data['mass_kg'], float)
    out = {}
    for frame in (0, -1):
        local = task.points_to_local(np.asarray(data['positions'][frame], float))
        out['first_frame' if frame == 0 else 'final_frame'] = [float(np.average(local[labels == c, 0], weights=masses[labels == c]))
                                                               for c in main]
    return out


def _goal_summary(evaluation):
    keep = ('status', 'verdict', 'success', 'integrity', 'outcome', 'quality', 'code_encoding')
    return {k: evaluation.get(k) for k in keep if k in evaluation}


def diagnose(spec, steps, data, metrics, context):
    """Compact FAIL diagnosis: bridges, stroke state, chip trajectory (TRIM) and where labels are."""
    out = {'bridges': metrics['bridges'], 'stroke': metrics['stroke'], 'plowing': metrics['plowing'],
           'components': metrics['components']['component_mass_fraction'][:6]}
    if 'chip' in metrics:
        chip = metrics['chip']
        out['chip'] = {k: chip[k] for k in chip if k != 'timeline'}
        out['code0_diagnosis'] = metrics.get('code0_diagnosis')
    return out


def git_state():
    state = matrix.git_state()
    state['runner_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    state['helper_sha256'] = L.sha256_file(Path(L.__file__))
    state['protocol_sha256'] = L.sha256_file(L.PHASE_F_PROTOCOL_PATH)
    state['phase_f_tasks_sha256'] = L.sha256_file(L.PHASE_F_TASKS_PATH)
    state['reused_scripts_sha256'] = {name: hashlib.sha256((ROOT / 'scripts' / f'{name}.py').read_bytes()).hexdigest()
                                      for name in ('run_phase1_matrix', 'run_phase_c4_trim_material', 'rescore_phase1_oblique')}
    state['package_file'] = robotic_cutting_sim.__file__
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--row', required=True, choices=sorted(ROWS))
    p.add_argument('--variant', default='formal', choices=sorted(F3_VARIANTS),
                   help='(spacing, grid, dt) configuration: F3 rows; on an F2 row a non-formal variant is a smoke run')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--history-stride', type=int, help='Control steps between recorded frames (default per row: 5; '
                                                      'F2.8 12; F2.11 20)')
    p.add_argument('--steps', type=int, help='Validation only: cap control steps (not formal)')
    a = p.parse_args()
    spec = ROWS[a.row]
    # A non-formal variant on an F2 row is a smoke/validation run only (verdict *_NONFORMAL).
    if 'law8' not in a.out.name or (('configured' in spec) != ('v5' in a.out.name)):
        p.error('--out name must contain law8 (and v5 exactly for the tlmpm_v5 row F2.11)')
    if a.history_stride is None:
        a.history_stride = spec.get('stride', 5)
    out = new_output(a.out)  # refuses to overwrite
    start = time.monotonic()
    spacing, grid, dt = F3_VARIANTS[a.variant]
    config = {**vars(a), 'out': str(a.out), 'seed': SEED}
    write_json(out / 'manifest.json', manifest(config, 'wp11_phaseF_law8' + ('_tlmpm_v5' if 'configured' in spec else '')))
    formal = a.variant == 'formal' and a.steps is None
    protocol_row = PROTOCOL['rows'][PROTOCOL_ROW[a.row]]
    base = {'row': a.row, 'variant': a.variant, 'protocol_id': PROTOCOL['protocol_id'], 'protocol_row': PROTOCOL_ROW[a.row],
            'protocol_setup': protocol_row['setup'], 'protocol_pass': protocol_row['pass'],
            'interpretation': ROW_INTERPRETATION[spec['kind']], 'command': sys.argv, **git_state(), 'seed': SEED,
            'cutting_law': LAW, 'solver': 'tlmpm_v5' if 'configured' in spec else 'tlmpm_v4',
            'particle_spacing_m': spacing, 'grid_dx_m': grid, 'material_dt_s': dt, 'control_dt_s': .02,
            'history_stride': a.history_stride, 'device': a.device, 'formal_resolution': formal,
            'runtime_source': 'kinematic_ideal_tracking', 'task_id': spec.get('task', spec.get('configured'))}
    try:
        steps, data, records, evaluation, context = run_episode(spec, a.variant, a, out)
        task = context.pop('task')
        checks, metrics = evaluate(spec, a.variant, task, steps, data, records, evaluation, context)
        outcome = 'PASS' if checks and all(checks.values()) else 'FAIL'
        if spec.get('f3') and a.variant != 'formal':
            verdict = 'CHARACTERIZATION'
        elif not formal:
            verdict = 'PASS_NONFORMAL' if outcome == 'PASS' else 'FAIL_NONFORMAL'
        else:
            verdict = outcome
        result = {**base, 'verdict': verdict, 'criterion_outcome': outcome, 'checks': checks, 'metrics': metrics,
                  'diagnosis': diagnose(spec, steps, data, metrics, context) if outcome == 'FAIL' else None,
                  'frozen_strict': evaluation, 'task': task.to_dict(), 'context': context, 'steps_run': len(steps),
                  'wall_s': time.monotonic() - start}
        write_json(out / 'result.json', jsonable(result))
        print(a.row, a.variant, verdict, outcome, checks, f'{time.monotonic() - start:.0f}s', flush=True)
    except Exception as error:
        write_json(out / 'result.json', jsonable({**base, 'verdict': 'ERROR', 'error': repr(error),
                                                  'wall_s': time.monotonic() - start}))
        raise


if __name__ == '__main__':
    main()
