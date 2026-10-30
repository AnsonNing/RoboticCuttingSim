#!/usr/bin/env python3
"""WP-12 Phase 4c (spec 4.5-4.6): workpiece-geometry and material-profile generalization runs (kinematic).

Every case is declared in ``CASES`` before running. A case is a compiled rcs_scene_v1 + rcs_goal_v1 on the
``blade_master_flat_board_v1`` flat board (``ruler_knife_v2_y``, tlmpm_v5 cutting law 8 unchanged, 1 mm particles,
4 mm grid, material dt 0.1 ms unless a material profile's stability rule requires less, seed 42), run on the
kinematic ideal-tracking backend with the goal planner's WaypointPolicy and scored by the goal evaluator
(``env.evaluate()``). Results are KINEMATIC (runtime_source = kinematic_ideal_tracking).

Subcommands (outputs are never overwritten):

* ``list``: case names.
* ``preflight --out FILE``: CPU only. Compile every case and every declared refusal case, sample the lattice,
  run ``geometry_providers.lattice_report`` and the material stability table. No solver.
* ``run --case NAME --device cuda:1 --out DIR``: one episode. Writes ``DIR/result.json`` and ``DIR/final_state.npz``.
  Per step (host copies after the control step, read-only): max particle speed, min det F (every 5 steps), blade
  reaction. At the end: mechanical components and mass fractions, direct code-0 bridges per cut digit,
  goal verdict, settle state at the first armed step, deformation near the cut, cut_aware_surface_v1 extraction
  (shells vs mechanical components) on the final state with the cut_events_v1 log.

  TMPDIR=/mnt/data/ningan/tmp .venv/bin/python scripts/run_wp12_phase4c.py run --case G01_bar_ref_CUT \\
      --device cuda:1 --out artifacts/wp12/phase4c_001/G01_bar_ref_CUT
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

SEED = 42
DT = 1e-4
BOARD_TOP_WORLD_Z = .01
SETTLE_RULE = {'max_speed_m_s': .01, 'max_displacement_m': .001, 'min_particle_z_m': 0.,
               'max_centroid_horizontal_drift_m': .0005,
               'when': 'state after the control step that first arms the blade (before blade contact: the edge is '
                       'at the planner cut clearance above the top)'}
DEFORMATION_BAND_M = .003


# ----------------------------------------------------------------------------- case declarations

def scene(scene_id, L, W, H, *, protrusions=(), provider=None, material='cutting_tasks_v1', xy=(0., 0.), yaw=0.,
          board_yaw=None, max_cuts=1, slots_per_node=None, description=''):
    workpiece = {'base_box_m': {'length_m': L, 'width_m': W, 'height_m': H}, 'protrusions': [dict(p) for p in protrusions],
                 'world_pose': {'translation_m': [xy[0], xy[1], BOARD_TOP_WORLD_Z], 'yaw_deg': yaw},
                 'material_profile': material}
    if provider is not None:
        workpiece['geometry_provider'] = deepcopy(provider)
    workcell = {'profile': 'blade_master_flat_board_v1'}
    if board_yaw is not None:   # co-rotated board, shortened and moved so that it stays on the table top
        workcell['overrides'] = {'cutting_board': {'yaw_deg': board_yaw, 'center_xy_world_m': [xy[0], xy[1]],
                                                   'length_x_m': .30, 'width_y_m': .16}}
    solver = {'module': 'tlmpm_v5', 'cutting_law': 8, 'max_cuts': max_cuts, 'particle_spacing_m': .001, 'grid_dx_m': .004}
    if slots_per_node is not None:
        solver['slots_per_node'] = slots_per_node
    return {'schema': 'rcs_scene_v1', 'scene_id': scene_id,
            'description': description or f'WP-12 Phase 4c case {scene_id}',
            'workpiece': workpiece, 'supports': {'kind': 'flat_board', 'workcell': workcell},
            'tool': {'profile_id': 'ruler_knife_v2_y', 'grasp_preset': 'default'},
            'solver': solver}


def sever_goal(duration=8.):
    goal = json.loads((ROOT/'configs/wp11/goals/sever_mid.json').read_text())
    goal['goal_id'] = 'p4c_sever_mid'
    goal['description'] = 'WP-12 Phase 4c: configs/wp11/goals/sever_mid.json thresholds, sever at local x = 0'
    goal['episode']['duration_s'] = duration
    return goal


def trim_goal(duration=14.):
    goal = json.loads((ROOT/'configs/wp11/goals/trim_bump_rho10.json').read_text())
    goal['goal_id'] = 'p4c_remove_all_rho10'
    goal['description'] = ('WP-12 Phase 4c: configs/wp11/goals/trim_bump_rho10.json thresholds (remove every '
                           'protrusion, rho = 10 deg, path_v2)')
    goal['episode']['duration_s'] = duration
    return goal


def dome(pid='p0', x=0., y=0., r=.010, h=.006, family='dome', ry=None, **extra):
    return {'id': pid, 'family': family, 'x_m': x, 'y_m': y, 'radius_x_m': r, 'radius_y_m': r if ry is None else ry,
            'height_m': h, **extra}


V2 = {'id': 'heightfield_v2'}
NOISE = {'id': 'heightfield_v2', 'surface_noise': {'amplitude_m': .0005, 'wavelength_m': .012, 'seed': 3, 'modes': 8}}
TUNNEL = {'id': 'sdf_csg_experimental_v1',
          'subtract': [{'shape': 'cylinder', 'axis': 'y', 'center_m': [0., .008], 'radius_m': .004}]}
ARCH = {'id': 'sdf_csg_experimental_v1',
        'subtract': [{'shape': 'box', 'center_m': [0., 0., 0.], 'half_extents_m': [.008, .013, .005]}]}

# Geometry cases: (scene kwargs, ops). Reference material cutting_tasks_v1.
GEOMETRIES = {
    'G01_bar_ref': (dict(L=.08, W=.024, H=.016), ('CUT',)),
    'G02_bar_short': (dict(L=.05, W=.024, H=.016), ('CUT',)),
    'G03_bar_long': (dict(L=.15, W=.024, H=.016), ('CUT',)),
    'G04_bar_narrow': (dict(L=.08, W=.012, H=.016), ('CUT',)),
    'G05_bar_wide': (dict(L=.08, W=.048, H=.016), ('CUT',)),
    'G06_bar_tall': (dict(L=.06, W=.024, H=.032), ('CUT',)),
    'G07_bar_low': (dict(L=.08, W=.024, H=.008), ('CUT',)),
    'G08_block': (dict(L=.05, W=.04, H=.03), ('CUT',)),
    'G09_thin_slab': (dict(L=.10, W=.04, H=.004), ('CUT',)),
    'G10_rod_aspect': (dict(L=.14, W=.01, H=.01), ('CUT',)),
    'G11_dome': (dict(L=.08, W=.024, H=.012, protrusions=[dome()]), ('CUT', 'TRIM')),
    'G12_gaussian': (dict(L=.08, W=.024, H=.012, protrusions=[dome(family='gaussian')]), ('CUT', 'TRIM')),
    'G13_elliptical': (dict(L=.08, W=.024, H=.012, protrusions=[dome(r=.014, ry=.007)]), ('CUT', 'TRIM')),
    'G14_multi_protrusions': (dict(L=.12, W=.024, H=.012, max_cuts=3,
                                   protrusions=[dome('p0', x=-.035), dome('p1', family='gaussian'),
                                                dome('p2', x=.035, family='ellipsoid_cap', r=.008, h=.005)]),
                              ('CUT', 'TRIM')),
    'G15_overlapping': (dict(L=.08, W=.024, H=.012, max_cuts=2,
                             protrusions=[dome('p0', x=-.006, family='gaussian'), dome('p1', x=.006, family='gaussian',
                                                                                         h=.005)]), ('CUT', 'TRIM')),
    # Added after G15_overlapping_TRIM raised SlotOverflowError (8 slots/node) at 02:55: same workpiece with the
    # existing opt-in scene slot capacity 16 (WP-11 Phase E round 2 precedent). The failed run is kept.
    'G15b_overlapping_slots16': (dict(L=.08, W=.024, H=.012, max_cuts=2, slots_per_node=16,
                                      protrusions=[dome('p0', x=-.006, family='gaussian'),
                                                   dome('p1', x=.006, family='gaussian', h=.005)]), ('TRIM',)),
    'G16_asymmetric_v2': (dict(L=.08, W=.030, H=.012, provider=V2,
                               protrusions=[dome(r=.014, ry=.007, yaw_deg=30., peak_offset=.5)]), ('CUT', 'TRIM')),
    'G17_irregular_v2': (dict(L=.08, W=.024, H=.012, provider=V2, protrusions=[dome(family='irregular')]),
                         ('CUT', 'TRIM')),
    'G18_surface_noise_v2': (dict(L=.08, W=.024, H=.012, provider=NOISE, protrusions=[dome()]), ('CUT', 'TRIM')),
    'G19_translated': (dict(L=.08, W=.024, H=.016, xy=(.08, .04)), ('CUT',)),
    'G20_yaw30_corotated_board': (dict(L=.08, W=.024, H=.016, yaw=30., board_yaw=30., xy=(0., .10)), ('CUT',)),
    'G21_sdf_tunnel': (dict(L=.08, W=.024, H=.016, provider=TUNNEL), ('CUT',)),
    'G22_sdf_arch_undercut': (dict(L=.08, W=.024, H=.016, provider=ARCH), ('CUT',)),
}

# Declared refusals (preflight only): each must be refused at compile/admission with the named reason.
REFUSALS = {
    'R01_yaw30_on_unrotated_board': (dict(L=.08, W=.024, H=.016, yaw=30.), 'workpiece_yaw_differs_from_board'),
    'R02_off_board': (dict(L=.08, W=.024, H=.016, xy=(.16, 0.)), 'workpiece_off_board'),
    'R03_sdf_split_in_two': (dict(L=.08, W=.024, H=.016, provider={'id': 'sdf_csg_experimental_v1', 'subtract': [
        {'shape': 'box', 'center_m': [0., 0., .008], 'half_extents_m': [.002, .02, .02]}]}), 'disconnected_material'),
    'R04_noise_too_coarse': (dict(L=.08, W=.024, H=.004, provider={'id': 'heightfield_v2', 'surface_noise': {
        'amplitude_m': .0025, 'wavelength_m': .012, 'seed': 1}}), 'spacing_too_coarse'),
    'R05_bump_outside_base': (dict(L=.08, W=.024, H=.012, provider=V2,
                                   protrusions=[dome(r=.014, ry=.004, yaw_deg=90.)]), 'protrusion_outside_base'),
}

MATERIALS = {   # (profile, material dt); run on the G01 CUT and G11 TRIM workpieces
    'M01_reference': ('wp12_reference_v1', DT),
    'M02_soft_elastic': ('wp12_soft_elastic_v1', DT),
    'M03_clay_plastic': ('wp12_clay_plastic_v1', DT),
    'M04_firm': ('wp12_firm_v1', DT),
    'M05_stiff_dt50us': ('wp12_stiff_v1', 5e-5),
    'M06_low_friction': ('wp12_low_friction_v1', DT),
    'M07_high_friction': ('wp12_high_friction_v1', DT),
    'M08_low_cut_resistance': ('wp12_low_cut_resistance_v1', DT),
    'M09_high_cut_resistance': ('wp12_high_cut_resistance_v1', DT),
    'M10_light': ('wp12_light_v1', DT),
    'M11_dense': ('wp12_dense_v1', DT),
}
MATERIAL_REFUSALS = {'MR01_stiff_at_100us': ('wp12_stiff_v1', DT)}

# DIAGNOSTIC rows, declared at 03:05 after G07/G09/G10/G21 CUT failed with bottom-layer code-0 bridges: the sever
# goal's default (frozen v1) controller profile ends the ARMED stroke with the edge 0.4 mm above the board (the
# last 0.4 mm is the disarmed HOLD), just above the measured kinematic clean-cut window [0, 0.375] mm
# (WP12_WORKCELL.md section 5). Same scene with goal.planner.controller_profile = path_v2 (armed stroke ends at
# 0.2 mm). Labelled DIAGNOSTIC_CONTROLLER_VARIANT; they do not replace the capability-matrix cells.
DIAGNOSTICS = {f'{g}_CUT_pathv2_DIAG': g for g in ('G07_bar_low', 'G09_thin_slab', 'G10_rod_aspect', 'G21_sdf_tunnel')}
# DIAGNOSTIC row, declared at 03:58 after G15b removed 0 % of both overlapping bumps: remove_protrusions clips each
# pass at the mid-gap between neighbours (goal_planner._protrusion_passes), so both passes end/start inside the
# merged bump. One trim_plane pass (configs/wp11/goals/trim_plane_z20.json semantics, plane z = 12 mm, rho 10 deg,
# path_v2) over the region |x| <= 20 mm instead. Labelled DIAGNOSTIC_GOAL_VARIANT.
TRIM_PLANE_DIAGNOSTICS = {'G15_overlapping_TRIMPLANE_DIAG': 'G15_overlapping'}


def cases():
    out = {}
    for name, (kwargs, ops) in GEOMETRIES.items():
        for op in ops:
            out[f'{name}_{op}'] = {'kind': 'geometry', 'geometry': name, 'op': op, 'scene_kwargs': kwargs,
                                   'material_profile': 'cutting_tasks_v1', 'material_dt_s': DT}
    for name, (profile, dt) in MATERIALS.items():
        for op, geometry in (('CUT', 'G01_bar_ref'), ('TRIM', 'G11_dome')):
            kwargs = dict(GEOMETRIES[geometry][0], material=profile)
            out[f'{name}_{op}'] = {'kind': 'material', 'geometry': geometry, 'op': op, 'scene_kwargs': kwargs,
                                   'material_profile': profile, 'material_dt_s': dt}
    for name, geometry in DIAGNOSTICS.items():
        out[name] = {'kind': 'diagnostic', 'label': 'DIAGNOSTIC_CONTROLLER_VARIANT', 'geometry': geometry, 'op': 'CUT',
                     'scene_kwargs': GEOMETRIES[geometry][0], 'material_profile': 'cutting_tasks_v1', 'material_dt_s': DT,
                     'goal_planner': {'controller_profile': 'path_v2'}}
    for name, geometry in TRIM_PLANE_DIAGNOSTICS.items():
        out[name] = {'kind': 'diagnostic', 'label': 'DIAGNOSTIC_GOAL_VARIANT', 'geometry': geometry, 'op': 'TRIM_PLANE',
                     'scene_kwargs': GEOMETRIES[geometry][0], 'material_profile': 'cutting_tasks_v1', 'material_dt_s': DT}
    return out


def trim_plane_goal(scene_kwargs, duration=40.):
    return {'schema': 'rcs_goal_v1', 'goal_id': 'p4c_trim_plane_base', 'type': 'trim_plane',
            'description': 'WP-12 Phase 4c diagnostic: trim to the base top over |x| <= 20 mm with one pass '
                           '(trim_plane_z20 thresholds), rho 10 deg, path_v2',
            'episode': {'duration_s': duration, 'control_dt_s': .02},
            'planner': {'controller_profile': 'path_v2', 'rake_deg': 10.},
            'plane': {'origin_local_m': [0., 0., scene_kwargs['H']], 'normal_local': [0., 0., 1.]},
            'region_local_m': {'x': [-.02, .02]}, 'edge_clearance_m': .0002}


def build(case_name, spec):
    """(scene dict, goal dict) of a case; the goal duration is the plan end + 1 s, rounded up to 0.5 s."""
    from robotic_cutting_sim.scene_config import compile_task
    sid = f"p4c_{spec['geometry']}" + ('' if spec['material_profile'] == 'cutting_tasks_v1' else f"_{spec['material_profile']}")
    sc = scene(sid, **spec['scene_kwargs'])
    goal = (sever_goal(30.) if spec['op'] == 'CUT' else trim_plane_goal(spec['scene_kwargs'])
            if spec['op'] == 'TRIM_PLANE' else trim_goal(40.))
    if 'goal_planner' in spec:
        goal['planner'] = dict(goal.get('planner', {}), **spec['goal_planner'])
        goal['goal_id'] += '_pathv2'
    probe = compile_task(sc, goal, seed=SEED)
    labels = probe.reference_phase_labels
    done = next((i for i, label in enumerate(labels) if label == 'DONE'), len(labels)-1)
    duration = math.ceil((probe.edge_position_knots_local_txyz[done][0]+1.)*2)/2
    goal['episode']['duration_s'] = float(duration)
    return sc, goal


# ----------------------------------------------------------------------------- helpers

def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def git_state():
    run = lambda *a: subprocess.run(['git', *a], cwd=ROOT, capture_output=True, text=True).stdout  # noqa: E731
    return {'git_sha': run('rev-parse', 'HEAD').strip(), 'git_dirty': bool(run('status', '--porcelain').strip()),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def direct_bridges(nodes, valid, digits):
    """digit-0 particles whose valid stencil nodes host both digit 1 and digit 2 (analyze_cut_bridges.py rule)."""
    nodes = np.asarray(nodes, np.int64)
    has = {}
    for d in (1, 2):
        mask = np.zeros(int(nodes.max())+1, bool)
        sel = digits == d
        mask[nodes[sel][valid[sel]]] = True
        has[d] = mask
    zero = np.flatnonzero(digits == 0)
    touches = lambda d: (has[d][nodes[zero]] & valid[zero]).any(1)  # noqa: E731
    return zero[touches(1) & touches(2)]


def deformation(reference, positions, det, mask_near, mask_far, top_mask):
    u = np.linalg.norm(positions-reference, axis=1)
    out = {}
    for name, mask in (('near', mask_near), ('far', mask_far)):
        if mask.any():
            out[name] = {'particles': int(mask.sum()), 'displacement_p50_m': float(np.median(u[mask])),
                         'displacement_p95_m': float(np.percentile(u[mask], 95)), 'displacement_max_m': float(u[mask].max()),
                         'abs_detF_minus_1_p95': float(np.percentile(np.abs(det[mask]-1), 95)),
                         'detF_min': float(det[mask].min())}
    if top_mask.any():
        sag = reference[top_mask, 2]-positions[top_mask, 2]
        out['top_surface_near_sag_max_m'] = float(sag.max())
        out['top_surface_near_sag_p95_m'] = float(np.percentile(sag, 95))
    return out


# ----------------------------------------------------------------------------- preflight (CPU)

def cmd_preflight(args):
    from robotic_cutting_sim import geometry_providers as gp
    from robotic_cutting_sim import materials_wp12 as mw
    from robotic_cutting_sim.scene_config import SceneConfigError, SceneInfeasible, compile_task
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    report = {'experiment': 'wp12_phase4c_preflight', 'command': [sys.executable, *sys.argv], **git_state(),
              'python': platform.python_version(), 'numpy': np.__version__, 'cases': {}, 'refusals': {},
              'material_stability': {}}
    for name, spec in cases().items():
        start = time.perf_counter()
        row = {'kind': spec['kind'], 'geometry': spec['geometry'], 'op': spec['op'],
               'material_profile': spec['material_profile'], 'material_dt_s': spec['material_dt_s']}
        try:
            sc, goal = build(name, spec)
            task = compile_task(sc, goal, seed=SEED)
            points, masses, lattice = task.sample_particles()
            row.update(compile='PASS', task_id=task.task_id, config_sha256=task.config_sha256,
                       duration_s=task.duration_s, particles=int(len(points)), mass_kg=float(masses.sum()),
                       lattice=gp.lattice_report(lattice), provider=gp.capability_receipt(task.geometry)['provider'],
                       footprint_on_board='PASS (check_placement)', min_particle_z_m=float(points[:, 2].min()))
            if spec['material_profile'] != 'cutting_tasks_v1':
                row['stability'] = mw.admit(spec['material_profile'], task.grid_dx_m, spec['material_dt_s'],
                                            task.particle_spacing_m)
        except (SceneConfigError, ValueError, NotImplementedError) as error:
            row.update(compile='FAIL', error=f'{type(error).__name__}: {error}', reason=getattr(error, 'reason', None))
        row['wall_s'] = time.perf_counter()-start
        report['cases'][name] = row
        print(name, row['compile'], row.get('particles'), row.get('duration_s'), row.get('error', ''), flush=True)
    for name, (kwargs, expected) in REFUSALS.items():
        try:
            compile_task(scene(f'p4c_{name}', **kwargs), sever_goal(), seed=SEED)
            row = {'refused': False, 'expected_reason': expected, 'result': 'FAIL (not refused)'}
        except SceneInfeasible as error:
            row = {'refused': True, 'reason': error.reason, 'expected_reason': expected, 'message': str(error),
                   'result': 'PASS' if error.reason == expected else 'FAIL (other reason)'}
        except SceneConfigError as error:
            row = {'refused': True, 'reason': 'SceneConfigError', 'expected_reason': expected, 'message': str(error),
                   'result': 'FAIL (other reason)'}
        report['refusals'][name] = row
        print(name, row['result'], row.get('reason'), flush=True)
    for name in mw.profiles():
        report['material_stability'][name] = {f'dt_{int(dt*1e6)}us': mw.stability(name, .004, dt, .001)
                                              for dt in (1e-4, 5e-5)}
    for name, (profile, dt) in MATERIAL_REFUSALS.items():
        try:
            mw.admit(profile, .004, dt, .001)
            report['refusals'][name] = {'refused': False, 'result': 'FAIL (not refused)'}
        except mw.MaterialStabilityError as error:
            report['refusals'][name] = {'refused': True, 'reason': 'MaterialStabilityError', 'message': str(error),
                                        'result': 'PASS'}
        print(name, report['refusals'][name]['result'], flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(report), indent=1))
    print(f'wrote {out}')


# ----------------------------------------------------------------------------- run (GPU)

def cmd_run(args):
    import warp as wp
    from robotic_cutting_sim import cut_aware_surface as cas
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim import geometry_providers as gp
    from robotic_cutting_sim.cut_topology import decode_digits, mechanical_components_packed
    from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition
    from robotic_cutting_sim.policies import WaypointPolicy
    from robotic_cutting_sim.scene_config import compile_task, make_configured_kinematic_env
    spec = cases()[args.case]
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'Refusing to overwrite {out}')
    out.mkdir(parents=True)
    wall_start = time.monotonic()
    result = {'experiment': 'wp12_phase4c_run', 'case': args.case, **spec, 'status': 'RUNNING',
              'command': [sys.executable, *sys.argv], 'cwd': str(ROOT), **git_state(), 'seed': SEED,
              'device': args.device, 'python': platform.python_version(), 'warp': wp.__version__,
              'tmpdir': os.environ.get('TMPDIR'), 'runtime_source': 'kinematic_ideal_tracking',
              'settle_rule': SETTLE_RULE}
    try:
        sc, goal = build(args.case, spec)
        task = compile_task(sc, goal, seed=SEED)
        result.update(task_id=task.task_id, config_sha256=task.config_sha256, scene=task.scene, goal=task.goal,
                      duration_s=task.duration_s, control_dt_s=task.control_dt_s,
                      geometry_capability=gp.capability_receipt(task.geometry))
        options = goal_planner.planner_options(task.goal, None)
        limits = goal_planner.action_limits(options, task.scene)
        steps_total = int(round(task.duration_s/task.control_dt_s))
        env = make_configured_kinematic_env(task, seed=SEED, device=args.device, material_dt=spec['material_dt_s'],
                                            delta_frame='world', ik_failure='reject', history_stride=10,
                                            action_limits=limits, cut_events='cut_events_v1')
        observation, _ = env.reset(seed=SEED)
        material = env.backend.material
        bed = material.bed
        result['material_receipt'] = material.parameter_receipt
        if spec['material_profile'] != 'cutting_tasks_v1':
            from robotic_cutting_sim import materials_wp12 as mw
            result['material_profile_receipt'] = mw.receipt(spec['material_profile'], task.grid_dx_m,
                                                            spec['material_dt_s'], task.particle_spacing_m)
        result['setup_wall_s'] = time.monotonic()-wall_start
        plan = goal_planner.plan(task.scene, task.goal, task.tool_profile, options)
        policy = WaypointPolicy(env.task, plan, limits, **goal_planner.policy_options(options))
        reference = np.asarray(material.points, np.float64)
        mass = np.asarray(material.masses, np.float64)
        lattice = np.asarray(material.lattice)
        dt = task.control_dt_s
        speeds, dets, forces, phases, armed_flags, clamps = [], [], [], [], [], 0
        settle, last_armed_snapshot, centroid0 = None, None, (mass@reference)/mass.sum()
        finite = True
        loop_start = time.monotonic()
        for step in range(steps_total):
            action = policy(observation)
            phase = policy.phase if not policy.done else 'DONE'
            observation, _, terminated, truncated, info = env.step(action)
            diag = env.backend.last_diagnostics['material']
            v = bed.v.numpy()
            speed = float(np.linalg.norm(v, axis=1).max())
            speeds.append(speed)
            finite &= bool(np.isfinite(speed))
            if step % 5 == 0 or action['cut_enabled']:
                det = np.linalg.det(bed.F.numpy().astype(np.float64))
                dets.append((step+1, float(det.min())))
            blade = np.asarray(diag['current_material_reaction_impulse_world'], float)[:3]/dt
            forces.append(float(np.linalg.norm(blade)))
            phases.append(phase)
            armed_flags.append(bool(action['cut_enabled']))
            clamps += bool((info.get('action_report') or {}).get('workspace_clamped', False))
            if action['cut_enabled'] and settle is None:
                x = bed.x.numpy().astype(np.float64)
                centroid = (mass@x)/mass.sum()
                u = np.linalg.norm(x-reference, axis=1)
                settle = {'step': step+1, 'time_s': env.time_s, 'max_speed_m_s': speed,
                          'max_displacement_m': float(u.max()), 'p95_displacement_m': float(np.percentile(u, 95)),
                          'min_particle_z_m': float(x[:, 2].min()),
                          'centroid_horizontal_drift_m': float(np.linalg.norm((centroid-centroid0)[:2])),
                          'centroid_vertical_drift_m': float((centroid-centroid0)[2])}
                settle['pass'] = bool(np.isfinite(x).all() and speed < SETTLE_RULE['max_speed_m_s']
                                      and settle['max_displacement_m'] < SETTLE_RULE['max_displacement_m']
                                      and settle['min_particle_z_m'] > SETTLE_RULE['min_particle_z_m']
                                      and settle['centroid_horizontal_drift_m'] < SETTLE_RULE['max_centroid_horizontal_drift_m'])
            if phase.startswith(('HOLD', 'SKIM')) or (spec['op'] == 'TRIM' and action['cut_enabled']):
                last_armed_snapshot = (step+1, phase, bed.x.numpy().astype(np.float64), bed.F.numpy().astype(np.float64))
            if step % 100 == 0:
                print(f'[{args.case}] step {step+1}/{steps_total} t={env.time_s:.2f} phase={phase} '
                      f'vmax={speed:.4f} wall={time.monotonic()-loop_start:.0f}s', flush=True)
            if terminated or truncated:
                break
        result['loop_wall_s'] = time.monotonic()-loop_start
        result['control_steps_run'] = len(speeds)
        result['ms_per_control_step'] = 1e3*result['loop_wall_s']/max(len(speeds), 1)
        result['plan_done'] = bool(policy.done)
        result['workspace_clamped_steps'] = int(clamps)
        slots = None
        try:
            slots = bed.check_slots() if hasattr(bed, 'check_slots') else None
        except Exception as error:
            slots = {'overflow': repr(error)}
        result['slots'] = slots
        recorder = material.cut_event_recorder
        try:
            recorder.check()
            log = recorder.log()
            result['cut_events'] = {'events': int(len(log)), 'counters': recorder.counters()}
        except Exception as error:
            log = None
            result['cut_events'] = {'error': repr(error)}
        x = bed.x.numpy().astype(np.float64)
        v = bed.v.numpy().astype(np.float64)
        F = bed.F.numpy().astype(np.float64)
        det = np.linalg.det(F)
        codes = bed.code.numpy().astype(np.int64)
        nodes, weights, gradients = bed.nodes.numpy(), bed.weights.numpy(), bed.gradients.numpy()
        valid = (weights > 0) | (np.linalg.norm(gradients, axis=-1) > 0)
        count, labels = mechanical_components_packed(nodes, weights, codes, task.max_cuts, gradients=gradients)
        cmass = np.bincount(labels, weights=mass, minlength=count)
        order = np.argsort(cmass)[::-1]
        digits = decode_digits(codes, task.max_cuts)
        bridges = {}
        for k in range(task.max_cuts):
            if (digits[:, k] > 0).any():
                b = direct_bridges(nodes, valid, digits[:, k])
                bridges[f'cut{k}'] = {'count': int(len(b)), 'labelled': int((digits[:, k] > 0).sum()),
                                      'reference_z_max_m': float(reference[b, 2].max()) if len(b) else None}
        armed = np.asarray(armed_flags)
        f = np.asarray(forces)
        result['physics'] = {
            'finite': bool(finite and np.isfinite(x).all() and np.isfinite(det).all()),
            'mechanical_components': int(count), 'component_mass_fractions': (cmass[order]/mass.sum())[:8].tolist(),
            'direct_code0_bridges': bridges, 'bridges_total': int(sum(r['count'] for r in bridges.values())),
            'labelled_particles': int((codes != 0).sum()),
            'min_detF_episode': float(min(d for _, d in dets)) if dets else None,
            'min_detF_final': float(det.min()), 'detF_below_0_5_final': int((det < .5).sum()),
            'max_particle_speed_episode_m_s': float(max(speeds)),
            'max_particle_speed_armed_m_s': float(np.asarray(speeds)[armed].max()) if armed.any() else None,
            'final_max_speed_m_s': float(np.linalg.norm(v, axis=1).max()),
            'blade_force_armed_mean_N': float(f[armed].mean()) if armed.any() else None,
            'blade_force_armed_max_N': float(f[armed].max()) if armed.any() else None,
            'blade_force_armed_p95_N': float(np.percentile(f[armed], 95)) if armed.any() else None,
            'armed_steps': int(armed.sum()),
            'min_particle_z_final_m': float(x[:, 2].min()),
            'particles_below_board_top_final': int((x[:, 2] < 0).sum())}
        result['settle'] = settle
        # deformation near the cut
        if last_armed_snapshot is not None:
            s_step, s_phase, xs, Fs = last_armed_snapshot
            ds = np.linalg.det(Fs)
            if spec['op'] == 'CUT':
                origin, normal = np.asarray(task.target_origin_local_m), np.asarray(task.target_normal_local)
                distance = (reference-origin)@normal
                top = reference[:, 2] >= np.max(reference[:, 2]) - 1.5e-3
                near, far = np.abs(distance) < DEFORMATION_BAND_M, np.abs(distance) > .015
                top_near = near & top
            else:
                plane_z = task.target_origin_local_m[2]
                below = reference[:, 2] <= plane_z
                near = below & (reference[:, 2] > plane_z-DEFORMATION_BAND_M)
                far = below & (reference[:, 2] < plane_z-.008)
                top_near = near & (reference[:, 2] > plane_z-1e-3)
            result['deformation'] = {'snapshot_step': s_step, 'snapshot_phase': s_phase, 'band_m': DEFORMATION_BAND_M,
                                     **deformation(reference, xs, ds, near, far, top_near),
                                     'rule': 'CUT: reference distance to the target plane < 3 mm (near) / > 15 mm (far) '
                                             'at the last HOLD step; TRIM: substrate below the trim plane, within 3 mm '
                                             '(near) / deeper than 8 mm (far) at the last SKIM step; displacement = '
                                             '|x - X| (includes rigid motion)'}
        # goal evaluation
        try:
            evaluation = env.evaluate()
        except Exception as error:
            evaluation = {'verdict': None, 'status': 'NOT_RUN', 'error': repr(error), 'traceback': traceback.format_exc()}
        result['evaluation'] = evaluation
        # cut-aware surface
        try:
            start = time.perf_counter()
            partition = CurrentMechanicalPartition(nodes, weights, gradients, task.max_cuts, encoding='packed2')
            surface = cas.CutAwareSurface(partition, task.particle_spacing_m, lattice, reference)
            vertices, faces, info = surface.extract(x, codes, F, cut_events_log=log)
            audit = info['audit']
            result['cut_aware_surface'] = {
                'wall_s': time.perf_counter()-start, 'vertices': int(len(vertices)), 'triangles': int(len(faces)),
                'mechanical_components': audit['mechanical_components'], 'mesh_shells': audit['mesh_shells'],
                'lattice_pieces': audit.get('lattice_pieces'), 'audit_ok': audit['ok'], 'audit_errors': audit['errors'],
                'visually_separated_mechanically_bridged': len(audit['visually_separated_mechanically_bridged']),
                'components': audit['components'], 'cut_triangles': info['faces']['cut_triangles'],
                'hermite': info['hermite']}
            np.savez_compressed(out/'cut_aware_mesh.npz', vertices=vertices.astype(np.float32), faces=faces,
                                face_kind=surface.last['face_kind'], vertex_component=surface.last['vertex_component'])
        except Exception as error:
            result['cut_aware_surface'] = {'error': repr(error), 'traceback': traceback.format_exc()}
        np.savez_compressed(out/'final_state.npz', reference_local=reference.astype(np.float32),
                            positions_local=x.astype(np.float32), codes=codes.astype(np.int32), lattice=lattice,
                            labels=labels.astype(np.int32), det_F=det.astype(np.float32),
                            speeds=np.asarray(speeds, np.float32), blade_force_N=f.astype(np.float32),
                            armed=armed, min_detF_samples=np.asarray(dets, np.float64))
        result['status'] = 'DONE'
    except Exception as error:   # preserved as a failed result, never lost
        result['status'] = 'ERROR'
        result['error'] = repr(error)
        result['traceback'] = traceback.format_exc()
    result['wall_s'] = time.monotonic()-wall_start
    (out/'result.json').write_text(json.dumps(jsonable(result), indent=1))
    print(f"[{args.case}] {result['status']} wall {result['wall_s']:.0f}s verdict "
          f"{(result.get('evaluation') or {}).get('verdict')} -> {out/'result.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('list')
    p = sub.add_parser('preflight')
    p.add_argument('--out', required=True)
    p = sub.add_parser('run')
    p.add_argument('--case', required=True, choices=sorted(cases()))
    p.add_argument('--device', default='cuda:1')
    p.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.command == 'list':
        print('\n'.join(cases()))
    elif args.command == 'preflight':
        cmd_preflight(args)
    else:
        cmd_run(args)


if __name__ == '__main__':
    main()
