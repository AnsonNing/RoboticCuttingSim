#!/usr/bin/env python3
"""WP-11 Phase C5b: read-only analysis of a live Native trim_bump_rho10_clear episode.

Runs the C5 analyzer (``scripts/analyze_phase_c5_live.py``: goal, home, per-phase
measured edge, lowest blade point, ADR 0013 angles, reaction, grasp, contacts,
chip components, C2 check of the measured joint path) into ``<out>/analysis.json``
and adds ``<out>/c5b_analysis.json``:

* the path_v3 controller trace: per phase the measured edge vs the planned
  reference (``policy_tracking``), the bias estimate, and during PRE_CUT the
  measured edge height vs the target and the lowest blade point vs the declared
  bar top (the C5b clearance audit: PASS iff the blade's lowest point never
  enters the declared uncut bar during PRE_CUT; SKIM is reported);
* the chip-clearing step re-scored with the MEASURED move window (first PUSH
  to last PUSH_HOLD step of the live controller) next to the live verdict,
  which used the planned reference-schedule window;
* the chip (selector particles) centroid timeline and whether it left the bar
  top (centroid past the bar end or below the bar top).

Oracle material state here is evaluation-only, never an observation. The
recording is never modified; ``<out>`` must not exist.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.recording import write_json  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


c5 = _load_script('analyze_phase_c5_live')


def measured_windows(rows, dt, goal):
    """{move step goal_id: [start, end]} from the live controller phases (PUSH* of move<k>)."""
    windows = {}
    for index, step in enumerate(goal.get('steps', [])):
        if step['type'] != 'move_piece':
            continue
        times = [r['time_s'] for r in rows if str(r.get('policy_phase', '')).split(':', 1)[-1] == f'move{index}'
                 and str(r.get('policy_phase', '')).startswith('PUSH')]
        if times:
            windows[step['goal_id']] = [float(min(times) - dt), float(max(times))]
    return windows


def chip_timeline(task, recording, goal):
    from robotic_cutting_sim.goals import selector_mask
    move = next(s for s in goal['steps'] if s['type'] == 'move_piece')
    reference = task.points_to_local(np.asarray(recording['reference_positions'], float))
    mask = selector_mask(move, reference)
    masses = np.asarray(recording['mass_kg'], float)[mask]
    base = task.scene['workpiece']['base_box_m']
    rows = []
    for time_s, positions in zip(recording['time_s'], recording['positions']):
        local = task.points_to_local(np.asarray(positions, float)[mask])
        centroid = masses @ local / masses.sum()
        rows.append(dict(time_s=float(time_s), centroid_local_m=centroid.tolist(),
                         x_min_local_m=float(local[:, 0].min()), z_min_local_m=float(local[:, 2].min())))
    final = rows[-1]
    return dict(selector_particles=int(mask.sum()), selector_mass_kg=float(masses.sum()),
                initial_centroid_local_m=rows[0]['centroid_local_m'], final_centroid_local_m=final['centroid_local_m'],
                final_displacement_x_m=final['centroid_local_m'][0] - rows[0]['centroid_local_m'][0],
                off_bar_top=bool(final['x_min_local_m'] > base['length_m'] / 2 or final['centroid_local_m'][2] < base['height_m']),
                bar_end_local_x_m=base['length_m'] / 2, bar_top_local_z_m=base['height_m'],
                timeline=rows[::max(1, len(rows) // 80)])


def controller_trace(rows, target_z, bar_top_z, task):
    out = {}
    for phase in dict.fromkeys(str(r.get('policy_phase')) for r in rows):
        sub = [r for r in rows if str(r.get('policy_phase')) == phase]
        tracking = [r.get('policy_tracking') or {} for r in sub]
        errors = [np.linalg.norm(t['measured_minus_previous_reference_m']) for t in tracking
                  if t.get('measured_minus_previous_reference_m') is not None]
        cross = [t['cross_track_error_m'] for t in tracking if t.get('cross_track_error_m') is not None]
        bias = [t['bias_position_m'] for t in tracking if t.get('bias_position_m') is not None]
        z = [r['measured']['tool_pose'][2] - target_z for r in sub]
        low = [c5.lowest_blade_z(task, r['measured']['tool_pose']) - bar_top_z for r in sub]
        out[phase] = dict(steps=len(sub), t=[sub[0]['time_s'], sub[-1]['time_s']],
                          measured_minus_reference_m=c5.stats(errors), cross_track_error_m=c5.stats(cross),
                          bias_position_first_m=bias[0] if bias else None, bias_position_last_m=bias[-1] if bias else None,
                          measured_edge_z_minus_target_m=c5.stats(z), lowest_blade_point_above_bar_top_m=c5.stats(low))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('run', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--skip-path-check', action='store_true')
    a = p.parse_args()
    sys.argv = ['analyze_phase_c5_live.py', str(a.run), '--out', str(a.out)] + (['--skip-path-check'] if a.skip_path_check else [])
    c5.main()  # creates <out> (refuses an existing one) and writes analysis.json
    base = json.loads((a.out / 'analysis.json').read_text())
    episode = a.run / 'episode_000'
    result = json.loads((episode / 'result.json').read_text())
    rows = [json.loads(line) for line in (episode / 'actions.jsonl').open()]
    from robotic_cutting_sim.goals import evaluate_goal
    from robotic_cutting_sim.scene_config import compile_task
    configured = result['configured_task']
    task = compile_task(configured['scene'], configured['goal'], planner_options=configured['planner_options'],
                        seed=result['seed'])
    goal = task.goal
    dt = task.control_dt_s
    target_z = float(task.T_world_workpiece[2, 3] + task.target_origin_local_m[2] + task.edge_clearance_m)
    bar_top_z = float(task.T_world_workpiece[2, 3] + task.geometry.base_height_m)
    recording = dict(np.load(episode / 'physics_state.npz'))
    recording.update(code_encoding='packed2', max_cuts=int(task.max_cuts))
    windows = measured_windows(rows, dt, goal)
    measured_eval = evaluate_goal(goal, recording, task=task, windows=windows)
    planned_eval = evaluate_goal(goal, recording, task=task)
    trace = controller_trace([r for r in rows if not str(r.get('policy_phase', '')).startswith('RETURN_HOME')],
                             target_z, bar_top_z, task)
    pre_cut = [v for k, v in trace.items() if k.startswith('PRE_CUT')]
    skim = [v for k, v in trace.items() if k.startswith('SKIM')]
    pre_low = min(v['lowest_blade_point_above_bar_top_m']['min'] for v in pre_cut) if pre_cut else None
    pre_dz = min(v['measured_edge_z_minus_target_m']['min'] for v in pre_cut) if pre_cut else None
    skim_low = min(v['lowest_blade_point_above_bar_top_m']['min'] for v in skim) if skim else None
    clearance = dict(status='PASS' if pre_low is not None and pre_low > 0 else 'FAIL',
                     rule='measured blade lowest point (tool-profile cross-section at the measured edge pose) above the '
                          'declared bar top during every PRE_CUT step',
                     pre_cut_lowest_blade_point_above_bar_top_min_m=pre_low,
                     pre_cut_measured_edge_below_target_max_m=None if pre_dz is None else max(0., -pre_dz),
                     skim_lowest_blade_point_above_bar_top_min_m=skim_low,
                     c2_measured_path_status=(base.get('measured_task_path_clearance') or {}).get('status'))
    steps = (measured_eval.get('outcome') or {}).get('steps', [])
    move = next((s for s in steps if s['type'] == 'move_piece'), None)
    trim = next((s for s in steps if s['type'] == 'remove_protrusions'), None)
    chip = chip_timeline(task, recording, goal)
    live = result.get('task_outcome') or {}
    extra = dict(
        run=str(a.run), task_id=task.task_id, controller_profile=(result.get('controller_profile') or {}).get('name'),
        live_goal_verdict=result.get('task_status'),
        live_steps=[dict(goal_id=s['goal_id'], status=s['outcome']['status'], checks=s['outcome']['checks'],
                         window=(s['outcome'].get('metrics') or {}).get('window_s'),
                         window_source=(s['outcome'].get('metrics') or {}).get('window_source'))
                    for s in (live.get('outcome') or {}).get('steps', [])],
        planned_window_reevaluation_verdict=planned_eval.get('verdict'),
        measured_windows_s=windows, measured_window_verdict=measured_eval.get('verdict'),
        measured_window_integrity=(measured_eval.get('integrity') or {}).get('status'),
        trim_step=None if trim is None else dict(status=trim['outcome']['status'], checks=trim['outcome']['checks'],
                                                 items=trim['outcome']['metrics'].get('items'),
                                                 substrate_share=trim['outcome']['metrics'].get('largest_component_below_plane_mass_share'),
                                                 detached_below=trim['outcome']['metrics'].get('detached_below_plane_mass_fraction'),
                                                 components=trim['outcome']['metrics'].get('mechanical_components')),
        move_step=None if move is None else dict(status=move['outcome']['status'], checks=move['outcome']['checks'],
                                                 metrics=move['outcome']['metrics']),
        clearance_audit=clearance, controller_trace=trace, chip=chip,
        home_outcome=base.get('home_outcome'),
        scope='Privileged/oracle evaluation of a recorded live episode; never an observation')
    write_json(a.out / 'c5b_analysis.json', c5.matrix.jsonable(extra))
    print(json.dumps(c5.matrix.jsonable({k: extra[k] for k in ('live_goal_verdict', 'measured_window_verdict', 'home_outcome',
                                                                'clearance_audit')}))[:1200])


if __name__ == '__main__':
    main()
