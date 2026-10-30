#!/usr/bin/env python3
"""Read-only ADR 0016 report of a live configured episode run with ``planner_options.perception``.

Reports the controller's own wrist-depth measurement (result.json ``policy_vision``)
next to the ORACLE displacement of the same piece between the calibration and
survey frames (physics_state.npz, nearest recorded frames; report-only, after the
episode, never a controller input), plus the per-plane cut-label boundary (the
live_001 SLICE-PUSH diagnostic definition). Writes one new JSON file, never
overwrites.

  .venv/bin/python scripts/analyze_perception_relocate.py --sample-dir DIR --run run_001 \\
      --planner-options '{"max_rotation_deg": 1.0, "max_translation_m": 0.001, "perception": "wrist_depth_relocate_v1"}' \\
      --out DIR/perception_001.json
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def _digit(encoding):
    return (lambda c, k: (c >> (2*k)) & 3) if encoding == 'packed2' else (lambda c, k: (c//3**k) % 3)


def main(argv=None):
    from robotic_cutting_sim.scene_config import compile_task
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sample-dir', type=Path, required=True)
    p.add_argument('--run', default='run_001')
    p.add_argument('--planner-options', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        p.error(f'refusing to overwrite {a.out}')
    episode = a.sample_dir/a.run/'episode_000'
    top = json.loads((a.sample_dir/a.run/'result.json').read_text())
    row = (top.get('rows') or [{}])[0]
    vision = row.get('policy_vision')
    goal = json.loads((a.sample_dir/'inputs/goal.json').read_text())
    task = compile_task(str(a.sample_dir/'inputs/scene.json'), str(a.sample_dir/'inputs/goal.json'),
                        planner_options=json.loads(a.planner_options), seed=42)
    sever = next(step for step in goal.get('steps', [goal]) if step['type'] == 'sever')
    planes = [row_['origin_local_m'][0] for row_ in sever['planes']]
    out = dict(schema='rcs_perception_relocate_report_v1', adr='docs/adr/0016-configured-perception-relocation.md',
               sample_dir=str(a.sample_dir), run=a.run, task_id=row.get('task_id'), recompiled_task_id=task.task_id,
               planes_local_x_m=planes, scope='measurement = controller input (wrist depth); truth = oracle particle '
               'state read here after the episode, report-only')
    if not vision:
        out['measurement'] = None
    else:
        surveys = []
        for survey in vision['surveys']:
            estimate, record = survey['estimate'], survey['record']
            face = ((record.get('measurement') or {}).get('end_face') or {})
            surveys.append(dict(op=survey['op'], next_cut_op=survey['next_cut_op'], valid=estimate['valid'],
                                reasons=estimate['reasons'], shift_m=estimate.get('shift_m'),
                                applied_shift_m=estimate['applied_shift_m'],
                                calibration_face_x_m=estimate.get('calibration_face_x_m'),
                                survey_face_x_m=estimate.get('survey_face_x_m'),
                                calibration_timestamp_s=estimate.get('calibration_timestamp_s'),
                                survey_timestamp_s=estimate.get('survey_timestamp_s'),
                                survey_lines_used=face.get('lines_used'), survey_line_spread_m=face.get('line_spread_m')))
        calibration = vision['calibration'] or {}
        cal_face = ((calibration.get('measurement') or {}).get('end_face') or {})
        out['measurement'] = dict(side=vision['side'], camera_id=vision['camera_id'],
                                  calibration=dict(valid=calibration.get('valid'), reason=calibration.get('reason'),
                                                   frame_timestamp_s=calibration.get('frame_timestamp_s'),
                                                   lines_used=cal_face.get('lines_used'),
                                                   line_spread_m=cal_face.get('line_spread_m'), x_m=cal_face.get('x_m')),
                                  surveys=surveys, applied_shift_m=vision['applied_shift_m'],
                                  applied_shift_world_m=vision['applied_shift_world_m'],
                                  shifted_labels=vision['shifted_labels'], inputs=vision['inputs'])
    physics = episode/'physics_state.npz'
    if physics.exists():
        material = json.loads((episode/'material.json').read_text())
        digit = _digit(material.get('code_encoding'))
        with np.load(physics) as d:
            times = np.asarray(d['time_s'], float)
            ref = task.points_to_local(np.asarray(d['reference_positions'], float))
            codes = np.asarray(d['codes'])

            def dx(i):
                return task.points_to_local(np.asarray(d['positions'][i], float))[:, 0]-ref[:, 0]
            truth = {}
            if vision:
                for survey in out['measurement']['surveys']:
                    t_cal, t_sur = survey['calibration_timestamp_s'], survey['survey_timestamp_s']
                    if t_cal is None or t_sur is None:
                        continue
                    i, j = int(np.argmin(np.abs(times-t_cal))), int(np.argmin(np.abs(times-t_sur)))
                    k = int(survey['next_cut_op'].replace('cut', ''))
                    x_cut, side = planes[k], out['measurement']['side']
                    bands = {'cut_layers_adjacent_to_plane': np.abs(ref[:, 0]-x_cut) <= .0005+1e-6,
                             'end_layer': (side*ref[:, 0]) >= (side*ref[:, 0]).max()-1e-6,
                             'outer_piece_beyond_previous_cut': (side*(ref[:, 0]-planes[k-1])) > .0005+1e-6}
                    disp_i, disp_j = dx(i), dx(j)
                    rows = {name: float(disp_j[m].mean()-disp_i[m].mean()) for name, m in bands.items()}
                    truth[survey['op']] = dict(calibration_frame_time_s=float(times[i]), survey_frame_time_s=float(times[j]),
                                               actual_displacement_m=rows,
                                               measured_minus_actual_cut_layers_m=None if survey['shift_m'] is None
                                               else float(survey['shift_m']-rows['cut_layers_adjacent_to_plane']))
            boundaries = []
            for k, x in enumerate(planes):
                dig = digit(codes[-1], k)
                if not ((dig == 1).any() and (dig == 2).any()):
                    boundaries.append(dict(plane_local_x_m=x, label_boundary_local_x_m=None))
                    continue
                b = float((ref[dig == 1, 0].max()+ref[dig == 2, 0].min())/2)
                first = next((i for i in range(len(times)) if (digit(codes[i], k) > 0).any()), None)
                near = np.abs(ref[:, 0]-x) < .005
                before = max((first or 1)-1, 0)
                boundaries.append(dict(plane_local_x_m=x, label_boundary_local_x_m=b, offset_m=b-x,
                                       material_shift_before_cut_m=float(dx(before)[near].mean()),
                                       frame_before_cut_s=float(times[before])))
        out['oracle_truth_report_only'] = dict(frames=dict(n=len(times), stride_s=float(np.median(np.diff(times)))),
                                               survey_truth=truth, cut_label_boundaries=boundaries)
    out['git'] = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(out, stream, indent=1, default=lambda v: v.tolist() if isinstance(v, np.ndarray) else str(v))
    print(json.dumps({k: out.get(k) for k in ('task_id', 'measurement', 'oracle_truth_report_only')}, indent=1,
                     default=str)[:4000])
    return 0


if __name__ == '__main__':
    sys.exit(main())
