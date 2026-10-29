#!/usr/bin/env python3
"""Offline check of the ADR 0009 wrist-depth shift estimator on a recorded Demo 3 episode.

The estimator (``robotic_cutting_sim.wrist_depth_survey``) reads only the
recorded wrist D405_0 depth frames, their K / T_world_camera and the measured
edge pose from actions.jsonl. The recorded particle state (physics_state.npz)
is read ONLY here, after the estimate, to score it against the actual shift.
Nothing in the episode is modified (input files are hashed before and after).

Frames: calibration = last wrist frame before the first ``cut_0`` action;
survey = last wrist frame of ``move_1`` (frame time <= input time of the first
``cut_1`` action). The robustness sweep repeats the estimate for every frame
pair in those settle windows, with seeded Gaussian depth noise, and runs the
ADR's literal kerf-face method where the kerf is in view (end of ``lift_0``).

  .venv/bin/python scripts/validate_wrist_survey_offline.py <episode_dir> --out <new.json>
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.task_scenarios import configure_task  # noqa: E402
from robotic_cutting_sim.tasks import make_task  # noqa: E402
from robotic_cutting_sim.wrist_depth_survey import (BarGeometry, SurveyConfig, estimate_shift,  # noqa: E402
                                                    survey_frame)

CAMERA = 'wrist_d405_0'
TARGET_M = .0005
NOISE_SIGMAS_M = (.00005, .0001, .0002)
NOISE_SEEDS = (0, 1, 2, 3, 4)
# Estimator development on this episode (documented, superseded versions; the final one is the module default).
ESTIMATOR_ITERATIONS = (
    dict(version=1, feature='silhouette: midpoint between the last bar-top hit and the next pixel ray at the hit height',
         live='+x end over frames 5.5-6.5 s (bar end static, camera moving) 80.44-81.03 mm; calibration 0.5 s '
              '79.60 mm; shift 0.5 s -> 6.5 s = 1.331 mm (error +0.14 mm vs the cut-1 layers)',
         verdict='rejected: view-dependent on the rendered rounded edge (0.6 mm frame-to-frame)'),
    dict(version=2, feature='top-profile crossing 0.3 mm below the local top height (first crossing walking outward)',
         live='survey frames 6.0-6.6 s 80.36-80.42 mm; calibration frames 0.1-0.5 s 79.16-79.34 mm; shift 1.231 mm '
              '(error +0.043 mm)', synthetic='1 mm fillet, no noise: max |error| 0.172 mm over 25 shifts in +-3 mm',
         verdict='superseded: pixel-phase error on the synthetic fillet and a noisy-top early exit'),
    dict(version=3, feature='crossing 0.2 mm below the local top, searched from the run end inward (module default)',
         live='see measurements/robustness below', synthetic='no noise max |error| 0.113 mm; depth noise 0.1 mm '
              '<= 0.09 mm, 0.2 mm <= 0.21 mm (6 seeds)', verdict='final'),
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def phases(episode):
    """[(input time, result time, phase, measured tool pose)] per control step."""
    rows = []
    with open(episode / 'actions.jsonl') as stream:
        for line in stream:
            row = json.loads(line)
            rows.append((float(row['input_timestamp_s']), float(row['time_s']), row.get('policy_phase'),
                         row['measured']['tool_pose']))
    return rows


def first_input(rows, phase):
    return next(t for t, _, p, _ in rows if p == phase)


def frames(episode):
    """{frame time (rounded to 1e-6): path} of the recorded wrist D405_0 depth frames."""
    out = {}
    for path in sorted((episode / 'observations').glob(f'{CAMERA}_*.npz')):
        with np.load(path) as data:
            out[round(float(data['physics_timestamp_s']), 6)] = path
    return out


def load(path, noise_sigma=0., seed=None):
    with np.load(path) as data:
        frame = {key: data[key] for key in ('depth_m', 'valid_mask', 'K', 'T_world_camera', 'physics_timestamp_s')}
    frame['camera_id'] = CAMERA
    if noise_sigma:
        rng = np.random.default_rng(seed)
        noisy = frame['depth_m'] + rng.normal(0., noise_sigma, frame['depth_m'].shape)
        frame['depth_m'] = np.where(frame['valid_mask'], noisy, 0.).astype(np.float32)
    return frame


def edge_at(rows, time_s):
    """Measured tool-edge pose at a frame time (the policy's edge feedback; used only to exclude the knife)."""
    for _, result_time, _, pose in rows:
        if abs(result_time - time_s) < 1e-9:
            return np.asarray(pose, float)
    return np.asarray(rows[0][3], float) if time_s == 0 else None


def oracle_displacements(episode, task, times):
    """VALIDATION ONLY: mean x displacement (m) of reference-x bands at the given frame times."""
    with np.load(episode / 'physics_state.npz') as data:
        frame_times = np.asarray(data['time_s'], float)
        reference = np.asarray(data['reference_positions_local'], float)
        reference_world = np.asarray(data['reference_positions'], float)
        index = {t: int(np.argmin(np.abs(frame_times - t))) for t in times}
        positions = {t: np.asarray(data['positions'][i], float) for t, i in index.items()}
    x = reference[:, 0]
    bands = {'coordinator_kerf_band_ref_x_-29.5_to_-20mm': (-.0295, -.020),
             'kerf_band_without_ref_x_-29.5_layer_-28.5_to_-20mm': (-.0285, -.020),
             'ref_x_-29.5mm_layer_only': (-.0295, -.0295),
             'left_part_ref_x_-79.5_to_-30.5mm': (-.0795, -.0305),
             'middle_ref_x_-28.5_to_+28.5mm': (-.0285, .0285),
             'cut1_layers_ref_x_+29.5_and_+30.5mm': (.0295, .0305),
             'right_part_ref_x_+31.5_to_+79.5mm': (.0315, .0795),
             'end_layer_ref_x_+79.5mm': (.0795, .0795),
             'middle_plus_right_ref_x_-28.5_to_+79.5mm': (-.0285, .0795)}
    out = {}
    for t in times:
        dx = positions[t][:, 0] - reference_world[:, 0]
        out[f'{t:.2f}'] = dict(frame_index=index[t], frame_time_s=float(frame_times[index[t]]),
                               mean_dx_m={name: float(dx[(x >= lo - 1e-6) & (x <= hi + 1e-6)].mean())
                                          for name, (lo, hi) in bands.items()})
    return out


def compact(record):
    keep = ('x_m', 'valid', 'reason', 'lines_used', 'line_spread_m', 'bracket_median_m')
    return {face: {k: record[face].get(k) for k in keep} for face in ('end_minus_x', 'end_plus_x', 'kerf_right')
            if face in record} | dict(physics_timestamp_s=record['physics_timestamp_s'], counts=record['counts'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('episode', type=Path)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    if a.out.exists():
        p.error(f'refusing to overwrite {a.out}')
    start = time.monotonic()
    episode = a.episode.resolve()
    inputs = ('actions.jsonl', 'physics_state.npz', 'task.json', 'frame_manifest.json')
    before = {name: sha256(episode / name) for name in inputs}
    task_json = json.loads((episode / 'task.json').read_text())
    task = configure_task(make_task(task_json['task_id']), scenario='blademaster_workcell_v1')
    if list(task.world_from_workpiece_xyzw) != task_json['world_from_workpiece_xyzw']:
        raise ValueError('Recorded workpiece pose differs from the blademaster_workcell_v1 task')
    geometry, config = BarGeometry.from_task(task), SurveyConfig()
    rows, wrist = phases(episode), frames(episode)
    cut0, move1, cut1, lift0 = (first_input(rows, name) for name in ('cut_0', 'move_1', 'cut_1', 'lift_0'))
    times = sorted(wrist)
    calibration_time = max(t for t in times if t < cut0 - 1e-9)
    survey_time = max(t for t in times if t <= cut1 + 1e-9)

    def measure(t, sigma=0., seed=None):
        return survey_frame(load(wrist[t], sigma, seed), geometry, config, edge_pose_world=edge_at(rows, t))

    calibration, survey = measure(calibration_time), measure(survey_time)
    shift, record = estimate_shift(calibration, survey, method='end_face_plus_x', config=config)
    kerf_at_survey = estimate_shift(calibration, survey, method='kerf_face', config=config)[1]

    # Robustness: every calibration frame of the start_0 approach/settle window x every survey frame of move_1.
    calibration_window = [t for t in times if 0. < t < cut0 - 1e-9]
    survey_window = [t for t in times if move1 + 1e-9 < t <= cut1 + 1e-9]
    lift_end_window = [t for t in times if lift0 + 1. <= t <= move1 + 1e-9]
    cal_records = {t: measure(t) for t in calibration_window}
    sur_records = {t: measure(t) for t in survey_window}
    pairs = [dict(calibration_s=c, survey_s=s, shift_m=estimate_shift(cal_records[c], sur_records[s], config=config)[0])
             for c in calibration_window for s in survey_window]
    kerf_literal = []
    for t in lift_end_window:
        frame_record = measure(t)
        estimate = estimate_shift(calibration, frame_record, method='kerf_face', config=config)[1]
        kerf_literal.append(dict(survey_s=t, kerf_right_face_x_m=frame_record['kerf_right'].get('x_m'),
                                 kerf_lines=frame_record['kerf_right'].get('lines_used'),
                                 shift_m=estimate['shift_m'], valid=estimate['valid'], reasons=estimate['reasons']))
    noise = []
    for sigma in NOISE_SIGMAS_M:
        for seed in NOISE_SEEDS:
            noisy = estimate_shift(measure(calibration_time, sigma, seed), measure(survey_time, sigma, 1000 + seed),
                                   config=config)
            noise.append(dict(depth_noise_sigma_m=sigma, seed=seed, shift_m=noisy[0], valid=noisy[1]['valid'],
                              reasons=noisy[1]['reasons']))

    # ---- oracle, validation only (after every estimate above) ----
    truth = oracle_displacements(episode, task, sorted({calibration_time, survey_time, *lift_end_window}))
    at_survey = truth[f'{survey_time:.2f}']['mean_dx_m']
    at_calibration = truth[f'{calibration_time:.2f}']['mean_dx_m']
    errors = {name: (None if shift is None else float(shift - value)) for name, value in at_survey.items()}
    pair_shifts = np.asarray([row['shift_m'] for row in pairs if row['shift_m'] is not None], float)
    noise_shifts = np.asarray([row['shift_m'] for row in noise if row['shift_m'] is not None], float)
    cut1_truth = at_survey['cut1_layers_ref_x_+29.5_and_+30.5mm']
    coordinator_truth = at_survey['coordinator_kerf_band_ref_x_-29.5_to_-20mm']
    after = {name: sha256(episode / name) for name in inputs}
    if after != before:
        raise AssertionError('The episode changed during validation')
    git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = dict(
        schema='demo3_wrist_survey_offline_v1', adr='docs/adr/0009-demo3-wrist-depth-retarget.md',
        command=' '.join([Path(sys.executable).name, *sys.argv]), git_head=git,
        episode=str(episode), episode_input_sha256=before, episode_unchanged=True,
        estimator_iterations=ESTIMATOR_ITERATIONS,
        kinematic_preflight=dict(status='NOT_RUN', reason='the kinematic preflight records no depth frames'),
        estimator=dict(module='src/robotic_cutting_sim/wrist_depth_survey.py', method='end_face_plus_x',
                       config=record['config'], inputs='wrist_d405_0 depth_m/valid_mask/K/T_world_camera + measured '
                       'edge pose (knife exclusion); physics_state.npz read only for the truth below'),
        frames=dict(calibration=dict(time_s=calibration_time, file=wrist[calibration_time].name,
                                     sha256=sha256(wrist[calibration_time]), rule='last wrist frame before cut_0 starts',
                                     cut_0_first_input_s=cut0),
                    survey=dict(time_s=survey_time, file=wrist[survey_time].name, sha256=sha256(wrist[survey_time]),
                                rule='last wrist frame of move_1 (<= first cut_1 input time)', move_1_first_input_s=move1,
                                cut_1_first_input_s=cut1)),
        measurements=dict(calibration=compact(calibration), survey=compact(survey)),
        estimate=dict(record, config=None),
        adr_literal_kerf_face=dict(
            at_survey_frame=dict(valid=kerf_at_survey['valid'], reasons=kerf_at_survey['reasons'],
                                 shift_m=kerf_at_survey['shift_m']),
            calibration_minus_x_end=compact(calibration)['end_minus_x'],
            at_lift_0_end_frames=kerf_literal,
            note='The ADR measures the -x end face at calibration and the cut-0 kerf face at the cut-1 start. In this '
                 'rig both are outside the usable view: the wrist camera sits +45 mm in x of the hand, so at the cut-0 '
                 'start the visible bar top spans about x = -33..+80 mm (the -x end at -80 mm is out of view and '
                 'behind the blade) and at the cut-1 start about x = +28..+81 mm (the kerf at -30 mm is out of view). '
                 'Where the kerf is in view (end of lift_0) the kerf-face formula is biased by the crushed kerf top '
                 'and by which 1 mm particle layer cut 0 severed.'),
        robustness=dict(
            frame_pairs=dict(calibration_frames_s=calibration_window, survey_frames_s=survey_window, n=len(pairs),
                             shift_min_m=float(pair_shifts.min()), shift_max_m=float(pair_shifts.max()),
                             shift_median_m=float(np.median(pair_shifts)),
                             max_abs_error_vs_cut1_layers_m=float(np.abs(pair_shifts - cut1_truth).max()),
                             invalid=int(len(pairs) - len(pair_shifts))),
            depth_noise=dict(rows=noise, n=len(noise), invalid=int(len(noise) - len(noise_shifts)),
                             max_abs_error_vs_cut1_layers_m=float(np.abs(noise_shifts - cut1_truth).max())
                             if len(noise_shifts) else None)),
        oracle_truth_validation_only=dict(at_survey=truth[f'{survey_time:.2f}'],
                                          at_calibration=truth[f'{calibration_time:.2f}'],
                                          at_lift_0_end={k: v for k, v in truth.items()
                                                         if float(k) in [round(t, 2) for t in lift_end_window]}),
        errors_m=errors,
        verdicts=dict(
            target_abs_error_m=TARGET_M,
            coordinator_metric=dict(truth_m=coordinator_truth, error_m=errors['coordinator_kerf_band_ref_x_-29.5_to_-20mm'],
                                    status='PASS' if shift is not None and abs(shift - coordinator_truth) <= TARGET_M
                                    else 'FAIL',
                                    caveat='The band includes the ref x = -29.5 mm layer, which cut 0 left on the LEFT '
                                           'part (it moved about -3 mm with it); its mean is not the shift of the part '
                                           'right of cut 0.'),
            cut1_plane_metric=dict(truth_m=cut1_truth, error_m=errors['cut1_layers_ref_x_+29.5_and_+30.5mm'],
                                   status='PASS' if shift is not None and abs(shift - cut1_truth) <= TARGET_M else 'FAIL',
                                   rationale='Cut 1 must separate the reference layers +29.5/+30.5 mm; their current '
                                             'x displacement is the retarget the controller needs.'),
            kerf_band_without_left_layer=dict(
                truth_m=at_survey['kerf_band_without_ref_x_-29.5_layer_-28.5_to_-20mm'],
                error_m=errors['kerf_band_without_ref_x_-29.5_layer_-28.5_to_-20mm'],
                status='PASS' if shift is not None and abs(errors['kerf_band_without_ref_x_-29.5_layer_-28.5_to_-20mm'])
                <= TARGET_M else 'FAIL')),
        calibration_time_displacement_note=dict(
            end_layer_dx_at_calibration_m=at_calibration['end_layer_ref_x_+79.5mm'],
            note='The end-face difference measures the shift between the calibration and survey frames; the bar end had '
                 'already moved this much from the reference when the calibration frame was taken.'),
        wall_s=time.monotonic() - start)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(shift_m=shift, errors_m=errors, verdicts={k: v.get('status') for k, v in result['verdicts'].items()
                                                                   if isinstance(v, dict)},
                          pairs=result['robustness']['frame_pairs'], wall_s=result['wall_s']), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
