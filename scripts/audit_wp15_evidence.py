#!/usr/bin/env python3
"""Read-only recording/media audit for agentic control robot research.

Recompute every retained attempt with the original mechanical evaluator. Write
the audit outside the recordings; output names are exclusive, preserving prior
audits. Public mode emits relative paths and a small allowlisted provenance.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.seeded_trim import evaluate_recording
from robotic_cutting_sim.recording import physical_recording_paths


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def evaluations_match(stored, recomputed, path=()):
    """Compare original decisions exactly; allow mass-sum reduction roundoff.

    Native and file evaluators can differ by two float64 ULPs in the reported
    total mass. Only that diagnostic allows four ULPs; outcome values, types,
    statuses, booleans and structure must agree exactly. Neither score changes.
    """
    if type(stored) is not type(recomputed):
        return False
    if isinstance(stored, dict):
        return stored.keys() == recomputed.keys() and all(
            evaluations_match(value, recomputed[key], path + (key,))
            for key, value in stored.items())
    if isinstance(stored, list):
        return len(stored) == len(recomputed) and all(
            evaluations_match(a, b, path + (index,))
            for index, (a, b) in enumerate(zip(stored, recomputed)))
    if isinstance(stored, float) and path == ('integrity', 'metrics', 'recorded_mass_kg'):
        return bool(math.isfinite(stored) and math.isfinite(recomputed) and
                    abs(stored - recomputed) <= 4 * max(math.ulp(stored), math.ulp(recomputed)))
    return stored == recomputed


def discover(root):
    """Include abandoned/failed attempts, including unindexed attempt folders."""
    root = Path(root).resolve()
    found, errors = set(), []
    if root.name.startswith('attempt_'):
        found.add(root)
    indexes = ([root / 'attempts.json'] if (root / 'attempts.json').is_file() else
               sorted(root.rglob('attempts.json')))
    for index in indexes:
        try:
            for number, row in enumerate(read_json(index)):
                expected = f'attempt_{number:03d}'
                if row['path'] != expected:
                    raise ValueError('Invalid attempt index path')
                found.add(index.parent / expected)
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(index.relative_to(root)), error_type=type(error).__name__))
    found.update(p for p in root.rglob('attempt_*') if p.is_dir())
    return sorted(found), errors


def camera_checks(rec):
    rows = read_json(rec / 'frame_manifest.json')
    times = np.asarray([row['time_s'] for row in rows], float)
    receipt = read_json(rec / 'episode_receipt.json')
    checks = dict(reset_to_end=bool(len(times) and times[0] == 0 and np.isfinite(times).all()
        and np.all(np.diff(times) > 0) and abs(times[-1] - receipt['simulation_duration_s']) < 1e-9))
    checks['raw_camera_streams'] = True
    for number, timestamp in enumerate(times):
        for camera in ('global_d455', 'wrist_d405_0'):
            with np.load(rec / 'observations' / f'{camera}_{number:04d}.npz', allow_pickle=False) as frame:
                rgb = frame['rgb']
                valid = bool(rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
                    and abs(float(frame['physics_timestamp_s']) - timestamp) < 1e-9)
                if camera == 'global_d455':
                    valid &= 'depth_m' not in frame
                else:
                    depth, mask = frame['depth_m'], frame['valid_mask']
                    valid &= bool(depth.shape == rgb.shape[:2] and mask.shape == depth.shape
                        and mask.dtype == np.bool_ and np.isfinite(depth[mask]).all())
                checks['raw_camera_streams'] &= valid
    return dict(status='PASS' if all(checks.values()) else 'FAIL', checks=checks,
                frames=len(times), capture_time_s=times.tolist())


def media_checks(rec, *, decode):
    manifest_path = rec / 'success_manifest.json'
    if not manifest_path.is_file():
        return dict(status='SUCCESS_MEDIA_PENDING', reason='No success manifest')
    manifest = read_json(manifest_path)
    relative = Path(manifest['mp4_path'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Invalid relative media path')
    movie = rec / relative
    checks = dict(mp4_hash=sha256(movie) == manifest['mp4_sha256'],
                  original_frames=True, recording_hashes=True)
    provenance = read_json(rec / 'provenance.json')
    # Scene/policy/simulation seeds and controller identity travel together.
    # A movie for another attempt or controller cannot complete this success.
    identity = ('scene_seed', 'policy_seed', 'sim_seed', 'scene_hash', 'solver_commit',
                'solver_version', 'agent_commit', 'candidate_id', 'controller_sha256',
                'controller_provenance', 'split')
    checks['source_identity'] = all(manifest.get(k) == provenance.get(k) for k in identity)
    checks['independent_physical_result'] = bool(manifest.get('task_success') is True
        and manifest.get('physical_result') == read_json(rec / 'independent_evaluation.json'))
    required_hashes = {p.relative_to(rec).as_posix() for p in physical_recording_paths(rec)} | {
        'compiled_task.json', 'independent_evaluation.json', 'provenance.json'}
    checks['recording_hashes'] = required_hashes.issubset(manifest.get('recording_hashes', {}))
    for name, expected in manifest.get('recording_hashes', {}).items():
        if Path(name).name != name:
            raise ValueError('Invalid recording hash path')
        checks['recording_hashes'] &= sha256(rec / name) == expected
    rows = read_json(rec / 'frame_manifest.json')
    checks['original_frames'] = len(rows) == manifest['original_frame_count'] == len(manifest['sources'])
    checks['capture_times'] = [row['time_s'] for row in rows] == manifest['capture_time_s']
    for i, (row, source) in enumerate(zip(rows, manifest['sources'])):
        checks['original_frames'] &= bool(row['time_s'] == source['time_s']
            and sha256(rec / 'observations' / f'global_d455_{i:04d}.npz') == source['global_sha256']
            and sha256(rec / 'observations' / f'wrist_d405_0_{i:04d}.npz') == source['wrist_sha256'])
    decode_status = 'NOT_RUN'
    if decode:
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams',
            '-of', 'json', str(movie)], text=True, stderr=subprocess.PIPE, timeout=60))
        video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        checks['codec_and_frames'] = bool(video['codec_name'] == 'h264' and video['pix_fmt'] == 'yuv420p'
            and int(video['nb_frames']) == len(rows))
        subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(movie), '-f', 'null', '-'],
                       capture_output=True, check=True, timeout=180)
        decode_status = 'PASS'
    return dict(status=('PASS' if decode else 'NOT_RUN_DECODE') if all(checks.values()) else 'FAIL',
                checks=checks, decode_status=decode_status, mp4_sha256=manifest['mp4_sha256'],
                mp4_path=relative.as_posix())


def audit_attempt(rec, *, decode=True, public=False):
    rec = Path(rec)
    checks, errors, hashes = {}, [], {}
    # Hash before rescoring, then check every file again. An audit never writes
    # evaluation/media back into the source recording or repairs it in place.
    for path in sorted(rec.rglob('*')):
        if path.is_symlink():
            errors.append(dict(path=path.relative_to(rec).as_posix(), error_type='SymlinkEvidence'))
        elif path.is_file():
            hashes[path.relative_to(rec).as_posix()] = sha256(path)
    result, provenance, cameras, media = {}, {}, {}, dict(status='NOT_REQUIRED')
    evaluation_comparison = {}
    try:
        result = evaluate_recording(rec)
        stored = read_json(rec / 'independent_evaluation.json')
        checks['stored_evaluation_matches'] = evaluations_match(stored, result)
        evaluation_comparison = dict(exact_match=stored == result,
            recorded_mass_diagnostic_max_ulps=4,
            stored_recorded_mass_kg=stored.get('integrity', {}).get('metrics', {}).get('recorded_mass_kg'),
            recomputed_recorded_mass_kg=result.get('integrity', {}).get('metrics', {}).get('recorded_mass_kg'))
        provenance = read_json(rec / 'provenance.json')
        if provenance.get('controller_provenance') == 'agent_candidate':
            required = ('scene_seed', 'policy_seed', 'sim_seed', 'scene_hash', 'solver_commit',
                        'solver_version', 'agent_commit', 'candidate_id', 'controller_sha256', 'split')
            checks['frozen_candidate_provenance'] = all(
                key in provenance and provenance[key] is not None and provenance[key] != '' for key in required)
        cameras = camera_checks(rec)
        checks['raw_camera_recording'] = cameras['status'] == 'PASS'
        actions = [json.loads(line) for line in (rec / 'actions.jsonl').read_text().splitlines()]
        checks['requested_and_executed_actions'] = bool(actions) and all(
            all(k in row.get('info', {}) for k in ('action_requested', 'action_executed')) for row in actions)
        if result.get('success'):
            media = media_checks(rec, decode=decode)
            checks['success_media_complete'] = media['status'] == 'PASS'
    except Exception as error:
        errors.append(dict(error_type=type(error).__name__, **({} if public else {'error': str(error)})))
    checks['recording_unchanged'] = all((rec / name).is_file() and sha256(rec / name) == digest
                                      for name, digest in hashes.items())
    checks['recording_unchanged'] &= set(hashes) == {
        p.relative_to(rec).as_posix() for p in rec.rglob('*') if p.is_file() and not p.is_symlink()}
    checks['raw_physics_present'] = all(p.relative_to(rec).as_posix() in hashes
                                        for p in physical_recording_paths(rec))
    provenance = {key: provenance.get(key) for key in ('scene_seed', 'policy_seed', 'sim_seed', 'scene_hash',
        'solver_commit', 'solver_version', 'agent_commit', 'candidate_id', 'controller_sha256',
        'controller_provenance', 'split', 'agent_success_eligible')}
    return dict(audit_status='PASS' if checks and all(checks.values()) and not errors else 'FAIL',
        checks=checks, errors=errors, provenance=provenance, physical_status=result.get('status', 'NOT_RUN'),
        task_success=bool(result.get('success')), integrity=result.get('integrity', {}),
        evaluation_comparison=evaluation_comparison,
        outcome=result.get('outcome', {}), quality=result.get('quality', {}),
        home_status=result.get('home_status', 'NOT_RUN'), raw_cameras=cameras, success_media=media,
        file_sha256=hashes)


def audit(roots, *, decode=True, public=False):
    episodes, discovery_errors = [], []
    seen = set()
    for root_number, root in enumerate(roots):
        root = Path(root).resolve()
        attempts, errors = discover(root)
        discovery_errors.extend(dict(root=root_number, **error) for error in errors)
        for rec in attempts:
            if rec in seen:
                continue
            seen.add(rec)
            episodes.append(dict(recording=f'root_{root_number}/{rec.relative_to(root)}' if public else str(rec),
                                 **audit_attempt(rec, decode=decode, public=public)))
    denominators = Counter()
    successes = Counter()
    for row in episodes:
        p = row['provenance']
        group = '|'.join(str(p.get(k) or 'unknown') for k in ('controller_provenance', 'split', 'candidate_id'))
        denominators[group] += 1
        successes[group] += row['task_success']
    successful = [row for row in episodes if row['task_success']]
    return dict(schema='wp15_recording_audit_v1', scope='Recorded evidence audit; no physics execution',
        status='PASS' if episodes and not discovery_errors and all(r['audit_status'] == 'PASS' for r in episodes) else 'FAIL',
        attempt_count=len(episodes), successful_attempt_count=len(successful),
        media_complete_count=sum(row['success_media']['status'] == 'PASS' for row in successful),
        success_media_status=('NOT_APPLICABLE_NO_SUCCESSES' if not successful else
            'PASS' if all(r['success_media']['status'] == 'PASS' for r in successful) else 'FAIL'),
        denominators=[dict(group=k, attempts=v, successes=successes[k]) for k, v in sorted(denominators.items())],
        discovery_errors=discovery_errors, attempts=episodes,
        evaluator_source_sha256={path.name: sha256(path) for path in
            [ROOT / 'src/robotic_cutting_sim' / name for name in ('seeded_trim.py', 'goals.py', 'cut_topology.py')]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--public', action='store_true')
    parser.add_argument('--skip-decode', action='store_true', help='Report decode NOT_RUN; cannot pass successful media')
    args = parser.parse_args()
    report = audit(args.roots, decode=not args.skip_decode, public=args.public)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('status', 'attempt_count', 'successful_attempt_count',
                                           'media_complete_count', 'success_media_status')}))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
