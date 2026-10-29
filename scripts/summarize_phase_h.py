#!/usr/bin/env python3
"""WP-11 Phase H evidence summarizer (read-only): docs/evidence/wp11/phaseH_profiling_001.json.

Reads the Phase H probe / dataset / profile artifacts under artifacts/wp11/phaseH/
(and the Phase G Native baselines) and writes one evidence file. It never modifies
an artifact and refuses to overwrite an existing evidence file.

  .venv/bin/python scripts/summarize_phase_h.py --out docs/evidence/wp11/phaseH_profiling_001.json
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
H = ROOT/'artifacts/wp11/phaseH'
G = ROOT/'artifacts/wp11/phaseG'

PROBES = (  # (directory, description)
    ('h1_probe_baseline_002', 'full env reset, stage_reset=reopen (legacy), cameras on'),
    ('h1_probe_camera_off_001', 'full env reset, reopen, cameras off'),
    ('h1_probe_stage_same_path_001', 'after one reset: open_stage of the same workcell file only'),
    ('h1_probe_stage_new_path_001', 'after one reset: export a new workcell copy and open it (the legacy reset path)'),
    ('h1_probe_stage_close_open_001', 'after one reset: close_stage then open_stage of the same file'),
    ('h1_probe_export_only_001', 'after one reset: workcell export only, no open'),
    ('h1_probe_stage_new_empty_001', 'after one reset: new empty stage'),
    ('h1_probe_stage_reload_001', 'after one reset: clear session layer + reload root layer (no open_stage)'),
    ('h1_probe_reload_in_place_001', 'full env reset, reload_in_place v1 (IPhysx.release_physics_objects), cameras on'),
    ('h1_probe_reload_in_place_tracemalloc_001', 'as above with tracemalloc (4 frames) and undo-stack clearing'),
    ('h1_probe_reload_v2_camera_on_001', 'full env reset, reload_in_place v2 (stage-update reset/detach/attach), cameras on'),
    ('h1_probe_reload_v3_camera_on_001', 'full env reset, reload_in_place v3 = committed (v2 + gc.collect), cameras on'),
)
CRASHED = (
    ('h1_probe_reload_in_place_camera_off_001', 'reload_in_place v1, cameras off: SIGSEGV in PhysX start_simulation at reset 1'),
    ('h1_probe_reload_v2_camera_off_001', 'reload_in_place v2, cameras off: SIGSEGV in PhysX start_simulation at reset 1'),
    ('h1_probe_reload_v3_camera_off_001', 'reload_in_place v3, cameras off: SIGSEGV in PhysX start_simulation at reset 1'),
    ('h1_probe_reload_v4_notices_camera_off_001', 'USD notices only (no PhysX release), cameras off: SIGSEGV at reset 1'),
)


def load(path):
    return json.loads(Path(path).read_text())


def slope(values, start=1):
    ys = np.asarray(values[start:], float)
    if len(ys) < 2:
        return None
    return float(np.polyfit(np.arange(len(ys)), ys, 1)[0])


def probe_row(name, description):
    path = H/name/'probe.json'
    if not path.exists():
        return {'dir': f'artifacts/wp11/phaseH/{name}', 'description': description, 'status': 'MISSING'}
    doc = load(path)
    rows = doc['resets']
    key = 'rss_after_malloc_trim'
    trim = [r[key] for r in rows]
    out = {'dir': f'artifacts/wp11/phaseH/{name}', 'description': description, 'args': doc['args'],
           'git_sha': doc['manifest']['git_sha'], 'git_dirty': doc['manifest']['git_dirty'],
           'resets': len(rows), 'rss_after_malloc_trim_mib': [round(v, 1) for v in trim],
           'slope_mib_per_reset_from_reset_1': slope(trim),
           'loaded_sdf_layers': sorted({r.get('loaded_layers') for r in rows})}
    if 'rss_after_steps' in rows[0]:
        out['rss_after_steps_mib'] = [round(r['rss_after_steps'], 1) for r in rows]
        out['slope_rss_after_steps_mib_per_reset_from_reset_1'] = slope([r['rss_after_steps'] for r in rows])
        out['malloc_trim_released_mib_mean'] = float(np.mean([r['rss_after_gc']-r['rss_after_malloc_trim'] for r in rows]))
        out['reset_wall_s'] = [round(r['reset_wall_s'], 2) for r in rows]
    if 'gc_objects' in rows[0]:
        objects = [r['gc_objects'] for r in rows]
        out['gc_objects_growth_per_reset_from_reset_1'] = slope(objects)
        out['gc_type_growth_last_reset'] = rows[-1].get('gc_type_growth', [])[:8]
    if 'tracemalloc_traced_mib' in rows[0]:
        traced = [r['tracemalloc_traced_mib'] for r in rows]
        out['python_traced_heap_mib'] = [round(v, 2) for v in traced]
        out['python_traced_heap_slope_mib_per_reset_from_reset_3'] = slope(traced, 3)
        out['note'] = ('tracemalloc snapshots themselves are retained between resets (reset 1-2 RSS); the traced '
                       'heap slope from reset 3 excludes them')
    return out


def setup_diff(runtime_dir):
    """Setup receipt keys that differ between reset 0 and each later reset (wall times excluded)."""
    def flat(node, prefix=''):
        out = {}
        if isinstance(node, dict):
            for k, v in node.items():
                out.update(flat(v, f'{prefix}/{k}'))
        elif isinstance(node, list):
            for i, v in enumerate(node):
                out.update(flat(v, f'{prefix}[{i}]'))
        else:
            out[prefix] = node
        return out
    base = flat(load(runtime_dir/'setup.json'))
    rows = []
    for path in sorted(runtime_dir.glob('reset_setup_*.json')):
        other = flat(load(path))
        keys = [k for k in sorted(set(base) | set(other)) if base.get(k) != other.get(k)
                and 'wall' not in k and not k.startswith('/stage_reset')]
        rows.append({'reset': path.stem, 'differing_keys': keys})
    return rows


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def dataset_summary(directory):
    directory = ROOT/Path(directory) if not Path(directory).is_absolute() else Path(directory)
    if not (directory/'index.json').exists():
        return {'dir': rel(directory), 'status': 'MISSING'}
    index, manifest = load(directory/'index.json'), load(directory/'dataset_manifest.json')
    out = {'dir': rel(directory), 'git_sha': manifest.get('git_sha'), 'git_dirty': manifest.get('git_dirty'),
           'command': ' '.join(manifest.get('command', [])), 'episodes': []}
    for row in index['episodes']:
        path = directory/row['dir']/'episode.json'
        if not path.exists():
            out['episodes'].append(row)
            continue
        meta = load(path)
        result = meta.get('task_result') or {}
        item = {k: meta.get(k) for k in ('episode_id', 'label', 'controller', 'seed', 'verdict', 'expect', 'wall_s',
                                         'termination_reason', 'acceptance_valid', 'steps_recorded', 'task_steps')}
        item['home_outcome'] = (meta.get('home_outcome') or {}).get('status') if isinstance(meta.get('home_outcome'), dict) else meta.get('home_outcome')
        item['host_rss_mib_after'] = meta['memory_after']['host_rss_mib']
        item['gpu_used_mib_after'] = meta['memory_after']['gpu_used_mib_total']
        item['reset_wall_s'] = (meta.get('reset_info') or {}).get('reset_wall_s')
        item['items'] = result.get('items') if isinstance(result.get('items'), list) else None
        outcome = result.get('outcome') if isinstance(result.get('outcome'), dict) else {}
        item['outcome_status'] = outcome.get('status')
        for key in ('removed_mass_fraction', 'substrate_share', 'piece_mass_fractions'):
            if key in result:
                item[key] = result[key]
        item['reset_checks'] = {k: meta['reset_checks'].get(k) for k in
                                ('tool_pose_world_xyzw', 'edge_pose_world_xyzw', 'q_active', 'grasp_reset_status',
                                 'tool_linear_speed_m_s', 'tool_angular_speed_rad_s', 'material_codes_sha256',
                                 'particles', 'labels_clean', 'material_at_reference_max_m')}
        if meta.get('profile'):
            item['profile'] = meta['profile']
        out['episodes'].append(item)
    for name in ('reliability.json', 'reliability_phaseG_reliability_v1.json', 'reliability_phaseG_reliability_v1a.json'):
        if (directory/name).exists():
            report = load(directory/name)
            out[name.replace('.json', '')] = {k: report.get(k) for k in ('version', 'POLICY_READY', 'failing', 'cross_episode',
                                                                     'memory', 'determinism', 'wall_s')}
    runtime = directory/'runtime'
    if (runtime/'setup.json').exists():
        out['reset_setup_differences_vs_reset_0'] = setup_diff(runtime)
    episodes = [e for e in out['episodes'] if 'reset_checks' in e]
    if episodes:
        first = episodes[0]['reset_checks']
        out['reset_repeat_vs_episode_0'] = [
            {'episode_id': e['episode_id'],
             **{f'{k}_identical': e['reset_checks'][k] == first[k] for k in
                ('tool_pose_world_xyzw', 'edge_pose_world_xyzw', 'q_active', 'material_codes_sha256',
                 'tool_linear_speed_m_s', 'tool_angular_speed_rad_s')}}
            for e in episodes[1:]]
    return out


def profile_summary(directory, episode='episode_000'):
    path = Path(directory)/episode/'profile.json'
    if not path.exists():
        return None
    doc = load(path)
    keep = {'control_steps', 'control_step_total_s', 'wall_s', 'device', 'mode', 'runtime', 'kernel_total_s', 'sync_device'}
    out = {k: doc.get(k) for k in keep}
    out['per_control_step'] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                               for k, v in sorted(doc['per_control_step'].items(), key=lambda kv: -kv[1]['mean_ms'])}
    out['episode_level'] = {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                            for k, v in sorted(doc['episode_level'].items(), key=lambda kv: -kv[1]['total_s'])}
    if 'kernel_groups' in doc:
        out['kernel_groups'] = doc['kernel_groups']
        out['kernels_top'] = dict(list(doc['kernels'].items())[:12])
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--native-run', action='append', default=[], help='Phase H Native dataset directory name(s)')
    p.add_argument('--kinematic-run', action='append', default=[], help='Phase H kinematic dataset directory name(s)')
    p.add_argument('--extra', type=Path, help='JSON with hand-entered context (commits, commands, notes) to merge')
    a = p.parse_args(argv)
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; evidence is never overwritten')
    evidence = {'evidence': 'WP-11 Phase H: Native reset memory, profiling, safe optimizations, runtime profiles',
                'h1_probes': [probe_row(n, d) for n, d in PROBES],
                'h1_crashed_probes': [{'dir': f'artifacts/wp11/phaseH/{n}', 'description': d,
                                       'log': f'artifacts/wp11/logs/phaseH_{n}.log',
                                       'reached_resets': (len(load(H/n/'probe.json')['resets'])
                                                          if (H/n/'probe.json').exists() else 0)} for n, d in CRASHED],
                'phaseG_baselines': {name: dataset_summary(G/name) for name in
                                     ('native_memory_probe_001', 'native_reliability_trim_001')},
                'native_runs': {name: dataset_summary(H/name) for name in a.native_run},
                'kinematic_runs': {}}
    for name in a.native_run:
        prof = profile_summary(H/name)
        if prof:
            evidence['native_runs'][name]['profile_episode_000'] = prof
    for name in a.kinematic_run:
        summary = dataset_summary(H/name)
        summary['profile'] = profile_summary(H/name)
        evidence['kinematic_runs'][name] = summary
    if a.extra:
        evidence.update(load(a.extra))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(evidence, indent=1, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v)) + '\n')
    print(f'wrote {a.out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
