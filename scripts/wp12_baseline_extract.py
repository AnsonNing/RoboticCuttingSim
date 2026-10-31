#!/usr/bin/env python3
"""WP-12 Phase 0.3/0.4(d): compact baseline package, views and surface metrics of one captured run (read-only).

Reads a run directory written by ``scripts/wp12_baseline_capture.py`` (the unchanged WP-11 runner output plus
``wp12_capture.npz``/``.json``) and writes three files into a NEW subdirectory ``wp12_extract_NNN`` (the next
free number; earlier attempts are kept):

* ``baseline_compact.npz``: initial/final particle positions (world and solver-local), reference positions,
  lattice, final cut codes and final mechanical component labels (actual eight-node stencil graph,
  ``cut_topology.mechanical_components[_packed]``), final F, the initial/final ParticleSurface meshes
  (solver-local), per-control-step blade poses (world) and cut digit, and the solver contact/plow diagnostics;
* ``baseline_summary.json``: scene id, seed/config/command, verdict, wall times, diagnostics summary,
  ``surface_metrics_v1`` cut-face metrics and mesh validity, and (with ``--frozen``) a comparison with the
  frozen WP-11 run of the same protocol row;
* ``baseline_views.png``: top and side views of the final particles by mechanical component, a cut-face
  close-up of the final ParticleSurface mesh (shaded by triangle normal) and a thin-slab section with the
  cut-link midpoints.

  .venv/bin/python scripts/wp12_baseline_extract.py --run artifacts/wp12/baseline/s3_testb_s1502_002 \
      --scene-id S3_TEST_B_s1502 --frozen artifacts/wp11/phaseE/round1_kinematic_001/E1_TEST_B_s1502
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import surface_metrics_v1 as SM  # noqa: E402
from robotic_cutting_sim.cut_topology import mechanical_components, mechanical_components_packed  # noqa: E402

PALETTE = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
OTHER = '#9a9a96'


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def components(state, codes):
    encoding = str(state['code_encoding']) if 'code_encoding' in state.files else 'ternary'
    cuts = int(state['max_cuts']) if 'max_cuts' in state.files else 1
    function = mechanical_components_packed if encoding == 'packed2' else mechanical_components
    count, labels = function(state['reference_stencil_nodes'], state['reference_stencil_weights'], codes, cuts,
                             gradients=state['reference_stencil_gradients'])
    # Relabel by decreasing mass so colours follow piece size, not solver order.
    mass = np.bincount(labels, weights=state['mass_kg'], minlength=count)
    order = np.argsort(-mass, kind='stable')
    rank = np.empty(count, int)
    rank[order] = np.arange(count)
    return int(count), rank[labels], (mass[order]/mass.sum()), encoding, cuts


def to_world(T, points):
    return points@T[:3, :3].T+T[:3, 3]


def pose_to_world(T, poses):
    from scipy.spatial.transform import Rotation
    out = np.empty_like(poses, dtype=np.float64)
    out[:, :3] = to_world(T, poses[:, :3].astype(float))
    out[:, 3:] = (Rotation.from_matrix(T[:3, :3])*Rotation.from_quat(poses[:, 3:].astype(float))).as_quat()
    return out


def verdict(result):
    row = {k: result.get(k) for k in ('class', 'class_reason', 'verdict', 'status') if k in result}
    evaluation = result.get('evaluation')
    if isinstance(evaluation, dict):
        row['evaluation_verdict'] = evaluation.get('verdict')
        row['evaluator_version'] = evaluation.get('evaluator_version')
        outcome = evaluation.get('outcome') or {}
        row['evaluation_outcome_checks'] = outcome.get('checks')
    if 'checks' in result:
        row['checks'] = result['checks']
    if isinstance(result.get('metrics'), dict):
        row['metrics'] = {k: v for k, v in result['metrics'].items() if not isinstance(v, (list, dict)) or k in
                          ('final_component_mass_fractions', 'piece_mass_fractions')}
    return row


def wall(result):
    w = result.get('wall_s')
    return w if isinstance(w, dict) else {'total': w}


def shade(normals, base_rgb):
    light = np.array([.35, -.45, .82])
    light /= np.linalg.norm(light)
    intensity = .35+.65*np.abs(normals@light)
    return np.clip(np.asarray(base_rgb)[None]*intensity[:, None], 0, 1)


def hex_rgb(h):
    return np.array([int(h[i:i+2], 16)/255 for i in (1, 3, 5)])


def colour(labels):
    return np.array([PALETTE[l] if l < len(PALETTE) else OTHER for l in labels])


def render(path, title, positions, labels, count, mesh_v, mesh_f, mesh_labels, locus, window_centre, normal):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(16, 9.5), dpi=110, facecolor='#fcfcfb')
    rng = np.random.default_rng(0)
    take = rng.choice(len(positions), min(len(positions), 60000), replace=False)
    p, c = positions[take]*1e3, colour(labels[take])
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(p[:, 0], p[:, 1], s=1.2, c=c, linewidths=0)
    ax.set_aspect('equal'); ax.set_title('Top view (solver-local x-y, mm), final particles by mechanical component', fontsize=9)
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(p[:, 0], p[:, 2], s=1.2, c=c, linewidths=0)
    ax.set_aspect('equal'); ax.set_title('Side view (local x-z, mm)', fontsize=9)
    handles = [plt.Line2D([], [], marker='o', ls='', color=PALETTE[i] if i < len(PALETTE) else OTHER,
                          label=f'component {i}' if i < len(PALETTE) else 'other') for i in range(min(count, 9))]
    ax.legend(handles=handles, fontsize=7, loc='upper right', frameon=False)
    if window_centre is not None:
        half = .008
        centroid = mesh_v[mesh_f].mean(1)
        inside = np.all(np.abs(centroid-window_centre) <= half, 1)
        faces = mesh_f[inside]
        if len(faces) > 40000:
            faces = faces[rng.choice(len(faces), 40000, replace=False)]
        tri = mesh_v[faces]*1e3
        n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
        base = np.array([hex_rgb(PALETTE[l]) if l < len(PALETTE) else hex_rgb(OTHER) for l in mesh_labels[faces[:, 0]]])
        ax = fig.add_subplot(2, 2, 3, projection='3d')
        ax.add_collection3d(Poly3DCollection(tri, facecolors=shade(n, np.ones(3))*base, edgecolors='none'))
        lo, hi = (window_centre-half)*1e3, (window_centre+half)*1e3
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1, 1, 1))
        elev = float(np.degrees(np.arcsin(np.clip(normal[2], -1, 1))))
        azim = float(np.degrees(np.arctan2(normal[1], normal[0])))
        ax.view_init(elev=np.clip(elev+20, -80, 80), azim=azim+35)
        ax.set_title('Cut-face close-up: final ParticleSurface (current_volume_v1), 16 mm window, local mm', fontsize=9)
        ax.tick_params(labelsize=6)
        ax = fig.add_subplot(2, 2, 4)
        slab = np.abs(positions[:, 1]-window_centre[1]) <= .0006
        box = slab & np.all(np.abs(positions[:, [0, 2]]-window_centre[[0, 2]]) <= half, 1)
        ax.scatter(positions[box, 0]*1e3, positions[box, 2]*1e3, s=9, c=colour(labels[box]), linewidths=0)
        m = locus['side_points'].reshape(-1, 3)
        sel = (np.abs(m[:, 1]-window_centre[1]) <= .0006) & np.all(np.abs(m[:, [0, 2]]-window_centre[[0, 2]]) <= half, 1)
        ax.scatter(m[sel, 0]*1e3, m[sel, 2]*1e3, s=6, c='#0b0b0b', marker='x', linewidths=.6,
                   label='side locus x + F(X_nb - X)/2')
        ax.set_aspect('equal'); ax.legend(fontsize=7, loc='upper right', frameon=False)
        ax.set_title('Section |y - y_c| <= 0.6 mm: particles by component, cut-link side locus (local x-z, mm)', fontsize=9)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--scene-id', required=True)
    p.add_argument('--frozen', type=Path, help='Frozen WP-11 run directory of the same protocol row (comparison only)')
    p.add_argument('--note', default='')
    a = p.parse_args()
    run = a.run
    index = 1
    while (run/f'wp12_extract_{index:03d}').exists():
        index += 1
    folder = run/f'wp12_extract_{index:03d}'
    folder.mkdir()  # a new directory per attempt; earlier attempts are never overwritten
    outputs = [folder/'baseline_compact.npz', folder/'baseline_summary.json', folder/'baseline_views.png']
    start = time.monotonic()
    state = np.load(run/'physics_state.npz')
    capture = np.load(run/'wp12_capture.npz')
    meta = json.loads((run/'wp12_capture.json').read_text())
    result = json.loads((run/'result.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    T = capture['T_world_local']
    codes = capture['final_codes']
    if not np.array_equal(codes, state['codes'][-1]):
        raise SystemExit('Final captured codes differ from the last recorded history frame')
    count, labels, fractions, encoding, cuts = components(state, codes)
    local = capture['final_positions'].astype(float)
    reference_local = capture['reference_local'].astype(float)
    F = capture['final_F'].astype(float)
    mesh_v, mesh_f = capture['final_surface_vertices_local'], capture['final_surface_faces']
    _, nearest = __import__('scipy.spatial', fromlist=['cKDTree']).cKDTree(local).query(mesh_v)
    mesh_labels = labels[nearest]
    t0 = time.monotonic()
    metrics = SM.cut_face_metrics(reference_local, capture['lattice'], local, codes, mesh_v, mesh_f, encoding=encoding,
                                  cuts=cuts, F=F, component_labels=labels)
    metrics_wall = time.monotonic()-t0
    validity_final = SM.mesh_validity(mesh_v, mesh_f)
    validity_initial = (SM.mesh_validity(capture['initial_surface_vertices_local'], capture['initial_surface_faces'])
                        if 'initial_surface_faces' in capture.files else None)
    locus = SM.cut_locus(reference_local, capture['lattice'], local, codes, encoding=encoding, cuts=cuts, F=F)
    per_digit = {}
    for digit in range(cuts):
        mask = locus['cut_digit'] == digit
        if mask.any():
            per_digit[str(digit)] = {'cut_links': int(mask.sum()),
                                     'midpoint_median_local_m': np.median(locus['midpoint'][mask], 0).tolist(),
                                     'mean_link_direction_local': (locus['direction_current'][mask].mean(0)).tolist()}
    centre, normal = None, np.array([1., 0, 0])
    if per_digit:
        best = max(per_digit, key=lambda k: per_digit[k]['cut_links'])
        mask = locus['cut_digit'] == int(best)
        side = locus['side_points'][mask].reshape(-1, 3)
        side_label = labels[locus['pairs'][mask].reshape(-1)]
        # Centre the close-up on the side locus of the largest piece touched by this cut (substrate for a TRIM).
        centre = np.median(side[side_label == side_label.min()], 0)
        normal = np.asarray(per_digit[best]['mean_link_direction_local'])
        normal /= max(np.linalg.norm(normal), 1e-12)
    step = {k[5:]: capture[k] for k in capture.files if k.startswith('step_')}
    stats = step.get('stats')
    armed = step['cut'] >= 0 if 'cut' in step else np.zeros(0, bool)
    diagnostics = {
        'control_steps': int(len(step.get('cut', []))), 'armed_steps': int(armed.sum()),
        'steps_with_tool_contact': int((stats[:, 1] > 0).sum()) if stats is not None else None,
        'max_tool_penetration_m': float(stats[:, 3].max()) if stats is not None else None,
        'max_fixture_or_table_penetration_m': float(stats[:, 0].max()) if stats is not None else None,
        'max_particle_overlap_m': float(stats[:, 6].max()) if stats is not None else None,
        'armed_steps_with_law4_motion_last_substep': int((step['segment_ok'][armed] > 0).sum()) if armed.any() else 0,
        'armed_steps_with_law8_plow_last_substep': int((step['segment_plow'][armed] > 0).sum()) if armed.any() else 0,
        'armed_steps_engaged_last_substep': int((step['engaged'][armed] > 0).sum()) if armed.any() else 0,
        'final_labelled_particles': int(np.count_nonzero(codes)),
        'peak_reaction_force_N': float(np.linalg.norm(step['reaction_impulse_local'][:, :3], axis=1).max()
                                       / meta['control_dt_s']) if 'reaction_impulse_local' in step else None,
        'per_substep_pose_rule': 'coupled_robot.interpolate_measured_poses(pose_start_local, pose_end_local, '
                                 f'{meta.get("substeps")}, {meta.get("control_dt_s")}) (linear position, slerp rotation)'}
    poses_world = pose_to_world(T, step['pose_end_local']) if 'pose_end_local' in step else None
    initial_world = state['positions'][0]
    np.savez_compressed(
        outputs[0], scene_id=a.scene_id,
        initial_positions_world=initial_world.astype(np.float32), final_positions_world=state['positions'][-1].astype(np.float32),
        final_positions_local=local.astype(np.float32), reference_positions_world=state['reference_positions'].astype(np.float32),
        reference_positions_local=reference_local.astype(np.float32), lattice=capture['lattice'], mass_kg=state['mass_kg'],
        initial_codes=state['codes'][0], final_codes=codes, code_encoding=encoding, max_cuts=cuts,
        final_component_labels=labels.astype(np.int32), final_F=F.astype(np.float32),
        final_plastic_strain=capture['final_plastic_strain'], final_cut_progress=capture['final_cut_progress'],
        final_surface_vertices_local=mesh_v, final_surface_faces=mesh_f,
        initial_surface_vertices_local=capture['initial_surface_vertices_local'] if 'initial_surface_vertices_local' in capture.files else np.zeros((0, 3), np.float32),
        initial_surface_faces=capture['initial_surface_faces'] if 'initial_surface_faces' in capture.files else np.zeros((0, 3), np.int32),
        T_world_local=T, step_time_s=step['time_s'], step_cut=step['cut'], blade_pose_end_world=poses_world.astype(np.float32),
        blade_pose_start_local=step['pose_start_local'], blade_pose_end_local=step['pose_end_local'],
        step_stats=stats, step_segment_ok=step['segment_ok'], step_segment_plow=step['segment_plow'],
        step_stroke=step['stroke'], step_engaged=step['engaged'], step_labelled=step['labelled'],
        step_reaction_impulse_local=step['reaction_impulse_local'],
        **{k: state[k] for k in state.files if k.startswith('law4_')})
    title = (f'{a.scene_id}  |  {manifest.get("config", {}).get("task_id") or result.get("task_id")}  |  '
             f'{count} mechanical components  |  {verdict(result).get("class") or verdict(result).get("verdict")}')
    render(outputs[2], title, local, labels, count, mesh_v, mesh_f, mesh_labels, locus, centre, normal)
    frozen = None
    if a.frozen is not None:
        f_result = json.loads((a.frozen/'result.json').read_text())
        frozen = {'dir': str(a.frozen), 'verdict': verdict(f_result), 'wall_s': wall(f_result),
                  'note': 'GPU float32 atomic accumulation is nondeterministic (Phase H); verdict-level reproduction is the criterion'}
        if (a.frozen/'physics_state.npz').exists():
            f_state = np.load(a.frozen/'physics_state.npz')
            f_count, _, f_fractions, _, _ = components(f_state, f_state['codes'][-1])
            same = len(f_state['codes'][-1]) == len(codes)
            frozen.update(physics_state='present', final_components=f_count,
                          final_component_mass_fractions=f_fractions[:10].tolist(),
                          final_labelled_particles=int(np.count_nonzero(f_state['codes'][-1])),
                          final_code_agreement_fraction=float((f_state['codes'][-1] == codes).mean()) if same else None,
                          reference_positions_identical=bool(same and np.array_equal(f_state['reference_positions'],
                                                                                     state['reference_positions'])))
        else:
            frozen['physics_state'] = 'MISSING (only result.json/manifest.json/actions.jsonl remain); comparison is verdict-level'
    summary = {'schema': 'wp12_phase0_baseline_v1', 'scene_id': a.scene_id, 'note': a.note, 'run_dir': str(run),
               'wrapped_command': meta['wrapped_command'], 'wrapped_exit': meta['wrapped_exit'],
               'git': {k: manifest.get(k) for k in ('git_sha', 'git_dirty')},
               'config': {k: manifest.get('config', {}).get(k) for k in ('task_id', 'seed', 'family', 'device', 'material_seed',
                                                                          'cutting_law', 'solver_module', 'solver', 'row', 'max_cuts',
                                                                          'particle_spacing_m', 'grid_dx_m', 'material_dt_s', 'control_steps')},
               'verdict': verdict(result), 'wall_s': wall(result),
               'particles': int(len(codes)), 'code_encoding': encoding, 'max_cuts': cuts,
               'final_components': count, 'final_component_mass_fractions': fractions[:10].tolist(),
               'diagnostics': diagnostics, 'cut_locus_per_digit': per_digit,
               'surface': {'profile': meta.get('final_surface', {}).get('profile'),
                           'policy': meta.get('final_surface', {}).get('policy'),
                           'raise_error': meta.get('final_surface', {}).get('raise_error'),
                           'extract_wall_s': meta.get('final_surface', {}).get('wall_s'),
                           'mesh_components_by_ranges': len(meta.get('final_surface', {}).get('component_ranges') or []),
                           'validity_final': validity_final, 'validity_initial': validity_initial},
               'surface_metrics_v1': metrics, 'surface_metrics_wall_s': metrics_wall,
               'frozen_comparison': frozen, 'extract_wall_s': time.monotonic()-start,
               'files': {k.name: k.stat().st_size for k in [*outputs[::2], run/'physics_state.npz', run/'wp12_capture.npz']}}
    outputs[1].write_text(json.dumps(jsonable(summary), indent=1))
    print(json.dumps(jsonable({k: summary[k] for k in ('scene_id', 'verdict', 'final_components', 'extract_wall_s')}), indent=1))


if __name__ == '__main__':
    main()
