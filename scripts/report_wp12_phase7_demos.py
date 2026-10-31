#!/usr/bin/env python3
"""WP-12 Phase 7: read-only analysis, media and evidence for the Native demo runs of
``scripts/run_wp12_phase7_demos.py`` (configs/wp12/phase7_demos_v1.json).

Nothing here re-simulates or changes a recording, physics, an evaluator or a threshold. Outputs are new files or
directories only (existing outputs are refused, never overwritten).

* ``analyze --run ROOT/DEMOk_X``: the unchanged Phase 2b analysis tool
  (``scripts/run_wp12_workcell_live_analysis.py``; class = ``live_classification`` / ``classify_live`` with the
  grasp warning gates and the task-phase IK / workspace-clamp counts) into ``.../analysis_001``, then the geometry
  block into ``.../geometry_001/geometry.json``.
* ``media --root ROOT --demo DEMOk --media docs/media/wp12/demos/<name>``: per profile the recorded start / progress
  / final close-ups and camera stills (``export_wp12_workcell_media.py``) and an MP4 (``render_packed_episode.py``);
  then A|B side-by-side close-ups of the same view and frame, an offline cut-face close-up of the final staged
  specimen mesh (A | B) and a "geometric reconstruction vs actual particle state" section panel.
* ``summarize --root ROOT --out docs/evidence/wp12/phase7_demos_001.json``.

Geometry (report-only): the final staged specimen mesh ``/World/ContinuousSpecimen`` of ``final_live_scene.usda``
(WORLD; the rendered surface, never the collider) against the final recorded particle state
(``physics_state.npz`` last frame) with ``surface_metrics_v1.surface_report`` in current space. The Native record
holds no deformation gradient F, so the side locus is translation-only (F = identity), and the reference-space
pull-back is not computed. The locus is the labelled link midpoint for every run (comparable A vs B); B runs also
report the recorder Hermite locus (``cut_events_log.npy``; reference = workpiece-local lattice, as the recorder).
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
DEMOS = ROOT/'configs/wp12/phase7_demos_v1.json'
PROFILES = {'A': 'particle_surface_v1', 'B': 'cut_aware_surface_v1'}
MEDIA_NAMES = {'DEMO1': 'demo1_clean_cut', 'DEMO2': 'demo2_oblique_slot_p20', 'DEMO3': 'demo3_trim_rho10',
               'DEMO4': 'demo4_four_slices', 'DEMO5': 'demo5_three_bumps', 'DEMO6': 'demo6_cut_push_trim'}
PY = str(ROOT/'.venv/bin/python')


def load(path):
    return json.loads(Path(path).read_text())


def write_new(path, payload):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'{path} exists; outputs are never overwritten')
    path.write_text(json.dumps(payload, indent=1, allow_nan=True,
                               default=lambda v: v.tolist() if isinstance(v, np.ndarray)
                               else v.item() if isinstance(v, np.generic) else str(v)))


def rel(path):
    path = Path(path)
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _abs(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT/path


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------------- geometry

def final_mesh(episode):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(episode/'final_live_scene.usda'))
    prim = stage.GetPrimAtPath('/World/ContinuousSpecimen')
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if len(counts) and not (counts == 3).all():
        raise ValueError('specimen mesh is not triangulated')
    xf = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()), float)
    v = np.asarray(mesh.GetPointsAttr().Get(), float)
    v = v@xf[:3, :3]+xf[3, :3]   # USD row-vector convention
    return v, np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), np.int64).reshape(-1, 3)


def final_state(episode):
    from robotic_cutting_sim.goals import packed_components
    with np.load(episode/'physics_state.npz') as z:
        d = {k: z[k] for k in ('time_s', 'mass_kg', 'reference_positions_local', 'reference_indices',
                               'reference_stencil_nodes', 'reference_stencil_weights', 'reference_stencil_gradients')}
        d['positions'], d['codes'] = z['positions'][-1].astype(float), z['codes'][-1]
        d['positions_first'] = z['positions'][0].astype(float)
    material = load(episode/'material.json')
    d['max_cuts'] = int(material.get('max_cuts', 1))
    d['count'], d['labels'] = packed_components(d['reference_stencil_nodes'], d['reference_stencil_weights'], d['codes'],
                                                gradients=d['reference_stencil_gradients'])
    return d


def _q(row, scale=1e3):
    if not isinstance(row, dict) or row.get('p95') is None:
        return None
    return {k: (row[k]*scale if row.get(k) is not None else None) for k in ('p50', 'p95', 'max')} | {'count': row.get('count')}


def compact_report(rep):
    if rep.get('status') != 'MEASURED':
        return {'status': rep.get('status'), 'topology': rep.get('topology'), 'mesh_validity': rep.get('mesh_validity')}
    top = rep['topology']
    return {'status': 'MEASURED', 'locus_source': rep['locus_source'], 'locus_rows_by_source': rep['locus_rows_by_source'],
            'conflicting_link_rows': rep['conflicting_link_rows'], 'vertices': rep['vertices'], 'triangles': rep['triangles'],
            'cut_face_triangles': rep['cut_face_triangles'], 'cut_face_area_mm2': rep['cut_face_area_m2']*1e6,
            'distance_to_side_locus_plane_mm': _q(rep['distance_to_side_locus_plane']),
            'distance_to_side_locus_plane_core_mm': _q(rep['distance_to_side_locus_plane_core']),
            'locus_to_mesh_distance_mm': _q(rep['locus_to_mesh_distance']), 'coverage': rep['coverage'],
            'normal_error_deg_core': _q(rep['normal_error_deg_core'], 1.),
            'thickness_bias_mm': None if rep['signed_bias']['thickness_bias_m'] is None else rep['signed_bias']['thickness_bias_m']*1e3,
            'roughness_mesh_core': rep.get('roughness_mesh_core'), 'roughness_locus_core': rep.get('roughness_locus_core'),
            'rim': rep.get('rim'),
            'topology': {k: top.get(k) for k in ('mesh_shells', 'mechanical_components', 'shells_equal_components',
                                                 'shells_spanning_components', 'components_without_shell',
                                                 'components_without_shell_mass_kg', 'faces_nearer_other_component')},
            'mesh_validity': rep.get('mesh_validity'),
            'per_cut': {k: {'cut_face_vertices': c['cut_face_vertices'],
                            'distance_to_side_locus_plane_mm': _q(c['distance_to_side_locus_plane']),
                            'coverage': c['coverage'], 'normal_error_deg_core': _q(c['normal_error_deg_core'], 1.)}
                        for k, c in rep['per_cut'].items()}}


def geometry(run_root):
    from robotic_cutting_sim import surface_metrics_v1 as SM
    episode = run_root/'run'/'episode_000'
    out = {'scope': __doc__.split('Geometry (report-only): ', 1)[1].strip(), 'run': rel(run_root)}
    try:
        v, f = final_mesh(episode)
        d = final_state(episode)
    except Exception as error:
        return {**out, 'status': 'ERROR', 'error': repr(error)}
    out.update(final_frame_time_s=float(d['time_s'][-1]), mechanical_components=int(d['count']),
               mesh_vertices=int(len(v)), mesh_triangles=int(len(f)))
    common = dict(spacing=.001, labels=d['labels'], masses=d['mass_kg'], cuts=d['max_cuts'], reference_space=False)
    ref = d['reference_positions_local'].astype(float)
    for name, events in (('midpoint_locus', None), ('hermite_locus', 'cut_events_log.npy')):
        if events is not None and not (episode/events).exists():
            continue
        try:
            log = None if events is None else np.load(episode/events)
            rep = SM.surface_report(ref, d['reference_indices'], d['positions'], d['codes'], v, f, events=log, **common)
            out[name] = compact_report(rep)
        except Exception as error:
            out[name] = {'status': 'ERROR', 'error': repr(error)}
    out['status'] = 'MEASURED'
    return out


def cmd_analyze(args):
    run_root = _abs(args.run)
    analysis = run_root/'analysis_001'
    if not analysis.exists():
        cmd = [PY, 'scripts/run_wp12_workcell_live_analysis.py', '--root', rel(run_root)]
        print(' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=False)
    gdir = run_root/'geometry_001'
    if not gdir.exists():
        gdir.mkdir()
        write_new(gdir/'geometry.json', {**geometry(run_root), 'command': sys.argv,
                                         'created': datetime.now().astimezone().isoformat()})
        print(json.dumps({'geometry': rel(gdir/'geometry.json')}), flush=True)
    return 0


# ----------------------------------------------------------------------------- media

def _label_for(run_root):
    a = run_root/'analysis_001'/'analysis.json'
    if not a.exists():
        return 'analysis missing'
    a = load(a)
    home = ((a.get('home') or {}).get('home_outcome') or {}).get('status')
    return f"{a['class']} | verdict {(a.get('task_outcome') or {}).get('verdict')} | home {home}"


def _cut_focus(d, cut=0):
    """Current-space centre and reference normal of the labelled cut-``cut`` locus (link midpoints)."""
    from robotic_cutting_sim import surface_metrics_v1 as SM
    locus = SM.material_locus(d['reference_positions_local'].astype(float), d['reference_indices'], d['positions'],
                              d['codes'], cuts=d['max_cuts'])
    rows = np.flatnonzero(locus['cut'] == cut)
    if not len(rows):
        rows = np.arange(len(locus['cut']))
    if not len(rows):
        return None
    # centre on the side points of the largest mechanical component (the substrate of a trim; either half of a sever)
    pts = locus['side_points'][rows].reshape(-1, 3)
    owner = d['labels'][locus['pairs'][rows].reshape(-1)]
    ids, counts = np.unique(d['labels'], return_counts=True)
    size = dict(zip(ids.tolist(), counts.tolist()))
    best = max(set(owner.tolist()), key=lambda c: size[c])
    n = locus['N'][rows].mean(0)
    n = n/max(np.linalg.norm(n), 1e-12)
    # the chosen piece's cut face looks along +N when it is on side 1 (pairs[:, 0]), along -N on side 2
    side1 = (owner.reshape(-1, 2)[:, 0] == best).sum() >= (owner.reshape(-1, 2)[:, 1] == best).sum()
    centre = np.median(pts[owner == best], 0)
    idx = locus['pairs'][rows].reshape(-1)[owner == best]
    anchor = int(idx[np.argmin(np.linalg.norm(d['positions'][idx]-centre, axis=1))])
    return {'centre': centre, 'normal': n, 'face_outward': n if side1 else -n, 'anchor_particle': anchor,
            'cut': int(cut), 'rows': int(len(rows)), 'centre_component_particles': int(size[best])}


def _section(v, f, y0):
    """Segments of the mesh cut by the plane y = y0 (x, z pairs)."""
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    s = np.stack([a[:, 1], b[:, 1], c[:, 1]], 1)-y0
    cross = (s.min(1) < 0) & (s.max(1) > 0)
    segs = []
    for tri, sd in zip(np.stack([a, b, c], 1)[cross], s[cross]):
        pts = []
        for i, j in ((0, 1), (1, 2), (2, 0)):
            if (sd[i] < 0) != (sd[j] < 0):
                t = sd[i]/(sd[i]-sd[j])
                p = tri[i]+t*(tri[j]-tri[i])
                pts.append((p[0], p[2]))
        if len(pts) == 2:
            segs.append(pts)
    return np.asarray(segs).reshape(-1, 2, 2)


def _component_colours(labels):
    from robotic_cutting_sim import demo_media as dm
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-counts)]
    palette = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3', '#937860', '#da8bc3', '#8c8c8c']
    lut = {int(c): palette[i] if i < len(palette)-1 else palette[-1] for i, c in enumerate(order)}
    _ = dm  # palette documented in demo_media; kept local for presentation
    return np.asarray([lut[int(x)] for x in labels])


def section_panel(runs, out_png, title):
    """Rows A, B: particle centres in the slab |y - y0| < 0.5 mm (coloured by mechanical component) and the final
    staged mesh's section with the plane y = y0; left the whole workpiece, right a 30 mm window on cut 0."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig, axes = plt.subplots(len(runs), 2, figsize=(16, 3.4*len(runs)), gridspec_kw={'width_ratios': [2.2, 1]},
                             squeeze=False)
    info = {}
    focus = y0 = None
    for row, (tag, run_root) in enumerate(runs):
        episode = run_root/'run'/'episode_000'
        v, f = final_mesh(episode)
        d = final_state(episode)
        if row == 0:   # one section plane and window for every row (from the first run's final state)
            focus = _cut_focus(d)
            y0 = float(focus['centre'][1]) if focus else float(np.median(d['positions'][:, 1]))
            # snap to a particle layer of the initial lattice so the slab holds one lattice layer
            ys = np.unique(np.round(d['positions_first'][:, 1], 4))
            y0 = float(ys[np.argmin(np.abs(ys-y0))])
        slab = np.abs(d['positions'][:, 1]-y0) < .0005
        p = d['positions'][slab]
        col = _component_colours(d['labels'])[slab]
        segs = _section(v, f, y0)
        cx, cz = (focus['centre'][0], focus['centre'][2]) if focus else (float(p[:, 0].mean()), float(p[:, 2].mean()))
        for k, ax in enumerate(axes[row]):
            ax.scatter(p[:, 0]*1e3, p[:, 2]*1e3, s=2 if k == 0 else 7, c=col, linewidths=0, alpha=.85)
            ax.add_collection(LineCollection(segs*1e3, colors='k', linewidths=.6 if k == 0 else 1.0))
            ax.set_aspect('equal')
            ax.set_xlabel('world x [mm]')
            ax.set_ylabel('world z [mm]')
            if k == 1:
                ax.set_xlim(cx*1e3-15, cx*1e3+15)
                ax.set_ylim(cz*1e3-12, cz*1e3+12)
                ax.set_title(f'{tag}: window on cut {focus["cut"] if focus else "-"}', fontsize=9)
            else:
                ax.set_title(f'{tag}: {PROFILES[tag[0]]} - section y = {y0*1e3:.1f} mm; dots = particle centres '
                             f'(colour = mechanical component, {d["count"]}), black = rendered mesh section', fontsize=9)
        info[tag] = {'y0_m': y0, 'particles_in_slab': int(slab.sum()), 'mesh_section_segments': int(len(segs)),
                     'components': int(d['count']), 'final_frame_time_s': float(d['time_s'][-1])}
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return info


def cutface_closeup(runs, out_png, title, half=14e-3):
    """Offline close-up of the final staged specimen mesh around cut 0, viewed about 25 deg off the cut-face
    normal: only the triangles of one piece (the mechanical component of an anchor particle next to the cut, chosen
    on the first run's state) inside a window are drawn, so the other piece does not hide the face. Lambert-shaded
    triangles; presentation only."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(7.2*len(runs), 6.4))
    info = {}
    focus = None
    for i, (tag, run_root) in enumerate(runs):
        episode = run_root/'run'/'episode_000'
        v, f = final_mesh(episode)
        d = final_state(episode)
        if i == 0:   # one view for every panel (from the first run's final state)
            focus = _cut_focus(d)
        ax = fig.add_subplot(1, len(runs), i+1, projection='3d')
        if focus is None:
            ax.set_title(f'{tag}: no conflicting links')
            continue
        from scipy.spatial import cKDTree
        c, n = focus['centre'], focus['face_outward']
        cen = v[f].mean(1)
        rel_c = cen-c
        # only the mesh of the chosen piece (the mechanical component of the anchor particle in THIS run's state;
        # the lattice is identical across runs), inside a window around the cut-face centre
        _, nearest = cKDTree(d['positions']).query(cen)
        keep = (d['labels'][nearest] == d['labels'][focus['anchor_particle']]) & (np.abs(rel_c).max(1) < half*1.6)
        lateral = rel_c-np.outer(rel_c@n, n)
        keep &= np.linalg.norm(lateral, axis=1) < half*1.4
        tri = v[f[keep]]
        nrm = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-18)
        up = np.array([0., 0., 1.]) if abs(n[2]) < .8 else np.array([1., 0., 0.])
        side = np.cross(up, n)
        side /= np.linalg.norm(side)
        view = n*np.cos(np.radians(25))+(up*np.sin(np.radians(20))+side*np.sin(np.radians(15)))
        view /= np.linalg.norm(view)
        light = view+.3*up
        light /= np.linalg.norm(light)
        shade = .25+.75*np.abs(nrm@light)
        colours = np.c_[np.outer(shade, [0.93, 0.80, 0.55]), np.ones(len(shade))]
        ax.add_collection3d(Poly3DCollection(tri*1e3, facecolors=np.clip(colours, 0, 1), edgecolors='none'))
        lo, hi = (c-half)*1e3, (c+half)*1e3
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=float(np.degrees(np.arcsin(np.clip(view[2], -1, 1)))),
                     azim=float(np.degrees(np.arctan2(view[1], view[0]))))
        ax.set_xlabel('x [mm]'); ax.set_ylabel('y [mm]'); ax.set_zlabel('z [mm]')
        ax.set_title(f'{tag}: {PROFILES[tag[0]]}\ncut {focus["cut"]} face, final staged mesh ({int(keep.sum())} triangles '
                     'shown: one piece only, window around the cut face)', fontsize=9)
        info[tag] = {'centre_world_m': c.tolist(), 'normal_reference': n.tolist(), 'triangles_shown': int(keep.sum()),
                     'view_direction': view.tolist()}
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=90)
    plt.close(fig)
    return info


def compose_ab(a_png, b_png, out_png, name):
    from PIL import Image, ImageDraw
    a, b = Image.open(a_png).convert('RGB'), Image.open(b_png).convert('RGB')
    if a.size != b.size:
        b = b.resize(a.size, Image.LANCZOS)
    band = 22
    canvas = Image.new('RGB', (a.width*2+6, a.height+band), (255, 255, 255))
    canvas.paste(a, (0, band))
    canvas.paste(b, (a.width+6, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 4), f'A particle_surface_v1 - {name}', fill=(0, 0, 0))
    draw.text((a.width+12, 4), f'B cut_aware_surface_v1 - {name}', fill=(0, 0, 0))
    canvas.save(out_png, optimize=True)


def cmd_media(args):
    root = _abs(args.root)
    media = _abs(args.media)
    media.mkdir(parents=True, exist_ok=False)
    demo = args.demo
    title = next(d['title'] for d in load(DEMOS)['demos'] if d['id'] == demo)
    receipt = {'demo': demo, 'root': rel(root), 'command': sys.argv, 'created': datetime.now().astimezone().isoformat(),
               'scope': 'Recorded live Native frames and the final staged mesh/particle state; presentation only. The '
                        'presentation close-up and overview cameras are never observations.', 'runs': {}}
    runs = []
    for tag, profile in PROFILES.items():
        run_root = root/f'{demo}_{tag}'
        episode = run_root/'run'/'episode_000'
        sub = media/profile
        entry = {'run': rel(run_root)}
        if not (episode/'frame_manifest.json').exists():
            entry['status'] = 'NO_EPISODE'
            receipt['runs'][tag] = entry
            continue
        subprocess.run([PY, 'scripts/export_wp12_workcell_media.py', '--root', rel(run_root), '--media', rel(sub)],
                       cwd=ROOT, check=True)
        render = run_root/'render_001'
        if not render.exists():
            label = _label_for(run_root)
            cmd = [PY, 'scripts/render_packed_episode.py', rel(episode), '--out', rel(render),
                   '--title', f'WP-12 Phase 7 {demo} {title} - {profile} (Isaac Sim Native, tlmpm_v5 law 8, seed 42)',
                   '--label', label]
            if demo in ('DEMO4', 'DEMO5', 'DEMO6'):
                cmd.append('--wide-piece-palette')
            subprocess.run(cmd, cwd=ROOT, check=True)
        mp4 = render/'demo.mp4'
        if mp4.exists():
            target = sub/'demo.mp4'
            if mp4.stat().st_size > args.max_mp4_mb*1e6:
                subprocess.run(['/usr/bin/ffmpeg', '-loglevel', 'error', '-n', '-i', str(mp4), '-c:v', 'libx264', '-crf', '30',
                                '-preset', 'slow', '-pix_fmt', 'yuv420p', '-an', str(target)], check=True)
                entry['mp4_reencoded_crf'] = 30
            else:
                shutil.copy2(mp4, target)
            entry['mp4'] = {'path': rel(target), 'bytes': target.stat().st_size, 'source': rel(mp4)}
        entry['status'] = 'EXPORTED'
        receipt['runs'][tag] = entry
        runs.append((tag, run_root))
    if len(runs) == 2:
        ab = {}
        for name in ('start_closeup', 'progress_closeup', 'final_closeup', 'final_overview'):
            a_png, b_png = media/PROFILES['A']/f'{name}.png', media/PROFILES['B']/f'{name}.png'
            if a_png.exists() and b_png.exists():
                out = media/f'ab_{name}.png'
                if out.exists():
                    raise FileExistsError(out)
                compose_ab(a_png, b_png, out, name)
                ab[name] = rel(out)
        receipt['ab'] = ab
    if runs:
        tags = [(f'{t} ({PROFILES[t]})', r) for t, r in runs]
        out = media/'ab_cutface_closeup.png'
        receipt['cutface_closeup'] = {'path': rel(out), 'info': cutface_closeup(
            [(t.split(' ')[0], r) for t, r in tags], out, f'{demo} {title}: cut-face close-up of the final staged mesh '
            '(render geometry, not the collider)')}
        out = media/'reconstruction_vs_particles.png'
        receipt['reconstruction_vs_particles'] = {'path': rel(out), 'info': section_panel(
            [(t.split(' ')[0], r) for t, r in tags], out,
            f'{demo} {title}: geometric reconstruction (rendered mesh section) vs actual particle state, final recorded frame')}
    receipt['files'] = {rel(p): p.stat().st_size for p in sorted(media.rglob('*')) if p.is_file()}
    receipt['total_bytes'] = sum(receipt['files'].values())
    write_new(media/'phase7_media_receipt.json', receipt)
    print(json.dumps({'media': rel(media), 'total_mb': receipt['total_bytes']/1e6}), flush=True)
    return 0


# ----------------------------------------------------------------------------- summary

def run_entry(run_root, demos_cfg):
    entry = {'root': rel(run_root)}
    attempt = load(run_root/'attempt.json') if (run_root/'attempt.json').exists() else None
    prep = load(run_root/'inputs'/'prepare.json') if (run_root/'inputs'/'prepare.json').exists() else None
    if attempt is None:
        entry['status'] = 'NOT_RUN' if not (run_root/'launch.json').exists() else 'CRASHED_OR_RUNNING'
        return entry
    episode = run_root/'run'/'episode_000'
    manifest = load(episode/'manifest.json') if (episode/'manifest.json').exists() else {}
    analysis = load(run_root/'analysis_001'/'analysis.json') if (run_root/'analysis_001'/'analysis.json').exists() else None
    geom = load(run_root/'geometry_001'/'geometry.json') if (run_root/'geometry_001'/'geometry.json').exists() else None
    SL = _module('wp12_sl_summary', ROOT/'scripts/summarize_wp12_surface_live.py')
    entry.update({
        'profile': attempt.get('profile'), 'seed': demos_cfg['common']['physics']['seed'],
        'commit': attempt.get('git_sha'), 'git_dirty': attempt.get('git_dirty'),
        'git_status_porcelain': attempt.get('git_status_porcelain'),
        'config_hashes': None if prep is None else {
            'demos_file_sha256': prep['demos_file_sha256'], 'source_scene_sha256': prep['source_scene_sha256'],
            'source_goal_sha256': prep['source_goal_sha256'], 'live_scene_sha256': sha256(run_root/'inputs'/'scene.json'),
            'live_goal_sha256': sha256(run_root/'inputs'/'goal.json'), 'live_config_sha256': prep['live_config_sha256'],
            'live_task_id': prep['live_task_id'], 'phase2b_task_id': prep.get('phase2b_task_id')},
        'inputs': None if prep is None else {k: prep[k] for k in ('duration_s', 'steps', 'history_stride', 'live_plan_end_s',
                                                                    'planner_overrides', 'max_cuts', 'particles', 'solver_module')},
        'versions': {'python': manifest.get('python'), 'packages': manifest.get('packages'), 'backend': manifest.get('backend')},
        'command': attempt.get('command'), 'returncode': attempt.get('returncode'), 'result_status': attempt.get('result_status'),
        'process_wall_s': attempt.get('wall_s'), 'started': attempt.get('started'), 'ended': attempt.get('ended'),
        'gpu_before': attempt.get('gpu_before'), 'gpu_after': attempt.get('gpu_after'), 'run_dir_bytes': attempt.get('run_dir_bytes'),
        'log': attempt.get('log')})
    if analysis is not None:
        a = analysis
        entry['class'] = a['class']
        entry['class_reason'] = a['class_reason']
        entry['verdict'] = (a.get('task_outcome') or {}).get('verdict')
        entry['failed_checks'] = (a.get('task_outcome') or {}).get('failed_checks')
        entry['task_id_match_cpu'] = a.get('task_id_match')
        entry['physical'] = {
            'goal_metrics': a.get('metrics'), 'outcome_metrics': (a.get('task_outcome') or {}).get('outcome_metrics'),
            'task_end_reason': a.get('task_end_reason'), 'plan_preflight': a.get('plan_preflight'), 'ik': a.get('ik'),
            'grasp_task_phase': {k: (a.get('grasp_task_phase') or {}).get(k) for k in ('grasp_lost', 'max_slip_m', 'max_rotation_rad')},
            'grasp_all': {k: (a.get('grasp_all') or {}).get(k) for k in ('grasp_lost', 'max_slip_m', 'max_rotation_rad')},
            'home_outcome': ((a.get('home') or {}).get('home_outcome') or {}).get('status'),
            'home_outcome_detail': (a.get('home') or {}).get('home_outcome'),
            'knife_vs_board': a.get('knife_vs_board'), 'material_vs_board': a.get('material_vs_board'),
            'forces': a.get('forces'), 'displacement': a.get('displacement'), 'bridges': a.get('bridges'),
            'measured_task_path_clearance': {k: (a.get('measured_task_path_clearance') or {}).get(k) for k in
                                             ('status', 'min_clearance_m', 'reason')},
            'finite_recorded_state': a.get('finite_recorded_state')}
        entry['analysis'] = rel(run_root/'analysis_001'/'analysis.json')
        entry['runtime'] = {'performance': a.get('performance')}
    else:
        entry['class'] = 'NOT_ANALYZED'
    entry.setdefault('runtime', {})['profiling'] = SL.profiling(run_root) if episode.exists() else None
    entry['geometry'] = geom
    entry['surface_provenance'] = SL.surface_summary(run_root) if (episode/'surface_provenance.json').exists() else None
    if 'B' == run_root.name.rsplit('_', 1)[-1] and entry['surface_provenance'] is None:
        entry['surface_provenance'] = {'status': 'MISSING'}
    return entry


def _leaves(tree, prefix=''):
    """Numeric leaves of nested dicts and lists (lists of dicts included, e.g. sequence step metrics)."""
    out = {}
    if isinstance(tree, dict):
        for key, value in tree.items():
            if key != 'definitions':
                out.update(_leaves(value, f'{prefix}.{key}' if prefix else str(key)))
    elif isinstance(tree, list):
        for i, value in enumerate(tree):
            out.update(_leaves(value, f'{prefix}[{i}]'))
    elif isinstance(tree, (int, float)) and not isinstance(tree, bool) and tree is not None:
        out[prefix] = float(tree)
    elif isinstance(tree, bool):
        out[prefix] = tree
    return out


def goal_metric_diffs(a_root, b_root):
    """Every goal-metric leaf of A and B (analysis.json ``metrics``, sequence steps included): |B - A|, booleans equal."""
    a = _leaves(load(a_root/'analysis_001/analysis.json').get('metrics'))
    b = _leaves(load(b_root/'analysis_001/analysis.json').get('metrics'))
    rows = {}
    for key in sorted(set(a) | set(b)):
        va, vb = a.get(key), b.get(key)
        if isinstance(va, bool) or isinstance(vb, bool):
            rows[key] = {'a': va, 'b': vb, 'equal': va == vb}
        elif va is None or vb is None:
            rows[key] = {'a': va, 'b': vb, 'equal': False}
        else:
            rows[key] = {'a': va, 'b': vb, 'abs_diff': abs(vb-va)}
    return rows


def cmd_summarize(args):
    root = _abs(args.root)
    out = _abs(args.out)
    if out.exists():
        raise FileExistsError(out)
    demos_cfg = load(DEMOS)
    SL = _module('wp12_sl_summary2', ROOT/'scripts/summarize_wp12_surface_live.py')
    tolerance = 0.02
    demos = {}
    for spec in demos_cfg['demos']:
        demo = spec['id']
        if 'scene' not in spec:
            demos[demo] = {'status': 'NOT_RUN', 'reason': 'Demo 7 comes from the Phase 5 held-out selection (later)'}
            continue
        runs = {tag: run_entry(root/f'{demo}_{tag}', demos_cfg) for tag in PROFILES}
        pair = None
        a_root, b_root = root/f'{demo}_A', root/f'{demo}_B'
        if (a_root/'analysis_001/analysis.json').exists() and (b_root/'analysis_001/analysis.json').exists():
            try:
                pair = SL.physics_comparison(b_root, a_root, tolerance)
                pair['rule'] = ('surface_live_v1 physics_unchanged_rule applied to the pair (B vs A on identical inputs): '
                                'task id, class, verdict, piece counts equal; fraction/removal metrics within 0.02 abs; '
                                'labelled particles and conflicting links reported. The "phase2b" keys here mean run A.')
            except Exception as error:
                pair = {'status': 'ERROR', 'error': repr(error)}
        if pair is not None and pair.get('status') in ('PASS', 'FAIL'):
            pair['goal_metrics_all_leaves'] = goal_metric_diffs(a_root, b_root)
        demos[demo] = {'title': spec['title'], 'scene': spec['scene'], 'goal': spec['goal'],
                       'scene_sha256': spec['scene_sha256'], 'goal_sha256': spec['goal_sha256'],
                       'configuration': spec.get('configuration'), 'label': spec.get('label'),
                       'describe_as': spec.get('describe_as'), 'disclose': spec.get('disclose'),
                       'evidence_before': spec.get('evidence_now'), 'runs': runs, 'pair_physics_agreement': pair,
                       'media': rel(ROOT/'docs/media/wp12/demos'/MEDIA_NAMES[demo])}
    git = lambda *x: subprocess.run(['git', *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()  # noqa: E731
    payload = {'schema': 'rcs_wp12_phase7_demos_evidence_v1', 'demos_file': rel(DEMOS), 'demos_file_sha256': sha256(DEMOS),
               'created': datetime.now().astimezone().isoformat(), 'git_sha': git('rev-parse', 'HEAD'),
               'git_dirty': bool(git('status', '--porcelain')), 'command': sys.argv, 'root': rel(root),
               'success_definition': demos_cfg['common']['success'],
               'scope': demos_cfg['scope'], 'demos': demos}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_new(out, payload)
    print(json.dumps({d: {t: (v.get('runs') or {}).get(t, {}).get('class') for t in PROFILES} if 'runs' in v else v['status']
                      for d, v in demos.items()}, indent=1))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('analyze')
    s.add_argument('--run', required=True)
    s.set_defaults(func=cmd_analyze)
    s = sub.add_parser('media')
    s.add_argument('--root', required=True)
    s.add_argument('--demo', required=True)
    s.add_argument('--media', required=True)
    s.add_argument('--max-mp4-mb', type=float, default=2.6)
    s.set_defaults(func=cmd_media)
    s = sub.add_parser('summarize')
    s.add_argument('--root', required=True)
    s.add_argument('--out', required=True)
    s.set_defaults(func=cmd_summarize)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
