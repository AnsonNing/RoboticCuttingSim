"""Resumable full-episode export from original live frames; agentic control robot research."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def export_success(recording_dir):
    """All original capture times, three allowed modalities, no physics rerun.

    Encoding failure leaves the successful physical outcome and raw data intact.
    Re-entry reuses a validated manifest, or uses a fresh export attempt name.
    """
    from PIL import Image, ImageDraw
    from .agentic_trim_env import write_json
    rec = Path(recording_dir)
    result = json.loads((rec/'independent_evaluation.json').read_text())
    if not result.get('success'):
        raise ValueError('Only independently successful recordings enter the success manifest')
    provenance = json.loads((rec/'provenance.json').read_text())
    episode_id = f"s{provenance['scene_seed']}_{hashlib.sha256(str(rec.resolve()).encode()).hexdigest()[:12]}_{rec.name}"
    manifest_file = rec/'success_manifest.json'
    if manifest_file.exists():
        prior = json.loads(manifest_file.read_text())
        if (rec/prior['mp4_path']).is_file() and sha256(rec/prior['mp4_path']) == prior['mp4_sha256']:
            subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(rec/prior['mp4_path']),
                            '-f', 'null', '-'], check=True, timeout=180)
            return prior
        raise ValueError('Existing success manifest does not match its media; preserve and investigate')
    rows = json.loads((rec/'frame_manifest.json').read_text())
    times = np.asarray([row['time_s'] for row in rows], float)
    if not len(times) or times[0] != 0 or np.any(np.diff(times) <= 0):
        raise ValueError('Full reset-to-end capture times must increase strictly')
    receipt = json.loads((rec/'episode_receipt.json').read_text())
    if abs(times[-1]-receipt['simulation_duration_s']) > 1e-9:
        raise ValueError('Final original camera frame is missing')
    attempt = 0
    while (rec/f'media_{attempt:03d}').exists():
        attempt += 1
    stage = rec/f'media_{attempt:03d}'
    stage.mkdir()
    sources = []
    for i, t in enumerate(times):
        gp = rec/'observations'/f'global_d455_{i:04d}.npz'
        wp = rec/'observations'/f'wrist_d405_0_{i:04d}.npz'
        with np.load(gp, allow_pickle=False) as global_frame, np.load(wp, allow_pickle=False) as wrist:
            for frame in (global_frame, wrist):
                if abs(float(frame['physics_timestamp_s'])-t) > 1e-9:
                    raise ValueError('Camera timestamp differs from the frame manifest')
            if 'depth_m' in global_frame:
                raise ValueError('The narrow Global stream must remain explicitly RGB-only')
            image = Image.new('RGB', (960, 544))
            image.paste(Image.fromarray(global_frame['rgb']), (0, 0))
            image.paste(Image.fromarray(wrist['rgb']).resize((320, 240)), (640, 0))
            depth = wrist['depth_m']
            valid = wrist['valid_mask']
            # Fixed optical-Z scale, in metres; invalid pixels black.
            u = np.clip((depth-.07)/.93, 0, 1)
            color = np.stack([255*(1-u), 255*(1-np.abs(2*u-1)), 255*u], -1).astype(np.uint8)
            color[~valid] = 0
            image.paste(Image.fromarray(color).resize((320, 240)), (640, 240))
        draw = ImageDraw.Draw(image)
        text = (f'agentic control robot | scene seed {provenance["scene_seed"]} | tlmpm_v7_2 | TASK PASS\n'
                f'{episode_id} | controller {str(provenance.get("controller_sha256"))[:12]} | physics {t:.3f} s | 1x\n'
                f'{provenance.get("controller_provenance")} | {provenance.get("candidate_id")}\n'
                'Live Global RGB / wrist_0 RGB / wrist_0 optical Z: 0.07-1.00 m; invalid black')
        draw.text((6, 486), text, fill='white')
        image.save(stage/f'frame_{i:05d}.png')
        sources.append(dict(global_sha256=sha256(gp), wrist_sha256=sha256(wp), time_s=float(t)))
    # VFR concat holds each measured frame until the next real capture. The
    # final frame has the muxer's ordinary terminal duration; no interpolated frames.
    lines = ['ffconcat version 1.0']
    for i in range(len(times)):
        lines.append(f"file 'frame_{i:05d}.png'")
        if i+1 < len(times):
            lines.append(f'duration {times[i+1]-times[i]:.12f}')
    (stage/'frames.ffconcat').write_text('\n'.join(lines)+'\n')
    movie = stage/'episode.mp4'
    cmd = ['ffmpeg', '-v', 'error', '-n', '-safe', '0', '-i', str(stage/'frames.ffconcat'),
           '-vsync', 'vfr', '-c:v', 'libx264', '-threads', '2', '-crf', '24', '-pix_fmt', 'yuv420p',
           '-movflags', '+faststart', str(movie)]
    subprocess.run(cmd, check=True, timeout=300)
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_streams', '-show_format',
                                               '-of', 'json', str(movie)], text=True))
    subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-i', str(movie), '-f', 'null', '-'],
                   check=True, timeout=180)
    video = next(stream for stream in probe['streams'] if stream['codec_type'] == 'video')
    if video['codec_name'] != 'h264' or video['pix_fmt'] != 'yuv420p' or int(video['nb_frames']) != len(times):
        raise ValueError('Media codec, pixel format or complete original frame count mismatch')
    metrics = result.get('outcome', {}).get('metrics', {})
    manifest = dict(**provenance, episode_id=episode_id,
        task_success=True, physical_result=result, home_status=receipt['home_status'],
        per_bump_removed_fraction={row['id']: row['removed_mass_fraction'] for row in metrics.get('items', [])},
        substrate_retention=metrics.get('largest_component_below_plane_mass_share'),
        mechanical_components=metrics.get('mechanical_components'),
        simulation_duration_s=float(times[-1]), wall_time_s=receipt['wall_time_s'],
        mp4_path=str(movie.relative_to(rec)), mp4_sha256=sha256(movie), media_decode_status='PASS',
        capture_time_s=times.tolist(), original_frame_count=len(times), sources=sources, command=cmd,
        recording_hashes={p.name: sha256(p) for p in (rec/'physics_state.npz', rec/'compiled_task.json',
                                                   rec/'independent_evaluation.json', rec/'provenance.json')})
    write_json(manifest_file, manifest)
    return manifest
