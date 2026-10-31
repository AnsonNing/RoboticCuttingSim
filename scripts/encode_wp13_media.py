#!/usr/bin/env python3
"""Encode genuine saved Native RGB frames; previews never synthesize physical motion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

from run_wp13 import ROOT, G


def encode_frames(files, out, title, *, view='overview_rgb'):
    first = np.load(files[0])['rgb']
    height, width = first.shape[:2]
    command = ['ffmpeg', '-v', 'error', '-n', '-f', 'rawvideo', '-pixel_format', 'rgb24',
               '-video_size', f'{width}x{height}', '-framerate', '5', '-i', '-',
               '-vf', f"scale=640:360,drawbox=x=0:y=0:w=iw:h=24:color=black@0.7:t=fill,"
                      f"drawtext=text='{title}':fontcolor=white:fontsize=13:x=8:y=5",
               '-c:v', 'libx264', '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out)]
    digest = hashlib.sha256(); times = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as child:
        for path in files:
            data = np.load(path)
            rgb = np.asarray(data['rgb'], np.uint8)
            child.stdin.write(rgb.tobytes())
            times.append(float(data['physics_timestamp_s']))
            digest.update(path.name.encode()); digest.update(path.read_bytes())
        child.stdin.close()
        if child.wait() != 0:
            raise RuntimeError(f'Native encoding failed: {out}')
    return {'view': view, 'frames': len(files), 'capture_time_s': times,
            'ordered_capture_sha256': digest.hexdigest(), 'ffmpeg_command': command}


def preview(source, out, *, start, duration, width=480, speed=1.):
    filters = (f'setpts=(PTS-STARTPTS)/{speed},fps=5,scale={width}:-2:flags=lanczos,split[a][b];'
               '[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=3')
    cmd = ['ffmpeg', '-v', 'error', '-n', '-ss', str(start), '-t', str(duration), '-i', str(source),
           '-filter_complex', filters, '-loop', '0', str(out)]
    subprocess.run(cmd, check=True)
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-root', type=Path, required=True)
    p.add_argument('--stage', type=Path, required=True)
    p.add_argument('--publish', type=Path, default=Path('docs/media/wp13'))
    p.add_argument('--provenance', type=Path, default=Path('docs/evidence/wp13/media_001.json'))
    a = p.parse_args(); a.stage.mkdir(parents=True, exist_ok=False)
    captures, sources = {}, {}
    names = [('C00_cut', 'cut'), ('T00_trim', 'trim'), ('M00_four_slices', 'multicut')]
    for case, label in names:
        for solver in ('tlmpm_v5', 'tlmpm_v7'):
            short = solver.replace('tlmpm_', '')
            episode = a.native_root/solver/case/'run/episode_000'
            result = json.loads((episode/'result.json').read_text())
            if result['runtime_source'] != 'live_isaac_sim':
                raise ValueError(f'{episode} is not live Native')
            obs = episode/'observations'
            files = sorted(obs.glob('overview_rgb_*.npz'))
            name = f'{short}_{label}_native.mp4'
            title = f'Live Native | {short} | {label} | seed 42 | particle_surface_v1'
            rec = encode_frames(files, a.stage/name, title)
            captures[name] = {**rec, 'source_episode': str(episode), 'runtime_source': result['runtime_source'],
                'task_id': result['task_id'], 'task_status': result['task_status'], 'seed': 42,
                'solver': solver, 'surface_profile': 'particle_surface_v1', 'source_result_sha256': G.sha(episode/'result.json')}
            close = sorted(obs.glob('closeup_rgb_*.npz'))
            close_name = f'{short}_{label}_closeup.mp4'
            rec_close = encode_frames(close, a.stage/close_name, title, view='closeup_rgb')
            sources[(short, label)] = (name, close_name, rec_close)
    # Full, matched captured-time CUT comparison. Close-ups are actual Native
    # presentation cameras; neither is a policy observation or a collider.
    left, right = sources[('v5', 'cut')], sources[('v7', 'cut')]
    ta, tb = left[2]['capture_time_s'], right[2]['capture_time_s']
    if len(ta) != len(tb) or not np.allclose(ta, tb, atol=1e-8, rtol=0):
        raise ValueError('CUT comparison lacks identical capture timestamps; do not fabricate a pairing')
    ab_name = 'cut_v5_v7_native_comparison.mp4'
    subprocess.run(['ffmpeg', '-v', 'error', '-n', '-i', str(a.stage/left[1]), '-i', str(a.stage/right[1]),
                    '-filter_complex', '[0:v][1:v]hstack=inputs=2[v]', '-map', '[v]', '-c:v', 'libx264',
                    '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(a.stage/ab_name)], check=True)
    media = []

    def add(gif, source, mp4, case, start, duration, width, speed, capture):
        cmd = preview(a.stage/source, a.stage/gif, start=start, duration=duration, width=width, speed=speed)
        media.append({'gif': str(a.publish/gif), 'mp4': str(a.publish/mp4), 'runtime_source': 'live_isaac_sim',
                      'case': case, 'preview_speed': speed, 'preview_source_interval_s': [start, start+duration],
                      'surface_profile': 'particle_surface_v1', 'capture_provenance': capture,
                      'gif_sha256': G.sha(a.stage/gif), 'mp4_sha256': G.sha(a.stage/mp4), 'ffmpeg_command': cmd})

    add('hero_native.gif', 'v5_cut_native.mp4', 'v5_cut_native.mp4', 'C00_cut', 3., 7., 640, 1., captures['v5_cut_native.mp4'])
    add('cut_preview.gif', sources[('v5', 'cut')][1], 'v5_cut_native.mp4', 'C00_cut', 3., 7., 480, 1., sources[('v5', 'cut')][2])
    add('trim_preview.gif', sources[('v5', 'trim')][1], 'v5_trim_native.mp4', 'T00_trim', 2., 7., 480, 1., sources[('v5', 'trim')][2])
    duration = len(captures['v5_multicut_native.mp4']['capture_time_s'])/5
    add('multicut_preview.gif', sources[('v5', 'multicut')][1], 'v5_multicut_native.mp4', 'M00_four_slices',
        0., duration, 480, 3., sources[('v5', 'multicut')][2])
    add('cut_v5_v7_preview.gif', ab_name, ab_name, 'C00_cut', 3., 7., 960, 1.,
        {'left': left[2], 'right': right[2], 'matched_capture_times': True,
         'candidate_status': 'REJECTED_FOR_PROMOTION: primary TRIM fails; localized CUT improvement only'})
    # Publish only full clips and previews, not redundant temporary close-up clips.
    a.publish.mkdir(parents=True, exist_ok=False)
    keep = [*captures, ab_name, *[Path(row['gif']).name for row in media]]
    for name in keep:
        shutil.copyfile(a.stage/name, a.publish/name)
    G.write_new(a.provenance, {'runtime_source': 'live_isaac_sim', 'media': media, 'full_videos': captures,
        'scope': 'Real recorded Native RGB. Full videos use nominal 5 fps capture cadence; exact physics timestamps '
                 'and source hashes retained. Preview timing/crops do not change physical data. '
                 'Both candidate failures and functioning v5 demonstrations remain explicit.'})


if __name__ == '__main__':
    main()
