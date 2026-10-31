#!/usr/bin/env python3
"""WP-12 Phase 2b: small PNG close-ups (start / progress / final) and camera stills of one live workcell episode.

Reads the recorded observation frames of ``ROOT/run/episode_000/observations`` (presentation close-up and
overview RGB, global D455 and wrist D405 RGB + depth; nothing is re-simulated) and writes PNGs of at most
``--max-width`` pixels to ``--media`` (a new directory per episode; existing files are never overwritten).
The progress frame is the last recorded frame inside the first armed (cut-enabled) segment.

  .venv/bin/python scripts/export_wp12_workcell_media.py --root artifacts/wp12/workcell_live_001 \\
      --media docs/media/wp12/workcell_live/cut
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def save(path, array, max_width):
    if path.exists():
        raise FileExistsError(path)
    image = Image.fromarray(array)
    if image.width > max_width:
        image = image.resize((max_width, round(image.height*max_width/image.width)), Image.LANCZOS)
    image.save(path, optimize=True)
    return str(path.relative_to(ROOT))


def depth_image(depth, valid):
    out = np.zeros(depth.shape, np.uint8)
    if valid.any():
        lo, hi = np.percentile(depth[valid], [1, 99])
        out[valid] = (40+215*(1-np.clip((depth[valid]-lo)/max(hi-lo, 1e-6), 0, 1))).astype(np.uint8)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--media', required=True, type=Path)
    p.add_argument('--max-width', type=int, default=640)
    a = p.parse_args(argv)
    root = a.root if a.root.is_absolute() else ROOT/a.root
    media = a.media if a.media.is_absolute() else ROOT/a.media
    media.mkdir(parents=True, exist_ok=False)
    episode = root/'run'/'episode_000'
    frames = json.loads((episode/'frame_manifest.json').read_text())
    times = [f['time_s'] for f in frames]
    armed = [f['frame'] for f in frames if f.get('cut_enabled')]
    first_run = []
    for index in armed:
        if first_run and index != first_run[-1]+1:
            break
        first_run.append(index)
    picks = {'start': 0, 'progress': first_run[-1] if first_run else len(frames)//2, 'final': len(frames)-1}
    written = {}
    obs = episode/'observations'
    for tag, index in picks.items():
        for name in ('closeup_rgb', 'overview_rgb'):
            f = obs/f'{name}_{index:04d}.npz'
            if f.exists():
                with np.load(f) as z:
                    written[f'{tag}_{name}'] = save(media/f'{tag}_{name.replace("_rgb", "")}.png', np.asarray(z['rgb'], np.uint8), a.max_width)
    for tag in ('start', 'final'):
        index = picks[tag]
        for name in ('global_d455', 'wrist_d405_0', 'wrist_d405_1'):
            f = obs/f'{name}_{index:04d}.npz'
            if not f.exists():
                continue
            with np.load(f) as z:
                written[f'{tag}_{name}_rgb'] = save(media/f'{tag}_{name}_rgb.png', np.asarray(z['rgb'], np.uint8), 320)
                written[f'{tag}_{name}_depth'] = save(media/f'{tag}_{name}_depth.png',
                                                      depth_image(np.asarray(z['depth_m'], float), np.asarray(z['valid_mask'], bool)), 320)
    receipt = {'root': str(root.relative_to(ROOT)), 'frames': {k: {'index': v, 'time_s': times[v]} for k, v in picks.items()},
               'written': written, 'command': sys.argv,
               'scope': 'Recorded live Native frames (presentation close-up/overview are presentation-only, never observations)'}
    (media/'media_receipt.json').write_text(json.dumps(receipt, indent=1)+'\n')
    print(json.dumps(receipt['frames']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
