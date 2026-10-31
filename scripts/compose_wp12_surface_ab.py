#!/usr/bin/env python3
"""WP-12 Layer 6: side-by-side A (particle_surface_v1, Phase 2b) vs B (cut_aware_surface_v1) close-up PNGs.

Reads the already exported stills (``scripts/export_wp12_workcell_media.py``) of the same view and frame of the
same task: ``--a DIR`` (e.g. docs/media/wp12/workcell_live/cut) and ``--b DIR`` (docs/media/wp12/surface_live/cut),
and writes ``ab_<name>.png`` (A left, B right, labelled) into ``--b`` (never overwrites). Presentation only.

  .venv/bin/python scripts/compose_wp12_surface_ab.py --a docs/media/wp12/workcell_live/cut \\
      --b docs/media/wp12/surface_live/cut
"""
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('start_closeup', 'progress_closeup', 'final_closeup', 'final_overview')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', required=True, type=Path)
    p.add_argument('--b', required=True, type=Path)
    p.add_argument('--names', nargs='*', default=list(NAMES))
    args = p.parse_args(argv)
    a_dir = args.a if args.a.is_absolute() else ROOT/args.a
    b_dir = args.b if args.b.is_absolute() else ROOT/args.b
    written = {}
    for name in args.names:
        a_path, b_path = a_dir/f'{name}.png', b_dir/f'{name}.png'
        if not (a_path.exists() and b_path.exists()):
            continue
        out = b_dir/f'ab_{name}.png'
        if out.exists():
            raise FileExistsError(out)
        a, b = Image.open(a_path).convert('RGB'), Image.open(b_path).convert('RGB')
        if a.size != b.size:
            b = b.resize(a.size, Image.LANCZOS)
        band = 22
        canvas = Image.new('RGB', (a.width*2+6, a.height+band), (255, 255, 255))
        canvas.paste(a, (0, band))
        canvas.paste(b, (a.width+6, band))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 4), f'A particle_surface_v1 (Phase 2b) - {name}', fill=(0, 0, 0))
        draw.text((a.width+12, 4), f'B cut_aware_surface_v1 - {name}', fill=(0, 0, 0))
        canvas.save(out, optimize=True)
        written[name] = {'a': str(a_path.relative_to(ROOT)), 'b': str(b_path.relative_to(ROOT)),
                         'out': str(out.relative_to(ROOT))}
    receipt = b_dir/'ab_receipt.json'
    if receipt.exists():
        raise FileExistsError(receipt)
    receipt.write_text(json.dumps({'written': written, 'scope': 'presentation only; same task, view and frame index'},
                                  indent=1))
    print(json.dumps(written, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
