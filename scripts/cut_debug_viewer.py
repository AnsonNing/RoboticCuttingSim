#!/usr/bin/env python3
"""WP-12 Phase 1 multi-layer cut debug viewer (debug only; privileged state, never a policy observation).

Renders selected layers in selected views to one PNG and optionally writes one PLY per layer for interactive
inspection (MeshLab, CloudCompare, Blender). Reads a ``wp12_cut_debug_package_v1`` written by
scripts/run_wp12_phase1_geometry.py or a Phase 0 ``baseline_compact.npz``.

Layers: original, current, labels, components, sweep, contact, particle_surface, cut_aware, rendered.
Views: top, side, blade, section, closeup.

  .venv/bin/python scripts/cut_debug_viewer.py --package artifacts/wp12/phase1_001/G3b_r0_package.npz \
      --layers labels,particle_surface,cut_aware,sweep --views section,closeup,top --digit 0 \
      --out artifacts/wp12/phase1_001/views/G3b_section.png --ply-dir artifacts/wp12/phase1_001/views/G3b_ply
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from robotic_cutting_sim import cut_debug_view as V  # noqa: E402


def triple(text):
    return None if text is None else [float(v) for v in text.split(",")]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="PNG path (must not exist)")
    p.add_argument("--layers", default="components,particle_surface,cut_aware,sweep")
    p.add_argument("--views", default="top,side,blade,section,closeup")
    p.add_argument("--digit", type=int, default=0)
    p.add_argument("--profile", default="current_volume_v1", choices=("legacy", "current_volume_v1"))
    p.add_argument("--plane-point", help="x,y,z (m) of the section plane; default: median side locus of the digit")
    p.add_argument("--plane-normal", help="nx,ny,nz of the section plane; default: the blade edge direction")
    p.add_argument("--slab", type=float, default=None, help="section slab thickness (m), default 1 spacing")
    p.add_argument("--window", type=float, default=.008, help="half window of section/close-up views (m)")
    p.add_argument("--title", default=None)
    p.add_argument("--dpi", type=int, default=90)
    p.add_argument("--ply-dir", type=Path, default=None, help="also write one PLY per layer here (must not exist)")
    a = p.parse_args()
    layers, views = a.layers.split(","), a.views.split(",")
    for name in layers:
        if name not in V.LAYERS:
            p.error(f"unknown layer {name}; choose from {V.LAYERS}")
    for name in views:
        if name not in V.VIEWS:
            p.error(f"unknown view {name}; choose from {V.VIEWS}")
    if a.out.exists() or (a.ply_dir is not None and a.ply_dir.exists()):
        p.error("refusing to overwrite an existing output")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    pkg = V.load(a.package)
    V.render(pkg, a.out, layers=layers, views=views, digit=a.digit, profile=a.profile, plane_point=triple(a.plane_point),
             plane_normal=triple(a.plane_normal), slab_m=a.slab, window_m=a.window, title=a.title, dpi=a.dpi)
    written = V.export_ply(pkg, a.ply_dir, layers=layers, digit=a.digit, profile=a.profile) if a.ply_dir else {}
    print(f"{a.out} ({a.out.stat().st_size/1e3:.0f} kB), {len(written)} PLY, {time.perf_counter()-start:.1f} s")


if __name__ == "__main__":
    main()
