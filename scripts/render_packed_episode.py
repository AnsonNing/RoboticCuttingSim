#!/usr/bin/env python3
"""Render a live episode recorded with tlmpm_v5 packed codes (K > 2) via scripts/render_demo_episode.py.

render_demo_episode.py colours the particle panel with ``cut_topology.mechanical_components``
on ternary codes (at most two digits). A tlmpm_v5 episode with K >= 3 cut digits records
packed 2-bit codes (``episode_000/material.json``: ``code_encoding = packed2``, ``max_cuts``),
which that table cannot index. This wrapper reads the encoding from the recording and, for
packed codes, replaces only ``demo_media.ComponentCache`` with one that calls
``cut_topology.mechanical_components_packed`` (the same actual stencil graph, bitmask
conflict test). Everything else, all arguments and every output, is the unchanged
renderer. The recording is never modified.

  .venv/bin/python scripts/render_packed_episode.py <run>/episode_000 --out DIR [renderer options] [--wide-piece-palette]

``--wide-piece-palette`` (opt-in, presentation only; consumed here, not passed on): the particle
panel gets five piece hues instead of three and treats components of >= 0.2 % of the particles
(instead of 1 %) as pieces, so e.g. three small removed chips and the substrate are all
distinguishable. Without it the renderer's colouring is unchanged.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from robotic_cutting_sim import demo_media as dm  # noqa: E402
from robotic_cutting_sim.cut_topology import mechanical_components_packed  # noqa: E402


class PackedComponentCache(dm.ComponentCache):
    """mechanical_components_packed() memoised per unique code vector (K from the recording)."""

    max_cuts = None

    def __call__(self, codes):
        codes = np.ascontiguousarray(codes)
        key = hashlib.sha1(codes.tobytes()).hexdigest()
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = mechanical_components_packed(self.nodes, self.weights, codes, self.max_cuts,
                                                           gradients=self.gradients)
        return self.cache[key]


WIDE_PIECE_COLORS = dm.COMPONENT_COLORS + ("#e5b53a", "#b46ad8")
WIDE_MINOR_FRACTION = .002


def wide_component_colors(labels, reference_positions, minor_fraction=None):
    """dm.component_colors with five piece slots and a 0.2 % piece threshold (presentation only)."""
    slot, sizes = dm.component_order(labels, reference_positions, WIDE_MINOR_FRACTION, slots=len(WIDE_PIECE_COLORS))
    palette = np.array([dm.hex_rgba(c) for c in WIDE_PIECE_COLORS] + [dm.hex_rgba(dm.MINOR_COLOR)])
    return palette[slot[np.asarray(labels)]], slot, sizes


def main():
    wide = '--wide-piece-palette' in sys.argv
    if wide:
        sys.argv.remove('--wide-piece-palette')
    episode = Path(sys.argv[1])
    material = json.loads((episode/'material.json').read_text())
    spec = importlib.util.spec_from_file_location('render_demo_episode', ROOT/'scripts'/'render_demo_episode.py')
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    if material.get('code_encoding') == 'packed2':
        PackedComponentCache.max_cuts = int(material['max_cuts'])
        renderer.dm.ComponentCache = PackedComponentCache
        print(f'packed2 codes, K = {PackedComponentCache.max_cuts}: particle panel uses mechanical_components_packed',
              flush=True)
    if wide:
        renderer.dm.component_colors = wide_component_colors
        print(f'wide piece palette: {len(WIDE_PIECE_COLORS)} hues, pieces >= {WIDE_MINOR_FRACTION:.1%} of particles',
              flush=True)
    renderer.main()


if __name__ == '__main__':
    main()
