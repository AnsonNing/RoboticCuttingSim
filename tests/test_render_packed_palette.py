"""render_packed_episode --wide-piece-palette (presentation only; default colouring unchanged)."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location('render_packed_episode', ROOT/'scripts'/'render_packed_episode.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wide_palette_keeps_substrate_and_chip_hues_as_chips_split_off_in_order():
    m = _module()
    n = 110286   # Test C s1602: chips of 473 / 2047 / 758 particles (0.4-1.9 %, below the default 1 % for two)
    reference = np.zeros((n, 3))
    reference[:, 0] = np.linspace(0, .2, n)
    chips = [slice(0, 473), slice(50000, 52047), slice(100000, 100758)]
    previous = None
    for stage in range(1, 4):
        labels = np.zeros(n, int)
        for c in range(stage):
            labels[chips[c]] = c+1
        colors, slot, _ = m.wide_component_colors(labels, reference)
        assert slot.tolist() == list(range(stage+1))      # every piece distinct; substrate always hue 0
        if previous is not None:
            assert np.array_equal(colors[labels < stage], previous[labels < stage])
        previous = colors
    _, default_slot, _ = m.dm.component_colors(labels, reference)
    assert (default_slot == -1).sum() == 2          # the stock palette greys out the two chips < 1 %


def test_default_colouring_is_untouched_by_import():
    m = _module()
    from robotic_cutting_sim import demo_media
    assert m.dm.component_colors is demo_media.component_colors
