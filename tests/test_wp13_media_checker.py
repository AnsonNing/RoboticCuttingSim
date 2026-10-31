"""Release checks must reject a still image presented as an animation."""
import importlib.util
import io
from pathlib import Path

from PIL import Image
import pytest

spec = importlib.util.spec_from_file_location('wp13_release_check',
    Path(__file__).resolve().parents[1]/'scripts/check_wp12_release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def gif(colors, loop=0):
    frames = [Image.new('RGB', (8, 8), c) for c in colors]
    out = io.BytesIO()
    frames[0].save(out, format='GIF', save_all=True, append_images=frames[1:], duration=100, loop=loop)
    return out.getvalue()


def test_checker_rejects_static_gif_and_finite_loop():
    with pytest.raises(ValueError, match='no decoded pixel motion'):
        module.animated_gif_metadata(gif(['red']))
    with pytest.raises(ValueError, match='loop indefinitely'):
        module.animated_gif_metadata(gif(['red', 'blue'], loop=1))


def test_checker_decodes_actual_distinct_motion_frames():
    result = module.animated_gif_metadata(gif(['red', 'blue']))
    assert result['frames'] == 2 and result['distinct_decoded_frames'] == 2 and result['loop'] == 0
