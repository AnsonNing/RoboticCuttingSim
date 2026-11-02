"""Explicit camera version and measured-state matching for the media refresh."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from robotic_cutting_sim.camera_rig import make_rig


def test_v9_physical_distance_and_wrist_identity():
    old, new = make_rig(), make_rig(rig_profile='wrist_camera_rig_v9')
    for name in ('wrist_d405_0', 'wrist_d405_1'):
        assert new.specs[name] == old.specs[name]
    global_view = new.specs['global_d455']
    eye = np.asarray(global_view['T_world_camera'])[:3, 3]
    target = np.asarray(global_view['look_at_target_world_m'])
    assert np.linalg.norm(eye-target) == pytest.approx(.60, abs=1e-9)
    forward = np.asarray(global_view['T_world_camera'])[:3, 2]
    assert np.dot(forward, (target-eye)/.60) == pytest.approx(1.)
    for key in ('K', 'width', 'height', 'native_near_m', 'native_far_m'):
        assert global_view[key] == old.specs['global_d455'][key]
    assert global_view['near_m'] == global_view['native_near_m'] == .52
    assert make_rig().specs == old.specs


def test_v9_target_translation_is_frozen_at_scene_creation():
    target=np.array([.02,-.01,.12])
    rig=make_rig(rig_profile='wrist_camera_rig_v9',global_target_world_m=target)
    view=rig.specs['global_d455']
    eye=np.asarray(view['T_world_camera'])[:3,3]
    np.testing.assert_allclose(view['look_at_target_world_m'],target)
    assert np.linalg.norm(eye-target)==pytest.approx(.6,abs=1e-9)


def test_replay_refuses_interpolation_and_missing_state():
    spec = importlib.util.spec_from_file_location('replay_global', Path('scripts/rerender_native_global.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.exact_index([0., .2, .4], .2) == 1
    with pytest.raises(ValueError, match='interpolation refused'):
        module.exact_index([0., .2, .4], .3)
