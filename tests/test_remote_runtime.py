"""CPU checks for endpoint contracts and physical-camera pose isolation."""
from pathlib import Path
from types import SimpleNamespace
import tomllib

import numpy as np
import pytest

from robotic_cutting_sim.remote_runtime import StreamingPorts, copy_camera_for_viewer, streaming_launch_config


def test_streaming_bootstrap_preserves_controlled_physics_and_gui(tmp_path):
    config = streaming_launch_config(SimpleNamespace(gpu=1), tmp_path, "100.111.162.17",
                                    ports={"signal": 49101, "media": 47999})
    assert config["headless"] and not config["hide_ui"]
    assert not config["disable_viewport_updates"] and not config["multi_gpu"]
    assert config["width"] == 1280 and config["height"] == 800
    assert "--/app/player/playSimulations=false" in config["extra_args"]
    assert "--/exts/omni.kit.livestream.app/primaryStream/signalPort=49101" in config["extra_args"]
    assert "--/exts/omni.kit.livestream.app/primaryStream/streamPort=47999" in config["extra_args"]
    assert not any("publicEndpointAddress" in arg for arg in config["extra_args"])
    root = Path(__file__).resolve().parents[1]
    legacy = tomllib.loads((root/"configs/isaac_sim_cutting.kit").read_text())
    gui = tomllib.loads((root/"configs/isaac_sim_cutting_streaming.kit").read_text())
    assert set(legacy["dependencies"]).issubset(gui["dependencies"])
    assert not gui["settings"]["app"]["player"]["playSimulations"]
    assert "omni.kit.livestream.app" in gui["dependencies"]


@pytest.mark.parametrize("value", [0, -1, 65536, True, 49100.5])
def test_reject_invalid_ports(value):
    with pytest.raises(ValueError):
        StreamingPorts(signal=value)


@pytest.mark.parametrize("ip", ["0.0.0.0", "224.0.0.1", "::1", "server.invalid"])
def test_reject_invalid_advertised_ip(ip, tmp_path):
    with pytest.raises(ValueError):
        streaming_launch_config(SimpleNamespace(gpu=1), tmp_path, ip)


def test_preview_follows_actual_moving_parent_without_editing_sensor():
    pytest.importorskip("pxr.Usd")
    from pxr import Gf, Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    hand = UsdGeom.Xform.Define(stage, "/World/Hand")
    translate = hand.AddTranslateOp()
    translate.Set(Gf.Vec3d(.2, .1, .3))
    physical = UsdGeom.Camera.Define(stage, "/World/Hand/Sensor")
    physical.AddTranslateOp().Set(Gf.Vec3d(.01, .02, .03))
    physical.GetFocalLengthAttr().Set(22.)
    before = stage.GetRootLayer().ExportToString()
    target = "/World/RemoteViewer/Preview"
    copy_camera_for_viewer(stage, str(physical.GetPath()), target)
    camera = UsdGeom.Camera.Get(stage, target)
    assert camera.GetFocalLengthAttr().Get() == 22.
    for x in (.2, .25):
        translate.Set(Gf.Vec3d(x, .1, .3))
        copy_camera_for_viewer(stage, str(physical.GetPath()), target)
        cache = UsdGeom.XformCache()
        np.testing.assert_array_equal(np.asarray(cache.GetLocalToWorldTransform(physical.GetPrim())),
                                      np.asarray(cache.GetLocalToWorldTransform(camera.GetPrim())))
    # A streamed orbit can modify only the disposable viewer camera.
    UsdGeom.Xformable(camera).GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetTranslate(Gf.Vec3d(9, 9, 9)))
    np.testing.assert_allclose(physical.GetLocalTransformation().ExtractTranslation(), (.01, .02, .03))
    assert physical.GetFocalLengthAttr().Get() == 22.
    with pytest.raises(ValueError):
        copy_camera_for_viewer(stage, str(physical.GetPath()), str(physical.GetPath()))
    assert "RemoteViewer" not in before
