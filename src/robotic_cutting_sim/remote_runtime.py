"""Official Kit WebRTC bootstrap for the existing native cutting environment.

Importing this module does not import Isaac or start another simulation. The
caller owns the one SimulationApp and advances LiveIsaacSimCutting explicitly.
The application framebuffer includes the Kit GUI and the same live USD scene.
"""
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import distribution
from ipaddress import IPv4Address
from pathlib import Path
import sys

from .recording import ROOT, write_json


def _sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass(frozen=True)
class StreamingPorts:
    signal: int = 49100
    media: int = 47998

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 65535:
                raise ValueError(f"{name} must be an integer port in1..65535")


def streaming_launch_config(config, output, ip, width=1280, height=800, ports=None):
    """Build the SimulationApp arguments without importing or initializing Kit."""
    address = IPv4Address(ip)
    if address.is_unspecified or address.is_multicast:
        raise ValueError("The advertised WebRTC IP must identify this server")
    for value in (width, height):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value % 2:
            raise ValueError("Streaming width/height must be positive even integers")
    ports = StreamingPorts(**ports) if isinstance(ports, dict) else ports or StreamingPorts()
    if not isinstance(ports, StreamingPorts):
        raise TypeError("ports must be StreamingPorts or a signal/media mapping")
    output = Path(output).resolve()
    prefix = "--/exts/omni.kit.livestream.app/primaryStream/"
    args = [f"--/log/file={output/'kit.log'}",
            prefix+"streamType=webrtc", prefix+f"publicIp={address}",
            prefix+f"signalPort={ports.signal}", prefix+f"streamPort={ports.media}",
            "--/app/player/playSimulations=false",
            "--ext-folder", str(ROOT/".deps/isaacsim-extension-cache")]
    return {"headless": True, "hide_ui": False, "disable_viewport_updates": False,
            "active_gpu": config.gpu, "physics_gpu": config.gpu, "multi_gpu": False,
            "width": width, "height": height, "window_width": width, "window_height": height,
            "renderer": "RaytracedLighting", "enable_crashreporter": False, "extra_args": args}


def launch_streaming_app(config, output, ip, width=1280, height=800, ports=None):
    """Launch ONE offscreen Kit GUI with official WebRTC streaming enabled.

    Construct LiveIsaacSimCutting using this returned app. Do not separately
    launch isaacsim.exp.full.streaming, Docker Isaac Sim, or the old launch_app.
    HTTP web-viewer hosting is external to this native process.
    """
    from .isaac_sim_runner import require_sim_62, runtime_versions
    require_sim_62(runtime_versions())
    launch = streaming_launch_config(config, output, ip, width, height, ports)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Reuse the exact Kit110.4 extensions already resolved by this project's
    # existing experience. Missing extensions still use NVIDIA's Kit registry.
    package = Path(distribution("isaacsim-kernel").locate_file("isaacsim"))
    caches = sorted((package/"kit/data/Kit/isaac_sim_cutting").glob("*/exts/3"))
    for cache in caches:
        launch["extra_args"].extend(["--ext-folder", str(cache.resolve())])
    experience = ROOT/"configs/isaac_sim_cutting_streaming.kit"
    receipt = {"status": "STARTUP_REQUESTED", "same_instance_required": True,
               "experience": str(experience), "experience_sha256": _sha256(experience),
               "bootstrap_sha256": _sha256(Path(__file__)), "packages": runtime_versions(),
               "launch_config": launch, "gui_framebuffer_stream": True,
               "web_viewer_hosted_separately": True, "client_connection_verified": False}
    write_json(output/"streaming_runtime.json", receipt)
    from isaacsim import SimulationApp
    argv = sys.argv
    try:
        sys.argv = argv[:1]
        app = SimulationApp(launch, experience=str(experience))
    finally:
        sys.argv = argv
    import carb.settings
    import omni.kit.app
    manager = omni.kit.app.get_app().get_extension_manager()
    names = ("omni.kit.livestream.app", "omni.kit.livestream.webrtc", "omni.kit.livestream.streamsdk",
             "omni.kit.viewport.window", "omni.kit.window.stage", "omni.kit.window.property")
    extensions = {name: manager.get_enabled_extension_id(name) for name in names}
    if not extensions["omni.kit.livestream.app"] or not extensions["omni.kit.viewport.window"]:
        app.close()
        raise RuntimeError(f"Required WebRTC/GUI extensions were not enabled: {extensions}")
    settings = carb.settings.get_settings()
    prefix = "/exts/omni.kit.livestream.app/primaryStream/"
    actual = {key: settings.get(prefix+key) for key in
              ("streamType", "publicIp", "signalPort", "streamPort", "targetFps")}
    expected_ports = StreamingPorts(**ports) if isinstance(ports, dict) else ports or StreamingPorts()
    expected = {"streamType": "webrtc", "publicIp": str(IPv4Address(ip)),
                "signalPort": expected_ports.signal, "streamPort": expected_ports.media}
    if any(actual[key] != value for key, value in expected.items()):
        app.close()
        raise RuntimeError(f"Actual WebRTC settings differ from requested endpoint: {actual}")
    receipt.update(status="APP_INITIALIZED_CLIENT_NOT_VERIFIED", extensions=extensions,
                   kit_version=omni.kit.app.get_app().get_kit_version(), actual_primary_stream=actual,
                   manual_physics=configure_manual_physics())
    write_json(output/"streaming_runtime.json", receipt)
    return app


def configure_manual_physics():
    """Keep GUI updates independent of the retained robot/material loop.

    Call after every native env.reset(). Stage loading/closing callbacks remain
    enabled; explicit initialize_physics()/SimulationManager.step still work.
    GUI Play/Pause/Reset must be translated into project control commands.
    """
    import carb.settings
    import omni.timeline
    from isaacsim.core.simulation_manager import SimulationManager
    settings = carb.settings.get_settings()
    settings.set_bool("/app/player/playSimulations", False)
    SimulationManager.enable_default_callbacks(enable_warm_start=False, enable_on_stop=False)
    timeline = omni.timeline.get_timeline_interface()
    timeline.set_auto_update(False)
    # Timeline writes are deferred until a frame/commit. Commit on the Kit
    # thread so the returned receipt reflects the applied state immediately.
    timeline.commit()
    return {"automatic_physics": settings.get_as_bool("/app/player/playSimulations"),
            "timeline_auto_update": timeline.is_auto_updating(),
            "callbacks": SimulationManager.get_default_callback_status(),
            "stepping_owner": "LiveIsaacSimCutting.step: measured PhysX articulation then retained TLMPM"}


def copy_camera_for_viewer(stage, source_path, target_path):
    """Copy current lens/world pose into a separate camera; never edit source."""
    from pxr import Usd, UsdGeom
    if source_path == target_path:
        raise ValueError("The viewer camera must be separate from the measurement camera")
    source = UsdGeom.Camera(stage.GetPrimAtPath(source_path))
    if not source:
        raise ValueError(f"The selected physical camera does not exist: {source_path}")
    target = UsdGeom.Camera.Define(stage, target_path)
    # Copy schema attributes, not sensor APIs or render-product relationships.
    for name in UsdGeom.Camera.GetSchemaAttributeNames(False):
        value = source.GetPrim().GetAttribute(name).Get()
        if value is not None:
            target.GetPrim().GetAttribute(name).Set(value)
    xform = UsdGeom.Xformable(target)
    xform.ClearXformOpOrder()
    xform.SetResetXformStack(True)
    xform.AddTransformOp().Set(UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(source.GetPrim()))
    target.GetPrim().SetCustomDataByKey("robotic_cutting_sim:viewer_only", True)
    target.GetPrim().SetCustomDataByKey("robotic_cutting_sim:source_camera", str(source_path))
    return target


class ViewportCameraPreview:
    """Safe free view and live optical-pose previews in the SAME USD stage.

    update() after each reset/control step and before GUI updates. Mouse orbit
    may change the disposable preview but never the physical sensor camera.
    A viewport preview is not substituted for recorded native RGB-D arrays.
    """
    preview_path = "/World/RemoteViewer/SensorPreview"
    free_path = "/World/RemoteViewer/FreeCamera"

    def __init__(self, env):
        self.env = env
        self.selected = "free"
        self._window = None

    @property
    def camera_paths(self):
        if getattr(self.env, "camera_specs", None):
            return {name: spec["optical_prim_path"] for name, spec in self.env.camera_specs.items()}
        return {"d455_0": "/World/D455_0/RSD455/Camera_Pseudo_Depth",
                "d455_1": "/World/D455_1/RSD455/Camera_Pseudo_Depth", "global": "/World/GlobalCamera"}

    @property
    def choices(self):
        return ["free", *self.camera_paths]

    def select(self, name):
        if name not in self.choices:
            raise ValueError(f"Unknown viewer camera {name!r}; choose from{self.choices}")
        self.selected = name
        return self.update()

    def update(self):
        from omni.kit.viewport.utility import create_viewport_window, get_active_viewport
        from pxr import Sdf
        stage = self.env.stage
        viewport = get_active_viewport()
        if viewport is None:
            self._window = create_viewport_window("Viewport")
            viewport = self._window.viewport_api
        paths = self.camera_paths
        if self.selected not in self.choices:
            self.selected = "free"
        if self.selected == "free":
            if not stage.GetPrimAtPath(self.free_path):
                source = paths.get("global_d455", paths.get("global", next(iter(paths.values()))))
                copy_camera_for_viewer(stage, source, self.free_path)
            target = self.free_path
            viewport.fill_frame = True
        else:
            copy_camera_for_viewer(stage, paths[self.selected], self.preview_path)
            target = self.preview_path
            viewport.fill_frame = False
        # Never assign an actual sensor to the manipulable ViewportAPI.
        viewport.camera_path = Sdf.Path(target)
        return {"selected": self.selected, "viewer_camera_path": target,
                "source_camera_path": paths.get(self.selected), "same_live_stage": True,
                "sensor_transform_edited": False, "sensor_rgbd_unchanged": True}
