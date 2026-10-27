#!/usr/bin/env python3
"""Isaac Sim 6.2 native D455 RGB and stereo-model depth runner.

Run with Isaac's python.sh and this project on PYTHONPATH. Missing runtime emits
BLOCKED/result.json and exit 2. There is deliberately no ideal-camera fallback.
"""
import argparse
import importlib.metadata
import importlib.util
import json
import time
import traceback
from pathlib import Path

import numpy as np

from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.sensor_contract import mode_contract


def runtime_audit():
    packages = {}
    for name in ["isaacsim", "isaacsim-core", "isaaclab"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    available = importlib.util.find_spec("isaacsim") is not None
    return {"isaacsim_import_available": available, "packages": packages,
            "nvidia_smi": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
            "required_runtime": "Isaac Sim 6.2.0; isaacsim.sensors.experimental.rtx",
            "driver_policy": "No driver/OS changes; official tested Linux driver 595.58.03",
            "official_sensor_docs": "https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera_depth.html"}


def as_numpy(value):
    if hasattr(value, "numpy"):
        return value.numpy()
    if hasattr(value, "__dlpack__"):
        return np.from_dlpack(value).copy()
    return np.asarray(value)


def capture(args, output):
    # Kit/Omni modules must be imported after SimulationApp initialization.
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "renderer": "RaytracedLighting"})
    try:
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from pxr import Usd, UsdLux
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, SingleViewDepthCameraSensor
        from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge, usd_camera_calibration
        context = omni.usd.get_context()
        if not context.open_stage(str(args.scene.resolve())):
            raise RuntimeError("Isaac failed to open the workcell USD")
        for _ in range(300):
            app.update()
            if not context.is_stage_loading():
                break
        else:
            raise TimeoutError("Stage loading exceeded 300 application updates")
        stage = context.get_stage()
        if Usd.GetVersion() < (0, 25, 0):
            raise RuntimeError("Unexpected old OpenUSD in Isaac runtime")
        light = UsdLux.DomeLight.Define(stage, "/World/RgbdProbeDome")
        light.CreateIntensityAttr(1500)
        sensors, calibration, settings = {}, {}, {}
        for index in (0, 1):
            name = f"d455_{index}"
            root = f"/World/D455_{index}/RSD455"
            depth_path = root + "/Camera_Pseudo_Depth"
            color_path = root + "/Camera_OmniVision_OV9782_Color"
            # Existing native cameras retain their authored optical transforms/lens.
            depth = SingleViewDepthCameraSensor(RtxCamera(depth_path), resolution=(args.height, args.width),
                        annotators=["depth_sensor_distance", "distance_to_image_plane"])
            depth.set_enabled_post_processing(True)
            color = CameraSensor(RtxCamera(color_path), resolution=(args.height, args.width), annotators=["rgb"])
            sensors[name] = (color, depth)
            calibration[name] = {"rgb": usd_camera_calibration(stage, color_path, args.width, args.height),
                                 "depth": usd_camera_calibration(stage, depth_path, args.width, args.height),
                                 "rgb_depth_aligned": False}
            settings[name] = {"baseline_mm": depth.get_sensor_baseline(),
                              "noise_parameters": depth.get_sensor_noise_parameters(),
                              "distance_cutoffs": depth.get_sensor_distance_cutoffs(),
                              "post_processing_enabled": depth.get_enabled_post_processing()}
        global_sensor = CameraSensor(RtxCamera("/World/GlobalCamera"), resolution=(args.height, args.width),
                                     annotators=["rgb", "distance_to_image_plane"])
        write_json(output / "calibration.json", calibration)
        write_json(output / "d455_settings.json", settings)
        paths = sorted(args.surface_dir.glob("*.npz"))
        if len(paths) < 2:
            raise ValueError("Need at least two measured surface snapshots")
        paths = [paths[i] for i in np.unique(np.linspace(0, len(paths) - 1, min(args.samples, len(paths)), dtype=int))]
        bridge = DynamicSurfaceBridge(stage)
        omni.timeline.get_timeline_interface().play()
        rows = []
        for index, path in enumerate(paths):
            data = np.load(path)
            timestamp = float(data["physics_timestamp_s"])
            receipt = bridge.update(data["vertices_m"], data["triangles"], timestamp)
            tick = time.perf_counter()
            # All geometry is held fixed while renderer/annotators settle. Timestamp
            # correspondence is recorded, but not declared validated without a live run.
            for warmup in range(60):
                rep.orchestrator.step(rt_subframes=3, delta_time=0.0)
                payloads = {}
                for name, (color, depth) in sensors.items():
                    payloads[name] = [color.get_data("rgb"), depth.get_data("depth_sensor_distance"),
                                      depth.get_data("distance_to_image_plane")]
                if warmup >= 2 and all(item[0] is not None for values in payloads.values() for item in values):
                    break
            else:
                raise TimeoutError("Native sensor annotators did not produce arrays within 60 renders")
            for name, values in payloads.items():
                rgb, sensor_depth, ideal_depth = [as_numpy(item[0]) for item in values]
                if rgb.shape[:2] != (args.height, args.width) or sensor_depth.shape[:2] != (args.height, args.width):
                    raise RuntimeError(f"Malformed native annotator output: {name}")
                # Preserve raw depth convention until verified against a known plane.
                # Do not rename an unverified RTX depth AOV to optical-Z depth_m.
                np.savez_compressed(output / "observations" / f"{name}_{index:04d}.npz",
                    rgb=rgb, depth_sensor_distance_raw=sensor_depth, distance_to_image_plane_raw=ideal_depth,
                    physics_timestamp_s=timestamp, surface_timestamp_s=timestamp,
                    capture_wall_time_ns=time.time_ns(), rgb_depth_aligned=False)
                write_json(output / "observations" / f"{name}_{index:04d}.json",
                           {"annotator_info": [str(item[1]) for item in values], "surface": receipt})
            global_rgb, global_info = global_sensor.get_data("rgb")
            global_depth, _ = global_sensor.get_data("distance_to_image_plane")
            if global_rgb is None or global_depth is None:
                raise RuntimeError("Global camera annotator produced no array")
            np.savez_compressed(output / "global_debug" / f"{index:04d}.npz", rgb=as_numpy(global_rgb),
                               distance_to_image_plane_raw=as_numpy(global_depth), physics_timestamp_s=timestamp)
            rows.append({**receipt, "capture_wall_s": time.perf_counter() - tick})
        return {"status": "PASS", "scope": "native Isaac D455 annotator array capture only",
                **mode_contract("simulated_d455"), "frames": rows,
                "depth_convention_validation": "NOT_RUN", "live_physics_sensor_sync": "NOT_RUN",
                "physical_cutting_validation": "NOT_ASSESSED", "vision_proven": False}
    finally:
        app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    (output / "observations").mkdir(); (output / "global_debug").mkdir()
    audit = runtime_audit()
    write_json(output / "runtime_audit.json", audit)
    write_json(output / "manifest.json", manifest({"scene": str(args.scene), "surface_dir": str(args.surface_dir),
               **mode_contract("simulated_d455")}, "isaac_6_2_native_d455"))
    if not audit["isaacsim_import_available"]:
        result = {"status": "BLOCKED", "reason": "Isaac Sim runtime is not installed in this Python environment",
                  "actual_rtx_capture": "NOT_RUN", "ideal_fallback_used": False, "vision_proven": False}
        code = 2
    else:
        try:
            result, code = capture(args, output), 0
        except Exception as error:
            (output / "failure_traceback.txt").write_text(traceback.format_exc())
            result, code = {"status": "FAIL", "reason": str(error), "vision_proven": False}, 1
    write_json(output / "result.json", result)
    print(json.dumps(result, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
