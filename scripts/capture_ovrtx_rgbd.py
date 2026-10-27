#!/usr/bin/env python3
"""Standalone NVIDIA OVRTX RGB-D, separate from Isaac Sim's D455 wrapper.

Run in an isolated environment containing ovrtx and ovstage. USD preparation uses
a subprocess so external OpenUSD and the renderer's bundled USD never share a
process. Native camera optics are retained; depth is ideal RTX image-plane depth.
"""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from robotic_cutting_sim.recording import command, manifest, new_output, write_json
from robotic_cutting_sim.sensor_contract import validate_frame
from robotic_cutting_sim.rtx_outputs import decode_camera_params


CAMERAS = {"d455_0": "/World/D455_0/RSD455/Camera_Pseudo_Depth",
           "d455_1": "/World/D455_1/RSD455/Camera_Pseudo_Depth",
           "global": "/World/GlobalCamera"}


def prepare(args):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdRender, UsdShade
    from robotic_cutting_sim.usd_surface_bridge import usd_camera_calibration
    output = args.output_dir
    stage = Usd.Stage.CreateNew(str(output / "rtx_scene.usda"))
    stage.GetRootLayer().subLayerPaths = [str(args.scene.resolve())]
    if args.base_scene:
        stage.GetRootLayer().subLayerPaths.append(str(args.base_scene.resolve()))
    source_stage = Usd.Stage.Open(str(args.scene))
    # Stage metadata is read from root/session layers, not inherited sublayers.
    # Omitting this produces OpenUSD's centimeter/Y-up defaults in RTX.
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source_stage))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source_stage))
    stage.SetTimeCodesPerSecond(source_stage.GetTimeCodesPerSecond())
    stage.SetFramesPerSecond(source_stage.GetFramesPerSecond())
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    source = stage.GetPrimAtPath("/World/ContinuousSpecimen")
    samples = source.GetAttribute("points").GetTimeSamples() if source else []
    if not samples and source_stage.GetEndTimeCode() > source_stage.GetStartTimeCode():
        samples = list(np.linspace(source_stage.GetStartTimeCode(), source_stage.GetEndTimeCode(), args.samples))
    samples = ([samples[i] for i in np.unique(np.linspace(0, len(samples) - 1, min(len(samples), args.samples), dtype=int))]
               if samples else [0.0])
    calibration = {}
    products = []
    cameras = {**CAMERAS, "global": args.global_camera_path}
    for name, path in cameras.items():
        calibration[name] = usd_camera_calibration(stage, path, args.width, args.height)
        product = UsdRender.Product.Define(stage, "/Render/" + name)
        product.GetResolutionAttr().Set(Gf.Vec2i(args.width, args.height))
        product.GetCameraRel().SetTargets([Sdf.Path(path)])
        product.GetPrim().CreateAttribute("deviceIds", Sdf.ValueTypeNames.UIntArray).Set([args.gpu])
        product.GetPrim().CreateAttribute("omni:rtx:rendermode", Sdf.ValueTypeNames.Token).Set("RealTimePathTracing")
        product.GetPrim().CreateAttribute("omni:rtx:quality", Sdf.ValueTypeNames.Int).Set(0)
        vars = []
        aovs = ["LdrColor", "DistanceToImagePlaneSD", "CameraParams"]
        if args.camera_mode == "simulated_d455" and name != "global":
            # NVIDIA's documented Single View Depth Camera renderer schema. This
            # is the same post-process model family as the Isaac sensor wrapper,
            # with provisional noise and nominal D45595mm baseline explicitly set.
            product.GetPrim().SetMetadata("apiSchemas", Sdf.TokenListOp.Create(prependedItems=["OmniSensorDepthSensorSingleViewAPI"]))
            parameters = {"enabled": True, "baselineMM": 95.0,
                "focalLengthPixel": float(calibration[name]["K"][0][0]), "sensorSizePixel": float(args.width),
                "maxDisparityPixel": 150.0, "confidenceThreshold": args.confidence, "noiseMean": .25,
                "noiseSigma": .25, "noiseDownscaleFactorPixel": 1.0, "minDistance": .52,
                "maxDistance": 6.0, "outlierRemovalEnabled": True, "rgbDepthOutputMode": 0,
                "effectiveDistance": args.ir_effective_distance}
            for key, value in parameters.items():
                kind = (Sdf.ValueTypeNames.Bool if isinstance(value, bool) else
                        Sdf.ValueTypeNames.Int if isinstance(value, int) else Sdf.ValueTypeNames.Float)
                product.GetPrim().CreateAttribute("omni:rtx:post:depthSensor:" + key, kind).Set(value)
            aovs += ["Camera3dPositionSD", "DepthSensorDistance", "DepthSensorPointCloudPosition"]
        for aov in aovs:
            var = UsdRender.Var.Define(stage, "/Render/Vars/" + aov)
            var.GetSourceNameAttr().Set(aov)
            vars.append(var.GetPath())
        product.GetOrderedVarsRel().SetTargets(vars)
        products.append(str(product.GetPath()))
    # Use simple USD PreviewSurface materials: MDL packages are not part of the
    # bounded official geometry cache and must not be silently downloaded.
    colors = {"Franka_0": (.8, .82, .85), "Franka_1": (.8, .82, .85),
              "D455_0": (.2, .25, .3), "D455_1": (.2, .25, .3),
              "ContinuousSpecimen": (.72, .32, .13), "KinematicBlade": (.38, .65, .85),
              "Table": (.25, .28, .3)}
    for name, color in colors.items():
        prim = stage.GetPrimAtPath("/World/" + name)
        if not prim:
            continue
        material = UsdShade.Material.Define(stage, "/World/ProbeMaterials/" + name)
        shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Surface")
        shader.CreateIdAttr("UsdPreviewSurface")
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.7)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material, UsdShade.Tokens.strongerThanDescendants)
    dome = UsdLux.DomeLight.Define(stage, "/World/ProbeDome")
    dome.CreateIntensityAttr(500)
    distant = UsdLux.DistantLight.Define(stage, "/World/ProbeSun")
    distant.CreateIntensityAttr(1000)
    UsdGeom.Xformable(distant).AddRotateXYZOp().Set(Gf.Vec3f(-35, 20, -45))
    stage.GetRootLayer().Save()
    write_json(output / "calibration.json", calibration)
    write_json(output / "prepared.json", {"products": products,
         "time_s": [float(tc) / stage.GetTimeCodesPerSecond() for tc in samples],
         "source_scene": str(args.scene.resolve()), "material_scope": "simple USD PreviewSurface overrides",
         "camera_mode": args.camera_mode,
         "depth_model_parameters": parameters if args.camera_mode == "simulated_d455" else None})


def render(args):
    import ovrtx
    import ovstage
    output = args.output_dir
    config = ovrtx.RendererConfig(sync_mode=True, active_cuda_gpus=str(args.gpu),
        log_file_path=str((output / "renderer.log").resolve()), log_level="warning",
        enable_crash_reporter=False, keep_system_alive=False)
    print("Initializing OVRTX; the first shader compilation can take several minutes", flush=True)
    renderer = ovrtx.Renderer(config=config)
    stage = ovstage.Stage("robotic_cutting_sim.rgbd")
    renderer.attach_ovstage(stage)
    prepared = json.loads((output / "prepared.json").read_text())
    calibration = json.loads((output / "calibration.json").read_text())
    rows, previous, changed = [], {}, {name: 0 for name in CAMERAS}
    try:
        ordinal = 1
        ovstage.population.open_usd(stage, str((output / "rtx_scene.usda").resolve()), ordinal=ordinal)
        stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
        products = None
        for index, timestamp in enumerate(prepared["time_s"]):
            tick = time.perf_counter()
            ordinal += 1
            ovstage.population.update_from_usd_time_async(stage, ordinal=ordinal, time=timestamp).wait()
            stage.advance_write_floor(ordinal, ovstage.Scope.ALL).wait()
            renderer.reset(time=timestamp)
            # Geometry is held at the measured source timestamp during convergence.
            # Renderer capture times are recorded separately from solver time.
            for _ in range(args.warmup):
                products = renderer.step(set(prepared["products"]), delta_time=1 / 60, ordinal=ordinal)
            row = {"physics_timestamp_s": timestamp, "surface_timestamp_s": timestamp,
                   "stage_ordinal": ordinal, "cameras": {}}
            for name in CAMERAS:
                product = products["/Render/" + name]
                if not product.frames:
                    raise RuntimeError(f"No OVRTX sensor frame from {name}")
                frame = product.frames[-1]
                arrays = {}
                aovs = ["LdrColor", "DistanceToImagePlaneSD", "CameraParams"]
                modeled = args.camera_mode == "simulated_d455" and name != "global"
                if modeled:
                    aovs += ["DepthSensorDistance", "DepthSensorPointCloudPosition"]
                for aov in aovs:
                    mapped = frame.render_vars["/Render/Vars/" + aov].map(device=ovrtx.Device.CPU)
                    view = np.from_dlpack(mapped)
                    arrays[aov] = view.copy()
                    del view
                    mapped.unmap()
                    del mapped
                rgb = arrays["LdrColor"][..., :3]
                runtime_camera = decode_camera_params(arrays["CameraParams"])
                if not np.isclose(runtime_camera["meters_per_scene_unit"], 1.0):
                    raise RuntimeError("RTX runtime did not retain SI stage metadata")
                if not np.allclose(runtime_camera["T_world_camera"], calibration[name]["T_world_camera"], atol=1e-5):
                    raise RuntimeError(f"Runtime camera pose differs from authored calibration: {name}")
                raw_depth = arrays["DepthSensorDistance" if modeled else "DistanceToImagePlaneSD"].squeeze(-1)
                valid = np.isfinite(raw_depth) & (raw_depth > 0) & (raw_depth < 20)
                if modeled:
                    valid &= (raw_depth >= .52) & (raw_depth <= 6.0)
                depth = np.where(valid, raw_depth, 0).astype(np.float32)
                payload = validate_frame(rgb, depth, timestamp, timestamp)
                payload["rgb_depth_aligned"] = not modeled
                target = output / ("global_debug" if name == "global" else "observations")
                np.savez_compressed(target / f"{name}_{index:04d}.npz", **payload,
                    camera_params_raw=arrays["CameraParams"], renderer_capture_start_s=frame.start_time,
                    renderer_capture_end_s=frame.end_time, raw_depth=raw_depth)
                if modeled:
                    np.savez_compressed(output / "camera_debug" / f"{name}_{index:04d}.npz",
                        pointcloud_position_raw=arrays["DepthSensorPointCloudPosition"],
                        depth_sensor_distance_raw=arrays["DepthSensorDistance"],
                        ideal_distance_to_image_plane_raw=arrays["DistanceToImagePlaneSD"])
                Image.fromarray(rgb).save(target / f"{name}_{index:04d}.png")
                if name in previous:
                    common = (depth > 0) & (previous[name] > 0)
                    changed[name] += int(np.sum(common & (np.abs(depth - previous[name]) > 1e-5)))
                previous[name] = depth.copy()
                row["cameras"][name] = {"rgb_shape": list(rgb.shape), "depth_shape": list(depth.shape),
                    "valid_pixels": int(valid.sum()), "renderer_capture_start_s": frame.start_time,
                    "renderer_capture_end_s": frame.end_time, "runtime_camera": runtime_camera}
                if modeled:
                    ideal = arrays["DistanceToImagePlaneSD"].squeeze(-1)
                    common = valid & np.isfinite(ideal) & (ideal > 0) & (ideal < 20)
                    differences = np.abs(depth[common] - ideal[common])
                    row["cameras"][name]["depth_model_diagnostics"] = {
                        "common_valid_pixels": int(common.sum()),
                        "mean_abs_difference_from_ideal_m": float(differences.mean()) if len(differences) else None,
                        "max_abs_difference_from_ideal_m": float(differences.max()) if len(differences) else None,
                        "ideal_valid_sensor_invalid_pixels": int(np.sum((ideal > 0) & (ideal < 20) & ~valid))}
            row["wall_s"] = time.perf_counter() - tick
            rows.append(row)
            if index == 0:
                write_json(output / "runtime_calibration.json", {name: row["cameras"][name]["runtime_camera"]
                                                               for name in CAMERAS})
            print(f"Captured OVRTX frame={index} physics_t={timestamp:.3f}s wall={row['wall_s']:.3f}s", flush=True)
        valid = all(c["valid_pixels"] > 0 for r in rows for c in r["cameras"].values())
        return {"status": "PASS" if valid else "FAIL", "scope": "standalone OVRTX programmable RGB-D of measured USD replay",
                "renderer": "NVIDIA OVRTX", "mode": args.camera_mode, "frames": rows,
                "changed_depth_pixels_10um": changed, "rtx_array_capture_validated": valid,
                "isaac_sim_6_2_runtime_validated": False,
                "single_view_postprocess_arrays_validated": valid and args.camera_mode == "simulated_d455",
                "simulated_d455_validated": False,
                "physical_d455_calibration_validated": False,
                "depth_registration_status": "NOT_VALIDATED" if args.camera_mode == "simulated_d455" else "aligned ideal RGB-D",
                "depth_model": ("OmniSensorDepthSensorSingleViewAPI / DepthSensorDistance" if args.camera_mode == "simulated_d455"
                                else "ideal RTX DistanceToImagePlaneSD; native D455 authored optics"),
                "global_in_observation": False, "oracle_in_observation": False,
                "live_physics_sensor_sync": "NOT_RUN", "physical_cutting": "NOT_ASSESSED",
                "gpu_memory_after_capture": command(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv"]),
                "gpu_peak_memory": "NOT_MEASURED",
                "limitations": ["Simple USD PreviewSurface materials replace unresolved MDL",
                    "Recorded solver surfaces, not live Kit synchronization",
                    "Independent OVRTX; Isaac SingleViewDepthCameraSensor wrapper was not run",
                    "D455 asset optics and95mm nominal baseline; stereo noise is provisional and uncalibrated",
                    "RGB comes from the depth optical frame, not the native separate color camera",
                    "Renderer clock and source physics timestamp are both recorded and differ during warmup"]}
    finally:
        products = None
        renderer.detach_ovstage()
        stage.destroy()
        renderer.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--base-scene", type=Path, help="Optional workcell beneath a measured robot-only replay")
    parser.add_argument("--global-camera-path", default="/World/GlobalCamera", help="Debug camera prim; native D455 views are retained")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--camera-mode", choices=["ideal_rgbd", "simulated_d455"], default="ideal_rgbd")
    parser.add_argument("--confidence", type=float, default=.7)
    parser.add_argument("--ir-effective-distance", type=float, default=0.0)
    parser.add_argument("--prepare-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args)
        return 0
    if min(args.width, args.height, args.samples, args.warmup) < 1:
        parser.error("Resolution, samples and warmup must be positive")
    output = new_output(args.output_dir)
    (output / "observations").mkdir(); (output / "global_debug").mkdir()
    (output / "camera_debug").mkdir()
    try:
        packages = {x: importlib.metadata.version(x) for x in ("ovrtx", "ovstage", "numpy")}
    except importlib.metadata.PackageNotFoundError as error:
        result = {"status": "BLOCKED", "reason": str(error), "ideal_fallback_used": False,
                  "rtx_array_capture_validated": False}
        write_json(output / "result.json", result)
        print(json.dumps(result, indent=2))
        return 2
    audit = {"packages": packages,
             "driver": command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"]),
             "gpu": args.gpu, "source_scene": str(args.scene.resolve()),
             "runtime_license": "NVIDIA Software License Agreement / Product Specific Terms for NVIDIA AI Products",
             "driver_docs": "https://nvidia-omniverse.github.io/ovrtx/driver_requirements.html"}
    write_json(output / "runtime_audit.json", audit)
    write_json(output / "manifest.json", manifest({**vars(args), "scene": str(args.scene.resolve()),
        "output_dir": str(output.resolve()), "base_scene": str(args.base_scene) if args.base_scene else None,
        "source_scene_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest()}, "standalone_ovrtx_rgbd"))
    try:
        arguments = [x for x in sys.argv[1:] if x != "--prepare-only"]
        subprocess.run([sys.executable, __file__, *arguments, "--prepare-only"], check=True)
        result = render(args)
    except Exception as error:
        (output / "failure_traceback.txt").write_text(traceback.format_exc())
        result = {"status": "FAIL", "reason": str(error), "rtx_array_capture_validated": False}
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
