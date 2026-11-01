#!/usr/bin/env python3
"""Capture known-geometry native camera controls in real Isaac Sim6.2/Kit.

No robot physics, material dynamics or cutting acceptance is claimed here. Raw
native depth/cloud and separate RTX reference remain unchanged. Geometry is
held at each fixture time while the native sensor renderer converges.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig, calibration_from_replicator, launch_app, numpy_data, runtime_versions
from robotic_cutting_sim.recording import manifest, new_output, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--noise", choices=["off", "on"], required=True)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    for directory in ("observations", "camera_debug", "global_debug"):
        (output / directory).mkdir()
    fixture = json.loads((args.fixture_dir / "fixture.json").read_text())
    protocol = json.loads((args.fixture_dir / "protocol.json").read_text())
    optics = protocol['runtime_profile']
    width, height = optics.get('width',640), optics.get('height',480)
    focal = optics.get('focal_length_px',317.0431137084961)
    disparity = optics.get('maximum_disparity_px',150.)
    write_json(output/"manifest.json", manifest({"fixture_dir":str(args.fixture_dir.resolve()),
        "protocol_sha256":fixture["protocol_sha256"], "gpu":args.gpu, "noise":args.noise},
        "real_isaac_sim_kit_native_camera_controls"))
    noise = 0. if args.noise == "off" else .25
    config = IsaacSimCuttingConfig(gpu=args.gpu, device=f"cuda:{args.gpu}", render_warmup=10)
    app = None
    try:
        app = launch_app(config, output)
        import omni.kit.app
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.sensors.experimental.rtx import CameraSensor, RtxCamera, SingleViewDepthCameraSensor
        from pxr import Gf, Usd, UsdGeom, UsdLux
        context = omni.usd.get_context()
        context.open_stage(fixture["scene"])
        for _ in range(5):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        stage.SetInterpolationType(Usd.InterpolationTypeHeld)
        dome = UsdLux.DomeLight.Define(stage, "/World/ProbeDome")
        dome.CreateIntensityAttr(500.)
        sun = UsdLux.DistantLight.Define(stage, "/World/ProbeSun")
        sun.CreateIntensityAttr(1000.)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-35, 20, -45))
        sensors, params, rgb_annotators = {}, {}, {}
        native_parameters = {"enabled": True, "baselineMM": 95., "focalLengthPixel": 317.0431137084961,
            "sensorSizePixel": 640., "maxDisparityPixel": 150., "confidenceThreshold": 1.,
            "noiseMean": noise, "noiseSigma": noise, "noiseDownscaleFactorPixel": 1.,
            "minDistance": .52, "maxDistance": 6., "outlierRemovalEnabled": True,
            "rgbDepthOutputMode": 0, "effectiveDistance": 4.}
        for name, (camera_path, _) in fixture["cameras"].items():
            if name == "global":
                sensor = CameraSensor(RtxCamera(camera_path, tick_rate=0.), resolution=(height,width),
                                      annotators=["rgb", "distance_to_image_plane"])
            else:
                sensor = SingleViewDepthCameraSensor(RtxCamera(camera_path, tick_rate=0.), resolution=(height,width),
                    annotators=["distance_to_image_plane", "depth_sensor_distance", "depth_sensor_point_cloud_position"])
                sensor.set_sensor_baseline(95.)
                sensor.set_sensor_focal_length(focal)
                sensor.set_sensor_size(float(width))
                sensor.set_sensor_disparity_confidence(1.)
                sensor.set_sensor_maximum_disparity(disparity)
                sensor.set_sensor_noise_parameters(noise, noise)
                sensor.set_sensor_disparity_noise_downscale(1.)
                sensor.set_sensor_distance_cutoffs(.52, 6.)
                sensor.set_enabled_outlier_removal(True)
                sensor.set_sensor_output_mode(0)
                sensor.set_enabled_post_processing(True)
                # API does not expose an IR emitter/effective-range setter.
                stage.GetPrimAtPath(str(sensor.render_product.GetPath())).GetAttribute(
                    "omni:rtx:post:depthSensor:effectiveDistance").Set(4.)
                rgb_annotators[name] = rep.AnnotatorRegistry.get_annotator("rgb")
                rgb_annotators[name].attach(str(sensor.render_product.GetPath()))
            sensors[name] = sensor
            params[name] = rep.AnnotatorRegistry.get_annotator("camera_params")
            params[name].attach(str(sensor.render_product.GetPath()))
        manager = omni.kit.app.get_app().get_extension_manager()
        extension_id = manager.get_enabled_extension_id("omni.sensors.nv.camera")
        extension = manager.get_extension_dict(extension_id)
        runtime = {"packages": runtime_versions(), "kit_version": omni.kit.app.get_app().get_kit_version(),
            "app_version": omni.kit.app.get_app().get_app_version(),
            "kit_kernel_version": omni.kit.app.get_app().get_kernel_version(),
            "extensions": {"omni.sensors.nv.camera": extension.get("package",{}).get("version")},
            "camera_extension_id": extension_id}
        write_json(output / "runtime_audit.json", runtime)
        expected = protocol["runtime_profile"]
        assert runtime["packages"]["isaacsim"] == expected["isaacsim"]
        assert runtime["extensions"]["omni.sensors.nv.camera"] == expected["omni.sensors.nv.camera"]
        assert runtime["kit_version"].startswith(expected["kit_major_minor"])
        native_prim = stage.GetPrimAtPath(str(sensors["d455_0"].render_product.GetPath()))
        native_parameters = {key: native_prim.GetAttribute("omni:rtx:post:depthSensor:"+key).Get()
                             for key in native_parameters}
        write_json(output / "prepared.json", {"time_s":[case["time_s"] for case in fixture["frames"]],
            "fixture_sha256": hashlib.sha256((args.fixture_dir/"fixture.json").read_bytes()).hexdigest(),
            "protocol_sha256":fixture["protocol_sha256"], "depth_model_parameters": native_parameters})
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        rows = []
        for index, case in enumerate(fixture["frames"]):
            tick = time.perf_counter()
            timeline.set_current_time(case["time_s"])
            for _ in range(3):
                app.update()
            latest = {}
            for warmup in range(40):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.)
                latest = {name: {a:sensor.get_data(a) for a in (["rgb", "distance_to_image_plane"] if name == "global" else
                    ["distance_to_image_plane", "depth_sensor_distance", "depth_sensor_point_cloud_position"])}
                    for name,sensor in sensors.items()}
                for name, annotator in rgb_annotators.items():
                    rgb = annotator.get_data()
                    latest[name]["rgb"] = (rgb if np.asarray(rgb).size else None, {})
                if warmup >= 9 and all(value[0] is not None for data in latest.values() for value in data.values()):
                    break
            else:
                raise TimeoutError("Native Kit camera fixture produced no arrays")
            cameras = {}
            for name, data in latest.items():
                camera = calibration_from_replicator(params[name].get_data())
                pose = np.asarray(camera["T_world_camera"])
                if not np.allclose(pose[:3,3], fixture["cameras"][name][1], atol=1e-6, rtol=0):
                    raise RuntimeError(f"Runtime camera alias/wrong location for {name}")
                rgb = numpy_data(data["rgb"][0])[..., :3]
                ideal = numpy_data(data["distance_to_image_plane"][0]).reshape(height,width,1)
                target = output / ("global_debug" if name == "global" else "observations")
                np.savez_compressed(target / f"{name}_{index:04d}.npz", rgb=rgb,
                    camera_calibration_json=json.dumps(camera), timestamp_s=case["time_s"],
                    surface_timestamp_s=case["time_s"], live_camera_source="Isaac Sim6.2 Kit native sensor controls")
                if name != "global":
                    raw = numpy_data(data["depth_sensor_distance"][0]).reshape(height,width,1)
                    xyz = numpy_data(data["depth_sensor_point_cloud_position"][0]).reshape(height,width,-1)
                    np.savez_compressed(output / "camera_debug" / f"{name}_{index:04d}.npz",
                        depth_sensor_distance_raw=raw, pointcloud_position_raw=xyz,
                        ideal_distance_to_image_plane_raw=ideal)
                    cameras[name] = {"depth_shape":list(raw.shape),"cloud_shape":list(xyz.shape),
                        "raw_valid_pixels":int(np.sum(np.isfinite(raw)&(raw>=.52)&(raw<=6))), "runtime_camera":camera}
            rows.append({"index":index,"case_id":case["id"],"time_s":case["time_s"],"wall_s":time.perf_counter()-tick,"cameras":cameras})
            write_json(output/"frames.json",rows)
            print(f"KIT camera control {index+1}/{len(fixture['frames'])}: {case['id']}",flush=True)
        write_json(output/"result.json",{"status":"PASS_CAPTURE","runtime":runtime,"frames":rows,
            "scope":"Real Kit native sensor known-geometry controls; no robot/material dynamics", "simulated_d455_validated":False})
        return 0
    except Exception as error:
        (output/"failure.log").write_text(traceback.format_exc())
        write_json(output/"result.json",{"status":"FAIL_CAPTURE","error":str(error),"simulated_d455_validated":False})
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False)


if __name__ == "__main__":
    raise SystemExit(main())
