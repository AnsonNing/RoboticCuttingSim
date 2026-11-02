#!/usr/bin/env python3
"""Render a new fixed Global camera from saved Native states; never step physics.

Original wrist captures remain byte-identical. These are explicitly camera
replays of measured states, not additional live cutting experiments. Missing
states, time mismatches and unmeshable material refuse rendering.
"""
import argparse
import gzip
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from robotic_cutting_sim.camera_rig import CameraRig, pose_matrix
from robotic_cutting_sim.demo_media import sha256_file
from robotic_cutting_sim.isaac_sim_runner import (IsaacSimCuttingConfig,
    calibration_from_replicator, launch_app, numpy_data, runtime_versions)


def load_json(path):
    return json.loads(path.read_text())


def exact_index(times, timestamp):
    i = int(np.argmin(abs(np.asarray(times) - timestamp)))
    if abs(float(times[i]) - timestamp) > 1e-7:
        raise ValueError(f'Missing measured state at {timestamp}; interpolation refused')
    return i


def require_saved_surfaces(episode, frames):
    """Positions alone cannot reproduce current-volume or cut-aware surfaces."""
    for row in frames:
        path=episode/'surface_debug'/f"surface_{row['frame']:04d}.npz"
        if not path.exists():
            raise ValueError(f'Missing exact saved surface {path}; re-record required, no substitute surface')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--indices', type=int, nargs='+')
    parser.add_argument('--rig-profile', default='wrist_camera_rig_v9')
    args = parser.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    (out / 'observations').mkdir()
    started = time.perf_counter()
    app = None
    try:
        app = launch_app(IsaacSimCuttingConfig(gpu=args.gpu, camera=False), out)
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.sensors.experimental.rtx import RtxCamera, SingleViewDepthCameraSensor
        from pxr import Gf, Usd, UsdGeom
        from robotic_cutting_sim.camera_rig import _set_transform
        from robotic_cutting_sim.cutting_surface import CurrentMechanicalPartition, CuttingSurface
        from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge

        episode = args.episode.resolve()
        frames = load_json(episode / 'frame_manifest.json')
        selected = frames if args.indices is None else [r for r in frames if r['frame'] in args.indices]
        if args.indices is not None and len(selected) != len(args.indices):
            raise ValueError('Requested capture index missing')
        require_saved_surfaces(episode,selected)
        robot_path = episode / 'robot_state.npz'
        robot = np.load(robot_path)
        body_paths = robot['body_prim_paths'].tolist()
        source_scene = episode / 'final_live_scene.usda'
        if not source_scene.exists():
            zipped = source_scene.with_suffix('.usda.gz')
            if not zipped.exists():
                raise FileNotFoundError(source_scene)
            source_scene = out / 'source_scene.usda'
            with gzip.open(zipped, 'rb') as src, source_scene.open('xb') as dest:
                dest.write(src.read())
        # A flattened live stage contains old Replicator compute graphs. They
        # are runtime products, not scene geometry; rebuilding them avoids
        # duplicate graph ownership when attaching the new native sensor.
        prepared = Usd.Stage.Open(str(source_scene))
        for path in ('/Render', '/Replicator', '/Orchestrator'):
            if prepared.GetPrimAtPath(path):
                prepared.RemovePrim(path)
        for prim in list(prepared.Traverse()):
            if prim.GetTypeName() in ('OmniGraph', 'RenderProduct'):
                prim.SetActive(False)
        prepared.GetRootLayer().Export(str(out / 'render_scene.usda'))
        context = omni.usd.get_context()
        context.open_stage(str(out / 'render_scene.usda'))
        for _ in range(10):
            app.update()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        timeline = omni.timeline.get_timeline_interface()
        timeline.stop()
        camera_path = frames[0]['camera_calibration']['global_d455']['prim_path']
        camera = stage.GetPrimAtPath(camera_path)
        if not camera:
            raise ValueError(f'Original Global camera missing: {camera_path}')
        task_path=episode/'task.json'
        if task_path.exists():
            task_record=load_json(task_path)
        else:
            task_record=load_json(episode/'setup.json')['task']
        task_frame=pose_matrix(task_record['world_from_workpiece_xyzw'])
        target=task_frame[:3,:3]@np.array([0.,0.,task_record['geometry']['base_height_m']])+task_frame[:3,3]
        profile = CameraRig(rig_profile=args.rig_profile,global_target_world_m=target).specs['global_d455']
        transform = np.asarray(profile['T_world_camera'])
        # Move the actual existing Global housing, retaining its optical offsets.
        root = stage.GetPrimAtPath('/World/WristCameraRig/GlobalD455')
        if not root or not camera_path.startswith(str(root.GetPath()) + '/'):
            raise ValueError('Unsupported source Global housing hierarchy')
        old_root = np.asarray(UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        old_optical = np.asarray(UsdGeom.Xformable(camera).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
        optical_local = np.linalg.inv(old_root) @ old_optical
        _set_transform(root, transform @ np.diag([1., -1., -1., 1.]) @ np.linalg.inv(optical_local))
        width, height = profile['width'], profile['height']
        sensor = SingleViewDepthCameraSensor(RtxCamera(camera_path, tick_rate=0.), resolution=(height, width),
            annotators=['distance_to_image_plane', 'depth_sensor_distance'])
        sensor.set_sensor_baseline(95.)
        sensor.set_sensor_focal_length(317.0431137)
        sensor.set_sensor_size(640.)
        sensor.set_sensor_maximum_disparity(150.)
        sensor.set_sensor_disparity_confidence(1.)
        sensor.set_sensor_distance_cutoffs(.52, 6.)
        sensor.set_sensor_noise_parameters(0., 0.)
        sensor.set_sensor_disparity_noise_downscale(1.)
        sensor.set_enabled_outlier_removal(True)
        sensor.set_sensor_output_mode(0)
        sensor.set_enabled_post_processing(True)
        product = str(sensor.render_product.GetPath())
        stage.GetPrimAtPath(product).GetAttribute('omni:rtx:post:depthSensor:effectiveDistance').Set(4.)
        rgb_reader = rep.AnnotatorRegistry.get_annotator('rgb')
        rgb_reader.attach(product)
        params = rep.AnnotatorRegistry.get_annotator('camera_params')
        params.attach(product)
        bridge = DynamicSurfaceBridge(stage)
        audit = []
        for row in selected:
            timestamp, index = row['time_s'], row['frame']
            bi = exact_index(robot['time_s'], timestamp)
            for path, pose in zip(body_paths, robot['body_pose'][bi]):
                prim = stage.GetPrimAtPath(path)
                if not prim:
                    raise ValueError(f'Missing recorded robot/tool body {path}')
                _set_transform(prim, pose_matrix(pose))
                UsdGeom.Xformable(prim).SetResetXformStack(True)
            mesh_path = episode / 'surface_debug' / f'surface_{index:04d}.npz'
            with np.load(mesh_path) as mesh:
                if abs(float(mesh['physics_timestamp_s']) - timestamp) > 1e-7:
                    raise ValueError('Stale saved mesh')
                vertices, triangles = mesh['vertices_m'], mesh['triangles']
                normals=mesh['face_varying_normals'] if 'face_varying_normals' in mesh else None
            mesh_info = {'method': 'original_saved_surface', 'source_sha256': sha256_file(mesh_path)}
            staged = (bridge.update(vertices, triangles, timestamp) if normals is None else
                bridge.update_face_varying_v1(vertices,triangles,normals,timestamp))
            # Rendering is stopped-time only: no simulation/PhysX/material stepping.
            for _ in range(10):
                rep.orchestrator.step(rt_subframes=1, delta_time=0.)
            body_error = max(float(np.abs(
                np.asarray(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
                - pose_matrix(pose)).max()) for path, pose in zip(body_paths, robot['body_pose'][bi]))
            if body_error > 1e-5:
                raise ValueError('Rendered robot/tool moved away from measured Native state')
            calibration = calibration_from_replicator(params.get_data())
            if not np.allclose(calibration['T_world_camera'], transform, atol=1e-5, rtol=0):
                raise ValueError('Native camera extrinsics differ from requested physical pose')
            if not np.allclose(calibration['K'], [[317.0431137, 0, 320], [0, 317.0431137, 240], [0, 0, 1]], atol=.002, rtol=0):
                raise ValueError('Native camera lens changed from controlled v5 optics')
            rgb = numpy_data(rgb_reader.get_data())[..., :3]
            raw = numpy_data(sensor.get_data('depth_sensor_distance')[0]).reshape(height, width)
            ideal = numpy_data(sensor.get_data('distance_to_image_plane')[0]).reshape(height, width)
            valid = np.isfinite(raw) & (raw >= profile['near_m']) & (raw <= profile['far_m'])
            original = episode / 'observations' / f'global_d455_{index:04d}.npz'
            with np.load(original) as old:
                native_time = float(old['native_physics_timestamp_s'])
            target = out / 'observations' / f'global_d455_{index:04d}.npz'
            np.savez_compressed(target, rgb=rgb, depth_m=np.where(valid, raw, 0), valid_mask=valid,
                native_depth_raw=raw, ideal_depth_m=ideal, K=calibration['K'], T_world_camera=calibration['T_world_camera'],
                physics_timestamp_s=timestamp, surface_timestamp_s=timestamp, camera_extrinsics_timestamp_s=timestamp,
                native_physics_timestamp_s=native_time, camera_id='global_d455',
                live_source='Isaac Sim6.2 native camera replay of measured Native states; no physics rerun',
                camera_render_source='saved_native_state_replay_v1', rig_profile=args.rig_profile)
            audit.append({'frame': index, 'time_s': timestamp, 'robot_state_index': bi, 'surface': mesh_info,
                'maximum_rendered_body_matrix_error': body_error,
                'staged': staged, 'capture_sha256': sha256_file(target), 'original_global_sha256': sha256_file(original),
                'valid_depth_fraction': float(valid.mean())})
            (out / 'progress.json').write_text(json.dumps(audit, indent=2) + '\n')
            print(f'GLOBAL replay {index}: {timestamp:.2f}s', flush=True)
        receipt = {'schema': 'native_global_camera_replay_v1', 'status': 'PASS_RENDER', 'source_episode': str(episode),
            'runtime_source': 'native_camera_replay', 'replay_inputs': True,
            'physics_runs': 0, 'source_native_outcome_unchanged': True, 'wrist_captures_modified': False,
            'rendering_only': True, 'profile': profile, 'calibration': calibration, 'frames': audit,
            'source_sha256': {str(p): sha256_file(p) for p in [robot_path, episode/'frame_manifest.json',
                episode/'final_live_scene.usda' if (episode/'final_live_scene.usda').exists() else episode/'final_live_scene.usda.gz',
                *[episode/'surface_debug'/f"surface_{row['frame']:04d}.npz" for row in selected]]},
            'implementation_sha256': sha256_file(Path(__file__)), 'versions': runtime_versions(),
            'profile_sha256': sha256_file(ROOT/'configs'/f'{args.rig_profile}.json'),
            'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'implementation_modules_sha256': {str(ROOT/'src'/'robotic_cutting_sim'/name):
                sha256_file(ROOT/'src'/'robotic_cutting_sim'/name) for name in
                ('camera_rig.py','cutting_surface.py','cut_topology.py','usd_surface_bridge.py')},
            'command': sys.argv, 'wall_s': time.perf_counter() - started}
        (out / 'replay_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    except Exception:
        (out / 'failure.txt').write_text(traceback.format_exc())
        raise
    finally:
        if app is not None:
            app.close(wait_for_replicator=False)


if __name__ == '__main__':
    main()
