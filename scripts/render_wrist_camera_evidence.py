#!/usr/bin/env python3
"""Encode captured wrist/global RGB-D arrays; no simulation, replay or renderer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from robotic_cutting_sim.recording import new_output, write_json


CAMERAS = ("wrist_d405_0", "wrist_d405_1", "global_d455")


class RecordedVideo:
    def __init__(self, path, fps):
        self.path, self.fps, self.count = path, fps, 0
        self.directory = path.parent/(path.stem+"_frames")
        self.directory.mkdir()

    def append_data(self, values):
        Image.fromarray(values).save(self.directory/f"frame_{self.count:05d}.png")
        self.count += 1

    def close(self):
        if self.count:
            encoder = shutil.which("ffmpeg")
            if encoder is None:
                try:
                    from imageio_ffmpeg import get_ffmpeg_exe
                except ImportError as error:
                    raise RuntimeError("No FFmpeg found; run this encoder with the Lab environment's pinned imageio-ffmpeg package") from error
                encoder = get_ffmpeg_exe()
            subprocess.run([encoder, "-n", "-loglevel", "error", "-framerate", str(self.fps),
                "-i", str(self.directory/"frame_%05d.png"), "-c:v", "libx264", "-threads", "2",
                "-pix_fmt", "yuv420p", str(self.path)], check=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_transforms(episode, output):
    source = episode/"camera_pose_trace.json"
    data = json.loads(source.read_text())
    rows = data["samples"] if isinstance(data, dict) else data
    times = np.asarray([row["timestamp_s"] for row in rows])
    world_camera = np.asarray([[row["cameras"][name]["T_world_camera"] for name in CAMERAS] for row in rows])
    camera_world = np.asarray([[row["cameras"][name]["T_camera_world"] for name in CAMERAS] for row in rows])
    world_hand = np.asarray([row["T_world_hand"] if "T_world_hand" in row else
        [row["cameras"][name]["T_world_hand"] for name in CAMERAS[:2]] for row in rows])
    hand_camera = np.asarray([[row["cameras"][name]["T_hand_camera"] for name in CAMERAS[:2]] for row in rows])
    if not np.allclose(hand_camera, hand_camera[0], rtol=0, atol=1e-12):
        raise ValueError("The recorded hand-to-camera mount changed within the episode")
    expected = world_hand @ hand_camera
    error = np.linalg.norm(expected[:, :, :3, 3]-world_camera[:, :2, :3, 3], axis=-1)
    np.savez_compressed(output/"camera_transforms.npz", time_s=times, camera_ids=np.asarray(CAMERAS),
        T_world_hand=world_hand, T_hand_camera=hand_camera, T_world_camera=world_camera,
        T_camera_world=camera_world, composition_position_error_m=error)
    values = np.column_stack([times, world_camera[:, :, :3, 3].reshape(len(rows), -1), error])
    header = "time_s,"+",".join(name+"_"+axis+"_m" for name in CAMERAS for axis in ("x", "y", "z"))+",composition_error_0_m,composition_error_1_m"
    np.savetxt(output/"camera_pose_trace.csv", values, delimiter=",", header=header, comments="")
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for index in (0, 1):
        predicted = expected[:, index, :3, 3]
        actual = world_camera[:, index, :3, 3]
        hand = world_hand[:, index, :3, 3]
        axes[0, index].plot(times, np.linalg.norm(predicted-predicted[0], axis=-1)*1000,
                            label="Measured hand @ fixed mount")
        axes[0, index].plot(times, np.linalg.norm(actual-actual[0], axis=-1)*1000, "--",
                            label="Recorded optical pose")
        axes[0, index].plot(times, np.linalg.norm(hand-hand[0], axis=-1)*1000, ":",
                            label="Measured hand translation")
        axes[0, index].set(xlabel="Simulation time (s)", ylabel="Translation from initial pose (mm)", title=CAMERAS[index])
        axes[0, index].legend()
        axes[1, 0].plot(times, error[:, index]*1e6, label=CAMERAS[index])
    axes[1, 0].set(xlabel="Simulation time (s)", ylabel="Composition translation error (micrometres)")
    axes[1, 0].legend()
    drift = np.linalg.norm(world_camera[:, 2, :3, 3]-world_camera[0, 2, :3, 3], axis=-1)
    axes[1, 1].plot(times, drift*1e6)
    axes[1, 1].set(xlabel="Simulation time (s)", ylabel="Fixed Global translation drift (micrometres)")
    fig.suptitle("Actual live hand and optical transforms; no pose animation or fitted alignment")
    fig.tight_layout()
    fig.savefig(output/"camera_pose_tracking.png", dpi=150)
    plt.close(fig)
    return {"file": str(source), "sha256": digest(source), "samples": len(rows)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    result = json.loads((args.episode/"result.json").read_text())
    manifest = json.loads((args.episode/"manifest.json").read_text())
    native = (args.episode/"frame_manifest.json").exists()
    if native:
        live = (manifest["backend"] == "LIVE_IsaacSim6.2_cutting" and
                result.get("robot_dynamics_source") == "Isaac PhysX" and result.get("replay_inputs") is False)
    else:
        live = manifest["backend"] == "live_isaac_lab" and result.get("source") == "live_isaac_lab"
    if not live or result.get("status") not in ("PASS_SMOKE", "PASS_EXECUTION"):
        raise ValueError("Expected an actual live Isaac capture with a successful execution receipt")
    if result.get("camera_layout") != "wrist_d405_global_d455":
        raise ValueError("Expected the named wrist D405 / fixed Global D455 layout")
    output = new_output(args.output_dir)
    if native:
        records = json.loads((args.episode/"frame_manifest.json").read_text())
        rows = [{"frame": row["frame"], "time_s": row["time_s"],
                 "calibration": row["camera_calibration"]} for row in records]
    else:
        rows = []
        for path in sorted((args.episode/"rgbd").glob("frame_*.npz")):
            index = int(path.stem.split("_")[-1])
            data = np.load(path)
            rows.append({"frame": index, "time_s": float(data["timestamp_s"]),
                         "calibration": json.loads((path.parent/f"calibration_{index:05d}.json").read_text())})
    if len(rows) < 2:
        raise ValueError("At least two captured timestamps are needed for a video")
    fps = 1/float(np.median(np.diff([row["time_s"] for row in rows])))
    writers = {name: RecordedVideo(output/(name+".mp4"), fps)
               for name in (*CAMERAS, *(name+"_depth" for name in CAMERAS), "synchronized_multiview")}
    inputs = []
    try:
        for row in rows:
            rgb_tiles, depth_tiles = [], []
            for name in CAMERAS:
                path = (args.episode/"observations"/f"{name}_{row['frame']:04d}.npz" if native else
                        args.episode/"rgbd"/f"frame_{row['frame']:05d}.npz")
                data = np.load(path)
                rgb = data["rgb" if native else name+"_rgb"]
                depth = data["depth_m" if native else name+"_depth_m"]
                valid = data["valid_mask" if native else name+"_valid_mask"]
                calibration = row["calibration"][name]
                if rgb.shape != (*depth.shape, 3) or depth.shape != valid.shape:
                    raise ValueError("Captured RGB-D dimensions disagree")
                if rgb.shape[:2] != (calibration["height"], calibration["width"]):
                    raise ValueError("Captured arrays disagree with recorded calibration")
                model = calibration["sensor_model"]
                title = f"{name} | {model} | t={row['time_s']:.2f}s"
                picture = Image.fromarray(rgb)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, picture.width, 25), fill="black")
                draw.text((5, 7), title, fill="white")
                writers[name].append_data(np.asarray(picture))
                rgb_tiles.append(np.asarray(picture.resize((640, 480))))
                near, far = calibration["near_m"], calibration["far_m"]
                colored = (matplotlib.colormaps["turbo"](np.clip((depth-near)/(far-near), 0, 1))[..., :3]*255).astype(np.uint8)
                colored[~valid] = 0
                picture = Image.fromarray(colored)
                draw = ImageDraw.Draw(picture)
                draw.rectangle((0, 0, picture.width, 25), fill="black")
                draw.text((5, 7), f"{name} | optical Z {near:.2f}-{far:.2f}m | invalid=black | t={row['time_s']:.2f}s", fill="white")
                writers[name+"_depth"].append_data(np.asarray(picture))
                depth_tiles.append(np.asarray(picture.resize((640, 480))))
                inputs.append({"frame": row["frame"], "camera_id": name,
                               "file": str(path), "sha256": digest(path)})
            mosaic = np.vstack([np.hstack(rgb_tiles), np.hstack(depth_tiles)])
            writers["synchronized_multiview"].append_data(mosaic)
            if row is rows[len(rows)//2]:
                Image.fromarray(mosaic).save(output/"preview.png")
    finally:
        for writer in writers.values():
            writer.close()
    files = [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest(path)}
             for path in sorted(output.glob("*.mp4"))]
    pose_source = export_transforms(args.episode, output)
    auxiliary = [{"file": name, "bytes": (output/name).stat().st_size, "sha256": digest(output/name)}
                 for name in ("preview.png", "camera_transforms.npz", "camera_pose_tracking.png", "camera_pose_trace.csv")]
    write_json(output/"provenance.json", {
        "source_episode": str(args.episode.resolve()), "source_manifest_sha256": digest(args.episode/"manifest.json"),
        "source_result_sha256": digest(args.episode/"result.json"), "source_backend": manifest["backend"],
        "source_execution_status": result["status"], "source_task_metric_status": result.get("task_goal_status", "SEPARATE_EVALUATION"),
        "encoder_source_sha256": digest(__file__), "source_kind": "Actual captured live RGB-D arrays",
        "simulation_or_renderer_invoked": False, "frames": len(rows), "playback_fps": fps,
        "camera_ids": CAMERAS, "sensor_models": {name: rows[0]["calibration"][name]["sensor_model"] for name in CAMERAS},
        "raw_arrays_modified": False, "invalid_pixels": "black; no filling or tail rejection",
        "display": "Individual videos preserve input dimensions; mosaic panels resized to640x480 for display only",
        "input_payload_sha256": inputs, "files": files, "pose_source": pose_source, "auxiliary_files": auxiliary})
    print(json.dumps({"output": str(output), "videos": len(files), "frames": len(rows), "fps": fps}))


if __name__ == "__main__":
    main()
