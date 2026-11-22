"""Read a completed Native recording and check the actual 50 Hz data on disk."""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np


CAMERAS = {
    "cut": {
        "wrist_d405_0": True,
        "wrist_d405_1": True,
        "global_d455": True,
        "overview_rgb": False,
        "closeup_rgb": False,
    },
    "trim": {"wrist_d405_0": True, "global_d455": False},
}
PRESENTATION = {"overview_rgb", "closeup_rgb"}


def read_frames(path, times, rgbd, calibrated):
    size = None
    initial_native_time = None
    for file, time_s in zip(path, times, strict=True):
        with np.load(file, allow_pickle=False) as frame:
            rgb = frame["rgb"]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(f"Invalid RGB payload in {file}")
            if size is None:
                size = rgb.shape[:2]
            if rgb.shape[:2] != size or abs(float(frame["physics_timestamp_s"]) - time_s) > 1e-9:
                raise ValueError(f"RGB shape or physics timestamp mismatch in {file}")
            if ("depth_m" in frame) != rgbd or ("valid_mask" in frame) != rgbd:
                raise ValueError(f"Depth modality mismatch in {file}")
            if rgbd:
                depth, mask = frame["depth_m"], frame["valid_mask"]
                if (depth.shape != size or depth.dtype != np.float32 or mask.dtype != np.bool_
                        or not np.isfinite(depth).all() or np.any(depth < 0)
                        or not np.array_equal(mask, depth > 0)):
                    raise ValueError(f"Invalid optical-Z payload in {file}")
            if calibrated:
                K, pose = frame["K"], frame["T_world_camera"]
                if (K.shape != (3, 3) or pose.shape != (4, 4) or not np.isfinite(K).all()
                        or not np.isfinite(pose).all() or min(K[0, 0], K[1, 1]) <= 0
                        or not np.allclose(pose @ frame["T_camera_world"], np.eye(4), rtol=0, atol=1e-5)):
                    raise ValueError(f"Invalid camera calibration in {file}")
                if any(abs(float(frame[key]) - time_s) > 1e-9 for key in
                       ("surface_timestamp_s", "camera_extrinsics_timestamp_s")):
                    raise ValueError(f"Camera and scene times differ in {file}")
                native_time = float(frame["native_physics_timestamp_s"])
                if initial_native_time is None:
                    initial_native_time = native_time
                if not np.isfinite(native_time) or abs(native_time - initial_native_time - time_s) > 1e-6:
                    raise ValueError(f"Native physics clock mismatch in {file}")
                if not rgbd and list(frame["modalities"]) != ["rgb"]:
                    raise ValueError(f"Missing RGB-only modality declaration in {file}")
    return {"frames": len(path), "height": size[0], "width": size[1], "rgb": True, "depth_m": rgbd}


def audit(episode, task, steps):
    times = np.arange(steps + 1, dtype=float) / 50
    rows = json.loads((episode / "frame_manifest.json").read_text())
    indices = [row["frame"] for row in rows]
    frame_times = [row["time_s"] for row in rows]
    if indices != list(range(steps + 1)) or not np.allclose(frame_times, times, rtol=0, atol=1e-9):
        raise ValueError("Frame manifest must contain step zero and every control timestamp")
    with (episode / "actions.jsonl").open() as log:
        actions = [json.loads(line) for line in log]
    if ([row["step"] for row in actions] != list(range(1, steps + 1))
            or not np.allclose([row["time_s"] for row in actions], times[1:], rtol=0, atol=1e-9)
            or any(not {"q", "qdot", "tool_pose"} <= row["measured"].keys() for row in actions)):
        raise ValueError("Submitted actions and measured robot/tool state are missing or misaligned")
    with np.load(episode / "robot_state.npz", allow_pickle=False) as state:
        if not np.allclose(state["time_s"], times, rtol=0, atol=1e-9) or len(state["body_pose"]) != len(times):
            raise ValueError("Recorded robot states do not match the full frame timeline")
    files = {}
    for path in (episode / "observations").glob("*.npz"):
        match = re.fullmatch(r"(.+)_(\d{4})\.npz", path.name)
        if match is None:
            raise ValueError(f"Unexpected camera frame name {path.name}")
        name, index = match.groups()
        files.setdefault(name, {})[int(index)] = path
    if files.keys() != CAMERAS[task].keys():
        raise ValueError(f"Camera streams differ from the benchmark rig: {sorted(files)}")
    streams = {}
    for name, by_index in files.items():
        if sorted(by_index) != indices:
            raise ValueError(f"Missing or surplus camera frames: {name}")
        paths = [by_index[index] for index in indices]
        streams[name] = read_frames(paths, times, CAMERAS[task][name], name not in PRESENTATION)
    return {"status": "PASS", "task": task, "actions": steps, "frame_times": len(times),
            "invalid_or_mismatched_timestamps": 0, "cameras": streams}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode", type=Path)
    p.add_argument("--task", choices=tuple(CAMERAS), required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    try:
        result = audit(args.episode, args.task, args.steps)
    except (OSError, KeyError, ValueError) as error:
        result = {"status": "FAIL", "reason": str(error)}
    result["recording"] = str(args.episode.resolve())
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    sys.exit(main())
