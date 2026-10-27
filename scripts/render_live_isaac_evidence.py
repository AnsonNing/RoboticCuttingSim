#!/usr/bin/env python3
"""Visualize already captured LIVE Kit data; never rerun/replay physics or render USD."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation


def video(directory,pattern,target,fps):
    subprocess.run(["ffmpeg","-n","-loglevel","error","-framerate",str(fps),"-i",str(directory/pattern),
        "-c:v","libx264","-pix_fmt","yuv420p",str(target)],check=True)


def measured_depth_frame(payload, camera_name, time_s, near, far):
    """Color captured measured values while preserving the native invalid mask."""
    cmap=colormaps["viridis"]
    z=payload["depth_m"];valid=payload["valid_mask"]
    rgb=(cmap(np.clip((z-near)/(far-near),0,1))[...,:3]*255).astype(np.uint8)
    rgb[~valid]=0
    depth=Image.fromarray(rgb)
    draw=ImageDraw.Draw(depth)
    draw.rectangle((0,0,depth.width-1,24),fill="black")
    draw.text((8,7),f"LIVE Isaac Sim6.2 | {camera_name} measured depth | t={time_s:.2f}s",fill="white")
    ramp=np.tile(np.linspace(0,1,300)[None,:],(12,1))
    bar=Image.fromarray((cmap(ramp)[...,:3]*255).astype(np.uint8))
    left=(depth.width-300)//2
    depth.paste(bar,(left,depth.height-32));draw=ImageDraw.Draw(depth)
    draw.text((left,depth.height-17),f"{near:.2f}m",fill="white")
    draw.text((left+268,depth.height-17),f"{far:.2f}m",fill="white")
    return depth


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("episode",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.mkdir()
    depth_directories={"d455_0":a.output/"depth_frames", "d455_1":a.output/"depth_frames_d455_1"}
    for directory in depth_directories.values():
        directory.mkdir()
    (a.output/"multiview_frames").mkdir()
    receipt=json.loads((a.episode/"result.json").read_text())
    if receipt.get("robot_dynamics_source")!="Isaac PhysX" or receipt.get("replay_inputs") is not False:
        raise ValueError("This visualization requires actual live Kit episode provenance")
    manifest=json.loads((a.episode/"frame_manifest.json").read_text())
    times=np.asarray([row["time_s"] for row in manifest])
    if len(times)<2 or not np.allclose(np.diff(times),np.diff(times)[0],rtol=0,atol=1e-8):
        raise ValueError("Video requires at least two uniformly timed live captures")
    fps=1/float(np.diff(times)[0])
    for name in ("global","d455_0","d455_1"):
        (a.output/"rgb_frames"/name).mkdir(parents=True)
    near,far=.52,1.2
    for row in manifest:
        i=row["frame"]
        depths={}
        for name,directory in depth_directories.items():
            with np.load(a.episode/"observations"/f"{name}_{i:04d}.npz") as payload:
                depths[name]=measured_depth_frame(payload,name,row["time_s"],near,far)
            depths[name].save(directory/f"frame_{i:04d}.png")
        canvas=Image.new("RGB",(1280,960))
        for name,location,label in [("global",(0,0),"Global: complete workstation"),
                 ("d455_0",(640,0),"D4550 RGB"),("d455_1",(0,480),"D4551 RGB")]:
            folder="global_debug" if name=="global" else "observations"
            picture=Image.open(a.episode/folder/f"{name}_{i:04d}.png").convert("RGB")
            text=ImageDraw.Draw(picture);text.rectangle((0,0,picture.width-1,24),fill="black")
            text.text((8,7),f"LIVE Isaac Sim6.2 | {label} | t={row['time_s']:.2f}s",fill="white")
            picture.save(a.output/"rgb_frames"/name/f"frame_{i:04d}.png")
            canvas.paste(picture.resize((640,480)),location)
        canvas.paste(depths["d455_0"],(640,480));canvas.save(a.output/"multiview_frames"/f"frame_{i:04d}.png")
    video(a.output/"depth_frames","frame_%04d.png",a.output/"live_measured_depth.mp4",fps)
    video(a.output/"depth_frames_d455_1","frame_%04d.png",a.output/"live_d455_1_measured_depth.mp4",fps)
    video(a.output/"multiview_frames","frame_%04d.png",a.output/"live_synchronized_multiview.mp4",fps)
    for name in ("global","d455_0","d455_1"):
        video(a.output/"rgb_frames"/name,"frame_%04d.png",a.output/f"live_{name}.mp4",fps)
    robot=np.load(a.episode/"robot_state.npz")
    actual=robot["tool_pose_xyzw"];desired=robot["desired_tool_pose_xyzw"];t=robot["time_s"]
    setup=json.loads((a.episode/"setup.json").read_text())
    task=setup.get("task")
    frame=np.asarray(setup["world_from_workpiece_xyzw"])
    rotation=Rotation.from_quat(frame[3:]).as_matrix()
    actual_local=(actual[:,:3]-frame[:3])@rotation
    desired_local=(desired[:,:3]-frame[:3])@rotation
    error=np.linalg.norm(actual[:,:3]-desired[:,:3],axis=1)
    angle=np.rad2deg((Rotation.from_quat(actual[:,3:]).inv()*Rotation.from_quat(desired[:,3:])).magnitude())
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    axes[0,0].plot(desired_local[:,0]*1000,desired_local[:,2]*1000,label="Commanded")
    axes[0,0].plot(actual_local[:,0]*1000,actual_local[:,2]*1000,label="Measured PhysX",linestyle="--")
    if task and task["goal_kind"]=="sever":
        axes[0,0].axvline(task["target_origin_local_m"][0]*1000,color="gray",label="Local sever plane")
    else:
        height=(task["target_origin_local_m"][2] if task else setup["material_setup"]["workpiece_local_target_height_m"])
        axes[0,0].axhline(height*1000,color="gray",label=f"Local target surface {height*1000:g}mm")
    axes[0,0].set(xlabel="Workpiece-local X(mm)",ylabel="Edge local Z(mm)");axes[0,0].legend()
    axes[0,1].plot(t,error*1000);axes[0,1].set(xlabel="Simulation time(s)",ylabel="Position error(mm)")
    axes[1,0].plot(t,actual_local[:,2]*1000,label="Measured");axes[1,0].plot(t,desired_local[:,2]*1000,label="Commanded")
    axes[1,0].set(xlabel="Simulation time(s)",ylabel="Edge local height(mm)");axes[1,0].legend()
    axes[1,1].plot(t,angle);axes[1,1].set(xlabel="Simulation time(s)",ylabel="Orientation error(deg)")
    fig.suptitle(f"LIVE Isaac Sim6.2 {task['task_id'] if task else 'D0'}: commanded versus actual articulated knife edge")
    fig.tight_layout();fig.savefig(a.output/"planned_vs_actual_trajectory.png",dpi=150);plt.close(fig)
    (a.output/"provenance.json").write_text(json.dumps({"source_episode":str(a.episode.resolve()),
        "source_manifest_sha256":hashlib.sha256((a.episode/"manifest.json").read_bytes()).hexdigest(),
        "visualization_script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source":"actual live Kit captured arrays, no USD replay or new renderer invocation",
        "depth_color_scale_m":[near,far],"native_invalid_depth_display":"black; no filling or error-tail rejection",
        "measured_depth_videos":{"d455_0":"live_measured_depth.mp4","d455_1":"live_d455_1_measured_depth.mp4"},
        "multiview_panels":["global RGB","d455_0 RGB","d455_1 RGB","d455_0 measured depth"],
        "frames":len(manifest),"playback_fps":fps},indent=2))


if __name__=="__main__":
    main()
