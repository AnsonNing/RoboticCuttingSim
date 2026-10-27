"""Numerical trace plot and particle video; explicitly a diagnostic view."""
import csv
import subprocess

import numpy as np


def make_preview(output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    data = np.load(output/"physics_state.npz")
    points, times = data["positions"], data["time_s"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    q = points[0]
    mask = np.abs(q[:,1]) < .003
    sc = ax.scatter(q[mask,0]*1000, q[mask,2]*1000, s=3, c=q[mask,2], cmap="copper", vmin=0, vmax=.025)
    ax.axhline(15, color="tab:green", linestyle="--", label="target plane")
    ax.set(xlim=(-75,75), ylim=(-10,65), xlabel="world x [mm]", ylabel="world z [mm]")
    ax.set_aspect("equal"); ax.legend(loc="upper left")
    half = data["tool_half_extents"]
    patch = Rectangle((0,0), 2*half[0]*1000, 2*half[2]*1000, color="steelblue", alpha=.7)
    ax.add_patch(patch)
    images = output/"preview_frames"; images.mkdir()
    for j,i in enumerate(np.unique(np.linspace(0,len(points)-1,min(90,len(points)),dtype=int))):
        q = points[i]; mask = np.abs(q[:,1]) < .003
        sc.set_offsets(q[mask][:,[0,2]]*1000); sc.set_array(q[mask,2])
        pose = data["tool_pose_xyzw"][max(i-1,0)]
        patch.set_xy(((pose[0]-half[0])*1000,(pose[2]-half[2])*1000))
        ax.set_title(f"MPM diagnostic slice | t={times[i]:.3f} s | no separation claim")
        fig.savefig(images/f"{j:04d}.png", dpi=100)
    plt.close(fig)
    r = subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","30","-i",str(images/"%04d.png"),
                        "-c:v","libx264","-pix_fmt","yuv420p",str(output/"preview.mp4")], capture_output=True,text=True)
    (output/"ffmpeg.log").write_text(r.stdout+r.stderr)
    with open(output/"trace.csv") as f:
        rows = list(csv.DictReader(f))
    fig, axes = plt.subplots(3,1,figsize=(8,7),sharex=True)
    t = [float(r["time_s"]) for r in rows]
    for k in ["blade_fx_N", "blade_fz_N"]:
        axes[0].plot(t,[float(r[k]) for r in rows],label=k)
    axes[0].legend(); axes[0].set_ylabel("force [N], debug")
    axes[1].plot(t,[float(r["max_z_m"])*1000 for r in rows]); axes[1].set_ylabel("max height [mm]")
    axes[2].plot(t,[float(r["max_speed_m_s"]) for r in rows]); axes[2].set_ylabel("max speed [m/s]")
    axes[2].set_xlabel("time [s]"); fig.tight_layout(); fig.savefig(output/"trace.png"); plt.close(fig)
