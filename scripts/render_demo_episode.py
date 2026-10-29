#!/usr/bin/env python3
"""Render a recorded live demo episode into demo.mp4, demo.gif and poster.png.

Read-only consumer of an episode directory written by scripts/run_wp10_live.py:
the recording is never modified. Writes a fresh output directory (refuses to
overwrite) with a render_receipt.json describing inputs, measured capture rate
and which panels were present. The particle panel is an oracle debug view
coloured by the actual MPM stencil mechanical components; it is not an
observation.
"""
import argparse
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robotic_cutting_sim import demo_media as dm  # noqa: E402
from robotic_cutting_sim.recording import write_json  # noqa: E402

FFMPEG = "/usr/bin/ffmpeg"
W, H, DPI = 1920, 1080, 100


def git_info():
    def run(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return r.stdout.strip()
    return dict(sha=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")),
                branch=run("rev-parse", "--abbrev-ref", "HEAD"))


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def deep(mapping, *keys, default=None):
    for key in keys:
        if not isinstance(mapping, dict) or key not in mapping:
            return default
        mapping = mapping[key]
    return mapping


def row_series(rows, control_dt):
    """Per-control-step diagnostics extracted from actions.jsonl (privileged)."""
    out = {k: [] for k in ("t", "force", "cut", "cut_index", "edge", "left", "right", "grasp")}
    for row in rows:
        pd = row.get("privileged_diagnostics") or {}
        out["t"].append(float(row["time_s"]))
        impulse = pd.get("current_material_reaction_impulse_world")
        out["force"].append(np.nan if impulse is None else dm.reaction_force_n(impulse, control_dt))
        out["cut"].append(bool(pd.get("active_cut", False)))
        out["cut_index"].append(int(pd.get("cut_index", -1)))
        pose = pd.get("measured_tool_pose", deep(row, "measured", "tool_pose"))
        out["edge"].append(np.full(7, np.nan) if pose is None else np.asarray(pose, float)[:7])
        roles = deep(pd, "grasp", "contacts", "roles", default={}) or {}
        out["left"].append(float(deep(roles, "left_finger", "force_magnitude_sum_n", default=np.nan)))
        out["right"].append(float(deep(roles, "right_finger", "force_magnitude_sum_n", default=np.nan)))
        grasp = row.get("grasp_status") or pd.get("grasp_status")
        if isinstance(grasp, dict):
            grasp = grasp.get("status")
        out["grasp"].append(str(grasp or deep(row, "info", "grasp_status", default="n/a")))
    return {k: (np.asarray(v) if k != "grasp" else v) for k, v in out.items()}


def load_frame(obs_dir, stream, index):
    path = obs_dir / f"{stream}_{index:04d}.npz"
    if stream is None or not path.is_file():
        return None
    with np.load(path) as data:
        return {k: data[k] for k in ("rgb", "depth_m", "valid_mask", "physics_timestamp_s") if k in data.files}


def style_axis(ax):
    ax.set_facecolor(dm.SURFACE)
    for spine in ax.spines.values():
        spine.set_color("#55544f")
    ax.tick_params(colors=dm.TEXT_SECONDARY, labelsize=8)
    ax.grid(True, color="#2e2e2c", linewidth=.6)


def image_axis(fig, rect, title):
    ax = fig.add_axes(rect)
    ax.set_facecolor("#000000")
    ax.set_xticks([]), ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#55544f")
    ax.set_title(title, color=dm.TEXT_PRIMARY, fontsize=11, loc="left", pad=3)
    return ax


def px(x0, y0, w, h):
    return [x0 / W, y0 / H, w / W, h / H]


def placeholder(ax, text):
    return ax.text(.5, .5, text, color=dm.TEXT_SECONDARY, ha="center", va="center",
                   fontsize=12, transform=ax.transAxes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default=None)
    parser.add_argument("--label", default=None, help="final evaluation status line (default: result.json)")
    parser.add_argument("--fps", type=float, default=None, help="video fps (default: measured capture rate)")
    parser.add_argument("--gif-width", type=int, default=640)
    parser.add_argument("--gif-speed", type=float, default=2.)
    parser.add_argument("--gif-max-mb", type=float, default=4.)
    parser.add_argument("--particle-stride", type=int, default=3)
    parser.add_argument("--poster-frame", type=int, default=-1)
    parser.add_argument("--keep-frames", action="store_true")
    args = parser.parse_args()
    start = time.perf_counter()
    episode = args.episode_dir.resolve()
    if not (episode / "frame_manifest.json").is_file():
        parser.error(f"not an episode directory: {episode}")
    out = args.out
    if out.exists():
        parser.error(f"refusing to overwrite existing output: {out}")
    out.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier render
    out = out.resolve()
    try:
        render(args, episode, out, start)
    except Exception:
        import traceback
        (out / "render_failure.txt").write_text(traceback.format_exc())
        raise


def render(args, episode, out, start):
    obs_dir = episode / "observations"
    manifest_rows = read_json(episode / "frame_manifest.json")
    result = read_json(episode / "result.json", {})
    episode_manifest = read_json(episode / "manifest.json", {})
    task = read_json(episode / "task.json", {})
    rows = dm.load_jsonl(episode / "actions.jsonl") if (episode / "actions.jsonl").is_file() else []
    physics = np.load(episode / "physics_state.npz")
    positions, codes = physics["positions"], physics["codes"]
    reference = physics["reference_positions"]
    n_frames = len(manifest_rows)
    if len(positions) != n_frames:
        raise ValueError(f"physics_state frames {len(positions)} != frame_manifest {n_frames}")
    frame_t = np.array([r["time_s"] for r in manifest_rows], float)

    runtime_source = result.get("runtime_source") or deep(episode_manifest, "config", "runtime_source") \
        or deep(result, "strict", "runtime_source")
    runtime = dm.runtime_display_name(runtime_source)
    controller = result.get("controller") or manifest_rows[0].get("controller", "unknown")
    task_id = result.get("task_id") or task.get("task_id")
    status, status_source = dm.status_line(args.label, result)
    control_dt = float(task.get("control_dt_s") or deep(episode_manifest, "config", "config", "control_dt")
                       or (np.median(np.diff([r["time_s"] for r in rows])) if len(rows) > 1 else .02))
    series = row_series(rows, control_dt) if rows else None
    half = dm.edge_half_length(task)

    streams = dm.discover_streams(p.name for p in obs_dir.glob("*.npz")) if obs_dir.is_dir() else {}
    panels = dm.choose_panels(streams)

    # Measured capture rate from actual camera timestamps (fallback: manifest times).
    timing_stream = panels["large"] or next(iter(streams), None)
    stamps = []
    if timing_stream:
        for i in range(n_frames):
            path = obs_dir / f"{timing_stream}_{i:04d}.npz"
            if path.is_file():
                with np.load(path) as data:
                    stamps.append(float(data["physics_timestamp_s"]) if "physics_timestamp_s" in data.files
                                  else np.nan)
    rate = dm.measured_rate(stamps) if stamps else dm.measured_rate(frame_t)
    rate["source"] = f"observations/{timing_stream} physics_timestamp_s" if stamps else "frame_manifest time_s"
    rate["manifest"] = dm.measured_rate(frame_t)
    fps = float(args.fps or round(rate["rate_hz"] or 5.))

    # Components (cached per unique code vector).
    cuts = dm.cut_count(codes)
    cache = dm.ComponentCache(physics["reference_stencil_nodes"], physics["reference_stencil_weights"],
                              physics["reference_stencil_gradients"] if "reference_stencil_gradients"
                              in physics.files else None, cuts)
    components, labelled = [], []
    for i in range(n_frames):
        components.append(cache(codes[i])[0])
        labelled.append(int(np.count_nonzero(codes[i])))
    stride = max(1, args.particle_stride)
    sub = np.arange(0, positions.shape[1], stride)

    # Fixed orthographic view of the particles along the Global D455 optical axes.
    view = None
    for i in range(n_frames):
        path = obs_dir / f"global_d455_{i:04d}.npz"
        if path.is_file():
            with np.load(path) as data:
                view = np.asarray(data["T_world_camera"], float)[:3, :3]
            break
    if view is None:
        from scipy.spatial.transform import Rotation
        view = Rotation.from_euler("zx", [-35, -125], degrees=True).as_matrix()
    project = lambda p: np.asarray(p) @ view  # columns: right, down, forward
    ref_q = project(reference)
    lo, hi = ref_q.min(0), ref_q.max(0)
    span = (hi - lo).max()
    center = (lo + hi) / 2

    # Depth colour range fixed over the episode.
    depth_samples = []
    for i in (0, n_frames // 2, n_frames - 1):
        frame = load_frame(obs_dir, panels["wrist_depth"], i) if panels["wrist_depth"] else None
        if frame and "depth_m" in frame:
            d = frame["depth_m"][frame.get("valid_mask", np.isfinite(frame["depth_m"]))]
            depth_samples.append(d[np.isfinite(d)])
    if depth_samples and np.concatenate(depth_samples).size:
        dmin, dmax = np.percentile(np.concatenate(depth_samples), [2, 98])
    else:
        dmin, dmax = 0., 1.

    # ---------------- figure ----------------
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=dm.SURFACE)
    title = args.title or f"Robotic cutting demo {task_id or ''}".strip()
    fig.text(12 / W, 1 - 28 / H, title, color=dm.TEXT_PRIMARY, fontsize=19, weight="bold", va="center")
    header = [fig.text(12 / W, 1 - 56 / H, "", color=dm.TEXT_PRIMARY, fontsize=13, va="center"),
              fig.text(12 / W, 1 - 80 / H, "", color=dm.TEXT_PRIMARY, fontsize=13, va="center",
                       family="DejaVu Sans Mono")]
    cut_badge = fig.text(1 - 12 / W, 1 - 30 / H, "", fontsize=17, weight="bold", ha="right", va="center",
                         bbox=dict(boxstyle="round,pad=0.35", fc="#3a3a37", ec="none"))

    names = dict(global_d455="Global D455 RGB", wrist_d405_0="Wrist D405_0 RGB", wrist_d405_1="Wrist D405_1 RGB")
    top, main_h = 1080 - 100, 640
    y0 = top - main_h
    image_artists, missing_text = {}, {}

    def image_panel(key, rect, label):
        ax = image_axis(fig, rect, label)
        stream = panels[key]
        if stream is None:
            placeholder(ax, "not recorded")
            return
        image_artists[key] = (ax, stream, None)
        missing_text[key] = ax.text(.5, .5, "", color="#ffd28a", ha="center", va="center", fontsize=13,
                                    transform=ax.transAxes, bbox=dict(fc="#1a1a19", ec="none", alpha=.8))

    large_name = panels["large"]
    image_panel("large", px(10, y0, 853, main_h - 22),
                (f"Overview RGB ({large_name})" if panels["overview"] else names.get(large_name, large_name or "large"))
                if large_name else "Overview / Global RGB")
    cw = 505
    ph = main_h // 2 - 26
    image_panel("wrist_rgb", px(880, y0 + main_h // 2 + 4, cw, ph), "Wrist D405_0 RGB")
    image_panel("wrist_depth", px(880, y0, cw, ph),
                f"Wrist D405_0 depth [m] ({dmin:.3f}-{dmax:.3f}, invalid = grey)")
    sec = panels["secondary"]
    image_panel("secondary", px(1400, y0 + main_h // 2 + 4, cw, ph), dm.panel_title(sec, names))

    # Particle panel.
    pax = fig.add_axes(px(1400, y0, cw, ph))
    style_axis(pax)
    pax.grid(False)
    pax.set_xticks([]), pax.set_yticks([])
    pax.set_title("Particles by mechanical component", color=dm.TEXT_PRIMARY, fontsize=11, loc="left", pad=3)
    pax.text(.01, .02, dm.ORACLE_LABEL, transform=pax.transAxes, color="#ffd28a", fontsize=9.5, va="bottom")
    # Limits matched to the panel's pixel aspect, so the orthographic scale is equal in x and y.
    half_x = .62 * span
    half_y = half_x * ph / cw
    pax.set_xlim(center[0] - half_x, center[0] + half_x)
    pax.set_ylim(-center[1] - half_y, -center[1] + half_y)
    scatter = pax.scatter(np.zeros(len(sub)), np.zeros(len(sub)), s=1.1, linewidths=0, rasterized=True)
    (edge_line,) = pax.plot([], [], color="#f0efec", linewidth=2.2, solid_capstyle="round")
    comp_text = pax.text(.99, .97, "", transform=pax.transAxes, color=dm.TEXT_PRIMARY, fontsize=9.5,
                         ha="right", va="top", family="DejaVu Sans Mono")
    pax.text(.99, .02, "white line: measured blade edge", transform=pax.transAxes,
             color=dm.TEXT_SECONDARY, fontsize=8.5, ha="right", va="bottom")

    # Diagnostics strip.
    strip_y, strip_h = 70, 220
    titles = ["Material reaction on blade |F| [N]", "Finger contact force [N]", "Edge height z [mm]",
              "Labelled particles (code != 0)", "Mechanical components"]
    gap, left = 22, 60
    width = (W - left - 20 - gap * 4 * 2.2) / 5
    axes = []
    for k, t in enumerate(titles):
        ax = fig.add_axes(px(left + k * (width + gap * 2.2), strip_y, width, strip_h))
        style_axis(ax)
        ax.set_title(t, color=dm.TEXT_PRIMARY, fontsize=11, loc="left", pad=4)
        ax.set_xlabel("sim time [s]", color=dm.TEXT_SECONDARY, fontsize=8.5, labelpad=1)
        ax.set_xlim(frame_t[0], max(frame_t[-1], series["t"][-1] if series else 0))
        axes.append(ax)
    c1, c2 = dm.COMPONENT_COLORS[0], dm.COMPONENT_COLORS[1]
    if series is not None:
        t = series["t"]
        cut = series["cut"]
        for ax in axes:
            edges = np.flatnonzero(np.diff(np.r_[0, cut.astype(int), 0]))
            for a, b in zip(edges[::2], edges[1::2]):
                ax.axvspan(t[a] - control_dt, t[b - 1], color="#4a4a2a", alpha=.55, linewidth=0, zorder=0)
        axes[0].plot(t, series["force"], color=c1, linewidth=1.6)
        axes[1].plot(t, series["left"], color=c1, linewidth=1.6, label="left finger")
        axes[1].plot(t, series["right"], color=c2, linewidth=1.6, label="right finger")
        axes[1].legend(loc="lower left", fontsize=8, facecolor=dm.SURFACE, edgecolor="#55544f",
                       labelcolor=dm.TEXT_PRIMARY)
        axes[2].plot(t, series["edge"][:, 2] * 1e3, color=c1, linewidth=1.6)
    else:
        for ax in axes[:3]:
            placeholder(ax, "no actions.jsonl")
    axes[3].step(frame_t, labelled, where="post", color=c1, linewidth=1.6)
    axes[4].step(frame_t, components, where="post", color=c1, linewidth=1.6)
    axes[4].set_ylim(0, max(components) + 1)
    axes[4].yaxis.get_major_locator().set_params(integer=True)
    grasp_text = axes[1].text(.98, .96, "", transform=axes[1].transAxes, ha="right", va="top",
                              fontsize=9.5, color=dm.TEXT_PRIMARY, family="DejaVu Sans Mono")
    force_text = axes[0].text(.98, .96, "", transform=axes[0].transAxes, ha="right", va="top",
                              fontsize=9.5, color=dm.TEXT_PRIMARY, family="DejaVu Sans Mono")
    cursors = [ax.axvline(0, color="#f0efec", linewidth=1.2, zorder=5) for ax in axes]
    fig.text(left / W, 18 / H, "Force: " + dm.FORCE_DEFINITION + ".  Shaded: active cut (privileged "
             "diagnostics).  All plots are privileged diagnostics, not observations.",
             color=dm.TEXT_SECONDARY, fontsize=9, va="center")

    row_index = dm.nearest_rows(series["t"], frame_t) if series is not None else np.full(n_frames, -1)
    frames_dir = out / ("frames" if args.keep_frames else ".frames_tmp")
    frames_dir.mkdir()
    missing = {}
    for i in range(n_frames):
        for key, (ax, stream, artist) in list(image_artists.items()):
            frame = load_frame(obs_dir, stream, i)
            if frame is None:
                missing.setdefault(stream, set()).add(i)
                missing_text[key].set_text(f"no {stream} frame {i:04d}\n(previous frame, dimmed)"
                                           if artist is not None else f"no {stream} frame {i:04d}")
                if artist is not None:
                    artist.set_alpha(.25)
                continue
            missing_text[key].set_text("")
            if artist is not None:
                artist.set_alpha(1.)
            if key == "wrist_depth":
                depth = np.ma.masked_where(~frame["valid_mask"] | ~np.isfinite(frame["depth_m"]), frame["depth_m"])
                if artist is None:
                    cmap = plt.get_cmap("viridis").with_extremes(bad="#5a5a58")
                    artist = ax.imshow(depth, cmap=cmap, vmin=dmin, vmax=dmax, interpolation="nearest")
                else:
                    artist.set_data(depth)
            else:
                if artist is None:
                    artist = ax.imshow(frame["rgb"], interpolation="bilinear")
                else:
                    artist.set_data(frame["rgb"])
            image_artists[key] = (ax, stream, artist)

        count, labels = cache(codes[i])
        colors, slot, sizes = dm.component_colors(labels, reference)
        q = project(positions[i][sub])
        order = np.argsort(-q[:, 2], kind="stable")
        scatter.set_offsets(np.c_[q[order, 0], -q[order, 1]])
        scatter.set_facecolors(colors[sub][order])
        major = int((slot >= 0).sum())
        comp_text.set_text(f"components {count}  (major {major}, minor {count - major})\n"
                           f"labelled {labelled[i]:,} / {len(codes[i]):,}")
        r = row_index[i]
        if r >= 0 and half is not None and np.isfinite(series["edge"][r]).all():
            seg = project(dm.edge_segment(series["edge"][r], half))
            edge_line.set_data(seg[:, 0], -seg[:, 1])
        else:
            edge_line.set_data([], [])
        active = bool(series["cut"][r]) if r >= 0 else False
        cut_index = int(series["cut_index"][r]) if r >= 0 else -1
        lines = dm.frame_annotation(runtime=runtime, controller=controller, time_s=frame_t[i],
                                    active_cut=active, cut_index=cut_index, status=status, task_id=task_id)
        header[0].set_text(lines[0] + f"  |  frame {i + 1}/{n_frames} @ {rate['rate_hz'] or 0:.1f} Hz capture")
        header[1].set_text(lines[1])
        cut_badge.set_text(f"CUT {'ON' if active else 'OFF'}  idx {cut_index}")
        cut_badge.set_color("#1a1a19" if active else dm.TEXT_PRIMARY)
        cut_badge.get_bbox_patch().set_facecolor("#eda100" if active else "#3a3a37")
        if r >= 0:
            grasp_text.set_text(f"grasp: {series['grasp'][r]}")
            force_text.set_text(f"{series['force'][r]:.2f} N")
        for c in cursors:
            c.set_xdata([frame_t[i], frame_t[i]])
        fig.canvas.draw()
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        plt.imsave(frames_dir / f"{i:05d}.png", rgb)
    plt.close(fig)

    # ---------------- encode ----------------
    mp4 = out / "demo.mp4"
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-n", "-framerate", f"{fps}",
                    "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-r", f"{fps}", str(mp4)], check=True)
    gif = out / "demo.gif"
    gif_record = None
    in_rate = rate["rate_hz"] or fps
    for attempt, plan in enumerate(dm.gif_plans(n_frames, in_rate * args.gif_speed, args.gif_width)):
        candidate = out / f".demo_try{attempt}.gif"
        filters = (f"fps={plan['fps']:.4f},scale={plan['width']}:-2:flags=lanczos,split[a][b];"
                   "[a]palettegen=max_colors=128:stats_mode=diff[p];"
                   "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-n", "-framerate",
                        f"{in_rate * args.gif_speed}", "-i", str(frames_dir / "%05d.png"),
                        "-filter_complex", filters, "-loop", "0", str(candidate)], check=True)
        size = candidate.stat().st_size
        gif_record = dict(plan, speed=args.gif_speed, bytes=size, attempt=attempt,
                          budget_bytes=int(args.gif_max_mb * 1e6), within_budget=size <= args.gif_max_mb * 1e6)
        if gif_record["within_budget"]:
            candidate.rename(gif)
            break
        candidate.unlink()
    else:
        gif_record["note"] = "no plan met the size budget; smallest attempt kept"
        subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-n", "-framerate",
                        f"{in_rate * args.gif_speed}", "-i", str(frames_dir / "%05d.png"),
                        "-filter_complex", filters, "-loop", "0", str(gif)], check=True)
    poster_index = args.poster_frame % n_frames
    shutil.copyfile(frames_dir / f"{poster_index:05d}.png", out / "poster.png")
    if not args.keep_frames:
        shutil.rmtree(frames_dir)

    ffmpeg_version = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    inputs = {name: dict(path=str(episode / name), sha256=dm.sha256_file(episode / name))
              for name in ("result.json", "physics_state.npz", "frame_manifest.json", "actions.jsonl",
                           "manifest.json", "task.json") if (episode / name).is_file()}
    outputs = {p.name: dict(bytes=p.stat().st_size, sha256=dm.sha256_file(p))
               for p in (mp4, gif, out / "poster.png")}
    panel_record = dict(
        assignment=panels, overview_present=panels["overview"] is not None,
        **({"closeup_present": True, "secondary_title": dm.CLOSEUP_TITLE} if panels.get("closeup") else {}),
        streams={k: len(v) for k, v in streams.items()},
        missing_frames={k: sorted(v) for k, v in missing.items()},
        particle_debug=True, diagnostics_strip=series is not None,
        layout=dm.panel_layout(panels),
        poster_frame=poster_index, edge_half_length_m=half)
    receipt = dm.render_receipt(
        episode=episode, out=out, inputs=inputs, frames=n_frames, rate=rate, video_fps=fps, gif=gif_record,
        panels=panel_record, components=components, cuts=cuts, particle_stride=stride, status=status,
        status_source=status_source, git=git_info(), wall_s=time.perf_counter() - start, command=sys.argv,
        outputs=outputs, label_source_runtime=dict(runtime_source=runtime_source, display=runtime,
                                                   controller=controller, task_id=task_id),
        versions=dict(python=platform.python_version(), numpy=np.__version__,
                      matplotlib=matplotlib.__version__, ffmpeg=ffmpeg_version))
    receipt["component_cache"] = dict(unique_code_vectors=len(cache.cache), calls=cache.calls)
    receipt["labelled_particles_per_frame"] = labelled
    write_json(out / "render_receipt.json", receipt)
    print(json.dumps(dict(out=str(out), frames=n_frames, rate_hz=rate["rate_hz"], wall_s=receipt["wall_time_s"],
                          outputs={k: v["bytes"] for k, v in outputs.items()}, gif=gif_record)))


if __name__ == "__main__":
    main()
