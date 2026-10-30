"""Multi-layer cut debug viewer (WP-12 Phase 1): layers, views, PNG panels and PLY exports (read-only, debug only).

Uses privileged material state (codes, components, Hermite data, contact work). It is a developer tool and must
never be wired into a policy observation.

Package formats read by ``load``: ``wp12_cut_debug_package_v1`` (written by scripts/run_wp12_phase1_geometry.py)
and the Phase 0 ``baseline_compact.npz``. Coordinates are solver-local, metres.

Layers (``LAYERS``):
  original          reference (t = 0) particle positions
  current           current particle positions
  labels            current particles coloured by the digit of one cut (0 unassigned grey, 1 blue, 2 orange)
  components        current particles coloured by mechanical component
  sweep             blade: edge line at the recorded poses of the selected digit and the blade cross-section
                    polygon (tool profile) swept along them
  contact           particles with solver cutting progress > 0 or contact work > 0
  particle_surface  ParticleSurface mesh (``--profile`` legacy / current_volume_v1)
  cut_aware         pluggable (``CUT_AWARE_PROVIDERS``); Phase 1 default ``hermite_side_points_v0`` = the final side
                    locus of the recorded Hermite points (a point cloud). Phase 3 registers its surface here.
  rendered          the surface the live path renders: current_volume_v1 for every configured/registered task
                    (isaac_sim_runner / kinematic_env defaults); legacy only for the legacy D0 path
Views (``VIEWS``): top (x-y), side (x-z), blade (blade plane coordinates of the selected digit), section
(arbitrary plane point + normal, slab thickness), closeup (3-D window around the cut face).
"""
import struct
from pathlib import Path

import numpy as np

LAYERS = ("original", "current", "labels", "components", "sweep", "contact", "particle_surface", "cut_aware", "rendered")
VIEWS = ("top", "side", "blade", "section", "closeup")
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
DIGIT = {0: "#b8b8b4", 1: "#2a78d6", 2: "#eb6834"}
RENDERED_PROFILE = "current_volume_v1"


def _hex(h):
    return np.array([int(h[i:i+2], 16)/255 for i in (1, 3, 5)])


def hermite_side_points_v0(pkg, digit=None):
    """Phase 1 cut-aware placeholder: final side locus points of the recorded Hermite data (point cloud)."""
    sp = pkg.get("side_points")
    if sp is None or not len(sp):
        return {"kind": "points", "points": np.zeros((0, 3)), "colors": np.zeros((0, 3))}
    cut = pkg["hermite_cut"]
    sel = np.ones(len(cut), bool) if digit is None else cut == digit
    pts = sp[sel].reshape(-1, 3)
    col = np.tile([_hex("#0b0b0b"), _hex("#6250d6")], (int(sel.sum()), 1))
    return {"kind": "points", "points": pts, "colors": col,
            "label": "cut-aware hook: Hermite side locus (recorded)" if pkg.get("hermite_recorded", False)
            else "cut-aware hook: link-midpoint side locus (fallback)"}


CUT_AWARE_PROVIDERS = {"hermite_side_points_v0": hermite_side_points_v0}


def load(path):
    """Standardised dict from a debug package or a Phase 0 baseline_compact.npz."""
    z = np.load(path, allow_pickle=False)
    f = set(z.files)
    pkg = {"path": str(path)}
    if "schema" in f:
        for k in z.files:
            pkg[k] = z[k]
        pkg["meshes"] = {p: (z[f"mesh_{p}_vertices"], z[f"mesh_{p}_faces"]) for p in ("legacy", "current_volume_v1")
                         if f"mesh_{p}_vertices" in f}
        pkg["hermite_recorded"] = bool(np.all(z["hermite_is_recorded"])) if "hermite_is_recorded" in f and len(z["hermite_is_recorded"]) else False
        pkg["spacing_m"] = float(z["spacing_m"])
        pkg["cuts"] = int(z["cuts"])
    else:   # Phase 0 baseline package
        from .surface_metrics_v1 import material_locus
        from .cut_topology import ternary_to_packed
        codes = z["final_codes"]
        cuts = int(z["max_cuts"])
        if str(z["code_encoding"]) != "packed2":
            codes = ternary_to_packed(codes, cuts)
        pkg.update(reference=z["reference_positions_local"], positions=z["final_positions_local"], codes=codes,
                   labels=z["final_component_labels"], lattice=z["lattice"], cuts=cuts, spacing_m=.001,
                   cut_progress=z["final_cut_progress"], contact_work=np.zeros(len(codes), np.float32),
                   meshes={"current_volume_v1": (z["final_surface_vertices_local"], z["final_surface_faces"])},
                   poses=z["blade_pose_end_local"], pose_cuts=z["step_cut"].astype(np.int8),
                   blade_polygon_xz=np.zeros((0, 2), np.float32), blade_half_length_m=.06, hermite_recorded=False)
        loc = material_locus(pkg["reference"], pkg["lattice"], pkg["positions"], codes, cuts=cuts, F=z["final_F"])
        pkg.update(side_points=loc["side_points"], hermite_cut=loc["cut"], hermite_P=loc["P"])
    return pkg


def digits(pkg, k):
    return (np.asarray(pkg["codes"], np.int64) >> (2*int(k))) & 3


def _rotate(quats, v):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(quats).apply(v)


def blade_frame(pkg, digit):
    """(edge direction, blade normal, advance) of the selected digit's recorded poses (median), or None."""
    poses, cuts = pkg.get("poses"), pkg.get("pose_cuts")
    if poses is None or not len(poses):
        return None
    sel = poses[cuts == digit] if np.any(cuts == digit) else poses
    q = sel[len(sel)//2, 3:]
    e = _rotate(q, [0, 1, 0])
    n = _rotate(q, [0, 0, 1])
    if len(sel) > 1:
        motion = sel[-1, :3]-sel[0, :3]
    else:
        motion = _rotate(q, [-1, 0, 0])
    return e, n, motion/max(np.linalg.norm(motion), 1e-12), sel


def layer(pkg, name, *, digit=0, profile=RENDERED_PROFILE, provider="hermite_side_points_v0"):
    """Geometry of one layer: {'kind': 'points'|'mesh'|'lines'|'polygons', ...}."""
    if name == "original":
        return {"kind": "points", "points": pkg["reference"], "colors": np.tile(_hex("#9a9a96"), (len(pkg["reference"]), 1))}
    if name == "current":
        return {"kind": "points", "points": pkg["positions"], "colors": np.tile(_hex("#52524e"), (len(pkg["positions"]), 1))}
    if name == "labels":
        d = digits(pkg, digit)
        return {"kind": "points", "points": pkg["positions"], "colors": np.array([_hex(DIGIT[int(v)]) for v in range(3)])[d]}
    if name == "components":
        lab = np.asarray(pkg["labels"])
        pal = np.array([_hex(c) for c in PALETTE]+[_hex("#9a9a96")])
        return {"kind": "points", "points": pkg["positions"], "colors": pal[np.minimum(lab, len(PALETTE))]}
    if name == "contact":
        m = (np.asarray(pkg.get("cut_progress", 0)) > 0) | (np.asarray(pkg.get("contact_work", 0)) > 0)
        return {"kind": "points", "points": pkg["positions"][m], "colors": np.tile(_hex("#e34948"), (int(m.sum()), 1)),
                "label": "solver cutting progress > 0 or contact work > 0"}
    if name in ("particle_surface", "rendered"):
        p = RENDERED_PROFILE if name == "rendered" else profile
        if p not in pkg["meshes"]:
            return {"kind": "mesh", "vertices": np.zeros((0, 3)), "faces": np.zeros((0, 3), int), "label": f"{p} missing"}
        v, f = pkg["meshes"][p]
        return {"kind": "mesh", "vertices": np.asarray(v, float), "faces": np.asarray(f, np.int64), "label": p}
    if name == "cut_aware":
        return CUT_AWARE_PROVIDERS[provider](pkg, digit)
    if name == "sweep":
        frame = blade_frame(pkg, digit)
        if frame is None:
            return {"kind": "lines", "segments": np.zeros((0, 2, 3))}
        e, n, adv, sel = frame
        L = float(pkg.get("blade_half_length_m", .06))
        step = max(1, len(sel)//60)
        P = sel[::step]
        E = _rotate(P[:, 3:], [0, 1, 0])
        seg = np.stack([P[:, :3]-L*E, P[:, :3]+L*E], 1)
        poly = np.asarray(pkg.get("blade_polygon_xz", np.zeros((0, 2))), float)
        polys = []
        if len(poly):
            for pose in P[::max(1, len(P)//12)]:
                local = np.c_[poly[:, 0], np.zeros(len(poly)), poly[:, 1]]
                polys.append(_rotate(np.tile(pose[3:], (len(poly), 1)), local)+pose[:3])
        return {"kind": "lines", "segments": seg, "polygons": polys, "label": f"edge sweep of digit {digit}"}
    raise ValueError(f"unknown layer {name}; choose from {LAYERS}")


def view_basis(view, pkg, digit=0, plane_point=None, plane_normal=None):
    """(origin, u, v, w): projection axes u, v and the viewing/slab normal w."""
    centre = np.median(pkg["positions"], 0)
    if view == "top":
        return centre, np.array([1., 0, 0]), np.array([0., 1, 0]), np.array([0., 0, 1])
    if view == "side":
        return centre, np.array([1., 0, 0]), np.array([0., 0, 1]), np.array([0., -1, 0])
    if view in ("blade", "section", "closeup"):
        frame = blade_frame(pkg, digit)
        sp = pkg.get("side_points")
        sel = None if sp is None or not len(sp) else sp[pkg["hermite_cut"] == digit].reshape(-1, 3)
        origin = np.median(sel, 0) if sel is not None and len(sel) else centre
        if plane_point is not None:
            origin = np.asarray(plane_point, float)
        if view == "blade":
            e, n, adv, _ = frame if frame is not None else (np.array([0., 1, 0]), np.array([1., 0, 0]), np.array([0, 0, -1.]), None)
            u = e
            v = np.cross(n, e)
            return origin, u, v/np.linalg.norm(v), n
        # section: plane normal w (default: the edge direction, so the cut face appears as a line)
        if plane_normal is not None:
            w = np.asarray(plane_normal, float)
        elif frame is not None:
            w = frame[0]
        else:
            w = np.array([0., 1, 0])
        w = w/np.linalg.norm(w)
        a = np.array([0., 0, 1]) if abs(w[2]) < .9 else np.array([1., 0, 0])
        u = np.cross(w, a)
        u /= np.linalg.norm(u)
        v = np.cross(u, w)
        if v[2] < 0:            # keep "up" up where the plane allows it
            u, v = -u, -v
        return origin, u, v, w
    raise ValueError(f"unknown view {view}; choose from {VIEWS}")


def section_segments(vertices, faces, origin, w):
    """Line segments where triangles cross the plane (origin, w)."""
    if not len(faces):
        return np.zeros((0, 2, 3))
    d = (np.asarray(vertices, float)-origin)@w
    df = d[faces]
    cross = (df.min(1) < 0) & (df.max(1) > 0)
    tri = np.asarray(vertices, float)[faces[cross]]
    dd = df[cross]
    segs = []
    for (i, j) in ((0, 1), (1, 2), (2, 0)):
        m = (dd[:, i] < 0) != (dd[:, j] < 0)
        t = np.where(m, dd[:, i]/np.where(m, dd[:, i]-dd[:, j], 1), 0)
        segs.append((m, tri[:, i]+t[:, None]*(tri[:, j]-tri[:, i])))
    pts = np.full((len(tri), 2, 3), np.nan)
    count = np.zeros(len(tri), int)
    for m, p in segs:
        for r in np.flatnonzero(m):
            if count[r] < 2:
                pts[r, count[r]] = p[r]
                count[r] += 1
    return pts[count == 2]


def render(pkg, out_png, *, layers=("components", "particle_surface", "cut_aware", "sweep"), views=VIEWS, digit=0,
           profile=RENDERED_PROFILE, plane_point=None, plane_normal=None, slab_m=None, window_m=.008, title=None,
           max_points=40000, dpi=90, seed=0):
    """One PNG with one panel per view; layers are drawn in the given order."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    s = float(pkg.get("spacing_m", .001))
    slab_m = 1.*s if slab_m is None else slab_m   # >= one lattice layer whatever the plane phase
    rng = np.random.default_rng(seed)
    geo = {name: layer(pkg, name, digit=digit, profile=profile) for name in layers}
    cols = min(3, len(views))
    rows = int(np.ceil(len(views)/cols))
    fig = plt.figure(figsize=(5.2*cols, 4.4*rows), dpi=dpi, facecolor="#fcfcfb")
    for i, view in enumerate(views):
        origin, u, v, w = view_basis(view, pkg, digit, plane_point, plane_normal)
        if view == "closeup":
            ax = fig.add_subplot(rows, cols, i+1, projection="3d")
            lo, hi = origin-window_m, origin+window_m
            for name, g in geo.items():
                if g["kind"] == "mesh" and len(g["faces"]):
                    c = g["vertices"][g["faces"]].mean(1)
                    keep = np.all(np.abs(c-origin) <= window_m, 1)
                    f = g["faces"][keep]
                    if len(f) > 25000:
                        f = f[rng.choice(len(f), 25000, replace=False)]
                    tri = g["vertices"][f]*1e3
                    n = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
                    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-30)
                    shade = .35+.65*np.abs(n@np.array([.35, -.45, .82])/np.linalg.norm([.35, -.45, .82]))
                    ax.add_collection3d(Poly3DCollection(tri, facecolors=np.c_[np.tile(shade[:, None], 3)*.85, np.ones(len(tri))*.9],
                                                         edgecolors="none"))
                elif g["kind"] == "points" and len(g["points"]):
                    keep = np.all(np.abs(g["points"]-origin) <= window_m, 1)
                    p = g["points"][keep]*1e3
                    ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=2, c=g["colors"][keep], depthshade=False, linewidths=0)
            ax.set_xlim(lo[0]*1e3, hi[0]*1e3); ax.set_ylim(lo[1]*1e3, hi[1]*1e3); ax.set_zlim(lo[2]*1e3, hi[2]*1e3)
            ax.set_box_aspect((1, 1, 1))
            nn = w if abs(w[2]) < .99 else np.array([1., 0, 0])
            ax.view_init(elev=25, azim=float(np.degrees(np.arctan2(nn[1], nn[0])))+55)
            ax.tick_params(labelsize=6)
            ax.set_title(f"close-up {2*window_m*1e3:.0f} mm window (mm)", fontsize=8)
            continue
        ax = fig.add_subplot(rows, cols, i+1)
        slab = view in ("section", "blade") or view == "section"
        for name, g in geo.items():
            if g["kind"] == "points" and len(g["points"]):
                p = g["points"]
                keep = np.ones(len(p), bool)
                if view == "section":
                    keep = np.abs((p-origin)@w) <= slab_m/2
                elif view == "blade":
                    keep = np.abs((p-origin)@w) <= 4*s
                idx = np.flatnonzero(keep)
                if len(idx) > max_points:
                    idx = rng.choice(idx, max_points, replace=False)
                q = (p[idx]-origin)
                ax.scatter(q@u*1e3, q@v*1e3, s=6 if view == "section" else 1.2, c=g["colors"][idx], linewidths=0,
                           marker="x" if name == "cut_aware" else "o", label=name)
            elif g["kind"] == "mesh" and len(g["faces"]):
                if view == "section":
                    seg = section_segments(g["vertices"], g["faces"], origin, w)
                    q = seg-origin
                    ax.add_collection(LineCollection(np.stack([q@u, q@v], -1)*1e3, colors="#0b0b0b", linewidths=.9,
                                                     label=f"{g.get('label', name)} section"))
                else:
                    c = g["vertices"][g["faces"]].mean(1)
                    if len(c) > max_points:
                        c = c[rng.choice(len(c), max_points, replace=False)]
                    q = c-origin
                    depth = q@w
                    ax.scatter(q@u*1e3, q@v*1e3, s=.6, c=depth, cmap="Greys", linewidths=0, alpha=.5)
            elif g["kind"] == "lines" and len(g.get("segments", [])):
                q = g["segments"]-origin
                if view == "section":
                    for poly in g.get("polygons", []):
                        qq = poly-origin
                        ax.add_collection(PolyCollection([np.c_[qq@u, qq@v]*1e3], facecolors="#eda10022",
                                                         edgecolors="#eda100", linewidths=.4))
                else:
                    ax.add_collection(LineCollection(np.stack([q@u, q@v], -1)*1e3, colors="#eda100", linewidths=.3, alpha=.6))
        ax.set_aspect("equal")
        ax.autoscale_view()
        if view in ("section", "blade"):
            ax.set_xlim(-window_m*1.5e3, window_m*1.5e3); ax.set_ylim(-window_m*1.5e3, window_m*1.5e3)
        ax.tick_params(labelsize=6)
        names = {"top": "top (x-y)", "side": "side (x-z)", "blade": "blade plane (edge, advance)",
                 "section": f"section, slab {slab_m*1e3:.1f} mm, normal {np.round(w, 2).tolist()}"}
        ax.set_title(f"{names[view]} (mm)", fontsize=8)
    handles = []
    import matplotlib.lines as mlines
    legend = {"labels": [(DIGIT[0], "digit 0"), (DIGIT[1], "digit 1"), (DIGIT[2], "digit 2")],
              "components": [(PALETTE[i], f"component {i}") for i in range(min(int(np.max(pkg['labels']))+1, 6))],
              "cut_aware": [("#0b0b0b", "cut-aware hook (side 1)"), ("#6250d6", "cut-aware hook (side 2)")],
              "contact": [("#e34948", "contact / cutting work")], "sweep": [("#eda100", "blade sweep")],
              "particle_surface": [("#0b0b0b", f"ParticleSurface {profile}")], "rendered": [("#0b0b0b", "rendered surface")],
              "original": [("#9a9a96", "original particles")], "current": [("#52524e", "current particles")]}
    for name in layers:
        for c, lab in legend.get(name, []):
            handles.append(mlines.Line2D([], [], color=c, marker="o", ls="", markersize=4, label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 7), fontsize=7, frameon=False)
    fig.suptitle(title or Path(pkg["path"]).name, fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .97))
    fig.savefig(out_png)
    plt.close(fig)
    return out_png


def write_ply_points(path, points, colors=None):
    points = np.asarray(points, np.float32)
    colors = np.zeros((len(points), 3), np.uint8) if colors is None else (np.clip(np.asarray(colors), 0, 1)*255).astype(np.uint8)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(points)}\nproperty float x\nproperty float y\n"
              "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    rec = np.zeros(len(points), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    rec["p"], rec["c"] = points, colors
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(rec.tobytes())


def write_ply_mesh(path, vertices, faces):
    vertices = np.asarray(vertices, np.float32)
    faces = np.asarray(faces, np.int32)
    header = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(vertices)}\nproperty float x\nproperty float y\n"
              f"property float z\nelement face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
    rec = np.zeros(len(faces), dtype=[("n", "u1"), ("i", "<i4", 3)])
    rec["n"], rec["i"] = 3, faces
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(vertices.astype("<f4").tobytes())
        f.write(rec.tobytes())


def read_ply_header(path):
    with open(path, "rb") as f:
        lines = []
        while True:
            line = f.readline().decode().strip()
            lines.append(line)
            if line == "end_header":
                break
    return lines


def export_ply(pkg, out_dir, *, layers=LAYERS, digit=0, profile=RENDERED_PROFILE):
    """One binary PLY per layer (points with colours, meshes, sweep as an edge point cloud)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = {}
    for name in layers:
        g = layer(pkg, name, digit=digit, profile=profile)
        path = out_dir/f"{name}.ply"
        if g["kind"] == "points":
            write_ply_points(path, g["points"], g["colors"])
        elif g["kind"] == "mesh":
            write_ply_mesh(path, g["vertices"], g["faces"])
        elif g["kind"] == "lines":
            seg = g["segments"]
            t = np.linspace(0, 1, 25)
            pts = (seg[:, :1]*(1-t[None, :, None])+seg[:, 1:]*t[None, :, None]).reshape(-1, 3) if len(seg) else np.zeros((0, 3))
            write_ply_points(path, pts, np.tile(_hex("#eda100"), (len(pts), 1)))
        written[name] = str(path)
    return written
