#!/usr/bin/env python3
"""Render a CPU geometry preview of the composed USD, without Kit or RTX.

Usage: .venv/bin/python scripts/preview_workcell.py WORKCELL.usda OUTPUT.png
This diagnostic uses the referenced mesh geometry with simplified flat colors.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pxr import Usd, UsdGeom


def main():
    source, output = map(Path, sys.argv[1:])
    if output.exists() or output.with_suffix(".json").exists():
        raise SystemExit("Refusing to overwrite an existing preview")
    stage = Usd.Stage.Open(str(source))
    xforms = UsdGeom.XformCache()
    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    count, all_polygons, all_colors = 0, [], []
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        if UsdGeom.Imageable(prim) and UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            if not len(vertices):
                continue
            sizes = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            faces, offset = [], 0
            for index, size in enumerate(sizes):
                polygon = indices[offset:offset + size]
                faces.extend((polygon[0], polygon[i], polygon[i + 1]) for i in range(1, size - 1))
                offset += size
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) / 2
            vertices = half * np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                       [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]])
            quads = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4),
                     (2, 3, 7, 6), (1, 2, 6, 5), (3, 0, 4, 7)]
            faces = [(a, b, c) for a, b, c, d in quads] + [(a, c, d) for a, b, c, d in quads]
        else:
            continue
        transform = np.array(xforms.GetLocalToWorldTransform(prim)).T
        vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
        name = str(prim.GetPath())
        if "ContinuousSpecimen" in name:
            color = (.75, .34, .15)
        elif "Tray" in name:
            color = (.38, .43, .48)
        elif "Table" in name:
            color = (.20, .24, .30)
        elif "KinematicBlade" in name:
            color = (.45, .68, .82)
        elif "D455" in name:
            color = (.22, .29, .36)
        elif "panda_hand" in name or "finger" in name:
            color = (.16, .18, .21)
        else:
            color = (.85, .86, .88)
        polygons = [vertices[list(face)] for face in faces]
        all_polygons.extend(polygons)
        all_colors.extend([color] * len(polygons))
        count += 1
    # One collection lets the software renderer sort faces across all objects.
    axis.add_collection3d(Poly3DCollection(all_polygons, facecolors=all_colors, linewidths=0,
                                         rasterized=True, shade=True))
    axis.set(xlim=(-.95, .95), ylim=(-.65, .65), zlim=(-.08, 1.15),
             xlabel="world x [m]", ylabel="world y [m]", zlabel="world z [m]")
    axis.set_box_aspect((1.9, 1.3, 1.23))
    axis.view_init(elev=24, azim=-58)
    axis.set_title("RoboticCuttingSim | static research workcell\n"
                   "Official Panda ×2 / D455 ×2 · simplified CPU geometry preview")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160)
    plt.close(figure)
    receipt = {"status": "PASS", "scope": "CPU geometry preview; complete visible mesh geometry with simplified colors",
               "source": str(source.resolve()), "drawn_mesh_or_cube_prims": count,
               "drawn_triangles": len(all_polygons),
               "rtx_validated": False, "robot_control_validated": False}
    output.with_suffix(".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
