#!/usr/bin/env python3
"""Validate measured dynamic topology staging on CPU; this is not an RTX test."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from pxr import Usd, UsdGeom

from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.usd_surface_bridge import DynamicSurfaceBridge


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("surface_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = new_output(args.output_dir)
    paths = sorted(args.surface_dir.glob("*.npz"))
    if len(paths) < 2:
        raise ValueError("Need at least two measured surface snapshots")
    stage = Usd.Stage.CreateNew(str(output / "dynamic_surface.usda"))
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetTimeCodesPerSecond(60)
    stage.SetInterpolationType(Usd.InterpolationTypeHeld)
    bridge = DynamicSurfaceBridge(stage)
    rows = []
    started = time.perf_counter()
    for path in paths:
        data = np.load(path)
        timestamp = float(data["physics_timestamp_s"])
        tick = time.perf_counter()
        receipt = bridge.update(data["vertices_m"], data["triangles"], timestamp, time_code=timestamp * 60)
        receipt.update(source=str(path.resolve()), stage_wall_s=time.perf_counter() - tick)
        rows.append(receipt)
    stage.SetStartTimeCode(rows[0]["physics_timestamp_s"] * 60)
    stage.SetEndTimeCode(rows[-1]["physics_timestamp_s"] * 60)
    stage.GetRootLayer().Save()
    reopened = Usd.Stage.Open(str(output / "dynamic_surface.usda"))
    mesh = UsdGeom.Mesh.Get(reopened, "/World/ContinuousSpecimen")
    for row, path in zip(rows, paths):
        source = np.load(path)
        tc = row["physics_timestamp_s"] * 60
        assert np.array_equal(np.asarray(mesh.GetPointsAttr().Get(tc)), source["vertices_m"])
        assert np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get(tc)).reshape(-1, 3), source["triangles"])
        assert mesh.GetPrim().GetAttribute("research:surfaceTimestampS").Get(tc) == row["surface_timestamp_s"]
    result = {"status": "PASS", "scope": "CPU OpenUSD measured dynamic topology and timestamp roundtrip",
              "frames": rows, "wall_s": time.perf_counter() - started,
              "unique_topology_hashes": len({row["topology_sha256"] for row in rows}),
              "rtx_capture": "NOT_RUN", "physical_cutting": "NOT_ASSESSED"}
    write_json(output / "manifest.json", manifest({"surface_dir": str(args.surface_dir.resolve())}, "cpu_usd_surface_bridge"))
    write_json(output / "result.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=2))


if __name__ == "__main__":
    main()
