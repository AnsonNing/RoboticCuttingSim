"""P6.5 Native rubber run (Kit; plan docs/reports/rl_ready/P2_REMAINING_PLAN.md §4.6, protocol RL-09).

Runs ``native_rubber_host`` in Isaac Sim 6.2: the static D0 workcell, Frankas held at the shared safe park, a
kinematic PhysX clamp, the P2-6 rubber material (``RubberStripTaskFamily.build_life``: same recipe, slot, profile and
contact law) advanced on ``cuda:<gpu>`` in the Kit process with the clamp pose PhysX measures, the strip surface through
``DynamicSurfaceBridge`` and the overview / close-up RGB-D every 10 steps. ``--gpu`` is the render (``active_gpu``),
PhysX (``physics_gpu``; PhysX itself runs on the CPU) and material device. Use the launcher
``launch_native_rubber.py`` (memory gate, nvidia-smi readings); it calls this script.

    .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/run_native_rubber.py --out artifacts/rl_ready/p2_6/<new> \
        --gpu 0 --task-config <id> --task-config-sha256 <sha> --protocol configs/rubber/<protocol>.json \
        --protocol-sha256 <sha> [--steps N --label smoke]

Outputs (new directory, never overwritten): meta.json, result.json, receipt.json (compared by RL-09 a), scene.json,
native_arrays.npz (material inputs + statistics), bridge.jsonl (surface integrity per update), rubber_records.jsonl.gz
(rcs_rubber_record_v1), captures/ (RGB png, depth npz, camera consistency), kit.log. A zero exit code is not success.
"""
import argparse
import gzip
import json
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_meta as NM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--task-config", required=True, help="task id or path (RubberStripTaskFamily config)")
    ap.add_argument("--task-config-sha256", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    ap.add_argument("--steps", type=int, default=None, help="default: the declared 100 (anything else is a smoke)")
    ap.add_argument("--label", choices=("formal", "smoke"), default="formal")
    ap.add_argument("--friction-table", type=float, default=None, help="smoke only: override the build friction")
    a = ap.parse_args()
    from robotic_cutting_sim import native_rubber_host as H
    out = Path(a.out)
    out = out if out.is_absolute() else ROOT/out
    out.mkdir(parents=True)                       # FileExistsError: never overwrite
    protocol = Path(a.protocol) if Path(a.protocol).is_absolute() else ROOT/a.protocol
    if NM.sha256(protocol) != a.protocol_sha256:
        raise SystemExit(f"protocol {protocol} sha256 {NM.sha256(protocol)} != {a.protocol_sha256}")
    episode = dict(H.EPISODE)
    if a.steps is not None:
        episode["steps"] = int(a.steps)
    if a.label == "formal" and (episode["steps"] != H.EPISODE["steps"] or a.friction_table is not None):
        raise SystemExit("a formal run uses the declared episode (100 steps, the family's build material)")
    episode.update(protocol={"path": str(protocol.relative_to(ROOT)), "sha256": a.protocol_sha256})
    result = {"schema": H.RECORD_SCHEMA, "kind": "native_rubber_run", "label": a.label, "status": "STARTED",
              "episode": episode, "labels": H.LABELS}
    sampler = NM.SmiSampler(a.gpu, period_s=5.0).start()
    walls = {"process_start": time.time()}
    app = None
    try:
        from robotic_cutting_sim.isaac_sim_runner import launch_app, runtime_versions
        t0 = time.perf_counter()
        app = launch_app(SimpleNamespace(gpu=a.gpu, width=640, height=480), out)
        walls["kit_launch_s"] = time.perf_counter()-t0
        import warp as wp
        wp.init()
        from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily
        family = RubberStripTaskFamily([episode["slot"]], config=a.task_config, expected_sha256=a.task_config_sha256)
        episode["task_config"] = family.id
        meta = NM.run_meta(a.gpu, protocol=protocol, task_config=family.config_path, extra={
            "label": a.label, "runtime_versions": runtime_versions(), "kit_render_device": {
                "active_gpu": a.gpu, "physics_gpu": a.gpu, "multi_gpu": False, "renderer": "RaytracedLighting",
                "source": "isaac_sim_runner.launch_app (unchanged)"},
            "material_device": f"cuda:{a.gpu}",
            "material_device_pci_bus_id": getattr(wp.get_device(f"cuda:{a.gpu}"), "pci_bus_id", None),
            "material_device_name": wp.get_device(f"cuda:{a.gpu}").name})
        NM.write_new(out/"meta.json", meta)
        slot = episode["slot"]
        geo = family.geometry(slot)
        origin = np.asarray(geo["origin"], np.float64)
        corners = np.asarray(geo["jaw_corners"], np.float64)
        scene = H.KitRubberScene(app, out, robot_substeps=episode["robot_substeps"])
        t0 = time.perf_counter()
        placement = scene.build((corners.min(0), corners.max(0)), origin)
        walls["scene_build_s"] = time.perf_counter()-t0
        material = family.build_material()
        if a.friction_table is not None:
            import dataclasses
            material = dataclasses.replace(material, friction_table=float(a.friction_table))
        t0 = time.perf_counter()
        live = family.build_life(slot, material, device=f"cuda:{a.gpu}", record=True)
        walls["material_build_s"] = time.perf_counter()-t0
        receipt = H.strip_receipt(family, live, slot)
        receipt["episode"] = episode
        NM.write_new(out/"receipt.json", receipt)
        surface = scene.make_surface()
        sc = live.scene
        shim = H.StripCameraShim(placement, geo["length"], family.width, geo["thickness"],
                                 float(sc.receipt_extra["jaw_length"]), origin)
        specs = scene.make_cameras(shim)
        (out/"captures").mkdir()
        consistency = []

        def capture(k):
            t = time.perf_counter()
            frames = scene.capture()
            # before the first advance the marker diagnostics buffer is not computed yet (native_rubber_001 used it
            # at step 0: all 9 markers at one point); the reference cross-section centroids are the step-0 markers
            markers = live.scene.marker_reference if k == 0 else live.markers()
            markers_world = placement.to_world(np.asarray(markers, np.float64))
            row = {"step": k, "capture_wall_s": None, "cameras": {}}
            from PIL import Image
            for name, d in frames.items():
                rgb = np.asarray(d["rgb"])[..., :3].astype(np.uint8)
                depth = np.asarray(d["distance_to_image_plane"], np.float32)
                Image.fromarray(rgb).save(out/"captures"/f"{name}_rgb_{k:03d}.png")
                np.savez_compressed(out/"captures"/f"{name}_depth_{k:03d}.npz", depth=depth)
                cal = specs[name]["authored_usd_readback"]
                row["cameras"][name] = H.marker_depth_consistency(markers_world, cal["K"], cal["T_world_camera"],
                                                                  depth)
            row["capture_wall_s"] = time.perf_counter()-t
            consistency.append(row)

        bridge_file = (out/"bridge.jsonl").open("x")

        def on_step(i, row):
            bridge_file.write(json.dumps(row)+"\n")
            bridge_file.flush()
            if i % 10 == 9:
                print(f"step {i+1}: wall {row['step_wall_s']:.3f} s material {row['material_wall_s']:.3f} s "
                      f"bridge err {row['max_abs_error_m']:.2e} m invalid {live.invalid_reason}", flush=True)

        arrays = H.EpisodeArrays(episode["steps"])
        t0 = time.perf_counter()
        try:
            arrays, rows = H.run_bridge_episode(live, scene, surface, placement, origin, episode=episode,
                                                arrays=arrays, on_step=on_step, capture=capture)
        finally:
            walls["episode_s"] = time.perf_counter()-t0
            arrays.save(out/"native_arrays.npz")
            bridge_file.close()
            with gzip.open(out/"rubber_records.jsonl.gz", "xt") as f:
                for r in live.records:
                    f.write(json.dumps(r, default=NM._default)+"\n")
        # initial bridge row (step 0) is written before the loop's first on_step: put every row in one file
        NM.write_new(out/"bridge_rows.json", rows)
        NM.write_new(out/"camera_consistency.json", consistency)
        NM.write_new(out/"scene.json", {**scene.receipt, "arm_state_end": scene.arm_state(),
                                        "live_receipt_device": live.receipt()["solver"]["device"]})
        step_walls = [r["step_wall_s"] for r in rows if "step_wall_s" in r]
        mat_walls = [r["material_wall_s"] for r in rows if "material_wall_s" in r]
        result.update(status="COMPLETED", completed_steps=int(arrays.completed),
                      invalid_reason_final=int(live.invalid_reason),
                      bridge_max_abs_error_m=max(r["max_abs_error_m"] for r in rows),
                      command_vs_measured_max_m=max(r.get("command_vs_measured_m", 0.0) for r in rows),
                      timing={**walls, "warm_step_mean_s": float(np.mean(step_walls[1:])) if len(step_walls) > 1
                              else None, "material_step_mean_s": float(np.mean(mat_walls[1:])) if len(mat_walls) > 1
                              else None, "captures": len(consistency)},
                      throughput_row={"N": 1, "particles_per_env": int(sc.n), "control_hz": 1/live.control_dt,
                                      "material_dt_s": live.dt, "substeps": live.substeps,
                                      "cameras": "overview + close-up RGB-D 960x540, every 10 steps",
                                      "recording": "rcs_rubber_record_v1 + arrays + surface every step",
                                      "warm": "steps 2..N mean", "gpu": a.gpu})
    except BaseException as exc:
        result.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["gpu_sharing"] = sampler.stop()
        result["gpu_samples_file"] = "gpu_samples.json"
        result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        NM.write_new(out/"gpu_samples.json", sampler.samples)
        NM.write_new(out/"result.json", result)        # before app.close(): Kit may exit Python there
        print(json.dumps({"status": result["status"], "out": str(out)}), flush=True)
        if app is not None:
            app.close(wait_for_replicator=False)
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
