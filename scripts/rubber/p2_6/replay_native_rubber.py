"""P6.5 headless GPU replays of a Native rubber run (plan §4.6; RL-09 reference): no Kit.

Feeds the RECORDED measured clamp poses (strip frame, float64, ``native_arrays.npz``) to fresh
``RubberStripTaskFamily.build_life`` lives (same recipe, slot, profile, contact law and build material as the Native
host; record on, CUDA graphs) through ``native_rubber_host.advance_measured`` — the same call the Native loop uses.
Repeats 0..19 form the ``rubber_l3_band_v4`` band, 20..29 are the held-out repeats (reported). One process, one
device, one live at a time (each is freed before the next is built).

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/replay_native_rubber.py \\
        --native artifacts/rl_ready/p2_6/native_rubber_001/run --out artifacts/rl_ready/p2_6/<new> \\
        --task-config <id> --task-config-sha256 <sha> --protocol configs/rubber/<p>.json --protocol-sha256 <sha>

Outputs (new directory): meta.json, replay_NN.npz, receipt_NN.json, result.json, gpu_samples.json.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_meta as NM  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", required=True, help="the Native run directory (…/run)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--gpu-index", type=int, default=0, help="nvidia-smi index of the device (sampling)")
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--task-config", required=True)
    ap.add_argument("--task-config-sha256", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--protocol-sha256", required=True)
    a = ap.parse_args()
    from robotic_cutting_sim import native_rubber_host as H
    native = Path(a.native) if Path(a.native).is_absolute() else ROOT/a.native
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT/a.out
    out.mkdir(parents=True)
    protocol = Path(a.protocol) if Path(a.protocol).is_absolute() else ROOT/a.protocol
    if NM.sha256(protocol) != a.protocol_sha256:
        raise SystemExit(f"protocol sha256 {NM.sha256(protocol)} != {a.protocol_sha256}")
    import warp as wp
    wp.init()
    from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily
    nat = H.EpisodeArrays.load(native/"native_arrays.npz")
    steps = int(nat["completed"])
    episode = dict(H.EPISODE)
    family = RubberStripTaskFamily([episode["slot"]], config=a.task_config, expected_sha256=a.task_config_sha256)
    episode.update(task_config=family.id, protocol={"path": str(protocol.relative_to(ROOT)),
                                                    "sha256": a.protocol_sha256})
    meta = NM.run_meta(a.gpu_index, protocol=protocol, task_config=family.config_path, extra={
        "out": str(out.relative_to(ROOT)), "native_run": str(native.relative_to(ROOT)), "native_arrays_sha256": NM.sha256(native/"native_arrays.npz"),
        "device": a.device, "device_name": wp.get_device(a.device).name,
        "device_pci_bus_id": getattr(wp.get_device(a.device), "pci_bus_id", None), "repeats": a.repeats,
        "band_repeats": "0..19", "heldout_repeats": "20..29", "steps": steps})
    NM.write_new(out/"meta.json", meta)
    sampler = NM.SmiSampler(a.gpu_index, period_s=10.0).start()
    slot = episode["slot"]
    origin = np.asarray(family.geometry(slot)["origin"], np.float64)
    result = {"schema": "rcs_native_rubber_replay_v1", "status": "STARTED", "repeats": []}
    try:
        for r in range(a.repeats):
            t0 = time.perf_counter()
            live = family.build_life(slot, device=a.device, record=True)
            build = time.perf_counter()-t0
            receipt = H.strip_receipt(family, live, slot)
            receipt["episode"] = episode
            NM.write_new(out/f"receipt_{r:02d}.json", receipt)
            t0 = time.perf_counter()
            arrays = H.replay_bridge_inputs(live, nat["material_p"][:steps], nat["material_q"][:steps], origin)
            wall = time.perf_counter()-t0
            arrays.save(out/f"replay_{r:02d}.npz")
            row = {"repeat": r, "build_s": build, "episode_s": wall, "step_mean_s": wall/max(steps, 1),
                   "invalid_reason_final": int(live.invalid_reason), "role": "band" if r < 20 else "heldout"}
            result["repeats"].append(row)
            print(json.dumps(row), flush=True)
            del live, arrays
            gc.collect()
        result["status"] = "COMPLETED"
    except BaseException as exc:
        result.update(status="FAILED", error=repr(exc), traceback=traceback.format_exc())
        print(result["traceback"], file=sys.stderr, flush=True)
    finally:
        result["gpu_sharing"] = sampler.stop()
        NM.write_new(out/"gpu_samples.json", sampler.samples)
        result["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        NM.write_new(out/"result.json", result)
    return 0 if result["status"] == "COMPLETED" else 1


if __name__ == "__main__":
    sys.exit(main())
