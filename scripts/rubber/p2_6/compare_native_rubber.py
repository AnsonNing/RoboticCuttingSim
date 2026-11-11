"""P6.5 RL-09 comparison (CPU, no Warp, no Kit): Native rubber run vs its headless GPU replays.

Applies ``rubber_native_compare.evaluate_rl09`` (clauses a-e, see that module) to a Native run directory and a replay
directory; repeats 0..19 are the band, 20..29 held out. Writes ``rl09.json`` into a NEW directory and, with
``--evidence``, the read-only evidence file (refuses to overwrite).

    CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/rubber/p2_6/compare_native_rubber.py \\
        --native artifacts/rl_ready/p2_6/native_rubber_001 --replays artifacts/rl_ready/p2_6/<replay dir> \\
        --out artifacts/rl_ready/p2_6/<new> [--evidence docs/evidence/rl_ready/p2_6_native_rubber_001.json]
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_meta as NM  # noqa: E402


def load_native(base):
    """``base``: the launcher directory (launch.json + run/) or the run directory itself."""
    from robotic_cutting_sim.native_rubber_host import EpisodeArrays
    base = Path(base)
    run = base/"run" if (base/"run").is_dir() else base
    launch = json.loads((run.parent/"launch.json").read_text()) if (run.parent/"launch.json").is_file() else None
    return {"run": run, "launch": launch, "meta": json.loads((run/"meta.json").read_text()),
            "result": json.loads((run/"result.json").read_text()),
            "receipt": json.loads((run/"receipt.json").read_text()),
            "scene": json.loads((run/"scene.json").read_text()),
            "arrays": EpisodeArrays.load(run/"native_arrays.npz"),
            "bridge_rows": json.loads((run/"bridge_rows.json").read_text()),
            "camera": json.loads((run/"camera_consistency.json").read_text())}


def load_replays(d, band=20):
    from robotic_cutting_sim.native_rubber_host import EpisodeArrays
    d = Path(d)
    files = sorted(d.glob("replay_*.npz"))
    arrays = {f.stem: EpisodeArrays.load(f) for f in files}
    receipts = {f"replay_{f.stem.split('_')[1]}": json.loads(f.read_text()) for f in sorted(d.glob("receipt_*.json"))}
    names = sorted(arrays)
    return ({n: arrays[n] for n in names[:band]}, {n: arrays[n] for n in names[band:]}, receipts,
            json.loads((d/"meta.json").read_text()), json.loads((d/"result.json").read_text()))


def camera_summary(rows):
    """Per camera over the captures after step 0. Step 0 of native_rubber_001 projected the uncomputed marker
    diagnostics buffer (all markers at one point; fixed in run_native_rubber.py afterwards), so it is excluded."""
    out = {"excluded": "step 0 (marker diagnostics not computed before the first advance)"}
    for row in rows:
        if row["step"] == 0:
            continue
        for cam, c in row["cameras"].items():
            o = out.setdefault(cam, {"captures": 0, "visible_markers": 0, "residual_min_m": None,
                                     "residual_max_m": None})
            o["captures"] += 1
            o["visible_markers"] += c["visible"]
            for key, fn in (("residual_min_m", min), ("residual_max_m", max)):
                if c[key] is not None:
                    o[key] = c[key] if o[key] is None else fn(o[key], c[key])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--native", required=True)
    ap.add_argument("--replays", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--evidence", default=None)
    a = ap.parse_args()
    from robotic_cutting_sim import rubber_native_compare as C
    from robotic_cutting_sim.native_rubber_host import EPISODE, STAT_NAMES
    native = load_native(ROOT/a.native)
    band, heldout, receipts, rmeta, rresult = load_replays(ROOT/a.replays)
    out = ROOT/a.out
    out.mkdir(parents=True)
    translation = native["scene"]["placement"]["world_from_strip_translation_m"]
    cam = camera_summary(native["camera"])
    ev = C.evaluate_rl09(native_receipt=native["receipt"], replay_receipts=receipts, native_arrays=native["arrays"],
                         band_arrays=band, heldout_arrays=heldout, stat_names=STAT_NAMES,
                         placement_translation=translation, bridge_rows=native["bridge_rows"],
                         expected_steps=EPISODE["steps"], camera_consistency=cam)
    m, s = native["meta"], native["result"]
    gpu = m["gpu_start"]
    native_row = {
        "run": str(native["run"].relative_to(ROOT)), "label": s.get("label"), "status": s.get("status"),
        "gpu": f"#{gpu['index']} {gpu['name']}", "driver": gpu["driver"],
        "render_device": m.get("kit_render_device"), "material_device": m.get("material_device"),
        "material_device_pci_bus_id": m.get("material_device_pci_bus_id"),
        "sharing": s.get("gpu_sharing", {}).get("sharing"), "job_vram_peak_mib": s.get("gpu_sharing", {}).get(
            "job_vram_peak_mib"), "other_compute_processes": s.get("gpu_sharing", {}).get("other_compute_processes"),
        "commit": m["commit"], "dirty_tree": m["dirty_tree"], "protocol": m["protocol"], "task_config": m["task_config"],
        "warp": m.get("warp"), "runtime_versions": m.get("runtime_versions"), "timing": s.get("timing"),
        "throughput_row": s.get("throughput_row")}
    payload = {"schema": "rcs_p2_6_native_rubber_rl09_v1", "test": "RL-09", "verdict": ev["verdict"],
               "calibration_status": "SIMULATOR_ONLY_UNCALIBRATED", "native_row": native_row,
               "launch": native["launch"], "replay_meta": rmeta,
               "replay_result": {k: v for k, v in rresult.items() if k != "traceback"},
               "evaluation": ev, "command": [sys.executable, *sys.argv]}
    NM.write_new(out/"rl09.json", payload)
    if a.evidence:
        ev_path = ROOT/a.evidence
        slim = dict(payload)
        slim["evaluation"] = {**ev, "clauses": {**ev["clauses"], "c_physics": {
            k: v for k, v in ev["clauses"]["c_physics"].items() if k != "native_rows"}}}
        NM.write_new(ev_path, slim)
    c = ev["clauses"]
    print(json.dumps({"verdict": ev["verdict"], **{k: v["verdict"] for k, v in c.items()},
                      "native_max_M_over_upper": c["c_physics"]["native_max_M_over_upper"],
                      "heldout_pass": c["c_physics"]["heldout"]["pass"],
                      "bridge_max_err_m": c["e_bridge_integrity"]["max_abs_error_m"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
