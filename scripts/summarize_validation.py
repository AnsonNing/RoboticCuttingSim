#!/usr/bin/env python3
"""Compact, hashed evidence index suitable for committing; raw evidence stays local."""
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=root/"docs/evidence"; out.mkdir(exist_ok=True)
invalid={
 "newton_smoke_001":"PRE_FIX: stale output-buffer blade transform before projection",
 "newton_settle_001":"PRE_FIX: superseded by corrected buffer-synchronized runs",
 "newton_indent_001":"PRE_FIX: stale output-buffer blade transform before projection",
 "newton_scrape_001":"PRE_FIX: stale output-buffer blade transform before projection",
 "newton_elastic_indent_001":"FAIL: YAML exponent loaded as string; fixed numeric config and rerun",
 "cressim_scrape_001":"INVALID_ADAPTER: C++ Bounds3 min/max confused with C API center/extents",
 "cressim_scrape_002":"UNCERTIFIED: upstream nodal mass cutoff, APIC/material/normal defects",
 "camera_indent_001":"SUPERSEDED: old input and camera spacing/provenance guards; plane math independently checked",
 "tlmpm_cut_001":"INVALID_FIXTURE: unbounded walls and inconsistent PIC position/MUSL velocity; preserved failure",
 "tlmpm_cut_002":"INVALID_FIXTURE: falling chip met an unbounded side-wall half-space; preserved failure",
 "tlmpm_cross_cuts_001":"INVALID_GUARD: absent support still tested against table z limit; not cut proof",
 "tlmpm_cross_cuts_002":"NEGATIVE_CONTROL: unsupported zero-g specimen moved away; no complete split",
 "ovrtx_rgbd_001":"INVALID_RENDER_SCALE: root stage SI/up-axis metadata missing; superseded",
 "ovrtx_coupled_001":"INTERMEDIATE_RENDER: inward blade facets; superseded by outward-wedge002, physics unchanged",
}
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
records=[]
artifact_root=root/"artifacts"
paths=set()
for pattern in ["*/result.json","*/*/result.json","*/*/*/result.json","*/failure.json","*/*/failure.json","*/*/*/failure.json"]:
    paths.update(p.parent for p in artifact_root.glob(pattern) if p.parent.name!="physics_preview")
for path in sorted(paths):
    result=path/"result.json"
    if not result.exists():
        result=path/"failure.json"
    if not result.exists(): continue
    relative=str(path.relative_to(artifact_root))
    payload=json.loads(result.read_text())
    # Retain compact endpoints and the full source-file hash rather than copying
    # dozens of repeated per-camera calibration payloads into the public index.
    for key in ("frames","mechanical_history"):
        if isinstance(payload.get(key),list) and len(payload[key])>2:
            values=payload[key]
            payload[key]={"count":len(values),"first":values[0],"last":values[-1],
                          "scope":"endpoints only; full arrays/records retained in hashed source JSON"}
    record={"run":relative,"local_path":str(path),"classification":invalid.get(relative,"CURRENT_EVIDENCE"),
            "result":payload}
    m=path/"manifest.json"
    if m.exists():
        manifest=json.loads(m.read_text())
        record["source"]={key:manifest.get(key) for key in ["git_sha","git_dirty","config_sha256","packages","source_checkouts","local_code_sha256"]}
    record["artifact_sha256"]={p.name:digest(p) for p in [result,m,path/"trace.csv",path/"physics_state.npz",path/"preview.mp4"] if p.is_file()}
    receipt=path.parent/(path.name+".receipt.json")
    if not receipt.is_file() and path.parent==artifact_root:
        receipt=root/"artifacts/runs"/(path.name+".receipt.json")
    if receipt.is_file():
        r=json.loads(receipt.read_text()); peaks={}
        for sample in r.get("gpu_samples",[]):
            for line in sample["gpus"].splitlines():
                values=[v.strip() for v in line.split(",")]
                if len(values)==3:
                    peaks[values[0]]=max(peaks.get(values[0],0),int(values[1]))
        record["execution"]={"returncode":r.get("returncode"),"wall_s":r.get("wall_s"),
                             "peak_sampled_total_device_MiB":peaks,"memory_sampled":bool(r.get("gpu_samples")),
                             "command":r.get("command",r.get("argv")),"environment":r.get("environment")}
    records.append(record)
summary={"date":"2026-10-28 Asia/Seoul","gate_states":{"PHYSICS_PROVEN":False,"VISION_PROVEN":False,"WORKCELL_PROVEN":False,"BENCHMARK_READY":False},
         "engineering_capabilities":{"persistent_cutting_D0":"PASS_EXPERIMENTAL","coupled_robot_cutting_D0":"PASS_EXPERIMENTAL",
             "live_ideal_rgbd":"PASS","standalone_rtx_rgbd":"PASS","simulated_d455_validated":False,
             "benchmark_executable":True,"full_isaac_runtime":"BLOCKED"},
         "classification_registry":invalid,
         "raw_evidence":"Local artifacts/ is ignored by Git; hashes are recorded here, binary data is not redistributed",
         "runs":records,"checks":{}}
for name in ["reset_reproducibility.json","replay_validation.json","handoff_validation.json"]:
    path=artifact_root/name
    if path.is_file():
        summary["checks"][name]=json.loads(path.read_text())
(out/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(root/"artifacts/RUN_VALIDITY.json").write_text(json.dumps(invalid,indent=2)+"\n")
print(out/"validation_summary.json")
