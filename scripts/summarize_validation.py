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
}
def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
records=[]
for path in sorted((root/"artifacts").iterdir()):
    result=path/"result.json"
    if not result.exists():
        result=path/"failure.json"
    if not result.exists(): continue
    record={"run":path.name,"local_path":str(path),"classification":invalid.get(path.name,"CURRENT_EVIDENCE"),
            "result":json.loads(result.read_text())}
    m=path/"manifest.json"
    if m.exists():
        manifest=json.loads(m.read_text())
        record["source"]={key:manifest.get(key) for key in ["git_sha","git_dirty","config_sha256","packages","source_checkouts"]}
    record["artifact_sha256"]={p.name:digest(p) for p in [result,m,path/"trace.csv",path/"physics_state.npz",path/"preview.mp4"] if p.is_file()}
    receipt=root/"artifacts/runs"/(path.name+".receipt.json")
    if receipt.is_file():
        r=json.loads(receipt.read_text()); peaks={}
        for sample in r["gpu_samples"]:
            for line in sample["gpus"].splitlines():
                values=[v.strip() for v in line.split(",")]
                if len(values)==3:
                    peaks[values[0]]=max(peaks.get(values[0],0),int(values[1]))
        record["execution"]={"returncode":r["returncode"],"wall_s":r["wall_s"],
                             "peak_sampled_total_device_MiB":peaks,"command":r["command"],"environment":r["environment"]}
    records.append(record)
summary={"date":"2026-10-28 Asia/Seoul","gate_states":{"PHYSICS_PROVEN":False,"VISION_PROVEN":False,"WORKCELL_PROVEN":False,"BENCHMARK_READY":False},
         "raw_evidence":"Local artifacts/ is ignored by Git; hashes are recorded here, binary data is not redistributed",
         "runs":records}
(out/"validation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
(root/"artifacts/RUN_VALIDITY.json").write_text(json.dumps(invalid,indent=2)+"\n")
print(out/"validation_summary.json")
