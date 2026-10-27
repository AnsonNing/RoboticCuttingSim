#!/usr/bin/env python3
"""Run one command, preserve exit status/log, and sample total per-GPU memory.

Example: .venv/bin/python scripts/run_logged.py artifacts/run.log -- command args
Memory is device-wide, includes baseline/co-tenants, and is sampled, not allocator peak.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

path=Path(sys.argv[1]); args=sys.argv[3:]
if sys.argv[2]!="--" or not args:
    raise SystemExit("run_logged.py LOG -- COMMAND [ARGS]")
path.parent.mkdir(parents=True,exist_ok=True)
if path.exists():
    raise SystemExit(f"Refusing to overwrite {path}")
samples=[]; start=time.perf_counter()
with path.open("w") as log:
    process=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
    while process.poll() is None:
        result=subprocess.run(["nvidia-smi","--query-gpu=index,memory.used,utilization.gpu","--format=csv,noheader,nounits"],
                              capture_output=True,text=True)
        samples.append({"wall_s":time.perf_counter()-start,"gpus":result.stdout.strip()})
        time.sleep(.5)
    code=process.returncode
record={"command":args,"returncode":code,"wall_s":time.perf_counter()-start,
        "environment":{k:os.environ.get(k) for k in ["CUDA_VISIBLE_DEVICES","OMP_NUM_THREADS","MPLBACKEND"]},
        "gpu_samples":samples,"memory_scope":"sampled total GPU MiB including baseline"}
path.with_suffix(".receipt.json").write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps({k:v for k,v in record.items() if k!="gpu_samples"},indent=2))
raise SystemExit(code)
