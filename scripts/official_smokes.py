#!/usr/bin/env python3
"""Actual official Lab CLI attempts, without claiming a missing GA stack is installed."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
env=os.environ.copy()
env["PYTHONPATH"]=os.pathsep.join(str(p) for p in (root/".deps/IsaacLab/source").iterdir() if p.is_dir())
records=[]
for kind,name,extra in [("example","mpm-granular",[]),("example","mpm-two-way-coupling",[]),
                         ("demo","snowball-smash",[]),("demo","teapot-fill",["--visualizer","newton_gl","--fluid_render_mode","surface"])]:
    args=[sys.executable,"-m","isaaclab",kind,name,"--max_steps","3",*extra]
    tick=time.perf_counter()
    result=subprocess.run(args,env=env,capture_output=True,text=True,timeout=90)
    (out/f"{name}.log").write_text(result.stdout+result.stderr)
    records.append({"name":name,"command":args,"returncode":result.returncode,
                    "wall_s":time.perf_counter()-tick,"status":"PASS" if result.returncode==0 else "BLOCKED",
                    "environment":"source-only CLI discovery, GA dependencies not installed",
                    "last_error":result.stderr.strip().splitlines()[-1] if result.stderr else None})
(out/"results.json").write_text(json.dumps(records,indent=2)+"\n")
print(json.dumps(records,indent=2))
raise SystemExit(2 if any(r["status"]=="BLOCKED" for r in records) else 0)
