#!/usr/bin/env python3
"""Fetch only pinned public source checkouts, without modifying installed projects."""
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
refs=json.loads((root/"configs/source_refs.json").read_text())
for name,ref in refs.items():
    target=root/".deps"/name
    if not target.exists():
        target.mkdir(parents=True)
        subprocess.run(["git","init",str(target)],check=True)
        subprocess.run(["git","-C",str(target),"remote","add","origin",ref["url"]],check=True)
        subprocess.run(["git","-C",str(target),"fetch","--depth","1","origin",ref["sha"]],check=True)
        subprocess.run(["git","-C",str(target),"checkout","--detach",ref["sha"]],check=True)
    actual=subprocess.check_output(["git","-C",str(target),"rev-parse","HEAD"],text=True).strip()
    if actual!=ref["sha"]:
        raise SystemExit(f"{name}: existing checkout differs from pinned source; refusing to overwrite")
    print(f"{name}: {actual}")
