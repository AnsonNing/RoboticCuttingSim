#!/usr/bin/env python3
"""Fetch small official root USDs only; audit availability, not full composition."""
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

root=Path(__file__).resolve().parents[1]
out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
kit=(root/".deps/IsaacLab/apps/isaaclab.python.kit").read_text()
asset_root=re.search(r'persistent\.isaac\.asset_root\.default\s*=\s*"([^"]+)"',kit).group(1)
assets={"d455":"Isaac/Sensors/RealSense/D455/rsd455.usd",
        "franka":"Isaac/IsaacLab/Robots/FrankaEmika/franka_panda.usda"}
records=[]
for name,path in assets.items():
    url=f"{asset_root}/{path}"
    record={"name":name,"url":url,"composition":"NOT_RUN","license":"Official NVIDIA/asset license; no redistribution"}
    try:
        with urllib.request.urlopen(url,timeout=30) as response:
            data=response.read(16*1024*1024+1)
            record["http_status"]=response.status
            record["response_etag"]=response.headers.get("ETag")
        if len(data)>16*1024*1024:
            raise RuntimeError("Root USD exceeds 16 MiB audit cap; complete asset not downloaded")
        target=out/(name+Path(path).suffix); target.write_bytes(data)
        record.update(status="PASS",bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),local=str(target))
    except Exception as error:
        record.update(status="BLOCKED",error=str(error))
    records.append(record)
(out/"results.json").write_text(json.dumps(records,indent=2)+"\n")
print(json.dumps(records,indent=2))
raise SystemExit(2 if any(r["status"]!="PASS" for r in records) else 0)
