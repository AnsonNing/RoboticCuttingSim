#!/usr/bin/env python3
"""P1-5 T5.8: launch the pre-declared Native tlmpm_v7_2 CUT episode (``cut_v7_2_v1.json`` native_reference).

Run it under the RCS GPU0 lock; it never waits for memory itself. On start it reads ``nvidia-smi`` for the GPU and
refuses (status ``BLOCKED``, the readings recorded) unless the free memory is at least ``--need-mib`` (10 GiB
for the Native episode plus the 4 GiB reserve for other users). Otherwise it runs the declared command verbatim
(only ``<out>`` substituted), with Kit on that GPU (``--gpu`` sets render, physics and material device), and
records ``launch.json`` (command, environment, GPU readings before/after, wall, return code). A zero return code
is not task success: the evaluator verdict in the run's ``episode_000/result.json`` is.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \
        .deps/isaacsim-venv/bin/python scripts/rl_ready/launch_native_cut.py --gpu 0
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts" / "rl_ready" / "p1_5"


def smi(gpu):
    q = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    uuid = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=uuid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip()
    index, total, used, util = [x.strip() for x in q.split(",")]
    return dict(query=q, total_mib=float(total), used_mib=float(used), free_mib=float(total) - float(used),
                util_pct=float(util), compute_apps=[a for a in apps if a.startswith(uuid)],
                time=time.strftime("%Y-%m-%dT%H:%M:%S%z"))


def new_dir(name):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = OUT_ROOT / f"{name}_{index:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("no free directory")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--need-mib", type=float, default=(10 + 4) * 1024.)
    p.add_argument("--timeout-s", type=float, default=3600.)
    a = p.parse_args()
    declaration = ROOT / "configs" / "rl_benchmark" / "cut_v7_2_v1.json"
    command = list(json.loads(declaration.read_text())["native_reference"]["t5_8_command"])
    gpu_flag = command.index("--gpu")
    if command[gpu_flag + 1] != str(a.gpu):
        raise SystemExit(f"the declared command runs on GPU {command[gpu_flag + 1]}, not {a.gpu}")
    base = new_dir("native_cut_v7_2")
    out = base / "run"
    command[command.index("--out") + 1] = str(out.relative_to(ROOT))
    before = smi(a.gpu)
    record = dict(schema="rcs_rl_p1_5_native_launch_v1", task="T5.8", declaration=str(declaration.relative_to(ROOT)),
                  command=command, cwd=str(ROOT), gpu=a.gpu, need_free_mib=a.need_mib, gpu_before=before,
                  git_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                          text=True).stdout.strip(),
                  git_dirty=bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                                capture_output=True, text=True).stdout.strip()),
                  env=dict(OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1",
                           TMPDIR=os.environ.get("TMPDIR", "/mnt/data/ningan/tmp")),
                  started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=os.getpid())
    if before["free_mib"] < a.need_mib:
        record.update(status="BLOCKED", reason=f"GPU{a.gpu} free {before['free_mib']:.0f} MiB < required "
                                                 f"{a.need_mib:.0f} MiB (10 GiB Native + 4 GiB reserve)")
        (base / "launch.json").write_text(json.dumps(record, indent=1))
        print(json.dumps(dict(status="BLOCKED", dir=str(base), free_mib=before["free_mib"])))
        return 2
    (base / "launch.json.started").write_text(json.dumps(record, indent=1))
    env = dict(os.environ, **record["env"])
    env.pop("CUDA_VISIBLE_DEVICES", None)
    log = base / "native.log"
    start = time.perf_counter()
    with log.open("x") as stream:
        try:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=a.timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    record.update(returncode=rc, wall_s=time.perf_counter() - start, gpu_after=smi(a.gpu),
                  ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"), log=str(log.relative_to(ROOT)),
                  status="EXITED", note="A zero return code is not task success; see episode_000/result.json")
    (base / "launch.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(dict(status="EXITED", returncode=rc, dir=str(base), wall_s=record["wall_s"])))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
