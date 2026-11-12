#!/usr/bin/env python3
"""P6.5 launcher of the Native rubber run (T5.8 pattern, ``scripts/rl_ready/launch_native_cut.py``).

Run it under the RCS GPU lock; it never waits for memory itself. It reads ``nvidia-smi`` for the GPU and refuses
(status ``BLOCKED``, readings recorded in ``launch.json``) unless the free memory is at least ``--need-mib``: the plan's
14 GiB for the Native run PLUS the 4 GiB reserve for other users (18 GiB; the stricter reading of the rule). Otherwise
it runs ``run_native_rubber.py`` in the Isaac Sim venv with ``--gpu`` = render (Kit ``active_gpu``), physics and
material device, ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` and no ``CUDA_VISIBLE_DEVICES``, and records the command, the GPU
readings before / after, the wall and the return code. A zero return code is not task success: RL-09 is.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock \\
        .venv/bin/python scripts/rubber/p2_6/launch_native_rubber.py --gpu 0 --task-config <id> \\
        --task-config-sha256 <sha> --protocol configs/rubber/<protocol>.json --protocol-sha256 <sha> [--smoke-steps N]
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import native_meta as NM  # noqa: E402

OUT_ROOT = ROOT/"artifacts"/"rl_ready"/"p2_6"
PYTHON = ROOT/".deps"/"isaacsim-venv"/"bin"/"python"


def new_dir(name):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for index in range(1, 1000):
        path = OUT_ROOT/f"{name}_{index:03d}"
        try:
            path.mkdir()
            return path
        except FileExistsError:
            continue
    raise RuntimeError("no free directory")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--need-mib", type=float, default=(14+4)*1024.)
    p.add_argument("--timeout-s", type=float, default=3600.)
    p.add_argument("--task-config", required=True)
    p.add_argument("--task-config-sha256", required=True)
    p.add_argument("--protocol", required=True)
    p.add_argument("--protocol-sha256", required=True)
    p.add_argument("--smoke-steps", type=int, default=None, help="label smoke, N steps (never the formal run)")
    p.add_argument("--friction-table", type=float, default=None, help="smoke only")
    a = p.parse_args()
    smoke = a.smoke_steps is not None
    if a.friction_table is not None and not smoke:
        raise SystemExit("--friction-table is a smoke-only override")
    base = new_dir("native_rubber_smoke" if smoke else "native_rubber")
    out = base/"run"
    command = [str(PYTHON.relative_to(ROOT)), "scripts/rubber/p2_6/run_native_rubber.py", "--out",
               str(out.relative_to(ROOT)), "--gpu", str(a.gpu), "--task-config", a.task_config,
               "--task-config-sha256", a.task_config_sha256, "--protocol", a.protocol,
               "--protocol-sha256", a.protocol_sha256]
    if smoke:
        command += ["--steps", str(a.smoke_steps), "--label", "smoke"]
        if a.friction_table is not None:
            command += ["--friction-table", str(a.friction_table)]
    before = NM.smi(a.gpu)
    status = NM.sh("git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts", "configs", "tests")
    record = dict(schema="rcs_p2_6_native_rubber_launch_v1", task="P6.5", label="smoke" if smoke else "formal",
                  command=command, cwd=str(ROOT), gpu=a.gpu, need_free_mib=a.need_mib, gpu_before=before,
                  git_head=NM.sh("git", "rev-parse", "HEAD"), git_dirty=bool(status), dirty_paths=status.splitlines(),
                  env=dict(OMNI_KIT_ACCEPT_EULA="YES", PYTHONUNBUFFERED="1", CUDA_DEVICE_ORDER="PCI_BUS_ID",
                           TMPDIR=os.environ.get("TMPDIR", "/mnt/data/ningan/tmp")),
                  render_device=f"GPU{a.gpu} (Kit active_gpu={a.gpu}, physics_gpu={a.gpu}, multi_gpu False)",
                  started=time.strftime("%Y-%m-%dT%H:%M:%S%z"), pid=os.getpid())
    if before["free_mib"] < a.need_mib:
        record.update(status="BLOCKED", reason=f"GPU{a.gpu} free {before['free_mib']:.0f} MiB < required "
                                                 f"{a.need_mib:.0f} MiB (14 GiB Native + 4 GiB reserve)")
        NM.write_new(base/"launch.json", record)
        print(json.dumps(dict(status="BLOCKED", dir=str(base), free_mib=before["free_mib"])))
        return 2
    NM.write_new(base/"launch.json.started", record)
    env = dict(os.environ, **record["env"])
    env.pop("CUDA_VISIBLE_DEVICES", None)
    log = base/"native.log"
    start = time.perf_counter()
    with log.open("x") as stream:
        try:
            rc = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                timeout=a.timeout_s).returncode
        except subprocess.TimeoutExpired:
            rc = "TIMEOUT"
    record.update(returncode=rc, wall_s=time.perf_counter()-start, gpu_after=NM.smi(a.gpu),
                  ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"), log=str(log.relative_to(ROOT)), status="EXITED",
                  note="A zero return code is not task success; see run/result.json and the RL-09 comparison")
    NM.write_new(base/"launch.json", record)
    print(json.dumps(dict(status="EXITED", returncode=rc, dir=str(base), wall_s=record["wall_s"])))
    return 0 if rc == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
