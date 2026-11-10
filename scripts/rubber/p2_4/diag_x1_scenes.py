"""DIAGNOSTIC (task X1; NOT a protocol result): run the scenes of ``run_contact_v1.py`` (INC / PEN / IMP / PINCH) with a
different contact law or law parameters, and evaluate them with the v1 evaluator for comparison.

The law is substituted by wrapping ``rubber_contact.ContactSpec`` (every ``ContactSpec(...)`` the v1 job functions build
gets ``law=`` and the given parameters); the scene code, loads and measurements are those of ``run_contact_v1.py``.
Keys of ``--set``: law, kn, kt, zt, zn (ContactSpec fields), and the job list ``--jobs-only`` (comma-separated v1 job names).

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/diag_x1_scenes.py \
        --out artifacts/rl_ready/p2_4/<dir> --set law=grid_node_penalty_coulomb_v2 [--only INC,PINCH] [--jobs 32]
"""
import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"/"rubber"/"p2_4"))

KEYS = {"law": ("law", str), "kn": ("normal_stiffness", float), "kt": ("tangential_stiffness", float),
        "zt": ("damping_ratio", float), "zn": ("normal_damping_ratio", float)}


def _patch(overrides):
    from robotic_cutting_sim import rubber_contact as rct
    base = rct.ContactSpec
    if getattr(base, "_x1_patched", False):
        return

    class Spec(base):
        _x1_patched = True

        def __init__(self, *args, **kw):
            kw = {**kw, **{k: v for k, v in overrides.items() if k not in kw or k == "law"}}
            base.__init__(self, *args, **kw)
    rct.ContactSpec = Spec


def _run(item):
    name, fn_name, args, overrides = item
    _patch(overrides)
    import run_contact_v1 as R
    try:
        return name, getattr(R, fn_name)(name, *args)
    except Exception as exc:
        import traceback
        return name, {"name": name, "error": repr(exc), "traceback": traceback.format_exc()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--only", default="INC,PEN,IMP,PINCH")
    ap.add_argument("--jobs-only", default="")
    ap.add_argument("--jobs", type=int, default=16)
    a = ap.parse_args()
    overrides = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        field, typ = KEYS[k]
        overrides[field] = typ(v)
    import run_contact_v1 as R
    cfg = R._cfg()
    table = R.jobs(cfg, set(a.only.split(",")))
    if a.jobs_only:
        keep = set(a.jobs_only.split(","))
        table = {k: v for k, v in table.items() if k in keep}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    (out/"jobs").mkdir()
    git = lambda *c: subprocess.run(["git", *c], capture_output=True, text=True, cwd=ROOT).stdout.strip()  # noqa
    meta = {"label": "DIAGNOSTIC (task X1; not a protocol result)", "overrides": overrides,
            "commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "argv": sys.argv, "start": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
    (out/"meta.json").write_text(json.dumps(meta, indent=1))
    items = [(n, fn.__name__, args, overrides) for n, (fn, args) in table.items()]
    R_ = {}
    with ProcessPoolExecutor(a.jobs) as ex:
        for name, res in ex.map(_run, items):
            (out/"jobs"/f"{name}.json").write_text(json.dumps(res, default=float))
            R_[name] = res
            print(name, "ERROR "+res["error"] if "error" in res else f"{res.get('wall_s', 0):.1f} s", flush=True)
    summary = R.evaluate(cfg, R_)
    (out/"summary_v1_evaluator.json").write_text(json.dumps(summary, indent=1, default=float))
    for tid, v in summary.items():
        if tid.startswith("RUBBER") and "criteria" in v:
            print(tid, v.get("verdict"), v.get("criteria"))


if __name__ == "__main__":
    main()
