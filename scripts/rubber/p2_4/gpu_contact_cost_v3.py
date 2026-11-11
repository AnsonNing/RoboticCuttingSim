"""P2-4 X3: single-env RUBBER-CONT-11 cost rows (reported, not scored) of the contact laws v2 and v3 on GPU0.

Runs ``scripts/rubber/p2_4/gpu_contact_check.py`` (workload E single env: no contact / contact / contact out of reach,
CUDA graph chunks, stage split, GPU vs CPU pinch) with every ``ContactSpec`` it builds forced to ``--law`` (v2 or v3,
the v3 spec with the protocol's zeta_d) and its protocol field pointed at ``rubber_contact_tests_v3`` (sha256 recorded).
Run once per law, under the GPU0 lock:

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \\
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python scripts/rubber/p2_4/gpu_contact_cost_v3.py \\
        --law grid_node_penalty_coulomb_v3 --out artifacts/rl_ready/p2_4/<dir>
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json"
LAWS = ("grid_node_penalty_coulomb_v2", "grid_node_penalty_coulomb_v3")


def main():
    law = sys.argv[sys.argv.index("--law")+1]
    if law not in LAWS:
        raise SystemExit(f"--law must be one of {LAWS}")
    del sys.argv[sys.argv.index("--law"):sys.argv.index("--law")+2]
    sys.path.insert(0, str(ROOT/"src"))
    from robotic_cutting_sim import rubber_contact as rct
    zd = json.loads(PROTOCOL.read_text())["common"]["law_parameters"]["slip_rate_damping_ratio_zeta_d"]
    base = rct.ContactSpec
    built = []

    class Spec(base):
        def __init__(self, *args, **kw):
            extra = {"slip_rate_damping_ratio": zd} if law == LAWS[1] else {}
            base.__init__(self, *args, **{**kw, "law": law, **extra})
            built.append(self.law)
    rct.ContactSpec = Spec
    path = ROOT/"scripts"/"rubber"/"p2_4"/"gpu_contact_check.py"
    spec = importlib.util.spec_from_file_location("gpu_contact_check", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PROTOCOL = PROTOCOL
    out = Path(sys.argv[sys.argv.index("--out")+1])
    mod.main()
    for p in sorted(out.glob("*.json")):
        r = json.loads(p.read_text())
        if isinstance(r, dict):
            r["x3_wrapper"] = {"script": "scripts/rubber/p2_4/gpu_contact_cost_v3.py", "law_forced": law,
                               "laws_built": sorted(set(built)), "protocol": str(PROTOCOL.relative_to(ROOT))}
            p.write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
