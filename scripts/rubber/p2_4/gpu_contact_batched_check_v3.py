"""P2-4 X3: RUBBER-CONT-09 (GPU per-env isolation of the batched contact path) and the batched RUBBER-CONT-11 cost rows
under ``configs/rubber/rubber_contact_tests_v3.json`` with the law ``grid_node_penalty_coulomb_v3``.

The B2-v2 script ``scripts/rubber/p2_4b/gpu_contact_batched_check_v2.py`` (gpu_cont09_v2_001, PASS for v2) is run
unchanged with three module constants replaced: the protocol path and its pinned sha256 (v3) and the law (v3, also as
the default of its scene builder ``make_lives``). Configuration, k = 5 A / 5 B interleaved episodes, the v2 rule
(lam 4, eps_rel 1e-6, read from the v3 protocol, which keeps the v2 thresholds), audits (now including the v3 patch
rows) and the cost rows are those of the B2-v2 script. Its meta label text names the v2 law; this wrapper rewrites the
label of ``result.json`` after the run and keeps the inherited text in ``label_inherited``.

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock env CUDA_VISIBLE_DEVICES=0 \\
        TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .venv/bin/python \\
        scripts/rubber/p2_4/gpu_contact_batched_check_v3.py --out artifacts/rl_ready/p2_4/<dir>
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROTOCOL = ROOT/"configs"/"rubber"/"rubber_contact_tests_v3.json"
PROTOCOL_SHA256 = "36d70beaf63a0ef73195dcc47c62e522596779bb95b423a5544e5f7dc9e6a856"
LAW = "grid_node_penalty_coulomb_v3"


def main():
    path = ROOT/"scripts"/"rubber"/"p2_4b"/"gpu_contact_batched_check_v2.py"
    spec = importlib.util.spec_from_file_location("gpu_contact_batched_check_v2", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.PROTOCOL, mod.PROTOCOL_SHA256, mod.LAW = PROTOCOL, PROTOCOL_SHA256, LAW
    d = list(mod.make_lives.__defaults__)
    d[-1] = LAW
    mod.make_lives.__defaults__ = tuple(d)
    out = Path(sys.argv[sys.argv.index("--out")+1])
    mod.main()
    res = out/"result.json"
    if res.exists():
        r = json.loads(res.read_text())
        r["meta"]["label_inherited"] = r["meta"]["label"]
        r["meta"]["label"] = ("RUBBER-CONT-09 under rubber_contact_tests_v3 (v2 rule; GPU per-env isolation, batched "
                              "contact, law grid_node_penalty_coulomb_v3) + RUBBER-CONT-11 batched cost rows of the v3 "
                              "law (reported, not scored); B2-v2 script via gpu_contact_batched_check_v3.py")
        r["meta"]["wrapper"] = "scripts/rubber/p2_4/gpu_contact_batched_check_v3.py"
        res.write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
