#!/usr/bin/env python3
"""Knife-motion probes for the TLMPM cutting laws (kinematic ruler blade).

Runs each motion under cutting law 1 and law 2 on a lattice block and reports
labels, severed reference bonds, pieces and the tool reaction. CPU at 4 mm for
quick checks; GPU at the protocol 1 mm for WP-10D:

  python scripts/probe_knife_motions.py --out artifacts/knife_motions_cpu.json
  python scripts/probe_knife_motions.py --device cuda:0 --spacing .001 --block .06 .024 .012 --out ...
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))


def motions(top):
    from robotic_cutting_sim.blade_probe import FLAT_SKIM, VERTICAL_CUT, Phase
    from scipy.spatial.transform import Rotation
    tilt = Rotation.from_euler("y", 20, degrees=True)*VERTICAL_CUT  # oblique about the edge line
    return {
        "push_down": (VERTICAL_CUT, [.001, 0, top+.004], [Phase((0, 0, -.05), .4)]),
        "oblique20_push": (tilt, [.001-.004*np.tan(np.radians(20)), 0, top+.004],
                           [Phase(tuple(tilt.apply([.05, 0, 0])), .45)]),
        "slow_push": (VERTICAL_CUT, [.001, 0, top+.001], [Phase((0, 0, -.01), .4)]),
        "slice_along_edge": (VERTICAL_CUT, [.001, -.03, top+.001], [Phase((0, .2, -.01), .4)]),
        "sideways_push": (VERTICAL_CUT, [-.034, 0, .002], [Phase((.05, 0, 0), .3)]),
        "flat_press": (FLAT_SKIM, [.010, 0, top+.002], [Phase((0, 0, -.02), .3)]),
        "cut_half_then_pull_out": (VERTICAL_CUT, [.001, 0, top+.004], [Phase((0, 0, -.05), .2), Phase((0, 0, .05), .2)]),
        "cut_half_then_sideways": (VERTICAL_CUT, [.001, 0, top+.004], [Phase((0, 0, -.05), .2), Phase((.05, 0, 0), .16)]),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spacing", type=float, default=.004)
    parser.add_argument("--block", type=float, nargs=3, default=(.06, .024, .012))
    parser.add_argument("--cutting-energy", type=float, default=5.)
    parser.add_argument("--laws", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}")
    from robotic_cutting_sim.blade_probe import block, drive_blade
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1
    rows = []
    for name, (rotation, start, phases) in motions(args.block[2]).items():
        for law in args.laws:
            points, masses = block(tuple(args.block), args.spacing)
            began = time.time()
            result = drive_blade(RULER_KNIFE_V1, rotation, start, phases, points=points, masses=masses,
                                 spacing=args.spacing, cutting_law=law, cutting_energy=args.cutting_energy,
                                 device=args.device)
            rows.append({"motion": name, "cutting_law": law, "labelled_after_phase": result["labelled_after_phase"],
                         "severed_bonds": result["severed_bonds"], "pieces": result["components"],
                         "largest_mass_fractions": [round(f, 4) for f in result["mass_fractions"][:3]],
                         "forces": result["forces"], "wall_s": round(time.time()-began, 2)})
            print(name, law, rows[-1]["severed_bonds"], rows[-1]["pieces"],
                  [round(v, 4) for v in rows[-1]["forces"][-1]["mean_N"]], flush=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"command": " ".join(sys.argv), "commit": commit, "profile": "ruler_knife_v1",
                               "scope": "Kinematic blade; uncalibrated provisional material; not robot evidence",
                               "rows": rows}, indent=1)+"\n")


if __name__ == "__main__":
    main()
