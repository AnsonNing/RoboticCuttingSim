"""G3 (ADR 0022 §8) analysis: B2 snapshot restore against a never-reset world, and the settle spread over rebuilds.

Input: ``probe_lab_batched_gpu.py --probe g3`` runs (N=2, hold driver, env 1 restored to the settled snapshot at
``--restore-step``). Per step the hand-relative tool pose ``inv(T_world_hand) T_world_tool`` is computed from the
saved body poses.

* restore vs fresh: env 1 for ``window`` steps after its restore against env 0 for its first ``window`` steps
  after the build settle (both start from the same snapshot state; env 1's MuJoCo warm start is zeroed by the
  restore, env 0's is the settle's). In deterministic mode any difference is the warm-start/restore effect.
* drift: each trajectory's hand-relative tool pose against the snapshot relative pose.
* rebuild spread: the snapshot relative pose of env 0 over separately built (and settled) processes.
Grasp tolerances: 2 mm / pi/60 rad (``ToolGraspConfig``).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

TRANSLATION_TOL, ROTATION_TOL = .002, np.pi/60


def relative(body_q, hand, tool):
    h, t = body_q[..., hand, :].astype(float), body_q[..., tool, :].astype(float)
    Rh, Rt = Rotation.from_quat(h[..., 3:]), Rotation.from_quat(t[..., 3:])
    p = Rh.inv().apply(t[..., :3]-h[..., :3])
    return p, Rh.inv()*Rt


def pose_diff(pa, Ra, pb, Rb):
    return np.linalg.norm(pa-pb, axis=-1), (Ra.inv()*Rb).magnitude()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--restore-runs", nargs="+", required=True)
    parser.add_argument("--baseline-run", default=None, help="deterministic run without restore (run-to-run check)")
    parser.add_argument("--restore-step", type=int, default=10)
    parser.add_argument("--window", type=int, default=25)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = {"restore_vs_fresh": {}, "tolerances": {"translation_m": TRANSLATION_TOL, "rotation_rad": ROTATION_TOL}}
    snapshots = {}
    for run in args.restore_runs:
        d = np.load(Path(run)/"states.npz")
        row = json.loads((Path(run)/"row.json").read_text())
        hand, tool = d["hand"], d["tool"]
        k0 = args.restore_step-1           # states[i] = state after step i+1; the restore precedes step restore_step
        w = args.window
        p0, R0 = relative(d["body_q"][:w], hand[0], tool[0])
        p1, R1 = relative(d["body_q"][k0:k0+w], hand[1], tool[1])
        snap = row.get("grasp_snapshot", {})
        Rs = Rotation.from_matrix(np.asarray(snap["relative_R"], float))
        ps = np.asarray(snap["relative_p"], float)
        dt, dr = pose_diff(p0, R0, p1, R1)
        d0t, d0r = pose_diff(p0, R0, ps[0][None], Rs[0])
        d1t, d1r = pose_diff(p1, R1, ps[1][None], Rs[1])
        out["restore_vs_fresh"][run] = dict(
            deterministic=row.get("deterministic_mode"), window_steps=w,
            max_translation_diff_m=float(dt.max()), max_rotation_diff_rad=float(dr.max()),
            restored_equals_fresh_exactly=bool(dt.max() == 0 and dr.max() == 0),
            fresh_max_drift=[float(d0t.max()), float(d0r.max())], restored_max_drift=[float(d1t.max()), float(d1r.max())],
            within_grasp_tolerance=bool(max(d0t.max(), d1t.max()) <= TRANSLATION_TOL and
                                        max(d0r.max(), d1r.max()) <= ROTATION_TOL),
            translation_diff_every_5_steps=[float(v) for v in dt[4::5]],
            grasp_final_status=row.get("grasp_trace_summary", {}).get("final_status"))
        snapshots[run] = (ps[0], Rs[0])
    names = list(snapshots)
    spread_t = [float(np.linalg.norm(snapshots[a][0]-snapshots[b][0])) for i, a in enumerate(names) for b in names[i+1:]]
    spread_r = [float((snapshots[a][1].inv()*snapshots[b][1]).magnitude()) for i, a in enumerate(names) for b in names[i+1:]]
    out["rebuild_spread_env0"] = dict(runs=names, max_translation_m=max(spread_t) if spread_t else None,
                                      max_rotation_rad=max(spread_r) if spread_r else None,
                                      within_grasp_tolerance=bool(spread_t and max(spread_t) <= TRANSLATION_TOL and
                                                                  max(spread_r) <= ROTATION_TOL))
    if args.baseline_run:
        a = np.load(Path(args.restore_runs[0])/"states.npz")
        b = np.load(Path(args.baseline_run)/"states.npz")
        bw = a["body_world"]
        out["deterministic_run_to_run_env0"] = dict(
            bitwise=bool(np.array_equal(a["body_q"][:, bw == 0], b["body_q"][:, bw == 0])),
            max_abs=float(np.abs(a["body_q"][:, bw == 0]-b["body_q"][:, bw == 0]).max()))
        k0 = args.restore_step-1
        out["deterministic_restored_vs_unrestored_env1"] = dict(
            max_abs_body_q_after_restore=float(np.abs(a["body_q"][k0:, bw == 1]-b["body_q"][k0:, bw == 1]).max()),
            note="unrestored env 1 kept holding for restore_step-1 steps; this is the restore jump, not warm start")
    Path(args.out).mkdir(parents=True, exist_ok=False)
    (Path(args.out)/"g3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
