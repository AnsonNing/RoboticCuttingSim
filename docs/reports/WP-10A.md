# WP-10A/B — policy interface, ruler knife, cutting law 2, kinematic validation

Date: 2026-10-28. Branch `claude/lucid-mendel-vt7n82`. **Scope: CPU only.**
Every GPU, Isaac Sim and Isaac Lab result for this work is **NOT_RUN**.
Plan and remaining work: [WP-10 specification](../WP10_SPECIFICATION.md);
usage: [POLICY_CONTROL.md](../POLICY_CONTROL.md); decisions:
[ADR 0003](../adr/0003-ruler-knife-and-policy-interface.md).

## Environment

Cloud container, 4 CPU cores, no GPU. Python 3.13 venv (the repo targets
3.12) with numpy 2, scipy, warp-lang 1.17.0, newton 1.6.1, usd-core,
gymnasium, matplotlib. Material solver on Warp's CPU backend.

## Tests

- Quick suite (`-m "not gpu and not expensive and not cpu_sim"`): 435 PASS
  (with the save_views test, 436), 2 FAIL in `test_bootstrap.py` only because
  that host has no Python 3.12 interpreter (environment, not code).
- `cpu_sim` suite (`test_cutting_law_v2.py`, `test_kinematic_end_to_end.py`,
  ruler sever test): 23 PASS (about 6 minutes).
- Bitwise: `tlmpm_v2` law 1 with the legacy triangle equals a renamed copy of
  frozen v1 bit for bit on CPU, including a cut (`test_tool_profile.py`).

## Evidence (`docs/evidence/wp10/`)

| File | What | Result |
| --- | --- | --- |
| `dh_reachability_sweep.jsonl` | `scripts/probe_dh_reachability.py` (DH approximation, no collisions) | oblique ±60°, yaw 360°, edge roll ±45° all reachable at the cut point; per-step joint jump ≤ 0.02 rad; yaw −150° leaves 0.16 rad joint margin |
| `v1_angle_probe_cpu.json` | frozen v1 law, legacy wedge, 2 mm | vertical, oblique 20°, yaw 30°, edge roll 15° and a 15° out-of-plane push all sever (the last is a legitimate oblique push); forces < 0.5 N |
| `knife_motions_cpu_001.json` | `scripts/probe_knife_motions.py`, ruler, 4 mm, Gc 5 J/m², law 1 vs 2 | push and oblique push sever under both; sideways push and flat press sever 0 bonds under both; slicing push resistance (z) law 2 0.0038 N vs slow push 0.059 N (law 1: 0.045 vs 0.055); pulling out adds nothing; half cut then sideways: 6 (law 1) / 10 (law 2) severed bonds |
| `kinematic_validation_cpu_001.json` | `scripts/run_kinematic_validation.py`, 4 mm, material dt 0.5 ms, full 8 s episodes, frozen v1 oracle | see below |

Validation run 001 (preserved as is):

| Task | reference | push | saw | gym saw | oblique 20° | yaw 30° | agent API |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CUT-01-R | PASS | PASS | PASS | PASS | FAIL location only | FAIL location only | ERROR (script bug) |
| CUT-02 | PASS | PASS | PASS | PASS | FAIL location only | FAIL location only | ERROR (script bug) |
| CUT-01 (v1 law) | PASS | PASS | PASS | PASS | FAIL location only | FAIL location only | ERROR (script bug) |
| TRIM-01-R | FAIL surface | FAIL surface + persistence | FAIL surface + persistence | FAIL surface + persistence | n/a | n/a | FAIL surface + persistence |

- Oblique/yawed cuts sever the bar; they fail only the frozen check that the
  cut lies on the vertical target plane, as the task defines.
- `agent_api` ERROR on CUT tasks was a validation-script bug (`_plan` looked up
  the agent controller), fixed afterwards; the agent API itself severs CUT-02 in
  `test_agent_api_session_severs_the_bar`.
- TRIM-01-R: the bump material is removed (removal check passes), but the
  strict 0.5 mm surface metrics cannot be met at 4 mm particles (v1 TRIM-01
  also fails them at 1 mm on GPU), and the policies' full-length skim finished
  after the protocol's 5.6 s post-cut window. `policies.skim` now spans only
  the bumps (finishes about 4.2 s). Run 002 with these fixes was started but
  stopped unfinished at handoff: **NOT_RUN — rerun it** (WP-10G).

Kinematic video (not a robot video): `docs/media/wp10_kinematic/cut01r_saw.mp4`
(CUT-01-R, saw policy, oracle PASS).

## Failures found and fixed during the work

- `tlmpm_v2`: the fixture setup overwrote the obstacle array, so the support
  floor was applied a second time in the blade frame and TRIM-01-R diverged
  before tool contact. Renamed, obstacles moved to their own kernel, regression
  test added.
- Law 2 first version opened gaps when the blade plane met a particle column
  (edge segments narrower than the particle spacing, rear window too short);
  fixed (segment = one spacing with neighbor check, v1 rear window).
- Policies sent world-frame deltas to an env defaulting to edge-frame deltas;
  actions may now name their frame.
- Reward v0 counted labels, which one-sided labels could inflate; v1 counts
  severed bonds.
- Earlier diagnosis that v1 wrongly severs a 15° misaligned push was withdrawn.

## Reproduce (CPU)

```bash
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/probe_knife_motions.py --out artifacts/knife_motions.json
PYTHONPATH=src python scripts/run_kinematic_validation.py --out artifacts/kinematic_validation.json
PYTHONPATH=src python scripts/render_kinematic_episode.py --task CUT-01-R --controller saw_policy --out artifacts/saw.mp4
```

## Limits

Kinematic backend has no robot, IK, grasp, slip or cameras; CPU runs used
4 mm particles and 0.5 ms material steps (protocol 1 mm / 0.1 ms); law 2
parameters and all materials are uncalibrated; TRIM-01-R finger clearance was
computed from estimated finger dimensions.
