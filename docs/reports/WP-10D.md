# WP-10D — partial GPU physics/material validation

Completion: incomplete; preserved for Claude at the user's request on
2026-10-29. These are kinematic material probes on a real RTX 5090, not live
robot successes. Frozen v1 physics and scoring are unchanged.

The declared command plan is `artifacts/wp10/d_physics_plan_001.json`; recorded
commands and source snapshots are in `d_physics_batch_001/{graph_check,law,materials}`.
All use seed 42, 1 mm particles, law 2 and nominal 0.1 ms material steps except
the declared material-dt sweep. Source is `02ceca6` plus each receipt's exact
snapshot. [Compact evidence](../evidence/wp10/wp10cd_handoff_v1.json) includes
commands, commits, hashes, case metrics and timings.

## Measured GPU probes

| Probe | Result | Measured values | Artifact under `artifacts/wp10/` |
| --- | --- | --- | --- |
| Declared qualitative law checks | PASS, limited scope | Slice/push vertical resistance ratio 0.037611; side and flat press 0 severed bonds | `d_law_gpu_001` |
| Aligned through-push | PASS | 2 components, 288 severed bonds, mean Fz 0.035799 N | `d_law_gpu_001/push` |
| Two digits | PASS | 3 actual mechanical components, 576 severed bonds; explicit multi-cut outcome PASS | `d_law_gpu_001/two_digits` |
| Oblique ±20/±40, yaw 30/60, roll ±15 | PASS separation in these probes | 2 components each; per-case force and plane error in evidence | `d_law_gpu_001` |
| Saw / through-slice | PASS | 2 components, 288 bonds each; mean Fz 0.000220 / 0.000203 N | `d_law_gpu_001/{saw_through,slice_through}` |
| Skim | FAIL separation | 1 component despite 1,006 severed reference bonds | `d_law_gpu_001/skim` |
| CUDA graph / direct position comparison | FAIL at frozen 1 µm tolerance | Same cut codes, 6,816 labelled particles; max position difference 5.375594 µm | `d_graph_check_gpu_001` |
| Two new food-profile dt sweeps | PASS finite-state screen only | All 8 cases finite; force/dt convergence is not established | `d_materials_gpu_001` |

The law summary's PASS is not a complete reproduction of every assertion in
`tests/test_cutting_law_v2.py`. The half-cut withdrawal probe disables cutting
on withdrawal; the required armed-withdrawal negative control remains
**NOT_RUN**. The blunt-handle probe cuts zero bonds but mean Fz is 0.271506 N,
below the CPU test's 0.5 N assertion. Law 1 was not rerun through this GPU
matrix. Keep these gaps visible when extending the runner. CUDA atomic-order
effects are a possible explanation for the graph difference, not a verified
diagnosis; do not relax the existing tolerance.

## Provisional materials

`configs/materials/food_profiles_v1.json` adds cream-cheese-like E = 26 kPa and
cheddar-like E = 480 kPa profiles. Their stiffness orders are motivated by
[Hutchings et al., Table 3](https://doi.org/10.1111/jtxs.12863). Density,
plastic yield, fracture energy, friction and Poisson ratio are assumptions,
not identified food parameters. See [ADR 0005](../adr/0005-wp10-workcell-materials-and-experimental-law.md).

At dt = 100 / 50 / 20 / 10 µs, measured cream-profile mean Fz is
0.029952 / 0.047321 / 0.062959 / 0.073979 N (two parts in each case).
Cheddar-profile mean Fz is 0.223718 / 0.194483 / 0.167134 / 0.319067 N;
only the 20 and 10 µs cases separate into two parts. Nominal selected dt is
100 µs for cream and 20 µs for cheddar, with elastic-wave grid CFL 0.1410 and
0.1212. These are finite-state/stability screens, not force convergence or
calibrated-food validation. Raw force curves and physical states are retained
as NPZ per case. Outer recorded wall times are 3.17 / 43.59 / 50.17 s for graph,
law and material jobs respectively.

The new read-only `multi_cut_evaluation.py` uses an explicit two-plane goal and
the actual conservative compatibility graph, including stress gradients and
code-zero bridges. Five CPU checks pass, including refusal of incomplete
one-sided labels. It does not replace the frozen v1 oracle. Live two-cut
execution and the full articulated motion matrix are **NOT_RUN**.

Experimental law 3 is preserved in `tlmpm_v3.py` with explicit selection only.
It changes swept-edge side classification and one-sided support closure.
CPU kernel compilation passes, but no law-3 physical rollout, negative-control
test or task score has been measured. It must not be used as established
cutting proof. See [handoff](../HANDOFF_WP10_CLAUDE_20261029.md).
