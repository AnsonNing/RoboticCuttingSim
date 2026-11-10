# P1-5 CUT — CUT in the RL headless path, configured-scene vector env, RL smoke

Date: 2026-11-10. Branch `claude/rl-p1-5-cut` (from `30b8a67`; integration merge of P1-2b/R0/P1-3a `6e1e544` in
`5b98f81`). Plan: [P1_5_PLAN.md](P1_5_PLAN.md) §2 (CUT decision (b) + (c)), §5 (smoke), §7 (T5.1–T5.8, T5.12).
Evidence: [`docs/evidence/rl_ready/p1_5_cut_001.json`](../../evidence/rl_ready/p1_5_cut_001.json). Raw runs:
`artifacts/rl_ready/p1_5/` (git-ignored, `/mnt/data`). Every GPU row is `SHARED_GPU`.

## Status

| Task | What | Status |
| --- | --- | --- |
| T5.1 | Pre-declared `cut_v7_2_v1` (scene `bar_flat_board_tlmpm_v7_2_v1`, goal, planner options, seed, driver, success rule, parity and escalation rules, T5.8 command), `workloads_v2.json` (C1–C3; `workloads_v1.json` unchanged); committed in `cb76a33` before any CUT run | PASS (8 CPU tests) |
| T5.3 | `sever_bond_progress_packed_v1` (`PROVISIONAL_SHAPING_NOT_SUCCESS`), goal-type dispatch | PASS (7 tests; NumPy vs Torch reducer) |
| T5.4 | `reference_command_policy_v1` (planner schedule → `edge_delta_v1`) | PASS (700 commands, residual 3.1e-11 m, digit schedule equal; Torch path) |
| T5.5 | WP-12 recording adapter (`rl/replay_wp12.py`) + v5 source-equivalence audit | PASS (Demo 1: 740 steps, pads verified, reference arrays equal) |
| T5.6 (i) | frozen v5 Demo 1 replay × 4, single_env_headless | PASS — evaluator PASS 4/4 (= Native); final codes = Native in 3/4, 2 particles in 1/4 |
| T5.6 (ii) | frozen single v7_2 reference × 3 (ideal tracking, T5.4 commands) | PASS — evaluator PASS 3/3; the escalation rule did not trigger |
| T5.8 | Native v7_2 CUT episode (Kit, GPU0) + 3 single v7_2 replays | PASS — Native evaluator PASS (2 pieces 0.5/0.5, home PASS); replays PASS 3/3, final codes = Native 1/3 (22, 32 particles in the others); batched replay N=1/N=4 NOT_RUN |
| T5.2 | `ConfiguredTaskFamily` in `open_vector_env`, reward dispatch, CUT CPU L0 | PASS (6 Torch CPU tests incl. shared contract, N=1 bitwise through finite-link events, N=3 bitwise + partial reset) |
| T5.7 | batched CUT N=1/4/8/16, evaluation stream, frozen evaluator per episode | PASS — N=1/4/8/16: 35/35 reference episodes evaluator PASS (36 frames), L2 PASS, final codes within the single band (§6) |
| T5.12 | RL smoke: writer, runner, consumer check; S-CPU, S-CUT N=4, S-TRIM N=8 | PASS — S-CPU (3 tests), S-CUT N=4 and S-TRIM N=8 consumer check PASS; first S-TRIM run had a configuration bug (scripted episodes FAIL, preserved), rerun PASS 4/4 |

CUT feature-parity row: **PARTIAL** as declared (batched CUT = versioned v7_2 scene, evaluator-decided; frozen v5 CUT =
single_env_headless + Native; batched v5 = `UNSUPPORTED_SOLVER`).

## 1. Runtimes, commands, seeds

- CPU tests: `.venv/bin/python -m pytest -p no:cacheprovider` (no Torch; Torch tests skip = NOT_RUN there) and
  `PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest -p no:cacheprovider`
  (Torch 2.11, Warp 1.17.0 CPU device).
- Tests: Torch venv (CPU) `test_rl_{cut_workload,reference_policy,reward_sever,replay_wp12,configured_family,smoke}`, the P1-2/P1-2b vector-env suites, reward/replay/policy/termination tests and the naming test: **92 passed, 2 skipped** (workload B contract needs the Lab venv: NOT_RUN), 800 s. `.venv` RL suite: 189 passed, 23 skipped (Torch-only tests skip there = NOT_RUN), 21 s.
- Headless GPU: `CUDA_VISIBLE_DEVICES=<g> TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaacsim-venv/bin/python`,
  no Kit, each process under `flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu<g>.lock`, gate free ≥ estimate +
  4 GiB, one material (one device `wp.HashGrid`) per process; the census is 0 → 1 → 1 in every run, and warm resets
  release the previous solver first (census 0 then 1).
- Native (T5.8): `flock …/gpu0.lock .venv/bin/python scripts/rl_ready/launch_native_cut.py --gpu 0`, which runs the
  declared command verbatim (`cut_v7_2_v1.json` `native_reference.t5_8_command`: the Demo 1 command with only the
  scene replaced; `--gpu 0` = render, physics and material device; no `CUDA_VISIBLE_DEVICES`). GPU0 free 29,887 MiB
  at launch (≥ 14 GiB rule met); co-tenants PID 642659 (766 MiB, untouched) and 1478763.
- Seeds: sim/material seed 42 everywhere (declared); batched env RNG seeds 1000+e; smoke seeds per `rl_smoke_v1.json`.
- Chains: `/mnt/data/ningan/tmp/p1_5_cut_chain*.sh`, `p1_5_t5_8_replay_chain*.sh`, `p1_5_rest_chain_v2.sh`,
  `p1_5_batched_chain_v2.sh` (logs `artifacts/rl_ready/p1_5/logs/`).

```bash
scripts/rl_ready/run_cut_single.py --mode replay --gpu-index <g>                       # C1: v5 Demo 1 (default)
scripts/rl_ready/run_cut_single.py --mode reference --gpu-index <g>                    # C2 single reference
scripts/rl_ready/run_cut_single.py --mode replay --recording artifacts/rl_ready/p1_5/native_cut_v7_2_001/run/episode_000 \
    --audit-commit= --gpu-index 0                                                       # T5.8 replay
scripts/rl_ready/run_cut_batched.py --num-envs 1 --gpu-index 0 --run-name m0_cut_n1
scripts/rl_ready/run_cut_batched.py --num-envs 4 --random-envs 2 3 --forced-resets 2:150 3:300 --gpu-index 0 --run-name m1_cut_n4
scripts/rl_ready/run_cut_batched.py --num-envs 8 --episodes 2 --gpu-index 0 --run-name m2_cut_n8
scripts/rl_ready/run_cut_batched.py --num-envs 16 --gpu-index 0 --run-name m2_cut_n16
scripts/rl_ready/run_rl_smoke.py --run S-CUT --gpu-index 0 ; scripts/rl_ready/run_rl_smoke.py --run S-TRIM --gpu-index 0
PYTHONPATH=src .venv/bin/python scripts/rl_ready/check_rl_rollout.py <rollout dir>
PYTHONPATH=src .venv/bin/python scripts/rl_ready/make_p1_5_cut_evidence.py ... --out docs/evidence/rl_ready/p1_5_cut_001.json
```

## 2. What was built

| File | Task | Content |
| --- | --- | --- |
| `configs/rl_benchmark/scenes/bar_flat_board_tlmpm_v7_2_v1.json` | T5.1 | line-for-line copy of the frozen `configs/wp12/scenes/bar_flat_board.json`; only `solver.module` (tlmpm_v5 → tlmpm_v7_2) and the description differ (`scene_id` stays `bar_flat_board`) |
| `configs/rl_benchmark/cut_v7_2_v1.json` | T5.1 | `PREDECLARED`: SHA256 pins (scene, frozen source scene, goal `configs/wp14/goals/standard_cut.json`), planner options `{1°, 1 mm, path_v2}`, seed 42, 700 steps, material (v7_2, law 8, K 4, S 8, optimizations on, `frozen_v1` contacts, handle box only), driver, evaluator, success rule (`goals.evaluate_goal` verdict PASS only), reward, parity rules C1/C2, escalation rule, T5.8 command, not-claimed list. Compiled id `CFG-bar_flat_board-sever_mid-e5d82380` (v5 counterpart `…-c32f94a4` = Demo 1) |
| `configs/rl_benchmark/workloads_v2.json` | T5.1 | C1 `cut_v5_native_replay`, C2 `cut_v7_2_batched`, C3 `cut_v7_2_lab_b1` (NOT_RUN); extends `workloads_v1.json` by SHA256 (unchanged) |
| `rl/cut_workload.py` | T5.1 | loader with pin verification, `workload_task_args`, compile of the v7_2 task and of its frozen v5 counterpart, scene diff |
| `rl/reward_sever.py` | T5.3 | `sever_bond_progress_packed_v1`: target = lattice bonds crossing a declared sever/divide plane (600 for the bar), off-target = bonds with midpoint > the goal's frozen 2 mm bound from every plane (271,800), neutral 7,000; the TRIM device reducer reused unchanged; `bonds_for_task` / `reward_class_for_version` dispatch by goal type |
| `rl/reference_policy.py` | T5.4 | `reference_schedule` / `reference_command_actions` (knot k target, digit of the k-1 → k motion; optional window start) and `ReferenceCommandPolicyV1` (device table) |
| `rl/replay_wp12.py` | T5.5 | `extract_wp12_trajectory` (streaming `actions.jsonl.gz`, `runtime/setup.json` payload, `finger_geometry_00.json`, `robot_state.npz`; RECORDED vs RECONSTRUCTED provenance as the WP-17 adapter), `Wp12NativeSamples` (lazy `physics_state.npz`), `source_equivalence_audit` |
| `rl/configured_family.py`, `rl/vector_env.py` | T5.2 | `ConfiguredTaskFamily` (per-slot scene/goal/planner options, per-slot RNG seeds, `from_cut_workload(N)`, `UNSUPPORTED_SOLVER` for non-v7_2 scenes), vector env accepts it; `reward=None` dispatches by goal type (explicit mismatch → `UNSUPPORTED_OPTION`); fidelity against `NATIVE_CUT_V7_2_REFERENCE` (the T5.8 run); sever metrics in the evaluation summary |
| `rl/rollout.py`, `scripts/rl_ready/{run_rl_smoke,check_rl_rollout}.py`, `configs/rl_benchmark/rl_smoke_v1.json` | T5.12 | `rcs_rl_rollout_v1` writer (device chunk buffers, one host copy per chunk, forced-reset marking, manifest with chunk SHA256), smoke runner, consumer check |
| `scripts/rl_ready/{run_cut_single,launch_native_cut,run_cut_batched,make_p1_5_cut_evidence}.py` | T5.6–T5.8 | runners and evidence assembler |

Nothing frozen changed: v1–v7_2 solvers, law 8, `bar_flat_board.json`, `sever_mid`/`standard_cut`, the goal evaluator,
WP-12/13/14/17 records and `workloads_v1.json` are byte-identical (`tests/test_rl_cut_workload.py` pins them). The
WP-12 Demo 1 recording (main checkout) was only read; `actions.jsonl.gz` hashes are asserted unchanged.

## 3. C1 — frozen v5 Demo 1 replay (single_env_headless, T5.5/T5.6 (i))

Source audit (`a960dc6` vs checkout): `tlmpm_v5.py`, `cut_topology`, `fixture_contacts`, `material_config`,
`finger_geometry`, `tool_profile`, `coupled_robot`, `scene_frames`, `goals`, `goal_planner`, `geometry`, `tasks`,
`flat_board_workcell`, `materials_wp12` byte-identical; `live_material.py` (+55 lines) and `scene_config.py` (+42) changed
only for other solvers / research profiles / the read-only `numerical_diagnostics` helper (whose `snapshot` mode equals
the `debug_state()` extrema Native used at `a960dc6`); the Native pad/advance call lines are identical. Replay input:
740 recorded edge poses (700 task + 40 return home), 137 armed steps (207–343), pad boxes reconstructed and verified
(knife body → edge max 2.8e-17 m; reset boxes bitwise); Native pads never touched the material (0 finger-wrench steps).

| run | GPU | evaluator replay / Native (same call) | final codes = Native | labelled digit 0 (Native 28,500) | max pos diff over time | samples with code diff | reaction contact on/off mismatch | invariants |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cut_v5_replay_001 | 1 | PASS / PASS | yes (0) | 28,500 | 3.8e-5 m | 0 | 0 of 740 | PASS |
| cut_v5_replay_002 | 1 | PASS / PASS | yes (0) | 28,500 | 3.5e-5 m | 0 | 0 | PASS |
| cut_v5_replay_003 | 0 | PASS / PASS | **no (2 particles)** | 28,502 | 6.0e-5 m | 42 (steps 330–740, 2 particles) | 0 | PASS |
| cut_v5_replay_004 | 0 | PASS / PASS | yes (0) | 28,500 | 3.4e-5 m | 0 | 0 | PASS |

Evaluator metrics identical to Native in all four: 2 mechanical components, 0.5 / 0.5, frozen 2 mm statistic
1.5 mm, 75 frames, integrity PASS. Total impulse max |diff| 1.1–6.8 % of the episode peak (the reaction is computed,
not fed back). Result: **PASS** (P1-0 rules: reported, not scored; 3/4 bitwise final codes, one run 2 particles apart —
the GPU float-atomic band, same class as P1-0's transient s15 differences).

## 4. C2 single reference — frozen single v7_2, ideal tracking (T5.6 (ii))

`reference_command_policy_v1` commands (700, residual 3.1e-11 m), handle box only, 71 snapshots, `device_reduction_v1`
diagnostics. **No escalation**: the optimized v7_2 severs the bar.

| run | evaluator | components / fractions | 2 mm statistic | labelled | final codes SHA256 | env-steps/s (warm) | min det F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cut_v7_2_reference_001 | PASS | 2 / 0.5, 0.5 | 1.5 mm | 28,380 | 04e280a1… | 5.09 | 0.9395 |
| cut_v7_2_reference_002 | PASS | 2 / 0.5, 0.5 | 1.5 mm | 28,410 | c56c6b28… | 3.82 | 0.9395 |
| cut_v7_2_reference_003 | PASS | 2 / 0.5, 0.5 | 1.5 mm | 28,380 | 04e280a1… | 5.26 | 0.9395 |

Band: 001 and 003 bitwise equal final codes; 002 differs by 30 particles; max final position difference 1.2e-5 m.
Shaping terms at step 700: 600/600 target bonds severed, 0 off-target. All three ran at `5bd358f` with an uncommitted
change to the evidence script only (manifest `git_dirty`; single-env sources hashed in each manifest).

## 5. T5.8 — Native v7_2 CUT (FULL_NATIVE_REFERENCE of C2) and replays

`native_cut_v7_2_001` (git `5d3f28d`, clean; GPU0; 350.7 s launcher wall, episode 313.9 s): task id
`CFG-bar_flat_board-sever_mid-e5d82380` as declared; status PASS_EXECUTION, **evaluator PASS** (integrity PASS,
2 pieces 0.5 / 0.5, 2 mm statistic 1.5 mm), home outcome PASS, plan finished, 740 steps (700 + 40 home), 137 armed
steps 207–343, labelled 28,382. Mean task step wall 0.322 s (material 0.159 s) vs Demo 1 (v5) 0.968 s (material
0.774 s) — both Native, cameras on, SHARED (Demo 1 with other compute apps present). Job VRAM was not sampled (only
before/after `nvidia-smi`): UNVERIFIED.

Replay through the frozen single v7_2 (WP-12 adapter, measured poses + pads):

| run | evaluator replay / Native | final codes = Native | labelled (Native 28,382) | max pos diff | samples with code diff | contact on/off mismatch |
| --- | --- | --- | --- | --- | --- | --- |
| cut_v7_2_native_replay_001 | PASS / PASS | no (32) | 28,410 | 1.7e-5 m | 43 (steps 320–740) | 0 |
| cut_v7_2_native_replay_002 | PASS / PASS | no (22) | 28,404 | 9.8e-6 m | 43 | 1 of 740 |
| cut_v7_2_native_replay_003 | PASS / PASS | **yes (0)** | 28,382 | 5.8e-6 m | 0 | 0 |

T5.8 batched replay (N=1, N=4 copies through `step_measured_edge`): NOT_RUN (the API path is covered by the CPU test
`test_measured_replay_drive_with_pads_on_the_cut_family`; a GPU run was not scheduled in this round).

## 6. T5.2 / T5.7 — batched CUT

CPU (Warp/Torch CPU, 2.5 mm `reduced_fidelity`): batched N=1 is bitwise equal to the frozen single v7_2 for 64 steps
from the first armed knot (blade contact, labels, finite-link completion events > 0); every env of N=3 copies is
bitwise equal for 14 steps through the first labels; a partial reset leaves envs 0/2 bitwise on the continuing single
and makes env 1 bitwise equal to a fresh single after one step; the shared `vector_env_contract_v1` passes on the CUT
family with the evaluation stream.

GPU (GPU0, one process each, `frozen_v1` contacts, evaluation stream every 20 steps → 36 frames per episode,
`strict=False`, env seeds 1000+e):

| run | N × episodes | drives | evaluator (reference episodes) | L2 | final codes vs single band | env-steps/s | s/batch step | own VRAM | foreign SM mean / max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m0_cut_n1_001 | 1 × 1 | reference | PASS 1/1 | PASS | state C (30/60/30 from refs) | 3.25 | 0.308 | 944 MiB | 45 / 93 % |
| m1_cut_n4_001 | 4 × 1 | 2 reference, 2 random, forced resets at 150/300 | PASS 2/2 (random: FAIL_INTEGRITY, incomplete episodes, 0 labels) | PASS | = ref001/003; state C | 6.88 | 0.581 | 1,502 MiB | 28 / 84 % |
| m2_cut_n8_001 | 8 × 2 | reference | PASS 16/16 | PASS | 14 = ref001/003, 1 = ref002, 1 state C | 9.05 | 0.884 | 2,170 MiB | 10 / 76 % |
| m2_cut_n16_001 | 16 × 1 | reference | PASS 16/16 | PASS | 11 = ref001/003, 1 = ref002, 3 state C, 1 within 13 of ref001 | 5.44 | 2.940 | 3,522 MiB | 43 / 98 % |

All 35 batched reference episodes: evaluator PASS (2 pieces 0.5/0.5), final codes within 0–60 particles of every single
reference (out of ~28,400 labelled), 25/35 bitwise equal to a single-reference final state; L2: 0 code-monotonicity
violations, min det F 0.9394, no non-finite state, no slot overflow, no invalid bits, grouped-hash foreign candidates 0
and mass bitwise constant at the audit steps; HashGrid census 1 throughout. Partial reset p95 3.2 ms (M1); full resets
0.16–0.27 s. Throughput is co-tenant dominated (foreign SM means 10–45 %): no N-scaling ratio is claimed from these rows
(N=16 ran under 43 % mean foreign load, N=8 under 10 %).

## 7. T5.12 — RL pipeline smoke

`rcs_rl_rollout_v1` chunks (≤ 256 steps; obs/next_obs/action/reward/terminated/truncated/episode/step_count/seed/drive/
action_status/invalid_reason/termination_reason/reset_kind), `episodes.jsonl`, manifest with chunk SHA256. Policies read
the live env counters. Consumer check items: chunk hashes, shapes/dtypes, finite, actions in box, `obs[t+1] ==
next_obs[t]` bitwise when not done, done ↔ episode increment ↔ step-count restart, `termination_reason` consistency,
returns vs `episodes.jsonl`, NumPy GAE finite, Torch GAE = NumPy, one shuffled minibatch pass, policy terms only.

| run | N × steps | episodes (time limit / invalid / forced) | evaluator (complete scripted episodes) | consumer check | smoke env-steps/s | forced reset p95 |
| --- | --- | --- | --- | --- | --- | --- |
| S-CPU (pytest, 2.5 mm, 10-step episodes) | 3 × 40 | ≥3 / – / ≥1 | evaluated (last_completed) | PASS | n/a | n/a |
| S-CUT (`smoke_s_cut_n4_001`) | 4 × 1,400 | 4 / 0 / 13 | PASS 4/4 (two-frame mode) | PASS (Torch GAE included) | 4.36 | 13 ms |
| S-TRIM first run (`smoke_s_trim_n8_001`, **configuration bug**) | 8 × 1,500 | 4 / 0 / 29 | **FAIL 4/4** — every env started at the planner HOME, 1.06 mm from the WP-17 reset edge the recorded edge deltas were encoded from | PASS | 4.13 | 10 ms |
| S-TRIM rerun (`smoke_s_trim_n8_002`, fix `9a51a86`: recorded-command envs start at the recorded reset edge, as P1-2 M1) | 8 × 1,500 | 4 / 0 / 29 | PASS 4/4 (two-frame mode) | PASS | 4.60 | 13 ms |

Smoke numbers are not benchmark rows. The S-TRIM initial-pose fix is a runner bug fix, not a change of the declared run (`rl_smoke_v1.json` names `recorded_command` drives, whose commands are relative to the recorded reset edge); both runs are kept. Smokes ran at `5bd358f`/`9a51a86` with the evidence script uncommitted (`git_dirty`). Evaluation inside the smoke uses the two-frame mode (no stream: with terminal
capture the stream would double its device memory).

## 8. Benchmark rows

| boundary | workload | warm/cold | N | particles/env | control Hz | material dt (substeps) | cameras | env-steps/s | sim-time/wall | VRAM peak | reset p95 | physics verdict | robot DOF | GPU | sharing (foreign SM mean/max) | source |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| single_env_native | C1-native-demo1 | warm | 1 | 96000 | 50 | 1e-4 s (200) | Native RGB-D rig | 1.033 | 0.0207 | n/a | n/a | PASS | 18 | #0 RTX 5090 | SHARED_GPU | wp12_evidence |
| single_env_native | C1-native-demo1 | cold | 1 | 96000 | 50 | 1e-4 s (200) | Native RGB-D rig | 0.787 | 0.0157 | n/a | n/a | PASS | 18 | #0 RTX 5090 | SHARED_GPU | wp12_evidence |
| single_env_native | C2-native-v7_2 | warm | 1 | 96000 | 50 | 1e-4 s (200) | Native RGB-D rig | 3.107 | 0.0621 | n/a | n/a | PASS | 18 | #0 RTX 5090 | SHARED_GPU | native_cut_v7_2_001 |
| single_env_native | C2-native-v7_2 | cold | 1 | 96000 | 50 | 1e-4 s (200) | Native RGB-D rig | 2.110 | 0.0422 | n/a | n/a | PASS | 18 | #0 RTX 5090 | SHARED_GPU | native_cut_v7_2_001 |
| single_env_headless | C1-v5-demo1-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 1.853 | 0.0371 | 682 MiB | 0.103 s | PASS | 0 | #1 RTX 5090 | SHARED_GPU (48/99 %) | cut_v5_replay_001 |
| single_env_headless | C1-v5-demo1-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 1.824 | 0.0365 | 682 MiB | 0.103 s | PASS | 0 | #1 RTX 5090 | SHARED_GPU (48/99 %) | cut_v5_replay_001 |
| single_env_headless | C1-v5-demo1-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 1.882 | 0.0376 | 682 MiB | 0.102 s | PASS | 0 | #1 RTX 5090 | SHARED_GPU (45/88 %) | cut_v5_replay_002 |
| single_env_headless | C1-v5-demo1-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 1.841 | 0.0368 | 682 MiB | 0.102 s | PASS | 0 | #1 RTX 5090 | SHARED_GPU (45/88 %) | cut_v5_replay_002 |
| single_env_headless | C1-v5-demo1-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 1.950 | 0.0390 | 682 MiB | 0.107 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (46/98 %) | cut_v5_replay_003 |
| single_env_headless | C1-v5-demo1-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 1.916 | 0.0383 | 682 MiB | 0.107 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (46/98 %) | cut_v5_replay_003 |
| single_env_headless | C1-v5-demo1-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 2.200 | 0.0440 | 682 MiB | 0.100 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (40/95 %) | cut_v5_replay_004 |
| single_env_headless | C1-v5-demo1-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 2.147 | 0.0429 | 682 MiB | 0.100 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (40/95 %) | cut_v5_replay_004 |
| single_env_headless | C2-v7_2-single-reference | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 5.092 | 0.1018 | 688 MiB | 0.123 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (32/82 %) | cut_v7_2_reference_001 |
| single_env_headless | C2-v7_2-single-reference | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 4.880 | 0.0976 | 688 MiB | 0.123 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (32/82 %) | cut_v7_2_reference_001 |
| single_env_headless | C2-v7_2-single-reference | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.819 | 0.0764 | 688 MiB | 0.161 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (48/94 %) | cut_v7_2_reference_002 |
| single_env_headless | C2-v7_2-single-reference | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.696 | 0.0739 | 688 MiB | 0.161 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (48/94 %) | cut_v7_2_reference_002 |
| single_env_headless | C2-v7_2-single-reference | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 5.256 | 0.1051 | 688 MiB | 0.129 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (30/81 %) | cut_v7_2_reference_003 |
| single_env_headless | C2-v7_2-single-reference | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 5.026 | 0.1005 | 688 MiB | 0.129 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (30/81 %) | cut_v7_2_reference_003 |
| single_env_headless | C2-v7_2-native-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.509 | 0.0702 | 694 MiB | 0.138 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (56/99 %) | cut_v7_2_native_replay_001 |
| single_env_headless | C2-v7_2-native-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.405 | 0.0681 | 694 MiB | 0.138 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (56/99 %) | cut_v7_2_native_replay_001 |
| single_env_headless | C2-v7_2-native-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.959 | 0.0792 | 694 MiB | 0.152 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (49/98 %) | cut_v7_2_native_replay_002 |
| single_env_headless | C2-v7_2-native-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.829 | 0.0766 | 694 MiB | 0.152 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (49/98 %) | cut_v7_2_native_replay_002 |
| single_env_headless | C2-v7_2-native-replay | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 5.564 | 0.1113 | 694 MiB | 0.145 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (23/61 %) | cut_v7_2_native_replay_003 |
| single_env_headless | C2-v7_2-native-replay | cold | 1 | 96000 | 50 | 1e-4 s (200) | none | 5.294 | 0.1059 | 694 MiB | 0.145 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (23/61 %) | cut_v7_2_native_replay_003 |
| batched_headless | C2-batched-N1 | warm | 1 | 96000 | 50 | 1e-4 s (200) | none | 3.247 | 0.0649 | 944 MiB | 0.243 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (45/93 %) | m0_cut_n1_001 |
| batched_headless | C2-batched-N4 | warm | 4 | 96000 | 50 | 1e-4 s (200) | none | 6.882 | 0.1376 | 1502 MiB | 0.003 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (28/84 %) | m1_cut_n4_001 |
| batched_headless | C2-batched-N8 | warm | 8 | 96000 | 50 | 1e-4 s (200) | none | 9.049 | 0.1810 | 2170 MiB | 0.240 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (10/76 %) | m2_cut_n8_001 |
| batched_headless | C2-batched-N16 | warm | 16 | 96000 | 50 | 1e-4 s (200) | none | 5.442 | 0.1088 | 3522 MiB | 0.268 s | PASS | 0 | #0 RTX 5090 | SHARED_GPU (43/98 %) | m2_cut_n16_001 |

All rows: control 50 Hz, 200 substeps of 0.1 ms; headless rows have no cameras, no robot (DOF 0), no recording in the timed calls; contacts = material tool/handle/fixture (+ two measured pads in the replays); warm = steps 2..T, cold = process start to the last step. Native rows: Demo 1 (`wp12_evidence`, tlmpm_v5, git a960dc6 dirty) and T5.8 (tlmpm_v7_2), job VRAM and driver not recorded (UNVERIFIED), sharing from the process list only. The full 24-column table (contacts, recording, co-tenant statistics, commits, notes) is `markdown_table` in the evidence.

## 9. Feature-parity rows this work moves (from evidence only)

| # | Capability | Before (P1-5 plan §3) | Now |
| --- | --- | --- | --- |
| 2 | CUT (representative) | NOT_RUN | **PARTIAL** as declared: batched = versioned v7_2 CUT, evaluator PASS N=1/4/8/16 (35/35 reference episodes), Native v7_2 reference PASS (T5.8); frozen v5 CUT single_env_headless (4/4 PASS = Native) + Native; batched v5 `UNSUPPORTED_SOLVER` |
| 1 | Scene/goal/material schema | PARTIAL | PARTIAL (labelled): configured v7_2 scenes via `ConfiguredTaskFamily`; other solvers refused |
| 5 | Cut history / persistent severing | PARTIAL | PASS for the solver contract on CUT: multi-frame (36-frame stream) integrity, 0 monotonicity violations, partial reset restores history (CPU bitwise); recorders stay UNSUPPORTED |
| 10 | Performance profiling | PASS | PASS + CUT rows (all SHARED_GPU) |
| 12 | Frozen independent evaluators | PARTIAL | PASS: the sever evaluator decides batched CUT episodes (stream mode); v1 `evaluate_oracle` stays single-env (labelled) |
| 13 | Recording / replay | PARTIAL | PASS for RL data and replay: `rcs_rl_rollout_v1` + consumer check, WP-12 replay adapter; physics recorders UNSUPPORTED |
| 15 | RL pipeline smoke | NOT_RUN | **PASS**: S-CPU, S-CUT N=4, S-TRIM N=8 consumer checks PASS; scripted episodes evaluator PASS (TRIM after the initial-pose fix); no training |

Not moved here: 4 (robot, B2), 6 (cameras), 9 (randomisation), 11 (rubber), 14 (termination reasons: consumed via
`termination_reason`, owned by P1-2b), 7/8 (unchanged).

## 10. Open / not done

- T5.8 batched replay of the Native v7_2 recording (N=1 / N=4 copies vs Native samples): NOT_RUN.
- Reward dispatch in the Lab env (`lab_batched_env`) for CUT and the B1 CUT (T5.9): NOT_RUN.
- `tasks=` items for configured scenes (`resolve_tasks` still refuses non-TRIM selections; per-slot scenes are given
  through `ConfiguredTaskFamily`).
- EXCLUSIVE_GPU CUT rows; Native job VRAM for T5.8 (not sampled).
- The WP-17 seed-512 open discrepancy is untouched; no optimized/unoptimized v7_2 equivalence is claimed.
- Failures preserved: `smoke_cut_batched_n2_001` (KeyError in the manifest step, fixed in `5bd358f`); skipped/stopped
  chain attempts are logged in `artifacts/rl_ready/p1_5/logs/`.
