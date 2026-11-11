# P2-6 rubber RL — RL-09 Full Native verification (P6.4 / P6.5)

This file holds the RL-09 section only. P6.6 adds the other P2-6 rows (RL-01 … RL-08, RL-10, RL-11).

Protocol `rubber_rl_tests_v2`, sha256 `6db54681…89548`. Task config `rubber_strip_drag_v2`, sha256 `7395e610…2addb`.
Profile `rubber_rl_env_v2`:

- contact law `grid_node_penalty_coulomb_v2`;
- table friction fixed at 0.3 (labelled limitation `contact_friction_envelope_v2_fixed_0.3`).

Branch `claude/rl-rubber-p26n`. Material status: **SIMULATOR_ONLY_UNCALIBRATED**.

The declarations were pinned at `400f5c5` and merged here in `1615285` before the formal run. An earlier 5-step Kit smoke
under the superseded v1 profile was a pipeline check only. It is labelled smoke and preserved in
`artifacts/rl_ready/p2_6/native_rubber_smoke_001`; it is never a result.

**Verdict: RL-09 PASS. All five declared clauses hold.**

## 1. What was built (P6.4)

| File | Role |
| --- | --- |
| `src/robotic_cutting_sim/native_rubber_host.py` | The Kit host. It uses the same material as the P2-6 env: `RubberStripTaskFamily.build_life`, with the same recipe, slot, profile and contact law. The loop keeps the shape of the TLMPM Native bridge; the steps are listed below. |
| `src/robotic_cutting_sim/rubber_native_compare.py` | RL-09 clauses (a)–(e), pure numpy / scipy (`rubber_l3.evaluate` for (c)). |
| `scripts/rubber/p2_6/launch_native_rubber.py` | Launcher (T5.8 pattern). It refuses with BLOCKED unless 14 GiB plus the 4 GiB reserve are free. Runs the Kit host with `--gpu`. |
| `scripts/rubber/p2_6/run_native_rubber.py` | The Kit run: Native arrays, `rcs_rubber_record_v1`, bridge rows, captures, receipts, nvidia-smi samples. |
| `scripts/rubber/p2_6/replay_native_rubber.py` | Headless GPU replays without Kit. Each repeat builds a fresh life and feeds it the recorded measured poses through the same `advance_measured` call. |
| `scripts/rubber/p2_6/compare_native_rubber.py` | CPU comparison. Writes `rl09.json` and the read-only evidence file, and never overwrites. |
| `scripts/rubber/p2_6/native_meta.py` | RL-10 run metadata and the GPU sharing sampler. |
| `tests/test_native_rubber_host.py`, `tests/test_native_rubber_compare.py` | CPU tests. Their contents are listed in §5. |

The Native loop, per control step:

1. Set the kinematic clamp target with the PhysX tensor `set_kinematic_targets`.
2. Step PhysX: 10 × 2 ms steps, with both Frankas held at the shared safe park.
3. Read the measured clamp pose. It comes from the tensor `get_world_poses` as float32 and is converted to float64 with
   `isaac_sim_runner.pose_from_wxyz`.
4. Convert it to the strip frame (a pure float64 translation).
5. Call `LiveRubberMaterial.advance(motions={0: grip_motion(measured)})` on `cuda:0` inside the Kit process.
6. Update the strip surface with `DynamicSurfaceBridge`. Its vertices are the surface particles under the strip pose.
7. Capture the overview and close-up RGB-D every 10 steps.

Feedback is off: the wrench is reported and never applied. `isaac_sim_runner.py`, `live_material.py`, the Phase 1
frozen modules, `configs/rl_benchmark/*`, `scripts/rl_ready/*` and the rubber protocols are unchanged.

## 2. Run `native_rubber_001` (P6.5)

Episode `native_rubber_episode_v1`, from plan §4.6:

- Slot L125_t4: 125 × 16 × 4 mm, n_t 4, 65,024 particles.
- Material: recipe nominal, friction 0.3, zero traction.
- Steps 0–24: hold.
- Steps 25–49: lift 20 mm and yaw 0.3 rad (smootherstep).
- Steps 50–99: hold. The tip stays on the table and is dragged sideways by the yaw: tip y moves 8.0 → 40.8 mm and tip
  z stays at t/2.

Commands (the T5.8 lock pattern; the chain script is in the session scratchpad):

```bash
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu0.lock .venv/bin/python scripts/rubber/p2_6/launch_native_rubber.py \
    --gpu 0 --task-config rubber_strip_drag_v2 --task-config-sha256 7395e610…2addb \
    --protocol configs/rubber/rubber_rl_tests_v2.json --protocol-sha256 6db54681…89548
flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env CUDA_VISIBLE_DEVICES=1 CUDA_DEVICE_ORDER=PCI_BUS_ID \
    PYTHONPATH=src .deps/isaacsim-venv/bin/python scripts/rubber/p2_6/replay_native_rubber.py \
    --native artifacts/rl_ready/p2_6/native_rubber_001/run --out artifacts/rl_ready/p2_6/native_rubber_001_replays \
    --repeats 30 --gpu-index 1 --task-config rubber_strip_drag_v2 … --protocol configs/rubber/rubber_rl_tests_v2.json …
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .venv/bin/python scripts/rubber/p2_6/compare_native_rubber.py \
    --native artifacts/rl_ready/p2_6/native_rubber_001 --replays artifacts/rl_ready/p2_6/native_rubber_001_replays \
    --out artifacts/rl_ready/p2_6/native_rubber_001_rl09 --evidence docs/evidence/rl_ready/p2_6_native_rubber_001.json
```

The replays first queued on GPU0. On the coordinator's request they were moved to GPU1. GPU1 is the same model
(RTX 5090), which is what plan §4.6 asks for ("same GPU model").

### Native row

| Field | Value |
| --- | --- |
| Run | `artifacts/rl_ready/p2_6/native_rubber_001` (17 MB). Launcher wall 188.9 s; episode 181.5 s; Kit start 4.1 s (warm caches) |
| GPU | #0 NVIDIA GeForce RTX 5090 (PCI 21:00), driver 570.211.01 |
| Render device | GPU0: Kit `active_gpu=0`, `physics_gpu=0`, `multi_gpu` False, RaytracedLighting (`isaac_sim_runner.launch_app`, unchanged) |
| PhysX | CPU, dt 2 ms × 10 per control step, no GPU dynamics, no fabric |
| Material device | `cuda:0`, PCI 21:00, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, no `CUDA_VISIBLE_DEVICES` |
| Free memory at launch | 31,435 MiB (≥ 14 GiB + 4 GiB rule met) |
| Sharing | **SHARED_GPU**: PID 642659 (another project, 766 MiB, untouched) in all 38 samples; GPU util peak 100 % |
| Job VRAM peak | 5,334 MiB |
| Commit / dirty | `1615285`, clean tree (`git status --porcelain` empty for src / scripts / configs / tests) |
| Protocol | `rubber_rl_tests_v2` sha256 `6db5468172e0cf7c507c51b97757550442715ef80221cf818f6fb69cdd189548` |
| Task config | `rubber_strip_drag_v2` sha256 `7395e6104cab16a247dc03afb133f8bd65c96d68a0d01d058bc59e6fbbf2addb` |
| Runtime | Isaac Sim 6.2.0.0; Kit `omni.warp.core` 1.17.0 (the replays use the venv `warp-lang` 1.17.0; same kernel module hashes); Python 3.12.12 |

The throughput row is reported only, never scored:

- N 1, 65,024 particles per env, control 50 Hz, material dt 3.827 µs (5,226 substeps);
- cameras: overview and close-up RGB-D at 960 × 540, every 10 steps (11 captures);
- recording: records, arrays and the surface every step;
- warm step (steps 2–100) 1.81 s, of which material 1.74 s;
- GPU0 shared, warm caches.

### Replays

`artifacts/rl_ready/p2_6/native_rubber_001_replays` (8.2 MB):

- 30 repeats: 0–19 form the band, 20–29 are held out.
- Commit `5e64347`, clean tree.
- GPU1 (PCI F1:00), `cuda:0` under `CUDA_VISIBLE_DEVICES=1`.
- **SHARED_GPU**: the external agentic optimizer's acceptance processes were present.
- Job VRAM peak 714 MiB; 1.72–2.11 s per step; invalid bits 0 in all 30.

Between `1615285` and `5e64347` only the Kit-only `KitRubberScene` changed (the rendering fix in §4) plus scripts. The
replay path (`advance_measured`, `replay_bridge_inputs`, `statistics_row`) is byte-identical.

## 3. RL-09 result (`rl09.json` sha256 `11106fca…2ff7`, evidence `docs/evidence/rl_ready/p2_6_native_rubber_001.json`)

| Clause | Declared | Measured | Verdict |
| --- | --- | --- | --- |
| completeness | 100 Native steps, 20 band replays | 100 / 100 steps, 20 band + 10 held-out | PASS |
| (a) receipts | identical to the headless reference except device / timing | 30 / 30 receipts identical leaf by leaf; exempt leaves only `material/build_wall_s`, `material/graph/capture_wall_s`, `material/solver/device` | PASS |
| (b) inputs | material inputs = recorded measured clamp poses, bitwise float64 | Native strip-frame pose = recorded measured pose minus placement, bitwise. Material pose, motion (R, b) and solver-staged `group_q1` / `group_b1` are bitwise equal in 30 / 30 replays | PASS |
| (c) physics | 46 statistics inside `rubber_l3_band_v4` (20 replays, α_family 0.05) | 0 / 46 outside. α_stat 1.087e-3. Max M/U **0.654** (marker1_z: M 1.8e-7 m, U 2.75e-7 m). Next: marker0_z 0.550, plane_wrench_ty 0.535, plane_wrench_fz 0.489, clamp_wrench_fz 0.477. Bound method: lognormal 45, Cantelli fallback 1 | PASS |
| (c) held-out (reported) | — | 10 / 10 within; max M/U 0.488–0.978 | reported |
| (d) invalid bits | 0 | 0 at all 100 steps (NONFINITE, J_DOMAIN, CFL, S1 self-penetration) | PASS |
| (e) bridge integrity | mesh vertices = surface particles under the strip pose ≤ 1e-6 m | 101 / 101 updates, max 7.45e-9 m, 19,664 vertices, topology equal | PASS |
| camera consistency | reported, not scored | 10 captures × 9 markers per camera, all visible. Rendered depth minus marker optical depth: overview −8.6 … +2.7 mm, close-up −7.8 … +1.6 mm | reported |

How to read the camera residuals:

- The markers are cross-section centroids 2 mm below the top face.
- At the cameras' ≈ 17° elevation, the ray reaches the top surface ≈ 6 mm before the centroid, so the expected
  residual is about −6 mm, and that is what is measured.
- The tip marker sits on the end face, and its ray passes the end onto the table, which gives the positive values.

The command-to-measured clamp distance is at most 2.2e-9 m (the float32 PhysX pose). The Frankas stayed within
1.7e-10 rad of the park pose.

Media (reported, not physics):

- `docs/media/p2_6/native_rubber_001_closeup_steps_0_30_50_100.png`
- `docs/media/p2_6/native_rubber_001_overview_step100.png`

![closeup](../../media/p2_6/native_rubber_001_closeup_steps_0_30_50_100.png)

## 4. Defects found and fixed after the formal run (rendering / reported quantities only; preserved, not re-scored)

1. **Rendered clamp jaw stayed at its reset pose** (visible in the media above).
   - Cause: PhysX does not write kinematic-body poses back to USD. In `native_rubber_001` the jaw cube was a child of
     the kinematic body (`clamp_visual rigid_child_v0`).
   - The material input is the PhysX tensor pose, so no scored clause is affected.
   - Fixed in `071d5bb`: the jaw is now a separate non-physics prim posed from the measured pose after every PhysX step
     (`measured_pose_v1`, now the default).
   - A labelled 100-step visual smoke with the fix was queued, then cancelled to free the shared GPU0 queue. It is
     NOT_RUN.
2. **Camera consistency at step 0 used the uncomputed marker buffer.** All 9 markers projected to one point.
   - Fixed in `5e64347`: step 0 now uses the reference centroids.
   - The `native_rubber_001` summary excludes step 0. Camera consistency is reported only.

## 5. Tests

CPU suites `tests/test_native_rubber_host.py` (12) and `tests/test_native_rubber_compare.py` (7), with `.venv`, run
twice (`CUDA_VISIBLE_DEVICES=''` and `=0`):

- With the P6.5 artifacts present: **19 passed** both times. This includes the recomputation of this evidence from the
  artifacts.
- Without the artifacts: 18 passed, 1 NOT_RUN skip.

What the tests check, without Kit:

- **Bridge**, with a fake PhysX that adds a tracking offset and a fake USD sink:
  - call order;
  - the material receives the measured pose, not the command;
  - motion and staged rows;
  - surface vertices = surface particles ≤ 1e-6 m;
  - the surface mesh is a closed outward 2-manifold with the exact lattice volume;
  - a CPU replay is bitwise, and one ulp is detected.
- **Family profile:** a `RubberStripTaskFamily.build_life` run on the cpu_parity slot.
- **Compare logic:**
  - receipt exemptions and type / sign / NaN rules;
  - one-ulp input detection on every key;
  - L3 band accept, shift-reject, NaN-reject, and a deterministic band;
  - invalid and bridge clauses;
  - end-to-end PASS / FAIL paths.

`tests/test_repo_naming.py` passed before each commit.

## 6. Open

- The rendered-clamp fix (`measured_pose_v1`) has not been exercised in a Kit run yet (visual smoke NOT_RUN).
- Both GPU rows are SHARED_GPU. No EXCLUSIVE Native row exists.
- Robot feedback coupling stays `NOT_IMPLEMENTED_PHASE2`. The clamp is kinematic and its wrench is reported only.
- Friction is the fixed 0.3 of `rubber_strip_drag_v2`, the labelled limitation.
