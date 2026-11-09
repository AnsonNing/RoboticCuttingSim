# ADR 0022 — Workload B (P1-1b): batched Franka articulation on Isaac Lab Newton, coupled to the batched v7_2 material

Date: 2026-11-10. **Status: accepted (orchestrator, 2026-11-10)**; originally proposed by the xhigh design analysis.
Acceptance note: stage S0 (CPU only) may run in parallel with workload A; GPU stages S1+ start after workload A M2.
Cameras in the kit-less batched path are UNSUPPORTED here; a vision profile for the batch needs its own decision
(P1-5 camera row stays PARTIAL until then). This ADR is the "own ADR" for robot batching that
ADR 0021 §9 and plan P1-1b require. Acceptance needs the stage gates in §9.

Nothing here changes a frozen solver, profile, threshold, evaluator or record. The following stay byte-identical:
v5 to v7_2, law 8, the v1 evaluator, evaluation v2, final_v4, the camera rigs v1 to v5, WP-11 to WP-17 evidence,
`isaac_lab_env.py`, `isaac_sim_runner.py` and `newton_tool.py`. Everything new is versioned and opt-in, under
`robotic_cutting_sim.batched` / `robotic_cutting_sim.rl` (D-04). Material results stay `SIMULATOR_ONLY_UNCALIBRATED`.

Path abbreviations used in the citations:

| Short | Path |
|---|---|
| `IL/` | `.deps/IsaacLab/source/` (v3.0.0, `e3f7d9f`) |
| `NW/` | `.deps/newton/newton/_src/` (1.6.1) |
| `MJW/` | `.deps/isaaclab-venv/lib/python3.12/site-packages/mujoco_warp/_src/` (3.12.0) |
| `FR/` | `artifacts/workcell_asset_cache/Isaac/IsaacLab/Robots/FrankaEmika/payloads/Physics/` |
| `RCS/` | `src/robotic_cutting_sim/` |

## 1. Context (verified)

### 1.1 Why the single Lab env cannot be batched by raising `num_envs`

| Fact | Evidence |
|---|---|
| Robots and tool use fixed `/World/Franka_{i}` and `/World/Tools/CuttingTool_0` paths; `num_envs=1` | `RCS/isaac_lab_env.py:221,260,263`; `RCS/newton_tool.py:15,113` |
| Under `separate_worlds`, MuJoCo forbids bodies and joints in the global world −1. Assets outside `{ENV_REGEX_NS}` are shared, so these paths cannot be replicated | `NW/solvers/mujoco/solver_mujoco.py:9483-9497`; `IL/isaaclab/isaaclab/cloner/replicate_session.py:253-256` |
| Tool authoring hard-codes `/World/Franka_0/PhysicalGripMaterial` and `"Franka_0/"` lookups | `RCS/newton_tool.py:25-26,52-53,85-92,201-213,228-231` |
| Global body indices; the edge lever arm is hard-coded to `wp.vec3(0,0,.18)` | `RCS/isaac_lab_env.py:44,333-337` |
| IK is one problem over the whole model, with host round trips | `RCS/isaac_lab_env.py:427,449,453,461` |
| `measured()` and `_pre_physics_step` copy to the host; the action read is `actions[0]` | `RCS/isaac_lab_env.py:477-488,733` |
| `NewtonTool.post_substep` copies about 20 arrays to the host every 2 ms, so it cannot be captured; hence `use_cuda_graph=False` | `RCS/newton_tool.py:252-284`; `RCS/isaac_lab_env.py:297` |
| Reset refreshes all worlds (`_update_mjc_data` and `fwd_position` over the whole `mjw_data`) and settles the grasp by stepping the whole sim | `RCS/isaac_lab_env.py:549-551,610-648` |
| No v5+ solver selection; compiled `ConfiguredTaskSpec`s are refused on Lab. **Seeded TRIM / v7_2 cannot run on the single Lab env** | `RCS/isaac_lab_env.py:159,188`; `RCS/live_env.py:660-662`; ADR 0021:446-447 |
| `legacy_fixed_tool` is refused for ruler (profile) tasks | `RCS/isaac_lab_env.py:179-180` |

### 1.2 Upstream facts (read from source)

**Cloning and scene**
- `InteractiveScene` plans clones with `ReplicateSession`. Origins come from `grid_transforms(N, env_spacing)`
  (`IL/isaaclab/isaaclab/scene/interactive_scene.py:146-169`; `IL/isaaclab/isaaclab/cloner/clone_plan.py:141-165`).
  `filter_collisions` is PhysX-only (`interactive_scene.py:171-173`).
- **Kit-less** runs author only the prototype (`env_0`) and the shared prims in USD. USD replication is added only
  with Kit (`replicate_session.py:97-98`). The prototype path is `template.format(first_world)` (`clone_plan.py:249`).
- Newton builds one builder per source, puts shared sources into world −1, and replicates the worlds natively with
  `builder.replicate` (`IL/isaaclab_newton/isaaclab_newton/cloner/newton_clone_utils.py:334-400`;
  `.../cloner/replicate.py:131-134,258-264`).
- Different worlds never collide; world −1 collides with all of them (`NW/geometry/broad_phase_common.py:256-282`).
  Newton recommends keeping all worlds at the origin (`NW/sim/builder.py:3150-3153`).

**MuJoCo-Warp solver**
- `separate_worlds` defaults to `world_count > 1` (`solver_mujoco.py:4021-4022`). `nconmax` and `njmax` are per world
  (`:3740-3748`).
- The contact buffer is **one shared pool** of `naconmax = nconmax·nworld`, so "one world may have more than nconmax
  contacts" (`MJW/io.py:1725-1726,1766`).
- `SolverMuJoCo.reset(world_mask, flags=0)` zeroes `qacc_warmstart`, `qfrc_applied`, `ctrl`, `act` and
  `xfrc_applied` **only in the flagged worlds** (`solver_mujoco.py:4181-4304`).
- After each step, `_update_newton_state` runs `eval_fk`, so `body_q` is the post-integration pose (`:5207-5269`).

**Isaac Lab Newton manager and DirectRLEnv**
- The manager calls a registered **state-force callback before every solver substep** and a **post-step callback
  once per `step()`**, both inside the captured graph. Staged `body_f` is re-applied every substep
  (`IL/isaaclab_newton/isaaclab_newton/physics/newton_manager.py:513-525,1674-1739,1772-1787,1920-1951`).
  `use_cuda_graph` defaults to True (`.../newton_manager_cfg.py:180`).
- With only `ImplicitActuatorCfg`s, the Newton actuator path is not activated
  (`.../assets/articulation/actuator_control.py:51-63`), so `_is_all_graphable()` and `handles_decimation()` are False
  (`newton_manager.py:1857-1868,2007-2013`). The graph then covers **one 2 ms substep**
  (`_simulate_physics_only`), and `DirectRLEnv` loops it `decimation` times
  (`IL/isaaclab/isaaclab/envs/direct_rl_env.py:406-416`). Post-step callbacks therefore fire every 2 ms.
- Writes through `_index` / `_mask` raise per-world FK and solver-reset masks (`newton_manager.py:1058-1101`;
  `.../articulation/articulation_data.py:167-207`; `.../rigid_object/rigid_object_data.py:128-162`). The next
  `forward()` resets only those worlds (`newton_manager.py:642-663,1562-1578`; `.../physics/mjwarp_manager.py:104-135`).
  Mask writes are the graph-safe form (`.../articulation/articulation.py:1172-1175,1231-1303`).
- `DirectRLEnv` calls `_get_dones` before `_get_rewards` (`direct_rl_env.py:423-425`). Auto-reset uses `nonzero()`
  (a host sync); it can be overridden to return `None` when `compute_final_obs=False` (`:660-682`).
- Determinism: `NewtonCfg.deterministic_mode="run_to_run"` is supported for MJWarp on the GPU when
  `disable_sensors=True` (`newton_manager.py:1448-1516`; `newton_manager_cfg.py:190`).

**IK and wrenches**
- Isaac Lab's Newton IK action solves batched LM IK on the **finalized single-asset prototype model** with
  `n_problems=num_envs`, entirely in Warp (`IL/isaaclab_newton/isaaclab_newton/envs/mdp/actions/newton_ik_actions.py:150-197,300-316`;
  `.../controllers/ik/newton_ik_solver.py:42-80`).
- LM damping is per row (`NW/sim/ik/ik_lm_optimizer.py:298`). `IKSolver.step` only launches kernels
  (`NW/sim/ik/ik_solver.py:348-390`).
- Wrench composers turn world wrenches into body-frame buffers **once per env step** and stage them unchanged across
  the substeps (`.../articulation/articulation.py:388-439`; `IL/isaaclab/isaaclab/utils/wrench_composer.py:80-88`).

**Franka finger coupling**
- The official asset's `mujoco` variant turns `panda_finger_joint1` into an `MjcEqualityJointAPI` targeting
  `panda_finger_joint2` and deletes its USD drive (`FR/mujoco.usda:99-109`). `physics.usda` also carries
  `NewtonMimicAPI` (`FR/physics.usda:162-171`).
- The Isaac Lab skill states that MJWarp drives one leader and moves the follower through the equality
  (`.deps/IsaacLab/.claude/skills/isaaclab-transferring-policies-sim-to-sim/reference.md:30-37`).

**No upstream batched TLMPM.** Isaac Lab's MPM manager wraps Newton's implicit MPM, and a subset-of-worlds reset is
not supported for shared grids (`IL/isaaclab_newton/isaaclab_newton/physics/mpm_manager.py:343-353`). ADR 0021 §9
already concluded this; the batched Warp TLMPM (`RCS/batched/material_batch.py`) stays the material.

### 1.3 CPU probes run for this ADR (development evidence, not acceptance)

- Runtime: `CUDA_VISIBLE_DEVICES='' .deps/isaaclab-venv/bin/python -I <scratch script>` (Warp 1.17.0, Newton 1.6.1,
  MJWarp 3.12.0, NumPy 2.5.2; Warp CPU device).
- The scripts were ad hoc and are not retained. Stage S0 (§9) turns them into tests.
- The model is the official Franka (`mujoco` variant, convex hulls) built as in `RCS/robot.py:108-140`, with
  replicated worlds and a global table.

| Probe | Result |
|---|---|
| Sizing at N=1/16: two Frankas plus a free box tool per world | 23 bodies, 31 shapes, 25 coords and 24 DOFs per world. MJWarp data is **0.224 MB/world** (largest array: `efc.J [N,1024,24]`, 98 KB/world); the model is 0.37–0.54 MB in total; Newton state is 1.9 KB/world. `naconmax = 256·N` |
| Finger coupling | One MuJoCo joint equality per world (`neq=1`, type joint, polycoef `[0,1]`) and `nu=8` actuators per world. USD drive mode is POSITION on finger 1 and NONE on finger 2 |
| Isolation, 3 worlds, 20 substeps; world 0 moves, world 1 is perturbed, world 2 gets `reset(world_mask=[F,F,T,F], flags=0)` | World 0 stays **bitwise** equal in both cases |
| N=1 (default `separate_worlds`) against world 0 of N=3 (`separate_worlds=True`), 60 substeps, 6–7 active contacts | **Bitwise equal** (max \|Δq\| = 0) |
| World 1 driven toward the table | World 0 stays bitwise equal; world 1 changes |
| B1 mount: the tool as a separate body under the Franka root, with a `UsdPhysics.FixedJoint` to `panda_hand` (`add_usd`, 2 worlds, 50 substeps) | Body per world with FIXED joint (parent = hand); authored mass of 0.08 kg kept; **no extra DOF** (9/world, MJWarp `nv=9`); hand-to-tool offset held to 5e-8 m |

Startup measured earlier for N=1 kit-less, two Frankas (main-checkout logs, untracked):
- `artifacts/isaac_lab_cut01_001.log:11,19-21`: scene creation 0.38 s; simulation start 1.08 s, of which solver
  init 0.96 s.
- `artifacts/wp08_lab_physical_trim_v3_001.log:11,19-21` (physical grasp): 0.51 s and 1.30 s, of which solver init
  1.16 s.

## 2. Decision

1. **Runtime.** One process: `IsaacLabBatchedCuttingEnv(DirectRLEnv)` on Newton + MuJoCo-Warp with N worlds, coupled
   on the same device stream to `BatchedCuttingMaterial` (ADR 0021).
   - Rejected: N Kit processes, a Python loop over single envs, and PhysX GPU (no cut-capable coupling exists; Native
     is PhysX **CPU**, `RCS/isaac_sim_runner.py:603-606`).
2. **Scene.** Frankas and tool live under `{ENV_REGEX_NS}`; the static workcell is shared (world −1).
   - **`env_spacing = 0`**: every world sits at the origin, so `T_world_local[e]` is the task's own transform (§3.3).
   - The rigid scene must be uniform per batch.
3. **Control.** The action is `edge_delta_v1` on the device commanded-pose state, the same object as workload A
   (`RCS/rl/actions.py:153-200`).
   - The edge target goes through batched LM IK on the Franka prototype model, with **the same objectives as Native
     and Lab** (`RCS/robot.py:166-176`; `RCS/isaac_lab_env.py:421-429`) and `n_problems=N`.
   - Joint targets go through `actuators.target_command` mask writes. No host sync.
4. **Tool.** Stage B1 first: a rigid mount, the ruler-knife v2 body on a FixedJoint to `panda_hand`, labelled
   `reduced_fidelity: rigid_tool_mount_no_grasp`. Stage B2 next: the physical grasp with a free tool body.
   - Both stages use one wrench receiver (the tool body), one frame chain and one graph-safe per-world wrench kernel,
     run in a state-force callback.
5. **Coupling.** The ADR 0021 §7 semantics are unchanged:
   - Material step k reads the edge pose measured at the end of rigid interval k.
   - Its reaction is applied at the current edge point in every 2 ms substep of interval k+1.
   - The two are serial; nothing overlaps.
6. **Reset.** Masked per-world rigid writes plus MJWarp per-world solver reset plus ADR 0021's masked material reset.
   B2 restores a per-slot settled-grasp snapshot taken once at build; re-settling one world is UNSUPPORTED.
7. **Parity.**
   - Claimed against Native: material only, on identical edge and pad sequences.
   - Batching correctness is proven bitwise on the CPU, and on the GPU in deterministic mode or else within a
     run-to-run band.
   - Robot motion, grasp and success are **not** claimed equivalent to Native (§8).

## 3. Scene (Q1)

### 3.1 Prim layout

| Asset | Prim path | Newton world | Config |
|---|---|---|---|
| Franka 0 (cutting arm) | `{ENV_REGEX_NS}/Franka_0` | e | Same USD, variants, `NewtonArticulationCfg(self_collision_enabled=True)`, gravcomp, actuators and init pose as `RCS/isaac_lab_env.py:214-241`. New spawn func: B1 adds the tool body and FixedJoint; B2 adds the finger grip material, both inside this subtree |
| Franka 1 (parked) | `{ENV_REGEX_NS}/Franka_1` | e | Identical to today, held at `SECONDARY_PARK_JOINTS` (`RCS/workcell.py:22-23`), the shared normal safe park |
| Tool (B2 only) | `{ENV_REGEX_NS}/Tool` | e | `RigidObjectCfg`; free joint; ToolSpec mass, inertia and shapes; grip material inside the tool subtree; `CuttingEdge` Xform |
| Table, board and fixtures | `/World/Workcell/...` (shared) | −1 | Static `CuboidCfg` colliders built from `task.table_top_world_z_m` and `task.fixture_boxes_local` exactly as `RCS/isaac_lab_env.py:267-275`. Flat-board boards come through `fixture_boxes_local` (`RCS/scene_config.py:536-537`) |

**Configuration changes relative to `make_lab_cfg`.**
- `InteractiveSceneCfg(num_envs=N, env_spacing=0.0, replicate_physics=True)`.
- `NewtonCfg(use_cuda_graph=True, load_visual_shapes=False, gap .001)`.
- `MJWarpSolverCfg(iterations=50, nconmax=256, njmax=1024, integrator="implicitfast", cone="elliptic")`. These are
  the same numbers as today; only the graph changes.
- An RCS subclass of `MJWarpSolverCfg` adds `separate_worlds=True`, so N=1 and N>1 take one code path. The kwarg
  filter forwards it (`newton_manager.py:1441-1445`), and the probe found the two modes bitwise equal.
- `sim.dt=.002`, `decimation=10`.
- `episode_length_s` comes from the task (24 s for TRIM), not 8 s.
- **No** `_setup_scene` copy of cameras or lights, and no `_bind_visuals`: there are no USD prims for worlds ≥1
  kit-less (§1.2).

**Tool authoring.** A new `author_batched_tool(stage, root, spec, pose, grasp, finger_root)` reproduces
`author_newton_tool` with path parameters. The frozen module is not edited. Test C4 checks the two produce the same
attributes after path rebasing.

**Pads.** `discover_pads` runs once on the `env_0` prototype (`RCS/finger_geometry.py:12-24`); the pad corners are
the same in every world.

### 3.2 Collision filtering

- Separate MuJoCo worlds never produce cross-world contacts. Shared statics collide with every world
  (`broad_phase_common.py:256-282`; probe §1.3).
- Self-collision inside a world, including Franka 0 against Franka 1, is unchanged.
- In B1, the finger–tool shape pairs are filtered: there is no grasp, and the welded handle would otherwise be
  squeezed. Pads are held at the nominal grasp opening (§5.1).

### 3.3 World origins and the material frame

- `T_world_local[e] = T_world_origin[e] · task_e.world_from_workpiece`. With `env_spacing = 0`, `T_world_origin = I`.
  Every world then uses the same float32 coordinates as the single Lab env, whose N=1 origin is also 0
  (`clone_plan.py:150-154`).
- So N copies see identical numbers, and the material's `T_local_world[e]` (ADR 0021 §3.2) is the task's own
  transform.
- Other spacings change float32 rounding and break the copy and parity tests, so they raise `UNSUPPORTED_ENV_SPACING`.
- The material never sees other worlds: its fixture boxes and table plane are per-env, task-local
  (ADR 0021 §3.2, "fixture boxes").

### 3.4 Headless, one process

- **Launch.** `launch_simulation(cfg, {"headless": True, "visualizer": []})` starts no Kit and no launcher for kit-less
  configs (`IL/isaaclab/isaaclab/app/sim_launcher.py:595-660`; used by `scripts/run_isaac_lab_cutting.py:146`).
- **What is loaded.**
  - Torch, Warp, Newton, MJWarp and Isaac Lab Python packages.
  - An in-memory USD stage (usd-exchange).
  - One Newton builder per source (2 Frankas, plus the tool in B2), replicated N times, then `finalize`.
  - `SolverMuJoCo` (`put_model` / `put_data`).
  - Graph capture after the first eager step (`newton_manager.py:760-765`).
  - The batched material pack (CPU frozen solvers, 0.09–0.16 s per env, ADR 0021:53) and its graph.
- **Startup.**
  - Measured at N=1: about 1.5–1.8 s for scene and simulation start (§1.3).
  - N-dependence: UNVERIFIED but expected small (one replicate pass); measured by G0/G2 (§9).
  - Imports and the Warp kernel cache are reported separately (cold versus warm).
  - B2 adds a 2 s simulated settle (1000 substeps of all worlds at once) per build.

## 4. Control (Q2)

1. **Action contract.** The action is exactly `EdgeDeltaActionV1.apply(actions[N,8], commanded[N,7] float64)`
   (`RCS/rl/actions.py:196`).
   - At reset, `commanded` is initialized to the **nominal hand-mount edge**, `hand FK ∘ task.hand_from_edge`. This
     is deployable (`RCS/control.py:151-160`; `RCS/live_env.py:98,140-144`).
   - It is updated to the target only when IK succeeds; otherwise the env holds (`RCS/live_env.py:225-248`).
   - The measured free-tool pose (B2) is privileged. It drives the material, never the action reference or the policy
     observation.
   - IK rejection is the only addition to the workload A contract. It maps to the frozen status
     `REJECTED_IK_UNREACHABLE` (`RCS/policy_env.py:49-52`) and closes `spec()["not_implemented"]["ik_rejection"]`.
2. **Batched IK.** `newton.ik.IKSolver(model=prototype, n_problems=N)`.
   - The prototype is Franka 0, from `NewtonManager._cl_protos[source].finalize()` as Lab's IK action does it.
   - The objectives are those of `RCS/robot.py:166-176`: position plus rotation on `panda_hand`, with
     `link_offset = hand_from_edge`, joint-limit weight 10, `lambda_initial=.01`, analytic Jacobian, 48 iterations
     (`ik_iterations`), one seed.
   - The seed is the live `joint_pos[N,9]`. With `env_spacing=0` the target frame equals the prototype frame.
   - Problems are independent: λ is per row.
   - The FK of the solution runs on an N-replicated copy of the prototype. A device kernel then computes the
     position and rotation residuals and writes `ik_status[N]`, using the existing 1 mm / `ik_rotation_tolerance_rad`
     gate (`RCS/isaac_lab_env.py:462-469`).
   - Rejected alternatives:
     - **one big IK over the batched model**: one λ couples all envs;
     - **`isaaclab.controllers` differential IK**: one Jacobian step per control step changes the IK semantics that
       Native and Lab share;
     - **host RCS IK** (`DualFrankaController`): host-side and n=1.
3. **Joint targets.**
   - Arm targets are clipped to the limits (`RCS/isaac_lab_env.py:456-458`).
   - Velocity feed-forward is `(q* − q_prev)/0.02` per env when enabled (`:756-757`); fingers get 0.
   - Franka 1 gets the park targets (`:750-760`).
   - All are written with `robot.actuators.target_command.set_position_mask` / `set_velocity_mask` from device
     tensors.
   - The gripper stays in "hold" (`grasp.closed_target_m`) in B2. In B1 it holds the nominal opening (§5.1). There
     is no gripper channel in `edge_delta_v1`.
4. **Mimic fingers.** B keeps the single Lab env's finger configuration unchanged: the `mujoco` variant equality and
   `ImplicitActuatorCfg("panda_finger_joint.*")` (`RCS/isaac_lab_env.py:237-241`).
   - **UNVERIFIED:** whether that config makes MJWarp drive one finger or both. The probe sees one actuator from USD
     defaults; Lab then sets gains on both DOFs. This differs from the skill's "one drive" advice.
   - B inherits this behaviour and does not re-tune it. Test T1 compares, per world, `joint_target_mode/ke/kd`,
     effort limits, `mjw_model.nu`, actuator gains and `neq` against the single Lab env's world 0, so the batch cannot
     silently change grip physics.
   - Native PhysX differs here (two drives, PhysX mimic), as ADR 0021:458-462 already records.

## 5. Tool, frames and wrenches (Q3)

### 5.1 Two mounts

| | B1 `rigid_mount_v1` | B2 `physical_grasp_v1` |
|---|---|---|
| Tool body | ToolSpec shapes, mass and inertia on a separate body; `FixedJoint(panda_hand → tool)` at `ToolSpec.hand_from_tool` (probe §1.3) | Same body, with a free joint, authored as `RCS/newton_tool.py:19-100` does with paths parameterized |
| Edge (material input) | `body_q[hand[e]] ∘ T_hand_tool ∘ T_tool_edge`. A test asserts it equals `hand ∘ task.hand_from_edge` | `body_q[tool[e]] ∘ T_tool_edge` (`RCS/physical_tool.py:297-299`) |
| Fingers | Held at the nominal grasp opening; finger–tool pairs filtered | `closed_target_m`, PD with 2000 N/m and 20 N limit, friction 0.6 (`RCS/physical_tool.py:20-36`) |
| Initialization | Declared TRIM HOME joints, no settle | Declared HOME, then a 2 s settle of all worlds at build, per-world PASS gates, snapshot (§6.2) |
| Fidelity | `reduced_fidelity: true`, reasons `rigid_tool_mount_no_grasp`, `finger_opening_fixed_nominal` | Not reduced. Difference noted: `grasp_restored_from_settled_snapshot` until §8 G3 measures it |

Both modes record `rigid_engine: newton_mjwarp` against the Native `physx_cpu` reference as a listed difference
(not a reduction).

### 5.2 Frame chain and lever arm (replaces the 0.18 m constant)

- The edge pose comes from a device kernel that computes in float64 from float32 `body_q`, cast as in ADR 0021's
  `pose_interp` notes: bitwise to the frozen NumPy compose. It writes `measured_edge[N,7]`, `nominal_edge[N,7]` and
  the pad boxes.
- **Material wrench.** The world wrench `(f, τ_edge)` acts at the **current** edge point `p_e`, on the receiver = tool
  body:

  `τ_com = τ_edge + (p_e − c_com) × f`, with `c_com = body_q[tool] ∘ body_com[tool]`.

  This is the `apply_tool_wrench` pattern (`RCS/newton_tool.py:103-122`) and Native's per-substep application at the
  current edge (`RCS/isaac_sim_runner.py:1537-1543`). B1 needs no special lever: the edge offset is
  `T_tool_edge.translation` in both modes.
- **Pad wrenches.** `pad_wrench[N,B,6]` from the material (`RCS/batched/material_batch.py:11-13`) goes to the two
  finger bodies, also at `p_e`: `τ + (p_e − c_finger) × f`. This matches `RCS/isaac_lab_env.py:52-63` and Native
  `:1544-1547`.
- **Pad boxes.** Each step, a device port of `pad_boxes_edge` (`RCS/finger_geometry.py:41-49`) writes the handle box
  plus 2 pad boxes into the material's `box_centers/halves[N·3]` before `advance`. This is the WP-17
  `handle_and_measured_pads_v1` input; 3 boxes as Native.

### 5.3 Graph-safe per-world application

- One kernel, `dim=N`, runs in the registered **state-force callback**, so it executes before every 2 ms substep and
  inside the graph.
- It reads `pending_wrench[N,6]`, `pad_wrench[N,3,6]`, `external_wrench[N,6]` and index tables `tool[N]`,
  `finger[N,2]`, and adds into `state.body_f` at per-world global indices.
- Indices are resolved once from labels (`/World/envs/env_{e}/Franka_0/.../panda_leftfinger`) and asserted
  `body_world[idx[e]] == e`. The probe found the tool body interleaved in the link order, so arithmetic layouts are
  not trusted.
- It also accumulates per-env applied impulse and pre-step hand wrench (`[N]` versions of `RCS/newton_tool.py:117-122`).
- **Rejected: the wrench composer.** It freezes the body-frame wrench at the step start and stages it unchanged for
  10 substeps (§1.2), whereas Native and Lab recompute the lever arm every substep.

## 6. Partial reset (Q4)

1. **Rigid, per world.** Masked writes:
   - `write_joint_state_to_sim_mask` for both Frankas, and `target_command.set_*_mask`;
   - in B2, `write_root_pose/velocity_to_sim_mask` for the tool;
   - `scene.reset(env_ids)`, which resets actuators and composers per env.

   These set `_world_reset_mask` / `_fk_reset_mask` only for the selected worlds. The next `forward()` runs masked FK
   and the per-world MJWarp buffer reset (§1.2).
   - Contacts carry no state across steps: MJWarp recollides every step from `qpos`, which is synced from the Newton
     state with `update_data_interval=1`.
   - Probe evidence: a masked reset of world 2 left world 0 bitwise unchanged on the CPU (§1.3).
2. **Forbidden on partial reset.** The single env's all-world `_update_mjc_data` and `fwd_position`
   (`RCS/isaac_lab_env.py:549-551`), any full-sim settle, and `sim.reset()`.
   - Contact audits of a reset world are skipped for its first post-reset step instead.
   - **Risk:** a write path whose `articulation_ids` is `None` falls back to "mark every world"
     (`newton_manager.py:1098-1101`), which would clear warm starts everywhere. Test T1 audits the masks after each
     partial reset.
3. **B2 settled-grasp snapshot.**
   - At build (and at `rebuild`), all worlds start at the declared HOME (Native, `RCS/isaac_sim_runner.py:470-477,537-541`)
     and settle 2 s together, the existing procedure (`RCS/isaac_lab_env.py:610-648`).
   - Gates are evaluated per world on the device, over the same window and thresholds as `_settle_physical_grasp`.
     Strict mode raises with env ids; training mode marks those envs invalid.
   - A snapshot per slot is then stored on the device:
     - Franka q/qd (2×9);
     - tool pose and twist;
     - actuator targets;
     - `initial_relative_tool`;
     - the measured edge and pad boxes, which become the material's initial pose and boxes.
   - `reset(env_ids)` writes the snapshot. The slot's scene is fixed for the batch lifetime (ADR 0021 §6), so the
     snapshot is exact for that slot.
   - **The only difference from a fresh settle is the MJWarp warm start (zeroed).** Its effect is UNVERIFIED and
     measured in G3.
   - Re-settling one world would step every world. A different initial pose, grasp randomization or a tool
     re-pickup on partial reset therefore raises `UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET`; `rebuild()` is the
     explicit path.
4. **Material and coupling.**
   - `BatchedCuttingMaterial.reset(env_ids, seeds)` runs as specified in ADR 0021 §6.
   - Its initial pose is the snapshot edge. `pending_wrench`, `pad_wrench`, `applied_previous` and `external_wrench`
     are set to 0.
   - The grasp monitor, IK-failure streak, commanded pose (= nominal edge of the snapshot) and step counter are reset.

## 7. Control-step loop and cost (Q5)

All work runs on the Warp device stream. That stream is blocking, so it is ordered with Torch's legacy default
stream (isaaclab-venv `site-packages/warp/_src/torch.py:416-421`); no other streams are allowed (ADR 0021 §7
"Streams").

```
_pre_physics_step(a[N,8])   EdgeDeltaActionV1.apply(a, commanded)                         Torch, device
                            IK (48 LM iterations, N problems) -> FK residual -> ik_status   Warp launches
                            hold on failure; targets + velocity feed-forward -> target_command mask writes
loop 10x (DirectRLEnv)      scene.write_data_to_sim(); NewtonManager.step():
                              forward() (masked reset, if any) -> graph replay of one substep:
                                [state-force cb: wrench kernel N]  -> MJWarp substep, all worlds
                                -> [post-step cb (B2): update_contacts + per-world contact sample N]
                            scene.update(dt)
_get_dones (first hook)     edge/pad kernel -> set_measured_poses (device copy) -> pad boxes in place
                            material.advance(): pose_interp -> 200-substep graph -> wrench kernel
                            -> pending_wrench for the NEXT interval; validity, grasp monitor, IK streak -> dones
_get_rewards / _get_observations   device kernels; _reset_envs_from_buffer -> None (autoreset=False, no nonzero())
```

**Graph capture.**
- The Newton graph is one 2 ms substep, replayed 10× per control step, because the actuators are implicit (§1.2).
- Folding the decimation into one graph would need explicit `IdealPD` actuators. That changes actuator physics, so it
  is not done.
- IK is launch-only and can be captured as its own graph later (P1-3).
- The material graph is ADR 0021's.
- **The material and the next rigid interval are serial by construction:** delayed coupling means the interval k+1
  forces need material step k. Overlapping them would require a 40 ms delay, which is a semantic change.

**Host syncs per step.**
- Zero in training mode.
- In strict mode, one int32 (material `invalid_count`) plus one int32 (rigid flags: contact-pool overflow
  `nacon > naconmax`, nonfinite q, IK streak).

**Cost per control step** (ESTIMATES, except the material line, which is measured):

| Stage | N=1 | N=16 | Basis |
|---|---|---|---|
| Material | 129–137 ms | 0.7–1.6 s | WP-17 measured (ADR 0021:55); ADR 0021 §12 estimate of 1.3–3× aggregate gain |
| Rigid: 10 graph launches plus Python per substep | 5–20 ms | 6–25 ms | Latency-bound; MJWarp per-world arrays are small (§1.3) |
| IK (48 iterations, eager) | 1.5–4 ms | 2–5 ms | About 250 launches |
| Edge, pad and wrench kernels | <0.5 ms | <0.5 ms | `dim=N` |
| Robot share of the step | ≈5–18% | ≈1–3% | Workload B ≈ workload A throughput minus a few % |

The G2 probe measures this stage breakdown; the rows are never merged with workload A rows.

## 8. Parity claims and batching-correctness tests (Q6)

**Claimable.**
1. **Material parity on identical edge and pad sequences.** Each step records `measured_edge[7]`, the cut digit and
   `boxes[3,2,3]`. Replaying them through the frozen single `LiveCuttingMaterial(tlmpm_v7_2)` must give:
   - L0: CPU bitwise on short excerpts, same venv;
   - L3: inside the GPU run-to-run band over full episodes.

   This single class is the exact material code Native runs (`RCS/isaac_sim_runner.py:1564-1567`), so this is the
   "Native confirmation" P1-1b asks for.
2. **The coupling contract, per env and per step.**
   - The wrench applied in interval k+1 equals the material output of step k: bitwise, one buffer. This mirrors the
     Native assertion at `RCS/isaac_sim_runner.py:1570-1571`.
   - The material input equals the edge-kernel output.
3. **Batching correctness inside the Lab runtime:** N=1 equals N copies, and perturbations and resets stay isolated.

**Not claimable.** Robot trajectories, grasp slip, contact forces, the reaction→motion loop and success, compared with
Native: PhysX CPU and MJWarp differ in contact model, solver and mimic (ADR 0021:458-462).
- **The single Lab env cannot be a coupled reference.** It cannot run v7_2 or compiled TRIM (§1.1).
- Its N=1 rigid agreement with batched N=1 is **not bitwise**: the IK model differs (whole model against prototype),
  MuJoCo model composition differs (global against `env_0` paths), and the GPU contact-pool atomics vary run to run.
- So it is a tolerance test (G4).
- Native outcome comparisons are descriptive only.

**Tests.** C tests run in `.venv`, Warp CPU, no Torch, with `PYTHONPATH=src` (ADR 0021 implementation notes). T tests
run in `.deps/isaaclab-venv` with the D-09 overlay; in `.venv` they are NOT_RUN. G probes run on the GPU under D-07.

| Id | What | Pass criterion |
|---|---|---|
| C1 | Robot kernels on a synthetic 3-world `body_f`: index tables; edge kernel (B1 and B2 chains) against NumPy compose; pad-box kernel against `pad_boxes_edge`; **wrench kernel: a nonzero test wrench in env k changes only env k's tool and finger entries**; values against a NumPy lever-arm reference | Index and isolation **bitwise**; edge ≤1 ulp (f64); pads ≤1 ulp (f32); wrench f32 tolerance |
| C2 | Batched IK, N problems, against N single-problem solves on the prototype; against `DualFrankaController.inverse_kinematics` on 50 TRIM targets | Bitwise on CPU; both pass the 1 mm / 0.01 rad gate (joint Δ reported, no bitwise claim) |
| C3 | MJWarp on Warp CPU, RCS rigid scene (2 Frankas, B1/B2 tool, board, table), 60 substeps: (a) N=1 against world 0 of N=3; (b) world 1 driven into the board; (c) masked reset of world 2 (joint write + `reset(world_mask)`); contacts > 0 asserted | Unaffected worlds **bitwise** (promotes the §1.3 probes) |
| C4 | USD authoring: batched tool equals `author_newton_tool` after rebasing; B1 FixedJoint structure; no `RigidBodyAPI` under shared paths | Exact |
| T1 | Model receipt per world: body and DOF counts (B2: 23 and 24; B1: 23 and 18); `neq`; actuator arrays equal to the single Lab env's world 0; no global bodies; after `reset(env_ids)`, `_world_reset_mask` holds exactly those worlds | Exact |
| T2 | No host sync in 20 steps: `torch.cuda.set_sync_debug_mode("error")`, and `wp.array.numpy` / `wp.synchronize*` patched to raise; the strict-mode reads are whitelisted and counted | Pass; count reported |
| T3 | Coupling contract (claim 2) for every env and step of a 50-step N=4 run | Bitwise |
| G1 | N=4, `deterministic_mode="run_to_run"` (`disable_sensors=True`), material L4 mode if available: (a) identical copies; (b) perturbing env 2 leaves envs 0, 1, 3 equal to the unperturbed run; (c) resetting env 1 at step 30 leaves the others equal; env 1 equals a fresh run of that env; (d) debug `body_f` and contact-world audits | Bitwise expected. On failure the bitwise claim is reported **FAIL**, then the band verdict (3 runs) is reported separately; never a silent downgrade |
| G3 | B2 restore: a restored env against a never-reset env over 0.5 s hold; relative-pose spread after 3 fresh `rebuild`s | Within grasp tolerances (2 mm / π/60); restored-versus-fresh distribution reported (closes or keeps the UNVERIFIED) |
| G4 | Batched N=1 B2 against the single `IsaacLabCuttingEnv` on the same registered ruler task: `feedback=False`, identical `joint_position_rad` sequence (no IK), 100 steps | Tolerance fixed before running = 3 × the single env's 3-run spread; divergence curve reported; not an acceptance gate for coupling |
| G5 | Material parity (claim 1) on 2 envs of a G2 batch: L0 for the first 20 steps on CPU; L3 for 1200 steps × ≥3 single replays; descriptive Native seed-512 run with the same scripted controller | L0 bitwise; L3 within band and passes the frozen evaluator thresholds |

## 9. Staged plan, files, probes (Q7)

**Files (new).**
- `RCS/batched/lab_scene.py`: scene configuration builder, B1/B2 spawn functions, `author_batched_tool`, uniformity
  checks.
- `RCS/batched/robot_kernels.py`: index tables, edge/pad, wrench, contact sample, grasp monitor, rigid validity.
- `RCS/batched/robot_ik.py`: prototype IK, residual and status.
- `RCS/batched/grasp_batch.py`: settle gates, snapshot, restore.
- `RCS/rl/lab_batched_env.py`: `IsaacLabBatchedCuttingEnv`, plus `open_vector_env(backend="isaac_lab_newton",
  tool_mount=...)` wiring.
- `scripts/rl_ready/probe_lab_batched_gpu.py`.
- Tests `tests/test_rl_lab_*.py`.
- New `UnsupportedBatchConfiguration` codes in `batched/errors.py` (§10).
- `BatchedCuttingMaterial` accepts `backend="isaac_lab_newton"` and an in-place box writer. Today it refuses any
  backend but kinematic (`RCS/batched/material_batch.py:308-311`).

**Stages** (each gate PASS before the next):
- **S0 (CPU only).** C1 to C4, plus T1 and T3 on the smallest build that runs. Prerequisite: workload A M2 (plan).
- **S1, B1 on GPU.**
  - G0: N=1 smoke, graph capture, receipts, stage timing.
  - T2.
  - G1 at N=4.
  - G2 at N=1/4/16: 3 batches × 1200 steps with staggered resets, VRAM slope, workload B rows labelled
    `reduced_fidelity`.
  - G5.
- **S2, B2.** S1 repeated with the free tool, plus G3 and G4. This is the first unreduced workload B row.
- **S3.** Representative TRIM (seeds 512/15) and a configured CUT scene compiled for `tlmpm_v7_2` (the CUT-01 family
  reuses the same builder; legacy v1 CUT-01 is `UNSUPPORTED_SOLVER`). Run them in the batch with the frozen evaluator
  at episode end. Fill the P1-5 rows "Robot articulation/contact" (B1 PARTIAL, B2 YES on Lab, Native reference) and
  "Vectorized reset/step".

**P1-1b acceptance mapping.**
- "N=1/4/16 real batched articulation + material + delayed wrench" maps to G2.
- "Per-env wrench isolation" maps to C1, G1(b) and T3.
- "Native confirmation of material parity" maps to G5.

**GPU budget** (ESTIMATES; under the D-07 lock, explicit device, ≥4 GB left free, rows labelled SHARED/EXCLUSIVE).
- Fixed overhead (CUDA context, Torch, Warp/MJWarp/Newton modules, graphs): 0.8–1.5 GB, UNVERIFIED.
- Material: 55 MB/env (ADR 0021 §12).
- Rigid: <1 MB/world (MJWarp data 0.224 MB/world measured on CPU, plus IK, contacts and state).

| N | Worlds × (bodies / DOF), B2 | VRAM estimate |
|---|---|---|
| 1 | 1 × (23 / 24) | 0.9–1.6 GB |
| 4 | 4 × (23 / 24) | 1.0–1.8 GB |
| 16 | 16 × (23 / 24) | 1.7–2.4 GB |

## 10. Explicit UNSUPPORTED (raise; never a fallback)

| Code | Covers |
|---|---|
| `UNSUPPORTED_SENSOR` | Cameras, vision profiles, RTX, the ideal Warp RGB-D path (no USD prims for worlds ≥1 kit-less) |
| `UNSUPPORTED_ENV_SPACING` (new) | Any `env_spacing ≠ 0` |
| `UNSUPPORTED_NONUNIFORM_RIGID_SCENE` (new) | Fixtures, supports, table, robot bases or HOME differ between envs; existing `UNSUPPORTED_PER_ENV_TOOL_PROFILE` for the tool |
| `UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET` (new) | Per-env initial-pose, grasp or friction randomization on reset; `tool_init_mode` free or pickup |
| `UNSUPPORTED_RIGID_CPU` (new) | The `use_mujoco_cpu` / CPU Lab runtime, which is not mask-aware (`mjwarp_manager.py:117-122`). CPU coverage is the C tests |
| `UNSUPPORTED_ACTION_SPEC` (new) | Anything other than `edge_delta_v1`. `joint_position_rad` is allowed only in the G4 test harness |
| `UNSUPPORTED_ACTUATOR_OVERRIDE` (new) | Gains, drive or mimic configuration that differ from the single Lab env |
| `UNSUPPORTED_RECORDER` | `NewtonTool` raw 2 ms traces, cut events, surfaces |
| `UNSUPPORTED_SOLVER` | Anything other than `tlmpm_v7_2`; legacy wedge CUT-01 |
| `UNSUPPORTED_BACKEND` | PhysX / Native batching and N Kit processes |

Also unsupported: `autoreset="same_step"` with `final_obs` (host `nonzero`), deferred.

## 11. Risks

| # | Risk | Detection or mitigation |
|---|---|---|
| 1 | The shared MJWarp contact pool couples worlds on overflow (`MJW/io.py:1725-1726`) | Device overflow flag; a batch-level infrastructure failure, never a task failure; `nconmax` per world unchanged |
| 2 | A write path marks all worlds dirty (FK fallback), clearing warm starts everywhere | T1 mask audit; C3 and G1 bitwise checks |
| 3 | GPU rigid run-to-run nondeterminism hides isolation | CPU bitwise (C3), deterministic-mode G1, band only as a reported fallback |
| 4 | Finger drive count (one or two) under Lab's regex actuator config | T1 receipt equality with the single Lab env; documented as an inherited, not a new, difference from Native |
| 5 | B2 snapshot restore is not equivalent to a fresh settle (warm start zeroed) | G3; the label stays until it is measured |
| 6 | B1 FixedJoint body under the Lab `Articulation` view: joint or body naming at the Lab level | UNVERIFIED (Newton level PASS, §1.3); S0/T1 build check |
| 7 | Post-step `update_contacts` capture safety; IK eager launch overhead | G0 capture test; P1-3 IK graph |
| 8 | Venv split: workload B runs in isaaclab-venv (NumPy 2.5.2), workload A in isaacsim-venv (2.3.1) | Compare only rows from the same venv; hash packed initial material states across both venvs; if they differ, use a B-specific L3 band |
| 9 | 10 Python-driven substeps per control step | Measured in G2; it matters only at large N; optional outer capture in P1-3 |
| 10 | Material `backend` / box-writer API changes in the ADR 0021 code | Opt-in additions only; ADR 0021 L0/L1 tests rerun |

## Implementation notes

Appended during implementation; the sections above are unchanged. Each subsection names its stage and records
where the code deviates from or refines the text above.

### Stage S0 (CPU only), 2026-11-10

Files: `src/robotic_cutting_sim/batched/{lab_scene,robot_kernels,robot_ik,grasp_batch}.py`,
`src/robotic_cutting_sim/rl/lab_batched_env.py`, `scripts/rl_ready/probe_lab_batched_{cpu,gpu}.py`,
tests `tests/test_rl_lab_{kernels,ik,rigid_isolation,scene_usd,env}.py`. No GPU was used (`CUDA_VISIBLE_DEVICES=''`).

Results (Warp 1.17.0, Newton 1.6.1, MJWarp 3.12.0; C tests in `.venv` NumPy 2.5.4 and again in
`.deps/isaaclab-venv` NumPy 2.5.2 / Torch 2.11 with the D-09 overlay; AMD Threadripper PRO 9975WX):

| Test | Result |
|---|---|
| C1 index tables, edge, pad, wrench, contact/grasp/validity kernels | PASS. Edge kernel **0 f64 ulp** (bitwise) against `compose_pose` / `compose_tool_pose` on 4,096 random and all real poses, B1 and B2 chains; pad boxes **0 f32 ulp** on 21,600 components (gate stays <= 1 ulp); wrench isolation bitwise, values within 2e-6 of the NumPy lever arm |
| C2 batched IK | PASS. N problems bitwise equal to N single-problem solves (B1 and B2 prototypes); against `DualFrankaController.inverse_kinematics` on 50 TRIM targets both pass the gate and the measured joint difference is **0** (reported, not claimed in general) |
| C3 MJWarp on Warp CPU, 60 substeps, B1 and B2 | PASS. N=1 vs world 0 of N=3, a world driven into the board plus a per-world wrench, and a masked reset of world 2 all leave the other worlds bitwise equal; contacts > 0 in every substep |
| C4 USD authoring | PASS. `author_batched_tool` equals `author_newton_tool` after rebasing (every attribute, relationship, schema, custom datum, binding); B1 FixedJoint structure; no rigid body outside `{ENV_REGEX_NS}`; all §10 refusals raise their codes |
| T1 model receipt and mask audit (Lab, N=3, B1 and B2) | PASS. 23 bodies / 18 DOFs (B1) and 23 / 24 (B2) per world, no global bodies; per-joint Newton drive/limit arrays and MuJoCo `nu`, `neq`, `eq_type`, `actuator_gainprm/biasprm/forcerange/gear/trnid` equal to the single Lab env; after `reset_envs([F,T,F])` `_world_reset_mask = [F,T,F,F]` and `_fk_reset_mask` = exactly world 1's articulations |
| T3 coupling contract (Lab, N=4, 50 steps) | PASS, bitwise. Coupling input = edge-kernel output every env and step; the wrench read in all 10 substeps of interval k+1 = the coupling output of step k (a synthetic test coupling, not material) |
| Lab isolation (CPU analogue of G1 a-c, N=3, 10 steps) | PASS, bitwise (N=1 vs world 0, perturbed env 2, masked reset of env 1) |
| B2 settle / snapshot / restore (Lab, N=2) | PASS. Both worlds `PASS_PHYSICAL_PREGRASP` with identical accumulators (26 window rows, bilateral fraction 1.0, drift 0.2 mm); HOLDING over 5 steps; masked restore of world 1 marks only world 1 and restores its settled joint state bitwise, world 0 untouched. Not G3 (no warm-start comparison) |
| T2, G0-G5 | NOT_RUN (GPU, stage S1+); `scripts/rl_ready/probe_lab_batched_gpu.py` is written and only compile-checked |

Deviations and refinements:

1. **The Lab runtime runs on the Warp CPU device in S0.** `BatchedLabOptions(device="cpu", cpu_test_mode=True)` builds
   the real `DirectRLEnv` with MJWarp on Warp CPU (`use_mujoco_cpu=False`). So T1, T3, Lab-level isolation and the B2
   settle ran in S0, not only "the smallest build". `UNSUPPORTED_RIGID_CPU` is refined: `use_mujoco_cpu=True` is
   always refused; `device="cpu"` is refused unless `cpu_test_mode=True` (as the material's `parity_mode`).
2. **The single Lab env has no world 0.** With `num_envs=1` and `/World/Franka_*` paths, all 23 of its bodies are in
   world -1 (`world_count=1`). T1 compares against those global bodies (`make_lab_cfg`, physical grasp,
   `TRIM-01-R`, with `use_mujoco_cpu` switched off on the returned cfg so both use MJWarp).
3. **Finger drive (closes the §4.4 UNVERIFIED for Lab):** in both the single env and the batch every DOF of a Franka
   has a position and a velocity actuator (`nu = 36` per world for two Frankas), so **both** finger DOFs are driven,
   plus one joint equality per Franka (`neq = 2` per world, `eq_type = joint`). The batch inherits this unchanged.
4. **B1 tool prim.** The welded tool is a child prim of the hand, `.../panda_hand/CuttingTool`, with the FixedJoint
   at `.../CuttingTool/HandMount`. The Lab `Articulation` lists `CuttingTool` among Franka 0's body names; its
   `joint_names` stay the 9 Panda joints (risk 6 resolved on CPU). The tool body is interleaved (`tool = hand + 1`).
   Its gravity compensation is 0, like the free tool of the single env. B1 uses the physical-grasp finger drive
   (2000 N/m, 20 N s/m, 20 N) and holds the fingers at the task HOME opening (0.0112 m).
5. **SciPy apply order.** The frozen paths compose one pose at a time; a single or length-1 `Rotation.apply` matches
   the dgemv FMA chain `fma(R2,v2, fma(R0,v0, R1*v1))` (the order `pose_interp` pins). A multi-row batch would use
   NumPy einsum's `(R0 v0 + R2 v2) + R1 v1` instead. The kernels use the single-pose order.
6. **Partial reset skips `scene.reset(env_ids)`.** `Articulation.reset` calls `actuators.reset(None)` (all envs). The
   batch uses implicit actuators only (asserted at bind; others raise `UNSUPPORTED_ACTUATOR_OVERRIDE`) and no wrench
   composers, so there is no per-env actuator or composer state to reset. `_reset_envs_from_buffer` returns None;
   terminated envs wait for an explicit `reset_envs(mask)`.
7. **IK hold.** On `REJECTED_IK_UNREACHABLE` the env keeps its commanded pose and previous joint targets (velocity
   feed-forward 0) and disarms the cut; the executed action is the HOLD row. An IK failure streak does not
   invalidate an env by itself (`IK_STREAK_LIMIT = 0`); the streak is kept per env.
8. **Material coupling in S0.** `BatchedCuttingMaterial` still refuses `backend="isaac_lab_newton"`, so the env
   drives a `MaterialCoupling` interface (ADR 0021 §7: `pending_wrench [N,6]` and `pad_wrench [N,B,6]` read in
   place, `box_centers/box_halves` written in place before `advance`, `set_measured_poses`, `advance`,
   `reset_masked`). The only built-in implementation is `RobotOnlyCoupling` (no material, zero reaction, label
   `robot_only_no_material_coupling`, `reduced_fidelity`). Reward is zero with `extras["reward_available"] = False`;
   `compute_task_reward()` and `evaluate()` raise `UNSUPPORTED_MATERIAL_COUPLING`.
9. **B2 initialization.** The tool starts at `compose_tool_pose(hand FK(HOME), hand_from_tool)`; the single env
   uses `reference_edge(0) o inv(T_tool_edge)` instead. The fingers start at the task HOME opening, then target
   `closed_target_m`. Later `reset()` calls restore the snapshot of every world; there is no re-settle and no
   `rebuild()` yet (a new env is the explicit path).
10. **Contact world.** A contact's world is `max(shape_world[a], shape_world[b])`, so a static (-1) shape resolves to
    the other side. Grasp drift angles are computed in float64 (a float32 `acos` near 1 quantizes to about 5e-4 rad).
11. **C3 test model.** `build_newton_rigid_scene` is a pure Newton model with the same USD authoring. Its drives are
    set on the builder as `DualFrankaController` does (`nu = 18` per world, not Lab's 36). It is a test model; the
    Lab-level equivalents are the T tests.
12. **Entry point.** `rl.lab_batched_env.open_lab_batched_env(num_envs, backend="isaac_lab_newton", tool_mount=...)`
    returns `(cfg, factory)`. The dispatch from `rl.vector_env.open_vector_env` (another work package) is not wired
    yet; it still refuses `isaac_lab_newton`.
13. **New UNSUPPORTED codes** are raised through `lab_scene.UnsupportedLabBatchConfiguration`, a subclass of
    `errors.UnsupportedBatchConfiguration`, until `errors.CODES` lists them: `UNSUPPORTED_ENV_SPACING`,
    `UNSUPPORTED_NONUNIFORM_RIGID_SCENE`, `UNSUPPORTED_GRASP_RESETTLE_ON_PARTIAL_RESET`, `UNSUPPORTED_RIGID_CPU`,
    `UNSUPPORTED_ACTION_SPEC`, `UNSUPPORTED_ACTUATOR_OVERRIDE`, and two not in §10: `UNSUPPORTED_TOOL_MOUNT` and
    `UNSUPPORTED_MATERIAL_COUPLING`.

Measured on CPU (development figures, not performance rows): Lab build plus reset for N=3 B1 takes about 1.8 s
(solver init 0.84 s). A 50-step N=4 coupling run takes 10.3 s wall including the build. The B2 2 s settle for N=2
takes 8.4 s wall. The full T suite takes 110 s.

### Stage S1 (GPU0, robot-only gates, then material rows), 2026-11-10

Report: `docs/reports/rl_ready/P1_1B_ROBOT_BATCHING.md`; evidence `docs/evidence/rl_ready/p1_1b_s1_001.json`; raw
outputs `artifacts/rl_ready/p1_1b/<run>/`. All rows `SHARED_GPU` (GPU0 co-tenants at 65-90 % SM).

Deviations and refinements:

1. **Table geometry (fix of §3.1).** §3.1 prescribed the static colliders "exactly as `isaac_lab_env.py:267-275`",
   i.e. the legacy 1.8 x 1.2 m table centered at the origin. For WP-12 flat-board tasks Native places the table
   from `task.table_box_world` (`isaac_sim_runner.py:956-966`; for seeded TRIM y from -0.075 to 0.6 m). With
   the legacy table the hand of the TRIM skim pose (hand horizontal, about 0.2 m to -y of the edge) hit the
   table and the edge stalled 6 cm above the cut plane. `lab_scene.table_box(task)` now follows Native (legacy
   table only when a task declares no box); `rigid_scene_key` includes it. Rows before commit `359ce0d`'s
   source state used the legacy table and are kept as superseded.
2. **Graph capture and Python state.** Callbacks captured into the Newton substep graph must not branch on Python
   state after binding: the B2 contact callback samples unconditionally once bound (an earlier `_sampling`
   flag would have been frozen at capture time). If a Newton graph was captured before the material existed
   (B2 captures during the settle), `_attach_material` drops it and Newton recaptures on the next step, so the
   wrench kernel reads the material's `pending_wrench` buffer.
3. **Host syncs.** `targets[:, cols] = value` (index_put with a device index) synchronized; replaced by masks.
   T2 then finds exactly one host read per step (`check_valid`, strict mode) and no Torch sync.
4. **Deterministic mode** sets `MJWarpSolverCfg.disable_sensors=True` (Newton's precondition; no MuJoCo sensors
   are used). The first deterministic build compiled its kernel variants for 9 minutes (cold cache).
5. **Material coupling.** `material="tlmpm_v7_2"` builds `BatchedCuttingMaterial(backend="isaac_lab_newton")`
   from the measured HOME edge and pad boxes at the first `reset()` (recipients: handle + both finger pads).
   Rewards stay zero (`reward_available` false): the packed reward is not wired into this env yet.
6. **GPU G-probe driver.** G2 material rows use a device reference-tracking driver of `edge_delta_v1` on the
   task's `reference_action` (not a policy, not the WP-17 controller).
