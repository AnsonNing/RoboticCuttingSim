# WP-08 — Physical Tool Grasping, Two-Way Cutting Physics & Generalized Cutting Framework

**Repository:** https://github.com/AnsonNing/RoboticCuttingSim

**Implementation target:** Isaac Sim 6.2 / Isaac Lab 3.0 / Franka Panda / PhysX / Newton / Warp TLMPM

**Primary objective:** Build a physically credible robotic cutting environment in which a Franka Panda genuinely holds a cutting tool using its two-finger gripper, and the interaction between the tool, robot, and cuttable object is simulated through coupled physical dynamics.

This is an engineering and physics integration task, not a visual enhancement task.

## 0. Non-negotiable design decisions

The following requirements are fixed.

1. Franka must physically grasp a real, independent rigid-body tool. The tool must have mass, inertia, collision geometry, contact friction, and measurable position and velocity.
2. The default cutting-task initialization is `pregrasped`: the gripper is already closed around the tool and maintains a physically generated gripping force.
3. The cutting task begins after the tool has reached a verified stable grasp. Tool acquisition and grasp stabilization are not included in the existing 8-second cutting episode.
4. During the episode, the tool is held through contact and friction between the fingers and handle. It must not be rigidly attached to `panda_hand` through a fixed joint, transform parenting, kinematic synchronization, or any hidden constraint.
5. Cutting forces must act on the independent tool body, then be transmitted to the gripper and Franka through actual contact dynamics.
6. The measured tool pose, rather than the nominal hand-to-tool transform, must drive the material solver.
7. The gripper maintains its hold command throughout normal cutting, but the physical simulation must permit slipping, rotating, or dropping the tool if the applied load exceeds the available grasp stability.
8. All robot-executed cutting tasks must share this system: CUT-01, TRIM-01, fixed/randomized variants, and future cutting tasks.
9. Preserve the existing fixed-tool implementation under an explicitly named legacy mode for scientific comparison.
10. Support new cuttable-object geometry and material backends without rewriting the entire robot/tool system. Do not claim physically valid cutting of arbitrary materials without an appropriate solver and validation.

Do not modify the RSI repository, unrelated projects, system drivers, or validated installations.

## 1. Audit and preserve the existing baseline

Before implementing, inspect the latest main and the following files:

- `src/robotic_cutting_sim/isaac_sim_runner.py`
- `src/robotic_cutting_sim/isaac_lab_env.py`
- `src/robotic_cutting_sim/robot.py`
- `src/robotic_cutting_sim/live_material.py`
- `src/robotic_cutting_sim/tlmpm.py`
- `src/robotic_cutting_sim/tasks.py`
- `src/robotic_cutting_sim/live_env.py`
- `src/robotic_cutting_sim/remote_viewer.py`
- `configs/cutting_tasks_v1.json`
- Existing physics, camera, deployment, and task-validation reports.

The present implementation places `LiveKnife` underneath `panda_hand` in the Isaac Sim stage, or attaches a knife shape directly to the hand body in the Newton controller.

The TLMPM solver receives a tool pose derived from the hand and applies reaction forces directly to the hand.

This must be replaced for the new physical-grasp mode.

Before any modification:

- Record the exact baseline commit.
- Freeze existing CUT/TRIM scoring criteria.
- Preserve existing implementation hashes, experiments, videos, and failures.
- Create an implementation branch.
- Keep a runnable `legacy_fixed_tool` mode.
- Execute existing CPU and live-runtime regression tests.

The new mode must never silently fall back to the legacy fixed tool.

## 2. Create a reusable physical cutting tool

### 2.1 Tool model

Create a reusable straight-edge cutting tool with three connected geometric regions:

**A. Grip handle**

The handle must be compatible with the Franka parallel gripper.

It requires:

- A defined grasp region.
- Physically reasonable handle width and thickness.
- Finger-contact collision geometry.
- Configurable friction coefficients.
- Defined nominal grasp pose.
- A surface that allows both fingers to generate opposing contact forces.

**B. Tool body**

Create a visible metal body connecting the handle to the cutting edge.

It must have:

- Realistic dimensions.
- Physical mass and inertia.
- Center of mass.
- Collision geometry for interaction with rigid scene objects.
- Correct geometric alignment with the handle and cutting edge.

**C. Cutting edge**

The cutting edge must provide:

- Accurate local position and orientation.
- Geometric parameters matching the existing TLMPM blade-contact model.
- A defined coordinate frame.
- Configurable blade length, width, thickness, and orientation.
- Explicit physical versus visual geometry relationships.

Do not simply draw a large decorative knife while preserving a disconnected physical wedge elsewhere.

### 2.2 Rigid-body requirements

The complete tool should initially be one independent rigid body with compound visual and collision geometry.

Its physical properties include:

- Total mass.
- Center of mass.
- Inertia tensor.
- Linear/angular velocity.
- Gravity.
- Rigid contact friction.
- Contact restitution/compliance where appropriate.
- Tool–finger and tool–fixture collision interactions.

Use actual geometry or an explicitly measured approximation to derive mass properties. Record any provisional assumptions.

The dynamic tool must exist outside the robot articulation hierarchy, for example:

`/World/Tools/CuttingTool_0`

It must not be a rigid child of `/World/Franka_0/.../panda_hand`.

### 2.3 Coordinate frames

Define separate transforms:

- `T_world_hand`: actual measured Franka hand pose.
- `T_world_tool`: actual measured tool rigid-body pose.
- `T_tool_grasp`: nominal grasp frame on the handle.
- `T_tool_edge`: cutting edge frame relative to the tool.
- `T_world_edge_actual = T_world_tool @ T_tool_edge`.

A nominal grasp transform can be used for initialization, IK references, and controller planning.

It must not overwrite the measured tool pose during simulation.

CUT-01 and TRIM-01 may use different blade-head geometries while sharing the same physical grasp interface.

## 3. Implement true Franka gripper contact

### 3.1 Finger articulation

Use the actual `panda_finger_joint1` and `panda_finger_joint2`.

The controller must support:

- Open.
- Close.
- Hold.
- Release.
- Configurable grip command and force/effort limits.
- Actual finger position and velocity feedback.
- Contact-force and contact-pair diagnostics.

Closing the gripper must cause the fingers to contact the handle physically.

Do not instantly reposition the tool to simulate a completed grasp.

### 3.2 Contact mechanics

Use the native contact mechanism of each runtime:

- Isaac Sim: PhysX rigid-body contacts.
- Isaac Lab: Newton/MuJoCo-Warp rigid-body contacts.

Read the actual contact state or solver contact data to verify the interaction.

The system must be able to distinguish:

- Contact on the left finger.
- Contact on the right finger.
- Contact between handle and fingers.
- Contact with robot palm or unintended geometry.
- Contact with table or fixtures.
- No contact.

Physical contact and friction are required; merely detecting that the tool lies geometrically between the fingers is insufficient.

### 3.3 Grasp stability

A stable grasp is not defined by a fixed tool-to-hand transform.

Evaluate stability using measured:

- Tool pose.
- Tool linear/angular velocity.
- Relative tool-to-gripper translation.
- Relative tool-to-gripper rotation.
- Finger positions.
- Finger contacts.
- External tool wrench.

Under inadequate gripping force or excessive external cutting load, the tool must be able to slip or fall.

Avoid unrealistic grip strengths, infinite friction, massless tools, or solver modifications intended only to force stability.

## 4. Pre-grasped initialization — default for all cutting tasks

Implement:

`tool_init_mode = "pregrasped"`

This becomes the target default for all cutting episodes after end-to-end validation.

### Reset sequence

1. Initialize the Franka articulation in a collision-free pre-cut pose.
2. Initialize the independent tool rigid body with its handle positioned between the two finger pads.
3. Set physically consistent initial velocities and angular velocities.
4. Set the gripper's nominal closing/holding command.
5. Advance the actual physics engine until both fingers establish contact and the tool stabilizes.
6. Verify tool support, contact forces, pose stability, and absence of unintended penetrations.
7. Initialize/reset the TLMPM material and cutting history in a way that does not retain settling artifacts.
8. Reset the task clock, observation timestamps, and evaluation history to the formal episode start.
9. Begin CUT/TRIM at t = 0 with a stable physical grasp.

The settling phase must not consume the existing 8-second cutting budget.

It must not alter the workpiece, perform cutting, or introduce hidden force/pose corrections.

If the grasp fails to stabilize, return a specific failure status and preserve the actual physics trace. Do not silently attach the tool.

### Default behavior during cutting

- The fingers maintain a closing/holding command.
- The gripper does not automatically open.
- The arm controller executes cutting actions.
- The tool remains dynamically independent.
- Slip/drop remains physically possible.
- The task reports grasp failure if the tool is lost.

A gripper can be commanded to hold without being artificially locked.

## 5. Implement true two-way tool–material–robot coupling

This is the highest-priority physics requirement.

### 5.1 Forward interaction

The simulation must follow:

`Robot joint motion → Finger motion/contact → Tool rigid-body motion → Actual cutting-edge motion → Material contact/deformation/separation`

The measured cutting-edge motion must drive TLMPM.

Do not compute physical cutting solely from the desired Cartesian target or hand pose.

### 5.2 Reaction forces

The return path must follow:

`TLMPM material reaction → Tool rigid body → Tool–finger contact impulses → Franka articulation dynamics`

The material reaction must be applied at the physically correct location on the independent tool.

Define the wrench reference frame and application point explicitly.

For example, when converting a wrench about the cutting edge into a wrench about the tool center of mass, include the appropriate lever-arm torque:

`tau_com = tau_edge + (p_edge - p_com) × F`

Verify signs, units, reference frames, timestamps, and applied impulses numerically.

Remove the old direct-hand material-wrench injection from the new physical-grasp mode.

### 5.3 Preserve physically correct rigid contacts

The tool rigid body must physically interact with:

- Both gripper fingers.
- Other relevant Franka surfaces.
- Table and fixtures.
- Tool rack where present.

The tool–soft-material interaction remains owned by TLMPM unless an explicitly validated alternative replaces it.

Do not compute the same tool–material contact through both PhysX/Newton and TLMPM, which would double-count forces.

If different parts of the tool use different collision representations, document exactly which solver owns each contact pair.

### 5.4 Coupling time steps

The current system uses approximately:

- Robot/control interval: 20 ms.
- Rigid simulation substeps: 2 ms.
- TLMPM substeps: 0.1 ms.
- Material reaction feedback: one control interval of delay.

Verify whether the current delayed coupling remains numerically stable with a free tool held by friction.

If it generates excessive slip, instability, or nonphysical energy changes, investigate a more tightly synchronized coupling method.

Do not reduce reported physical errors by overriding the tool pose or adding artificial fixation.

Record:

- Tool linear/angular motion.
- Contact force and torque.
- Material reaction force and torque.
- Applied wrench timing.
- Tool–gripper load transfer.
- Contact penetration.
- Slip.
- Momentum/impulse diagnostics.
- Interface-work residual.

The existing coupling-work failures must remain visible until independently resolved.

### 5.5 Negative controls

Include actual experiments in which:

- Material reaction is disabled.
- Material reaction is enabled.
- Grip friction is reduced.
- Grip force is reduced.
- An external wrench is applied to the tool.
- The gripper opens.

These cases must produce physically explainable changes.

If changing material load does not affect the tool/finger/robot dynamics as expected, the two-way coupling should fail validation.

## 6. Generalize the cutting architecture beyond one fixed object

The long-term requirement is that the same robot, gripper, tool, and controller can operate on different cuttable objects.

Do not hard-code the new grasping or tool dynamics into CUT-01 or TRIM-01.

### 6.1 Separate four concepts

**Tool specification**

Defines physical tool geometry, grip frame, cutting edge, inertia, and collision parameters.

**Workpiece specification**

Defines object geometry, dimensions, world pose, initial placement, and support/fixture configuration.

**Material specification**

Defines constitutive behavior, deformation, friction, damage/fracture/cutting law, and required physical solver.

**Task specification**

Defines what the robot should accomplish and how success is independently evaluated.

Each of these should be independently replaceable without changing the gripper's physical mechanism.

Use small Python dataclasses/functions and the existing task framework. Do not create a large plugin registry or new orchestrator unless multiple real implementations demonstrate a need.

### 6.2 Initial supported objects

The first fully verified material remains the existing provisional Plasticine-like TLMPM material.

The new architecture must work with:

- CUT-01 continuous rectangular bar.
- TRIM-01 continuous bump surface.
- Existing randomized lengths, poses, and cutting locations.
- Existing randomized 1–3 bump configurations.
- At least one additional workpiece geometry that is genuinely different from the fixed reference task.

The additional geometry should be processed by the same physical material pipeline, not implemented as a special visual-only exception.

### 6.3 Future object support

Design explicit extension points for:

- Arbitrary procedural surface profiles.
- Additional connected soft-material geometries.
- Imported watertight mesh-to-volume sampling when feasible.
- Configurable workpiece fixtures.
- Different tool geometries.
- Material-specific cutting and fracture models.
- Multiple cutting operations.
- Future two-arm object holding and cutting.

If importing an arbitrary USD/STL/OBJ object is supported, validate that the object can be turned into a coherent simulation volume with correct mass and collision behavior.

Do not label an unsupported rigid or deformable object as cuttable merely because it can be imported visually.

The current TLMPM is not a universal solver for wood, metal, fabric, food, or arbitrary CAD models. New materials must report their actual support and validation status.

### 6.4 Multiple cutting operations

The present task-level cutting history has limited support for repeated cuts and arbitrary cutting planes.

Preserve the current single-cut benchmark, but remove object-specific assumptions from the tool/grasp interface.

Implement the cleanest feasible route toward multiple cuts; where the existing material solver cannot support a requested sequence, report the limitation rather than faking a cut.

## 7. Integrate all execution environments

### 7.1 Isaac Sim 6.2

Implement the physical tool as an independent native PhysX rigid body.

Use the actual articulated Franka finger geometry and contacts.

Verify that:

- The tool is not welded to the robot.
- The tool has nonzero mass/inertia.
- Gravity acts on the tool.
- Finger contact supports the tool.
- Releasing the grip allows the tool to move independently.
- Measured tool poses enter TLMPM.
- Material reaction acts on the tool.
- RGB-D contains the real tool geometry.

### 7.2 Isaac Lab 3.0

Implement equivalent semantics in the actual Isaac Lab/NewtonManager execution path.

The knife must be an independent free rigid body in the Newton physical model, not a shape attached to the hand link.

Verify actual contact behavior with the pinned MuJoCo-Warp solver.

Do not assume that passing the PhysX implementation automatically proves Newton grasping behavior.

Record differences in contact force, slip, trajectory, and physical task outcomes.

### 7.3 Shared API

Preserve the existing:

- `reset()`
- `step(action)`
- `observe()`
- `render()`
- `evaluate()`
- `close()`

Add only the necessary tool/gripper configuration and state fields.

Suggested interfaces include:

- `tool_init_mode`: `pregrasped`, `pickup`, `free`.
- `tool_attachment_mode`: `physical_grasp`, `legacy_fixed_tool`.
- `gripper_control_mode`: `hold`, `position`, `force`, `release`.
- `tool_id` / `tool_spec`.
- Measured gripper width, finger joint positions, and velocities.
- Separate measured tool pose/velocity and grasp diagnostics.

These are suggested API names; reconcile them with existing project conventions.

Maintain compatibility with current 7D Cartesian-edge and Joint Action commands.

An edge-target action should be converted into an appropriate nominal hand target for IK. The controller must not correct a slipping physical tool by teleporting it back to the commanded edge pose.

Keep simulator-only tool truth, exact contact impulse, cut topology, material particles, and oracle task scores outside the default deployable Policy Observation.

### 7.4 Dataset and Policy compatibility

Ensure future datasets can record:

- Both Wrist D405 RGB-D streams.
- Global D455 RGB-D.
- Robot joint positions/velocities.
- Finger positions and gripper state.
- Issued arm/gripper actions.
- Actual tool motion.
- Task outcome.
- Optional privileged contact/material debug in separately labeled data.

Maintain accurate timestamps and action-observation alignment.

No policy training is required for WP-08, but the environment should remain usable for IL, RL, VLA, and autonomous Agent control.

### 7.5 Web Viewer

Update the existing same-instance WebRTC Viewer.

The GUI must show the complete knife, handle, and actual gripping fingers.

Add useful controls for:

- Pregrasped Reset.
- Optional Open / Close / Release.
- CUT / TRIM task selection.
- Camera switching.
- Grasp status.
- Actual tool pose and slip diagnostics.

The live viewer must show the same rigid tool and physics as the headless environment.

Do not display a fake knife or independent animation.

## 8. Add optional pickup and release capabilities

These capabilities are not part of default cutting episodes, but should be available for future research.

Implement a basic:

- Tool Rack.
- `pickup_tool` demonstration.
- `hold_tool` controller.
- `release_tool` action.
- `place_tool` demonstration.
- Free-tool drop test.

At minimum, a scripted pickup should be able to approach the handle, close the actual fingers, lift the tool, and demonstrate physical support.

Autonomous tool selection, vision-guided grasp planning, and learned pickup policies are future research tasks, not requirements for the initial Cutting Benchmark.

These additional functions must use the same rigid tool and contact mechanism as `pregrasped`.

## 9. Required validation — no visual-only passes

Build independent validation scripts.

### A. Independent rigid tool

- Tool is a free dynamic body.
- Nonzero mass/inertia.
- Not a fixed child of the hand.
- No permanent Fixed Joint or hidden attachment.
- Gravity causes motion when unsupported.
- Dynamic pose is read from the physics solver.

### B. Physical gripper grasp

- Both finger pads contact the handle.
- The grip is generated through actual finger actuation.
- The tool can be lifted and supported against gravity.
- The tool does not require a pose correction every frame.
- Contact and slip can be measured.

### C. Stable hold

Run a no-cutting hold test over a declared interval.

Measure tool-to-hand relative translation and rotation, contact persistence, finger opening, and velocities.

Set provisional numeric stability tolerances before the test. Do not tune the acceptance criteria after measuring the result.

### D. External load response

Apply controlled forces and torques at the tool cutting edge.

Use loads around the actual recorded CUT/TRIM reaction-force scale and additional higher-load stress cases.

Confirm that:

- The tool responds dynamically.
- Contact forces change.
- The robot responds to the load.
- Grip stability is finite and physically explainable.

### E. Weak grip / low friction

Reduce grip effort or contact friction.

The tool must demonstrate measurable slip or loss of grasp under an appropriate load.

A tool that remains perfectly attached independently of friction should fail.

### F. Release

Open the fingers in a safe test configuration.

The tool must become unsupported and follow physically simulated motion, including gravity and collisions.

It cannot remain suspended at a fixed transform relative to the hand.

### G. Physical cutting

Run complete CUT-01 and TRIM-01 episodes using `physical_grasp`.

Verify:

- The independent knife remains physically supported during normal cutting.
- The actual measured knife edge contacts material.
- TLMPM performs material deformation and separation.
- Reaction forces act on the independent knife.
- The fingers transmit the reaction into the articulation.
- Physics remains finite and contact diagnostics are recorded.
- Cutting evaluation uses the unchanged independent criteria.

CUT-01 should be tested against its frozen severing score.

TRIM-01 must preserve its original strict surface-score failure unless new physics genuinely improves it. Do not weaken thresholds.

### H. Randomization

Test multiple seeds for both CUT and TRIM.

Check:

- Different workpiece positions.
- Different workpiece lengths.
- Different cut locations.
- Multiple bumps.
- Actual grasp stability.
- Tool/material collision.
- Camera visibility.
- Reset repeatability.

The physical tool must not rely on hard-coded alignment to the seed-42 reference object.

### I. Camera validation

Verify both Wrist D405 cameras and Global D455 capture the real knife and its cutting interaction.

Check handle visibility, blade visibility, self-occlusion, tool/camera clearance, calibration, and image timestamps.

Preserve the existing known D405 mounting/occlusion limitations until fixed and independently retested.

### J. Independent solver comparison

Validate the physical-grasp behavior separately in:

- Isaac Sim / PhysX.
- Isaac Lab / Newton.

Do not report unsupported API calls, skipped physics execution, or visual playback as successful live simulation.

## 10. Implementation phases and acceptance

### WP-08.1 — Physical Tool Asset

Deliver the real tool specification, geometry, mass/inertia, independent rigid-body integration, and free-fall/collision tests.

**Exit condition:** The tool exists independently and is physically dynamic.

### WP-08.2 — Franka Physical Grasp

Implement gripper actuation, two-sided contact, `pregrasped` reset, hold, slip, and release.

**Exit condition:** The Franka can genuinely support and release the tool through finger contact, without hidden attachment.

### WP-08.3 — Tool–Material–Robot Coupling

Connect the actual tool edge to TLMPM and route reaction forces through the independent tool and gripper contacts.

**Exit condition:** Matched force-on/force-off and grip perturbation tests prove that material load changes actual tool and robot dynamics.

### WP-08.4 — CUT/TRIM Integration

Integrate the shared physical-grasp mechanism into all robot-executed CUT/TRIM fixed and randomized cases.

Preserve historical legacy baselines.

**Exit condition:** End-to-end task results, including failures, are independently scored and reproducible.

### WP-08.5 — Isaac Lab Parity

Complete the Newton/MuJoCo-Warp implementation and compare it against PhysX.

**Exit condition:** The same grasp semantics and core physical tests execute in both runtimes, with differences measured rather than assumed away.

### WP-08.6 — Future Tool Skills & Object Extensibility

Add optional pickup/release/tool-rack tools and exercise the new workpiece specification with a genuinely different geometry.

**Exit condition:** The existing hand/tool implementation can be reused without copying CUT/TRIM-specific physics logic.

### WP-08.7 — Viewer, Dataset and Deployment

Update the Viewer, shared research API, recordings, documentation, and portable deployment workflow.

**Exit condition:** A new Clone can reproduce the declared physical-grasp smoke tests in both installed runtimes.

## 11. Required experiment reporting

For each work package, provide:

- Exact Git commit/source hashes.
- Runtime/package/driver versions.
- Commands and configuration.
- CPU/GPU test results.
- Actual physics logs.
- Joint and finger trajectories.
- Measured tool pose, linear/angular velocity.
- Finger contact pairs and contact forces.
- Grip-slip metrics.
- Applied tool/material forces and torques.
- Physical cutting metrics.
- RGB-D screenshots and videos.
- Runtime performance.
- Known failures and unresolved limitations.

Keep separate statuses for:

- `TOOL_RIGID_BODY_VALIDATED`
- `PHYSICAL_GRASP_VALIDATED`
- `TWO_WAY_COUPLING_VALIDATED`
- `CUT_TASK_VALIDATED`
- `TRIM_TASK_VALIDATED`
- `CAMERA_VALIDATED`
- `GENERAL_OBJECT_SUPPORT_VALIDATED`
- `FULL_PHYSICS_ACCEPTANCE`

A successful task must not automatically mark every physics validation as passed.

Negative results must remain available. Do not silently replace failed runs or change previously frozen thresholds.

## 12. Repository and implementation rules

- Work only in `AnsonNing/RoboticCuttingSim`.
- Reuse the existing PhysX, Newton, Warp, TLMPM, Task and Recording infrastructure.
- Avoid unnecessary new frameworks, services, or registries.
- No system driver upgrades.
- No modification to RSI or unrelated Python environments.
- No physical cheating to make a test pass.
- No hard-coded tool-pose synchronization during an episode.
- No force injection directly into the hand in the new mode.
- No unverified universal cutting claims.
- Do not overwrite previous experiment artifacts.
- Make incremental commits with independent tests and short reports.
- Update README, API documentation, physics documentation, deployment tests, and `docs/PROGRESS.md`.
- Push the tested work to GitHub using the repository's normal branch/review workflow.

Once a physical-grasp implementation passes its end-to-end requirements, make `physical_grasp + pregrasped` the default for robot-executed cutting tasks.

Keep `legacy_fixed_tool` explicitly selectable for regression and historical reproducibility.

If any required core capability remains unverified, report that limitation rather than shipping a fake physical-grasp fallback.

## Final completion criteria

WP-08 is complete only when the following end-to-end behavior is demonstrated in real simulation:

**1.** Franka begins the cutting task with a real independent knife held between the two fingers.

**2.** The fingers maintain contact and physical gripping force while the arm moves.

**3.** The knife's mass, inertia, friction, and contact dynamics affect its actual motion.

**4.** Cutting the workpiece creates material deformation/separation and a physical reaction on the knife.

**5.** That reaction is transmitted through the fingers into the Franka articulation.

**6.** An insufficient grasp permits slip or dropping; releasing the fingers causes real free-tool behavior.

**7.** CUT/TRIM and their randomized versions use the same mechanism without object-specific gripping hacks.

**8.** Both Isaac Sim and Isaac Lab have independent live evidence, with any remaining physics or solver limitations explicitly recorded.

**9.** The new architecture supports adding a different cuttable workpiece without redesigning the gripper and tool dynamics.

**10.** The default task starts with a stabilized physical grasp, so Policy/Agent research can concentrate on cutting rather than tool acquisition.

The final product should be a reusable physical robotic tool-use research environment, not simply a knife animation attached to a robot hand.
