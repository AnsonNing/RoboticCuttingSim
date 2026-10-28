RoboticCuttingSim — Physical Surface Cutting Simulation Environment Implementation Plan
Codex Execution Edition | v1.0 | 2026-10-27
Project type: independent Isaac Lab / Isaac Sim simulation research platform
Primary users: researchers in robotics, physical simulation, visual perception and machine learning
This document is a plan to execute, not a report of completed implementation or validated performance.

Current camera addendum (2026-10-28): the user's later specification supersedes
the fixed-camera layout in this original plan. The active shared API uses two
hand-mounted `ideal_d405_prototype` cameras and a closer fixed Global D455.
See [current camera configuration](docs/WRIST_CAMERAS.md) and
[actual validation](docs/WRIST_CAMERA_VALIDATION.md). Original stage gates,
fixed-camera experiments and failed results remain in the historical record.

0. Execution instructions for Codex (read first)
Build a new independent repository, provisionally named RoboticCuttingSim, for a physical simulation environment in which a robot uses a knife to remove irregular bumps from soft/plastic material fixed to a table, bringing its surface close to a target plane.

This phase implements only the simulation environment and verifiable research tools, without implementing or training RSI, RL, VLA, Diffusion Policy or Agent manipulation policies. The scene must retain standard interfaces for future algorithms. Do not modify AnsonNing/RSI, any existing RSI working directory, or its environment.

Execution rules:
1. Read official version-compatibility documentation before changing code. Pin versions, dependencies, GPU/driver requirements, asset sources and usable APIs; do not mix examples from older Isaac Lab 2.x / Isaac Sim 4.x releases.
2. Prioritize physics and rendering feasibility. Build a blade-only cutting testbed with a kinematic tool and deformable material before integrating dual Frankas. A parallel testbed must specifically verify whether dynamic MPM surfaces can enter actual RGB-D outputs.
3. Fake cutting cannot replace the final physics. Pure heightfield reduction, direct geometry deletion, or removing a predefined rigid body when contacted are diagnostic baselines only, and cannot be called completed physical cutting. Removed material must have actual separation, independent subsequent dynamics and mass accounting.
4. Evaluate both Newton MPM and CRESSim-MPM in execution. Documentation alone is insufficient: provide reproducible minimal examples, an API/coupling-limit inventory and the same single-cut experiment. If a candidate cannot build, record its exact blocker and evidence.
5. Each work package requires code/configuration, tests, a minimal demo, commands, results (video or images), performance and known limitations. Update docs/PROGRESS.md when complete; record important design decisions in docs/adr/.
6. Do not invent passing tests, physical realism or speedups; explicitly mark unavailable GPUs/APIs BLOCKED / NOT_RUN and deliver usable CPU tests and reproducible scripts.
7. Respect repository authorization: if no GitHub remote is specified, create a local Git repository first; do not independently push, publish large models/assets or modify other projects. Reference external assets and third-party solvers under their actual licenses, preserving sources and commits.
8. Independently consult documentation, read source, run experiments and make reasonable engineering decisions. Seek human decisions only for changes to the core task definition, unavailable permissions/hardware, or required payment. For unavoidable technical limits, retain independently working parts and comparative evidence; do not rewrite acceptance criteria to conceal failures.
The first formal acceptance target is measurable persistent material separation by one blade in the same initially continuous soft specimen: chips detach and continue experiencing forces/collisions; the remaining surface is independently measurable from physics and RGB-D; reset, reproduction and quantitative validation are available. RL success rate is not the initial target.
1. Scope, core specifications and non-goals
1.1 Task definition
The workstation contains a table, a fixed soft-material specimen, two Franka Pandas, two RealSense RGB-D cameras and one global camera. The first Franka holds a knife and removes one or more bumps by shaving / planing / scraping the surface; the second initially remains safely parked. The original planar base and bumps must form one continuum, not separate rigid bodies prepared to fall off after contact.
- Main material: an elastoplastic approximation of Plasticine-like oil-based modelling clay. The user does not require clay exclusively; soft wax, foam and other materials may be added later, but the first version focuses on one material.
- Core physics: 3D deformation, plasticity, tool–material contact, explainable failure / separation, persistent cuts, chip dynamics and mass bookkeeping.
- Observations: primarily RGB-D; contact force/torque must not enter future policy observations. The simulator must still solve contact forces internally. Provide optional vision_only and vision_proprio presets; do not insert oracle particles, stresses or target planes into official observations.
- Robots: Franka × 2, initially arm_0 with a fixed tool and arm_1 parked. Leave room for another active arm and tool pickup/exchange later; autonomous tool grasping is outside the first version.
- Cameras: fixed D455 × 2 + global camera × 1. Global defaults to recording/debug only; implement ideal D455 depth first, then noisy depth consistent with the sensor model.
- Experiment modes: physics_only (kinematic blade without a robot), workcell (complete dual-arm scene), benchmark (unified step/reset/observation/metrics).
1.2 Out of scope this phase
- No RSI repository integration, autonomous agents, RL/imitation-learning/VLA training or algorithm-performance comparisons.
- No tactile sensors, policy-input force sensing, real-robot deployment, ROS 2 network control or simultaneous coordinated dual-arm cutting.
- Visual realism does not replace verifiable physics; an animation that deletes predefined material is not a material-separation result.
- Do not assume thousand-environment parallelism, real-time FPS or established sim-to-real transfer; prioritize single-environment physics quality, reproducibility and evaluation.
1.3 Layered acceptance (important)
Level	Meaning	Minimum evidence
PHYSICS_PROVEN	An actual cutting-physics prototype is established	Material separation, persistent cut surfaces, subsequent chip motion, mass accounting, numerical stability and complete logs
VISION_PROVEN	Changing material surfaces can be measured by cameras	Dynamic pre/post-cut RGB-D, depth geometry compared with the physical surface, timestamps/calibration
WORKCELL_PROVEN	A complete dual-arm workstation can safely drive the blade	Articulations, IK/Cartesian paths, no unexpected collisions and tool–material interaction
BENCHMARK_READY	Other researchers can directly run experiments	Reset/seed, observation/action specifications, variants, evaluation, CI/tests and reproduction documentation


Do not claim a higher level prematurely: a geometry-removal animation remains GEOMETRIC_BASELINE_ONLY even if the workstation and API are complete; it is not PHYSICS_PROVEN.
2. Recommended technical baseline and hardware diagnostics
2.1 Pinned compatible versions (as of 2026-10-27)
Component	Recommended baseline	Notes
Isaac Lab	v3.0.0 GA release	Prefer the official release/tag; record the complete commit SHA
Isaac Sim	6.2.0	For USD, Franka, RTX and RealSense D455; the entire workflow need not depend on Kit
Newton	1.6.1	Corresponding to the above release; MPM/MJWarp coupling is experimental
NVIDIA Warp	1.17	Follow officially pinned dependencies rather than mixing versions
Python	3.12	Follow the official environment
PyTorch / CUDA	Combination pinned by the Isaac Lab release	Release notes: Torch 2.11, CUDA 13; record the actual installation
CRESSim-MPM	Pin one tested commit	Second candidate, independently compiled; do not assume the same ABI as Isaac Lab


Official baseline: https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0 and https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html
2.2 Machine audit before project startup (first P0 step)
- Collect uname -a, OS, CPU, RAM, free disk, GPU model/VRAM, driver, CUDA runtime, nvcc, Python and nvidia-smi.
- Compare Isaac Sim 6.2 requirements; official x86_64 minimums list 32 GB RAM, 16 GB VRAM and an RTX GPU. Measure headroom for complex scenes. Missing RTX hardware does not prevent testing Newton MPM, but RTX/D455 rendering cannot be declared validated.
- Check asset links, network access and availability of D455/Franka USDs; preserve source information and hashes.
- Isolate Isaac Lab and CRESSim in separate venvs/containers; record driver/CUDA/CMake compatibility. Keep large datasets, videos, checkpoints and asset caches outside the repository to avoid committing binaries.
- Produce artifacts/environment/system_report.json and docs/COMPATIBILITY.md; execution-generated artifact files are not committed to Git.
2.3 Run official smoke tests first
From the official Isaac Lab v3.0.0 working directory:
uv run isaaclab example mpm-granular
uv run isaaclab example mpm-two-way-coupling
uv run isaaclab demo snowball-smash
uv run isaaclab demo teapot-fill --visualizer newton_gl --fluid_render_mode surface
If a command/extra differs between builds, follow the checkout's --help and official 3.0 documentation, and record the adjustment in this repository instead of ignoring the error. Record startup status, GPU memory, frame time, asset-download problems, log screenshots and exact commands for every test.
3. System architecture (decoupled physics, rendering and tasks)
                     Project configuration (YAML/Hydra)
                                  |
                     Parameterized scene generator
           +----------------------+-------------------+
           |                      |                   |
    Robot/fixture/tool       Material generator    Cameras / lights
   (USD / articulations)   (volume -> particles) (D455 x2 + global)
           |                      |                   |
           +--------------+-------+                   |
                          v                           |
                   Cutting physics                    |
                +---------+---------+                 |
                |                   |                 |
            Newton MPM        CRESSim testbed          |
          + cutting extension   + adapter              |
                |                   |                 |
                +---------+---------+                 |
                          |                           |
            Physical state + separated chips          |
                          |                           |
                Dynamic surface extraction ------------+
                          |
                RGB / Depth sensor outputs
                          |
             Environment API / evaluator / logger
                          |
               Future: RL / Policy / Agent
                    (NOT this phase)
3.1 Data contracts that must be fixed
- SceneSpec:robot placement, camera extrinsics, fixture, tool geometry, material geometry seed, solver config.
- MaterialState: particle positions/velocities/masses/volumes, constitutive state, separation labels/damage variables if supported, and chip metadata; excluded from official observations.
- CuttingToolState: blade-edge frame, pose, twist, tool collider ID and actual swept tool volume.
- RenderState: reconstructed surface geometry, version/timestamp and alignment between visual/collision representations.
- Observation: rgb[d455_0, d455_1], depth_m[d455_0, d455_1], camera intrinsics/extrinsics, timestamps and optional robot proprioception.
- Metrics: remaining bump volume, overcut volume, chip mass, mass-conservation error, separation state, maximum penetration, solver residual and elapsed time.
- EpisodeManifest: version SHA, config digest, seed, physics/renderer backend, asset IDs, environment mode and test results.
Use explicit SI units (m, kg, s, Pa) in every module and output camera depth in m. Document coordinate conventions and quaternion order in docs/COORDINATES.md. Implement explicit conversions and unit tests at Isaac/external-solver boundaries.
3.2 Solver interface (extend only as needed)
Do not prebuild a large abstraction framework for nonexistent backends; a thin interface in the first cutting PoC should still permit comparison:
class MaterialCuttingSolver:
    def initialize(self, specimen_spec, material_params, tool_spec): ...
    def reset(self, initial_state): ...
    def step(self, dt, blade_pose, blade_twist): ...
    def get_material_state(self): ...
    def get_surface_data(self): ...
    def get_diagnostics(self): ...
Do not put actual Franka articulations directly into this interface: validate the solver with a kinematic blade first, then exchange rigid states and forces through a coupling adapter in the full scene.
4. Material construction and scene parameters
4.1 First continuous specimen
Suggested starting sample (engineering initial values only, to be adjusted through spacing/cutting experiments):
- Specimen: a 120 × 80 × 15 mm continuous base supported by a rigid tray/side stops; do not lock every particle.
- Bump: 1 smooth dome, approximately 20–30 mm in diameter and 5–10 mm above the original plane at its peak.
- Goal: scrape/shave excess material above the bump toward the reference plane; multiple passes are allowed, but every cut must be measurable.
- Initial tool: rigidly fix the blade to a kinematic proxy so grasping errors do not interfere with physics validation; mount it to Franka's tool frame after arm integration. A rigid attachment does not automatically provide contact-force feedback; verify coupling separately.
Define the uncut shape with a continuous height function h(x,y), then volume-sample material points on a grid / particles-per-cell within z_bottom ≤ z ≤ h(x,y), preserving volume connectivity. Add parameterized gaussian_bumps, ellipsoid_cap and ridge families; test reproducibility for the same seed and variation across seeds.
4.2 Initial material model
Approximate Plasticine-like behavior with a deformable/plastic constitutive model, including density, Young's modulus, Poisson ratio, yield/cohesion/hardening, contact friction and optional adhesion/rate dependence. Do not present these as measured parameters for any commercial clay; mark configs/materials/plasticine_provisional.yaml uncalibrated.

Perform three independent material-calibration/plausibility experiments:
1. Free settling/support: no severe nonphysical collapse, penetration or self-oscillation.
2. Flat-punch indentation and unloading: irreversible plastic deformation; preserve force–displacement curves for future calibration, outside Policy observations.
3. Single-blade lateral advance: plausible shear, contact and local flow, with stable stepping after contact.
4.3 Randomization (after the full solver passes)
- D0: one fixed bump with fixed material, lighting and cameras.
- D1: one bump with randomized position, height, width/orientation.
- D2: 2–5 bumps mixing Gaussian domes, ridges and irregular shapes.
- D3: perturb material properties, initial positions, camera extrinsics, lighting, blade angles and related parameters.
Sampling must not produce initial states penetrating the table/tool, unreachable by the robot, or violating the specimen definition. Record actual sampled parameters, particle counts and material mass on every reset.
5. Cutting Physics Solver: selection, experiments and improvements
5.1 Required physical phenomena
A cutting module must distinguish:
- Plastic flow (deformation): shape change does not itself establish cutting.
- Material failure: local failure based on yield, accumulated plastic strain and damage/fracture criteria.
- Displacement discontinuity (separation): motion degrees of freedom/grid transfers on opposite sides must not be incorrectly coupled.
- Persistent discontinuity: separation must not disappear after blade withdrawal; cut faces may recontact but must not reweld.
- Post-cut dynamics: chips retain mass and momentum and can fall, rotate, slide and contact the table.
- Mass conservation and traceability: record the reason and mass destination for any particle inactivation/deletion/spawning.
Not final solutions: directly deleting meshes/particles along the blade path; pre-cutting bumps into independent bodies; masking bumps only during rendering; updating heightmaps without updating actual physical contact.
5.2 Candidates
ID	Approach	Required execution checks	Main risks
S-A	Newton implicit MPM + custom failure/persistent separation	Official MPM, collisions and solver extension interfaces; whether CPIC/side labels can enter internal P2G/G2P or contact	Invasive solver changes and particle/grid compatibility
S-B	CRESSim-MPM (MLS/PB/Standard MPM, cutting geometry) + adapter	Successful build, QuadSlicer-like examples, persistent cuts/chips, particle export and C API	Not thread-safe; external rigid coupling, mesh restrictions and camera integration
S-C	DiSECt/FEM cutting (research reference / conditional prototype)	Whether virtual nodes + cutting springs fit surface trimming better than bisection	Geometry remeshing, old versions, licensing and integration cost
S-D	Reproduce CPIC/BladeMaster/STC-MPM techniques	Persistent side labels, damage evolution and tool swept interfaces	Redesign/reproduction required; directly usable code may be unavailable


Prioritize S-A and S-B; use S-C/S-D for algorithm review and targeted improvements. DiSECt source uses the NVIDIA Source Code License and its dataset also has CC BY-NC restrictions; confirm permissions before merging or redistributing. CRESSim-MPM is BSD-3-Clause and taichi_mpm is MIT; preserve notices and attribution.
5.3 Unified solver bake-off tests
Compare using the same specimen.json, blade_trajectory.csv, initial material parameters and units:
- T1 indent_unload: material plasticity and rebound.
- T2 shallow_scrape: blade contact/local shear; distinguish pushing from actual removal.
- T3 through_bump_cut: blade moves near the target plane and produces chip separation.
- T4 blade_withdrawal: cut surfaces remain independent after the blade retreats.
- T5 two_pass: two nonparallel cuts, chip recontact and unintended rewelding.
- T6 grid_dt_refinement: at least three grid/timestep configurations, measuring convergence rather than comparing animation alone.
- T7 reinitialize: same-seed reset reproduces initial particles, internal state and cutting outcomes within numerical tolerances.
- T8 surface_export: post-cut surfaces/chips export to external geometry for cameras/analysis.
Save raw data, curves, frames, failure classifications, timing, mass errors, GPU VRAM, simulation seconds / wall seconds, versions and commits for every test.
5.4 Solver hard requirements and weighted scoring
Pass these hard requirements before weighted comparison:
- At least one through_bump_cut case creates persistent material separation, allowing chips and substrate to move independently.
- No hidden large mass loss, explosive velocities or systematic penetration.
- Reset and particle/separation-state export are available; define a surface suitable for D455 rendering.
- Legal, maintainable licensing and an API/kernel modification path exist.
After passing: separation physics 30%, numerical/material behavior 15%, surface/camera integration 15%, rigid coupling 15%, stability/reset 10%, computational efficiency 10%, maintenance cost 5%. List measurements and rationale in docs/reports/SOLVER_SELECTION.md.

If neither S-A nor S-B passes the hard requirements, suspend formal solver selection, reproduce CPIC/persistent labels on a narrower material-separation problem and deliver a clear failure report. Do not substitute geometry deletion and claim success.
5.5 Resolution and timesteps
- Start exploratory voxel_size near 4 mm, then test 2 mm and 1 mm configurations; this is a test matrix, not a promise that 1 mm is executable.
- Choose substeps/dt from contact stability, blade thickness and grid convergence; do not promise 240 Hz or true real-time performance in advance.
- Configure physics, control and camera steps separately; record timestamps and synchronization, preventing cameras from showing the previous physics step's cut surface.
- If cutting needs sub-grid accuracy, compare tool SDFs, proxy contact and local geometry reconstruction; do not shrink the mesh while ignoring computational costs.
6. Dynamic surface reconstruction and camera imaging (second priority research track)
6.1 Questions to verify first
Newton MPM state consists of particles. Official ParticleSurface / ParticleSurfaceRenderer can reconstruct and render surfaces, but documentation states that Newton GL/RTX visualizers support reconstructed surfaces while the Kit visualizer directly displays MPM particles. An attractive visible surface and a new cut surface readable by a D455 depth annotator are separate acceptance conditions.

Build these two independent PoCs:
1. Newton renderer path: build ParticleSurface from the official teapot-fill surface-rendering example; export post-cut vertices/triangles/normals and timestamps. Test experimental camera output and record the renderer used.
2. Isaac Sim RTX / D455 path: supply the same time-varying surface to an Isaac Sim render product and obtain before/after RGB/depth through official RtxCamera and SingleViewDepthCameraSensor APIs. If intermediate dynamic mesh staging is needed, measure update latency, GPU memory, scene synchronization and validity first.
Any path providing only GUI images without programmatically readable RGB-D tensors is VISUALIZER_ONLY, not VISION_PROVEN.
6.2 Camera plan
- Official Isaac Sim 6.2 RealSense D455 asset: Isaac/Sensors/RealSense/D455/rsd455.usd, resolved against the actual asset root. Ideal D455 range is approximately 0.6–6 m; minimum depth at maximum resolution is approximately 0.52 m.
- D455_0: fixed elevated oblique camera, initially approximately 0.65–0.85 m away, covering the specimen and full tool path.
- D455_1: fixed front-side camera, initially approximately 0.6–0.85 m away, viewing blade entry and chips. If millimeter depth cannot be reliably estimated at close range, do not put D455 inside its minimum working distance; use ground-truth meshes for additional physics QA.
- Global: wide-angle recording/debug camera, excluded from official observations by default.
- Begin with configurable 640x480 resolution for testing; follow official specifications separately for physical D455 pixels and postprocessing options.
Provide ideal_depth and realsense_modeled_depth presets. The former validates geometry/occlusion; the latter uses supported stereo postprocessing, noise and disparity/outlier logic. Preserve intrinsics, extrinsics and depth encoding for both, including units in m and invalid-pixel definitions.
6.3 Vision QA
- Fixed geometry checkerboard/calibration board: approximately correct intrinsics, extrinsics and frame mappings.
- Static plane: ideal-depth/geometric-distance error stays within explainable numerical/renderer limits.
- Dynamic bump: height changes before/after deformation are identifiable from the camera viewpoint.
- Actual cut surfaces: post-cut geometry agrees with RGB-D cut boundaries/chip positions; rendering stale surfaces is unacceptable.
- Test valid masks, occlusion regions, tool masks and discontinuity boundaries; correct mean depth must not conceal missing cuts.
- Compare physical truth using ideal_depth first; report realsense_modeled_depth error separately, without substituting sensor-noise artifacts for geometry changes.
7. Dual-Franka workstation (integrate after Physics / Vision proof)
7.1 Scene assets
- Franka Panda × 2: use official USD/Isaac Lab configurations; check articulation, joint-drive and collision support for the selected assets under newton_mjwarp.
- Table:rigid/static support.
- Fixture: tray/side mechanical stops prevent the whole specimen being pushed away; validate support through indentation/lateral-push experiments.
- Knife: high-quality visual mesh and separately configurable collision mesh/SDF; record edge thickness, blade angle, effective edge length, friction and tool frame.
- Cameras x3: each has a fixed pose, intrinsics, extrinsics and render preset. Define workstation world and tool/specimen local frames.
SimReady Central can help locate OpenUSD tables, fixtures, supports and robots, but assets lacking collision profiles for the chosen physics backend must be corrected/replaced. It is not a cutting engine.
7.2 Staged coupling
- workcell_static: load both Frankas without cutting; validate poses, joint limits, collisions, IK reachability and camera occlusion.
- kinematic_blade: drive the material solver with a kinematically specified blade path for deterministic physics tests.
- coupled_arm_blade: bring arm_0 and its blade in only after solver validation. Start from a maintained Newton MJWarp + MPM proxy-coupling example; initially use lagged coupling, compare one-way/two-way force-feedback motion, then adjust coupling iterations.
- Initially lock arm_1 in a verified collision-free park pose that does not block important views; removing the second arm entirely does not satisfy the specification.
MPM itself does not solve articulated rigid dynamics: directly prescribing blade poses proves deformable contact but not reaction-force coupling to Franka. A robot cutting material requires verified rigid/deformable bidirectional feedback, with force excluded from future policy observations.
7.3 Tool and control interfaces
Provide a deterministic scripted trajectory player first; learning is unnecessary:
- joint_position: articulation joint commands for both arms.
- cartesian_pose / cartesian_delta: arm_0 tool-frame targets with IK/controller; public observations remain vision-first.
- kinematic_blade_trajectory: physics QA only, reading CSV time/pose/velocity without Franka for comparable solver tests.
Record robot-controller solutions and knife-tip poses separately so control errors are not mistaken for material-physics errors.
8. Benchmark API and research data (build now, without model training)
8.1 Intended public API
A Python facade can provide Gymnasium-like semantics:
sim = CuttingEnv(config)
obs, info = sim.reset(seed=42)
obs, reward, terminated, truncated, info = sim.step(action)
frame = sim.render()
sim.close()
This is the target API, not a claim that Isaac Lab already provides an identically named CuttingEnv. Implement it in this repository with a minimal example. Reward may initially be a fixed benchmark score; no training or reward-shaping optimization is required this phase.

Observation profile:
- vision_only: two D455 RGB/depth streams, valid masks, timestamps and calibration; no force or ground-truth physical state. The executor may still use robot encoders internally for low-level control.
- vision_proprio: the above plus q, qdot and gripper/tool pose for a future baseline that may converge more easily.
- debug_oracle: particles, surface meshes, target planes, material parameters and true chip labels, for research/evaluation only; do not mix with the first two profiles.
- global_camera: recording/debug, optionally enabled for other experiments by config, but excluded from official benchmark observations by default.
8.2 Required metrics
For a specified ROI, reference plane z0 and actual post-cut surface h(x,y):
- remaining_bump_volume = ∬ROI max(0, h(x,y)-z0) dxdy
- overcut_volume = ∬ROI max(0, z0-h(x,y)) dxdy
- remaining_height_p95 = P95(max(0, h-z0))
- roughness_rmse = sqrt(mean((h-z0)^2)) (explicitly specify valid ROI/mask)
- chip_mass, substrate_mass, material_mass_error:|M0-(M_substrate+M_chips+M_accounted)|/M0
- persistent_separation_valid: verify using actual solver cut surfaces/side labels/connectivity and subsequent motion, rather than pixel distance alone.
- blade_material_penetration, nonfinite_count, solver_residual, physics_walltime, render_walltime.
If the surface is not a single-valued heightfield, for example curled material or overlying chips, use 3D meshes/signed distances/voxels/particles and score substrate and chips separately. Do not silently keep only the visible highest surface and distort scoring.
8.3 Episode and data output
- Give each episode a unique run_id, seed, scenario variant, solver name/version, renderer name/version, material/camera parameters, git SHA and timestamp.
- Episodes may export manifest.json, metrics.json, physics traces (npz/parquet as needed), camera images, low-rate previews, failure screenshots and solver logs.
- Put rendered RGB-D and oracle physics state in separate paths to prevent misuse by future training pipelines.
- Each module documents fields, dtypes, shapes, frames, units and missing-value handling.
9. Experimental acceptance and quality thresholds
9.1 Required test checklist
Test ID	Test	Required evidence	Acceptance logic
PHY-01	Material settling	Drift/penetration/time-series	No explosions, stable shape and acceptable drift relative to target geometry
PHY-02	Indentation/unloading	Load–displacement and residual deformation	Plastic/elastic components can be exhibited and reproduced
PHY-03	Blade contact/friction	Blade poses, contact traces and cross-sections	No severe nonphysical penetration or solver crashes
PHY-04	Single cut	Cut surfaces, chip mass and separation labels	Actual separation with different subsequent chip/base motion possible
PHY-05	Blade withdrawal and cut-face recontact	Withdrawal + rest traces	Contact allowed without automatic welding into a continuum
PHY-06	Mass accounting	Mass-budget CSV	No unexplained mass loss; calibrate thresholds through material/discretization tests
PHY-07	Grid/dt convergence	Metric curves for 3 configurations	Primary observables approach convergence under refinement, across more than one seed
PHY-08	Reset reproducibility	Repeated-seed results	Reproducible within tolerances under the same backend/GPU conditions
VIS-01	D455 static depth	Ideal depth vs known plane	Correct units/extrinsics/valid masks
VIS-02	Dynamic material surfaces enter cameras	Before/during/after RGB-D	Programmatically readable tensors, not viewer-only images
VIS-03	Dynamic cut/chip geometry	Depth + physical-surface overlays	New surfaces, occlusions and chips in correct camera coordinates
ROB-01	Dual-arm scene	Stage, articulations and collisions	Both arms load correctly and the parked arm has no unexpected collisions
ROB-02	arm_0 executes a fixed blade trajectory	Tool-pose logs and video	Reachability and quantifiable position error
ROB-03	Coupled rigid reaction	Kinematic/coupled A/B	Explainable mechanical feedback from bidirectional coupling, not a policy force sensor
ENV-01	reset/step/observe/render/close	API smoke and repeated operation	No stale state; explicit exit/error reporting
ENV-02	D0–D2 procedural variants	Seed/shape sample set	Randomization preserves valid initial states
ENV-03	Separate benchmark/scoring	Observation-contract tests	vision_only excludes oracle state/force


9.2 Defining numerical thresholds
Do not already claim fixed mass/force/depth accuracy. In the P2 research report, Codex should propose achievable thresholds and rationale from measurements; for example material_mass_error <= 0.5% is an initial investigation target, but algorithms that do not conserve mass must not pass by widening tolerances. Define ideal_depth and modeled_depth limits separately, incorporating grid spacing and scene scale. Freeze all numerical thresholds before formal benchmarking and place reviewed values in config.

Report numerical convergence separately from matching real oil-based clay: without real material properties/experimental data, the strongest final claim is PHYSICS_VALIDATED_IN_SIMULATION, not completed real-world calibration.
10. Repository layout (suggested target; do not generate many empty shells at once)
RoboticCuttingSim/
├── README.md
├── AGENTS.md                         # Summary of section 0 and execution rules
├── pyproject.toml
├── uv.lock / requirements lock       # Choose according to the official installation strategy
├── .gitignore
├── configs/
│   ├── simulation/{physics_only,workcell,benchmark}.yaml
│   ├── materials/plasticine_provisional.yaml
│   ├── solvers/{newton_mpm,cressim_mpm}.yaml
│   ├── cameras/{ideal,d455_modeled}.yaml
│   └── tasks/{d0,d1,d2,d3}.yaml
├── assets/
│   ├── tools/                        # Project-owned tool USD/meshes
│   ├── fixtures/                     # Project-owned fixture USD/meshes
│   └── ASSET_SOURCES.md              # Official/third-party sources, licenses and commits
├── src/robotic_cutting_sim/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── scene/{workcell,robots,fixture,tool}.py
│   ├── materials/{geometry,particles,constitutive}.py
│   ├── physics/{base,newton_mpm,cressim_adapter,cutting}.py
│   ├── rendering/{surface,camera,d455_bridge}.py
│   ├── envs/{cutting_env,observations,actions}.py
│   ├── evaluation/{metrics,validation,recording}.py
│   └── utils/{transforms,units,seeds}.py
├── scripts/
│   ├── doctor.py
│   ├── run_physics_probe.py
│   ├── run_solver_comparison.py
│   ├── validate_depth.py
│   ├── launch_workcell.py
│   └── run_benchmark.py
├── tests/
│   ├── unit/                         # CPU-compatible tests
│   ├── integration/                  # GPU / IsaacLab
│   └── physics/                      # Strict material/cutting tests
├── experiments/
│   ├── materials/
│   ├── cutting/
│   └── renderer/
├── docs/
│   ├── SETUP.md
│   ├── COMPATIBILITY.md
│   ├── COORDINATES.md
│   ├── PHYSICS_VALIDATION.md
│   ├── CAMERA_PIPELINE.md
│   ├── BENCHMARK_SPEC.md
│   ├── PROGRESS.md
│   ├── adr/
│   └── reports/
└── artifacts/                         # gitignored: outputs, videos, traces
Use .gitignore for large files, downloaded asset caches, GPU traces, videos and models. Keep small reproducible scene/test fixtures; obtain third-party code through pinned submodules/source references or reproducible builds, rather than copying license-incompatible implementations.
10.1 Intended final CLI (to be implemented by Codex; these commands are not already available)
python -m robotic_cutting_sim.cli doctor
python -m robotic_cutting_sim.cli physics-probe --case through_bump_cut --solver newton
python -m robotic_cutting_sim.cli physics-probe --case through_bump_cut --solver cressim
python -m robotic_cutting_sim.cli compare-solvers --suite cut_v1
python -m robotic_cutting_sim.cli camera-check --scene dynamic_clay
python -m robotic_cutting_sim.cli scene --preset workcell --seed 42
python -m robotic_cutting_sim.cli benchmark --variant D0 --seed 42 --trajectory reference
The full CLI need not exist initially; add subcommands with real end-to-end tests as each package is completed. Every command needs --help, nonzero error exit codes, machine-readable JSON and an explicit --output-dir.
11. Codex work packages and implementation order (follow dependencies)
WP-00 — Repo / hardware / compatibility (first priority)
- [ ] Create an independent Git repo, README, AGENTS and basic config/test structure; do not touch RSI.
- [ ] Machine diagnostics and an Isaac Lab 3.0 GA / Isaac Sim 6.2 compatible environment.
- [ ] Run official MPM smoke demos; record actual versions, commands, GPU memory and video.
- [ ] Confirm available D455/Franka assets, licenses and download strategy.
- DoD: another researcher can install from docs/SETUP.md or clearly reproduce the blocker; environment records and smoke logs are complete.
WP-01 — Procedural material and blade-only Newton testbed
- [ ] Implement continuous base+bump geometry, particle sampling, material-mass computation and deterministic-seed tests.
- [ ] Load plastic material, a rigid tray and kinematic blade in Newton MPM; run settling/indent/scrape.
- [ ] Implement a CSV tool-trajectory player and physics traces.
- DoD: reproduce material/tool interaction from commands without Franka; demonstrate plastic deformation/contact without prematurely claiming separation.
WP-02 — Cutting solver bake-off (first research milestone)
- [ ] S-A Newton: inspect internal extension points and identify missing P2G/G2P/side-label/damage support for separation.
- [ ] S-B CRESSim: pin a commit, build, run official cutting demos, and audit C API, material export, thread safety and mesh warnings.
- [ ] Run T1–T8; if both remain incomplete, minimally reproduce gaps in CPIC/cutting geometry/persistent discontinuity.
- [ ] Produce SOLVER_SELECTION.md, minimal videos, curves, source attribution and a decision ADR.
- Gate G2: evidence of actual severing, chip separation and postwithdraw persistence, with a selected primary solver; otherwise explicitly record the blocker and continue independent renderer/workcell work without claiming completed cutting.
WP-03 — Dynamic surface + RGB-D proof (can run alongside WP-02)
- [ ] Generate deformable-particle time series and test ParticleSurface extraction, mesh updates and normal/abnormal topology.
- [ ] D455 assets, sensors, ideal depth and modeled depth; both RealSense cameras and Global output correctly.
- [ ] Verify rendering uses the latest material state, including pre/post-cut depth overlays and errors.
- Gate G3: obtain continuous-material/chip RGB-D tensors through actual camera APIs; visualizer-only success is insufficient.
WP-04 — Dual Franka workcell and coupled cutting
- [ ] Official OpenUSD workcell (table, tray, tool, Franka×2, D455×2, Global), with path/asset metadata.
- [ ] Safely park the second arm; first-arm tool mounting, IK paths, joint-limit and collision checks.
- [ ] Bridge tool/material with the selected solver; for Newton, adapt a maintained MJWarp–MPM proxy-coupling example and validate two-way forces and stability.
- [ ] Measure contact/motion differences between kinematic and robot-coupled trajectories.
- Gate G4: an actual arm_0 cutting trajectory executes inside the scene, beyond a detached blade-proxy demo.
WP-05 — Benchmark, randomization, evaluation
- [ ] CuttingEnv reset/step/render/close, action/observation profiles.
- [ ] D0→D2 randomization; test seeds, valid initial states and distribution coverage across seeds.
- [ ] Use tool trajectories/scripted references for demonstration and regression, without policy training.
- [ ] Actual chip/substrate geometry and mass metrics, with separate RGB-D consistency checks.
- [ ] Track performance, reliability and config hashes; separate formal samples from tuning/dev seeds.
- Gate G5: a third-party researcher can launch an episode from README and obtain cameras/metrics without knowing solver internals.
WP-06 — Release / reproducibility / handoff
- [ ] CPU unit tests, GPU smoke/physics tests, known issues and external-asset sources/licenses.
- [ ] Images/short videos and complete architecture/material/camera/benchmark documentation.
- [ ] Measured performance table: GPU memory, stage wall times, simulation seconds / wall seconds and final per-episode cost.
- [ ] Freeze official versions and provide reproduce_demo.sh or an equivalent script.
- [ ] README walkthrough for new researchers, PR guide, test commands and a guide to adding scene materials.
- DoD: auditable evidence for PHYSICS_PROVEN, VISION_PROVEN, WORKCELL_PROVEN and BENCHMARK_READY. Mark any unmet condition separately rather than declaring the whole project complete.
11.1 Priority and parallel work
WP-00
  |
  +-- WP-01 ---- WP-02 (cutting solver bake-off) --+
  |                                               |
  +------------- WP-03 (dynamic surface/RGB-D) ---+--> WP-04 --> WP-05 --> WP-06
WP-04 need not wait for real-world calibration of every material parameter, but requires a credible basic separation implementation. If WP-02 is blocked, continue WP-03 and static workcell tests; cosmetic cuts cannot replace the final benchmark.
12. Development and collaboration rules
12.1 Research loop
Iterate each package through hypothesis -> implementation -> minimal test -> metrics -> diagnosis -> revision -> reproducible artifact. Preserve reproducible failures, configurations, data and minimal fixes for important problems; screenshots alone do not prove correctness.
12.2 Documentation and Git practices
- docs/PROGRESS.md: record date, commit, test execution/results, unresolved issues and next steps after every subpackage.
- docs/adr/: record decisions affecting architecture/solver/material/render backends and rejected alternatives.
- docs/reports/WP-XX.md: concise reports covering What changed / Commands / Evidence / Numbers / Known limitations / Next step.
- One PR/branch per independently verifiable package; do not combine large physics-solver, USD-asset and benchmark-API changes into one difficult review.
- Publish CONTRIBUTING.md, CPU-only unit tests, GPU-required markers and asset acquisition instructions for other researchers; do not commit keys, private data or huge logs.
12.3 Test layers
1. unit/cpu: geometry, numerical units, seeds, transforms, metric mathematics and config schemas; run in ordinary CI every time.
2. integration/gpu: Isaac/Newton loading, cameras, Franka and scene reset; run with compatible GPUs only.
3. physics/expensive: grid convergence, repeated cuts, multiple seeds and large RGB-D comparisons; run after key solver changes and before releases.
4. manual/visual: debugging images/videos linked to physical measurements, never standalone correctness tests.
Unsupported-hardware skips are not passes; reports list PASS / FAIL / SKIP / BLOCKED / NOT_RUN with reasons.
12.4 Safety and version management
- Operate only in the new repository; check licenses, dependencies and sources before downloading third-party code or running untrusted examples.
- GPU training/simulation must not preempt other researchers' work on the machine; inspect utilization first. Resource-intensive runs must be interruptible and preserve results.
- Store research data/output by experiment/commit; never silently overwrite formal results.
- Choose a manageable complete version combination first; do not blindly upgrade the entire Isaac Lab major version for a new feature.
13. Risks, fallbacks and decision triggers
Risk	Detection	Engineering response	Prohibited shortcut
MPM pushes but cannot permanently sever	T3/T4/T5	Compare CRESSim, CPIC and persistent labels; customize the solver if needed	Masking material and claiming cutting is complete
Clay parameters are difficult to establish	Inconsistent indentation/cutting behavior	Use explicitly uncalibrated provisional material and maintain calibration tasks	Claiming agreement with real material
MPM particles cannot enter D455 RGB-D	VIS-02	Investigate surface extraction, dynamic mesh staging and renderer pipelines; record limits	GUI screenshots instead of sensor tensors
CUDA/GPU/RTX compatibility problems	WP-00 smoke	Follow official pinned configurations, isolate solver environments and run available CPU tests	Mixing older packages while ignoring ABI problems
CRESSim cutting creates no independent chips	T3/T4/T5	Inspect cutting semantics, add persistent separation or choose another approach	Treating a visible groove as success
Cutting runs too slowly	Benchmark profiler	Reduce scale/region, cache and profile GPUs; separate high/low fidelity	Claiming large-scale RL without measurement
Material history persists across reset	T7	Explicitly reset solver history and verify same-seed replay	Resetting positions while retaining plastic history
Large numbers of large files	Disk dashboard	Sample videos, use artifact TTLs, retain baseline data and representative failures	Committing raw dumps directly to Git


Newton explicitly notes that resetting independent MPM worlds requires solver-history handling; ordinary asset reset is not necessarily a complete reset. Add regression tests for this.
14. Specific first-round Codex deliverables
Complete WP-00 and WP-01 first and begin WP-02/WP-03 feasibility validation. Do not generate hundreds of template files, integrate every RL framework or mark all phases done in the first round.

Expected first-round delivery:
1. New local repo, AGENTS.md, README.md, docs/PROGRESS.md, docs/SETUP.md and version/hardware/license audits.
2. Successful official MPM examples or exact failure logs.
3. Code generating a continuous base with one bump, fixed fixtures and tool trajectories in a simplified scene, with deformation traces.
4. Initial Newton/CRESSim feasibility reports: minimal examples actually executed, cutting semantics and precise blockers.
5. A minimal ParticleSurface → camera-imaging verification plan and first results, distinguishing viewable surfaces from camera depth.
6. Priorities for the next package based on actual tests, rather than selecting a solver solely from this document's assumptions.
The first round does not require a complete dual-Franka scene, every camera, D3 randomization, final real-world calibration or RL training; these unfinished parts must not become prerequisites for an otherwise completed physical PoC.
15. Primary research sources (start with official/original sources)
Official versions/architecture
1. Isaac Lab 3.0 GA release:https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0
2. Isaac Lab 3.0 Implicit MPM:https://docs.nvidia.com/isaaclab/3.0/concepts/implicit-mpm
3. Isaac Lab Coupled Solvers:https://docs.nvidia.com/isaaclab/3.0/concepts/coupled-solvers
4. Isaac Lab Agent Skills (for Codex developing Isaac Lab, not RSI in this phase): https://docs.nvidia.com/isaaclab/3.0/developer-tools/agent-skills
5. Isaac Sim 6.2 release notes:https://docs.isaacsim.omniverse.nvidia.com/6.2.0/overview/release_notes.html
6. Isaac Sim 6.2 requirements:https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/requirements.html
7. Isaac Sim 6.2 D455 assets:https://docs.isaacsim.omniverse.nvidia.com/6.2.0/assets/usd_assets_depth_sensors.html
8. Isaac Sim 6.2 stereo-depth processing:https://docs.isaacsim.omniverse.nvidia.com/6.2.0/sensors/isaacsim_sensors_camera_depth.html
9. NVIDIA SimReady Central:https://huggingface.co/blog/nvidia/simready-central-assets-robotics-ai-factories
Cutting physics/materials
10. CRESSim-MPM(BSD-3-Clause):https://github.com/yafei-ou/CRESSim-MPM
11. MLS-MPM + CPIC(SIGGRAPH 2018, MIT reference):https://github.com/yuanming-hu/taichi_mpm
12. DiSECt (FEM/Virtual Nodes/Cutting Springs; check licensing): https://github.com/NVlabs/DiSECt
13. BladeMaster(2026, persistent discontinuities):https://jango6324.github.io/blademaster/
14. PlasticineLab:https://plasticinelab.csail.mit.edu/
15. RoboCook:https://hshi74.github.io/robocook/
Source-use rules: read each repository's README, actual source, license, issues and current APIs; do not assume paper implementations embed unchanged in Newton. Record cited versions, licenses and verified/unverified claims.
16. Required definition of completion (final acceptance contract)
Call the first version a research-ready cutting simulation environment if and only if every condition below is met:
- [ ] A continuous initial specimen and at least procedural D0/D1 variants.
- [ ] Plausible elastoplastic response with solver-supported persistent cut surfaces and separated chips.
- [ ] Subsequent dynamics/collisions among chips, substrate, tool and table, without large nonphysical mass loss.
- [ ] An independent physics-validation suite with convergence/reset/conservation experiments, conclusions and explicit measured limits.
- [ ] API-readable RGB-D observations of deforming material before/after cutting, beyond UI animation.
- [ ] A complete tabletop scene with two Frankas, two RealSense D455 cameras and Global camera; one arm executes a fixed cutting trajectory with a held blade.
- [ ] Verified two-way rigid/material interaction in robot-coupled mode, without force in official observations.
- [ ] CuttingEnv supports reset, step, render, record, score and fixed seeds, allowing future researchers to connect different policy frameworks.
- [ ] Clear installation, versions, GPU requirements, reproduction scripts, tests and contribution instructions.
If some conditions remain unmet, still deliver usable physics testbeds, camera workcells or APIs, marked accurately PARTIAL / BLOCKED; do not lower the conditions to fake cutting and declare completion.
Ready-to-use Codex startup instruction
Use RoboticCuttingSim_Implementation_Plan.md as the execution plan for a new independent repository. Read sections 0, 2, 9, 11, 14 and 16 first, create the local Git repo and complete WP-00; then implement WP-01 and begin WP-02 Newton/CRESSim cutting-prototype comparisons and WP-03 ParticleSurface-to-RGB-D feasibility tests. Independently consult official documentation/source, implement and test in small steps, produce reports per package and update PROGRESS. Do not modify RSI, train Policy/RL/Agent models, substitute geometry deletion for real cutting, or claim completion without test evidence. If GPUs/assets are unavailable, report exact blockers, retain executable independent work and propose the next minimal verifiable action.
