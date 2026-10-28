# Policy, RL and agent control (WP-10)

How a policy, an RL agent or an LLM agent drives the knife, and what each
piece guarantees. Everything here is opt-in; the historical API and the frozen
v1 tasks, physics and scores are unchanged unless you select the new options.
Status of each item (CPU-verified vs. GPU/live NOT_RUN) is in
[WP-10 specification](WP10_SPECIFICATION.md).

## Tasks and tool

| Task | Goal | Tool | Notes |
| --- | --- | --- | --- |
| `CUT-01`, `TRIM-01` | sever / flatten | retained wedge | frozen v1 |
| `CUT-01-R` | sever the 160 x 30 x 20 mm bar | `ruler_knife_v1` | CUT-01 thresholds |
| `TRIM-01-R` | remove the dome bump | `ruler_knife_v1`, 25 deg rake | TRIM-01 thresholds |
| `CUT-02` | sever a 160 x 60 x 25 mm bar | `ruler_knife_v1` | CUT-01 thresholds |

`ruler_knife_v1` (`tool_profile.py`): 120 mm straight edge, 40 mm deep, 1.5 mm
steel, 25 deg double bevel, 100 x 22 x 40 mm handle along the spine with the
edge center on the grasp axis. Colliders, mass, the TLMPM blade polygon and the
hand-to-edge mount all derive from this one profile. v2 tasks
(`configs/cutting_tasks_v2.json`) inherit their base task's v1 thresholds.

## Actions

The env (`live_env.CuttingTaskEnv`) accepts per step one mapping:

```python
{"mode": "cartesian_edge_delta", "target": [dx, dy, dz, rx, ry, rz],   # m, rotation vector (rad)
 "frame": "world" | "edge",          # optional; default = env delta_frame
 "cut_enabled": True,                # arm cutting
 "cut_index": 0}                     # optional persistent cut digit (0 or 1; material max_cuts)
{"mode": "cartesian_edge_pose_world", "target": [x, y, z, qx, qy, qz, qw]}  # absolute edge pose
{"mode": "joint_position_rad", "target": q7}                                  # live runtimes only
{"gripper": "hold" | "close" | "open" | "release"}                             # physical_grasp only
```

The edge frame: +x is the cutting (advance) direction, y runs along the edge,
z is the blade normal. Delta rotations turn about the edge center.

`ActionLimits()` (default 2 mm and 2 deg per 20 ms step, plus a workspace box
in workpiece coordinates) bounds delta actions, and absolute ones when passed
as `action_limits=`. `ik_failure="reject"` turns an unreachable target into a
hold at the last accepted target instead of an exception;
`max_consecutive_ik_failures=N` terminates after N rejections. Every step's
`info` has `action_report` (requested/applied motion, which limit clamped),
`ik_status`, `termination_reason`.

## Observations

`observation_profile="state"`: measured proprioception only (fast);
`"vision_proprio"`: wrist RGB-D + proprioception; `"vision_only"`. In
`physical_grasp` mode the held tool's true pose is privileged and is not
observed (hand FK and finger encoders are). Force, particles and cut labels
never enter observations.

## Reward

`reward.CuttingReward(task)` (`cutting_reward_v1`, provisional shaping, uses
privileged state): progress = fraction of target-plane columns with a severed
reference bond (labels on one side earn nothing), minus off-target severed
bonds, gouge below a flatten plane, sideways blade motion inside the
workpiece, IK rejections, clamping, termination and time. Pass
`reward=CuttingReward(task)` to the env. Success is still `env.evaluate()`.

## Gym (RL)

```python
from robotic_cutting_sim.gym_env import CuttingGymEnv
gym_env = CuttingGymEnv(env)   # actions: Box[-1,1]^6 scaled by ActionLimits, cutting armed
```

Observations: a flat state vector (`state`) or a Dict with the state vector and
each wrist camera's RGB and depth (`vision_proprio`).

## Scripted motions (policy baselines)

`policies.WaypointPolicy(task, waypoints)` reads only observations and emits
bounded world-frame deltas. Builders: `through_cut(task, tilt_deg=, yaw_deg=,
roll_deg=, saw_amplitude_m=, cut_speed_m_s=)` for push, oblique, yawed, rolled
and sawing cuts; `skim(task, speed_m_s=, saw_amplitude_m=)` for surface trimming.

## LLM agent tools

`agent_api.CuttingAgentAPI(env)`: `describe_task()`, `reset()`, `observe()`,
`status()`, `save_views(directory)` (current RGB views as PNG files for the agent to read), `move_edge_to(position_m, rpy_deg)`, `move_edge_by(delta_m,
delta_rpy_deg, frame=)`, `cut_along("advance" | world_vector, distance_m,
speed_m_s=, saw_amplitude_m=)`, `wait(steps)`, `finish_episode()`,
`set_gripper(cmd)`, `evaluate()`. Each returns a JSON-serializable dict and
executes as bounded delta actions, so the same limits and IK handling apply.

## Kinematic backend (CPU/GPU, no robot)

```python
from robotic_cutting_sim.kinematic_env import make_kinematic_env
env = make_kinematic_env("CUT-01-R", particle_spacing_m=.004, material_dt=.0005, device="cpu",
                         action_limits=ActionLimits(), ik_failure="reject")
```

The edge goes exactly where commanded (`runtime_source =
kinematic_ideal_tracking`). Use it to validate controllers, pre-train RL and
probe the cutting law; confirm on a live runtime (reachability, grasp, slip,
cameras are absent here). Full-episode checks:
`python scripts/run_kinematic_validation.py --out FILE`; a particle/knife
video: `python scripts/render_kinematic_episode.py --task CUT-01-R
--controller saw_policy --out FILE.mp4` (not a robot video).

## Cutting law 2 (`tlmpm_v2`, provisional)

v2 tasks use it by default; v1 tasks keep the frozen law. Cutting work counts
push work and drag along the edge (slicing/sawing), weighted by how much of the
edge's relative motion lies in the blade plane; separation is written only
where the edge has actually cut (energy-gated). Sideways pushes and flat
presses sever nothing; slicing needs far less push force; the handle is a
blunt obstacle. `blade_probe.drive_blade` runs kinematic probes. Parameters
are uncalibrated (the v1 material is very soft: E = 50 kPa, Gc = 0.05 J/m^2).
