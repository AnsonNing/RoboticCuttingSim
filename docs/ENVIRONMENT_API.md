# Research environment API

This is the original Newton deformation baseline. For articulated persistent
cutting and full-scene live RGB-D use [ResearchCuttingEnv](RESEARCH_ENVIRONMENT.md).

`CuttingEnv` advances Newton MPM and a kinematic blade with live ideal RGB-D. The
separate USD scene includes static robot/sensor assets. There is no training framework.

```python
from robotic_cutting_sim.env import CuttingEnv, EnvConfig
with CuttingEnv(EnvConfig(case="through_bump_cut", variant="D1")) as env:
    obs, info = env.reset(seed=42)
    obs, reward, terminated, truncated, info = env.step()
    obs, reward, terminated, truncated, info = env.step([-.04,0,.03,0,0,0,1])
    global_rgb = env.render()
```

`reset` constructs a fresh model/state/solver and surface context. None action
plays the reference trajectory; a7-vector is tool-center XYZ[m]+quaternion XYZW
in world frame, reached over one control step with linear translation/SLERP.
Switching back to None resumes the time-based script, rather than holding pose.
`observe` returns the current surface/timestamp; step after truncation requires reset.

Observation keys: rgb,depth_m,valid_mask,calibration,timestamp_s. Each camera
dictionary has d455_0/d455_1, with calibration explicitly labeling ideal pinholes.
Arrays are640×480 RGB uint8, depth float32 optical Z[m], mask bool. Global is
returned by render() only. Live views contain material/current tool proxy/tray/table;
Franka is in the workcell USD and not in this ideal renderer. Tool occlusion is
included. Oracle particles/stresses/forces never enter formal observation.
`debug_material_state()` is an explicit separate API.

vision_proprio requires an explicit real proprioception callback; the helper
permits q/qdot/tool_pose and filters other fields. Authored nominal park values
are not fabricated as measured state. Default is vision_only.

Reward is0, terminated=false, truncated follows max_steps. These are research-loop
semantics, not a task score. Info reports timings and supported capabilities.
There are no training, RSI, checkpoint or autonomous-agent paths.

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python -m robotic_cutting_sim.cli env-demo --steps 120 --output-dir artifacts/live_fresh
```

Output separates observations/,global_debug/,surface/ and oracle_debug/. Sparse
measured material/tool snapshots can be combined with the workcell in USD replay.
The first live120-step episode executed in8.215s wall for2s simulated, including
reset and periodic output. This is not a real-time guarantee.

The integrated handoff with9 saved material/surface snapshots also executed
120steps and exported a complete workcell replay. A separate actual-GPU API
check passed direct blade actions, truncation, D1 same-seed reset and close:

```bash
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 .venv/bin/python scripts/check_env_api.py artifacts/api_fresh
```

Reference local handoff: artifacts/research_handoff_001. The integrated script
and composed-geometry/tool-timestamp checks passed; raw arrays/videos remain local.
