# Supported v7 research workflow

For the measured Straight CUT configurations, select **`v7_cut_research_v1`** in the normal Scene/Goal API.
It uses the original, source-bound `tlmpm_v7`, the frozen controller and sufficient completion time.
Untagged scenes retain v5 / Law 8 as the legacy baseline. Selection never falls back during execution.
The profile does not qualify TRIM, arbitrary policy actions, all parameter combinations or a random success rate.
[Measured scope and failures](reports/WP14_RESULTS.md) · [immutable profile](../configs/wp14/v7_cut_research_v1.json).

```python
from robotic_cutting_sim.scene_config import compile_task, configured_task_payload, task_from_payload

# Configuration only: no robot or material simulation is started.
task = compile_task("configs/wp12/scenes/bar_flat_board.json", "sever_mid",
                    solver_profile="v7_cut_research_v1", seed=42)
assert task.solver_module == "tlmpm_v7"
assert task.duration_s == 14.0
assert task_from_payload(configured_task_payload(task)).task_id == task.task_id
```

Alternatively put `"research_profile": "v7_cut_research_v1"` beside `"module": "tlmpm_v7"` in the scene's
`solver` object. The normal compiler supplies the profile controller; contradictory tagged solver/law,
unsupported geometry, numerical or material overrides raise. A sufficient requested duration is retained;
a shorter duration receives the existing paired-driver completion margin. Caller dictionaries are copied.
The previous `compile_research_task` helper remains compatible.

The same selection is available at environment creation:

```python
from robotic_cutting_sim.live_env import open_cutting_env

# Run in the pinned Native environment, with a fresh output directory.
with open_cutting_env(runtime="isaac_sim", preset="cutting_research_v1",
                      scene="configs/wp12/scenes/bar_flat_board.json", goal="sever_mid",
                      solver_profile="v7_cut_research_v1", seed=42,
                      material_dt=0.0001, output_dir="artifacts/my_v7_native") as env:
    observation, info = env.reset(seed=42)
    # Supply actions through the normal policy API; new policies need evaluation.
    # observation, reward, terminated, truncated, info = env.step(action)
```

`runtime="kinematic"` accepts the same scene/goal/profile; it has ideal tracking and no rendered cameras.
Without the policy preset, pass `observation_profile="state"` on that runtime. A compiled `task=` also works.
The compiled payload carries identical solver, material, controller and goal selection into Native,
recording and the read-only evaluator. Native runs validate it again before material evolution.
Force, cut codes and oracle material state stay outside deployable observations.
[Policy actions and recording](POLICY_READY_API.md).

For an existing complete scripted Native CUT, the maintained scene launcher remains available:

```bash
PYTHONPATH=src .venv/bin/python scripts/run_wp14_scene.py launch \
  --scene standard_cut --gpu 0 --out artifacts/wp14/my_new_cut
```

The two-protrusion follow-up uses an explicitly versioned, experimental interface solver, separately from
this validated original-v7 CUT profile. Its results, camera configuration and limitations are documented in
[the follow-up report](reports/WP14_FOLLOWUP_RESULTS.md). Neither that experiment nor a functioning Native
TRIM movie establishes general TRIM support or changes the untagged v5 default.
