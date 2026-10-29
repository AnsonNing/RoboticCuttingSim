"""WP-11 Phase H: opt-in profiler (bitwise neutral), runtime profiles, Native in-place stage reset (USD part).

Kinematic episodes run on Warp CPU at coarse spacing; the Native reset itself is
validated on Isaac Sim separately (docs/reports/WP11_PHASE_H_PROFILING.md).
"""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
TRIM = dict(scene="trim01_wp11_workcell", goal="trim_bump_rho10",
            planner_options={"max_rotation_deg": 1.0, "max_translation_m": 0.001})


def _episode(out, steps=12, profiler=None, **options):
    """A short scripted episode through the policy env; returns (final codes/positions sha, evaluation, rows)."""
    from robotic_cutting_sim import goal_planner
    from robotic_cutting_sim.live_env import open_cutting_env
    from robotic_cutting_sim.policies import WaypointPolicy
    kwargs = dict(particle_spacing_m=.004, material_dt=.0005)
    kwargs.update(options)
    with open_cutting_env(runtime="kinematic", output_dir=out, preset="cutting_research_v1", device="cpu",
                          **TRIM, **kwargs) as env:
        if profiler is not None:
            profiler.install()
        try:
            observation, info = env.reset(seed=42)
            task = env.task
            plan_options = task.controller["planner"]
            policy = WaypointPolicy(task, goal_planner.plan(task.scene, task.goal, task.tool_profile, plan_options),
                                    task.action_limits(), **goal_planner.policy_options(plan_options))
            rows = []
            for _ in range(steps):
                observation, reward, terminated, truncated, step_info = env.step(policy(observation))
                rows.append((step_info["action_status"], np.asarray(observation["edge_command_world_xyzw"]).tobytes()))
            env.end_task("STEP_BUDGET")
            result = env.evaluate()
            state = env.backend.material.history_state()
            digest = {k: hashlib.sha256(np.ascontiguousarray(state[k]).tobytes()).hexdigest()
                      for k in ("positions", "velocities", "codes")}
        finally:
            if profiler is not None:
                profiler.uninstall()
        return digest, result, rows, info


def _strip(result):
    """Evaluation without wall-clock fields."""
    text = json.dumps(result, sort_keys=True, default=lambda v: np.asarray(v).tolist())
    doc = json.loads(text)

    def clean(node):
        if isinstance(node, dict):
            return {k: clean(v) for k, v in node.items() if "wall" not in k and k != "timestamp"}
        if isinstance(node, list):
            return [clean(v) for v in node]
        return node
    return clean(doc)


# ----------------------------------------------------------------------------- profiler

def test_profiler_install_uninstall_restores_every_attribute():
    from robotic_cutting_sim import profiling
    import warp
    targets = []
    for _, target in profiling.STAGES:
        try:
            owner, attribute = profiling._resolve(target)
        except (ImportError, AttributeError):
            continue
        targets.append((owner, attribute, owner.__dict__.get(attribute, None), attribute in owner.__dict__))
    launch, capture_launch = warp.launch, warp.capture_launch
    profiler = profiling.StageProfiler(kernels=True).install()
    assert profiling.StageProfiler._installed is profiler and warp.launch is not launch
    with pytest.raises(RuntimeError):
        profiling.StageProfiler().install()
    profiler.uninstall()
    assert profiling.StageProfiler._installed is None
    assert warp.launch is launch and warp.capture_launch is capture_launch
    for owner, attribute, original, own in targets:
        assert (attribute in owner.__dict__) == own, (owner, attribute)
        if own:
            assert owner.__dict__[attribute] is original, (owner, attribute)


def test_simulator_modules_do_not_import_the_profiler():
    for path in sorted((ROOT/"src/robotic_cutting_sim").glob("*.py")):
        if path.name in ("profiling.py",):
            continue
        assert "profiling" not in path.read_text(), path.name


def test_kernel_groups_cover_the_v5_substep():
    from robotic_cutting_sim.profiling import kernel_group
    assert kernel_group("_p2g") == "p2g" and kernel_group("_grid_update") == "grid_update"
    assert kernel_group("_claim_slots") == "slot_claim" and kernel_group("_g2p") == "g2p"
    assert kernel_group("_assign_band") == "label_stroke" and kernel_group("fixture_contacts") == "contacts"
    assert kernel_group("something_else") == "other_kernels"


@pytest.mark.cpu_sim
def test_profiler_is_bitwise_neutral_on_cpu(tmp_path):
    pytest.importorskip("warp")
    from robotic_cutting_sim.profiling import StageProfiler
    base = _episode(tmp_path/"plain")
    staged_profiler = StageProfiler()
    staged = _episode(tmp_path/"stages", profiler=staged_profiler)
    kernel_profiler = StageProfiler(kernels=True)
    kernels = _episode(tmp_path/"kernels", profiler=kernel_profiler)
    for other in (staged, kernels):
        assert other[0] == base[0]                      # positions, velocities, codes: identical bytes
        assert other[2] == base[2]                      # statuses and commanded edges per step
        assert _strip(other[1]) == _strip(base[1])      # evaluation (no wall times)
    summary = staged_profiler.summary()
    assert summary["control_steps"] == 12
    assert {"control_step", "backend_step", "material_advance", "tlmpm_substep_eager"} <= set(summary["per_control_step"])
    assert summary["per_control_step"]["control_step"]["share_of_control_step"] == pytest.approx(1.)
    kernel_summary = kernel_profiler.summary()
    assert {"p2g", "grid_update", "g2p", "slot_claim"} <= set(kernel_summary["kernel_groups"])


# ----------------------------------------------------------------------------- runtime profiles

def test_runtime_profiles_are_declared_and_versioned():
    from robotic_cutting_sim.runtime_profiles import load_runtime_profiles, runtime_profile
    doc = load_runtime_profiles()
    assert doc["version"] == "runtime_profiles_v1" and set(doc["profiles"]) == {"full_physics_v1", "research_fast_v1"}
    full, fast = runtime_profile("full_physics_v1"), runtime_profile("research_fast_v1")
    assert full["acceptance_valid"] is True and fast["acceptance_valid"] is False
    assert full["require"] == {"particle_spacing_m": .001, "grid_dx_m": .004, "material_dt_s": .0001, "control_dt_s": .02}
    assert fast["solver_override"] == {"particle_spacing_m": .002, "grid_dx_m": .008} and fast["material_dt_s"] == .0002
    limits = json.dumps(fast["fidelity_limits"])
    for number in ("0.816", "0.956", "3.0 mm"):
        assert number in limits
    with pytest.raises(ValueError):
        runtime_profile("fast")


def test_full_physics_profile_requires_the_formal_resolution():
    from robotic_cutting_sim.policy_env import compile_policy_task
    from robotic_cutting_sim.runtime_profiles import check_caller_overrides, check_task, runtime_profile
    full = runtime_profile("full_physics_v1")
    formal = compile_policy_task(TRIM["scene"], TRIM["goal"], planner_options=TRIM["planner_options"])
    assert check_task(full, formal, .0001)["particle_spacing_m"] == .001
    coarse = compile_policy_task(TRIM["scene"], TRIM["goal"], planner_options=TRIM["planner_options"],
                                 particle_spacing_m=.004)
    with pytest.raises(ValueError, match="requires"):
        check_task(full, coarse, .0001)
    with pytest.raises(ValueError, match="requires"):
        check_task(full, formal, .0005)
    for override in (dict(particle_spacing_m=.004), dict(backend_options={"material_dt": .0005}),
                     dict(backend_options={"history_stride": 5})):
        with pytest.raises(ValueError, match="fixes"):
            check_caller_overrides(full, **override)


def test_research_fast_scene_override_is_recorded_in_the_task_id():
    from robotic_cutting_sim.policy_env import compile_policy_task
    from robotic_cutting_sim.runtime_profiles import check_task, profile_scene, runtime_profile
    fast = runtime_profile("research_fast_v1")
    scene = profile_scene(TRIM["scene"], fast)
    assert scene["solver"]["particle_spacing_m"] == .002 and scene["solver"]["grid_dx_m"] == .008
    task = compile_policy_task(scene, TRIM["goal"], planner_options=TRIM["planner_options"])
    formal = compile_policy_task(TRIM["scene"], TRIM["goal"], planner_options=TRIM["planner_options"])
    assert task.task_id != formal.task_id and task.grid_dx_m == .008 and task.particle_spacing_m == .002
    assert check_task(fast, task, .0002)["grid_dx_m"] == .008


@pytest.mark.cpu_sim
def test_research_fast_episode_is_labelled_not_acceptance_valid(tmp_path):
    pytest.importorskip("warp")
    digest, result, rows, info = _episode(tmp_path/"fast", steps=6, particle_spacing_m=None, material_dt=None,
                                          runtime_profile="research_fast_v1")
    assert info["acceptance_valid"] is False and info["runtime_profile"]["profile_id"] == "research_fast_v1"
    assert info["runtime_profile"]["resolution"] == {"particle_spacing_m": .002, "grid_dx_m": .008,
                                                     "control_dt_s": .02, "material_dt_s": .0002}
    assert result["acceptance_valid"] is False and result["runtime_profile"] == "research_fast_v1"


@pytest.mark.cpu_sim
def test_no_profile_keeps_the_legacy_reset_info_and_evaluation(tmp_path):
    pytest.importorskip("warp")
    _, result, _, info = _episode(tmp_path/"legacy", steps=2)
    assert "runtime_profile" not in info and "acceptance_valid" not in info
    assert "runtime_profile" not in result and "acceptance_valid" not in result


def test_profile_refuses_caller_numerics_before_anything_runs(tmp_path):
    from robotic_cutting_sim.live_env import open_cutting_env
    with pytest.raises(ValueError, match="fixes"):
        with open_cutting_env(runtime="kinematic", output_dir=tmp_path/"x", preset="cutting_research_v1",
                              runtime_profile="full_physics_v1", particle_spacing_m=.004, device="cpu", **TRIM):
            pass
    assert not (tmp_path/"x").exists()


# ----------------------------------------------------------------------------- Native stage reset (config + USD part)

def test_isaac_stage_reset_is_opt_in_and_validated():
    from robotic_cutting_sim.isaac_sim_runner import STAGE_RESET_MODES, IsaacSimCuttingConfig
    assert IsaacSimCuttingConfig().stage_reset == "reopen"
    assert STAGE_RESET_MODES == ("reopen", "reload_in_place")
    assert IsaacSimCuttingConfig(stage_reset="reload_in_place",
                                 tool_attachment_mode="physical_grasp").stage_reset == "reload_in_place"
    with pytest.raises(ValueError, match="stage_reset"):
        IsaacSimCuttingConfig(stage_reset="recycle")
    for bad in (dict(camera=False, tool_attachment_mode="physical_grasp"), dict()):   # validated with cameras + grasp only
        with pytest.raises(ValueError, match="requires camera=True"):
            IsaacSimCuttingConfig(stage_reset="reload_in_place", **bad)


def test_renew_stage_layers_equals_a_fresh_open(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Gf, Sdf, Usd, UsdGeom
    from robotic_cutting_sim.isaac_sim_runner import renew_stage_layers
    sub = tmp_path/"asset.usda"
    asset = Usd.Stage.CreateNew(str(sub))
    UsdGeom.Cube.Define(asset, "/Asset/Cube").CreateSizeAttr(2.)
    asset.SetDefaultPrim(asset.GetPrimAtPath("/Asset"))
    asset.GetRootLayer().Save()
    path = tmp_path/"workcell.usda"
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, "/World")
    stage.DefinePrim("/World/Asset").GetReferences().AddReference(str(sub))
    UsdGeom.Mesh.Define(stage, "/World/ContinuousSpecimen")
    stage.GetRootLayer().Save()
    fresh = Usd.Stage.Open(str(path)).ExportToString()
    stage = Usd.Stage.Open(str(path))
    stage.SetEditTarget(stage.GetSessionLayer())          # the runner's reset edits
    UsdGeom.Cube.Define(stage, "/World/LiveKnife").CreateSizeAttr(.1)
    stage.GetPrimAtPath("/World/ContinuousSpecimen").SetActive(False)
    stage.GetPrimAtPath("/World/Asset/Cube").GetAttribute("size").Set(5.)
    with Usd.EditContext(stage, stage.GetRootLayer()):     # an edit that reached the root layer
        UsdGeom.Xform.Define(stage, "/World/Stray").AddTranslateOp().Set(Gf.Vec3d(1, 2, 3))
    assert stage.ExportToString() != fresh
    receipt = renew_stage_layers(stage)
    assert receipt["session_layer_empty"] and receipt["dirty_layers_before"] == [stage.GetRootLayer().identifier]
    assert stage.ExportToString() == fresh
    assert not any(layer.dirty for layer in stage.GetUsedLayers() if layer != stage.GetSessionLayer())


def test_isaac_surface_update_is_opt_in_and_validated():
    from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
    assert IsaacSimCuttingConfig().surface_update == "every_step"
    assert IsaacSimCuttingConfig(surface_update="on_capture").surface_update == "on_capture"
    with pytest.raises(ValueError, match="surface_update"):
        IsaacSimCuttingConfig(surface_update="never")


def test_specimen_surface_mesh_has_no_physics_api():
    """surface_update='on_capture' is physics-neutral only because the bridged mesh is visual."""
    text = (ROOT/"src/robotic_cutting_sim/usd_surface_bridge.py").read_text()
    for api in ("CollisionAPI", "RigidBodyAPI", "MeshCollisionAPI", "PhysxSchema"):
        assert api not in text
