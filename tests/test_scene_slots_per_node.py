"""Opt-in scene solver ``slots_per_node`` (tlmpm_v5 slot capacity) reaches the kinematic and live material (CPU)."""
from copy import deepcopy

import pytest

from robotic_cutting_sim.scene_config import (SceneConfigError, compile_task, load_json, make_configured_kinematic_env,
                                              validate_scene)

SCENE = load_json("configs/wp11/scenes/bar_three_bumps.json")


def with_slots(slots, module="tlmpm_v5"):
    doc = deepcopy(SCENE)
    doc["solver"]["module"] = module
    if module == "tlmpm_v4":
        doc["solver"]["max_cuts"] = 2
    doc["solver"]["slots_per_node"] = slots
    return doc


def test_slots_per_node_is_opt_in_and_v5_only():
    assert "slots_per_node" not in validate_scene(SCENE)["solver"]
    assert validate_scene(with_slots(16))["solver"]["slots_per_node"] == 16
    with pytest.raises(SceneConfigError, match="tlmpm_v5 only"):
        validate_scene(with_slots(16, "tlmpm_v4"))
    with pytest.raises(SceneConfigError, match="slots_per_node"):
        validate_scene(with_slots(0))
    goal = load_json("configs/wp11/goals/remove_all_protrusions.json")
    plain = compile_task(SCENE, goal, particle_spacing_m=.004)
    sixteen = compile_task(with_slots(16), goal, particle_spacing_m=.004)
    assert plain.task_id != sixteen.task_id and sixteen.scene["solver"]["slots_per_node"] == 16


def test_scene_slots_reach_the_kinematic_config_and_the_live_material():
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.live_material import LiveCuttingMaterial
    goal = load_json("configs/wp11/goals/remove_all_protrusions.json")
    task = compile_task(with_slots(16), goal, particle_spacing_m=.004)
    env = make_configured_kinematic_env(task, device="cpu")
    assert env.backend.config.slots_per_node == 16 and env.backend.config.solver == "tlmpm_v5"
    with pytest.raises(ValueError, match="differs"):
        make_configured_kinematic_env(task, device="cpu", slots_per_node=8)
    config = CouplingConfig(control_dt=task.control_dt_s, material_dt=1e-4, dx=task.grid_dx_m, device="cpu")
    # The Native runner builds LiveCuttingMaterial(task=..., solver='tlmpm_v5') without slots: the scene decides.
    material = LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law, solver="tlmpm_v5")
    assert material.slots_per_node == 16
    with pytest.raises(ValueError, match="differs"):
        LiveCuttingMaterial(config, task=task, max_cuts=task.max_cuts, cutting_law=task.cutting_law, solver="tlmpm_v5",
                            slots_per_node=8)
    plain = compile_task(SCENE, goal, particle_spacing_m=.004)
    assert LiveCuttingMaterial(config, task=plain, max_cuts=plain.max_cuts, cutting_law=plain.cutting_law,
                               solver="tlmpm_v5").slots_per_node == 8
