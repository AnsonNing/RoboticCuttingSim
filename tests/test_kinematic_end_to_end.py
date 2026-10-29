"""End-to-end control on the ideal-tracking kinematic backend (CPU, 4 mm).

Policies, the Gym adapter and the agent API emit bounded delta actions; the
backend executes them exactly; the ruler knife cuts the material. These are
pipeline checks: robot reachability, grasp and cameras are not involved.
"""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

pytest.importorskip("warp")
pytestmark = pytest.mark.cpu_sim

from robotic_cutting_sim.agent_api import CuttingAgentAPI
from robotic_cutting_sim.control import DELTA_MODE, ActionLimits
from robotic_cutting_sim.cut_topology import reference_components
from robotic_cutting_sim.kinematic_env import make_kinematic_env
from robotic_cutting_sim.policies import WaypointPolicy, skim, through_cut

COARSE = {"particle_spacing_m": .004, "material_dt": .0005}


def mass_fractions(env):
    material = env.backend.material
    _, component = reference_components(material.points, material.lattice, material.bed.code.numpy(),
                                        cuts=material.material_config.cuts)
    return np.sort(np.bincount(component, weights=material.masses))[::-1]/material.masses.sum(), component


def run_policy(env, policy, max_steps=400):
    observation, _ = env.reset()
    limits = env.action_limits
    while not policy.done and max_steps:
        action = policy(observation)
        if policy.done:
            break
        before = env.backend.edge.copy()
        observation, _, terminated, truncated, info = env.step(action)
        report = info["action_report"]
        # Commanded bounds hold and the backend executed exactly the bounded target.
        assert report["applied_translation_m"] <= limits.max_translation_m+1e-9
        assert report["applied_rotation_rad"] <= limits.max_rotation_rad+1e-9
        moved = np.linalg.norm(env.backend.edge[:3]-before[:3])
        assert moved == pytest.approx(report["applied_translation_m"], abs=1e-9)
        assert info["ik_status"] == "ok" and not terminated and not truncated
        max_steps -= 1
    assert policy.done
    return observation


@pytest.mark.parametrize("motion", ["push", "saw", "oblique20", "yaw30"])
def test_scripted_cuts_sever_the_bar(motion):
    env = make_kinematic_env("CUT-01-R", action_limits=ActionLimits(), **COARSE)
    options = {"push": {}, "saw": {"saw_amplitude_m": .01, "saw_period_s": .6},
               "oblique20": {"tilt_deg": 20.}, "yaw30": {"yaw_deg": 30.}}[motion]
    policy = WaypointPolicy(env.task, through_cut(env.task, cut_speed_m_s=.03, **options))
    run_policy(env, policy)
    fractions, _ = mass_fractions(env)
    assert fractions[1] > .35 and fractions[:2].sum() > .98


def test_skim_cuts_off_the_bump():
    env = make_kinematic_env("TRIM-01-R", action_limits=ActionLimits(), **COARSE)
    policy = WaypointPolicy(env.task, skim(env.task, speed_m_s=.04))
    run_policy(env, policy)
    material, task = env.backend.material, env.task
    _, component = mass_fractions(env)
    plane = task.target_plane_world()
    height = material.world_points@plane[:3]+plane[3]
    substrate = np.bincount(component[height < -task.particle_spacing_m]).argmax()
    above = height > task.particle_spacing_m
    detached = material.masses[above & (component != substrate)].sum()/material.masses[above].sum()
    assert detached > .9


def test_gym_adapter_drives_a_saw_cut():
    gym = pytest.importorskip("gymnasium")
    from robotic_cutting_sim.gym_env import CuttingGymEnv
    limits = ActionLimits()
    env = make_kinematic_env("CUT-01-R", action_mode=DELTA_MODE, action_limits=limits, delta_frame="world", **COARSE)
    adapter = CuttingGymEnv(env)
    policy = WaypointPolicy(env.task, through_cut(env.task, cut_speed_m_s=.03, saw_amplitude_m=.01, saw_period_s=.6))
    observation = env.observe()
    for _ in range(400):
        action = policy(observation)
        if policy.done:
            break
        normalized = np.r_[action["target"][:3]/limits.max_translation_m, action["target"][3:]/limits.max_rotation_rad]
        assert np.all(np.abs(normalized) <= 1+1e-9) and adapter.action_space.contains(normalized.astype(np.float32))
        state, reward, terminated, truncated, info = adapter.step(normalized)
        assert adapter.observation_space.contains(state)
        observation = env.observe()
    fractions, _ = mass_fractions(env)
    assert policy.done and fractions[1] > .35


def test_agent_api_session_severs_the_bar():
    env = make_kinematic_env("CUT-02", **COARSE)
    agent = CuttingAgentAPI(env)
    description = agent.describe_task()
    assert description["goal"].startswith("Cut the workpiece") and description["tool"] == "ruler_knife_v1"
    assert agent.reset()["runtime_source"] == "kinematic_ideal_tracking"
    top = description["workpiece_size_m"][2]
    rpy = Rotation.from_quat(env.task.reference_edge_pose_world(0.)[3:]).as_euler("xyz", degrees=True)
    moved = agent.move_edge_to([0., 0., top+.008], rpy)
    assert moved["reached"] and moved["ik_rejected_steps"] == 0
    cut = agent.cut_along("advance", top+.012, speed_m_s=.03, saw_amplitude_m=.008)
    assert cut["reached"] and cut["edge_position_m"][2] < 0
    retract = agent.move_edge_by([0, 0, .04])
    assert retract["reached"] and not retract["done"]
    fractions, _ = mass_fractions(env)
    assert fractions[1] > .35
    assert set(agent.status()) >= {"time_s", "edge_position_m", "termination_reason"}
