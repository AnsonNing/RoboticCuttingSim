"""Deployable state observation and opt-in privileged critic block (ADR 0021 §8).

Guard tests are torch-free (run in .venv); tensor tests need torch (.deps/isaacsim-venv, plan D-09) and are
NOT_RUN where torch is missing."""
import numpy as np
import pytest

from robotic_cutting_sim.policy_env import PRIVILEGED_KEYS
from robotic_cutting_sim.rl import observations as O


def test_policy_terms_are_deployable_and_layout_is_fixed():
    assert O.assert_policy_terms_deployable(O.POLICY_TERMS)
    obs = O.StateObservation(4, max_steps=1200, control_dt=.02)
    meta = obs.metadata()
    assert meta["policy"]["dim"] == 18 and "critic" not in meta
    assert [(t["name"], t["slice"]) for t in meta["policy"]["terms"]] == [
        ("commanded_edge_pose_world", [0, 7]), ("time_remaining_s", [7, 8]), ("previous_action", [8, 16]),
        ("arm_flag", [16, 17]), ("step_fraction", [17, 18])]


@pytest.mark.parametrize("name", sorted(PRIVILEGED_KEYS)+[
    "cut_codes_packed", "contact_force_world", "reaction_wrench", "particle_positions", "material_state_x",
    "bond_progress", "protrusion_count", "tool_truth_pose", "oracle_components", "slot_overflow", "pad_contact"])
def test_guard_rejects_privileged_names(name):
    with pytest.raises(O.PrivilegedObservationLeak):
        O.assert_policy_terms_deployable(O.POLICY_TERMS+((name, 1, "", ""),))


def test_guard_rejects_privileged_term_injected_into_the_policy_block(monkeypatch):
    monkeypatch.setattr(O, "POLICY_TERMS", O.POLICY_TERMS+(("material_wrench_world", 6, "", ""),))
    with pytest.raises(O.PrivilegedObservationLeak):
        O.StateObservation(2, max_steps=10, control_dt=.02)


def test_profile_validation_and_structure_guard():
    with pytest.raises(ValueError):
        O.StateObservation(2, max_steps=10, control_dt=.02, profile="deployable")
    with pytest.raises(O.PrivilegedObservationLeak):
        O.assert_observation_structure({"policy": 0, "critic": 0}, "state")
    with pytest.raises(O.PrivilegedObservationLeak):
        O.assert_observation_structure({"policy": {"codes": 0}}, "privileged_state")
    with pytest.raises(O.PrivilegedObservationLeak):
        O.assert_observation_structure({"policy": 0, "material": 0}, "privileged_state")
    meta = O.StateObservation(2, max_steps=10, control_dt=.02, profile="privileged_state").metadata()
    assert meta["critic"]["dim"] == 18+9 and meta["critic"]["deployable"] is False
    assert meta["critic"]["terms"][1]["slice"] == [18, 19]


def _inputs(torch, N=3):
    pose = torch.tensor([[.1, -.02, .04, 0., 0., 0., 1.], [0., 0., .05, 0., 0., -.6, -.8],
                         [.02, .03, .06, .6, 0., 0., .8]], dtype=torch.float64)[:N]
    step = torch.tensor([0, 600, 1199])[:N]
    prev = torch.tensor([[0.]*8, [.5, -.5, 1., 0., 0., .25, 1., -1.], [0, 0, 0, 0, 0, 0, -1., 0]])[:N]
    armed = torch.tensor([False, True, False])[:N]
    return pose, step, prev, armed


def test_state_values_dtype_and_no_critic():
    torch = pytest.importorskip("torch")
    pose, step, prev, armed = _inputs(torch)
    builder = O.StateObservation(3, max_steps=1200, control_dt=.02)
    obs = builder.compute(commanded_pose=pose, step_count=step, previous_action=prev, armed=armed)
    assert set(obs) == {"policy"}
    p = obs["policy"]
    assert p.shape == (3, 18) and p.dtype == torch.float32 and p.device == pose.device
    np.testing.assert_allclose(p[:, 0:3].numpy(), pose[:, :3].numpy(), rtol=1e-7)
    np.testing.assert_allclose(p[1, 3:7].numpy(), [0., 0., .6, .8], rtol=1e-7)   # canonical w >= 0
    np.testing.assert_allclose(p[:, 7].numpy(), [24., 12., .02], rtol=1e-6)
    np.testing.assert_array_equal(p[:, 8:16].numpy(), prev.numpy())
    np.testing.assert_array_equal(p[:, 16].numpy(), [0., 1., 0.])
    np.testing.assert_allclose(p[:, 17].numpy(), [0., .5, 1199/1200], rtol=1e-7)
    with pytest.raises(O.PrivilegedObservationLeak):
        builder.compute(commanded_pose=pose, step_count=step, previous_action=prev, armed=armed,
                        privileged={"material_wrench_world": torch.zeros(3, 6)})


def test_privileged_profile_appends_critic_and_keeps_policy_identical():
    torch = pytest.importorskip("torch")
    pose, step, prev, armed = _inputs(torch)
    state = O.StateObservation(3, max_steps=1200, control_dt=.02).compute(
        commanded_pose=pose, step_count=step, previous_action=prev, armed=armed)
    builder = O.StateObservation(3, max_steps=1200, control_dt=.02, profile="privileged_state")
    priv = {"target_bond_progress_fraction": torch.tensor([0., .5, 1.]),
            "off_target_bond_fraction": torch.tensor([0., .01, .2]),
            "material_wrench_world": torch.arange(18.).reshape(3, 6), "invalid_flag": torch.tensor([0., 0., 1.])}
    obs = builder.compute(commanded_pose=pose, step_count=step, previous_action=prev, armed=armed, privileged=priv)
    assert set(obs) == {"policy", "critic"}
    assert torch.equal(obs["policy"], state["policy"])
    assert obs["critic"].shape == (3, 27) and torch.equal(obs["critic"][:, :18], obs["policy"])
    np.testing.assert_array_equal(obs["critic"][:, 20:26].numpy(), priv["material_wrench_world"].numpy())
    with pytest.raises(ValueError):
        builder.compute(commanded_pose=pose, step_count=step, previous_action=prev, armed=armed)
    with pytest.raises(ValueError):
        builder.compute(commanded_pose=pose, step_count=step, previous_action=prev, armed=armed,
                        privileged={**priv, "material_wrench_world": torch.zeros(3, 5)})
