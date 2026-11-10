"""Review finding 12: the ``rcs_rl_rollout_v2`` field roles (CPU, no Torch). The writer / consumer-check round trip
runs in the Torch venv (``tests/test_rl_smoke.py``)."""
import pytest

from robotic_cutting_sim.rl import rollout as R


def test_every_written_field_has_a_role():
    assert set(R.FIELDS) | {"obs_critic", "next_obs_critic"} == set(R.FIELD_ROLES)


def test_oracle_fields_are_privileged_and_policy_inputs_are_only_the_policy_obs():
    privileged = {k for k, v in R.FIELD_ROLES.items() if v == R.PRIVILEGED}
    assert privileged == {"invalid_reason", "termination_reason", "obs_critic", "next_obs_critic"}
    assert {k for k, v in R.FIELD_ROLES.items() if v == R.DEPLOYABLE} == {"obs_policy", "next_obs_policy"}
    assert set(R.ORACLE_DERIVED) >= privileged | {"rewards", "terminated"}
    assert not set(R.ORACLE_DERIVED) & {"obs_policy", "next_obs_policy"}


def test_v1_and_v2_are_the_readable_schemas():
    assert R.SCHEMAS == ("rcs_rl_rollout_v1", "rcs_rl_rollout_v2") and R.SCHEMA == "rcs_rl_rollout_v1"


def test_load_rollout_refuses_unknown_schemas(tmp_path):
    (tmp_path / "manifest.json").write_text('{"schema": "rcs_rl_rollout_v9"}')
    with pytest.raises(ValueError, match="rcs_rl_rollout_v1 / rcs_rl_rollout_v2"):
        R.load_rollout(tmp_path)


def test_privileged_info_keys_include_the_oracle_done_flags():
    """Follow-up review finding 6: ``info["autoreset"]`` / ``info["terminal_captured"]`` are the done flags derived
    from the oracle ``termination_reason``; they are labelled with the other oracle-derived info keys."""
    from robotic_cutting_sim.rl.vector_contract import PRIVILEGED_INFO_KEYS, STEP_INFO
    assert {"autoreset", "terminal_captured", "invalid_reason", "termination_reason"} <= set(PRIVILEGED_INFO_KEYS)
    assert len(set(PRIVILEGED_INFO_KEYS)) == len(PRIVILEGED_INFO_KEYS)
    assert {"step_count", "seed", "episode", "drive", "action_status"}.isdisjoint(PRIVILEGED_INFO_KEYS)
    assert set(STEP_INFO) - set(PRIVILEGED_INFO_KEYS) == {"step_count", "action_status", "cut_digit", "action_clamped",
                                                          "action_out_of_box", "drive", "seed", "episode"}


def test_vector_env_info_keys_are_all_classified():
    """Every key ``VectorCuttingEnv.step`` writes into ``info`` is either a contract key or a labelled privileged key
    (source scan, CPU)."""
    import re
    from pathlib import Path
    from robotic_cutting_sim.rl import vector_env
    from robotic_cutting_sim.rl.vector_contract import PRIVILEGED_INFO_KEYS, STEP_INFO
    written = set(re.findall(r'info\["(\w+)"\] = ', Path(vector_env.__file__).read_text()))
    assert {"autoreset", "terminal_captured", "final_obs", "input_guard_reason"} <= written
    unlabelled = written - set(STEP_INFO) - set(PRIVILEGED_INFO_KEYS) - {"final_obs"}   # final_obs: the policy obs
    assert not unlabelled, unlabelled
