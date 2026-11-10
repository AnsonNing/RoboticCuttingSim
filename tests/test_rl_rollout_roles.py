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
