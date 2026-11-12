"""P6.7 review fixes F2-F5 (small additive API changes; physics and the policy observation unchanged).

F2: ``env.metadata["privileged_info_keys"]`` labels the oracle-derived step-info keys as the Phase 1 contract does.
F3: a strict-mode invalid rubber env is a task outcome for ``rl.termination`` (``InvalidRubberEnvs`` is an
``InvalidBatchEnvs``). F4: Phase 1 options on the rubber backend raise ``UNSUPPORTED_OPTION``, not ``TypeError``.
F5: ``rebuild()`` to new slots behaves like a fresh env; ``reset(seeds=None)`` repeats the episode's draws (documented
behaviour). Needs Torch (Torch venv) and is skipped otherwise (NOT_RUN, not a pass); the Torch-free F3 / F2 checks
are in tests/test_rubber_rl_f3_classification.py."""
import numpy as np
import pytest

from robotic_cutting_sim.rl import termination as TM


torch = pytest.importorskip("torch")

from robotic_cutting_sim.rl import vector_env as VE  # noqa: E402
from robotic_cutting_sim.rl.rubber_contract import (PRIVILEGED_STEP_INFO, check_rubber_vector_env_contract,  # noqa
                                                    contract_actions)
from robotic_cutting_sim.rl.rubber_env import PRIVILEGED_INFO_KEYS, RubberVectorEnv  # noqa: E402
from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily, UnsupportedRubberConfiguration  # noqa: E402
from robotic_cutting_sim.rl.vector_contract import PRIVILEGED_INFO_KEYS as PHASE1_KEYS  # noqa: E402

PAR = "cpu_parity_v1"


def _env(slots, strict=False):
    return RubberVectorEnv(len(slots), task=RubberStripTaskFamily(slots, variant=PAR), device="cpu", parity_mode=True,
                           strict=strict)


def test_f2_privileged_info_keys_labelled():
    env = _env(["P16_t4", "P12_t6"])
    assert env.metadata["privileged_info_keys"] == PRIVILEGED_INFO_KEYS == PRIVILEGED_STEP_INFO
    assert env.env_metadata["privileged_info_keys"] == PRIVILEGED_INFO_KEYS
    env.reset(seeds=[1, 2])
    _, _, _, _, info = env.step(contract_actions(2, 0, "cpu"))
    assert set(info) & set(PHASE1_KEYS) <= set(env.metadata["privileged_info_keys"])
    assert {"invalid_reason", "termination_reason", "invalid_penalty_applied"} <= set(info)
    # the policy observation and its metadata are unchanged by the label
    assert env.observation_metadata()["info_privileged"] == ["invalid_reason"]
    report = check_rubber_vector_env_contract(_env(["P16_t4"]))
    assert "receipt/spaces/metadata" in report["clauses"]


def test_f3_strict_env_raises_a_task_classified_error():
    env = _env(["P16_t4", "P12_t6"], strict=True)
    env.reset(seeds=[1, 2])
    pre = env.family.preload_steps
    for k in range(pre):
        env.step(contract_actions(2, k, "cpu"))
    inv = env.material.invalid_reason.numpy().copy()
    inv[1] |= 2                                                         # RUBBER_J_DOMAIN
    env.material.invalid_reason.assign(inv)
    with pytest.raises(FloatingPointError) as err:
        env.step(contract_actions(2, pre, "cpu"))
    assert TM.classify_exception(err.value) == ("task", "MATERIAL_INVALID")


@pytest.mark.parametrize("option", [{"evaluation_stream": True}, {"terminal_capture": True}, {"input_guard": "v1"},
                                    {"tool_mount": "wrist"}, {"grip_contact_profile": "mjwarp_grip_v1"},
                                    {"no_such_option": 1}])
def test_f4_phase1_options_raise_unsupported_option(option):
    with pytest.raises(UnsupportedRubberConfiguration) as err:
        with VE.open_vector_env(1, task=RubberStripTaskFamily(["P16_t4"], variant=PAR),
                                backend="kinematic_grip_batched", device="cpu", parity_mode=True, **option):
            pass
    assert err.value.code == "UNSUPPORTED_OPTION" and next(iter(option)) in str(err.value)


def _particles(env):
    return np.array(env.material.solver.x.numpy(), copy=True)


def test_f5_rebuild_to_new_slots_equals_a_fresh_env():
    env = _env(["P16_t4", "P20_t5"])
    env.reset(seeds=[3, 4])
    env.step(contract_actions(2, 0, "cpu"))
    new = RubberStripTaskFamily(["P12_t6", "P16_t4"], variant=PAR)
    env.rebuild(new)
    fresh = _env(["P12_t6", "P16_t4"])
    a, _ = env.reset(seeds=[5, 6])
    b, _ = fresh.reset(seeds=[5, 6])
    assert list(env.family.env_slots) == ["P12_t6", "P16_t4"]
    assert a["policy"].numpy().tobytes() == b["policy"].numpy().tobytes()
    assert _particles(env).tobytes() == _particles(fresh).tobytes()
    for k in range(2):
        a = env.step(contract_actions(2, k, "cpu"))[0]
        b = fresh.step(contract_actions(2, k, "cpu"))[0]
        assert a["policy"].numpy().tobytes() == b["policy"].numpy().tobytes()
    check_rubber_vector_env_contract(env)


def test_f5_reset_without_seeds_repeats_the_draws():
    env = _env(["P16_t4"])
    env.reset(seeds=[11])
    goal, params = env.goal.clone(), env.params5.clone()
    env.step(contract_actions(1, 0, "cpu"))
    env.reset()                                                          # seeds=None: the env's seed is reused
    assert torch.equal(env.goal, goal) and torch.equal(env.params5, params)
    env.reset(seeds=[12])
    assert not torch.equal(env.goal, goal)
