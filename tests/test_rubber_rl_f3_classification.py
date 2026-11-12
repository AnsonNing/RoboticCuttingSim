"""P6.7 review F3 / F2, Torch-free (runs in .venv and CI): a strict-mode invalid rubber env is a task outcome for
``rl.termination`` (``InvalidRubberEnvs`` is a Phase 1 ``InvalidBatchEnvs``); the rubber contract's copy of the Phase 1
privileged info-key list equals ``rl.vector_contract.PRIVILEGED_INFO_KEYS``."""
import traceback

import pytest

from robotic_cutting_sim.rl import termination as TM


def test_f3_strict_invalid_rubber_env_is_a_task_outcome():
    rmb = pytest.importorskip("robotic_cutting_sim.batched.rubber_material_batch")
    from robotic_cutting_sim.batched.material_batch import InvalidBatchEnvs
    err = rmb.InvalidRubberEnvs([1], {1: ["RUBBER_J_DOMAIN"]})
    assert isinstance(err, InvalidBatchEnvs) and isinstance(err, FloatingPointError)
    assert err.env_ids == [1] and err.reasons == {1: ["RUBBER_J_DOMAIN"]}
    assert TM.classify_exception(err) == ("task", "MATERIAL_INVALID")
    try:
        raise err
    except FloatingPointError:
        tail = traceback.format_exc()
    out = TM.classify_process_exit(1, tail)
    assert (out["category"], out["kind"]) == ("task", "STRICT_INVALID_ENV")


def test_phase1_privileged_list_copy_is_current():
    from robotic_cutting_sim.rl.rubber_contract import PHASE1_PRIVILEGED_INFO_KEYS
    from robotic_cutting_sim.rl.vector_contract import PRIVILEGED_INFO_KEYS
    assert PHASE1_PRIVILEGED_INFO_KEYS == PRIVILEGED_INFO_KEYS
