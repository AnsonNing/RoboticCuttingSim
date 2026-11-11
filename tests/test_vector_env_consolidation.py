"""Phase 1 consolidation on the Warp/Torch CPU device (workload A): ``evaluate()`` of a non-finite env returns a
structured ``FAIL_INTEGRITY`` entry (P1-4 finding: it raised ``ValueError`` from ``transform_points``), the material
``invalid_reason`` -> ``termination_reason`` mapping on the device, and ``tasks=`` entries for configured scenes.

Torch test: run in ``.deps/isaacsim-venv`` with the D-09 overlay; in ``.venv`` (no torch) it is skipped = NOT_RUN.
Coarse, shortened TRIM / CUT tasks (2.5 mm spacing, a few steps): every env here is ``reduced_fidelity``."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import warp as wp  # noqa: E402

from test_vector_env_api import env_of, push  # noqa: E402
from robotic_cutting_sim.batched.material_batch import INVALID_REASONS, InvalidBatchEnvs  # noqa: E402
from robotic_cutting_sim.rl import ConfiguredTaskFamily, UnsupportedVectorConfiguration, open_vector_env  # noqa: E402
from robotic_cutting_sim.rl import termination as T  # noqa: E402
from robotic_cutting_sim.rl import vector_env as VE  # noqa: E402
from robotic_cutting_sim.rl.configured_family import NATIVE_CUT_V7_2_REFERENCE  # noqa: E402
from robotic_cutting_sim.rl.evaluation_stream import INTEGRITY_FAILURE_VERSION  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def poison(env, e):
    """Fault injection (as P1-4 ``nan_env``): one NaN position in env ``e``'s material state."""
    s, pk = env.material.solver, env.material.packed
    wp.to_torch(s.x)[int(pk.particle_offset[e])] = float("nan")


def assert_integrity_failure(entry, e, cause_prefix):
    assert entry["env"] == e and entry["verdict"] == "FAIL_INTEGRITY"
    assert entry["summary"]["verdict"] == "FAIL_INTEGRITY" and entry["summary"]["integrity"] == "FAIL"
    assert entry["summary"]["outcome"] == "NOT_EVALUATED" and entry["result"]["success"] is False
    assert entry["integrity_failure"] == INTEGRITY_FAILURE_VERSION and entry["integrity_cause"].startswith(cause_prefix)
    assert entry["physics_metrics"] is None and entry["labelled"] is None and entry["final_codes_sha256"] is None
    assert entry["summary"]["removed_mass_fraction"] == {} and entry["summary"]["mechanical_components"] is None
    assert "not called" in entry["evaluator"]


# ----------------------------------------------------------------------------- task 2: evaluate() of a non-finite env

def test_evaluate_returns_fail_integrity_for_a_nonfinite_env_and_scores_the_others():
    with env_of(strict=False) as env, env_of(strict=False) as clean:
        env.reset()
        clean.reset()
        env.step(push(2))
        clean.step(push(2))
        poison(env, 1)
        _, _, terminated, _, info = env.step(push(2))
        clean.step(push(2))
        assert int(info["invalid_reason"][1]) & 1 and int(info["invalid_reason"][0]) == 0    # NONFINITE_STATE
        assert info["termination_reason"].tolist()[1] & T.TERM_MATERIAL_INVALID and terminated.tolist()[1]
        results = env.evaluate()
        expected = clean.evaluate([0])[0]
    assert [r["env"] for r in results] == [0, 1]
    good, bad = results
    assert good["final_codes_sha256"] == expected["final_codes_sha256"]     # env 0 evaluated normally
    assert good["summary"]["verdict"] == expected["summary"]["verdict"] and good["invalid_reasons"] == []
    assert good["termination_reasons"] == [] and good["result"] is not None
    assert_integrity_failure(bad, 1, "nonfinite_material_bit")
    assert "NONFINITE_STATE" in bad["invalid_reasons"] and bad["invalid_reason"] & 1
    assert "MATERIAL_INVALID" in bad["termination_reasons"] and bad["step_count"] == 2


def test_evaluate_of_a_nonfinite_snapshot_without_a_material_bit_and_strict_step_unchanged():
    with env_of(strict=True) as env:
        env.reset()
        env.step(push(2))
        poison(env, 0)                       # no step after the fault: no material bit yet, the snapshot is NaN
        with pytest.raises(ValueError, match="Expected finite"):
            env.snapshot_world(0)            # public snapshot keeps refusing (a ValueError, as before)
        first, second = env.evaluate()
        assert_integrity_failure(first, 0, "nonfinite_snapshot")
        assert first["invalid_reasons"] == [] and second["env"] == 1
        assert second["result"] is not None and "integrity_failure" not in second
        with pytest.raises(InvalidBatchEnvs):     # strict step(): the material's own raise, unchanged
            env.step(push(2))
        assert env.infrastructure_error is None
        again = env.evaluate([0])[0]
        assert_integrity_failure(again, 0, "nonfinite_material_bit")


def test_evaluate_last_completed_of_a_nonfinite_episode():
    with env_of(seeds=(512, 15), strict=False, autoreset="same_step", terminal_capture=True,
                evaluation_stream={"every": 1}) as env:
        env.reset()
        env.step(push(2))
        poison(env, 1)
        _, _, terminated, _, info = env.step(push(2))       # env 1 terminates and is autoreset in the same call
        assert terminated.tolist() == [False, True] and info["terminal_captured"].tolist() == [False, True]
        done = env.evaluate([1], episode="last_completed")[0]
        assert_integrity_failure(done, 1, "nonfinite_snapshot")
        assert done["invalid_reasons"] is None and done["step_count"] == 2       # bits cleared by the autoreset
        current = env.evaluate([0])[0]
        assert current["result"] is not None


# ----------------------------------------------------------------------------- task 3: device mapping

def test_material_bits_map_to_termination_bits_on_the_device():
    bits = torch.tensor([0, *INVALID_REASONS, 128 | 256, 1 | 128], dtype=torch.int32)
    out = T.material_termination_bits(bits)
    assert out.dtype == torch.int32 and out.device == bits.device
    assert out.tolist() == [T.material_termination_bits(int(v)) for v in bits.tolist()]
    assert out.tolist()[list(INVALID_REASONS).index(256) + 1] == T.TERM_MATERIAL_INVALID | T.TERM_INFRASTRUCTURE
    with env_of(strict=False) as env:
        env.reset()
        cut = torch.tensor([5, -1], dtype=torch.int32)     # digit 5 >= K: INVALID_CUT_DIGIT in env 0
        _, _, _, _, info = env.step_measured_edge(env.commanded_pose.clone(), cut)
        assert info["termination_reason"].tolist() == [T.TERM_MATERIAL_INVALID, 0]
        detail = T.decode_termination_detail(info["termination_reason"][0], info["invalid_reason"][0])
        assert detail == {"termination": ["MATERIAL_INVALID"], "material_invalid": ["INVALID_CUT_DIGIT"],
                          "infrastructure": False}
        table = env.receipt()["termination"]
        assert set(table["material_invalid"]["bits"]) == {str(b) for b in INVALID_REASONS}


# ----------------------------------------------------------------------------- task 4: tasks= with configured scenes

CUT_SPACING = .0025


def test_tasks_entries_for_configured_scenes():
    base = ConfiguredTaskFamily.from_cut_workload(2, particle_spacing_m=CUT_SPACING, max_episode_steps=2)
    resolved = VE.resolve_tasks(2, base, [{"family_id": "configured_scene_v1", "scene": "cut_v7_2_v1", "seed": 3},
                                          {"scene": "cut_v7_2_v1"}])
    assert isinstance(resolved, ConfiguredTaskFamily) and resolved.env_seeds == (3, base.sim_seed)
    assert resolved.particle_spacing_m == CUT_SPACING and resolved.max_episode_steps == 2
    assert resolved.native_reference["id"] == NATIVE_CUT_V7_2_REFERENCE["id"] and resolved.workload_id == "cut_v7_2_v1"
    with open_vector_env(2, task=base, tasks=[{"scene": "cut_v7_2_v1", "seed": 3}, {"scene": "cut_v7_2_v1"}],
                         device="cpu", parity_mode=True) as env:
        receipt = env.receipt()
        assert [t["initial_seed"] for t in receipt["tasks"]] == [3, base.sim_seed]
        assert receipt["task_family"]["kind"] == "ConfiguredTaskFamily"
        _, info = env.reset()
        assert info["seed"].tolist() == [3, base.sim_seed]
        env.step(push(2))
    # a TRIM seed and a CUT scene in one batch: the batched layout differs (K, episode length, reward) -> refused
    with pytest.raises(UnsupportedVectorConfiguration, match="^UNSUPPORTED_NONUNIFORM_BATCH") as error:
        VE.resolve_tasks(2, None, [512, {"scene": "cut_v7_2_v1"}])
    assert "max_cuts" in str(error.value) and "duration_s" in str(error.value)
