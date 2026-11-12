"""P6.2: the ``open_vector_env`` dispatch branch for the rubber backend (plan §4.2, ADR 0023 §23.2 decision 7).

``backend="kinematic_grip_batched"`` with a ``RubberStripTaskFamily`` opens ``RubberVectorEnv``; a rubber task with
another backend raises ``UNSUPPORTED_BACKEND``; the rubber backend with a cutting task raises ``UNSUPPORTED_TASK``; the
cutting defaults (action spec, reward, material) map to the rubber defaults and every other non-rubber value is
refused; ``tasks=`` (cutting per-env selection) is refused. The default-backend path is unchanged (its own suites).
Needs Torch (Torch venv); skipped otherwise (NOT_RUN).
"""
import pytest

torch = pytest.importorskip("torch")

from robotic_cutting_sim.rl import vector_env as VE  # noqa: E402
from robotic_cutting_sim.rl.rubber_env import RubberVectorEnv  # noqa: E402
from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily, UnsupportedRubberConfiguration  # noqa: E402

PAR = "cpu_parity_v1"


def _fam(n=1):
    return RubberStripTaskFamily(["P16_t4"]*n, variant=PAR)


def test_dispatch_opens_the_rubber_env_and_maps_cutting_defaults():
    with VE.open_vector_env(1, task=_fam(), backend="kinematic_grip_batched", device="cpu", parity_mode=True,
                            strict=False) as env:
        assert isinstance(env, RubberVectorEnv)
        r = env.receipt()
        assert r["backend"] == "kinematic_grip_batched" and r["action_spec"]["id"] == "grip_delta_v1"
        assert r["reward"]["id"] == "rubber_tip_tracking_v1"
        obs, info = env.reset(seeds=[3])
        assert tuple(obs["policy"].shape) == (1, 78)
    assert env._closed


@pytest.mark.parametrize("backend", ["kinematic_ideal_tracking_batched", "isaac_lab_newton"])
def test_rubber_task_with_another_backend_is_refused(backend):
    with pytest.raises(UnsupportedRubberConfiguration) as err:
        with VE.open_vector_env(1, task=_fam(), backend=backend, device="cpu", parity_mode=True):
            pass
    assert err.value.code == "UNSUPPORTED_BACKEND"


def test_rubber_backend_with_a_cutting_task_is_refused():
    for task in (None, VE.TaskFamily(scene_seeds=[512])):
        with pytest.raises(UnsupportedRubberConfiguration) as err:
            with VE.open_vector_env(1, task=task, backend="kinematic_grip_batched", device="cpu", parity_mode=True):
                pass
        assert err.value.code == "UNSUPPORTED_TASK"


@pytest.mark.parametrize("kw,code", [({"action_spec": "edge_delta_v2"}, "UNSUPPORTED_OPTION"),
                                     ({"reward": "trim_bond_progress_v0"}, "UNSUPPORTED_OPTION"),
                                     ({"material": "law8"}, "UNSUPPORTED_OPTION"),
                                     ({"tasks": [512]}, "UNSUPPORTED_OPTION"),
                                     ({"autoreset": "same_step"}, "UNSUPPORTED_OPTION"),
                                     ({"cameras": ["wrist"]}, "UNSUPPORTED_SENSOR"),
                                     ({"observation_profile": "vision_rgbd"}, "UNSUPPORTED_SENSOR"),
                                     ({"recorders": ["cut_events"]}, "UNSUPPORTED_RECORDER"),
                                     ({"runtime": "rl_cpu"}, "UNSUPPORTED_RUNTIME")])
def test_refusals(kw, code):
    with pytest.raises(UnsupportedRubberConfiguration) as err:
        with VE.open_vector_env(1, task=_fam(), backend="kinematic_grip_batched", device="cpu", parity_mode=True,
                                **kw):
            pass
    assert err.value.code == code


def test_cpu_needs_parity_mode():
    with pytest.raises(UnsupportedRubberConfiguration) as err:
        with VE.open_vector_env(1, task=_fam(), backend="kinematic_grip_batched", device="cpu"):
            pass
    assert err.value.code == "UNSUPPORTED_CPU_WITHOUT_PARITY_MODE"


def test_default_backend_path_is_unchanged_for_cutting_tasks():
    """The dispatch branch does not see a cutting call: a TaskFamily with the default backend still reaches
    VectorCuttingEnv's own validation (an unsupported option is refused with the Phase 1 error type)."""
    with pytest.raises(VE.UnsupportedVectorConfiguration) as err:
        with VE.open_vector_env(1, task=VE.TaskFamily(scene_seeds=[512]), device="cpu", parity_mode=True,
                                material="law8"):
            pass
    assert err.value.code == "UNSUPPORTED_OPTION"
