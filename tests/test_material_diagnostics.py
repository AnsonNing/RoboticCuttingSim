"""CPU execution of the optional read-only live safety reduction."""

from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from robotic_cutting_sim.isaac_sim_runner import IsaacSimCuttingConfig
from robotic_cutting_sim.live_material import LiveCuttingMaterial


def material(count=130):
    rng = np.random.default_rng(1601)
    values = {"positions": rng.normal(size=(count, 3)).astype(np.float32),
              "velocities": rng.normal(size=(count, 3)).astype(np.float32),
              "F": (np.eye(3) + .2*rng.normal(size=(count, 3, 3))).astype(np.float32),
              "plastic_strain": -rng.uniform(.1, 2., size=count).astype(np.float32)}
    bed = SimpleNamespace(x=wp.array(values["positions"], dtype=wp.vec3, device="cpu"),
                          v=wp.array(values["velocities"], dtype=wp.vec3, device="cpu"),
                          F=wp.array(values["F"], dtype=wp.mat33, device="cpu"),
                          plastic=wp.array(values["plastic_strain"], dtype=float, device="cpu"))
    bed.snapshot_calls = 0

    def snapshot():
        bed.snapshot_calls += 1
        return dict(positions=bed.x.numpy(), velocities=bed.v.numpy(), F=bed.F.numpy(),
                    plastic_strain=bed.plastic.numpy())

    bed.snapshot = snapshot
    live = LiveCuttingMaterial.__new__(LiveCuttingMaterial)
    live.bed = bed
    return live, values


@pytest.mark.parametrize("count", [1, 64, 65, 130])
def test_reduction_matches_snapshot_and_preserves_all_source_arrays(count):
    live, original = material(count)
    expected = live.numerical_diagnostics("snapshot")
    actual = live.numerical_diagnostics("device_reduction_v1")
    assert live.bed.snapshot_calls == 1
    assert actual.keys() == expected.keys()
    # Double determinant uses an explicit 3x3 formula rather than NumPy's LU;
    # both round back to float32. They may differ at the last output bit.
    np.testing.assert_allclose(list(actual.values()), list(expected.values()), rtol=2e-6, atol=0.)
    assert actual["maximum_plastic_strain"] < 0.
    for name, value in live.debug_state().items():
        np.testing.assert_array_equal(value, original[name])


@pytest.mark.parametrize("component", range(16))
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_every_state_component_remains_subject_to_nonfinite_rejection(component, bad):
    live, values = material(65)
    if component < 3:
        values["positions"][-1, component] = bad
        live.bed.x.assign(values["positions"])
    elif component < 6:
        values["velocities"][-1, component-3] = bad
        live.bed.v.assign(values["velocities"])
    elif component < 15:
        row, column = divmod(component-6, 3)
        values["F"][-1, row, column] = bad
        live.bed.F.assign(values["F"])
    else:
        values["plastic_strain"][-1] = bad
        live.bed.plastic.assign(values["plastic_strain"])
    for mode in ("snapshot", "device_reduction_v1"):
        with pytest.raises(FloatingPointError, match="Nonfinite live TLMPM state"):
            live.numerical_diagnostics(mode)


def test_reused_scratch_has_no_stale_extrema_or_nonfinite_flag_and_reset_rebinds():
    live, values = material(65)
    first = live.numerical_diagnostics("device_reduction_v1")
    reducer = live._numerical_reduction
    values["positions"][0, 0] = np.nan
    live.bed.x.assign(values["positions"])
    with pytest.raises(FloatingPointError):
        live.numerical_diagnostics("device_reduction_v1")
    live.bed.x.zero_()
    live.bed.v.zero_()
    live.bed.F.assign(np.tile(np.eye(3, dtype=np.float32), (65, 1, 1)))
    live.bed.plastic.assign(np.full(65, -3., dtype=np.float32))
    assert live.numerical_diagnostics("device_reduction_v1") == {
        "minimum_detF": 1., "maximum_material_speed_m_s": 0., "maximum_plastic_strain": -3.}
    assert live._numerical_reduction is reducer and first["maximum_material_speed_m_s"] > 0.
    replacement, _ = material(1)
    live.bed = replacement.bed
    np.testing.assert_allclose(list(live.numerical_diagnostics("device_reduction_v1").values()),
                               list(replacement.numerical_diagnostics().values()), rtol=2e-6)
    assert live._numerical_reduction is not reducer


@pytest.mark.parametrize("matrix", [
    [[1., 1., 1.], [1., 1.0001, 1.], [1., 1., 1.0001]],
    [[1e10, 1e10, 1.], [1e10, 1e10+1024, 1.], [1., 1., 1.]],
    [[1e-20, 0., 0.], [0., 1e-20, 0.], [0., 0., 1.]],
    [[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]],
    [[-1., 0., 0.], [0., 2., 0.], [0., 0., 3.]],
])
def test_determinant_cancellation_singular_subnormal_and_negative_values(matrix):
    live, _ = material(1)
    live.bed.F.assign(np.asarray([matrix], dtype=np.float32))
    actual = live.numerical_diagnostics("device_reduction_v1")["minimum_detF"]
    expected = live.numerical_diagnostics()["minimum_detF"]
    assert actual == pytest.approx(expected, rel=2e-6, abs=1e-44)


def test_finite_inputs_with_overflowed_derived_statistics_keep_original_behavior():
    live, _ = material(1)
    live.bed.v.assign(np.full((1, 3), 1e30, dtype=np.float32))
    live.bed.F.assign(np.asarray([np.eye(3)*1e30], dtype=np.float32))
    with np.errstate(over="ignore"):
        expected = live.numerical_diagnostics()
    assert live.numerical_diagnostics("device_reduction_v1") == expected
    assert np.isinf(expected["minimum_detF"]) and np.isinf(expected["maximum_material_speed_m_s"])


def test_config_keeps_explicit_snapshot_rollback_and_rejects_unknown_modes():
    assert IsaacSimCuttingConfig().material_diagnostics == "snapshot"
    assert IsaacSimCuttingConfig(material_diagnostics="device_reduction_v1").material_diagnostics == "device_reduction_v1"
    with pytest.raises(ValueError, match="material_diagnostics"):
        IsaacSimCuttingConfig(material_diagnostics="skip_checks")
    live, _ = material(1)
    with pytest.raises(ValueError, match="material_diagnostics"):
        live.numerical_diagnostics("skip_checks")


@pytest.mark.parametrize("selection", [None, "device_reduction_v1"])
def test_existing_agentic_adapter_forwards_selected_diagnostics_without_contract_changes(tmp_path, monkeypatch, selection):
    from contextlib import contextmanager
    from robotic_cutting_sim import agentic_trim_env, live_env

    calls = []

    @contextmanager
    def fake_native(**options):
        calls.append(options)
        yield object()

    finalized = []
    monkeypatch.setattr(live_env, "open_cutting_env", fake_native)
    monkeypatch.setattr(agentic_trim_env, "compile_seed", lambda *a, **kw: (object(), {"particle_count": 2}))
    monkeypatch.setattr(agentic_trim_env, "AgenticTrimEnv", lambda *a, **kw:
                        SimpleNamespace(finalize=lambda reason: finalized.append(reason)))
    options = {} if selection is None else {"material_diagnostics": selection, "v7_2_optimizations": False}
    with agentic_trim_env.open_env(output_dir=tmp_path/"native", seed=512, stream_physics=True, **options):
        pass
    assert calls[0]["material_diagnostics"] == (selection or "snapshot")
    assert calls[0]["v7_2_optimizations"] is (selection is None)
    assert calls[0]["history_mode"] == "latest"
    assert calls[0]["active_camera_ids"] == agentic_trim_env.CAMERAS
    assert calls[0]["material_dt"] == .0001 and calls[0]["dx"] == .004
    assert finalized == ["WORKER_EXIT"]


@pytest.mark.parametrize("bad", [0, 1, None, "false"])
def test_optimization_switch_rejects_nonbool_at_each_runtime_entry(tmp_path, bad):
    from robotic_cutting_sim.agentic_trim_env import open_env
    with pytest.raises(ValueError, match="v7_2_optimizations must be a bool"):
        IsaacSimCuttingConfig(v7_2_optimizations=bad)
    with pytest.raises(ValueError, match="v7_2_optimizations must be a bool"):
        LiveCuttingMaterial(v7_2_optimizations=bad)
    with pytest.raises(ValueError, match="v7_2_optimizations must be a bool"):
        with open_env(output_dir=tmp_path/"unused", seed=512, v7_2_optimizations=bad):
            pass
    assert not (tmp_path/"unused").exists()


@pytest.mark.parametrize("solver", ["tlmpm_v5", "tlmpm_v7", "tlmpm_v7_2"])
@pytest.mark.parametrize("enabled", [True, False])
def test_only_v7_2_explicit_rollback_overrides_solver_defaults(monkeypatch, solver, enabled):
    import importlib
    from robotic_cutting_sim.coupled_robot import CouplingConfig
    from robotic_cutting_sim.tool_profile import RULER_KNIFE_V1

    captured = {}

    def stop_before_allocation(*args, **kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop_before_solver_allocation")

    monkeypatch.setattr(importlib.import_module("robotic_cutting_sim."+solver), "TLMPM", stop_before_allocation)
    live = LiveCuttingMaterial(CouplingConfig(device="cpu"), solver=solver, tool_profile=RULER_KNIFE_V1,
                               cutting_law=8, v7_2_optimizations=enabled)
    with pytest.raises(RuntimeError, match="stop_before_solver_allocation"):
        live.reset([0., 0., .05, 0., 0., 0., 1.], seed=42)
    overrides = {key:captured[key] for key in ("optimize_contacts",) if key in captured}
    assert overrides == ({"optimize_contacts":False}
                         if solver == "tlmpm_v7_2" and not enabled else {})
    assert IsaacSimCuttingConfig().v7_2_optimizations is True
