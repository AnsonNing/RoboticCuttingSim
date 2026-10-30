"""WP-11 review R1: versioned surface_unmeshable plumbing and the reload_in_place failure path (no Isaac launch)."""
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from robotic_cutting_sim.isaac_sim_runner import (IsaacSimCuttingConfig, LiveIsaacSimCutting, SurfaceNotScorable,
                                                  UnmeshableSurfaceLog)
from robotic_cutting_sim.scene_config import compile_task, configured_task_payload


@pytest.fixture(scope="module")
def configured_common():
    task = compile_task("trim01_wp11_workcell", "trim_bump_rho10")
    return dict(task_id=task.task_id, configured_task=configured_task_payload(task),
                tool_attachment_mode="physical_grasp", max_cuts=1, camera_mode="native_single_view",
                camera_layout="wrist_d405_global_d455")


def test_unmeshable_policy_resolves_to_raise_except_for_configured_tasks(configured_common):
    assert IsaacSimCuttingConfig().surface_unmeshable == "raise"                       # legacy D0
    trim = IsaacSimCuttingConfig(task_id="TRIM-01-R", tool_attachment_mode="physical_grasp")
    assert trim.surface_unmeshable == "raise"                                           # registered v1/v2 tasks
    configured = IsaacSimCuttingConfig(**configured_common)
    assert configured.surface_unmeshable == "skip_presentation_v1"                      # WP-11 configured path
    assert asdict(configured)["surface_unmeshable"] == "skip_presentation_v1"           # recorded resolved value
    assert IsaacSimCuttingConfig(**configured_common, surface_unmeshable="raise").surface_unmeshable == "raise"
    explicit = IsaacSimCuttingConfig(task_id="TRIM-01-R", tool_attachment_mode="physical_grasp",
                                     surface_unmeshable="skip_presentation_v1")
    assert explicit.surface_unmeshable == "skip_presentation_v1"
    with pytest.raises(ValueError, match="surface_unmeshable must be one of"):
        IsaacSimCuttingConfig(surface_unmeshable="skip")


def _info(skipped=None, components=3):
    info = {"mechanical_components": components, "component_ranges": []}
    if skipped is not None:
        info["unmeshable_skipped"] = skipped
    return info


def _skip(substrate, particles=11):
    return {"policy": "skip_presentation_v1", "fragments": 1, "particles": particles,
            "substrate_component_skipped": substrate,
            "components": [{"mechanical_component": 2, "particles": particles, "substrate_anchor_particles": int(bool(substrate))}],
            "rule": "fixture"}


def test_skip_log_records_each_skip_with_its_step_and_summarizes(tmp_path):
    log = UnmeshableSurfaceLog("skip_presentation_v1", tmp_path/"surface_unmeshable_00.jsonl")
    assert log.record(_info(), reset_index=0, step_index=0, time_s=0.) is None
    assert not (tmp_path/"surface_unmeshable_00.jsonl").exists()      # nothing written until something is skipped
    assert log.step_receipt() == {"policy": "skip_presentation_v1", "extractions": 1, "extractions_with_skips": 0,
                                  "new_skips": []}
    log.record(_info(_skip(False)), reset_index=0, step_index=7, time_s=.14)
    log.record(_info(_skip(False, particles=3)), reset_index=0, step_index=8, time_s=.16)
    receipt = log.step_receipt()
    assert receipt["extractions"] == 3 and receipt["extractions_with_skips"] == 2
    assert [row["step_index"] for row in receipt["new_skips"]] == [7, 8]
    assert log.step_receipt()["new_skips"] == []                      # each skip reaches exactly one step record
    rows = [json.loads(line) for line in (tmp_path/"surface_unmeshable_00.jsonl").read_text().splitlines()]
    assert [(r["step_index"], r["particles"], r["substrate_component_skipped"]) for r in rows] == [(7, 11, False),
                                                                                                    (8, 3, False)]
    summary = log.summary(_info())
    assert summary["extractions"] == 3 and summary["extractions_with_skips"] == 2
    assert summary["max_particles"] == 11 and summary["substrate_skipped_extractions"] == 0
    assert (summary["first_skip_step_index"], summary["last_skip_step_index"]) == (7, 8)
    assert summary["current_surface_skipped"] is None
    assert log.summary(_info(_skip(True)))["current_surface_skipped"] == {
        "fragments": 1, "particles": 11, "substrate_component_skipped": True}


def _runner(config, info, log=None, time_s=1.):
    runner = LiveIsaacSimCutting.__new__(LiveIsaacSimCutting)
    runner.config, runner.surface_info, runner.unmeshable_log = config, info, log
    runner.time_s = runner._surface_time = time_s     # surface already current: no re-extraction
    return runner


@pytest.mark.parametrize("substrate", [True, None])
def test_a_surface_omitting_substrate_is_not_scorable(substrate):
    config = IsaacSimCuttingConfig(task_id="TRIM-01-R", tool_attachment_mode="physical_grasp",
                                   surface_unmeshable="skip_presentation_v1")
    runner = _runner(config, _info(_skip(substrate)))
    with pytest.raises(SurfaceNotScorable, match="NOT_RUN"):
        runner.refresh_surface_for_evaluation()
    assert isinstance(SurfaceNotScorable("x"), RuntimeError)   # live_env's configured path records it as NOT_RUN


def test_non_substrate_skips_and_raise_mode_stay_scorable(tmp_path):
    config = IsaacSimCuttingConfig(task_id="TRIM-01-R", tool_attachment_mode="physical_grasp",
                                   surface_unmeshable="skip_presentation_v1")
    log = UnmeshableSurfaceLog("skip_presentation_v1", tmp_path/"log.jsonl")
    runner = _runner(config, _info(_skip(False)), log)
    runner.refresh_surface_for_evaluation()
    assert runner.surface_unmeshable_summary()["policy"] == "skip_presentation_v1"
    plain = _runner(IsaacSimCuttingConfig(), _info())
    plain.refresh_surface_for_evaluation()
    assert plain.surface_unmeshable_summary() is None          # raise mode: no key reaches evaluation results


class _Context:
    def __init__(self):
        self.opened = []

    def get_stage(self):
        return object()

    def open_stage(self, path):
        self.opened.append(path)
        return True


def test_failed_reload_in_place_forgets_the_stage_so_the_next_reset_reopens(tmp_path):
    scene = tmp_path/"workcell.usda"
    scene.write_text("#usda 1.0\n")
    config = IsaacSimCuttingConfig(scene=str(scene), stage_reset="reload_in_place", camera=True,
                                   tool_attachment_mode="physical_grasp")
    runner = LiveIsaacSimCutting.__new__(LiveIsaacSimCutting)
    runner.config, runner.output, runner.reset_index = config, tmp_path, 0
    context = _Context()
    first = runner._renew_stage(context, export=None)
    assert context.opened == [str(scene.resolve())] and runner.stage_reset_receipt["mode"] == "open_stage"
    assert runner._stage_source is None        # set only when reset() completes
    runner._stage_source = first               # what a completed reset() stores

    def failing_reload(context, source):
        raise RuntimeError("reload failed after detach")
    runner._reload_stage_in_place = failing_reload
    runner.reset_index = 1
    with pytest.raises(RuntimeError, match="reload failed"):
        runner._renew_stage(context, export=None)
    assert runner._stage_source is None        # no retry of an in-place reload on a detached stage
    runner.reset_index = 2
    runner._renew_stage(context, export=None)
    assert context.opened == [str(scene.resolve())]*2 and runner.stage_reset_receipt["mode"] == "open_stage"

    calls = []
    runner._stage_source = first
    runner._reload_stage_in_place = lambda context, source: calls.append(Path(source)) or {"mode": "reload_in_place"}
    assert runner._renew_stage(context, export=None) == first
    assert calls == [scene.resolve()] and runner.stage_reset_receipt == {"mode": "reload_in_place"}
