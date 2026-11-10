"""Benchmark rows, Markdown table, nvidia-smi telemetry parsing and the RL benchmark workloads config."""
import json
from pathlib import Path

import pytest

from robotic_cutting_sim.rl.benchmark import (BOUNDARIES, MAIN_COLUMNS, BenchmarkRow, co_tenant_stats,
                                              markdown_table, percentile, rows_from_json, rows_to_json)
from robotic_cutting_sim.rl.telemetry import (GpuTelemetry, parse_compute_apps, parse_pids_query, parse_pmon,
                                              parse_query_gpu, summarize, take_sample)

ROOT = Path(__file__).resolve().parents[1]

QUERY_GPU = "1, GPU-bbb, NVIDIA GeForce RTX 5090, 570.211.01, 55, 5794, 32607\n"
COMPUTE_APPS = ("642659, GPU-aaa, 766\n"
                "4242, GPU-bbb, 2100\n"
                "777, GPU-bbb, 300\n")
PIDS = """
==============NVSMI LOG==============
GPU 00000000:F1:00.0
    Processes
        GPU instance ID                   : N/A
        Compute instance ID               : N/A
        Process ID                        : 4060
            Type                          : G
            Name                          : /usr/lib/xorg/Xorg
            Used GPU Memory               : 47 MiB
        Process ID                        : 615272
            Type                          : G
            Name                          : /some/env/bin/python
            Used GPU Memory               : 715 MiB
        Process ID                        : 4242
            Type                          : C
            Name                          : python
            Used GPU Memory               : 2100 MiB
        Process ID                        : 777
            Type                          : C
            Name                          : python
            Used GPU Memory               : 300 MiB
"""
PMON = """# gpu         pid   type     sm    mem    enc    dec    jpg    ofa    command
# Idx           #    C/G      %      %      %      %      %      %    name
    1       4060     G      -      -      -      -      -      -    Xorg
    1     615272     G     12      3      -      -      -      -    python
    1       4242     C     40     20      -      -      -      -    python
"""


def make_row(**overrides):
    values = dict(boundary="single_env_headless", workload_id="A/D-s512", num_envs=1, particles_per_env=102732,
                  control_hz=50.0, material_dt_s=1e-4, substeps_per_control_step=200, cameras="none", robot_dof=0,
                  contacts="tool+pads (replayed)", recording="none", warm_cold="warm", env_steps_per_s=7.8,
                  aggregate_sim_time_per_wall=0.156, vram_peak_mib=900.0, reset_mean_s=2.0, reset_p95_s=2.4,
                  physics_verdict="REPORTED_NOT_SCORED", reduced_fidelity=False, gpu_name="NVIDIA GeForce RTX 5090",
                  gpu_index=1, gpu_driver="570.211.01", gpu_sharing="EXCLUSIVE_GPU", co_tenant={},
                  source_commit="abc123", source="measured", notes="")
    values.update(overrides)
    return BenchmarkRow(**values)


def test_row_round_trips_through_json():
    rows = [make_row(), make_row(boundary="batched_headless", num_envs=16, env_steps_per_s=None,
                                 aggregate_sim_time_per_wall=None, physics_verdict="NOT_RUN", warm_cold="n/a",
                                 gpu_sharing="UNKNOWN")]
    assert rows_from_json(rows_to_json(rows)) == rows


@pytest.mark.parametrize("bad", [dict(boundary="vectorized"), dict(warm_cold="hot"), dict(gpu_sharing="maybe"),
                                 dict(physics_verdict="OK"), dict(reduced_fidelity="no"),
                                 dict(env_steps_per_s=float("nan")), dict(physics_verdict="NOT_RUN")])
def test_row_rejects_invalid_values(bad):
    with pytest.raises(ValueError):
        make_row(**bad)


def test_row_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown"):
        BenchmarkRow.from_dict({**make_row().to_dict(), "speedup": 3.0})


def test_markdown_table_has_required_columns_first():
    table = markdown_table([make_row(), make_row(boundary="single_env_native", cameras="global_d455+wrist_d405_0",
                                                 source="wp17_evidence")])
    lines = table.strip().splitlines()
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    assert tuple(header[:10]) == MAIN_COLUMNS
    assert header[10:12] == ["boundary", "workload"] and "sharing" in header and "source" in header
    assert len(lines) == 4 and all(line.count("|") == lines[0].count("|") for line in lines)
    assert "7.800" in lines[2] and "wp17_evidence" in lines[3] and "single_env_native" in lines[3]
    assert set(BOUNDARIES) == {"single_env_native", "single_env_headless", "batched_headless"}


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([5], 95) == 5
    assert percentile([], 95) is None


def test_parsers_read_canned_nvidia_smi_output():
    gpu = parse_query_gpu(QUERY_GPU)[0]
    assert gpu == dict(index=1, uuid="GPU-bbb", name="NVIDIA GeForce RTX 5090", driver="570.211.01", util_pct=55.,
                       memory_used_mib=5794., memory_total_mib=32607.)
    apps = parse_compute_apps(COMPUTE_APPS)
    assert [a["pid"] for a in apps] == [642659, 4242, 777] and apps[1]["memory_mib"] == 2100.
    pids = parse_pids_query(PIDS)
    assert [(p["pid"], p["type"], p["display"]) for p in pids] == [
        (4060, "G", True), (615272, "G", False), (4242, "C", False), (777, "C", False)]
    sm = parse_pmon(PMON)
    assert [(r["pid"], r["sm_pct"]) for r in sm] == [(4060, None), (615272, 12.), (4242, 40.)]


def canned_runner(gpu=QUERY_GPU, apps=COMPUTE_APPS, pids=PIDS, pmon=PMON):
    def run(args, timeout=10.):
        if any(a.startswith("--query-gpu") for a in args):
            return gpu
        if any(a.startswith("--query-compute-apps") for a in args):
            return apps
        if "PIDS" in args:
            return pids
        if "pmon" in args:
            return pmon
        return None
    return run


def test_sample_separates_own_foreign_and_display_processes():
    sample = take_sample(1, {777}, runner=canned_runner())
    assert sample["ok"] and sample["own_memory_mib"] == 300.
    assert sample["foreign_compute"] == {4242: 2100.}            # GPU-aaa process is on another GPU
    assert sample["foreign_graphics"] == {615272: 715.}          # Xorg is display, not a co-tenant
    assert sample["foreign_memory_mib"] == 2815. and sample["display_pids"] == [4060]
    assert sample["foreign_sm_pct"] == 52. and sample["own_sm_pct"] == 0.


def test_summary_labels_shared_and_exclusive():
    shared = summarize([dict(take_sample(1, {777}, runner=canned_runner()), phase="run")])
    assert shared["gpu_sharing"] == "SHARED_GPU" and shared["own_vram_peak_mib"] == 300.
    assert "foreign_compute_process" in shared["sharing_reasons"] and shared["foreign_compute_pids"] == [4242]
    alone_apps = "777, GPU-bbb, 300\n"
    alone_pids = PIDS.split("        Process ID                        : 615272")[0]
    alone_pmon = "# gpu pid\n    1    777     C     88      1      -      -      -      -    python\n"
    quiet = canned_runner(apps=alone_apps, pids=alone_pids, pmon=alone_pmon)
    samples = [dict(take_sample(1, {777}, runner=quiet), phase="baseline"),
               dict(take_sample(1, {777}, runner=quiet), phase="run")]
    samples[0]["gpu"] = dict(samples[0]["gpu"], util_pct=0.)
    exclusive = summarize(samples)
    assert exclusive["gpu_sharing"] == "EXCLUSIVE_GPU", exclusive["sharing_reasons"]
    assert exclusive["foreign_sm_pct"]["max"] == 0. and exclusive["own_sm_pct"]["max"] == 88.
    samples[0]["gpu"] = dict(samples[0]["gpu"], util_pct=30.)
    assert summarize(samples)["sharing_reasons"] == ["baseline_total_utilisation"]
    assert summarize([])["gpu_sharing"] == "UNKNOWN"
    stats = co_tenant_stats(shared)
    assert stats["foreign_processes_max"] == 2 and stats["foreign_sm_pct_max"] == 52.


def test_background_sampler_collects_and_stops():
    telemetry = GpuTelemetry(1, interval_s=0.01, own_pid=777, runner=canned_runner())
    telemetry.set_phase("baseline")
    telemetry.sample()
    telemetry.set_phase("run")
    with telemetry:
        import time
        time.sleep(0.05)
    summary = telemetry.summary()
    assert summary["valid_samples"] >= 2 and summary["baseline_samples"] == 1
    assert summary["gpu_name"] == "NVIDIA GeForce RTX 5090" and summary["gpu_driver"] == "570.211.01"


def test_failed_nvidia_smi_gives_unknown():
    telemetry = GpuTelemetry(1, own_pid=1, runner=lambda args, timeout=10.: None)
    telemetry.sample()
    assert telemetry.summary()["gpu_sharing"] == "UNKNOWN"


def test_workloads_config_declares_a_to_e():
    config = json.loads((ROOT / "configs" / "rl_benchmark" / "workloads_v1.json").read_text())
    assert config["schema"] == "rcs_rl_benchmark_workloads_v1"
    workloads = {w["id"]: w for w in config["workloads"]}
    assert sorted(workloads) == ["A", "B", "C", "D", "E"]
    assert {i for i, w in workloads.items() if w["p1_0_status"] == "NOT_RUN"} == {"B", "C", "E"}
    assert {i for i, w in workloads.items() if w["p1_0_status"] == "MEASURED"} == {"A", "D"}
    a = workloads["A"]["material"]
    assert (a["solver"], a["cutting_law"], a["particle_spacing_m"], a["grid_dx_m"], a["material_dt_s"],
            a["substeps_per_control_step"], a["control_hz"]) == ("tlmpm_v7_2", 8, 0.001, 0.004, 0.0001, 200, 50)
    assert workloads["A"]["driver"]["real_cutting"] is True
    assert workloads["D"]["scene_seeds"] == [512, 15]
    assert all(w["physics"] == "full" and w["reduced_fidelity"] is False for w in workloads.values())
    assert tuple(config["row_columns"]) == MAIN_COLUMNS
