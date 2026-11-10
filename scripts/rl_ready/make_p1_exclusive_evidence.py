#!/usr/bin/env python3
"""Assemble the compact P1 exclusive-window performance evidence (``p1_exclusive_001.json``) and print its tables.

Inputs (raw, not committed): ``<D>`` = ``artifacts/rl_ready/exclusive_001`` with chain A (workload A, P1-6 R0
harness, seed 512, N=1/4/8/16 x {frozen_v1, alias_free_v1}) and chain B (single_env_headless replays, workload A
seed 15 N=8, batched CUT, workload B1/B2) outputs, the chain logs and the chain-level sampler files. Rows are built
with the frozen row helpers of the benchmark checkout (``--frozen``, integration commit 691b571): ``BenchmarkRow``,
``co_tenant_stats`` and P1-0's ``headless_rows``. single_env_native rows are copied from the P1-0 evidence
(``source: wp17_evidence``), never rerun.

Classification rule (stated in the evidence as ``classification_rule``): a row is EXCLUSIVE_GPU only if
(1) the row's own in-process telemetry (``rl.telemetry.GpuTelemetry``, ~1 Hz incl. a pre-work baseline) labels it
EXCLUSIVE_GPU, i.e. no foreign compute PID, no foreign non-display graphics PID, foreign per-PID SM max <= 5 % and
baseline total utilisation max <= 5 %; (2) every chain-sampler sample (2 s, own = the chain shell's process tree)
inside the row's start..end window shows no foreign compute or non-display graphics PID and foreign SM <= 5 %
(when the sampler was not yet running for a row this is recorded as unavailable, not as passed); (3) the
nvidia-smi compute-app list taken just before (and, chain B, just after) the row holds no process. Otherwise
SHARED_GPU with the measured load. The display server (Xorg) never counts.

    PYTHONPATH=<frozen>/src .venv/bin/python scripts/rl_ready/make_p1_exclusive_evidence.py \
        --out docs/evidence/rl_ready/p1_exclusive_001.json
"""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np

FROZEN = Path("/home/ningan/RoboticCuttingSim-rl-review")
D = Path("/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/exclusive_001")
P15 = Path("/mnt/data/ningan/RoboticCuttingSim_artifacts/rl_ready/p1_5")
SM_LIMIT = 5.
A_CONTACTS = "handle + two measured finger pads (handle_and_measured_pads_v1); reaction computed, not fed back"
CUT_CONTACTS = "material tool/handle box/fixture (ideal tracking, reaction computed, not fed back)"
B_CONTACTS = ("MJWarp robot/tool/fingers/table + material tool/handle/pads/fixture, reaction wrench fed back "
              "(interval k applied in k+1)")


def load(path):
    return json.loads(Path(path).read_text())


def frozen_modules(frozen):
    sys.path.insert(0, str(frozen / "src"))
    from robotic_cutting_sim.rl import benchmark
    spec = importlib.util.spec_from_file_location("make_p1_0_evidence", frozen / "scripts/rl_ready/make_p1_0_evidence.py")
    p10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p10)
    return benchmark, p10


# ----------------------------------------------------------------------------------------------- chain logs
LINE = re.compile(r"^(\S+) (start|end|before|after|skip|moved|move) (\S+)(.*)$")


def parse_chain(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        m = LINE.match(line)
        if not m:
            continue
        ts, kind, name, rest = m.groups()
        t = datetime.fromisoformat(ts).timestamp()
        r = rows.setdefault(name, {"name": name})
        if kind == "start":
            r["start"], r["start_iso"] = t, ts
            fb = re.search(r"foreign_before=\[(.*?)\]", rest)
            if fb:
                r["before_compute_apps"] = [x for x in fb.group(1).split(";") if x.strip()]
            cmd = re.search(r"cmd=\[(.*)\]", rest)
            if cmd:
                r["command"] = cmd.group(1)
        elif kind == "end":
            r["end"], r["end_iso"] = t, ts
            r["rc"] = int(re.search(r"rc=(-?\d+)", rest).group(1))
        elif kind in ("before", "after"):
            apps = re.search(r"compute_apps=\[(.*?)\]", rest).group(1)
            r[f"{kind}_compute_apps"] = [x for x in apps.split(";") if x.strip()]
            r[f"{kind}_util_mem"] = re.search(r"util_mem=\[(.*?)\]", rest).group(1)
            r[f"{kind}_pmon"] = re.search(r"pmon_pid_sm=\[(.*?)\]", rest).group(1).strip()
        elif kind == "skip":
            r["skipped"] = rest.strip()
        elif kind == "moved":
            r["moved"] = rest.strip()
    return rows


def sampler(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def window_stats(samples, start, end):
    w = [s for s in samples if s.get("ok") and start <= s["t"] <= end]
    if not w:
        return dict(available=False, samples=0)
    names = {}
    for s in w:
        for app in s.get("compute_apps_named") or []:
            names[app["pid"]] = app["name"]
    fc = sorted({int(p) for s in w for p in s.get("foreign_compute", {})})
    fg = sorted({int(p) for s in w for p in s.get("foreign_graphics", {})})
    fsm = [s["foreign_sm_pct"] for s in w if s.get("foreign_sm_pct") is not None]
    tot = [s["gpu"]["util_pct"] for s in w if s["gpu"].get("util_pct") is not None]
    own = [s.get("own_memory_mib") or 0. for s in w]
    fmem = [s.get("foreign_memory_mib") or 0. for s in w]
    return dict(available=True, samples=len(w), coverage_s=w[-1]["t"] - w[0]["t"], window_s=end - start,
                foreign_compute_pids=fc, foreign_compute_names={str(p): names.get(p) for p in fc},
                foreign_graphics_pids=fg,
                foreign_sm_pct_max=max(fsm) if fsm else None, foreign_sm_pct_mean=float(np.mean(fsm)) if fsm else None,
                foreign_memory_mib_max=max(fmem), total_util_pct_mean=float(np.mean(tot)) if tot else None,
                own_memory_mib_max=max(own), pmon_samples=len(fsm))


def classify(own_tel, win, chain_row):
    checks = {}
    checks["own_telemetry"] = dict(label=own_tel.get("gpu_sharing"), reasons=own_tel.get("sharing_reasons"),
                                   foreign_compute_pids=own_tel.get("foreign_compute_pids"),
                                   foreign_graphics_pids=own_tel.get("foreign_graphics_pids"),
                                   foreign_sm_pct=own_tel.get("foreign_sm_pct"),
                                   baseline_total_util_pct=own_tel.get("baseline_total_util_pct"),
                                   samples=own_tel.get("valid_samples"))
    ok1 = own_tel.get("gpu_sharing") == "EXCLUSIVE_GPU"
    if win.get("available"):
        ok2 = (not win["foreign_compute_pids"] and not win["foreign_graphics_pids"]
               and (win["foreign_sm_pct_max"] is None or win["foreign_sm_pct_max"] <= SM_LIMIT))
    else:
        ok2 = None
    checks["chain_sampler"] = dict(win, passed=ok2)
    before = chain_row.get("before_compute_apps")
    after = chain_row.get("after_compute_apps")
    ok3 = (before == [] or before is None) and (after == [] or after is None)
    checks["snapshots"] = dict(before=before, after=after, before_util_mem=chain_row.get("before_util_mem"),
                               after_util_mem=chain_row.get("after_util_mem"), before_pmon=chain_row.get("before_pmon"),
                               after_pmon=chain_row.get("after_pmon"), passed=ok3,
                               note=None if before is not None else "no before snapshot recorded")
    exclusive = ok1 and ok2 is not False and ok3 and before is not None
    label = "EXCLUSIVE_GPU" if exclusive else "SHARED_GPU"
    load_parts = []
    if not exclusive:
        sm = (own_tel.get("foreign_sm_pct") or {})
        if sm:
            load_parts.append(f"foreign SM mean {sm.get('mean', 0):.0f} % / max {sm.get('max', 0):.0f} %")
        pids = sorted(set(own_tel.get("foreign_compute_pids") or []) | set(win.get("foreign_compute_pids") or []))
        if pids:
            load_parts.append(f"foreign compute PIDs {pids}")
    return dict(label=label, checks=checks, chain_sampler_used=ok2 is not None,
                measured_load="; ".join(load_parts) or None)


def labelled_from_codes(path):
    with np.load(path) as f:
        keys = list(f.keys())
        codes = f[keys[0]]
    return [int(np.count_nonzero(c)) for c in np.asarray(codes).reshape(len(codes), -1)] if codes.ndim > 1 \
        else [int(np.count_nonzero(codes))]


# ----------------------------------------------------------------------------------------------- row builders
def a_row(B, name, chain_row, samples, seed):
    run = D / name
    r = load(run / "result.json")
    if r.get("status") not in ("OK", "COMPLETE", "PASS") and r.get("error"):
        return dict(name=name, status=r.get("status"), error=str(r.get("error"))[-600:]), None
    row = r["rows"][0]
    tel = r["telemetry"]
    win = window_stats(samples, chain_row.get("start", 0), chain_row.get("end", 0))
    cls = classify(tel, win, chain_row)
    variant = (r.get("variant") or {}).get("particle_contact_mode", "frozen_v1") if isinstance(r.get("variant"), dict) \
        else "frozen_v1"
    rnd = r["rounds"][0] if r.get("rounds") else {}
    invalid = rnd.get("invalid_reason")
    br = B.BenchmarkRow(
        boundary="batched_headless", workload_id=f"A-s{seed}-{variant}", num_envs=row["num_envs"],
        particles_per_env=row["particles_per_env"], control_hz=row["control_hz"], material_dt_s=row["material_dt_s"],
        substeps_per_control_step=row["substeps"], cameras="none", robot_dof=0, contacts=A_CONTACTS,
        recording="none", warm_cold="warm", env_steps_per_s=row["env_steps_per_s"],
        aggregate_sim_time_per_wall=row["aggregate_sim_time_per_wall"], vram_peak_mib=row["vram_peak_mib"],
        reset_mean_s=row["reset_mean_ms"] / 1e3, reset_p95_s=row["reset_p95_ms"] / 1e3,
        physics_verdict="REPORTED_NOT_SCORED", reduced_fidelity=True, gpu_name=tel.get("gpu_name"),
        gpu_index=tel.get("gpu_index"), gpu_driver=tel.get("gpu_driver"), gpu_sharing=cls["label"],
        co_tenant=B.co_tenant_stats(tel), source_commit=r["environment"]["commit"], source=f"measured ({name})",
        notes=(f"P1-6 R0 harness throughput mode, variant {variant}, one process per row (round child), warmup "
               f"{r['warmup_steps']} + {row['timed_steps']} timed batch steps (WP-17 replay steps "
               f"{r['warmup_steps'] + 1}..{r['warmup_steps'] + row['timed_steps']}; tool-material contact starts at "
               f"~314 (s512) / ~321 (s15), so the timed window is pre-contact except the last N=1 steps); 10 timed "
               f"full resets; invalid_reason {invalid}; reduced fidelity: replayed edge, no robot."))
    extra = dict(name=name, chain="A" if seed == 512 else "B", variant=variant, seed=seed, rc=chain_row.get("rc"),
                 start=chain_row.get("start_iso"), end=chain_row.get("end_iso"), status=r.get("status"),
                 timed_steps=row["timed_steps"], warmup_steps=r["warmup_steps"], vram_source=row.get("vram_source"),
                 allocator_peak_mib=row.get("allocator_peak_mib"), final_labelled=rnd.get("final_labelled"),
                 final_step_count=rnd.get("final_step_count"), invalid_reason=invalid,
                 particle_contact_receipt=rnd.get("particle_contact_receipt"),
                 device_hashgrids=dict(after_build=rnd.get("device_hashgrids_after_build"),
                                       after_release=rnd.get("device_hashgrids_after_release")),
                 dumps=r.get("dumps"), wall_s=r.get("wall_s"), commit=r["environment"]["commit"],
                 tree_dirty=r["environment"].get("dirty"), classification=cls)
    return extra, br


def single_rows(B, P10, name, chain_row, samples):
    run = Path(chain_row.get("moved", "").split(" -> ")[-1]) if chain_row.get("moved") else None
    if run is None or not (run / "result.json").exists():
        return dict(name=name, status="NO_RUN_DIR", rc=chain_row.get("rc"), skipped=chain_row.get("skipped")), []
    result = load(run / "result.json")
    if result.get("status") != "COMPLETE":
        return dict(name=name, run=run.name, status=result.get("status"), error=(result.get("error") or "")[-600:]), []
    manifest = load(run / "manifest.json")
    rows, _ = P10.headless_rows(run, manifest["git_head"])
    win = window_stats(samples, chain_row["start"], chain_row["end"])
    cls = classify(result["telemetry"], win, chain_row)
    for row in rows:
        row.gpu_sharing = cls["label"]
        row.source = f"measured ({run.name})"
        row.notes += " Frozen single LiveCuttingMaterial (tlmpm_v7_2, law 8), P1-0 replay command, --warm-resets 5."
    t = result["timing"]
    extra = dict(name=name, run=run.name, seed=result["seed"], rc=chain_row.get("rc"), start=chain_row.get("start_iso"),
                 end=chain_row.get("end_iso"), status=result["status"], git_head=manifest["git_head"],
                 git_dirty=manifest["git_dirty"], warm_step_mean_s=t.get("warm_step_mean_s"),
                 warm_step_p95_s=t.get("warm_step_p95_s"), warm_resets_s=t.get("warm_resets_s"),
                 first_reset_s=t.get("first_reset_s"), invariants=result["invariants"].get("status"),
                 parity_final_codes_equal=(result.get("parity") or {}).get("final_codes_equal"),
                 wall_s=result.get("wall_s"), classification=cls)
    parity = result.get("parity") or {}
    extra["parity"] = {k: parity.get(k) for k in ("final_codes_equal", "final_code_differences",
                                                   "max_position_diff_m", "final_max_position_diff_m")
                       if k in parity}
    return extra, rows


def cut_rows(B, name, chain_row, samples):
    run = Path(chain_row.get("moved", "").split(" -> ")[-1]) if chain_row.get("moved") else None
    if run is None or not (run / "result.json").exists():
        return dict(name=name, status="NO_RUN_DIR", rc=chain_row.get("rc"), skipped=chain_row.get("skipped")), []
    r = load(run / "result.json")
    t, tele = r.get("timing", {}), r["telemetry"]
    singles = {}
    for n in ("cut_v7_2_reference_001", "cut_v7_2_reference_002", "cut_v7_2_reference_003"):
        f = P15 / n / "final_state.npz"
        if f.exists():
            with np.load(f) as z:
                singles[n] = dict(codes=z["codes"].copy(), positions=z["positions"].astype(float))
    episodes = []
    for ep in r.get("episodes", []):
        row = {k: ep.get(k) for k in ("env", "episode", "end", "drive", "step_count", "frames", "verdict", "integrity",
                                       "outcome", "mechanical_components", "piece_mass_fractions", "labelled",
                                       "final_codes_sha256")}
        if ep.get("final_state_file") and singles and ep["drive"] == "reference_command_policy_v1":
            with np.load(run / ep["final_state_file"]) as f:
                codes, pos = f["codes"], f["positions"].astype(float)
            row["different_code_particles_vs_single_refs"] = {
                n: int(np.count_nonzero(codes != s["codes"])) for n, s in singles.items()}
        episodes.append(row)
    ref = [e for e in episodes if e["drive"] == "reference_command_policy_v1" and e["end"] == "TRUNCATED_TIME_LIMIT"]
    verdict = ("PASS" if ref and all(e["verdict"] == "PASS" for e in ref) and (r.get("l2") or {}).get("status") == "PASS"
               else "FAIL") if r["status"] == "COMPLETE" else "FAIL"
    win = window_stats(samples, chain_row["start"], chain_row["end"])
    cls = classify(tele, win, chain_row)
    N = r.get("num_envs")
    extra = dict(name=name, run=run.name, rc=chain_row.get("rc"), start=chain_row.get("start_iso"),
                 end=chain_row.get("end_iso"), status=r["status"], error=(r.get("error") or "")[-600:] or None,
                 num_envs=N, episodes_requested=r.get("episodes_requested"), timing=t,
                 l2={k: v for k, v in (r.get("l2") or {}).items() if k != "audits"},
                 hashgrid_census=r.get("hashgrid_census"), episodes=episodes, verdict=verdict,
                 reference_episodes_pass=f"{sum(e['verdict'] == 'PASS' for e in ref)}/{len(ref)}",
                 git_head=r.get("git_head"), wall_s=r.get("wall_s"), classification=cls)
    if r["status"] != "COMPLETE":
        return extra, []
    br = B.BenchmarkRow(
        boundary="batched_headless", workload_id=f"C2-CUT-v7_2-N{N}", num_envs=N,
        particles_per_env=(r.get("particles_per_env") or [None])[0], control_hz=50., material_dt_s=1e-4,
        substeps_per_control_step=200, cameras="none", robot_dof=0, contacts=CUT_CONTACTS,
        recording="evaluation stream every 20 steps on the device (36 frames/episode); evaluation outside timing",
        warm_cold="warm", env_steps_per_s=t["warm_env_steps_per_s"],
        aggregate_sim_time_per_wall=t["warm_aggregate_sim_time_per_wall"], vram_peak_mib=tele.get("own_vram_peak_mib"),
        reset_mean_s=float(np.mean(t["full_resets_s"])) if t.get("full_resets_s") else None,
        reset_p95_s=t.get("partial_reset_p95_s") or t.get("full_reset_p95_s"), physics_verdict=verdict,
        reduced_fidelity=True, gpu_name=tele.get("gpu_name"), gpu_index=tele.get("gpu_index"),
        gpu_driver=tele.get("gpu_driver"), gpu_sharing=cls["label"], co_tenant=B.co_tenant_stats(tele),
        source_commit=r.get("git_head"), source=f"measured ({run.name})",
        notes=(f"P1-5 run_cut_batched.py, scene bar_flat_board_tlmpm_v7_2_v1, {r.get('episodes_requested')} x "
               f"{r.get('steps_per_episode')}-step episodes, reference_command_policy_v1, frozen_v1 contacts; "
               f"{t['steps']} synchronized env.step calls, mean {t['warm_step_mean_s']:.3f} s/batch step; reset = full "
               "resets; physics verdict = every reference episode frozen sever evaluator PASS and L2 PASS. "
               "reduced_fidelity: ideal tracking, no robot, no reaction feedback, handle box only."))
    return extra, [br]


def b_rows(B, name, chain_row, samples):
    run = D / name
    if not (run / "row.json").exists():
        tail = (D / f"{name}.log").read_text()[-1500:] if (D / f"{name}.log").exists() else None
        return dict(name=name, status="FAIL_NO_ROW" if chain_row.get("start") else "NOT_RUN", rc=chain_row.get("rc"),
                    skipped=chain_row.get("skipped"), log_tail=tail), []
    r = load(run / "row.json")
    if r.get("status") == "BLOCKED":
        return dict(name=name, status="BLOCKED", reason=r.get("reason")), []
    w, tele = r.get("warm") or {}, r.get("telemetry") or {}
    evals = r.get("evaluation") or []
    verdicts = [(e.get("summary") or {}).get("verdict") for e in evals]
    term = r.get("termination_reason")
    multi = (r.get("hashgrids") or {}).get("multi_grid_process")
    complete = w.get("episode_steps_driven") == 1200 and len(evals) == r["N"]
    verdict = "PASS" if complete and all(v == "PASS" for v in verdicts) and not any(term or []) and not multi else "FAIL"
    win = window_stats(samples, chain_row["start"], chain_row["end"])
    cls = classify(tele, win, chain_row)
    b2 = r["tool_mount"] == "physical_grasp_v1"
    wid = "B2-physical-grasp-mjwarp_grip_v1" if b2 else "B1-rigid-mount"
    dof = (r.get("receipt") or {}).get("dofs_per_world")
    extra = dict(name=name, rc=chain_row.get("rc"), start=chain_row.get("start_iso"), end=chain_row.get("end_iso"),
                 N=r["N"], tool_mount=r["tool_mount"], scene_seeds=r.get("scene_seeds"), driver=r.get("driver"),
                 reduced_fidelity=r.get("reduced_fidelity"), fidelity_reasons=r.get("reasons"),
                 listed_differences=r.get("listed_differences"), graph_captured=r.get("graph_captured"),
                 cold=r.get("cold"), warm=w, evaluation=[dict(env=e.get("env"), verdict=v, labelled=e.get("labelled"),
                                                             final_codes_sha256=e.get("final_codes_sha256"))
                                                        for e, v in zip(evals, verdicts)],
                 termination_reason=term, reward_fractions=r.get("reward_fractions"),
                 grasp_trace_summary=r.get("grasp_trace_summary"), hashgrids=r.get("hashgrids", {}).get("status"),
                 hashgrid_cuda_count=r.get("hashgrids", {}).get("cuda_count"), dofs_per_world=dof,
                 particles_per_env=r.get("particles_per_env"), commit=r.get("commit"), verdict=verdict,
                 vram_own_slope_mib_per_s_warm=r.get("vram_own_slope_mib_per_s_warm"),
                 wall_s_total_process=r.get("wall_s_total_process"), classification=cls)
    if not w.get("env_steps_per_s"):
        return extra, []
    parts = r.get("particles_per_env") or [None]
    br = B.BenchmarkRow(
        boundary="batched_headless", workload_id=f"{wid}-N{r['N']}", num_envs=r["N"],
        particles_per_env=int(round(float(np.mean(parts)))) if parts[0] is not None else None, control_hz=50.,
        material_dt_s=1e-4, substeps_per_control_step=int(r.get("material_substeps_per_control_step") or 200),
        cameras="none", robot_dof=dof, contacts=B_CONTACTS, recording="none", warm_cold="warm",
        env_steps_per_s=w["env_steps_per_s"], aggregate_sim_time_per_wall=w["aggregate_sim_time_per_wall"],
        vram_peak_mib=r.get("vram_own_peak_mib"), reset_mean_s=None, reset_p95_s=None, physics_verdict=verdict,
        reduced_fidelity=bool(r.get("reduced_fidelity")), gpu_name=tele.get("gpu_name"), gpu_index=tele.get("gpu_index"),
        gpu_driver=tele.get("gpu_driver"), gpu_sharing=cls["label"], co_tenant=B.co_tenant_stats(tele),
        source_commit=(r.get("commit") or {}).get("sha"), source=f"measured ({name})",
        notes=(f"Isaac Lab Newton/MJWarp kit-less, probe_lab_batched_gpu.py g2, {r['tool_mount']}"
               f"{' + mjwarp_grip_v1 (opt-in profile, physical grasp)' if b2 else ' (rigid mount, reduced fidelity)'}, "
               f"recorded_command_policy_v1 (WP-17 seeds {r.get('scene_seeds')}), 1200 steps, strict, rigid 10 x 2 ms "
               f"substeps + material 200 x 0.1 ms per 50 Hz step, CUDA graph; warm = steps 2..1200 with a device sync "
               f"per step (mean {w.get('step_wall_s_mean', 0):.3f} s, p95 {w.get('step_wall_s_p95', 0):.3f} s); no "
               f"resets in the episode (cold reset {(r.get('cold') or {}).get('reset_s', 0):.2f} s"
               f"{' incl. grasp settle' if b2 else ''}); physics verdict = frozen evaluator PASS in every env, "
               "termination reason 0, one device HashGrid."))
    return extra, [br]


# ----------------------------------------------------------------------------------------------- comparisons
def ratio(a, b):
    return None if not a or not b else a / b


def comparisons(rows_by_key, single, prior):
    out = {}
    # batched vs batched N=1 (same workload/variant/seed, same chain window)
    scal = {}
    for wid in sorted({k[0] for k in rows_by_key}):
        n1 = rows_by_key.get((wid, 1))
        if n1 is None:
            continue
        scal[wid] = {str(n): dict(env_steps_per_s=row.env_steps_per_s, ratio_vs_n1=ratio(row.env_steps_per_s,
                                                                                         n1.env_steps_per_s),
                                  sharing=row.gpu_sharing, n1_sharing=n1.gpu_sharing)
                     for (w, n), row in sorted(rows_by_key.items()) if w == wid}
    out["batched_vs_batched_n1"] = scal
    # batched vs single env (single_env_headless warm, same seed) -- workload A only (same material input)
    vs_single = {}
    for seed in (512, 15):
        s = single.get(seed)
        if s is None:
            continue
        vs_single[f"s{seed}"] = {f"{wid}-N{n}": dict(env_steps_per_s=row.env_steps_per_s,
                                                     single_env_steps_per_s=s.env_steps_per_s,
                                                     ratio=ratio(row.env_steps_per_s, s.env_steps_per_s),
                                                     sharing=row.gpu_sharing, single_sharing=s.gpu_sharing)
                                 for (wid, n), row in sorted(rows_by_key.items()) if wid.startswith(f"A-s{seed}-")}
    out["batched_vs_single_env_headless"] = vs_single
    # alias_free_v1 vs frozen_v1 per N and seed (one variant per process)
    af = {}
    for (wid, n), row in sorted(rows_by_key.items()):
        if wid.endswith("-alias_free_v1"):
            base = rows_by_key.get((wid.replace("alias_free_v1", "frozen_v1"), n))
            if base is not None:
                af[f"{wid.split('-')[1]}-N{n}"] = dict(alias_free_v1=row.env_steps_per_s, frozen_v1=base.env_steps_per_s,
                                                       ratio=ratio(row.env_steps_per_s, base.env_steps_per_s),
                                                       vram_alias_free_v1=row.vram_peak_mib, vram_frozen_v1=base.vram_peak_mib,
                                                       sharing=[row.gpu_sharing, base.gpu_sharing])
    out["alias_free_v1_vs_frozen_v1"] = af
    out["vs_prior_shared_rows"] = prior(rows_by_key, single)
    return out


def prior_rows(frozen):
    def ev(name):
        return load(frozen / "docs/evidence/rl_ready" / name)

    def prior(rows_by_key, single):
        res = {}
        p10 = [r for r in ev("p1_0_baseline_001.json")["rows"] if r["boundary"] == "single_env_headless"
               and r["warm_cold"] == "warm"]
        for seed in (512, 15):
            s = single.get(seed)
            for r in p10:
                if r["workload_id"] == f"A/D-s{seed}" and s is not None:
                    res.setdefault("p1_0_single_env_headless", []).append(dict(
                        seed=seed, prior_env_steps_per_s=r["env_steps_per_s"], prior_sharing=r["gpu_sharing"],
                        prior_source=r["source"], prior_notes_run=r["notes"][-40:], exclusive_now=s.env_steps_per_s,
                        now_sharing=s.gpu_sharing, ratio_now_vs_prior=ratio(s.env_steps_per_s, r["env_steps_per_s"])))
        p12 = ev("p1_2_milestones_001.json")
        for r in p12["rows"]:
            if r["boundary"] != "batched_headless" or r["warm_cold"] != "warm" or not r.get("env_steps_per_s"):
                continue
            n = r["num_envs"]
            now = rows_by_key.get(("A-s512-frozen_v1", n))
            res.setdefault("p1_2_batched_workload_a", []).append(dict(
                n=n, prior_env_steps_per_s=r["env_steps_per_s"], prior_sharing=r["gpu_sharing"],
                prior_foreign_sm_mean=(r.get("co_tenant") or {}).get("foreign_sm_pct_mean"), prior_source=r["source"],
                exclusive_now_s512_frozen_v1=None if now is None else now.env_steps_per_s,
                now_sharing=None if now is None else now.gpu_sharing,
                ratio_now_vs_prior=None if now is None else ratio(now.env_steps_per_s, r["env_steps_per_s"]),
                note="P1-2 rows: frozen_v1, replay drive, envs cycle seeds 512/15 at N>=8 (s512 only at N=1 m2_n1_001), "
                     "longer windows incl. contact; now: harness throughput window, all envs s512"))
        p15 = ev("p1_5_cut_001.json")
        for r in p15.get("rows", []):
            if r["boundary"] == "batched_headless":
                n = r["num_envs"]
                now = rows_by_key.get((f"C2-CUT-v7_2-N{n}".rsplit("-N", 1)[0], n))
                res.setdefault("p1_5_batched_cut", []).append(dict(
                    n=n, prior_env_steps_per_s=r["env_steps_per_s"], prior_sharing=r["gpu_sharing"],
                    prior_foreign_sm_mean=(r.get("co_tenant") or {}).get("foreign_sm_pct_mean"), prior_source=r["source"],
                    exclusive_now=None if now is None else now.env_steps_per_s,
                    now_sharing=None if now is None else now.gpu_sharing,
                    ratio_now_vs_prior=None if now is None else ratio(now.env_steps_per_s, r["env_steps_per_s"])))
        return res
    return prior


# ----------------------------------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frozen", type=Path, default=FROZEN)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    B, P10 = frozen_modules(args.frozen)
    chain_a = parse_chain(D / "chain_a.log")
    chain_b = parse_chain(D / "chain_b.log") if (D / "chain_b.log").exists() else {}
    sa, sb = sampler(D / "sampler_chain_a.jsonl"), sampler(D / "sampler_chain_b.jsonl")
    runs, rows, failures = {}, [], []

    p10 = load(args.frozen / "docs/evidence/rl_ready/p1_0_baseline_001.json")
    native = [B.BenchmarkRow.from_dict(r) for r in p10["rows"] if r["boundary"] == "single_env_native"]
    rows += native

    single = {}
    for seed in (15, 512):
        name = f"single_s{seed}"
        if name not in chain_b:
            runs[name] = dict(name=name, status="NOT_RUN")
            continue
        extra, rr = single_rows(B, P10, name, chain_b[name], sb)
        runs[name] = extra
        rows += rr
        for row in rr:
            if row.warm_cold == "warm":
                single[seed] = row
        if not rr:
            failures.append(extra)

    rows_by_key = {}
    for chain, seed, log in (("A", 512, chain_a), ("B", 15, chain_b)):
        for name, cr in log.items():
            if not name.startswith("a_n"):
                continue
            if "end" not in cr:
                runs[name] = dict(name=name, status="NOT_FINISHED", start=cr.get("start_iso"))
                failures.append(runs[name])
                continue
            extra, br = a_row(B, name, cr, sa if chain == "A" else sb, seed)
            runs[name] = extra
            if br is None:
                failures.append(extra)
                continue
            rows.append(br)
            rows_by_key[(br.workload_id, br.num_envs)] = br
            if not extra["classification"]["chain_sampler_used"]:
                extra["classification"]["note"] = "chain sampler started after this row began; rule (2) unavailable"

    for name in ("cut_n1", "cut_n8", "cut_n16"):
        cr = chain_b.get(name)
        if cr is None or "end" not in cr:
            runs[name] = dict(name=name, status="NOT_RUN" if cr is None or "start" not in cr else "NOT_FINISHED",
                              skipped=(cr or {}).get("skipped"))
            continue
        extra, rr = cut_rows(B, name, cr, sb)
        runs[name] = extra
        rows += rr
        for row in rr:
            rows_by_key[("C2-CUT-v7_2", row.num_envs)] = row
        if not rr or extra.get("verdict") != "PASS":
            failures.append({k: extra.get(k) for k in ("name", "status", "verdict", "error")})

    for name in ("b1_n1", "b1_n4", "b1_n8", "b1_n16", "b2_n4"):
        cr = chain_b.get(name)
        if cr is None or "end" not in cr:
            runs[name] = dict(name=name, status="NOT_RUN" if cr is None or "start" not in cr else "NOT_FINISHED",
                              skipped=(cr or {}).get("skipped"))
            continue
        extra, rr = b_rows(B, name, cr, sb)
        runs[name] = extra
        rows += rr
        for row in rr:
            rows_by_key[(row.workload_id.rsplit("-N", 1)[0], row.num_envs)] = row
        if not rr or extra.get("verdict") != "PASS":
            failures.append({k: extra.get(k) for k in ("name", "status", "verdict", "rc", "log_tail")})

    comps = comparisons(rows_by_key, single, prior_rows(args.frozen))
    gpu = next((r for r in rows if r.gpu_index == 1 and r.gpu_driver), None)
    evidence = dict(
        schema="rcs_rl_p1_exclusive_perf_v1",
        purpose="Phase 1 acceptance performance rows measured in an exclusive window on GPU1",
        benchmark_checkout=str(args.frozen), benchmark_commit="691b57157c89dd876ebdd1b377fd25aeba4a55ce",
        gpu=dict(name=gpu.gpu_name if gpu else None, index=1, driver=gpu.gpu_driver if gpu else None,
                 memory_total_mib=32607, cuda_runtime_reported_by_driver="12.8"),
        raw_outputs=str(D),
        classification_rule=__doc__.split("Classification rule")[1].split("    PYTHONPATH")[0].strip(),
        boundaries_note=("single_env_native (WP-17 Native, Kit, cameras, PhysX robot; copied from P1-0 evidence), "
                         "single_env_headless (frozen single LiveCuttingMaterial replay, no Kit) and batched_headless "
                         "(VectorCuttingEnv / Isaac Lab batch) are separate measurement boundaries; no speedup merges "
                         "them."),
        fidelity_labels=dict(
            A="reduced fidelity: replayed WP-17 edge and measured pads through step_measured_edge, no robot, reaction "
              "computed but not fed back",
            C2_CUT="reduced fidelity: versioned v7_2 CUT scene (bar_flat_board_tlmpm_v7_2_v1), ideal tracking, no robot",
            B1="rigid tool mount (reduced fidelity: rigid_tool_mount_no_grasp, finger_opening_fixed_nominal), MJWarp "
               "robot + material with reaction feedback",
            B2="physical grasp (physical_grasp_v1) with the opt-in mjwarp_grip_v1 contact profile, not reduced"),
        sampler_coverage=dict(chain_a=dict(samples=len(sa), first=sa[0]["t"] if sa else None,
                                           last=sa[-1]["t"] if sa else None),
                              chain_b=dict(samples=len(sb), first=sb[0]["t"] if sb else None,
                                           last=sb[-1]["t"] if sb else None)),
        runs=runs, rows=[r.to_dict() for r in rows], comparisons=comps, failures_or_not_run=failures,
        markdown_table=B.markdown_table(rows))
    out = Path(args.out)
    with out.open("x") as stream:
        json.dump(evidence, stream, indent=1, sort_keys=True, default=str)
    print(evidence["markdown_table"])
    print(json.dumps(comps, indent=1, default=str)[:6000])
    print(hashlib.sha256(out.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
