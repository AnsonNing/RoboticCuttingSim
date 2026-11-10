"""P1-6 R0 (decision D-11): the performance acceptance config, the harness pieces and the judge's verdict logic.

CPU only. The judge runs on synthetic outputs (small synthetic Native references injected through
``Judge(natives=...)``); the real WP-17 references and the GPU harness are exercised by the R0 GPU smoke
(``docs/evidence/rl_ready/p1_6_r0_001.json``). One Torch-gated test runs the harness end to end on the CPU for one
control step (about a minute).
"""
import fnmatch
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from robotic_cutting_sim.rl import perf_acceptance as PA

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "rl_benchmark" / "acceptance_v1.json"
HARNESS = ROOT / "scripts" / "rl_ready" / "run_perf_acceptance.py"
JUDGE = ROOT / "scripts" / "rl_ready" / "judge_perf_acceptance.py"


def load_script(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def judge_mod():
    return load_script(JUDGE, "judge_perf_acceptance_under_test")


@pytest.fixture(scope="module")
def harness_mod():
    return load_script(HARNESS, "run_perf_acceptance_under_test")


def tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [x for x in out.decode().split("\0") if x]


# ----------------------------------------------------------------------------- config

def test_acceptance_v1_config_is_complete_and_matches_the_contract():
    cfg = json.loads(CONFIG.read_text())
    assert PA.config_problems(cfg) == []
    assert cfg["schema"] == "rcs_perf_acceptance_v1" and cfg["version"] == "acceptance_v1"
    for key in ("schema", "version", "harness", "judge", "evaluator_files", "reference_variant", "seeds",
                "gpu_mem_mib", "plan", "performance", "cpu_tests", "baseline_measure", "frozen_paths", "dev_limits"):
        assert key in cfg     # the external caller's CONFIG_KEYS
    assert cfg["reference_variant"] == {"particle_contact_mode": "alias_free_v1"}
    assert cfg["seeds"] == {"dev": [512], "heldout": [15]}
    assert (ROOT / cfg["harness"]).is_file() and (ROOT / cfg["judge"]).is_file()
    th = cfg["thresholds"]
    assert (th["primary_median_min"], th["primary_pairs_above_1"], th["all_median_min"]) == (1.05, 4, 0.97)
    assert (th["vram_ratio_max"], th["vram_slack_mib"]) == (1.10, 256)
    assert (th["reset_p95_ratio_max"], th["reset_p95_slack_ms"], th["clock_tolerance"]) == (1.25, 2.0, 0.05)
    assert th["reset_p95_shared_slack_ms"] == 10.0 and "shared_gpu_reset_bimodality" in th["reset_p95_rule"]
    perf = cfg["performance"]
    assert perf["num_envs"] == [1, 8, 16] and perf["primary_num_envs"] == 16 and perf["pairs"] == 5
    assert perf["order"] == "BCCBBCCBBC"
    gates = {e["gate"]: e for e in cfg["plan"]}
    assert set(gates) == {"G4", "G6", "G7", "G8"}
    assert gates["G6"]["seeds"] == [512, 15] and gates["G6"]["repeats"] == 3 and gates["G6"]["steps"] == 1200
    assert gates["G7"]["num_envs"] == 4 and gates["G8"]["variant"] == "default"
    # the P1-0 min-max band is a swappable, descriptive input (never a hard gate until a pre-declared band lands)
    band = cfg["physics"]["position_band"]
    assert band == {"path": "configs/rl_benchmark/position_band_p1_0.json", "mode": "descriptive"}
    loaded = PA.load_position_band(ROOT, band)
    assert loaded["high_m"] == {"512": 3.9e-05, "15": 4.6e-04} and loaded["status"] == "descriptive_only"
    assert set(cfg["gpu_mem_mib"]) >= {str(n) for n in perf["num_envs"]} | {"4"}


def test_reference_variant_is_an_accepted_open_vector_env_option():
    cfg = json.loads(CONFIG.read_text())
    assert PA.check_variant(cfg["reference_variant"]) == cfg["reference_variant"]


def test_evaluator_files_and_frozen_paths_cover_the_evaluator():
    cfg = json.loads(CONFIG.read_text())
    files = tracked()

    def covered(path, globs):
        return any(fnmatch.fnmatchcase(path, g) for g in globs)
    for glob in cfg["evaluator_files"] + cfg["frozen_paths"]:
        assert any(fnmatch.fnmatchcase(f, glob) for f in files), glob
    for path in (cfg["harness"], cfg["judge"], "configs/rl_benchmark/acceptance_v1.json",
                 "src/robotic_cutting_sim/rl/perf_acceptance.py", "src/robotic_cutting_sim/rl/replay.py",
                 "src/robotic_cutting_sim/rl/benchmark.py", "src/robotic_cutting_sim/rl/telemetry.py",
                 "src/robotic_cutting_sim/goals.py", "src/robotic_cutting_sim/seeded_trim.py",
                 "src/robotic_cutting_sim/recording.py", "scripts/rl_ready/run_p1_3a.py",
                 "scripts/rl_ready/run_vector_milestones.py", "tests/test_perf_acceptance.py"):
        assert covered(path, cfg["evaluator_files"]), path
    for path in ("src/robotic_cutting_sim/tlmpm_v5.py", "src/robotic_cutting_sim/tlmpm_v7_2.py",
                 "src/robotic_cutting_sim/live_material.py", "src/robotic_cutting_sim/goals.py"):
        assert covered(path, cfg["frozen_paths"]), path
    # the code a candidate is meant to change is not frozen
    for path in ("src/robotic_cutting_sim/batched/particle_contacts.py", "src/robotic_cutting_sim/rl/vector_env.py"):
        assert not covered(path, cfg["frozen_paths"] + cfg["evaluator_files"]), path


@pytest.mark.parametrize("mutate, message", [
    (lambda c: c["seeds"].update(heldout=[512]), "seeds"),
    (lambda c: c["performance"].update(order="BBCC"), "performance"),
    (lambda c: c["plan"][0].update(steps=1201), "warmup_steps + steps"),
    (lambda c: c["thresholds"].pop("clock_tolerance"), "thresholds"),
    (lambda c: c.pop("physics"), "missing physics"),
    (lambda c: c.update(harness="/abs/path.py"), "harness"),
    (lambda c: c["physics"]["position_band"].update(mode="hard"), "position_band"),
])
def test_config_problems_are_reported(mutate, message):
    cfg = json.loads(CONFIG.read_text())
    mutate(cfg)
    assert any(message in p for p in PA.config_problems(cfg))


# ----------------------------------------------------------------------------- marks, rounds, variants, codec

def test_marks_round_trip_and_reject_malformed_lines():
    line = PA.format_mark("warm_end", 20, 16)
    assert line.startswith("RCS_PERF_MARK {")
    assert PA.parse_mark(line) == {"event": "warm_end", "global_step": 20, "num_envs": 16}
    assert json.loads(line.split(" ", 1)[1]) == {"event": "warm_end", "global_step": 20, "num_envs": 16}
    for bad in ("RCS_PERF_MARK {", 'RCS_PERF_MARK {"event": "other", "global_step": 1, "num_envs": 1}',
                'RCS_PERF_MARK {"event": "warm_end", "global_step": 1.5, "num_envs": 1}',
                'RCS_PERF_MARK {"event": "warm_end", "global_step": true, "num_envs": 1}',
                'x RCS_PERF_MARK {"event": "warm_end", "global_step": 1, "num_envs": 1}', "Warp 1.17.0 initialized"):
        assert PA.parse_mark(bad) is None
    with pytest.raises(ValueError):
        PA.format_mark("start", 0, 1)


def test_outer_rate_uses_last_warm_end_and_first_episode_end_after_it():
    def m(t, event, step, n=2):
        return {"t_mono": t, "mark": {"event": event, "global_step": step, "num_envs": n}}
    marks = [m(0, "warm_begin", 0), m(1, "warm_end", 2), m(3, "episode_end", 12),
             m(4, "warm_begin", 12), m(5, "warm_end", 14), m(9, "episode_end", 34), m(11, "episode_end", 99)]
    rate = PA.outer_rate(marks)
    assert rate["vector_steps"] == 20 and rate["env_steps"] == 40 and rate["seconds"] == 4
    assert rate["env_steps_per_s"] == pytest.approx(10.)
    assert PA.outer_rate(marks[:2]) is None


def test_plan_rounds_one_batch_at_a_time():
    assert PA.plan_rounds("episode", [512, 15], 1) == [(512,), (15,)]
    assert PA.plan_rounds("episode", [512, 15], 4) == [(512, 15, 512, 15)]
    assert PA.plan_rounds("throughput", [512], 16) == [(512,) * 16]
    assert PA.plan_rounds("samestate", [512, 15], 2) == [(512, 512), (15, 15)]
    for bad in (([], 1), ([512, 512], 1), ([512], 0)):
        with pytest.raises(ValueError):
            PA.plan_rounds("episode", *bad)


@pytest.mark.parametrize("variant", [{"bogus_option": 1}, {"num_envs": 3}, {"task": None}, {"device": "cpu"}, [1]])
def test_unknown_or_reserved_variant_option_raises_unsupported_option(variant):
    from robotic_cutting_sim.batched.errors import UnsupportedBatchConfiguration
    with pytest.raises(UnsupportedBatchConfiguration) as info:
        PA.check_variant(variant)
    assert info.value.code == "UNSUPPORTED_OPTION"


def test_known_variant_options_pass():
    assert PA.check_variant({}) == {}
    assert PA.check_variant({"particle_contact_mode": "alias_free_v1",
                             "particle_contact_options": {"speed_guard": "fallback"}})["particle_contact_mode"] == \
        "alias_free_v1"


def test_position_codec_bounds_and_escapes():
    rng = np.random.default_rng(0)
    native = rng.uniform(-0.3, 0.3, (5, 40, 3))
    pos = native + rng.normal(0, 3e-5, native.shape)
    pos[1, 3, 0] += 0.01                    # a large delta (escape, exact)
    pos[2, 5, 2] = np.nan                   # non-finite (escape)
    enc = PA.encode_positions(native, pos)
    assert enc["dq"].dtype == np.int8 and enc["escape_index"].size == 2
    dec = PA.decode_positions(native, **{"dq": enc["dq"], "escape_index": enc["escape_index"],
                                         "escape_delta": enc["escape_delta"]})
    finite = np.isfinite(pos)
    assert np.max(np.abs(dec[finite] - pos[finite])) <= 0.5e-6 + 1e-12
    assert dec[1, 3, 0] == pytest.approx(pos[1, 3, 0], abs=1e-12) and np.isnan(dec[2, 5, 2])
    with pytest.raises(ValueError):
        PA.decode_positions(native, enc["dq"], enc["escape_index"][:1], enc["escape_delta"][:1])


def test_junit_summary(tmp_path):
    (tmp_path / "j.xml").write_text(
        '<testsuites><testsuite><testcase classname="a" name="t1"/><testcase classname="a" name="t2">'
        '<failure/></testcase><testcase classname="a" name="t3"><skipped/></testcase></testsuite></testsuites>')
    s = PA.junit_summary(tmp_path / "j.xml")
    assert (s["tests"], s["passed"], s["failed"], s["skipped"]) == (3, 1, 1, 1)
    assert s["failed_ids"] == ["a::t2"] and s["skipped_ids"] == ["a::t3"]


# ----------------------------------------------------------------------------- harness command line

def test_harness_refuses_existing_out_and_writes_nothing(harness_mod, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (tmp_path / "v.json").write_text("{}")
    rc = harness_mod.main(["--variant-json", str(tmp_path / "v.json"), "--mode", "throughput", "--num-envs", "1",
                           "--seeds", "512", "--steps", "1", "--device", "cpu", "--out", str(out)])
    assert rc == 2 and list(out.iterdir()) == []


def test_harness_unknown_variant_raises_unsupported_option_with_a_result(harness_mod, tmp_path, capsys):
    (tmp_path / "v.json").write_text(json.dumps({"made_up_flag": True}))
    out = tmp_path / "run"
    rc = harness_mod.main(["--variant-json", str(tmp_path / "v.json"), "--mode", "episode", "--num-envs", "1",
                           "--seeds", "512", "--steps", "1", "--device", "cpu", "--out", str(out)])
    res = json.loads((out / "result.json").read_text())
    assert rc == 3 and res["status"] == "ERROR" and res["error_code"] == "UNSUPPORTED_OPTION"
    assert res["schema"] == "rcs_perf_result_v1" and res["environment"]["package_root_inside_checkout"]
    assert "RCS_PERF_MARK" not in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--steps", "0"], ["--steps", "1200", "--warmup-steps", "1"],
                                  ["--steps", "5", "--seeds", "512,512"], ["--steps", "5", "--num-envs", "0"],
                                  ["--steps", "5", "--mode", "train"]])
def test_harness_argument_validation(harness_mod, tmp_path, argv):
    base = {"--variant-json": "v.json", "--mode": "throughput", "--num-envs": "1", "--seeds": "512",
            "--out": str(tmp_path / "o")}
    for i in range(0, len(argv), 2):
        base[argv[i]] = argv[i + 1]
    with pytest.raises(SystemExit) as info:
        harness_mod.parse_args([x for kv in base.items() for x in kv])
    assert info.value.code == 2


def test_harness_child_argv_round_trips(harness_mod, tmp_path):
    args = harness_mod.parse_args(["--variant-json", "v.json", "--mode", "episode", "--num-envs", "1", "--seeds",
                                   "512,15", "--steps", "5", "--gpu-uuid", "GPU-x", "--out", str(tmp_path / "o")])
    child = harness_mod.parse_args(harness_mod.child_argv(args, 1, 5))
    assert (child.child_round, child.child_global_step, child.seed_list, child.gpu_uuid) == (1, 5, [512, 15], "GPU-x")
    assert (child.mode, child.num_envs, child.steps, child.out) == ("episode", 1, 5, args.out)


def test_harness_cpu_one_step_end_to_end(tmp_path):
    """The real harness process on the CPU (Torch venv only): marks after the steps, result.json, dumps."""
    pytest.importorskip("torch")
    (tmp_path / "v.json").write_text("{}")
    out = tmp_path / "run"
    proc = subprocess.run([sys.executable, str(HARNESS), "--variant-json", str(tmp_path / "v.json"), "--mode",
                           "throughput", "--num-envs", "1", "--seeds", "512", "--warmup-steps", "0", "--steps", "1",
                           "--device", "cpu", "--out", str(out)], cwd=ROOT, capture_output=True, text=True,
                          timeout=1800, env=dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUDA_VISIBLE_DEVICES=""))
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
    marks = [PA.parse_mark(x) for x in proc.stdout.splitlines() if PA.parse_mark(x)]
    assert [m["event"] for m in marks] == ["warm_begin", "warm_end", "episode_end"]
    assert [m["global_step"] for m in marks] == [0, 0, 1] and {m["num_envs"] for m in marks} == {1}
    res = json.loads((out / "result.json").read_text())
    row = res["rows"][0]
    assert res["status"] == "COMPLETE" and row["num_envs"] == 1 and row["particles_per_env"] > 10 ** 5
    assert row["control_hz"] == 50 and row["material_dt_s"] == pytest.approx(1e-4) and row["substeps"] == 200
    assert row["cameras"] == "none" and row["recording"] == "none" and row["robot_dof"] == 0
    assert res["rounds"][0]["final_step_count"] == [1] and len(res["rounds"][0]["reset_ms"]) == 10
    assert all((out / d["path"]).is_file() for d in res["dumps"])
    assert res["process_model"].startswith("one child process per round") and len(res["round_pids"]) == 1


# ----------------------------------------------------------------------------- judge on synthetic outputs

P = 40
SEED = 512


class FakeTask:
    particle_spacing_m = 0.001

    def evaluate_goal(self, data):
        ok = int(np.asarray(data["codes"])[-1].sum()) > 0
        return {"verdict": "PASS" if ok else "FAIL", "integrity": {"status": "PASS"},
                "outcome": {"status": "PASS" if ok else "FAIL"}}


class FakeNatives:
    """121 Native samples (0.2 s apart) of P particles; codes label half the particles from step 600 on."""

    def __init__(self):
        rng = np.random.default_rng(1)
        time_s = np.arange(121) * 0.2
        base = rng.uniform(-0.1, 0.1, (P, 3))
        pos = np.stack([base + 1e-4 * k for k in range(121)])
        codes = np.zeros((121, P), np.int32)
        codes[60:, : P // 2] = 1
        ref = dict(mass_kg=np.full(P, 1e-3), reference_positions=base, reference_stencil_nodes=np.zeros((P, 8), int),
                   reference_stencil_weights=np.zeros((P, 8), np.float32),
                   reference_stencil_gradients=np.zeros((P, 8, 3), np.float32))
        self.entry = dict(data=dict(positions=pos, codes=codes, time_s=time_s), task=FakeTask(), ref=ref,
                          steps=np.rint(time_s / 0.02).astype(int), particles=P)

    def get(self, seed):
        return self.entry


def synthetic_config():
    cfg = json.loads(CONFIG.read_text())
    cfg["version"] = "acceptance_v1_test"
    cfg["plan"] = [dict(id="G4_samestate", gate="G4", mode="samestate", num_envs=1, seeds=[SEED], warmup_steps=0,
                        steps=300, variant="proposal"),
                   dict(id="G6_replay", gate="G6", mode="episode", num_envs=1, seeds=[SEED], warmup_steps=0,
                        steps=1200, variant="proposal", repeats=2)]
    cfg["performance"] = dict(num_envs=[1, 2], primary_num_envs=2, pairs=2, order="BCCB", warmup_steps=2, steps=10,
                              seeds=[SEED])
    cfg["thresholds"]["primary_pairs_above_1"] = 2
    cfg["torch_tests"] = None
    cfg["seeds"] = {"dev": [SEED], "heldout": [15]}
    assert PA.config_problems(cfg) == []
    return cfg


class Acceptance:
    """Synthetic manifest + results + dumps in the layout the external caller and the harness produce."""

    def __init__(self, root, judge_mod):
        self.root, self.judge_mod = Path(root), judge_mod
        self.cfg = synthetic_config()
        self.cfg_path = self.root / "acceptance.json"
        self.cfg_path.write_text(json.dumps(self.cfg))
        self.native = FakeNatives().entry
        self.checkouts = {s: self.root / s for s in ("baseline", "candidate")}
        for c in self.checkouts.values():
            (c / "src" / "robotic_cutting_sim").mkdir(parents=True)
        self.commits = {"baseline": "b" * 40, "candidate": "c" * 40}
        self.proposal = {"particle_contact_mode": "fast_v2"}
        self.runs = []
        self.shas = dict(harness_sha256=PA.sha256_file(HARNESS),
                         shared_module_sha256=PA.sha256_file(Path(PA.__file__).resolve()))
        # knobs for the failure scenarios
        self.candidate_rate = 1.2
        self.candidate_inner_factor = 1.0
        self.candidate_code_flip = 0
        self.tamper_candidate_dump = False
        self.drop_baseline_results = False
        self.candidate_error = False
        self.candidate_position_offset = 2e-6
        self.sharing = "SHARED_GPU"
        self.baseline_reset_ms = 7.0
        self.candidate_reset_ms = 7.0

    def variant(self, side, key):
        return {} if key == "default" else self.cfg["reference_variant"] if side == "baseline" else self.proposal

    def write_result(self, rid, side, mode, n, warm, steps, vkey, rows, rounds, dumps):
        d = self.root / "runs" / rid / "harness"
        (d / "dumps").mkdir(parents=True)
        meta = []
        for name, info, arrays in dumps:
            np.savez_compressed(d / "dumps" / name, **arrays)
            if self.tamper_candidate_dump and side == "candidate":
                with (d / "dumps" / name).open("ab") as f:
                    f.write(b"\0")
            meta.append(dict(info, path=f"dumps/{name}", sha256=PA.sha256_file(d / "dumps" / name)
                             if not (self.tamper_candidate_dump and side == "candidate") else "0" * 64))
        status = "ERROR" if (self.candidate_error and side == "candidate") else "COMPLETE"
        res = dict(schema=PA.SCHEMA_RESULT, harness=PA.HARNESS_VERSION, status=status, error_code=None,
                   mode=mode, num_envs=n, seeds=[SEED], warmup_steps=warm, steps=steps,
                   variant=self.variant(side, vkey), rows=rows, rounds=rounds, dumps=meta,
                   environment=dict(commit=self.commits[side],
                                    package_root=str(self.checkouts[side] / "src" / "robotic_cutting_sim"),
                                    **self.shas))
        (d / "result.json").write_text(json.dumps(res))
        return None if (self.drop_baseline_results and side == "baseline") else str(d / "result.json")

    def add_run(self, rid, side, gate, layer, mode, n, warm, steps, vkey, rows, rounds, dumps, outer=None, **extra):
        result = self.write_result(rid, side, mode, n, warm, steps, vkey, rows, rounds, dumps)
        self.runs.append(dict(id=rid, side=side, commit=self.commits[side], gate=gate, layer=layer, mode=mode,
                              num_envs=n, seeds=[SEED], variant=vkey, rc=0, wall_s=1.0, result=result,
                              infra=None if result else "no result.json (exit 1)", outer=outer,
                              gpu={"label": "SHARED_GPU", "util_p50": 60}, environment_problems=[],
                              gpu_guard="PASS", cmd=["python", "run_perf_acceptance.py", "--out",
                                                     str(self.root / "runs" / rid / "harness")], **extra))

    def episode_dump(self, side):
        nat = self.native
        steps = nat["steps"][1:]
        idx = np.arange(1, 121)
        pos = nat["data"]["positions"][idx] + (self.candidate_position_offset if side == "candidate" else 2e-6)
        enc = PA.encode_positions(nat["data"]["positions"][idx], pos)
        codes = nat["data"]["codes"][idx].copy()
        final_codes = codes[-1].copy()
        if side == "candidate" and self.candidate_code_flip:
            final_codes[-self.candidate_code_flip:] = 2
            codes[-1] = final_codes
        arrays = dict(seed=np.int64(SEED), env=np.int64(0), steps=np.int64(1200), sample_step=steps.astype(np.int64),
                      sample_index=idx.astype(np.int64), sample_codes=codes, sample_dq=enc["dq"],
                      sample_escape_index=enc["escape_index"], sample_escape_delta=enc["escape_delta"],
                      final_positions=pos[-1], final_velocities=np.zeros((P, 3), np.float32), final_codes=final_codes,
                      final_step_count=np.int64(1200), contacts=np.full(1200, 7, np.int64), invalid_reason=np.int64(0))
        rounds = [dict(round=0, invalid_reason=[0], final_step_count=[1200],
                       l2_candidate_reported=dict(code_monotone_violations=[0], nonfinite_steps=[0], status="PASS"))]
        return [("r0_e0_s512.npz", dict(kind="episode", round=0, env=0, seed=SEED, samples=120), arrays)], rounds

    def samestate_dump(self, side):
        imp = np.zeros((P, 3), np.float32)
        imp[:5] = 1e-3
        stats = np.zeros((1, 8))
        stats[0, 4] = 5
        codes = self.native["data"]["codes"][30].astype(np.int32)
        arrays = {"impulse_a_k33": imp, "impulse_b_k33": imp, "stats_sum_a_k33": stats, "stats_sum_b_k33": stats,
                  "stats_max_a_k33": stats, "stats_max_b_k33": stats, "final_codes_e0": codes,
                  "final_positions_e0": self.native["data"]["positions"][30], "contacts": np.full((300, 1), 3)}
        rounds = [dict(round=0, final_step_count=[300])]
        return [("r0_samestate_s512.npz", dict(kind="samestate", round=0, seed=SEED, env_seeds=[SEED], samples=[33],
                                               steps=300), arrays)], rounds

    def build(self):
        for side in ("baseline", "candidate"):
            dumps, rounds = self.samestate_dump(side)
            self.add_run(f"G4_samestate_{side}_r0", side, "G4", "PHYSICS", "samestate", 1, 0, 300, "reference"
                         if side == "baseline" else "proposal", [], rounds, dumps, plan_id="G4_samestate", repeat=0)
        for r in range(2):
            for side in ("baseline", "candidate"):
                dumps, rounds = self.episode_dump(side)
                self.add_run(f"G6_replay_{side}_r{r}", side, "G6", "PHYSICS", "episode", 1, 0, 1200, "reference"
                             if side == "baseline" else "proposal", [], rounds, dumps, plan_id="G6_replay", repeat=r)
        perf = self.cfg["performance"]
        for n in perf["num_envs"]:
            for i, ch in enumerate(perf["order"]):
                side = "baseline" if ch == "B" else "candidate"
                rate = 5.0 * n * (self.candidate_rate if side == "candidate" else 1.0)
                inner = rate * (self.candidate_inner_factor if side == "candidate" else 1.0)
                steps = perf["steps"]
                outer = dict(env_steps=steps * n, seconds=steps * n / rate, env_steps_per_s=rate, num_envs=n,
                             vector_steps=steps)
                row = dict(num_envs=n, env_steps_per_s=inner, vram_peak_mib=1000., vram_source="nvidia_smi_process",
                           reset_p95_ms=self.candidate_reset_ms if side == "candidate" else self.baseline_reset_ms,
                           control_hz=50., gpu_sharing=self.sharing)
                rounds = [dict(round=0, final_step_count=[12] * n, final_labelled=[{"0": {"labelled": 0}}] * n)]
                self.add_run(f"perf_n{n}_{i:02d}_{ch}", side, "G9", "PERFORMANCE", "throughput", n, 2, steps,
                             "reference" if side == "baseline" else "proposal", [row], rounds, [], outer=outer,
                             plan_id=f"perf_n{n}", pair=i // 2, order=i)
        suite = dict(tests=10, passed=10, failed=0, errors=0, skipped=1, failed_ids=[], skipped_ids=["t::gpu"], rc=0)
        manifest = dict(schema=PA.SCHEMA_MANIFEST, config="acceptance_v1_test", config_path=str(self.cfg_path),
                        config_sha256=PA.sha256_file(self.cfg_path),
                        baseline=dict(commit=self.commits["baseline"], variant=self.cfg["reference_variant"],
                                      checkout=str(self.checkouts["baseline"])),
                        candidate=dict(commit=self.commits["candidate"], variant=self.proposal,
                                       checkout=str(self.checkouts["candidate"])),
                        evaluator=dict(commit=self.commits["baseline"], files_sha256="e" * 64, files=10),
                        runs=self.runs, cpu_tests={"baseline": {"cpu_tests": suite}, "candidate": {"cpu_tests": suite}},
                        wp17=str(self.root / "wp17"), seeds=self.cfg["seeds"], blocked=None, clock_tolerance=0.05)
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(manifest))
        return self

    def verdict(self, band_mode=None):
        J = self.judge_mod
        cfg = PA.load_config(self.cfg_path)
        if band_mode:
            cfg["physics"]["position_band"]["mode"] = band_mode
        manifest = json.loads(self.manifest_path.read_text())
        return J.Judge(cfg, self.cfg_path, manifest, self.manifest_path, natives=FakeNatives()).run()


def gates(v, status):
    return sorted(g["id"] for g in v["gates"] if g["status"] == status)


def test_judge_pass(judge_mod, tmp_path):
    v = Acceptance(tmp_path, judge_mod).build().verdict()
    assert v["schema"] == "rcs_perf_verdict_v1"
    assert v["layers"] == {"PHYSICS": "PASS", "PERFORMANCE": "PASS", "INTEGRITY": "PASS"}, v["gates"]
    assert {"G1", "G4", "G6", "G9:n1", "G9:n2", "G9:n2:vram", "G9:n2:reset", "G9:n2:work"} <= set(gates(v, "PASS"))
    assert v["performance"]["ratios"]["2"]["median"] == pytest.approx(1.2)
    assert v["performance"]["ratios"]["2"]["pairs"] == pytest.approx([1.2, 1.2])
    assert v["failed_gates"] == [] and v["judge"]["version"] == PA.JUDGE_VERSION
    assert len(v["rows"]) == 8 and v["rows"][0]["clock"] == "outer (caller)"


def test_judge_physics_fail_on_final_codes(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.candidate_code_flip = 3
    v = acc.build().verdict()
    assert v["layers"]["PHYSICS"] == "FAIL" and v["layers"]["PERFORMANCE"] == "PASS"
    assert "G6" in v["failed_gates"]
    g6 = next(g for g in v["gates"] if g["id"] == "G6")
    assert "codes_equal_native" in g6["detail"]


def test_judge_performance_fail_when_slower(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.candidate_rate = 0.9
    v = acc.build().verdict()
    assert v["layers"] == {"PHYSICS": "PASS", "PERFORMANCE": "FAIL", "INTEGRITY": "PASS"}
    assert {"G9:n1", "G9:n2"} <= set(v["failed_gates"])


def test_judge_integrity_fail_on_clock_disagreement_fails_performance_too(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.candidate_inner_factor = 1.5          # the in-process clock claims 50 % more than the caller measured
    v = acc.build().verdict()
    assert v["layers"]["INTEGRITY"] == "FAIL" and v["layers"]["PERFORMANCE"] == "FAIL"
    assert v["layers"]["PHYSICS"] == "PASS"
    assert any(g.startswith("I-clock:perf_n") for g in v["failed_gates"])


def test_judge_integrity_fail_on_tampered_dump_fails_physics_too(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.tamper_candidate_dump = True
    v = acc.build().verdict()
    assert v["layers"]["INTEGRITY"] == "FAIL" and v["layers"]["PHYSICS"] == "FAIL"
    assert any(g.startswith("I-run:G6_replay_candidate") for g in v["failed_gates"])


def test_judge_candidate_harness_error_is_a_fail(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.candidate_error = True
    v = acc.build().verdict()
    assert v["layers"]["PHYSICS"] == "FAIL" and v["layers"]["PERFORMANCE"] == "FAIL"


def test_judge_blocked_when_outputs_are_missing(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.drop_baseline_results = True
    v = acc.build().verdict()
    assert v["layers"]["PHYSICS"] == "BLOCKED" and v["layers"]["PERFORMANCE"] == "BLOCKED"
    assert v["failed_gates"] == []


def test_judge_blocked_when_runs_are_absent_or_caller_blocked(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod).build()
    manifest = json.loads(acc.manifest_path.read_text())
    manifest["runs"] = [r for r in manifest["runs"] if r.get("gate") != "G6"]
    acc.manifest_path.write_text(json.dumps(manifest))
    v = acc.verdict()
    assert v["layers"]["PHYSICS"] == "BLOCKED" and "G6" in gates(v, "BLOCKED")
    manifest["blocked"] = "gpu_wait: timeout"
    acc.manifest_path.write_text(json.dumps(manifest))
    v = acc.verdict()
    assert v["layers"]["PHYSICS"] == "BLOCKED" and v["layers"]["PERFORMANCE"] == "BLOCKED"


def test_judge_cli_writes_verdict_and_never_overwrites(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod).build()
    out = tmp_path / "judge"
    # the real CLI uses the WP-17 references; with a missing wp17 tree the verdict is BLOCKED, never a FAIL
    rc = judge_mod.main(["--config", str(acc.cfg_path), "--manifest", str(acc.manifest_path), "--out", str(out)])
    v = json.loads((out / "verdict.json").read_text())
    assert rc == 0 and v["schema"] == "rcs_perf_verdict_v1"
    assert v["layers"]["PHYSICS"] in ("BLOCKED",) and "FAIL" not in v["layers"].values()
    assert judge_mod.main(["--config", str(acc.cfg_path), "--manifest", str(acc.manifest_path),
                           "--out", str(out)]) == 2


def test_judge_position_band_is_descriptive_by_default(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.candidate_position_offset = 1e-4      # final max diff ~1.7e-4 m > the P1-0 s512 high 3.9e-5 m
    acc.build()
    v = acc.verdict()
    assert v["layers"] == {"PHYSICS": "PASS", "PERFORMANCE": "PASS", "INTEGRITY": "PASS"}
    assert v["position_band"]["mode"] == "descriptive" and v["position_band"]["version"] == "p1_0_minmax_v1"
    assert len(v["warnings"]) == 2 and all(w.startswith("WARN position band") for w in v["warnings"])
    g6 = next(g for g in v["gates"] if g["id"] == "G6")
    assert "position_band" not in g6["threshold"]
    assert {x["values"]["position_band_within"] for x in g6["value"] if x["side"] == "candidate"} == {False}
    # the same outputs with the band switched to a gate (as band v2 will be): PHYSICS FAIL
    v = acc.verdict(band_mode="gate")
    assert v["layers"]["PHYSICS"] == "FAIL" and "position_band@" in next(
        g for g in v["gates"] if g["id"] == "G6")["detail"]


# B = 7 ms: strict limit 1.25 x 7 + 2 = 10.75 ms, shared limit 1.25 x 7 + 10 = 18.75 ms
@pytest.mark.parametrize("sharing, candidate_ms, status, warned", [
    ("EXCLUSIVE_GPU", 10.7, "PASS", False),
    ("EXCLUSIVE_GPU", 12.0, "FAIL", False),      # exclusive: strict rule
    ("SHARED_GPU", 10.7, "PASS", False),
    ("SHARED_GPU", 12.0, "PASS", True),          # shared: above strict, within +10 ms -> WARN, not a failure
    ("SHARED_GPU", 19.0, "FAIL", False),         # shared: beyond +10 ms -> FAIL
])
def test_judge_reset_p95_rule_exclusive_vs_shared(judge_mod, tmp_path, sharing, candidate_ms, status, warned):
    acc = Acceptance(tmp_path, judge_mod)
    acc.sharing, acc.candidate_reset_ms = sharing, candidate_ms
    v = acc.build().verdict()
    gates_ = {g["id"]: g for g in v["gates"]}
    for n in (1, 2):
        g = gates_[f"G9:n{n}:reset"]
        assert g["status"] == status
        assert g["value"]["regime"] == ("exclusive" if sharing == "EXCLUSIVE_GPU" else "shared")
        assert g["threshold"]["strict_max_ms"] == pytest.approx(10.75)
        assert ("shared_max_ms" in g["threshold"]) == (sharing != "EXCLUSIVE_GPU")
    warnings = [w for w in v["warnings"] if "shared_gpu_reset_bimodality" in w]
    assert len(warnings) == (2 if warned else 0)
    assert v["layers"]["PERFORMANCE"] == ("FAIL" if status == "FAIL" else "PASS")
    assert v["layers"]["PHYSICS"] == "PASS" and v["layers"]["INTEGRITY"] == "PASS"


def test_judge_mixed_sharing_uses_the_shared_rule(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    acc.candidate_reset_ms = 12.0
    acc.build()
    manifest = json.loads(acc.manifest_path.read_text())
    for r in manifest["runs"]:                # make every baseline row EXCLUSIVE, candidate rows stay SHARED
        if r.get("gate") == "G9" and r["side"] == "baseline":
            res = json.loads(Path(r["result"]).read_text())
            res["rows"][0]["gpu_sharing"] = "EXCLUSIVE_GPU"
            Path(r["result"]).write_text(json.dumps(res))
    v = acc.verdict()
    g = next(g for g in v["gates"] if g["id"] == "G9:n2:reset")
    assert g["value"]["regime"] == "shared" and g["status"] == "PASS" and g["value"]["warning"]


def test_judge_known_harness_only_for_rejudging_preserved_outputs(judge_mod, tmp_path):
    acc = Acceptance(tmp_path, judge_mod)
    old = dict(harness_sha256="a" * 64, shared_module_sha256="b" * 64)
    acc.shas = dict(old)                     # outputs of an earlier harness version
    acc.build()
    v = acc.verdict()
    assert v["layers"]["INTEGRITY"] == "FAIL" and v["layers"]["PHYSICS"] == "FAIL"
    cfg = PA.load_config(acc.cfg_path)
    v = judge_mod.Judge(cfg, acc.cfg_path, json.loads(acc.manifest_path.read_text()), acc.manifest_path,
                        natives=FakeNatives(), known_harness=[dict(old, note="earlier version")]).run()
    assert v["layers"] == {"PHYSICS": "PASS", "PERFORMANCE": "PASS", "INTEGRITY": "PASS"}
    assert any(n.startswith("re-judge:") for n in v["notes"])
