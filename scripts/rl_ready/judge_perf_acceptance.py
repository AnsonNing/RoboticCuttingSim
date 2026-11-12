#!/usr/bin/env python3
"""judge_perf_acceptance_v1 — RCS's own judge of a performance acceptance (P1-6 R0b, decision D-11).

Runs once per acceptance, CPU only, in its own process, from the frozen baseline (evaluator) checkout:

    CUDA_VISIBLE_DEVICES= PYTHONPATH=<evaluator>/src <python> scripts/rl_ready/judge_perf_acceptance.py \\
        --config configs/rl_benchmark/acceptance_v2.json --manifest MANIFEST.json --out DIR

Reads the manifest (``rcs_perf_accept_manifest_v1``), every run's ``result.json`` and dumps, the WP-17 Native
records and the caller's outer clock; writes only ``DIR/verdict.json`` (``rcs_perf_verdict_v1``). Physics is
recomputed from the dumps with this checkout's code (never the candidate's ``evaluate()``):

* episode plan entries (G6, G7), hard gates: final codes vs Native, labelled counts vs Native, sample invariants
  (finite positions, valid digits, no digit rewrites, invalid_reason 0), the frozen TRIM evaluator
  (``goals.evaluate_goal`` via the compiled task) on 2 frames and on the Native sample series, per-env contact
  totals vs the baseline side; candidate-reported L2 vs the recomputed invariants (disagreement = INTEGRITY).
  The final maximum position difference vs Native is compared with the swappable band file
  ``physics.position_band.path`` (high value per seed, or the reference variant's own value for that seed if
  larger): ``mode: descriptive`` (the P1-0 min-max band) only adds a WARN to ``verdict.warnings``; ``mode: gate``
  makes it a check;
* samestate plan entries (G4, G8): same recorded state, candidate vs baseline: non-zero impulse particle sets and
  stats[4] identical, impulse within the relative tolerance; the short replay's final codes and labelled counts
  equal, positions within tolerance, contact totals within tolerance;
* G1 / G2: the CPU suites (0 failed, no skip beyond the baseline's in the same environment);
* G9: per N the candidate / baseline outer-clock env-steps/s of each ABBA pair, VRAM peak, reset p95, the same work
  (final labelled counts) and the inner/outer clock agreement;
* ``G9:window`` (only when the config declares ``performance.window_check``, ``acceptance_v2``): the G9 timed window
  must hold the tool-material contact phase and the cutting. Measured on the episode dumps of the listed plan
  entries (G6 / G7: the same variants replaying the G9 seeds from reset, not the G9 processes themselves): in every
  such env the window has >= ``min_tool_contact_steps`` steps with non-zero tool-material contacts (stats slot 1,
  ``tool_contacts``) and >= ``min_new_labels`` new labelled particles (Native-sample codes). Particle-particle
  (cut-face) contacts (stats slot 4, ``contacts``) are reported per env, never gated. A dump without
  ``tool_contacts`` (a harness before acceptance_v2's) cannot pass;
* ``performance.phases`` (only with ``performance.phases``): per-phase median in-process step walls per side and
  their ratio from the G9 runs' ``step_walls_s`` (candidate-reported, descriptive, never a gate).

The config path is the required ``--config`` (``acceptance_v1.json`` and ``acceptance_v2.json`` are both judged by
this file; v1 has neither optional block and is judged exactly as before).

Layers PHYSICS / PERFORMANCE / INTEGRITY are PASS / FAIL / BLOCKED / NOT_RUN; an integrity problem also sets the layer
it invalidates (candidate side: FAIL; baseline side or caller: BLOCKED). Exit 0 whenever ``verdict.json`` was
written (a judge error writes a BLOCKED verdict); exit 2 for usage errors (nothing written).
"""
import argparse
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
import traceback

T0 = time.perf_counter()
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from robotic_cutting_sim.rl import perf_acceptance as PA  # noqa: E402

HARNESS = ROOT / "scripts" / "rl_ready" / "run_perf_acceptance.py"
JUDGE_THRESHOLD_KEYS = ("reset_p95_shared_slack_ms",)   # judged here; perf_acceptance.THRESHOLD_KEYS has the rest
OOM = re.compile(r"out of memory|CUDA_ERROR_OUT_OF_MEMORY|cudaErrorMemoryAllocation|MemoryError", re.IGNORECASE)
LIMITATIONS = [
    "VRAM peak and reset p95 are measured inside the process under test (candidate-reported, not independently "
    "verified); env-steps/s comes from the caller's outer clock and is cross-checked against the in-process value",
    "intermediate Native-sample positions are dumped as int8 micrometre deltas to the Native sample (|error| <= "
    "0.5 um, larger or non-finite deltas exact); final states are exact",
    "the evaluator reference arrays (mass, reference positions, stencils) are taken from the WP-17 Native recording",
    "samestate uses the baseline harness helpers (run_p1_3a.load_state / contact_stage) on the solver attributes "
    "x, v, code, impulse, stats_sum, stats_max, hash / particle_contact",
    "workload A is reduced fidelity (ideal edge tracking, no robot dynamics, reaction not fed back)",
]


def git(*args):
    try:
        out = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def opt(cmd, flag):
    """Value of ``flag`` in a command list (the harness command inside a possibly wrapped one)."""
    cmd = list(cmd or [])
    for i in range(len(cmd) - 1, -1, -1):
        if cmd[i] == flag and i + 1 < len(cmd):
            return cmd[i + 1]
    return None


def rel(a, b):
    return abs(float(a) - float(b)) / max(abs(float(b)), 1e-300)


# ----------------------------------------------------------------------------- Native references

class Natives:
    """WP-17 Native samples, compiled tasks and evaluator references per seed (loaded once, read-only)."""

    def __init__(self, wp17):
        self.root = Path(wp17)
        self.cache = {}

    def record(self, seed):
        dirs = sorted(d for d in self.root.iterdir() if (d / f"seed_{seed}").is_dir())
        if not dirs:
            raise FileNotFoundError(f"{self.root}: no WP-17 record seed_{seed}")
        preferred = [d for d in dirs if d.name == "native_optimized_001"]
        return (preferred or dirs)[0] / f"seed_{seed}" / "box" / "env" / "recording" / "attempt_000"

    def get(self, seed):
        seed = int(seed)
        if seed not in self.cache:
            from robotic_cutting_sim.rl.replay import native_samples
            from robotic_cutting_sim.rl.vector_env import TaskFamily, _compile
            data, _ = native_samples(self.record(seed))
            task = _compile(TaskFamily(scene_seeds=(seed,)), seed)
            ref = {k: np.asarray(data[k]) for k in ("mass_kg", "reference_positions", "reference_stencil_nodes",
                                                    "reference_stencil_weights", "reference_stencil_gradients")}
            steps = np.rint(np.asarray(data["time_s"], float) / PA.CONTROL_DT_S).astype(int)
            self.cache[seed] = dict(data=data, task=task, ref=ref, steps=steps, particles=len(ref["mass_kg"]))
        return self.cache[seed]


def evaluate(native, times, positions, codes):
    """Frozen seeded_trim goal evaluation (``task.evaluate_goal``) of a WORLD series starting at the reset frame."""
    ref = native["ref"]
    n = native["particles"]
    data = dict(ref, time_s=np.asarray([0.] + list(times), float),
                positions=np.stack([ref["reference_positions"]] + list(positions)),
                codes=np.stack([np.zeros(n, np.int32)] + [np.asarray(c, np.int32) for c in codes]),
                code_encoding="packed2", max_cuts=2, particle_spacing_m=native["task"].particle_spacing_m)
    result = native["task"].evaluate_goal(data)
    return dict(verdict=result.get("verdict"), integrity=result.get("integrity", {}).get("status"),
                outcome=result.get("outcome", {}).get("status"), frames=len(times) + 1)


# ----------------------------------------------------------------------------- the judge

class Judge:
    def __init__(self, cfg, cfg_path, manifest, manifest_path, natives=None, known_harness=None):
        """``natives``: an object with ``get(seed)`` (as :class:`Natives`); tests pass synthetic references."""
        missing = [k for k in JUDGE_THRESHOLD_KEYS if not isinstance(cfg["thresholds"].get(k), (int, float))]
        if missing:
            raise ValueError(f"config thresholds: needs numbers {missing}")
        self.cfg, self.cfg_path, self.man, self.manifest_path = cfg, Path(cfg_path), manifest, Path(manifest_path)
        self.known_harness = list(known_harness or [])
        self.gates, self.rows, self.notes, self.warnings = [], [], [], []
        self.episode_rows = []          # recomputed episode envs without dump problems (G9:window)
        self.phase_report = {}
        self.band, self.band_error = None, None
        try:   # the band file is a swappable input of this (evaluator) checkout
            self.band = PA.load_position_band(ROOT, cfg["physics"]["position_band"])
        except (OSError, ValueError, KeyError) as error:
            self.band_error = f"{type(error).__name__}: {error}"
        self.natives = natives
        self.harness_sha256 = PA.sha256_file(HARNESS)
        self.shared_sha256 = PA.sha256_file(Path(PA.__file__).resolve())

    # -- gate bookkeeping
    def gate(self, gid, layer, status, value=None, threshold=None, detail=None, **extra):
        g = dict(id=gid, layer=layer, status=status, value=value, threshold=threshold, detail=detail, **extra)
        self.gates.append(g)
        return g

    def integrity(self, gid, side, affects, detail, run=None):
        status = "FAIL" if side == "candidate" else "BLOCKED"
        return self.gate(gid, "INTEGRITY", status, detail=detail, affects=list(affects), side=side, run=run)

    def layers(self):
        out = {}
        for layer in ("PHYSICS", "PERFORMANCE"):
            words = [g["status"] for g in self.gates if g["layer"] == layer]
            words += [g["status"] for g in self.gates if g["layer"] == "INTEGRITY" and layer in g.get("affects", ())]
            out[layer] = PA.worst(words)
        out["INTEGRITY"] = PA.worst([g["status"] for g in self.gates if g["layer"] == "INTEGRITY"])
        return out

    # -- inputs
    def side_variant(self, side, key):
        if key == "default":
            return {}
        if key == "reference":
            return self.cfg["reference_variant"]
        return (self.man.get(side) or {}).get("variant")

    def load_run(self, run, expect):
        """Validated view of one manifest run. ``expect``: mode, num_envs, seeds, warmup_steps, steps, layer."""
        side = run.get("side")
        v = dict(run=run, id=run.get("id"), side=side, ok=False, kind=None, result=None, dir=None, problems=[])
        rp = run.get("result")
        if not rp:
            out = opt(run.get("cmd") or run.get("harness_cmd"), "--out")
            err = None
            if out and (Path(out) / "error.json").is_file():
                try:
                    err = read_json(Path(out) / "error.json")
                except (OSError, ValueError):
                    err = None
            text = json.dumps(err or {}) + " " + str(run.get("infra") or "")
            v["kind"] = "oom" if OOM.search(text) else "no_result"
            v["detail"] = (str((err or {}).get("error_code")) + ": " + str((err or {}).get("error") or "")[-300:]
                           if err else str(run.get("infra") or f"no result.json (rc {run.get('rc')})"))
            return v
        try:
            res = read_json(rp)
        except (OSError, ValueError) as error:
            v["kind"] = "no_result"
            v["detail"] = f"result.json unreadable: {error}"
            return v
        v["result"], v["dir"] = res, Path(rp).parent
        p = v["problems"]
        if res.get("schema") != PA.SCHEMA_RESULT or res.get("harness") != PA.HARNESS_VERSION:
            p.append(f"result schema/harness {res.get('schema')}/{res.get('harness')}")
        env = res.get("environment") or {}
        sides = self.man.get(side) or {}
        if side not in ("baseline", "candidate"):
            p.append(f"side {side!r}")
        if env.get("commit") != run.get("commit") or run.get("commit") != sides.get("commit"):
            p.append(f"commit {str(env.get('commit'))[:12]} / run {str(run.get('commit'))[:12]} / "
                     f"{side} {str(sides.get('commit'))[:12]}")
        checkout = sides.get("checkout")
        root = str(env.get("package_root") or "")
        if not checkout or not os.path.realpath(root).startswith(os.path.realpath(str(checkout)) + os.sep):
            p.append(f"package_root {root!r} is not inside the {side} checkout {checkout!r}")
        pair = (env.get("harness_sha256"), env.get("shared_module_sha256"))
        if pair != (self.harness_sha256, self.shared_sha256):
            known = next((k for k in self.known_harness
                          if (k.get("harness_sha256"), k.get("shared_module_sha256")) == pair), None)
            if known is None:
                p.append("harness or perf_acceptance module differs from the evaluator checkout's")
            else:
                self.notes.append(f"re-judge: run {run.get('id')} harness bytes accepted from --known-harness "
                                  f"({known.get('note')})")
        for key in ("mode", "num_envs", "warmup_steps", "steps"):
            if res.get(key) != expect[key]:
                p.append(f"{key} {res.get(key)!r} != {expect[key]!r}")
        if [int(s) for s in res.get("seeds") or []] != [int(s) for s in expect["seeds"]]:
            p.append(f"seeds {res.get('seeds')} != {expect['seeds']}")
        want = self.side_variant(side, run.get("variant"))
        if want is None or PA.canonical_json(res.get("variant")) != PA.canonical_json(want):
            p.append(f"variant {res.get('variant')} != expected {want} ({run.get('variant')})")
        for d in res.get("dumps") or []:
            path = str(d.get("path") or "")
            if path.startswith("/") or ".." in Path(path).parts:
                p.append(f"dump path {path!r}")
                continue
            f = v["dir"] / path
            if not f.is_file() or PA.sha256_file(f) != d.get("sha256"):
                p.append(f"dump {path} missing or sha256 differs")
        for problem in run.get("environment_problems") or []:
            p.append(f"caller: {problem}")
        if run.get("gpu_guard") not in (None, "PASS"):
            p.append(f"gpu_guard {run.get('gpu_guard')}")
        if res.get("status") != "COMPLETE":
            v["kind"] = "harness_error"
            v["detail"] = f"{res.get('error_code')}: {str(res.get('error') or '')[-300:]}"
        if p:
            self.integrity(f"I-run:{v['id']}", side, [expect["layer"]], "; ".join(p)[:600], run=v["id"])
            v["kind"] = v["kind"] or "integrity"
        elif v["kind"] is None:
            v["ok"] = True
        return v

    def run_failure_status(self, views):
        """Gate status from runs that did not produce valid outputs; None when every run is usable."""
        bad = [v for v in views if not v["ok"]]
        if not bad:
            return None, None
        base_ok = all(v["ok"] for v in views if v["side"] == "baseline")
        cand_bad = [v for v in bad if v["side"] == "candidate"]
        detail = "; ".join(f"{v['id']}: {v['kind']} {str(v.get('detail') or '')[:160]}" for v in bad)[:800]
        if base_ok and cand_bad and all(v["kind"] in ("harness_error", "integrity", "no_result") for v in cand_bad):
            return "FAIL", detail
        return "BLOCKED", detail

    # -- G1 / G2
    def cpu_gates(self):
        suites = self.man.get("cpu_tests") or {}
        for gid, kind in (("G1", "cpu_tests"), ("G2", "torch_tests")):
            if not self.cfg.get(kind):
                continue
            b = (suites.get("baseline") or {}).get(kind)
            c = (suites.get("candidate") or {}).get(kind)
            if not b or not c or b.get("error") or c.get("error") or not c.get("tests"):
                self.gate(gid, "PHYSICS", "BLOCKED", value={"baseline": b, "candidate": c},
                          threshold="0 failed, no new skips", detail="suite results missing")
                continue
            new_skips = sorted(set(c.get("skipped_ids") or []) - set(b.get("skipped_ids") or []))
            c_fail = set(c.get("failed_ids") or [])
            failed = int(c.get("failed") or 0) + int(c.get("errors") or 0)
            if failed == 0 and not new_skips:
                status = "PASS"
            elif not new_skips and c_fail and c_fail <= set(b.get("failed_ids") or []):
                status = "BLOCKED"   # the baseline fails the same tests in this environment
            else:
                status = "FAIL"
            self.gate(gid, "PHYSICS", status, value=dict(tests=c.get("tests"), failed=failed,
                                                         skipped=c.get("skipped"), new_skips=new_skips[:20],
                                                         failed_ids=sorted(c_fail)[:20]),
                      threshold="0 failed; skips within the baseline's", detail=f"baseline failed "
                      f"{int(b.get('failed') or 0) + int(b.get('errors') or 0)}, skipped {b.get('skipped')}")

    # -- physics plan entries
    def plan_gates(self):
        plan = self.cfg["plan"]
        gate_counts = {}
        for e in plan:
            gate_counts[e["gate"]] = gate_counts.get(e["gate"], 0) + 1
        for e in plan:
            gid = e["gate"] if gate_counts[e["gate"]] == 1 else f"{e['gate']}:{e['id']}"
            layer = e.get("layer") or ("PERFORMANCE" if e["gate"] == "G9" else "PHYSICS")
            expect = dict(mode=e["mode"], num_envs=int(e["num_envs"]), seeds=e["seeds"],
                          warmup_steps=int(e["warmup_steps"]), steps=int(e["steps"]), layer=layer)
            runs = [r for r in self.man.get("runs") or [] if r.get("plan_id") == e["id"]]
            sides = e.get("sides") or ["baseline", "candidate"]
            want = int(e.get("repeats", 1)) * len(sides)
            views = [self.load_run(r, expect) for r in runs]
            if len(runs) != want or {v["side"] for v in views} != set(sides):
                self.gate(gid, layer, "BLOCKED", value=len(runs), threshold=want,
                          detail=f"{e['id']}: {len(runs)} runs in the manifest, expected {want} ({sides})")
                continue
            status, detail = self.run_failure_status(views)
            if status:
                self.gate(gid, layer, status, detail=f"{e['id']}: {detail}", plan_id=e["id"])
                continue
            try:
                if e["mode"] == "episode":
                    self.episode_gate(gid, layer, e, views)
                elif e["mode"] == "samestate":
                    self.samestate_gate(gid, layer, e, views)
                else:
                    self.work_gate(gid, layer, e, views)
            except Exception:   # an unexpected judge error is never a verdict on the candidate
                self.gate(gid, layer, "BLOCKED", detail=f"{e['id']}: judge error {traceback.format_exc()[-600:]}",
                          plan_id=e["id"])

    def native(self, seed):
        if self.natives is None:
            self.natives = Natives(self.man["wp17"])
        return self.natives.get(seed)

    def episode_env(self, v, d):
        """Recomputed physics of one episode dump (one env of one run)."""
        ph = self.cfg["physics"]
        with np.load(v["dir"] / d["path"], allow_pickle=False) as z:
            a = {k: z[k] for k in z.files}
        seed = int(a["seed"])
        nat = self.native(seed)
        P = nat["particles"]
        out = dict(run=v["id"], side=v["side"], seed=seed, env=int(a["env"]), checks={}, values={}, problems=[])
        si, ss = a["sample_index"].astype(int), a["sample_step"].astype(int)
        steps = int(a["steps"])
        S = len(si)
        shapes_ok = (a["final_positions"].shape == (P, 3) and a["final_codes"].shape == (P,)
                     and a["sample_codes"].shape == (S, P) and a["sample_dq"].shape == (S, P, 3)
                     and a["sample_dq"].dtype == np.int8 and len(ss) == S)
        if not shapes_ok or (S and (si.min() < 0 or si.max() >= len(nat["steps"]))) or \
                not np.array_equal(ss, nat["steps"][si] if S else ss) or np.any(np.diff(ss) <= 0) or \
                int(a["final_step_count"]) != steps or steps != v["result"]["warmup_steps"] + v["result"]["steps"]:
            out["problems"].append("dump shapes, sample steps or step count inconsistent")
            return out
        expected_samples = int(np.count_nonzero((nat["steps"] >= 1) & (nat["steps"] <= steps)))
        if S != expected_samples:
            out["problems"].append(f"{S} samples, expected {expected_samples}")
            return out
        pos = PA.decode_positions(np.asarray(nat["data"]["positions"])[si], a["sample_dq"], a["sample_escape_index"],
                                  a["sample_escape_delta"]) if S else np.zeros((0, P, 3))
        final_pos, final_codes = a["final_positions"].astype(np.float64), a["final_codes"].astype(np.int64)
        codes = a["sample_codes"].astype(np.int64)
        # invariants on the sampled frames and the final state
        chain = [np.zeros(P, np.int64)] + list(codes) + [final_codes]
        rewrites = sum(PA.digit_rewrites(chain[i], chain[i + 1]) for i in range(len(chain) - 1))
        digits_ok = all(int((((c >> (2 * k)) & 3)).max(initial=0)) <= 2 for c in chain for k in range(2)) and \
            all(int(c.min(initial=0)) >= 0 for c in chain)
        finite = bool(np.isfinite(pos).all() and np.isfinite(final_pos).all()
                      and np.isfinite(a["final_velocities"]).all())
        last_sample_eq_final = bool(not S or ss[-1] != steps or np.array_equal(codes[-1], final_codes))
        out["values"].update(digit_rewrites=rewrites, finite=finite, invalid_reason=int(a["invalid_reason"]),
                             samples=S, steps=steps)
        out["checks"]["invariants"] = bool(rewrites == 0 and digits_ok and finite and int(a["invalid_reason"]) == 0
                                           and last_sample_eq_final)
        # Native comparison at the final step (when it is a Native sample step)
        hit = np.flatnonzero(nat["steps"] == steps)
        if hit.size:
            k = int(hit[0])
            nat_codes = np.asarray(nat["data"]["codes"][k]).astype(np.int64)
            from robotic_cutting_sim.rl.replay import digit_counts
            dp = np.linalg.norm(final_pos - np.asarray(nat["data"]["positions"][k], np.float64), axis=1)
            out["values"].update(different_code_particles=int(np.count_nonzero(final_codes != nat_codes)),
                                 labelled=digit_counts(final_codes, 2), labelled_native=digit_counts(nat_codes, 2),
                                 final_max_position_diff_m=float(dp.max()) if finite else None,
                                 max_position_diff_m_over_samples=float(np.max(np.linalg.norm(
                                     pos - np.asarray(nat["data"]["positions"])[si], axis=2))) if finite and S
                                 else None)
            out["checks"]["codes_equal_native"] = out["values"]["different_code_particles"] == 0
            out["checks"]["labelled_equal_native"] = out["values"]["labelled"] == out["values"]["labelled_native"]
            if steps == PA.EPISODE_STEPS and self.band and str(seed) in self.band["high_m"]:
                out["values"]["position_band_high_m"] = self.band["high_m"][str(seed)]
        # frozen TRIM evaluator: 2 frames (reset + final) and the Native sample series (+ final)
        if finite:
            t_final = steps * PA.CONTROL_DT_S
            ev2 = evaluate(nat, [t_final], [final_pos], [final_codes])
            keep = ss < steps
            ev_series = evaluate(nat, list(ss[keep] * PA.CONTROL_DT_S) + [t_final], list(pos[keep]) + [final_pos],
                                 list(codes[keep]) + [final_codes])
            out["values"].update(evaluator_2=ev2, evaluator_series=ev_series)
            if steps == PA.EPISODE_STEPS:
                out["checks"]["evaluator_2_frames"] = ev2["verdict"] == "PASS"
                out["checks"]["evaluator_series"] = ev_series["verdict"] == "PASS"
        else:
            out["checks"]["evaluator_2_frames"] = False
            out["checks"]["evaluator_series"] = False
        out["values"]["contacts_total"] = int(np.sum(a["contacts"]))
        out["series"] = dict(contacts=np.asarray(a["contacts"], np.int64).reshape(-1),
                             tool_contacts=(np.asarray(a["tool_contacts"], np.int64).reshape(-1)
                                            if "tool_contacts" in a else None),
                             labelled_at_step={int(st): int(np.count_nonzero(codes[i])) for i, st in enumerate(ss)})
        # candidate-reported L2 vs the recomputation
        rnd = next((r for r in v["result"].get("rounds") or [] if r.get("round") == d.get("round")), {})
        l2 = rnd.get("l2_candidate_reported") or {}
        e = out["env"]
        try:
            reported_clean = (l2.get("code_monotone_violations", [None])[e] == 0
                              and l2.get("nonfinite_steps", [None])[e] == 0
                              and (rnd.get("invalid_reason") or [None])[e] == int(a["invalid_reason"]))
        except (IndexError, TypeError):
            reported_clean = None
        out["values"]["l2_reported_clean"] = reported_clean
        if reported_clean is None or (reported_clean and not (rewrites == 0 and finite)):
            out["problems"].append(f"candidate-reported L2 {reported_clean} disagrees with the dumps "
                                   f"(rewrites {rewrites}, finite {finite})")
        return out

    def episode_gate(self, gid, layer, e, views):
        ph = self.cfg["physics"]
        envs = []
        for v in views:
            dumps = [d for d in v["result"].get("dumps") or [] if d.get("kind") == "episode"]
            expect = sum(len(r) for r in PA.plan_rounds("episode", e["seeds"], int(e["num_envs"])))
            if len(dumps) != expect:
                self.integrity(f"I-dumps:{v['id']}", v["side"], [layer], f"{len(dumps)} episode dumps, expected "
                               f"{expect}", run=v["id"])
                self.gate(gid, layer, "FAIL" if v["side"] == "candidate" else "BLOCKED", plan_id=e["id"],
                          detail=f"{v['id']}: dumps missing")
                return
            for d in dumps:
                row = self.episode_env(v, d)
                if row["problems"]:
                    self.integrity(f"I-dumps:{v['id']}:e{row['env']}", v["side"], [layer],
                                   "; ".join(row["problems"]), run=v["id"])
                envs.append(row)
        self.episode_rows += [dict(r, plan_id=e["id"]) for r in envs if not r["problems"]]
        if any(r["problems"] for r in envs):
            bad_c = any(r["problems"] for r in envs if r["side"] == "candidate")
            self.gate(gid, layer, "FAIL" if bad_c else "BLOCKED", plan_id=e["id"],
                      detail="dumps inconsistent (see INTEGRITY gates)")
            return
        # relative rules need the baseline of the same seed
        base = {}
        for r in envs:
            if r["side"] == "baseline":
                base.setdefault(r["seed"], []).append(r)
        for r in envs:
            b = base.get(r["seed"]) or []
            high = r["values"].get("position_band_high_m")
            fmax = r["values"].get("final_max_position_diff_m")
            if high is not None:
                b_max = max([x["values"].get("final_max_position_diff_m") or 0. for x in b] or [0.])
                limit = max(float(high), b_max)
                within = fmax is not None and fmax <= limit
                r["values"]["position_limit_m"] = limit
                r["values"]["position_band_within"] = within
                if self.band["mode"] == "gate":
                    r["checks"]["position_band"] = within
                elif not within:
                    self.warnings.append(
                        f"WARN position band ({self.band['version']}, descriptive, not a gate): {gid} {r['run']} "
                        f"s{r['seed']} e{r['env']} final max position diff {fmax} m > {limit} m")
            if b:
                totals = [x["values"]["contacts_total"] for x in b]
                med = float(np.median(totals))
                spread = (max(totals) - min(totals)) / med if med else 0.
                tol = max(float(ph["contact_total_rel_tol"]), 2 * spread)
                r["values"]["contacts_rel_diff_vs_baseline"] = rel(r["values"]["contacts_total"], med) if med else 0.
                r["values"]["contacts_rel_tol"] = tol
                r["checks"]["contacts_vs_baseline"] = r["values"]["contacts_rel_diff_vs_baseline"] <= tol
        self.combine(gid, layer, e, envs)

    def combine(self, gid, layer, e, rows):
        """Gate from per-row checks: a candidate failure is FAIL unless the baseline fails the same check for the
        same seed (then BLOCKED: the reference itself does not meet it in this environment)."""
        names = sorted({k for r in rows for k in r["checks"]})
        failed, blocked = [], []
        for name in names:
            c_bad = [r for r in rows if r["side"] == "candidate" and r["checks"].get(name) is False]
            b_bad_seeds = {r["seed"] for r in rows if r["side"] == "baseline" and r["checks"].get(name) is False}
            for r in c_bad:
                (blocked if r["seed"] in b_bad_seeds else failed).append(f"{name}@{r['run']}:s{r['seed']}e{r['env']}")
            if not c_bad and b_bad_seeds:
                self.notes.append(f"{gid}: baseline fails {name} for seeds {sorted(b_bad_seeds)} (candidate passes)")
        status = "FAIL" if failed else "BLOCKED" if blocked else "PASS"
        value = [dict(run=r["run"], side=r["side"], seed=r["seed"], env=r["env"], checks=r["checks"],
                      values={k: v for k, v in r["values"].items() if k not in ("evaluator_2", "evaluator_series")
                              } | {k: (r["values"].get(k) or {}).get("verdict")
                                   for k in ("evaluator_2", "evaluator_series") if k in r["values"]})
                 for r in rows]
        self.gate(gid, layer, status, value=value, threshold=sorted(names), plan_id=e["id"],
                  detail="; ".join(failed or blocked)[:600] or f"{len(rows)} env runs, checks {names}")

    def samestate_gate(self, gid, layer, e, views):
        ph = self.cfg["physics"]
        tol_imp, tol_pos = float(ph["impulse_rel_tol"]), float(ph["replay_position_tol_m"])
        tol_contacts = float(ph["contact_total_rel_tol"])
        loaded = {}
        for v in views:
            for d in v["result"].get("dumps") or []:
                if d.get("kind") == "samestate":
                    with np.load(v["dir"] / d["path"], allow_pickle=False) as z:
                        loaded[(v["side"], int(v["run"].get("repeat") or 0), int(d["seed"]))] = (
                            v, d, {k: z[k] for k in z.files})
        rows = []
        for (side, rep, seed), (v, d, a) in sorted(loaded.items()):
            if side != "candidate":
                continue
            ref = loaded.get(("baseline", rep, seed)) or next(
                (x for k, x in sorted(loaded.items()) if k[0] == "baseline" and k[2] == seed), None)
            if ref is None or d.get("samples") != ref[1].get("samples"):
                self.integrity(f"I-dumps:{v['id']}:s{seed}", "baseline", [layer], "no baseline samestate dump "
                               "with the same samples", run=v["id"])
                rows.append(dict(run=v["id"], side=side, seed=seed, env=0, checks={"baseline_present": None},
                                 values={}))
                continue
            b = ref[2]
            checks, values = {}, {"samples": {}}
            for k in d.get("samples") or []:
                key = f"k{k}"
                try:
                    ca, ba, cb = (a[f"impulse_a_{key}"], b[f"impulse_a_{key}"], a[f"impulse_b_{key}"])
                    cs, bs = a[f"stats_sum_a_{key}"], b[f"stats_sum_a_{key}"]
                except KeyError:
                    checks["dump_complete"] = False
                    continue
                if ca.shape != ba.shape or cs.shape != bs.shape:
                    checks["dump_complete"] = False
                    continue
                nz_c, nz_b, nz_cb = (np.any(x != 0, axis=1) for x in (ca, ba, cb))
                scale = float(np.abs(ba.astype(np.float64)).max())
                diff = float(np.abs(ca.astype(np.float64) - ba.astype(np.float64)).max()) if ca.size else 0.
                row = dict(nonzero=[int(nz_b.sum()), int(nz_c.sum())], nonzero_sets_identical=bool(
                    np.array_equal(nz_c, nz_b) and np.array_equal(nz_c, nz_cb)),
                    stats4_identical=bool(np.array_equal(cs[:, 4], bs[:, 4])),
                    impulse_max_rel_diff=diff / scale if scale else (0. if diff == 0 else float("inf")))
                values["samples"][str(k)] = row
                checks[f"nonzero_sets_{key}"] = row["nonzero_sets_identical"]
                checks[f"stats4_{key}"] = row["stats4_identical"]
                checks[f"impulse_{key}"] = row["impulse_max_rel_diff"] <= tol_imp
            n = len(d.get("env_seeds") or [])
            for i in range(n):
                fc, fb = a.get(f"final_codes_e{i}"), b.get(f"final_codes_e{i}")
                pc, pb = a.get(f"final_positions_e{i}"), b.get(f"final_positions_e{i}")
                if fc is None or fb is None or pc is None or pb is None or fc.shape != fb.shape or \
                        pc.shape != pb.shape:
                    checks["dump_complete"] = False
                    continue
                finite = bool(np.isfinite(pc).all())
                dp = float(np.linalg.norm(pc - pb, axis=1).max()) if finite else float("inf")
                values[f"replay_e{i}"] = dict(different_codes=int(np.count_nonzero(fc != fb)), max_position_diff_m=dp)
                checks[f"replay_codes_e{i}"] = bool(np.array_equal(fc, fb))
                checks[f"replay_positions_e{i}"] = dp <= tol_pos
            if "contacts" in a and "contacts" in b and a["contacts"].shape == b["contacts"].shape:
                tc, tb = int(a["contacts"].sum()), int(b["contacts"].sum())
                values["contacts_total"] = [tb, tc]
                checks["replay_contacts"] = (rel(tc, tb) if tb else float(tc != 0)) <= tol_contacts
            else:
                checks["dump_complete"] = False
            rows.append(dict(run=v["id"], side=side, seed=seed, env=0, checks=checks, values=values))
        if not rows:
            self.gate(gid, layer, "BLOCKED", plan_id=e["id"], detail="no candidate samestate dumps")
            return
        self.combine(gid, layer, e, rows)

    def work_gate(self, gid, layer, e, views):
        """Throughput / profile plan entries: completed, same final labelled counts as the baseline side."""
        base = next(v for v in views if v["side"] == "baseline")
        want = [r.get("final_labelled") for r in base["result"].get("rounds") or []]
        bad = [v["id"] for v in views if [r.get("final_labelled") for r in v["result"].get("rounds") or []] != want]
        self.gate(gid, layer, "FAIL" if bad else "PASS", value=bad, threshold="final labelled counts = baseline",
                  plan_id=e["id"])

    # -- G9
    def performance_gates(self):
        perf, th = self.cfg["performance"], self.cfg["thresholds"]
        tol = float(th["clock_tolerance"])
        if abs(float(self.man.get("clock_tolerance", tol)) - tol) > 1e-12:
            self.integrity("I-clock-tolerance", "caller", ["PERFORMANCE"],
                           f"manifest clock_tolerance {self.man.get('clock_tolerance')} != config {tol}")
        ratios = {}
        for n in perf["num_envs"]:
            n = int(n)
            steps = int((perf.get("steps_by_num_envs") or {}).get(str(n), perf["steps"]))
            expect = dict(mode="throughput", num_envs=n, seeds=perf["seeds"], warmup_steps=int(perf["warmup_steps"]),
                          steps=steps, layer="PERFORMANCE")
            runs = sorted((r for r in self.man.get("runs") or [] if r.get("gate") == "G9" and
                           int(r.get("num_envs") or 0) == n), key=lambda r: int(r.get("order") or 0))
            gid = f"G9:n{n}"
            order = "".join("B" if r.get("side") == "baseline" else "C" for r in runs)
            if order != perf["order"]:
                self.gate(gid, "PERFORMANCE", "BLOCKED", value=order, threshold=perf["order"],
                          detail="ABBA runs missing or out of order")
                continue
            views = [self.load_run(r, expect) for r in runs]
            status, detail = self.run_failure_status(views)
            if status:
                self.gate(gid, "PERFORMANCE", status, detail=detail)
                continue
            # outer clock (the caller's) and its agreement with the in-process clock
            rates, clock_bad = {}, []
            for v in views:
                o = v["run"].get("outer") or {}
                inner = next((x.get("env_steps_per_s") for x in v["result"].get("rows") or []
                              if x.get("num_envs") == n), None)
                ok_outer = (o.get("num_envs") == n and o.get("vector_steps") == steps and o.get("env_steps") == steps * n
                            and float(o.get("seconds") or 0) > 0 and o.get("env_steps_per_s") and
                            rel(o["env_steps"] / o["seconds"], o["env_steps_per_s"]) < 1e-6)
                if not ok_outer or not inner:
                    clock_bad.append((v, f"outer {o or None}, inner {inner}"))
                    continue
                ratio = float(inner) / float(o["env_steps_per_s"])
                if abs(ratio - 1) > tol:
                    clock_bad.append((v, f"inner/outer {ratio:.4f} outside +-{tol}"))
                rates[v["id"]] = float(o["env_steps_per_s"])
                self.rows.append(self.row(v, o))
            for v, why in clock_bad:
                self.integrity(f"I-clock:{v['id']}", v["side"], ["PERFORMANCE"], why, run=v["id"])
            if clock_bad:
                self.gate(gid, "PERFORMANCE", "FAIL" if any(v["side"] == "candidate" for v, _ in clock_bad)
                          else "BLOCKED", detail="clock check failed (see INTEGRITY)")
                continue
            gpus = {json.dumps([(v["result"].get("gpu") or {}).get(k) for k in ("gpu_uuid", "uuid", "name")])
                    for v in views}
            if len(gpus) != 1:
                self.integrity(f"I-gpu:n{n}", "caller", ["PERFORMANCE"], f"runs on different GPUs {sorted(gpus)}")
            pairs = []
            for i in range(int(perf["pairs"])):
                pv = [v for v in views if int(v["run"].get("pair", -1)) == i]
                bv = [v for v in pv if v["side"] == "baseline"]
                cv = [v for v in pv if v["side"] == "candidate"]
                if len(bv) == 1 and len(cv) == 1:
                    pairs.append(rates[cv[0]["id"]] / rates[bv[0]["id"]])
            if len(pairs) != int(perf["pairs"]):
                self.gate(gid, "PERFORMANCE", "BLOCKED", value=pairs, detail="incomplete pairs")
                continue
            med = statistics.median(pairs)
            above = sum(1 for x in pairs if x > 1.0)
            ratios[str(n)] = dict(median=med, pairs=pairs, above_1=above, mean=statistics.fmean(pairs))
            primary = n == int(perf["primary_num_envs"])
            need = float(th["primary_median_min"] if primary else th["all_median_min"])
            ok = med >= need and (not primary or above >= int(th["primary_pairs_above_1"]))
            self.gate(gid, "PERFORMANCE", "PASS" if ok else "FAIL", value=dict(median=med, pairs=pairs, above_1=above),
                      threshold=dict(median_min=need, **({"pairs_above_1_min": th["primary_pairs_above_1"]}
                                                          if primary else {})),
                      detail="candidate / baseline outer-clock env-steps/s per ABBA pair")
            # VRAM, reset p95, same work
            row = {v["id"]: next((x for x in v["result"].get("rows") or [] if x.get("num_envs") == n), {})
                   for v in views}
            side_rows = {s: [row[v["id"]] for v in views if v["side"] == s] for s in ("baseline", "candidate")}
            sources = {x.get("vram_source") for rs in side_rows.values() for x in rs}
            vb = [x.get("vram_peak_mib") for x in side_rows["baseline"]]
            vc = [x.get("vram_peak_mib") for x in side_rows["candidate"]]
            if len(sources) != 1 or None in vb + vc:
                self.gate(f"G9:n{n}:vram", "PERFORMANCE", "BLOCKED", value=dict(baseline=vb, candidate=vc),
                          detail=f"VRAM sources {sorted(map(str, sources))}")
            else:
                limit = max(vb) * float(th["vram_ratio_max"]) + float(th["vram_slack_mib"])
                self.gate(f"G9:n{n}:vram", "PERFORMANCE", "PASS" if max(vc) <= limit else "FAIL",
                          value=dict(baseline_max=max(vb), candidate_max=max(vc), source=sources.pop()),
                          threshold=dict(max_mib=limit, rule="C <= 1.10 x B + 256 MiB (config)"))
            rb = [x.get("reset_p95_ms") for x in side_rows["baseline"]]
            rc = [x.get("reset_p95_ms") for x in side_rows["candidate"]]
            if None in rb + rc:
                self.gate(f"G9:n{n}:reset", "PERFORMANCE", "BLOCKED", value=dict(baseline=rb, candidate=rc))
            else:
                self.reset_gate(n, rb, rc, {s: [x.get("gpu_sharing") for x in side_rows[s]] for s in side_rows})
            base_v = next(v for v in views if v["side"] == "baseline")
            want = [r.get("final_labelled") for r in base_v["result"].get("rounds") or []]
            differ = [v["id"] for v in views if [r.get("final_labelled") for r in v["result"].get("rounds") or []]
                      != want or any(c != expect["warmup_steps"] + steps for r in v["result"].get("rounds") or []
                                     for c in r.get("final_step_count") or [])]
            self.gate(f"G9:n{n}:work", "PERFORMANCE", "FAIL" if differ else "PASS", value=differ,
                threshold="every run: W+K steps, final labelled counts = the first baseline run's")
            if perf.get("phases"):
                self.phase_report[str(n)] = self.phases(views, perf["phases"], expect["warmup_steps"] + steps)
        return ratios

    def phases(self, views, phases, steps):
        """Descriptive per-phase speed of the G9 runs at one N from the harness's in-process ``step_walls_s``
        (candidate-reported): per run the median step wall of each phase, per side the median over runs, and
        baseline / candidate (> 1: the candidate is faster in that phase). Never a gate."""
        out = {}
        for name, (first, last) in phases.items():
            if last > steps:
                out[name] = dict(steps=[first, last], status="NOT_RUN", detail=f"phase beyond the {steps} run steps")
                continue
            per = {"baseline": [], "candidate": []}
            for v in views:
                walls = [r.get("step_walls_s") for r in v["result"].get("rounds") or []]
                if len(walls) != 1 or not isinstance(walls[0], list) or len(walls[0]) != steps:
                    continue
                per[v["side"]].append(float(np.median(walls[0][first - 1:last])))
            if not per["baseline"] or not per["candidate"]:
                out[name] = dict(steps=[first, last], status="NOT_RUN",
                                 detail="no step_walls_s in the results (harness before acceptance_v2)")
                continue
            b, c = statistics.median(per["baseline"]), statistics.median(per["candidate"])
            out[name] = dict(steps=[first, last], baseline_median_step_s=b, candidate_median_step_s=c,
                             speed_ratio_c_over_b=b / c if c > 0 else None, runs=per, status="REPORTED")
        return out

    def window_gate(self):
        """``G9:window`` (config ``performance.window_check``): the G9 timed window holds the tool-contact phase and the
        cutting in every episode env of the listed plans for the G9 seeds (see the module docstring)."""
        perf = self.cfg["performance"]
        check = perf.get("window_check")
        if not check:
            return
        windows = sorted(set(PA.g9_windows(perf).values()))
        seeds = {int(s) for s in perf["seeds"]}
        rows = [r for r in self.episode_rows if r["plan_id"] in check["plans"] and r["seed"] in seeds]
        threshold = dict(windows=[list(w) for w in windows], min_tool_contact_steps=check["min_tool_contact_steps"],
                         min_new_labels=check["min_new_labels"], plans=check["plans"], seeds=sorted(seeds),
                         particle_contacts="reported, not gated")
        if not any(r["side"] == "candidate" for r in rows) or not any(r["side"] == "baseline" for r in rows):
            self.gate("G9:window", "PERFORMANCE", "BLOCKED", threshold=threshold,
                      detail="no usable episode dumps of the window_check plans on both sides")
            return
        checked = []
        for r in rows:
            contacts, labels = r["series"]["contacts"], r["series"]["labelled_at_step"]
            tool = r["series"].get("tool_contacts")
            vals, ok = [], True
            for first, last in windows:
                tool_steps = int(np.count_nonzero(tool[first - 1:last])) if tool is not None and len(tool) >= last \
                    else None
                particle_steps = int(np.count_nonzero(contacts[first - 1:last])) if len(contacts) >= last else None
                before = [s for s in labels if s <= first - 1]
                after = [s for s in labels if s <= last]
                new = (labels[max(after)] - (labels[max(before)] if before else 0)) if after else None
                vals.append(dict(window=[first, last], tool_contact_steps=tool_steps,
                                 tool_contact_steps_episode=None if tool is None else int(np.count_nonzero(tool)),
                                 new_labels=new, label_steps=[max(before) if before else 0,
                                                              max(after) if after else None],
                                 particle_contact_steps_reported=particle_steps,
                                 particle_contact_steps_episode_reported=int(np.count_nonzero(contacts))))
                ok = ok and tool_steps is not None and new is not None \
                    and tool_steps >= check["min_tool_contact_steps"] and new >= check["min_new_labels"]
            checked.append(dict(run=r["run"], side=r["side"], seed=r["seed"], env=r["env"], plan_id=r["plan_id"],
                                checks={"window_contact_and_cutting": ok}, values={"windows": vals}))
        c_bad = [x for x in checked if x["side"] == "candidate" and not x["checks"]["window_contact_and_cutting"]]
        b_bad = [x for x in checked if x["side"] == "baseline" and not x["checks"]["window_contact_and_cutting"]]
        status = "BLOCKED" if b_bad else "FAIL" if c_bad else "PASS"
        bad = b_bad or c_bad
        self.gate("G9:window", "PERFORMANCE", status, value=checked, threshold=threshold,
                  detail="; ".join(f"{x['run']}:s{x['seed']}e{x['env']} {x['values']['windows']}" for x in bad)[:600]
                  or f"{len(checked)} episode envs: the G9 window holds the tool-contact phase and the cutting")

    def reset_gate(self, n, rb, rc, sharing):
        """``G9:n<N>:reset`` on the per-side medians of reset p95 over every ABBA process at N. Both sides
        EXCLUSIVE_GPU: FAIL above ratio x B + slack. Otherwise (SHARED_GPU or UNKNOWN on either side; smoke_001
        showed per-process reset modes of ~1 ms or ~7 ms independent of the variant): the strict limit still decides
        PASS, a value between the strict and the shared limit (ratio x B + shared slack) is a WARN
        (``shared_gpu_reset_bimodality``), FAIL only above the shared limit."""
        th = self.cfg["thresholds"]
        mb, mc = statistics.median(rb), statistics.median(rc)
        ratio = float(th["reset_p95_ratio_max"])
        strict = mb * ratio + float(th["reset_p95_slack_ms"])
        shared_limit = mb * ratio + float(th["reset_p95_shared_slack_ms"])
        exclusive = all(x == "EXCLUSIVE_GPU" for xs in sharing.values() for x in xs) and all(sharing.values())
        value = dict(baseline_median=mb, candidate_median=mc, baseline=rb, candidate=rc, sharing=sharing,
                     regime="exclusive" if exclusive else "shared")
        threshold = dict(strict_max_ms=strict, rule=th.get("reset_p95_rule"))
        if exclusive:
            status = "PASS" if mc <= strict else "FAIL"
        else:
            threshold["shared_max_ms"] = shared_limit
            status = "PASS" if mc <= shared_limit else "FAIL"
            if strict < mc <= shared_limit:
                value["warning"] = "shared_gpu_reset_bimodality"
                self.warnings.append(
                    f"WARN shared_gpu_reset_bimodality: G9:n{n}:reset candidate median reset p95 {mc:.3f} ms > strict "
                    f"{strict:.3f} ms (1.25 x B {mb:.3f} + {th['reset_p95_slack_ms']} ms) but <= shared limit "
                    f"{shared_limit:.3f} ms (+ {th['reset_p95_shared_slack_ms']} ms); sharing {sharing}")
        self.gate(f"G9:n{n}:reset", "PERFORMANCE", status, value=value, threshold=threshold)

    def row(self, v, outer):
        r = next((x for x in v["result"].get("rows") or [] if x.get("num_envs") == v["run"].get("num_envs")), {})
        g = v["run"].get("gpu") or {}
        keep = ("num_envs", "particles_per_env", "control_hz", "material_dt_s", "cameras", "env_steps_per_s",
                "aggregate_sim_time_per_wall", "vram_peak_mib", "vram_source", "reset_p95_ms", "physics_verdict",
                "substeps", "robot_dof", "contacts", "recording", "warm", "gpu_sharing", "co_tenant", "gpu")
        out = {k: r.get(k) for k in keep}
        out.update(run=v["id"], side=v["side"], pair=v["run"].get("pair"), order=v["run"].get("order"),
                   in_process_env_steps_per_s=r.get("env_steps_per_s"), env_steps_per_s=outer["env_steps_per_s"],
                   aggregate_sim_time_per_wall=outer["env_steps_per_s"] / float(r.get("control_hz") or 50.),
                   clock="outer (caller)", caller_gpu_label=g.get("label"), caller_util_p50=g.get("util_p50"),
                   physics_verdict="NOT_JUDGED_THROUGHPUT_RUN (work check: final labelled counts)")
        return out

    # -- top level
    def run(self):
        cfg_sha = PA.sha256_file(self.cfg_path)
        if self.man.get("schema") != PA.SCHEMA_MANIFEST:
            self.integrity("I-manifest", "caller", ["PHYSICS", "PERFORMANCE"], f"schema {self.man.get('schema')!r}")
        if self.man.get("config_sha256") not in (None, cfg_sha):
            self.integrity("I-config", "caller", ["PHYSICS", "PERFORMANCE"],
                           f"manifest config_sha256 {str(self.man.get('config_sha256'))[:12]} != {cfg_sha[:12]}")
        seeds = self.man.get("seeds")
        if seeds is not None and seeds != self.cfg["seeds"]:
            self.integrity("I-seeds", "caller", ["PHYSICS"], f"manifest seeds {seeds} != config {self.cfg['seeds']}")
        if PA.canonical_json((self.man.get("baseline") or {}).get("variant")) != \
                PA.canonical_json(self.cfg["reference_variant"]):
            self.integrity("I-reference", "caller", ["PHYSICS", "PERFORMANCE"],
                           "manifest baseline variant is not the config reference_variant")
        if not isinstance((self.man.get("candidate") or {}).get("variant"), dict):
            self.integrity("I-proposal", "caller", ["PHYSICS", "PERFORMANCE"], "candidate variant missing")
        if self.band_error:
            if self.cfg["physics"]["position_band"]["mode"] == "gate":
                self.gate("position_band", "PHYSICS", "BLOCKED", detail=f"band file: {self.band_error}")
            else:
                self.warnings.append(f"WARN position band file unusable (descriptive): {self.band_error}")
        ratios = {}
        if self.man.get("blocked"):
            self.gate("blocked", "PHYSICS", "BLOCKED", detail=str(self.man["blocked"])[:300])
            self.gate("blocked", "PERFORMANCE", "BLOCKED", detail=str(self.man["blocked"])[:300])
        else:
            self.cpu_gates()
            self.plan_gates()
            ratios = self.performance_gates()
            self.window_gate()
        integrity_checked = not any(g["layer"] == "INTEGRITY" for g in self.gates)
        if integrity_checked:
            self.gate("I-all", "INTEGRITY", "PASS", detail="results, dumps (sha256), commits, package roots, "
                      "harness bytes, variants, outer/inner clocks and GPUs consistent")
        layers = self.layers()
        shared = sorted({r.get("gpu_sharing") or "UNKNOWN" for r in self.rows})
        limitations = list(LIMITATIONS)
        if self.cfg["performance"].get("window_check"):
            limitations.append("G9:window is measured on the episode-mode runs of the window_check plans (same "
                               "variants, same seeds, replay from reset), not inside the G9 throughput processes; it "
                               "gates tool-material contact steps (stats slot 1) and new labels, particle-particle "
                               "contacts (slot 4) are reported only")
        if self.cfg["performance"].get("phases"):
            limitations.append("performance.phases are in-process step walls (candidate-reported), descriptive only")
        if "EXCLUSIVE_GPU" not in shared or len(shared) > 1:
            limitations.append(f"G9 rows sharing labels {shared}: absolute rows are not EXCLUSIVE_GPU; the paired "
                               "ratios are relative (same GPU, same hold, interleaved)")
        return dict(schema=PA.SCHEMA_VERDICT, config=self.cfg.get("version"), config_sha256=cfg_sha,
                    manifest_sha256=PA.sha256_file(self.manifest_path),
                    judge=dict(version=PA.JUDGE_VERSION, commit=git("rev-parse", "HEAD"),
                               sha256=PA.sha256_file(Path(__file__).resolve()), checkout=str(ROOT),
                               python=sys.version.split()[0], interpreter=sys.executable,
                               cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES")),
                    baseline=self.man.get("baseline"), candidate=self.man.get("candidate"),
                    layers=layers, gates=self.gates, failed_gates=sorted({g["id"] for g in self.gates
                                                                         if g["status"] == "FAIL"}),
                    performance=dict(ratios=ratios, **({"phases": self.phase_report} if self.phase_report else {})),
                    rows=self.rows, notes=self.notes, warnings=self.warnings,
                    position_band=self.band or dict(error=self.band_error, **self.cfg["physics"]["position_band"]),
                    limitations=limitations,
                    wall_s=time.perf_counter() - T0, peak_rss_mib=peak_rss_mib())


def peak_rss_mib():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.


def blocked_verdict(error, cfg_path=None):
    return dict(schema=PA.SCHEMA_VERDICT, judge=dict(version=PA.JUDGE_VERSION, commit=git("rev-parse", "HEAD")),
                layers={"PHYSICS": "BLOCKED", "PERFORMANCE": "BLOCKED", "INTEGRITY": "BLOCKED"},
                gates=[dict(id="judge_error", layer="INTEGRITY", status="BLOCKED", value=None, threshold=None,
                            detail=str(error)[-1500:], affects=["PHYSICS", "PERFORMANCE"])],
                failed_gates=[], performance=dict(ratios={}), rows=[], notes=[], limitations=list(LIMITATIONS),
                wall_s=time.perf_counter() - T0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--known-harness", default=None,
                   help="RE-JUDGE OF PRESERVED OUTPUTS ONLY (never in an acceptance): JSON list of {harness_sha256, "
                        "shared_module_sha256, note} of an earlier harness version whose outputs are accepted")
    args = p.parse_args(argv)
    out = Path(args.out)
    if out.exists() and (not out.is_dir() or (out / "verdict.json").exists() or any(out.iterdir())):
        print(f"judge_perf_acceptance: --out {out} is not an empty directory (never overwritten)", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    try:
        cfg = PA.load_config(args.config)
        manifest = read_json(args.manifest)
        known = json.loads(Path(args.known_harness).read_text()) if args.known_harness else None
        verdict = Judge(cfg, args.config, manifest, args.manifest, known_harness=known).run()
        if known:
            verdict["rejudge"] = dict(known_harness=known, note="re-judge of preserved outputs produced by an earlier "
                                      "harness version; not an acceptance")
            verdict["limitations"].append("re-judge: harness bytes of the listed earlier version accepted")
    except Exception:
        verdict = blocked_verdict(traceback.format_exc())
    with (out / "verdict.json").open("x") as stream:
        json.dump(verdict, stream, indent=1, sort_keys=True, allow_nan=True, default=str)
    print(json.dumps(dict(verdict=str(out / "verdict.json"), layers=verdict["layers"],
                          failed_gates=verdict.get("failed_gates"),
                          ratios={k: v.get("median") for k, v in verdict["performance"]["ratios"].items()})))
    return 0


if __name__ == "__main__":
    sys.exit(main())
