"""Shared pieces of the versioned performance acceptance (P1-6 R0, decision D-11): constants, the stdout mark,
seed rounds, variant validation, the sample-position codec, junit summaries and the acceptance config check.

Used by ``scripts/rl_ready/run_perf_acceptance.py`` (the harness, in the process under test) and
``scripts/rl_ready/judge_perf_acceptance.py`` (the judge, CPU, from the frozen evaluator checkout). NumPy and the
standard library only; nothing here imports Torch or Warp. Part of the evaluator file set of
``configs/rl_benchmark/acceptance_v1.json``: a candidate may not change it.
"""
import hashlib
import inspect
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

HARNESS_VERSION = "perf_acceptance_v1"
JUDGE_VERSION = "judge_perf_acceptance_v1"
SCHEMA_RESULT = "rcs_perf_result_v1"
SCHEMA_VERDICT = "rcs_perf_verdict_v1"
SCHEMA_CONFIG = "rcs_perf_acceptance_v1"
SCHEMA_MANIFEST = "rcs_perf_accept_manifest_v1"
MARK = "RCS_PERF_MARK"
MARK_EVENTS = ("warm_begin", "warm_end", "episode_end")
MODES = ("throughput", "episode", "samestate", "profile")
WORDS = ("PASS", "FAIL", "BLOCKED", "NOT_RUN")
LAYERS = ("PHYSICS", "PERFORMANCE", "INTEGRITY")
EPISODE_STEPS = 1200            # WP-17 TRIM: 24 s at 50 Hz
CONTROL_DT_S = 0.02
# Native sample indices (0.2 s apart; sample k is control step 10 k) whose recorded state the samestate mode
# writes into the batch before running the contact stage alone. Particle contacts exist at samples 32-36 (s512)
# and 33-85 (s15) of the P1-3a replays; 15 is a contact-free control; 33 / 43 are the P1-3a same-state samples.
SAMESTATE_SAMPLES = (15, 32, 33, 34, 35, 36, 43, 60, 80)
# Intermediate sample positions are stored as int8 multiples of this quantum relative to the Native sample
# (|error| <= 0.5 um); deltas outside +-127 quanta or non-finite are stored exactly (float64) as escapes.
POSITION_QUANTUM_M = 1e-6
POSITION_ESCAPE = -128
# open_vector_env keywords that define the workload; a variant may not set them.
RESERVED_OPTIONS = ("num_envs", "task", "device", "runtime", "material", "backend", "action_spec", "reward",
                    "observation_profile", "autoreset", "strict", "output_dir", "parity_mode", "cameras", "sensors",
                    "recorders", "debug_sync", "reward_config")
CONFIG_KEYS = ("schema", "version", "harness", "judge", "evaluator_files", "reference_variant", "seeds",
               "gpu_mem_mib", "plan", "performance", "cpu_tests", "baseline_measure", "frozen_paths", "dev_limits")
PLAN_KEYS = ("id", "gate", "mode", "num_envs", "seeds", "warmup_steps", "steps", "variant")
PERF_KEYS = ("num_envs", "primary_num_envs", "pairs", "order", "warmup_steps", "steps", "seeds")
THRESHOLD_KEYS = ("primary_median_min", "primary_pairs_above_1", "all_median_min", "vram_ratio_max",
                  "vram_slack_mib", "reset_p95_ratio_max", "reset_p95_slack_ms", "clock_tolerance")
PHYSICS_KEYS = ("position_band_m", "impulse_rel_tol", "contact_total_rel_tol", "replay_position_tol_m")


# ----------------------------------------------------------------------------- marks

def format_mark(event, global_step, num_envs):
    """``RCS_PERF_MARK {...}`` (one line; printed by the harness after a device synchronize)."""
    if event not in MARK_EVENTS:
        raise ValueError(f"mark event {event!r}; one of {MARK_EVENTS}")
    return MARK + " " + json.dumps({"event": event, "global_step": int(global_step), "num_envs": int(num_envs)},
                                   sort_keys=True)


def parse_mark(line):
    """The mark dict of a stdout line, None for any other line or a malformed mark."""
    text = line.strip()
    if not text.startswith(MARK + " "):
        return None
    try:
        mark = json.loads(text[len(MARK) + 1:])
    except ValueError:
        return None
    if not isinstance(mark, dict) or mark.get("event") not in MARK_EVENTS:
        return None
    for key in ("global_step", "num_envs"):
        if not isinstance(mark.get(key), int) or isinstance(mark.get(key), bool):
            return None
    return mark


def outer_rate(marks):
    """Env-steps/s between the last ``warm_end`` and the first ``episode_end`` after it; ``marks`` are
    ``{"t_mono": seconds, "mark": {...}}`` rows in arrival order (the caller's clock)."""
    warm = [m for m in marks if m["mark"]["event"] == "warm_end"]
    if not warm:
        return None
    w = warm[-1]
    ends = [m for m in marks if m["mark"]["event"] == "episode_end" and m["t_mono"] >= w["t_mono"]]
    if not ends:
        return None
    e = ends[0]
    steps = int(e["mark"]["global_step"]) - int(w["mark"]["global_step"])
    dt = float(e["t_mono"]) - float(w["t_mono"])
    n = int(e["mark"]["num_envs"])
    if steps <= 0 or dt <= 0:
        return None
    return {"env_steps": steps * n, "seconds": dt, "env_steps_per_s": steps * n / dt, "num_envs": n,
            "vector_steps": steps}


# ----------------------------------------------------------------------------- rounds and variants

def plan_rounds(mode, seeds, num_envs):
    """Per-env scene seeds of each round (one batch = one round, built and released in turn, never two live).

    ``samestate``: one round per seed (all envs that seed). Otherwise one round with seeds assigned cyclically
    when ``num_envs >= len(seeds)`` (N=4, seeds 512,15 -> 512,15,512,15), else one round per seed."""
    seeds = [int(s) for s in seeds]
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}; one of {MODES}")
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("seeds: a nonempty list of distinct nonnegative ints")
    if not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive int")
    if mode == "samestate" or num_envs < len(seeds):
        return [tuple([s] * num_envs) for s in seeds]
    return [tuple(seeds[i % len(seeds)] for i in range(num_envs))]


def unsupported_option(detail):
    from ..batched.errors import UnsupportedBatchConfiguration
    try:
        from .vector_env import UnsupportedVectorConfiguration
        return UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", detail)
    except ImportError:   # pragma: no cover - vector_env always importable on CPU
        return UnsupportedBatchConfiguration("UNSUPPORTED_OPTION", detail)


def accepted_options():
    """Keyword names ``open_vector_env`` passes on (its own and ``VectorCuttingEnv.__init__``'s), or None when
    either takes ``**kwargs`` beyond the pass-through (then the env itself must refuse unknown options)."""
    from .vector_env import VectorCuttingEnv, open_vector_env
    names = set()
    for fn in (open_vector_env, VectorCuttingEnv.__init__):
        for p in inspect.signature(fn).parameters.values():
            if p.kind == p.VAR_KEYWORD:
                if fn is open_vector_env:
                    continue
                return None
            if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY) and p.name != "self":
                names.add(p.name)
    return names - set(RESERVED_OPTIONS)


def check_variant(variant, accepted="introspect"):
    """Raise ``UNSUPPORTED_OPTION`` for a variant that is not a JSON object of known, non-workload options."""
    if not isinstance(variant, dict):
        raise unsupported_option(f"variant must be a JSON object, got {type(variant).__name__}")
    if accepted == "introspect":
        accepted = accepted_options()
    reserved = sorted(k for k in variant if k in RESERVED_OPTIONS)
    if reserved:
        raise unsupported_option(f"{reserved}: reserved by the harness (they define the workload)")
    if accepted is not None:
        unknown = sorted(k for k in variant if k not in accepted)
        if unknown:
            raise unsupported_option(f"{unknown}: not an open_vector_env option")
    return dict(variant)


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------------------------------------------------------- sample position codec

def encode_positions(native, positions, quantum=POSITION_QUANTUM_M):
    """``positions - native`` as int8 quanta plus exact escapes (flat indices, float64 deltas)."""
    native = np.asarray(native, np.float64)
    positions = np.asarray(positions, np.float64)
    if native.shape != positions.shape:
        raise ValueError(f"shape {positions.shape} vs native {native.shape}")
    delta = positions - native
    with np.errstate(invalid="ignore"):
        q = np.rint(delta / quantum)
        ok = np.isfinite(q) & (np.abs(q) <= 127)
    dq = np.where(ok, q, POSITION_ESCAPE).astype(np.int8)
    idx = np.flatnonzero(~ok).astype(np.int64)
    return dict(dq=dq, escape_index=idx, escape_delta=delta.reshape(-1)[idx].astype(np.float64))


def decode_positions(native, dq, escape_index, escape_delta, quantum=POSITION_QUANTUM_M):
    native = np.asarray(native, np.float64)
    dq = np.asarray(dq)
    if dq.shape != native.shape or dq.dtype != np.int8:
        raise ValueError(f"dq {dq.shape}/{dq.dtype} vs native {native.shape}")
    out = native + dq.astype(np.float64) * quantum
    idx = np.asarray(escape_index, np.int64)
    if idx.size:
        if idx.min() < 0 or idx.max() >= out.size or not np.all(dq.reshape(-1)[idx] == POSITION_ESCAPE):
            raise ValueError("escape indices do not match escape markers")
        flat = out.reshape(-1)
        flat[idx] = native.reshape(-1)[idx] + np.asarray(escape_delta, np.float64)
    if np.count_nonzero(dq == POSITION_ESCAPE) != idx.size:
        raise ValueError("escape markers without escape values")
    return out


# ----------------------------------------------------------------------------- small helpers

def worst(words):
    """FAIL > BLOCKED > NOT_RUN > PASS; an empty list is NOT_RUN."""
    words = list(words)
    for w in ("FAIL", "BLOCKED", "NOT_RUN"):
        if w in words:
            return w
    return "PASS" if words else "NOT_RUN"


def percentile(values, q):
    values = sorted(float(v) for v in values)
    if not values:
        return None
    position = (len(values) - 1) * q / 100.
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def digit_rewrites(previous, current, max_cuts=2):
    """Particles whose already-assigned packed digit changed between two code arrays (labels are persistent)."""
    a, b = np.asarray(previous, np.int64), np.asarray(current, np.int64)
    out = 0
    for k in range(max_cuts):
        x, y = (a >> (2 * k)) & 3, (b >> (2 * k)) & 3
        out += int(np.count_nonzero((x != 0) & (y != x)))
    return out


def junit_summary(path):
    """``{tests, passed, failed, errors, skipped, failed_ids, skipped_ids}`` of a pytest junit file."""
    path = Path(path)
    if not path.is_file():
        return {"error": f"no junit file {path.name}"}
    root = ET.parse(path).getroot()
    failed, skipped, errors, n = [], [], [], 0
    for tc in root.iter("testcase"):
        n += 1
        tid = f"{tc.get('classname', '')}::{tc.get('name', '')}"
        if tc.find("failure") is not None:
            failed.append(tid)
        elif tc.find("error") is not None:
            errors.append(tid)
        elif tc.find("skipped") is not None:
            skipped.append(tid)
    return {"tests": n, "passed": n - len(failed) - len(errors) - len(skipped), "failed": len(failed),
            "errors": len(errors), "skipped": len(skipped), "failed_ids": sorted(failed + errors),
            "skipped_ids": sorted(skipped)}


# ----------------------------------------------------------------------------- acceptance config

def _ints(x):
    return isinstance(x, list) and bool(x) and all(isinstance(v, int) and not isinstance(v, bool) for v in x)


def config_problems(cfg):
    """Everything the harness, the judge and the external caller need of an acceptance config."""
    if not isinstance(cfg, dict):
        return ["not a JSON object"]
    p = [f"missing {k}" for k in CONFIG_KEYS + ("thresholds", "physics") if k not in cfg]
    if p:
        return p
    if cfg["schema"] != SCHEMA_CONFIG:
        p.append(f"schema {cfg['schema']!r} is not {SCHEMA_CONFIG}")
    for k in ("harness", "judge"):
        if not isinstance(cfg[k], str) or cfg[k].startswith("/") or ".." in cfg[k]:
            p.append(f"{k}: a repository-relative path")
    if not isinstance(cfg["evaluator_files"], list) or not all(isinstance(x, str) for x in cfg["evaluator_files"]):
        p.append("evaluator_files: a list of fnmatch globs")
    if not isinstance(cfg["reference_variant"], dict):
        p.append("reference_variant: open_vector_env kwargs of the baseline side")
    s = cfg["seeds"]
    if not (isinstance(s, dict) and _ints(s.get("dev")) and _ints(s.get("heldout"))
            and not set(s["dev"]) & set(s["heldout"])):
        p.append("seeds: {dev: [..], heldout: [..]} disjoint int lists")
    if not isinstance(cfg["gpu_mem_mib"], dict) or not all(str(k).isdigit() and isinstance(v, int)
                                                             for k, v in cfg["gpu_mem_mib"].items()):
        p.append('gpu_mem_mib: {"N": MiB}')
    ids = []
    for i, e in enumerate(cfg["plan"] if isinstance(cfg["plan"], list) else [None]):
        if not isinstance(e, dict) or any(k not in e for k in PLAN_KEYS):
            p.append(f"plan[{i}]: needs {', '.join(PLAN_KEYS)}")
            continue
        ids.append(e["id"])
        if e["mode"] not in MODES or e["variant"] not in ("proposal", "default") or not _ints(e["seeds"]):
            p.append(f"plan[{i}] {e.get('id')}: mode in {MODES}, variant proposal|default, seeds ints")
        if int(e["warmup_steps"]) + int(e["steps"]) > EPISODE_STEPS:
            p.append(f"plan[{i}] {e.get('id')}: warmup_steps + steps > {EPISODE_STEPS}")
        if set(e.get("sides") or ("baseline", "candidate")) - {"baseline", "candidate"}:
            p.append(f"plan[{i}] {e.get('id')}: sides baseline|candidate")
    if len(set(ids)) != len(ids):
        p.append("plan: ids must be unique")
    perf = cfg["performance"]
    if not isinstance(perf, dict) or any(k not in perf for k in PERF_KEYS):
        p.append(f"performance: needs {', '.join(PERF_KEYS)}")
    elif not (isinstance(perf["order"], str) and set(perf["order"]) <= {"B", "C"}
              and len(perf["order"]) == 2 * int(perf["pairs"]) and _ints(perf["num_envs"])
              and perf["primary_num_envs"] in perf["num_envs"]
              and all(perf["order"][2 * i:2 * i + 2] in ("BC", "CB") for i in range(int(perf["pairs"])))):
        p.append("performance: order of B/C pairs (BC or CB each), 2 x pairs long; primary_num_envs in num_envs")
    if not isinstance(cfg["cpu_tests"], dict) or not isinstance(cfg["cpu_tests"].get("args"), list):
        p.append("cpu_tests: {args: [pytest arguments]}")
    if not isinstance(cfg["baseline_measure"], list):
        p.append("baseline_measure: a list of {mode, num_envs, seeds, warmup_steps, steps}")
    if not isinstance(cfg["frozen_paths"], list):
        p.append("frozen_paths: fnmatch globs the candidate may not change")
    d = cfg["dev_limits"]
    if not isinstance(d, dict) or any(not isinstance(d.get(k), int) for k in ("max_num_envs", "max_steps",
                                                                            "episode_max_steps")):
        p.append("dev_limits: {max_num_envs, max_steps, episode_max_steps} ints")
    if cfg.get("judge_python", "rcs") not in ("rcs", "sim"):
        p.append("judge_python: rcs | sim")
    th = cfg["thresholds"]
    if not isinstance(th, dict) or any(not isinstance(th.get(k), (int, float)) for k in THRESHOLD_KEYS):
        p.append(f"thresholds: needs numbers {', '.join(THRESHOLD_KEYS)}")
    ph = cfg["physics"]
    if not isinstance(ph, dict) or any(k not in ph for k in PHYSICS_KEYS):
        p.append(f"physics: needs {', '.join(PHYSICS_KEYS)}")
    elif not all(str(k).isdigit() and isinstance(v, list) and len(v) == 2 for k, v in ph["position_band_m"].items()):
        p.append('physics.position_band_m: {"seed": [low, high]}')
    return p


def load_config(path):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    bad = config_problems(cfg)
    if bad:
        raise ValueError(f"{path}: " + "; ".join(bad))
    return cfg


__all__ = ["HARNESS_VERSION", "JUDGE_VERSION", "SCHEMA_RESULT", "SCHEMA_VERDICT", "SCHEMA_CONFIG",
           "SCHEMA_MANIFEST", "MARK", "MODES", "WORDS", "LAYERS", "format_mark", "parse_mark", "outer_rate",
           "plan_rounds", "check_variant", "accepted_options", "encode_positions", "decode_positions", "worst",
           "percentile", "digit_rewrites", "junit_summary", "config_problems", "load_config"]
