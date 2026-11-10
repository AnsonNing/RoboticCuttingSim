#!/usr/bin/env python3
"""P1-4 long-rollout CHILD process (T4.1 / T4.4): one attempt of one run of ``configs/rl_benchmark/long_rollout_v1.json``.

Normally launched by ``scripts/rl_ready/supervise_rollout.py`` (which owns the attempt directory, watchdog,
VRAM sampling, classification and restart). The child writes ``<attempt-dir>/ledger.jsonl`` (append-only,
fsync per record; ``rl.supervision.Ledger``): ``start``, ``start_segments``, ``census``, ``heartbeat``,
``sample``, ``reset``, ``episode``, ``l2_audit``, ``infra_error`` (top-level handler), ``end``.

Backends (one material = one device HashGrid per process):

* ``trim`` (GPU, ``.deps/isaacsim-venv``): workload A ``VectorCuttingEnv`` (``strict`` from the run), drives per
  env from the run (replay canary through ``step_measured_edge``, ``RecordedCommandPolicyV1``,
  ``RandomPolicyV1``; all in one batched call per step), per-episode L2 on the device, the frozen evaluator at
  every episode end, canary comparison with the Native final codes, HashGrid / graph-capture census hooks.
* ``fake`` (CPU, NumPy only): a small synthetic batched env with the same info contract (``legacy`` or
  ``termination_reason`` style). It exercises the identical schedule / ledger / classification / fault code
  for the CPU fault-injection tests. It is never a measurement.

Fault injection (this script only; ``RCS_FAULT_INJECT``, comma-separated ``kind@batch_step``): exception,
memoryerror, cuda_error, cuda_oom (simulated error strings), sigsegv, sigabrt, sigkill, hang, disk,
strict_invalid, nan_env:E, second_grid, device_oom (trim backend only: real device OOM confined to this
process via ``torch.cuda.set_per_process_memory_fraction``). A fault ``@k`` fires right after batch step k.
"""
import argparse
import faulthandler
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from robotic_cutting_sim.rl import supervision as S  # noqa: E402
from robotic_cutting_sim.rl.long_rollout import RolloutSchedule  # noqa: E402

DRIVES = ("replay_canary", "recorded_command_policy_v1", "random_policy_v1")


# ----------------------------------------------------------------------------- fault injection (child only)

def parse_faults(text):
    faults = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        kind, _, step = item.rpartition("@")
        env = None
        for prefix in ("nan_env", "bad_pose", "infra_env", "reset_leak"):
            if kind.startswith(prefix + ":"):
                kind, env = prefix, int(kind.split(":", 1)[1])
        faults.append(dict(kind=kind, step=int(step), env=env))
    known = {"exception", "memoryerror", "cuda_error", "cuda_oom", "sigsegv", "sigabrt", "sigkill", "hang", "disk",
             "strict_invalid", "nan_env", "second_grid", "device_oom", "bad_pose", "infra_env", "reset_leak"}
    bad = [f["kind"] for f in faults if f["kind"] not in known]
    if bad:
        raise ValueError(f"unknown RCS_FAULT_INJECT kinds {bad}")
    return faults


class FaultInjector:
    def __init__(self, faults, ledger):
        self.faults, self.ledger = list(faults), ledger

    def after_step(self, global_step, backend):
        for f in [f for f in self.faults if f["step"] == global_step]:
            self.ledger.write("fault_injected", global_step=global_step, **f)
            kind = f["kind"]
            if kind == "exception":
                raise RuntimeError("injected exception (RCS_FAULT_INJECT)")
            if kind == "memoryerror":
                raise MemoryError("injected host allocation failure (RCS_FAULT_INJECT)")
            if kind == "cuda_error":
                raise RuntimeError("CUDA error: an illegal memory access was encountered (simulated by RCS_FAULT_INJECT)")
            if kind == "cuda_oom":
                raise RuntimeError("CUDA out of memory. Tried to allocate 20.00 GiB (simulated by RCS_FAULT_INJECT)")
            if kind == "disk":
                raise OSError(28, "No space left on device (simulated by RCS_FAULT_INJECT)")
            if kind == "strict_invalid":
                raise _invalid_batch_envs([0], {0: ["NONFINITE_STATE"]}, "(simulated by RCS_FAULT_INJECT)")
            if kind in ("sigsegv", "sigabrt", "sigkill"):
                sys.stderr.flush()
                os.kill(os.getpid(), {"sigsegv": signal.SIGSEGV, "sigabrt": signal.SIGABRT,
                                      "sigkill": signal.SIGKILL}[kind])
                time.sleep(5)
            if kind == "hang":
                while True:
                    time.sleep(1.)
            if kind == "nan_env":
                backend.inject_nan(f["env"])
            if kind == "bad_pose":
                backend.inject_bad_pose(f["env"])
            if kind == "infra_env":
                backend.inject_infra(f["env"])
            if kind == "reset_leak":
                backend.inject_reset_leak(f["env"])
            if kind == "second_grid":
                backend.second_grid()
            if kind == "device_oom":
                backend.device_oom()


def _invalid_batch_envs(ids, reasons, note=""):
    """The material's strict-mode exception (``InvalidBatchEnvs``) when importable, else a FloatingPointError."""
    try:
        from robotic_cutting_sim.batched.material_batch import InvalidBatchEnvs
        error = InvalidBatchEnvs(ids, reasons)
        error.args = (f"{error.args[0]} {note}".strip(),)
        return error
    except ImportError:
        return FloatingPointError(f"Invalid batched envs {ids}: {reasons} {note}")


def env_exception_category(error):
    """The env's own classification of an exception (``rl.termination.classify_exception``) when available."""
    try:
        from robotic_cutting_sim.rl.termination import classify_exception
    except ImportError:        # pre-P1-2b checkout
        return None, None
    try:
        return classify_exception(error)
    except Exception:          # never let classification hide the original error
        return None, None


# ----------------------------------------------------------------------------- fake backend (CPU tests)

class FakeBackend:
    """NumPy stand-in of the batched env for CPU tests (never a measurement)."""

    name = "fake"
    builds = 0          # process-wide, like the trim backend's census hooks
    live = 0            # open fake materials (one "device grid" each)

    def __init__(self, run, *, info_style="legacy", strict=False, step_s=0., particles=32):
        FakeBackend.builds += 1
        FakeBackend.live += 1
        self.N, self.drives = run["num_envs"], [e["drive"] for e in run["envs"]]
        self.scene_seeds = [e["scene_seed"] for e in run["envs"]]
        self.episode_steps, self.info_style, self.strict = run["episode_steps"], info_style, strict
        self.step_s, self.P = float(step_s), int(particles)
        self.x = np.zeros((self.N, self.P))
        self.codes = np.zeros((self.N, self.P), np.int32)
        self.step_count = np.zeros(self.N, np.int64)
        self.seed = np.zeros(self.N, np.int64)
        self.invalid = np.zeros(self.N, np.int64)
        self.infra = np.zeros(self.N, np.int64)
        self.target = np.array([self.P // 2 if d != "random_policy_v1" else self.P // 8 for d in self.drives])
        self.grids, self.extra_inits = 1, 0
        self.native = {s: self._pattern(self.P // 2) for s in set(self.scene_seeds)}

    def _pattern(self, k):
        out = np.zeros(self.P, np.int32)
        out[:k] = 1
        return out

    def receipt(self):
        return dict(backend="fake", num_envs=self.N, particles_per_env=self.P, info_style=self.info_style,
                    strict=self.strict, measurement=False)

    def reset(self, ids, seeds):
        t = time.perf_counter()
        ids = np.asarray(ids, int)
        self.x[ids] = 0.
        self.codes[ids] = 0
        self.step_count[ids] = 0
        self.invalid[ids] = 0
        self.infra[ids] = 0
        self.seed[ids] = np.asarray(seeds, np.int64)
        return time.perf_counter() - t

    def set_episode_index(self, ids, indices):
        pass

    def step(self):
        if self.step_s:
            time.sleep(self.step_s)
        self.step_count += 1
        self.x += 1e-3
        frac = np.minimum(self.step_count / self.episode_steps, 1.)
        for e in range(self.N):
            self.codes[e, :int(self.target[e] * frac[e])] = 1
        bad = ~np.isfinite(self.x).all(1)
        self.invalid |= np.where(bad, 1, 0)
        if self.strict and self.invalid.any():
            ids = np.flatnonzero(self.invalid).tolist()
            raise _invalid_batch_envs(ids, {e: ["NONFINITE_STATE"] for e in ids})
        info = {"step_count": self.step_count.copy(), "invalid_reason": self.invalid.copy(),
                "seed": self.seed.copy()}
        truncated = self.step_count >= self.episode_steps
        if self.info_style == "termination_reason":      # termination_reason_v1 bits (rl.termination)
            reason = (np.where(self.invalid != 0, S.TERM_MATERIAL_INVALID, 0)
                      | np.where(self.infra != 0, S.TERM_INFRASTRUCTURE, 0)
                      | np.where(truncated, S.TERM_TRUNCATED, 0))
            info["termination_reason"] = reason
            terminated = (reason & S.TERMINAL_BITS) != 0
        else:
            terminated = self.invalid != 0
        return S.step_status(info, terminated, truncated, num_envs=self.N)

    def evaluate(self, ids):
        out = {}
        for e in ids:
            labelled = int(np.count_nonzero(self.codes[e]))
            out[e] = dict(verdict="PASS" if labelled >= self.P // 4 else "FAIL", integrity="PASS",
                          outcome="PASS" if labelled >= self.P // 4 else "FAIL", labelled=labelled,
                          final_codes_sha256=hashlib.sha256(np.ascontiguousarray(self.codes[e])).hexdigest(),
                          step_count=int(self.step_count[e]), wall_s=0.)
        return out

    def canary(self, e):
        native = self.native[self.scene_seeds[e]]
        return dict(codes_equal_native=bool(np.array_equal(self.codes[e], native)),
                    different_code_particles=int(np.count_nonzero(self.codes[e] != native)),
                    labelled=int(np.count_nonzero(self.codes[e])), native_labelled=int(np.count_nonzero(native)))

    def l2_take(self, ids):
        return {e: dict(status="PASS" if np.isfinite(self.x[e]).all() else "FAIL") for e in ids}

    def l2_update(self):
        pass

    def l2_audit(self, global_step):
        return None

    def memory(self, census=False):
        out = FakeBackend.process_memory(census)
        out["hashgrid_device_inits"] += self.extra_inits
        if census:
            out["live_device_hashgrids"] += self.extra_inits
        return out

    @staticmethod
    def process_memory(census=False):
        out = dict(S.proc_status() or {}, mempool_current_bytes=None, mempool_high_bytes=None,
                   torch_allocated_bytes=None, torch_reserved_bytes=None, graph_captures=FakeBackend.builds,
                   hashgrid_device_inits=FakeBackend.builds, warp_modules=3)
        if census:
            out["live_device_hashgrids"] = FakeBackend.live
        return out

    @staticmethod
    def process_own_vram_mib(gpu_index):
        return None

    def inject_nan(self, e):
        self.x[e, 0] = np.nan

    def second_grid(self):
        self.grids += 1
        self.extra_inits += 1

    def inject_bad_pose(self, e):
        self.invalid[e] |= 4                       # INVALID_MEASURED_POSE (the material's own bit)

    def inject_reset_leak(self, e):
        self.leak = getattr(self, "leak", []) + [int(e)]

    def state_checksums(self):
        return {"x": np.array([hash(self.x[e].tobytes()) for e in range(self.N)]),
                "code": np.array([hash(self.codes[e].tobytes()) for e in range(self.N)]),
                "step_count": self.step_count.copy()}

    def reset_check(self, ids, reference):
        for e in getattr(self, "leak", []):        # fault: a reset that left state behind
            if e in ids:
                self.x[e, 0] = 1.
        self.leak = [e for e in getattr(self, "leak", []) if e not in ids]
        current = self.state_checksums()
        return {int(e): [n for n in sorted(reference) if current[n][e] != reference[n][e]] for e in ids}

    def canary_sample(self, e):
        k = int(self.target[e] * min(self.step_count[e] / self.episode_steps, 1.))
        return dict(env=int(e), step=int(self.step_count[e]),
                    different_code_particles=int(np.count_nonzero(self.codes[e] != self._pattern(k))),
                    max_position_diff_m=0., labelled=int(np.count_nonzero(self.codes[e])))



    def inject_infra(self, e):
        self.infra[e] = 1

    def device_oom(self):
        raise RuntimeError("device_oom needs the trim backend")

    def close(self):
        if self.grids:
            FakeBackend.live -= 1
        self.grids = 0          # the material releases its grid


# ----------------------------------------------------------------------------- TRIM backend (GPU)

def _site(stack):
    """Creator of a wp.HashGrid: the innermost frame outside Warp and outside the counting hook."""
    frames = [f for f in stack[:-1] if f"{os.sep}warp{os.sep}" not in f.filename]
    f = frames[-1] if frames else stack[-1]
    return f"{Path(f.filename).name}:{f.lineno}:{f.name}"


_HOOKS = None


def census_hooks():
    """Process-wide census hooks, installed once before any material is built: every ``wp.HashGrid.__init__``
    (device, creator) and every ``wp.capture_begin``."""
    global _HOOKS
    if _HOOKS is None:
        import warp as wp
        hooks = dict(hashgrids=[], captures=[0])
        init = wp.HashGrid.__init__

        def counted_init(grid, *a, **k):
            init(grid, *a, **k)
            hooks["hashgrids"].append(dict(device=str(grid.device), created_by=_site(traceback.extract_stack())))
        wp.HashGrid.__init__ = counted_init
        begin = wp.capture_begin

        def counted_begin(*a, **k):
            hooks["captures"][0] += 1
            return begin(*a, **k)
        wp.capture_begin = counted_begin
        _HOOKS = hooks
    return _HOOKS


class TrimBackend:
    """Workload A ``VectorCuttingEnv`` driven per the run's env table (see the module docstring)."""

    name = "trim"

    def __init__(self, run, common, output_dir):
        import torch
        import warp as wp
        self.torch, self.wp = torch, wp
        hooks = census_hooks()
        self.hashgrids, self.capture_count = hooks["hashgrids"], hooks["captures"]
        sys.path.insert(0, str(ROOT / "scripts" / "rl_ready"))
        import run_vector_milestones as rvm
        self.rvm = rvm
        from robotic_cutting_sim.rl.vector_env import VectorCuttingEnv
        from robotic_cutting_sim.rl.policies_batched import (MeasuredTrajectoryDriver, RandomPolicyV1,
                                                             RecordedCommandPolicyV1, recorded_command_actions)
        N = run["num_envs"]
        self.N = N
        self.drives = [e["drive"] for e in run["envs"]]
        self.scene_seeds = [int(e["scene_seed"]) for e in run["envs"]]
        bad = [d for d in self.drives if d not in DRIVES]
        if bad or len(self.drives) != N:
            raise ValueError(f"run env table: {N} envs with drives in {DRIVES} required")
        order = sorted(set(self.scene_seeds))
        inputs = rvm.load_inputs(order)
        canary_seeds = [s for s in order if any(d == "replay_canary" and self.scene_seeds[e] == s
                                                for e, d in enumerate(self.drives))]
        natives = {s: rvm.native_final_codes(s) for s in canary_seeds}
        self.natives = {s: v[0] for s, v in natives.items()}
        self.native_data = {s: v[1] for s, v in natives.items()}
        options = dict(particle_contact_mode=run.get("particle_contact_mode", "frozen_v1"))
        if "input_guard" in run:                  # long_rollout_v2: declared per run (None = the env default, on)
            options["input_guard"] = run["input_guard"]
        self.env = VectorCuttingEnv(N, task=rvm.family_for(tuple(self.scene_seeds), inputs), device="cuda",
                                    strict=bool(common.get("strict", False) if run.get("strict") is None
                                                else run["strict"]),
                                    output_dir=output_dir, **options)
        env = self.env
        self.payload_check = rvm.check_payloads(env, inputs)
        rvm._grouped_query_audit = rvm._define_kernels()
        dev = env.torch_device
        replay = [d == "replay_canary" for d in self.drives]
        self.replay_mask = torch.tensor(replay, dtype=torch.bool, device=dev)
        self.any_replay = any(replay)
        self.random_mask = torch.tensor([d == "random_policy_v1" for d in self.drives], dtype=torch.bool, device=dev)
        self.driver = MeasuredTrajectoryDriver([inputs[s]["trajectory"] for s in order],
                                               [order.index(s) if replay[e] else -1
                                                for e, s in enumerate(self.scene_seeds)], device=dev)
        tables, table_of = [], {}
        for e, d in enumerate(self.drives):
            s = self.scene_seeds[e]
            if d == "recorded_command_policy_v1" and s not in table_of:
                rec = inputs[s]["commands"]
                task = env.tasks[e]
                actions, _, report = recorded_command_actions(rec["targets"], rec["cut_enabled"], rec["cut_index"],
                                                              rec["reset_edge"], env.action_limits,
                                                              task.T_world_workpiece, task.max_cuts)
                table_of[s] = len(tables)
                tables.append(actions)
        self.recorded = RecordedCommandPolicyV1(tables or [np.zeros((1, 8), np.float32)],
                                                [table_of.get(s, -1) if d == "recorded_command_policy_v1" else -1
                                                 for d, s in zip(self.drives, self.scene_seeds)], device=dev)
        self.random = RandomPolicyV1()
        self.episode_index = torch.zeros(N, dtype=torch.int64, device=dev)
        self.monitor = _episode_l2(rvm)(env)
        self.extra_grids = []
        self.bad_pose = []

    def receipt(self):
        r = self.env.receipt()
        return dict(backend="trim", num_envs=self.N, drives=self.drives, scene_seeds=self.scene_seeds,
                    particles_per_env=r["material"]["layout"]["particle_counts"],
                    memory_bytes_per_env=r["material"]["memory_bytes"] / self.N, fidelity=r["fidelity"],
                    device_hashgrid_guard=r["material"].get("device_hashgrid_guard"),
                    particle_contact_mode=getattr(self.env, "particle_contact_mode", None),
                    input_guard=str(getattr(self.env, "input_guard_config", "n/a (pre-P1-2b env)")),
                    termination=r.get("termination"), vector_api_revision=r.get("vector_api_revision"),
                    payload_check=self.payload_check, build_wall_s=self.env.build_wall_s, versions=r["versions"],
                    strict=self.env.strict, measurement=True)

    def reset(self, ids, seeds):
        torch = self.torch
        torch.cuda.synchronize()
        t = time.perf_counter()
        if len(ids) == self.N:
            self.env.reset(seeds=list(seeds))
        else:
            self.env.reset(env_ids=list(ids), seeds=list(seeds))
        torch.cuda.synchronize()
        wall = time.perf_counter() - t
        self.monitor.on_reset(list(ids))
        return wall

    def set_episode_index(self, ids, indices):
        for e, i in zip(ids, indices):
            self.episode_index[int(e)] = int(i) if i >= 0 else (1 << 40) + int(e)

    def step(self):
        torch, env = self.torch, self.env
        policy_info = dict(step_count=env._step_count, seed=env._seed, episode=self.episode_index)
        actions = torch.where(self.random_mask[:, None], self.random(None, policy_info),
                              self.recorded(None, policy_info))
        if self.any_replay:
            pose, cut, pads = self.driver.inputs(env._step_count)
            if self.bad_pose:                        # T4.2 re-check: one non-finite measured pose per injection
                pose = pose.clone()
                for e in self.bad_pose:
                    pose[e] = float("nan")
                self.bad_pose = []
            _, _, terminated, truncated, info = env.step_measured_edge(pose, cut, pad_boxes=pads,
                                                                       env_mask=self.replay_mask, actions=actions)
        else:
            _, _, terminated, truncated, info = env.step(actions)
        self.monitor.update()
        status = S.step_status(info, terminated, truncated, num_envs=self.N)   # host copy = the step's sync
        return status

    def evaluate(self, ids):
        out = {}
        for r in self.env.evaluate(list(ids)):
            s = r["summary"]
            out[r["env"]] = dict(verdict=s["verdict"], integrity=s["integrity"], outcome=s["outcome"],
                                 labelled=r["labelled"], final_codes_sha256=r["final_codes_sha256"],
                                 step_count=r["step_count"], frames=r["frames"], wall_s=r["wall_s"],
                                 removed_mass_fraction=s.get("removed_mass_fraction"),
                                 mechanical_components=s.get("mechanical_components"))
        return out

    def canary(self, e):
        from robotic_cutting_sim.rl.replay import digit_counts
        native = self.natives[self.scene_seeds[e]]
        codes = self.env.snapshot_world(e)["codes"]
        return dict(codes_equal_native=bool(np.array_equal(codes, native)),
                    different_code_particles=int(np.count_nonzero(codes != native)),
                    labelled=int(np.count_nonzero(codes)), native_labelled=int(np.count_nonzero(native)),
                    labelled_digits=digit_counts(codes, 2))

    def l2_update(self):
        pass   # updated inside step()

    def l2_take(self, ids):
        return self.monitor.take(list(ids))

    def l2_audit(self, global_step):
        row = self.monitor.audit(global_step)
        return dict(step=row["step"], range_audit_nonzero={k: v for k, v in row["range_audit"].items() if v},
                    grouped_query_visited=row["grouped_query_visited"],
                    grouped_query_foreign=row["grouped_query_foreign"],
                    mass_bitwise_constant=row["mass_bitwise_constant"])

    def memory(self, census=False):
        return TrimBackend.process_memory(census)

    @staticmethod
    def process_memory(census=False):
        """Process-level memory and census (no reference to any env: usable after close)."""
        import torch
        import warp as wp
        from warp._src import context as wctx
        hooks = census_hooks()
        out = dict(S.proc_status() or {},
                   mempool_current_bytes=int(wp.get_mempool_used_mem_current("cuda:0")),
                   mempool_high_bytes=int(wp.get_mempool_used_mem_high("cuda:0")),
                   torch_allocated_bytes=int(torch.cuda.memory_allocated()),
                   torch_reserved_bytes=int(torch.cuda.memory_reserved()),
                   graph_captures=int(hooks["captures"][0]),
                   hashgrid_device_inits=sum(1 for g in hooks["hashgrids"] if g["device"].startswith("cuda")),
                   warp_modules=len(wctx.user_modules))
        if census:
            from robotic_cutting_sim.batched.material_batch import live_device_hashgrids
            out["live_device_hashgrids"] = len(live_device_hashgrids())
        return out

    def inject_nan(self, e):
        torch, wp = self.torch, self.wp
        s, pk = self.env.material.solver, self.env.material.packed
        p0 = int(pk.particle_offset[e])
        with self.env._stream():
            wp.to_torch(s.x)[p0] = float("nan")
        torch.cuda.synchronize()

    def second_grid(self):
        self.extra_grids.append(self.wp.HashGrid(8, 8, 8, device="cuda:0"))

    # -- LR-B: bitwise reset check through per-env checksums of every array the reset kernels write
    PARTICLE_ARRAYS = ("x", "v", "x_start", "impulse", "link_velocity_start", "stress", "F", "progress",
                       "contact_work", "code", "link_codes", "front_crossings", "front_reached", "pending_q",
                       "pending_cut", "slot")
    NODE_SLOT_ARRAYS = ("slot_code", "gm", "gp", "gj", "old", "new_v")
    ENV_SOLVER_ARRAYS = ("edge_flags", "segment_ok", "segment_plow", "stroke_engaged", "stroke_any_motion", "stroke",
                         "stroke_travel", "box_centers", "box_halves", "box_reactions", "slot_overflow",
                         "slot_max_occupied", "previous_cut", "reaction", "poses", "twists", "cuts", "previous_pose",
                         "link_previous_pose", "link_diagnostics", "stats_sum", "stats_max", "contact_totals_sum",
                         "contact_totals_max", "contact_guard_flag", "contact_guard_violations")
    ENV_MATERIAL_ARRAYS = ("pending_wrench", "applied_previous", "pad_wrench", "step_count", "last_pose",
                           "measured_pose", "cut_input", "pose_status", "invalid_reason")
    ENV_TENSORS = ("commanded_pose", "previous_action", "armed", "termination_reason")

    def _words(self, t):
        torch = self.torch
        t = t.contiguous()
        if t.dtype == torch.bool or t.element_size() < 4:
            t = t.to(torch.int32)
        return t.reshape(t.shape[0], -1).view(torch.int32).to(torch.int64) if t.dim() > 1 else \
            t.reshape(-1, 1).view(torch.int32).to(torch.int64)

    def _group_hash(self, t, group, local):
        torch = self.torch
        words = self._words(t)
        width = words.shape[1]
        w = torch.arange(width, dtype=torch.int64, device=words.device) * 2654435761 + 1        # odd weights
        row = (words * w[None]).sum(1)
        row = row * (local * 0x9E3779B1 * 2 + 1)                                               # odd per row
        out = torch.zeros(self.N, dtype=torch.int64, device=words.device)
        out.index_add_(0, group, row)
        return out

    def state_checksums(self):
        """Per-env 64-bit position-weighted checksums of the bit patterns of every array a reset writes (except the
        seeds): one odd weight per word and per row, so any single changed word changes the checksum."""
        torch, wp, env = self.torch, self.wp, self.env
        s, mt = env.material.solver, env.material
        if not hasattr(self, "_groups"):
            dev = env.torch_device
            pof = torch.as_tensor(np.asarray(mt.packed.particle_offset, np.int64), device=dev)
            nof = torch.as_tensor(np.asarray(mt.packed.node_offset, np.int64), device=dev)
            ep = wp.to_torch(s.env_of_particle).to(torch.int64)
            en = wp.to_torch(s.env_of_node).to(torch.int64)
            slots = int(s.slots_per_node)
            node_env = en.repeat_interleave(slots)
            node_local = torch.arange(len(node_env), device=dev) - nof[node_env] * slots
            self._groups = dict(particle=(ep, torch.arange(len(ep), device=dev) - pof[ep]),
                                node=(node_env, node_local))
        out = {}
        with env._stream():
            names = list(self.PARTICLE_ARRAYS) + [v.name for v in s.model.per_particle_state]
            for name in names:
                if hasattr(s, name):
                    out[name] = self._group_hash(wp.to_torch(getattr(s, name)), *self._groups["particle"])
            for name in self.NODE_SLOT_ARRAYS:
                if hasattr(s, name):
                    out[name] = self._group_hash(wp.to_torch(getattr(s, name)), *self._groups["node"])
            ids = torch.arange(self.N, device=env.torch_device)
            zero = torch.zeros(self.N, dtype=torch.int64, device=env.torch_device)
            for owner, names in ((s, self.ENV_SOLVER_ARRAYS), (mt, self.ENV_MATERIAL_ARRAYS)):
                for name in names:
                    a = getattr(owner, name, None)
                    if a is None:
                        continue
                    out[name] = self._group_hash(wp.to_torch(a).reshape(self.N, -1), ids, zero)
            for name in self.ENV_TENSORS:
                t = getattr(env, name, None)
                if t is not None:
                    out["env." + name] = self._group_hash(t.reshape(self.N, -1), ids, zero)
            counts = getattr(env.reward, "counts", None)
            if counts is not None:
                out["env.reward.counts"] = self._group_hash(counts.reshape(-1, self.N).T.contiguous(), ids, zero)
        return out

    def reset_check(self, ids, reference):
        """Arrays whose checksum of a reset env differs from the reference (post initial full reset)."""
        current = self.state_checksums()
        idx = self.torch.as_tensor(list(ids), dtype=self.torch.int64, device=self.env.torch_device)
        names = sorted(reference)
        diff = self.torch.stack([(current[n][idx] != reference[n][idx]) for n in names]).cpu().numpy()
        return {int(e): [names[k] for k in np.flatnonzero(diff[:, j])] for j, e in enumerate(ids)}

    def canary_sample(self, e):
        """Codes / positions of canary env ``e`` vs the Native sample at its current step (None if no sample)."""
        native = self.native_data.get(self.scene_seeds[e])
        if native is None:
            return None
        snap = self.env.snapshot_world(e)
        steps = np.rint(np.asarray(native["time_s"]) / .02).astype(int)
        hit = np.flatnonzero(steps == snap["step_count"])
        if not len(hit):
            return None
        i = int(hit[0])
        codes = np.asarray(native["codes"][i])
        return dict(env=int(e), step=snap["step_count"], different_code_particles=int(np.count_nonzero(
            snap["codes"] != codes)), max_position_diff_m=float(np.abs(snap["positions"] - np.asarray(
                native["positions"][i])).max()), labelled=int(np.count_nonzero(snap["codes"])))

    @staticmethod
    def process_own_vram_mib(gpu_index):
        if gpu_index is None:
            return None
        from robotic_cutting_sim.rl.telemetry import take_sample, process_tree
        sample = take_sample(gpu_index, process_tree(os.getpid()), pmon=False, pids_query=False)
        return sample.get("own_memory_mib")

    def inject_bad_pose(self, e):
        if self.drives[e] != "replay_canary":
            raise ValueError("bad_pose:E needs a replay (measured pose) env")
        self.bad_pose.append(int(e))

    def inject_infra(self, e):
        raise ValueError("infra_env:E is a fake-backend fault (no per-env infrastructure injection in workload A)")

    def device_oom(self):
        torch = self.torch
        torch.cuda.set_per_process_memory_fraction(0.02)
        self._oom = torch.empty(int(8 * 1024 ** 3), dtype=torch.uint8, device="cuda")   # raises OutOfMemoryError

    def close(self):
        self.env.close()


def _episode_l2(rvm):
    class EpisodeL2(rvm.L2Monitor):
        """``run_vector_milestones.L2Monitor`` with per-episode accumulators (read and cleared at episode end)."""

        def take(self, ids):
            torch = self.torch
            idx = torch.as_tensor(ids, dtype=torch.int64, device=self.env.torch_device)
            rows = dict(code_monotone_violations=self.violations[idx].cpu().tolist(),
                        minimum_detF=self.min_det[idx].cpu().tolist(),
                        maximum_material_speed_m_s=self.max_speed[idx].cpu().tolist(),
                        nonfinite_steps=self.nonfinite_steps[idx].cpu().tolist(),
                        slot_overflow_max=self.slot_overflow_max[idx].cpu().tolist(),
                        invalid_any=self.invalid_any[idx].cpu().tolist())
            self.violations[idx] = 0
            self.min_det[idx] = float("inf")
            self.max_speed[idx] = 0.
            self.nonfinite_steps[idx] = 0
            self.slot_overflow_max[idx] = 0
            self.invalid_any[idx] = 0
            out = {}
            for k, e in enumerate(ids):
                row = {key: values[k] for key, values in rows.items()}
                row["status"] = "PASS" if (row["code_monotone_violations"] == 0 and row["minimum_detF"] > 0
                                           and row["nonfinite_steps"] == 0 and row["slot_overflow_max"] == 0) \
                    else "FAIL"
                out[e] = row
            return out
    return EpisodeL2


# ----------------------------------------------------------------------------- rollout

def git(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
    return out.stdout.strip() if out.returncode == 0 else None


def resumed_state(resume_dirs, N):
    records = []
    for d in resume_dirs:
        records += S.read_ledger(Path(d) / "ledger.jsonl")[0]
    return S.completed_quota(records, N)


def rollout(backend, run, cadence, numerical_invalid, ledger, faults, *, resume_dirs=(), max_global_steps=0,
            progress=None, declared_native=None, split="p1_4_v1"):
    N = run["num_envs"]
    completed, next_index = resumed_state(resume_dirs, N)
    schedule = RolloutSchedule(N, episode_steps=run["episode_steps"], episodes_per_env=run["episodes_per_env"],
                               offset_steps=run["offset_steps_per_env"], base_seed=run["base_seed"],
                               completed=completed, next_index=next_index)
    canary = [e["drive"] == "replay_canary" for e in run["envs"]]
    drives = [e["drive"] for e in run["envs"]]
    scene = [e["scene_seed"] for e in run["envs"]]
    sample_every, beat_every = cadence["sample_every_batch_steps"], cadence["heartbeat_every_batch_steps"]
    audit_every = cadence.get("l2_audit_every_steps", 0)

    segs = schedule.initial()
    wall = backend.reset([s["env"] for s in segs], [s["seed"] for s in segs])
    backend.set_episode_index([s["env"] for s in segs], [s["episode_index"] for s in segs])
    ledger.write("start_segments", global_step=0, segments=segs, wall_s=wall, kind="full", ids=list(range(N)),
                 completed_at_start=completed.tolist())
    ledger.write("census", where="after_full_reset", global_step=0, **backend.memory(census=True))
    audit = backend.l2_audit(0)
    if audit:
        ledger.write("l2_audit", global_step=0, **audit)
    global_step, walls, adapter_source = 0, [], None
    stopped_early = False
    while not schedule.done():
        t = time.perf_counter()
        status = backend.step()
        walls.append(time.perf_counter() - t)
        global_step += 1
        if progress is not None:
            progress["global_step"] = global_step
        if adapter_source is None:
            adapter_source = status.source
            ledger.write("adapter", source=status.source, infra_key=status.infra_key, info_keys=status.extra_keys)
        faults.after_step(global_step, backend)
        ends = schedule.on_step(global_step, status.terminated)
        if ends:
            evaluate = [end["env"] for end in ends if end["cause"] in ("full", "terminated")]
            results = backend.evaluate(evaluate) if evaluate else {}
            l2 = backend.l2_take([end["env"] for end in ends])
            restart = []
            for end in ends:
                e = end["env"]
                res = results.get(e)
                reason, invalid, infra = (int(status.termination_reason[e]), int(status.invalid_reason[e]),
                                          int(status.infra[e]))
                cls = S.classify_episode_end(cause=end["cause"], termination_reason=reason, invalid_reason=invalid,
                                             infra=infra, verdict=res and res["verdict"],
                                             numerical_invalid=numerical_invalid, split=split)
                rec = dict(end, end_class=cls, drive=drives[e], scene_seed=scene[e], evaluation=res, l2=l2.get(e),
                           termination_reason=reason,
                           termination_names=S.decode_bits(reason, S.TERMINATION_REASONS),
                           invalid_reason=invalid, invalid_names=S.decode_bits(invalid, S.INVALID_REASONS),
                           infra_flag=infra, adapter_source=status.source)
                if canary[e] and end["episode_index"] >= 0 and end["cause"] in ("full", "terminated"):
                    comp = backend.canary(e) if end["cause"] == "full" else None
                    declared = (declared_native or {}).get(str(scene[e]))
                    if comp is not None:
                        comp["declared_native_labelled"] = declared
                    ok = bool(comp and comp["codes_equal_native"] and comp["labelled"] == comp["native_labelled"]
                              and (declared is None or comp["labelled"] == declared)
                              and res and res["verdict"] == "PASS")
                    rec["canary"] = dict(comp or {}, status="PASS" if ok else "FAIL",
                                         reason=None if ok else ("terminated" if comp is None else "mismatch"))
                ledger.write("episode", **rec)
                schedule.record_end(e, cls, end["episode_index"])
                restart.append(e)
            new = schedule.begin(restart, global_step)
            kinds = {end["env"]: {"full": "episode_end", "terminated": "terminated", "forced": "staggered"}[end["cause"]]
                     for end in ends}
            wall = backend.reset([s["env"] for s in new], [s["seed"] for s in new])
            backend.set_episode_index([s["env"] for s in new], [s["episode_index"] for s in new])
            kind = sorted(set(kinds.values()))
            ledger.write("reset", global_step=global_step, ids=[s["env"] for s in new], segments=new, wall_s=wall,
                         kind=kind[0] if len(kind) == 1 else "+".join(kind))
            gc.collect()
            ledger.write("sample", kind="episode_end_post_gc", global_step=global_step,
                         **backend.memory(census=True))
        if global_step % sample_every == 0:
            ledger.write("sample", kind="periodic", global_step=global_step, **backend.memory())
        if audit_every and global_step % audit_every == 0:
            audit = backend.l2_audit(global_step)
            if audit:
                ledger.write("l2_audit", global_step=global_step, **audit)
        if global_step % beat_every == 0:
            recent = walls[-200:]
            ledger.write("heartbeat", global_step=global_step, median_step_s=float(np.median(recent)),
                         walls_s=[round(w, 6) for w in walls[-beat_every:]],
                         completed=schedule.completed.tolist())
        if max_global_steps and global_step >= max_global_steps:
            stopped_early = True
            break
    for end in schedule.in_flight(global_step):
        ledger.write("episode", **dict(end, end_class=S.TRUNCATED_RUN_END, drive=drives[end["env"]],
                                       scene_seed=scene[end["env"]], evaluation=None, l2=None))
    ledger.write("census", where="end", global_step=global_step, **backend.memory(census=True))
    return dict(global_steps=global_step, completed=schedule.completed.tolist(),
                completed_this_attempt=(schedule.completed - schedule.start_completed).tolist(),
                stopped_early_at_max_global_steps=stopped_early, adapter_source=adapter_source,
                step_wall_s=dict(n=len(walls), median=float(np.median(walls)) if walls else None,
                                 mean=float(np.mean(walls)) if walls else None,
                                 p95=float(np.percentile(walls, 95)) if walls else None))


def _uniform(base, e, step):
    """Counter-based uniform in [0, 1) for the forced-reset schedule of LR-B."""
    return (S.splitmix64((S.splitmix64(int(base) + 0xF0F0 + int(e)) + int(step)) & S.MASK64) >> 11) * 2. ** -53


def reset_stress(backend, run, cadence, numerical_invalid, ledger, faults, *, max_global_steps=0, progress=None,
                 split="termination_reason_v1"):
    """LR-B: canaries untouched; every other env force-reset with probability p per step (SplitMix64 schedule);
    after every reset the reset envs' state checksums are compared with those after the initial full reset."""
    N, steps, prob, base = run["num_envs"], run["batch_steps"], run["forced_reset_probability"], run["base_seed"]
    drives = [e["drive"] for e in run["envs"]]
    scene = [e["scene_seed"] for e in run["envs"]]
    untouched = [d == "replay_canary" for d in drives]
    canary_every = run.get("canary_sample_every_steps", 100)
    seg_seq = np.zeros(N, np.int64)
    current, seg_steps = [None] * N, np.zeros(N, np.int64)

    def begin(ids, global_step):
        out = []
        for e in ids:
            index = int(seg_seq[e])
            seg = dict(env=int(e), segment=index, kind="episode", episode_index=index,
                       seed=int(S.episode_seed(base, e, index)), global_step_start=int(global_step), beyond_quota=False)
            seg_seq[e] += 1
            seg_steps[e] = 0
            current[e] = seg
            out.append(seg)
        return out

    segs = begin(range(N), 0)
    wall = backend.reset(list(range(N)), [g["seed"] for g in segs])
    backend.set_episode_index(list(range(N)), [g["episode_index"] for g in segs])
    ledger.write("start_segments", global_step=0, segments=segs, wall_s=wall, kind="full", ids=list(range(N)))
    reference = backend.state_checksums()
    ledger.write("reset_check_reference", global_step=0, arrays=sorted(reference))
    ledger.write("census", where="after_full_reset", global_step=0, **backend.memory(census=True))
    walls, checks, mismatched, canary_rows, forced_total = [], 0, [], [], 0
    gstep = 0
    while gstep < steps:
        t = time.perf_counter()
        status = backend.step()
        walls.append(time.perf_counter() - t)
        gstep += 1
        seg_steps += 1
        if gstep == 1:
            ledger.write("adapter", source=status.source, infra_key=status.infra_key, info_keys=status.extra_keys)
        if progress is not None:
            progress["global_step"] = gstep
        faults.after_step(gstep, backend)
        ends = []
        for e in range(N):
            if status.terminated[e]:
                ends.append((e, "terminated"))
            elif not untouched[e] and _uniform(base, e, gstep) < prob:
                ends.append((e, "forced"))
        if ends:
            evaluate = [e for e, c in ends if c == "terminated"]
            results = backend.evaluate(evaluate) if evaluate else {}
            l2 = backend.l2_take([e for e, _ in ends])
            for e, cause in ends:
                reason, invalid, infra = (int(status.termination_reason[e]), int(status.invalid_reason[e]),
                                          int(status.infra[e]))
                res = results.get(e)
                cls = S.classify_episode_end(cause=cause, termination_reason=reason, invalid_reason=invalid,
                                             infra=infra, verdict=res and res["verdict"],
                                             numerical_invalid=numerical_invalid, split=split)
                ledger.write("episode", **dict(current[e], cause=cause, steps=int(seg_steps[e]),
                                                global_step_end=gstep, end_class=cls, drive=drives[e],
                                                scene_seed=scene[e], evaluation=res, l2=l2.get(e),
                                                termination_reason=reason,
                                                termination_names=S.decode_bits(reason, S.TERMINATION_REASONS),
                                                invalid_reason=invalid,
                                                invalid_names=S.decode_bits(invalid, S.INVALID_REASONS),
                                                infra_flag=infra, adapter_source=status.source))
            ids = [e for e, _ in ends]
            forced_total += sum(1 for _, c in ends if c == "forced")
            new = begin(ids, gstep)
            wall = backend.reset(ids, [g["seed"] for g in new])
            backend.set_episode_index(ids, [g["episode_index"] for g in new])
            check = backend.reset_check(ids, reference)
            checks += len(ids)
            bad = {e: v for e, v in check.items() if v}
            if bad:
                mismatched.append(dict(global_step=gstep, envs=bad))
            kinds = sorted({c for _, c in ends})
            ledger.write("reset", global_step=gstep, ids=ids, segments=new, wall_s=wall,
                         kind="+".join("staggered" if k == "forced" else k for k in kinds),
                         reset_check_mismatch={str(e): v for e, v in bad.items()})
        if gstep % canary_every == 0 or gstep == steps:
            for e in [e for e in range(N) if untouched[e]]:
                row = backend.canary_sample(e)
                if row is not None:
                    canary_rows.append(row)
                    ledger.write("canary_sample", global_step=gstep, **row)
        if gstep % cadence["sample_every_batch_steps"] == 0:
            ledger.write("sample", kind="periodic", global_step=gstep, **backend.memory())
            gc.collect()
            ledger.write("sample", kind="post_gc", global_step=gstep, **backend.memory(census=True))
        if gstep % cadence["heartbeat_every_batch_steps"] == 0:
            ledger.write("heartbeat", global_step=gstep, median_step_s=float(np.median(walls[-200:])),
                         walls_s=[round(w, 6) for w in walls[-cadence["heartbeat_every_batch_steps"]:]])
        if max_global_steps and gstep >= max_global_steps:
            break
    for e in range(N):
        ledger.write("episode", **dict(current[e], cause="run_end", steps=int(seg_steps[e]), global_step_end=gstep,
                                        end_class=S.TRUNCATED_RUN_END, drive=drives[e], scene_seed=scene[e],
                                        evaluation=None, l2=None))
    ledger.write("census", where="end", global_step=gstep, **backend.memory(census=True))
    limit = run.get("canary_max_different_code_particles", 1)
    return dict(global_steps=gstep, forced_resets=forced_total, reset_checks=checks,
                reset_check_mismatches=len(mismatched), reset_check_mismatch_detail=mismatched[:20],
                reset_check_status="PASS" if checks and not mismatched else ("FAIL" if mismatched else "NOT_RUN"),
                canary_samples=len(canary_rows),
                canary_max_different_code_particles=max((r["different_code_particles"] for r in canary_rows),
                                                        default=None),
                canary_max_position_diff_m=max((r["max_position_diff_m"] for r in canary_rows), default=None),
                canary_status="PASS" if canary_rows and all(r["different_code_particles"] <= limit
                                                            for r in canary_rows) else "FAIL",
                canary_rule=f"different code particles <= {limit} at every sample (M0 history: 0-1)",
                stopped_early_at_max_global_steps=bool(max_global_steps and gstep < steps),
                step_wall_s=dict(n=len(walls), median=float(np.median(walls)) if walls else None,
                                 p95=float(np.percentile(walls, 95)) if walls else None))


def rebuild_cycles(make_backend, run, ledger, faults, *, gpu_index=None, progress=None):
    """LR-C: ``cycles`` x (build -> full reset -> ``steps_per_cycle`` steps -> close) in ONE process; census and
    memory after every build and after every close (post gc)."""
    rows = []
    for c in range(run["cycles"]):
        ledger.write("heartbeat", phase=f"cycle_{c}_build", global_step=None)
        t = time.perf_counter()
        backend = make_backend(c)
        build_s = time.perf_counter() - t
        seeds = [int(S.episode_seed(run["base_seed"], e, c)) for e in range(backend.N)]
        backend.reset(list(range(backend.N)), seeds)
        backend.set_episode_index(list(range(backend.N)), [c] * backend.N)
        open_census = backend.memory(census=True)
        ledger.write("census", where=f"cycle_{c}_open", global_step=c, cycle=c, **open_census)
        walls = []
        for k in range(run["steps_per_cycle"]):
            t = time.perf_counter()
            status = backend.step()
            walls.append(time.perf_counter() - t)
            if progress is not None:
                progress["global_step"] = c * run["steps_per_cycle"] + k + 1
            faults.after_step(c * run["steps_per_cycle"] + k + 1, backend)
            if k % 5 == 4:
                ledger.write("heartbeat", global_step=c * run["steps_per_cycle"] + k + 1,
                             median_step_s=float(np.median(walls)), walls_s=[round(w, 6) for w in walls[-5:]])
        l2 = backend.l2_take(list(range(backend.N)))
        terminated = [int(e) for e in np.flatnonzero(status.terminated)]
        probe_memory, probe_vram = type(backend).process_memory, type(backend).process_own_vram_mib
        t = time.perf_counter()
        backend.close()
        close_s = time.perf_counter() - t
        del backend                      # no bound method or other reference may keep the closed env alive
        gc.collect()
        after = probe_memory(census=True)
        own_after = probe_vram(gpu_index)
        row = dict(cycle=c, build_s=build_s, close_s=close_s, steps=len(walls), median_step_s=float(np.median(walls)),
                   l2_pass=all(v.get("status") == "PASS" for v in l2.values()), terminated_envs=terminated,
                   live_device_hashgrids_open=open_census.get("live_device_hashgrids"),
                   live_device_hashgrids_after_close=after.get("live_device_hashgrids"),
                   own_vram_mib_after_close=own_after)
        rows.append(row)
        ledger.write("cycle", **row)
        ledger.write("sample", kind="post_close", global_step=c, cycle=c, own_vram_mib=own_after, **after)
    return dict(cycles=len(rows), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs" / "rl_benchmark" / "long_rollout_v1.json"))
    parser.add_argument("--run", required=True)
    parser.add_argument("--attempt-dir", required=True)
    parser.add_argument("--resume-from", action="append", default=[])
    parser.add_argument("--backend", choices=("trim", "fake"), default="trim")
    parser.add_argument("--max-global-steps", type=int, default=0, help="smoke only: stop early (REDUCED run)")
    parser.add_argument("--fake-step-s", type=float, default=0.)
    parser.add_argument("--fake-info-style", choices=("legacy", "termination_reason"), default="legacy")
    parser.add_argument("--fake-strict", action="store_true")
    parser.add_argument("--gpu-index", type=int, help="physical nvidia-smi index (LR-C own-VRAM samples)")
    args = parser.parse_args()
    faulthandler.enable()
    attempt_dir = Path(args.attempt_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    ledger = S.Ledger(attempt_dir / "ledger.jsonl")
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    run = dict(config["runs"][args.run])
    common = config.get("common", {})
    cadence = dict(config["cadence"], l2_audit_every_steps=common.get("l2_audit_every_steps", 0))
    numerical = config.get("classification", {}).get("numerical_invalid_reasons", list(S.DEFAULT_NUMERICAL_INVALID))
    split = config.get("classification", {}).get("split", "p1_4_v1")
    faults = FaultInjector(parse_faults(os.environ.get("RCS_FAULT_INJECT")), ledger)
    ledger.write("start", run=args.run, config=os.path.relpath(args.config, ROOT) if args.config.startswith(str(ROOT))
                 else args.config, config_sha256=hashlib.sha256(config_bytes).hexdigest(), backend=args.backend,
                 pid=os.getpid(), argv=sys.argv, git_head=git("rev-parse", "HEAD"),
                 git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
                 cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), python=sys.version.split()[0],
                 resumed_from=args.resume_from, fault_inject=os.environ.get("RCS_FAULT_INJECT"),
                 max_global_steps=args.max_global_steps, reduced=bool(args.max_global_steps), run_config=run,
                 split=split, mode=run.get("mode", "rollout"))
    backend, status, t0, progress = None, 1, time.time(), dict(global_step=None)
    try:
        ledger.write("heartbeat", phase="build_start", global_step=None)
        mode = run.get("mode", "rollout")

        def make_backend(tag=None):
            if args.backend == "fake":
                return FakeBackend(run, info_style=args.fake_info_style, strict=args.fake_strict,
                                   step_s=args.fake_step_s)
            return TrimBackend(run, common, attempt_dir / ("env" if tag is None else f"env_cycle_{tag:02d}"))
        if mode == "rebuild_cycles":
            result = rebuild_cycles(make_backend, run, ledger, faults, gpu_index=args.gpu_index, progress=progress)
        else:
            backend = make_backend()
            ledger.write("built", receipt=backend.receipt(), **backend.memory(census=True))
            if mode == "reset_stress":
                result = reset_stress(backend, run, cadence, numerical, ledger, faults,
                                      max_global_steps=args.max_global_steps, progress=progress, split=split)
            else:
                result = rollout(backend, run, cadence, numerical, ledger, faults, resume_dirs=args.resume_from,
                                 max_global_steps=args.max_global_steps, progress=progress,
                                 declared_native=None if args.backend == "fake"
                                 else config.get("canary_rule", {}).get("native_labelled"), split=split)
            backend.close()
            backend = None
        ledger.write("end", wall_s=time.time() - t0, **result)
        status = 0
    except BaseException as error:   # recorded, then re-raised as a non-zero exit (never hidden)
        if isinstance(error, (SystemExit, KeyboardInterrupt)):
            raise
        category, kind = env_exception_category(error)
        ledger.write("infra_error", exc_type=type(error).__name__, message=str(error)[:4000],
                     errno=getattr(error, "errno", None), traceback=traceback.format_exc()[-6000:],
                     global_step=progress["global_step"], env_category=category, env_kind=kind)
        traceback.print_exc()
        status = 1
    finally:
        ledger.close()
    sys.exit(status)


if __name__ == "__main__":
    main()
