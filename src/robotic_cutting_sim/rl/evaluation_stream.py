"""Opt-in per-env evaluation stream and the shared frozen-evaluator call of the vector envs (P1-2b task 4).

The Native WP-17 episodes stream one physical sample (positions, velocities, packed codes of every particle) at
reset and then every ``--record-every 10`` control steps, plus one at termination/truncation
(``scripts/run_wp10_live.py`` ``EpisodeRecorder.capture``; ``recording.PhysicsStreamWriter``): 121 samples over
the 1200-step TRIM episode, which ``seeded_trim.evaluate_recording`` hands to the frozen ``goals.evaluate_goal``.

``EvaluationStream`` records the same samples for every env of a batch, on the device, with no host sync: one
Warp kernel over all particles copies the solver's LOCAL ``x``, ``v`` (float32) and ``code`` (int32) of the envs
whose sample is due into a preallocated ``[S, P]`` slot (the slot index is the env's own sample counter, read on
the device), and one kernel over the envs records the step and advances the counter. A sample is due at reset
(step 0), when ``step_count % every == 0`` and when the env is done. Storing the solver's own float32 local state is
lossless; the WORLD float64 transform (``scene_frames.transform_points``, as the Native ``history_state`` and
``snapshot_world``) happens on the host in ``evaluate()``. Memory: ``S x P x 28`` bytes (S = max_steps // every + 2;
about 350 MB per 1e5-particle env for a 1200-step episode). Device storage only; there is no host-pinned mode.

``evaluate_frozen`` is the one place both backends build the evaluator input. Modes (``evaluation_mode`` in each
result): ``two_frame_reset_and_current_v1`` (the default, unchanged since P1-2: reset reference state at t = 0 plus
the current state) and ``evaluation_stream_v1`` (the recorded samples plus the current state when it is newer).

``integrity_failure_result`` (``INTEGRITY_FAILURE_VERSION``, P1 consolidation) is what ``evaluate()`` of both backends
returns for an env whose episode is not numerically valid: its material ``invalid_reason`` has a non-finite bit
(``NONFINITE_STATE`` / ``NONFINITE_WRENCH``) or a host snapshot / stream sample it would score is non-finite
(``NonFiniteSnapshot``). The frozen evaluator is NOT called for that env (it cannot take non-finite positions):
verdict ``FAIL_INTEGRITY`` (the frozen evaluator's own word for a failed integrity layer), the decoded material and
termination reasons, and no physics metrics. The other envs of the call are evaluated normally.
"""
from dataclasses import asdict, dataclass
import hashlib
import time

import numpy as np
import warp as wp

STREAM_VERSION = "evaluation_stream_v1"
DEFAULT_MODE = "two_frame_reset_and_current_v1"
NATIVE_RECORD_EVERY = 10
INTEGRITY_FAILURE_VERSION = "nonfinite_env_integrity_failure_v1"
# material invalid_reason bits that make an episode numerically invalid (batched.material_batch INVALID_NONFINITE_*)
NONFINITE_MATERIAL_BITS = 1 | 16


class NonFiniteSnapshot(ValueError):
    """A host snapshot or stream sample of one env holds non-finite positions or velocities (raised before the
    WORLD transform; a ``ValueError`` like ``scene_frames.transform_points``'s own refusal)."""


def finite_or_raise(x, v, what):
    if not (np.isfinite(x).all() and np.isfinite(v).all()):
        raise NonFiniteSnapshot(f"Expected finite (...,3) points: non-finite {what}")


@dataclass(frozen=True)
class EvaluationStreamConfig:
    every: int = NATIVE_RECORD_EVERY
    capacity: int | None = None      # samples per env; None: max_episode_steps // every + 2
    storage: str = "device"

    def validate(self):
        if isinstance(self.every, bool) or not isinstance(self.every, int) or self.every < 1:
            raise ValueError("evaluation_stream.every must be a positive int")
        if self.capacity is not None and (not isinstance(self.capacity, int) or self.capacity < 2):
            raise ValueError("evaluation_stream.capacity must be an int >= 2 or None")
        if self.storage != "device":
            from .vector_env import UnsupportedVectorConfiguration
            raise UnsupportedVectorConfiguration("UNSUPPORTED_OPTION", f"evaluation_stream storage "
                                                 f"{self.storage!r}; only 'device'")
        return self


def stream_config(value):
    """``open_vector_env(evaluation_stream=...)``: None/False (off), True (defaults), a dict or a config."""
    if value is None or value is False:
        return None
    if value is True:
        return EvaluationStreamConfig().validate()
    if isinstance(value, EvaluationStreamConfig):
        return value.validate()
    if isinstance(value, dict):
        return EvaluationStreamConfig(**value).validate()
    raise ValueError("evaluation_stream must be None, a bool, a dict or an EvaluationStreamConfig")


@wp.kernel
def _capture_particles(mask: wp.array(dtype=wp.int32), bank: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                       steps: wp.array2d(dtype=wp.int64), step_count: wp.array(dtype=wp.int64),
                       env_of_particle: wp.array(dtype=int), capacity: int, num_envs: int, x: wp.array(dtype=wp.vec3),
                       v: wp.array(dtype=wp.vec3), code: wp.array(dtype=wp.int32),
                       out_x: wp.array2d(dtype=wp.vec3), out_v: wp.array2d(dtype=wp.vec3),
                       out_code: wp.array2d(dtype=wp.int32)):
    p = wp.tid()
    e = env_of_particle[p]
    if mask[e] == 0:
        return
    b = bank[e]
    s = count[b * num_envs + e]
    if s >= capacity:
        return
    row = b * capacity + s
    if s > 0:
        if steps[row - 1, e] == step_count[e]:
            return      # already sampled at this step
    out_x[row, p] = x[p]
    out_v[row, p] = v[p]
    out_code[row, p] = code[p]


@wp.kernel
def _capture_envs(mask: wp.array(dtype=wp.int32), bank: wp.array(dtype=wp.int32), count: wp.array(dtype=wp.int32),
                  steps: wp.array2d(dtype=wp.int64), step_count: wp.array(dtype=wp.int64), capacity: int,
                  num_envs: int, dropped: wp.array(dtype=wp.int32)):
    e = wp.tid()
    if mask[e] == 0:
        return
    b = bank[e]
    c = b * num_envs + e
    s = count[c]
    row = b * capacity + s
    if s > 0:
        if steps[row - 1, e] == step_count[e]:
            return
    if s >= capacity:
        dropped[c] = dropped[c] + 1
        return
    steps[row, e] = step_count[e]
    count[c] = s + 1


@wp.kernel
def _capture_terminal(mask: wp.array(dtype=wp.int32), env_of_particle: wp.array(dtype=int), x: wp.array(dtype=wp.vec3),
                      v: wp.array(dtype=wp.vec3), code: wp.array(dtype=wp.int32), out_x: wp.array(dtype=wp.vec3),
                      out_v: wp.array(dtype=wp.vec3), out_code: wp.array(dtype=wp.int32)):
    p = wp.tid()
    if mask[env_of_particle[p]] != 0:
        out_x[p] = x[p]
        out_v[p] = v[p]
        out_code[p] = code[p]


class EvaluationStream:
    """Device sample store of one batch (module docstring). ``material``: a ``BatchedCuttingMaterial``.

    ``banks=2`` (only with terminal capture under same-step autoreset, review F5): each env writes into its active
    bank; ``flip(mask)`` at an autoreset keeps the finished episode's samples in the other bank until that env
    finishes its next episode. Memory doubles."""

    def __init__(self, material, max_episode_steps, torch_device, config, banks=1):
        import torch
        self.torch, self.config = torch, config
        self.material = material
        s = material.solver
        self.num_envs, self.particles = int(material.num_envs), int(s.total_particles)
        self.every = int(config.every)
        self.capacity = int(config.capacity or (int(max_episode_steps) // self.every + 2))
        if banks not in (1, 2):
            raise ValueError("banks is 1 or 2")
        self.banks = int(banks)
        dev = torch_device
        S, P, N, B = self.capacity, self.particles, self.num_envs, self.banks
        self.x = torch.zeros((B * S, P, 3), dtype=torch.float32, device=dev)
        self.v = torch.zeros((B * S, P, 3), dtype=torch.float32, device=dev)
        self.code = torch.zeros((B * S, P), dtype=torch.int32, device=dev)
        self.steps = torch.full((B * S, N), -1, dtype=torch.int64, device=dev)
        self.count = torch.zeros(B * N, dtype=torch.int32, device=dev)
        self.dropped = torch.zeros(B * N, dtype=torch.int32, device=dev)
        self.bank = torch.zeros(N, dtype=torch.int32, device=dev)
        self._mask = torch.zeros(N, dtype=torch.int32, device=dev)
        self.bytes = int(sum(t.element_size() * t.numel() for t in (self.x, self.v, self.code, self.steps)))
        self._wp = dict(x=wp.from_torch(self.x, dtype=wp.vec3), v=wp.from_torch(self.v, dtype=wp.vec3),
                        code=wp.from_torch(self.code), steps=wp.from_torch(self.steps),
                        count=wp.from_torch(self.count), dropped=wp.from_torch(self.dropped),
                        mask=wp.from_torch(self._mask), bank=wp.from_torch(self.bank))
        self.captures = 0

    def due(self, step_count, done=None):
        """``bool [N]``: a sample is due at ``step_count % every == 0`` or when done."""
        due = (step_count % self.every) == 0
        return due if done is None else due | done

    def capture(self, mask):
        """Record the current material state of the envs in ``mask`` (bool ``[N]``), on the current stream."""
        s, w, N = self.material.solver, self._wp, self.num_envs
        self._mask.copy_(mask.to(self.torch.int32))
        device = self.material.device
        wp.launch(_capture_particles, self.particles, [w["mask"], w["bank"], w["count"], w["steps"],
                  self.material.step_count, s.env_of_particle, self.capacity, N, s.x, s.v, s.code, w["x"], w["v"],
                  w["code"]], device=device)
        wp.launch(_capture_envs, N, [w["mask"], w["bank"], w["count"], w["steps"], self.material.step_count,
                  self.capacity, N, w["dropped"]], device=device)
        self.captures += 1

    def _active(self, mask):
        """``bool [B*N]`` counter mask of the active bank of the envs in ``mask``."""
        torch = self.torch
        banks = torch.arange(self.banks, device=mask.device, dtype=torch.int32)[:, None]
        return ((banks == self.bank[None, :]) & mask[None, :]).reshape(-1)

    def reset(self, mask):
        """Forget the active-bank samples of the envs in ``mask`` (call before capturing their reset state)."""
        torch = self.torch
        active = self._active(mask)
        self.count.copy_(torch.where(active, torch.zeros_like(self.count), self.count))
        self.dropped.copy_(torch.where(active, torch.zeros_like(self.dropped), self.dropped))
        rows = active.view(self.banks, self.num_envs)                      # [B, N]
        rows = rows[:, None, :].expand(self.banks, self.capacity, self.num_envs).reshape(-1, self.num_envs)
        self.steps.copy_(torch.where(rows, torch.full_like(self.steps, -1), self.steps))

    def flip(self, mask):
        """Autoreset of the envs in ``mask``: their finished samples stay in the current bank, new ones go to the
        other bank (needs ``banks=2``)."""
        if self.banks != 2:
            raise RuntimeError("flip() needs banks=2")
        self.bank.copy_(self.torch.where(mask, 1 - self.bank, self.bank))

    def host_frames(self, e, T_world_local, control_dt, completed=False):
        """Host WORLD samples of env ``e`` (``snapshot_world`` dicts, time order) of its active bank, or of the
        other bank (``completed=True``: the episode that ended at its last autoreset). Host reads: evaluation only."""
        from ..scene_frames import transform_points
        pk = self.material.packed
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e + 1])
        b = int(self.bank[e].item())
        if completed:
            if self.banks != 2:
                raise RuntimeError("completed-episode samples need banks=2")
            b = 1 - b
        c, base = b * self.num_envs + e, b * self.capacity
        n = int(self.count[c].item())
        # .clone() before .numpy(): on CPU .cpu() is the same tensor and the arrays would alias the store
        steps = self.steps[base:base + n, e].cpu().clone().numpy()
        x = self.x[base:base + n, p0:p1].cpu().clone().numpy()
        v = self.v[base:base + n, p0:p1].cpu().clone().numpy()
        code = self.code[base:base + n, p0:p1].cpu().clone().numpy()
        T = np.asarray(T_world_local, float)
        finite_or_raise(x, v, f"stream sample of env {e}")
        return [dict(positions=transform_points(T, x[k]), velocities=v[k].astype(np.float64) @ T[:3, :3].T,
                     codes=np.asarray(code[k]), step_count=int(steps[k]), time_s=int(steps[k]) * float(control_dt))
                for k in range(n)], int(self.dropped[c].item())

    def receipt(self):
        return {"version": STREAM_VERSION, "config": asdict(self.config), "capacity_samples_per_env": self.capacity,
                "banks": self.banks, "device_bytes": self.bytes, "particles": self.particles,
                "due": f"reset (step 0), step_count % {self.every} == 0, done",
                "stored": "solver LOCAL x, v (float32), code (int32); WORLD float64 transform on the host at evaluate",
                "native_reference": "WP-17 EpisodeRecorder --record-every 10 (121 samples / 1200 steps)",
                "host_sync": "none per step (device kernels; counters read only in evaluate())"}


class TerminalCapture:
    """Opt-in terminal snapshot of envs that finish inside ``step()`` under ``autoreset="same_step"`` (review F5).

    Before the masked autoreset, one Warp kernel copies the finished envs' solver ``x``, ``v`` and ``code`` into a
    ``[P]`` device store, and their step count / episode index are recorded (no host sync). ``evaluate(env_ids,
    episode="last_completed")`` scores that state (with the evaluation stream: the finished episode's samples kept
    in the stream's other bank). Memory: ``P x 28`` bytes."""

    def __init__(self, material, torch_device):
        import torch
        self.torch, self.material = torch, material
        P, N = int(material.solver.total_particles), int(material.num_envs)
        self.x = torch.zeros((P, 3), dtype=torch.float32, device=torch_device)
        self.v = torch.zeros((P, 3), dtype=torch.float32, device=torch_device)
        self.code = torch.zeros(P, dtype=torch.int32, device=torch_device)
        self.step = torch.full((N,), -1, dtype=torch.int64, device=torch_device)
        self.episode = torch.full((N,), -1, dtype=torch.int64, device=torch_device)
        self.captures = torch.zeros(N, dtype=torch.int64, device=torch_device)
        self._mask = torch.zeros(N, dtype=torch.int32, device=torch_device)
        self._wp = dict(x=wp.from_torch(self.x, dtype=wp.vec3), v=wp.from_torch(self.v, dtype=wp.vec3),
                        code=wp.from_torch(self.code), mask=wp.from_torch(self._mask))
        self.bytes = int(sum(t.element_size() * t.numel() for t in (self.x, self.v, self.code)))

    def capture(self, mask, step_count, episode):
        torch, s, w = self.torch, self.material.solver, self._wp
        self._mask.copy_(mask.to(torch.int32))
        wp.launch(_capture_terminal, int(s.total_particles), [w["mask"], s.env_of_particle, s.x, s.v, s.code, w["x"],
                  w["v"], w["code"]], device=self.material.device)
        self.step.copy_(torch.where(mask, step_count.to(torch.int64), self.step))
        self.episode.copy_(torch.where(mask, episode.to(torch.int64), self.episode))
        self.captures.add_(mask.to(torch.int64))

    def host_snapshot(self, e, T_world_local, control_dt):
        """WORLD snapshot of env ``e``'s last terminal state, or None when it never finished under autoreset."""
        from ..scene_frames import transform_points
        step = int(self.step[e].item())
        if step < 0:
            return None
        pk = self.material.packed
        p0, p1 = int(pk.particle_offset[e]), int(pk.particle_offset[e + 1])
        T = np.asarray(T_world_local, float)
        x = self.x[p0:p1].cpu().clone().numpy()
        v = self.v[p0:p1].cpu().clone().numpy().astype(np.float64)
        finite_or_raise(x, v, f"terminal state of env {e}")
        return dict(positions=transform_points(T, x), velocities=v @ T[:3, :3].T,
                    codes=self.code[p0:p1].cpu().clone().numpy(), step_count=step, time_s=step * float(control_dt),
                    episode=int(self.episode[e].item()))

    def receipt(self):
        return {"enabled": True, "version": "terminal_capture_v1", "device_bytes": self.bytes,
                "captured": "solver LOCAL x, v, code of envs done in step(), before the same-step autoreset",
                "evaluate": "evaluate(env_ids, episode='last_completed')", "host_sync": "none per step"}


def evaluate_frozen(task, reference, final, series=None, *, mode=DEFAULT_MODE, env=None, extra=None):
    """Frozen ``task.evaluate_goal`` (``goals.evaluate_goal``) of one env; the one input builder of both backends.

    ``reference``: ``evaluation_reference(e)``; ``final``: ``snapshot_world(e)`` (current state); ``series``: earlier
    ``snapshot_world``-like samples in time order. A t = 0 frame is the reference state (positions, zero velocities
    and codes) unless ``series`` starts with a step-0 sample (stream mode). Samples at or after the final step are
    replaced by the final state; times increase strictly."""
    from .vector_env import summarize_evaluation
    start = time.perf_counter()
    n = len(reference["mass_kg"])
    samples = [s for s in (series or []) if s["step_count"] < final["step_count"]]
    if final["step_count"] > 0 or not samples:
        samples = samples + [final]
    reference_frame = mode == DEFAULT_MODE or not samples or samples[0]["step_count"] != 0
    frames = [s for s in samples if s["step_count"] > 0] if reference_frame else samples
    times = ([0.] if reference_frame else []) + [s["time_s"] for s in frames]
    data = dict(reference, time_s=np.asarray(times, float),
                positions=np.stack(([reference["reference_positions"]] if reference_frame else [])
                                   + [s["positions"] for s in frames]),
                velocities=np.stack(([np.zeros((n, 3))] if reference_frame else [])
                                    + [np.asarray(s["velocities"], float) for s in frames]),
                codes=np.stack(([np.zeros(n, np.int32)] if reference_frame else [])
                               + [np.asarray(s["codes"]).astype(np.int32) for s in frames]),
                code_encoding="packed2", max_cuts=int(task.max_cuts), particle_spacing_m=task.particle_spacing_m)
    result = task.evaluate_goal(data)
    out = dict(env=env, task_id=task.task_id, step_count=final["step_count"], time_s=final["time_s"],
               frames=len(data["time_s"]), evaluation_mode=mode, reference_frame_at_t0=reference_frame,
               final_codes_sha256=hashlib.sha256(np.ascontiguousarray(final["codes"])).hexdigest(),
               labelled=int(np.count_nonzero(final["codes"])), summary=summarize_evaluation(result), result=result,
               evaluator="goals.evaluate_goal via ConfiguredTaskSpec.evaluate_goal (frozen goal "
                         f"{task.goal.get('goal_id')}); host snapshot at episode end")
    out.update(extra or {})
    out["wall_s"] = time.perf_counter() - start
    return out


def integrity_failure_result(task, *, env, step_count, control_dt, mode, cause, invalid_reason=None,
                             termination_reason=None, extra=None):
    """Structured ``evaluate()`` result of an env whose episode is not numerically valid (module docstring).

    ``invalid_reason`` / ``termination_reason``: the env's current material / termination bits (None when not
    captured, e.g. ``episode="last_completed"`` after the same-step autoreset cleared them). The frozen evaluator is
    not called; ``result`` carries only the integrity layer, labelled as this module's, never as the evaluator's."""
    from .termination import decode_material_invalid, decode_termination
    from .vector_env import summarize_evaluation
    checks = {"finite_material_state": False}
    result = {"verdict": "FAIL_INTEGRITY", "success": False, "task_id": task.task_id,
              "integrity": {"status": "FAIL", "checks": checks, "cause": cause},
              "outcome": {"status": "NOT_EVALUATED", "checks": None, "metrics": {}},
              "source": f"rl.evaluation_stream.{INTEGRITY_FAILURE_VERSION} (frozen evaluator not called: "
                        "non-finite material state)"}
    out = dict(env=env, task_id=task.task_id, step_count=int(step_count), time_s=int(step_count) * float(control_dt),
               frames=0, evaluation_mode=mode, reference_frame_at_t0=None, final_codes_sha256=None, labelled=None,
               verdict="FAIL_INTEGRITY", integrity_failure=INTEGRITY_FAILURE_VERSION, integrity_cause=cause,
               invalid_reason=None if invalid_reason is None else int(invalid_reason),
               invalid_reasons=None if invalid_reason is None else decode_material_invalid(invalid_reason),
               termination_reason=None if termination_reason is None else int(termination_reason),
               termination_reasons=None if termination_reason is None else decode_termination(termination_reason),
               physics_metrics=None, summary=summarize_evaluation(result), result=result,
               evaluator="not called (integrity failure: non-finite material state); no physics metrics")
    out.update(extra or {})
    out["wall_s"] = 0.
    return out


__all__ = ["EvaluationStream", "EvaluationStreamConfig", "TerminalCapture", "evaluate_frozen", "stream_config", "STREAM_VERSION",
           "DEFAULT_MODE", "NATIVE_RECORD_EVERY", "INTEGRITY_FAILURE_VERSION", "NONFINITE_MATERIAL_BITS",
           "NonFiniteSnapshot", "finite_or_raise", "integrity_failure_result"]
