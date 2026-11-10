"""P1-4 long-rollout episode schedule (host state machine; NumPy only).

Plan ``docs/reports/rl_ready/P1_5_PLAN.md`` §6.2 (LR-A): env ``k`` first runs a *stagger* segment that is
force-reset at batch step ``offset * k`` (env 0 has none), so later resets fall while the other envs are
mid-episode; afterwards every env runs full episodes back to back. An episode ends when the env is terminated
(``StepStatus.terminated``) or reaches ``episode_steps``; the env is then partially reset and starts the next
schedule episode. The run is done when every env has completed ``episodes_per_env`` quota episodes
(``supervision.QUOTA_CLASSES``); envs that finish early keep running further (``beyond_quota``) episodes, and the
episodes still in flight at the end (including ones started at the last step) are ``TRUNCATED_RUN_END``,
so every started segment has exactly one end record.

Seeds: episode ``i`` of env ``e`` uses ``supervision.episode_seed(base_seed, e, i)``; a resumed attempt continues
at the next schedule index of each env (read from the previous attempts' ledgers), so its seeds and policy
streams are the ones the uninterrupted run would have used.
"""
import numpy as np

from .supervision import QUOTA_CLASSES, episode_seed

STAGGER, EPISODE = "stagger", "episode"
STAGGER_SEED_OFFSET = 1 << 32      # stagger segment seeds never collide with episode seeds


class RolloutSchedule:
    def __init__(self, num_envs, *, episode_steps, episodes_per_env, offset_steps, base_seed, completed=None,
                 next_index=None):
        self.N, self.episode_steps, self.quota = int(num_envs), int(episode_steps), int(episodes_per_env)
        self.offset_steps, self.base_seed = int(offset_steps), int(base_seed)
        self.completed = np.zeros(self.N, np.int64) if completed is None else np.asarray(completed, np.int64).copy()
        self.next_index = np.zeros(self.N, np.int64) if next_index is None else np.asarray(next_index, np.int64).copy()
        self.steps = np.zeros(self.N, np.int64)
        self.segment_seq = np.zeros(self.N, np.int64)
        self.current = [None] * self.N
        self.start_completed = self.completed.copy()

    def offset(self, e):
        return self.offset_steps * int(e)

    def _segment(self, e, kind, global_step):
        if kind == STAGGER:
            index, seed = -1, episode_seed(self.base_seed, e, STAGGER_SEED_OFFSET + int(self.next_index[e]))
        else:
            index = int(self.next_index[e])
            seed = episode_seed(self.base_seed, e, index)
            self.next_index[e] += 1
        seg = dict(env=int(e), segment=int(self.segment_seq[e]), kind=kind, episode_index=index, seed=int(seed),
                   global_step_start=int(global_step), beyond_quota=bool(kind == EPISODE
                                                                          and self.completed[e] >= self.quota))
        self.segment_seq[e] += 1
        self.steps[e] = 0
        self.current[e] = seg
        return seg

    def initial(self):
        """Segments of the initial full reset (global step 0)."""
        return [self._segment(e, STAGGER if self.offset(e) > 0 else EPISODE, 0) for e in range(self.N)]

    def on_step(self, global_step, terminated):
        """Advance every env by one batch step; return the segments that ended with their cause."""
        self.steps += 1
        ends = []
        for e in range(self.N):
            seg = self.current[e]
            cause = None
            if bool(terminated[e]):
                cause = "terminated"
            elif seg["kind"] == EPISODE and self.steps[e] >= self.episode_steps:
                cause = "full"
            elif seg["kind"] == STAGGER and global_step >= self.offset(e):
                cause = "forced"
            if cause:
                ends.append(dict(seg, cause=cause, steps=int(self.steps[e]), global_step_end=int(global_step)))
        return ends

    def record_end(self, e, end_class, episode_index):
        if episode_index >= 0 and end_class in QUOTA_CLASSES:
            self.completed[e] += 1

    def begin(self, env_ids, global_step):
        """Start the next (full) episode of every env in ``env_ids``; returns the new segments."""
        return [self._segment(e, EPISODE, global_step) for e in env_ids]

    def done(self):
        return bool((self.completed >= self.quota).all())

    def in_flight(self, global_step):
        return [dict(seg, cause="run_end", steps=int(self.steps[seg["env"]]), global_step_end=int(global_step))
                for seg in self.current if seg is not None]

    def expected_batch_steps(self):
        """Batch steps of an uninterrupted run where every episode is full length (LR-A arithmetic)."""
        return int(self.offset(self.N - 1) + self.quota * self.episode_steps)


__all__ = ["RolloutSchedule", "STAGGER", "EPISODE"]
