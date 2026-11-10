"""``rcs_rl_rollout_v1``: RL transition chunks written from a batched vector env (P1-5 T5.12, plan section 5).

The writer records, for every batch step, the transition of every env as consumed by a standard RL loop:
``obs_policy[t]`` (observation the action was computed from), ``actions[t]`` (executed ``edge_delta_v1`` action as
the env reports it, ``info`` not needed), ``rewards[t]``, ``terminated[t]``, ``truncated[t]``, ``next_obs_policy[t]``
(the observation after the step; for an env that ended inside the step under ``autoreset="same_step"`` it is
``info["final_obs"]``, the pre-reset observation), plus ``episode``, ``step_count``, ``seed``, ``drive``,
``action_status``, ``invalid_reason``, ``termination_reason`` and ``reset_kind`` (0 none, 1 autoreset, 2 forced).

Forced resets (``env.reset(env_ids=...)`` between steps, time-limit-style truncation): before the reset the caller
calls ``mark_forced_reset(ids, obs)``, which sets ``truncated`` and ``reset_kind = 2`` on the last stored transition
of those envs and stores ``obs`` (their current, pre-reset observation) as its ``next_obs``. Rows are held in device
buffers and copied to the host once per chunk (``chunk_size`` steps), never per step; the last chunk row stays
editable until the next row is added, so a forced-reset mark never touches a written file.

Files (a new directory, never overwritten): ``manifest.json`` (schema, env receipt, spaces, N, T, chunk list with
SHA256, policies, drive and reset schedule, git, versions), ``chunk_NNNNN.npz`` and ``episodes.jsonl`` (one line per
finished episode, written by the caller through ``episode_record``). The consumer check is
``scripts/rl_ready/check_rl_rollout.py``. Privileged ``obs_critic`` is written only when ``critic=True`` and the
manifest then flags it non-deployable.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = "rcs_rl_rollout_v1"
RESET_NONE, RESET_AUTO, RESET_FORCED = 0, 1, 2
END_TIME_LIMIT = "TRUNCATED_TIME_LIMIT"
END_INVALID = "TERMINATED_INVALID"
END_FORCED = "TRUNCATED_BY_FORCED_RESET"
FIELDS = {  # name: (trailing shape key, numpy dtype)
    "obs_policy": ("D", np.float32), "next_obs_policy": ("D", np.float32), "actions": ("A", np.float32),
    "rewards": (None, np.float32), "terminated": (None, np.bool_), "truncated": (None, np.bool_),
    "episode": (None, np.int64), "step_count": (None, np.int64), "seed": (None, np.int64), "drive": (None, np.int8),
    "action_status": (None, np.int32), "invalid_reason": (None, np.int32), "termination_reason": (None, np.int32),
    "reset_kind": (None, np.int8)}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RolloutWriter:
    def __init__(self, out_dir, num_envs, *, obs_dim, action_dim=8, chunk_size=256, device="cpu", critic_dim=None,
                 manifest=None):
        import torch
        self.torch = torch
        self.out = Path(out_dir)
        if self.out.exists() and any(self.out.iterdir()):
            raise FileExistsError(f"{self.out} exists and is not empty (never overwritten)")
        self.out.mkdir(parents=True, exist_ok=True)
        if not 1 <= chunk_size <= 256:
            raise ValueError("chunk_size must lie in [1, 256]")
        self.N, self.chunk_size, self.device = int(num_envs), int(chunk_size), torch.device(device)
        dims = dict(D=int(obs_dim), A=int(action_dim))
        fields = dict(FIELDS)
        if critic_dim is not None:
            dims["C"] = int(critic_dim)
            fields["obs_critic"] = ("C", np.float32)
            fields["next_obs_critic"] = ("C", np.float32)
        self.fields = fields
        self.buffers = {}
        for name, (key, dtype) in fields.items():
            shape = (self.chunk_size, self.N) + ((dims[key],) if key else ())
            self.buffers[name] = torch.zeros(shape, dtype=getattr(torch, np.dtype(dtype).name), device=self.device)
        self.rows = 0            # rows in the current (unflushed) chunk
        self.total = 0
        self.chunks = []
        self.manifest_base = dict(manifest or {})
        self.critic = critic_dim is not None
        self._episodes = (self.out / "episodes.jsonl").open("x")

    def _flush(self):
        if self.rows == 0:
            return
        index = len(self.chunks)
        path = self.out / f"chunk_{index:05d}.npz"
        host = {name: buf[:self.rows].cpu().numpy() for name, buf in self.buffers.items()}  # one copy per chunk
        with path.open("xb") as stream:
            np.savez_compressed(stream, **host)
        self.chunks.append(dict(file=path.name, rows=self.rows, first_step=self.total - self.rows,
                                sha256=_sha256(path)))
        self.rows = 0

    def add(self, *, obs, actions, reward, terminated, truncated, next_obs, info, reset_kind):
        """One batch step (device tensors; no host sync unless the previous chunk is full)."""
        torch = self.torch
        if self.rows == self.chunk_size:
            self._flush()
        i = self.rows
        b = self.buffers
        b["obs_policy"][i].copy_(obs["policy"])
        b["next_obs_policy"][i].copy_(next_obs["policy"])
        if self.critic:
            b["obs_critic"][i].copy_(obs["critic"])
            b["next_obs_critic"][i].copy_(next_obs["critic"])
        b["actions"][i].copy_(actions.to(torch.float32))
        b["rewards"][i].copy_(reward.to(torch.float32))
        b["terminated"][i].copy_(terminated)
        b["truncated"][i].copy_(truncated)
        for name in ("episode", "step_count", "seed", "drive", "action_status", "invalid_reason", "termination_reason"):
            b[name][i].copy_(info[name].to(b[name].dtype))
        b["reset_kind"][i].copy_(reset_kind.to(torch.int8))
        self.rows += 1
        self.total += 1

    def mark_forced_reset(self, env_ids, obs):
        """Time-limit-style end of ``env_ids`` before ``env.reset(env_ids=...)`` (see the module docstring)."""
        if self.rows == 0:
            raise RuntimeError("no stored transition to mark (forced reset before the first step)")
        ids = self.torch.as_tensor(list(env_ids), dtype=self.torch.int64, device=self.device)
        i = self.rows - 1
        b = self.buffers
        b["truncated"][i, ids] = True
        b["reset_kind"][i, ids] = RESET_FORCED
        b["next_obs_policy"][i, ids] = obs["policy"][ids]
        if self.critic:
            b["next_obs_critic"][i, ids] = obs["critic"][ids]

    def episode_record(self, **row):
        self._episodes.write(json.dumps(row, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist")
                                        else str(o)) + "\n")
        self._episodes.flush()

    def close(self, **extra):
        self._flush()
        self._episodes.close()
        manifest = dict(schema=SCHEMA, version=1, num_envs=self.N, steps=self.total, chunk_size=self.chunk_size,
                        chunks=self.chunks, fields={k: dict(dtype=np.dtype(v[1]).name, trailing=v[0])
                                                    for k, v in self.fields.items()},
                        episodes_file="episodes.jsonl", episodes_sha256=_sha256(self.out / "episodes.jsonl"),
                        obs_critic=("PRIVILEGED_NON_DEPLOYABLE" if self.critic else None),
                        reset_kind={"0": "none", "1": "autoreset_same_step", "2": "forced_reset"},
                        conventions=dict(
                            next_obs="observation after the step; info['final_obs'] for envs that ended in the step "
                                     "(autoreset same_step); for a forced reset, the pre-reset observation",
                            done="terminated | truncated; forced resets set truncated on the last stored transition",
                            episode="per-env episode counter of the env (increments at every reset)"),
                        **self.manifest_base, **extra)
        with (self.out / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=1, sort_keys=True,
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        return manifest


def load_rollout(path):
    """``(manifest, arrays concatenated over chunks, episodes list)``; verifies the chunk SHA256s."""
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} rollout")
    parts = {}
    for chunk in manifest["chunks"]:
        file = path / chunk["file"]
        if _sha256(file) != chunk["sha256"]:
            raise ValueError(f"{file}: sha256 mismatch")
        with np.load(file, allow_pickle=False) as data:
            for key in data.files:
                parts.setdefault(key, []).append(data[key])
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    episodes = [json.loads(line) for line in (path / "episodes.jsonl").read_text().splitlines() if line.strip()]
    return manifest, arrays, episodes


__all__ = ["RolloutWriter", "load_rollout", "SCHEMA", "RESET_NONE", "RESET_AUTO", "RESET_FORCED", "END_TIME_LIMIT",
           "END_INVALID", "END_FORCED"]
