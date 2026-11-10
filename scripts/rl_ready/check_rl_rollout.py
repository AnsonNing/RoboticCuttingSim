#!/usr/bin/env python3
"""Consumer check of an ``rcs_rl_rollout_v1`` directory (P1-5 T5.12, plan section 5): can a standard RL loop use it?

Items (each PASS or FAIL; the exit code is 0 only when every item passes, and an exit code alone is never the
report): chunk SHA256s; shapes/dtypes vs the declared spaces; finite values; actions in [-1, 1];
``obs[t+1] == next_obs[t]`` bitwise wherever the env is not done at t; ``done = terminated | truncated`` matches the
episode increments and the step-count restarts per env; episode returns recomputed from ``rewards`` equal
``episodes.jsonl``; a CleanRL-style GAE (gamma 0.99, lambda 0.95, zero value function, bootstrap on truncation
only) gives finite advantages (NumPy, and on Torch when importable); one pass of a shuffled minibatch iterator
over T x N (a PPO-style data loader, no update); ``obs_policy`` carries only ``rl.observations.POLICY_TERMS``
(dimension and the manifest's observation metadata).

    PYTHONPATH=src .venv/bin/python scripts/rl_ready/check_rl_rollout.py <rollout dir> [--json out.json]
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def gae_numpy(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    """CleanRL-style GAE over [T, N]: bootstrap from next_values on truncation, zero on termination; the
    advantage recursion stops at every episode end."""
    T = len(rewards)
    adv = np.zeros_like(rewards, dtype=np.float64)
    last = np.zeros(rewards.shape[1])
    done = terminated | truncated
    for t in reversed(range(T)):
        boot = np.where(terminated[t], 0., next_values[t])
        delta = rewards[t] + gamma * boot - values[t]
        last = delta + gamma * lam * np.where(done[t], 0., 1.) * last
        adv[t] = last
    return adv


def gae_torch(rewards, values, next_values, terminated, truncated, gamma=.99, lam=.95):
    import torch
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards, dtype=torch.float64)
    last = torch.zeros(rewards.shape[1], dtype=torch.float64, device=rewards.device)
    done = terminated | truncated
    for t in reversed(range(T)):
        boot = torch.where(terminated[t], torch.zeros_like(next_values[t]), next_values[t])
        delta = rewards[t].double() + gamma * boot - values[t]
        last = delta + gamma * lam * (~done[t]).double() * last
        adv[t] = last
    return adv


def check(path):
    from robotic_cutting_sim.rl.rollout import RESET_AUTO, RESET_FORCED, RESET_NONE, load_rollout
    items = {}

    def item(name, ok, detail=None):
        items[name] = dict(status="PASS" if ok else "FAIL", detail=detail)
    try:
        manifest, a, episodes = load_rollout(path)
        item("chunks_sha256", True, f"{len(manifest['chunks'])} chunks")
    except Exception as error:   # a broken chunk is a FAIL, reported
        item("chunks_sha256", False, repr(error))
        return dict(status="FAIL", items=items)
    N, T = manifest["num_envs"], manifest["steps"]
    D = manifest["single_observation_space"]["shape"][0]
    A = manifest["single_action_space"]["shape"][0]
    shapes = dict(obs_policy=((T, N, D), "float32"), next_obs_policy=((T, N, D), "float32"),
                  actions=((T, N, A), "float32"), rewards=((T, N), "float32"), terminated=((T, N), "bool"),
                  truncated=((T, N), "bool"), episode=((T, N), "int64"), step_count=((T, N), "int64"),
                  seed=((T, N), "int64"), drive=((T, N), "int8"), action_status=((T, N), "int32"),
                  invalid_reason=((T, N), "int32"), termination_reason=((T, N), "int32"), reset_kind=((T, N), "int8"))
    bad = [k for k, (shape, dtype) in shapes.items() if k not in a or a[k].shape != shape or str(a[k].dtype) != dtype]
    item("shapes_and_dtypes", not bad, bad or None)
    floats = [k for k in a if a[k].dtype.kind == "f"]
    item("finite", all(np.isfinite(a[k]).all() for k in floats), floats)
    item("actions_in_box", bool((np.abs(a["actions"]) <= 1.).all()), float(np.abs(a["actions"]).max()))
    done = a["terminated"] | a["truncated"]
    cont = ~done[:-1]
    same = np.all(a["obs_policy"][1:] == a["next_obs_policy"][:-1], axis=2)
    item("obs_chain_bitwise_when_not_done", bool(same[cont].all()), int((~same & cont).sum()))
    ep_change = a["episode"][1:] != a["episode"][:-1]
    item("done_matches_episode_increment", bool(np.array_equal(ep_change, done[:-1])),
         int((ep_change != done[:-1]).sum()))
    restart = a["step_count"][1:] == 1
    item("step_count_restarts_after_done", bool(np.array_equal(restart, done[:-1])),
         int((restart != done[:-1]).sum()))
    progress = a["step_count"][1:] == a["step_count"][:-1] + 1
    item("step_count_advances_within_episode", bool(progress[cont].all()), int((~progress & cont).sum()))
    reset_kinds = set(np.unique(a["reset_kind"]).tolist())
    item("reset_kind_values", reset_kinds <= {RESET_NONE, RESET_AUTO, RESET_FORCED}
         and bool(np.array_equal(a["reset_kind"] != RESET_NONE, done)), sorted(reset_kinds))
    from robotic_cutting_sim.rl.termination import TERM_TRUNCATED, TERMINAL_BITS
    reason = a["termination_reason"]
    auto = a["reset_kind"] == RESET_AUTO
    item("termination_reason_consistent",
         bool(np.array_equal(a["terminated"][auto], (reason[auto] & TERMINAL_BITS) != 0))
         and bool(((reason[a["terminated"]] & TERMINAL_BITS) != 0).all()), None)
    # returns: per env, segment rewards at episode ends
    rows = {}
    for e in range(N):
        start = 0
        for t in np.flatnonzero(done[:, e]):
            rows.setdefault(e, []).append(float(a["rewards"][start:t + 1, e].astype(np.float64).sum()))
            start = t + 1
    recorded = {}
    for row in episodes:
        recorded.setdefault(row["env"], []).append(float(row["undiscounted_return"]))
    mismatch = [(e, i) for e in rows for i, (x, y) in enumerate(zip(rows[e], recorded.get(e, [])))
                if not np.isclose(x, y, rtol=1e-5, atol=1e-6)]
    counts_equal = all(len(rows.get(e, [])) == len(recorded.get(e, [])) for e in range(N))
    item("episode_returns_match_episodes_jsonl", counts_equal and not mismatch,
         dict(episodes=sum(len(v) for v in rows.values()), mismatches=mismatch[:5]))
    values = np.zeros((T, N))
    adv = gae_numpy(a["rewards"].astype(np.float64), values, values, a["terminated"], a["truncated"])
    item("gae_numpy_finite", bool(np.isfinite(adv).all()), float(np.abs(adv).max()))
    try:
        import torch
        tv = torch.zeros((T, N), dtype=torch.float64)
        tadv = gae_torch(torch.as_tensor(a["rewards"]), tv, tv, torch.as_tensor(a["terminated"]),
                         torch.as_tensor(a["truncated"]))
        item("gae_torch_matches_numpy", bool(np.allclose(tadv.numpy(), adv, rtol=1e-12, atol=1e-12)), None)
    except ImportError:
        items["gae_torch_matches_numpy"] = dict(status="NOT_RUN", detail="torch not importable")
    rng = np.random.default_rng(0)
    flat = T * N
    order = rng.permutation(flat)
    seen = np.zeros(flat, bool)
    batches = 0
    obs_flat = a["obs_policy"].reshape(flat, D)
    for start in range(0, flat, max(1, flat // 8)):
        idx = order[start:start + max(1, flat // 8)]
        batch = dict(obs=obs_flat[idx], actions=a["actions"].reshape(flat, A)[idx], adv=adv.reshape(flat)[idx])
        seen[idx] = True
        batches += int(batch["obs"].shape[0] > 0)
    item("minibatch_pass_covers_every_transition", bool(seen.all()), dict(minibatches=batches))
    from robotic_cutting_sim.rl.observations import POLICY_TERMS
    policy_meta = manifest["env_receipt"]["observation"]["policy"]
    names = [t["name"] for t in policy_meta["terms"]]
    allowed = {term[0] for term in POLICY_TERMS}
    item("policy_obs_terms_only", set(names) <= allowed and int(policy_meta["dim"]) == D
         and int(policy_meta["dim"]) == sum(int(term[1]) for term in POLICY_TERMS if term[0] in names),
         dict(terms=names, dim=D))
    item("critic_flagged_non_deployable", ("obs_critic" not in a) or manifest.get("obs_critic") == "PRIVILEGED_NON_DEPLOYABLE",
         manifest.get("obs_critic"))
    status = "PASS" if all(v["status"] in ("PASS", "NOT_RUN") for v in items.values()) else "FAIL"
    if any(v["status"] == "NOT_RUN" for v in items.values()) and status == "PASS":
        status = "PASS_WITH_NOT_RUN"
    return dict(status=status, items=items, num_envs=N, steps=T, episodes=len(episodes))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path)
    p.add_argument("--json", type=Path)
    a = p.parse_args()
    report = check(a.path)
    for name, row in report["items"].items():
        print(f"{row['status']:9s} {name}  {row['detail'] if row['detail'] is not None else ''}")
    print(f"OVERALL {report['status']}")
    if a.json is not None:
        with a.json.open("x") as stream:
            json.dump(report, stream, indent=1, default=str)
    return 0 if report["status"].startswith("PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
