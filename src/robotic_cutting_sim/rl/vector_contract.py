"""The public vector-env contract (P1-2b), one checker for both backends.

``check_vector_env_contract(env, ...)`` drives an open ``open_vector_env`` env (workload A
``VectorCuttingEnv`` or workload B ``LabVectorCuttingEnv``) through reset, steps, a partial reset and
``evaluate()``, and asserts the same things for both: observation keys/shapes/dtypes/device and the deployable
``"policy"`` block, the opt-in ``"critic"`` block, spaces, ``info`` keys/shapes/dtypes, the ``termination_reason``
bitfield and its relation to ``terminated`` / ``truncated``, the partial-reset bookkeeping, the receipt fields and
the evaluation mode. It raises ``AssertionError`` naming the first violated clause and returns a JSON-able report.
The tests (CPU, workload A) and the GPU smoke (both backends) call it with the same arguments.
"""
CONTRACT_VERSION = "vector_env_contract_v1"
POLICY_DIM = 18
RESET_INFO = {"step_count": "int64", "seed": "int64", "episode": "int64", "drive": "int8"}
STEP_INFO = {"step_count": "int64", "action_status": "int32", "invalid_reason": "int32", "cut_digit": "int32",
             "action_clamped": "bool", "action_out_of_box": "bool", "drive": "int8", "seed": "int64",
             "episode": "int64", "termination_reason": "int32", "progress_delta": "float64",
             "off_target_delta": "float64", "target_bond_progress_fraction": "float64",
             "off_target_bond_fraction": "float64", "severed_target_bonds": "int64",
             "severed_substrate_bonds": "int64", "invalid_penalty_applied": "bool"}
# step info keys computed from oracle state (material solver validity / codes, simulator validity): a deployable
# policy must never consume them (review finding 12). Both backends list them in env.metadata["privileged_info_keys"]
# (``input_guard_reason`` only with the guard, ``rigid_reason`` on workload B).
PRIVILEGED_INFO_KEYS = ("invalid_reason", "termination_reason", "progress_delta", "off_target_delta",
                        "target_bond_progress_fraction", "off_target_bond_fraction", "severed_target_bonds",
                        "severed_substrate_bonds", "invalid_penalty_applied", "input_guard_reason", "rigid_reason")
RECEIPT_KEYS = ("version", "api_revision", "runtime", "backend", "num_envs", "termination", "input_guard",
                "evaluation_stream", "evaluation_mode", "fidelity", "observation", "action_spec", "reward", "tasks",
                "episode", "versions", "calibration", "success_source", "terminal_capture")
EVALUATION_KEYS = ("env", "scene_seed", "task_id", "step_count", "frames", "evaluation_mode", "summary", "result",
                   "final_codes_sha256", "labelled")


def contract_actions(num_envs, step, device):
    """Small deterministic edge_delta_v1 actions (armed, digit 0, +x/-z motion), identical for both backends."""
    import torch
    a = torch.zeros((num_envs, 8), dtype=torch.float32, device=device)
    a[:, 0] = .3
    a[:, 2] = -.2 if step % 2 == 0 else .1
    a[:, 6] = 1.
    a[:, 7] = -.5
    return a


def _check(condition, clause):
    if not condition:
        raise AssertionError(f"vector env contract: {clause}")


def _check_tensor(value, shape, dtype, device, clause):
    import torch
    _check(isinstance(value, torch.Tensor), f"{clause} is a torch tensor")
    _check(tuple(value.shape) == tuple(shape), f"{clause} shape {tuple(value.shape)} == {tuple(shape)}")
    _check(str(value.dtype).replace("torch.", "") == dtype, f"{clause} dtype {value.dtype} == {dtype}")
    _check(value.device == device, f"{clause} on {device} (got {value.device})")


def _check_obs(env, obs, clause):
    from .observations import privileged_name_reasons
    N, device = env.num_envs, env.torch_device
    privileged = env.observation_profile == "privileged_state"
    _check(set(obs) == ({"policy", "critic"} if privileged else {"policy"}), f"{clause} observation keys {sorted(obs)}")
    _check_tensor(obs["policy"], (N, POLICY_DIM), "float32", device, f"{clause} obs['policy']")
    meta = env.observation.metadata()
    _check(meta["policy"]["deployable"] is True, "policy block is deployable")
    names = [t["name"] for t in meta["policy"]["terms"]]
    _check(not any(privileged_name_reasons(n) for n in names), "policy terms carry no privileged names")
    if privileged:
        _check_tensor(obs["critic"], (N, env.observation.critic_dim), "float32", device, f"{clause} obs['critic']")
        import torch
        _check(torch.equal(obs["critic"][:, :POLICY_DIM], obs["policy"]), "critic starts with the policy block")
    else:
        _check("critic" not in meta, "no critic block under the deployable 'state' profile")


def check_vector_env_contract(env, *, steps=3, reset_ids=(1,), reset_step=1, seeds=None, evaluate=True):
    """Run the contract on an open env (module docstring); returns a report dict."""
    import torch
    from .termination import TERM_TRUNCATED, TERMINAL_BITS, decode_termination
    N, device = env.num_envs, env.torch_device
    report = {"contract": CONTRACT_VERSION, "num_envs": N, "steps": steps, "clauses": []}
    receipt = env.receipt()
    for key in RECEIPT_KEYS:
        _check(key in receipt, f"receipt has {key!r}")
    report.update(backend=receipt["backend"], runtime=receipt["runtime"], api_revision=receipt["api_revision"],
                  evaluation_mode=receipt["evaluation_mode"], observation_profile=env.observation_profile,
                  reduced_fidelity=receipt["fidelity"].get("reduced_fidelity"),
                  input_guard=bool(receipt["input_guard"].get("enabled", True)))
    _check(receipt["runtime"] == "rl_gpu_headless", "runtime is rl_gpu_headless")
    _check(set(receipt["termination"]["bits"].values()) >= {"RIGID_INVALID", "MATERIAL_INVALID", "GRASP_LOST",
                                                             "IK_FAILURE_STREAK", "TRUNCATED_HORIZON",
                                                             "INFRASTRUCTURE_ERROR"}, "termination table complete")
    # spaces and metadata
    _check(tuple(env.single_action_space.shape) == (8,) and tuple(env.action_space.shape) == (N, 8), "action spaces")
    _check(tuple(env.single_observation_space["policy"].shape) == (POLICY_DIM,), "single observation space")
    _check(tuple(env.observation_space["policy"].shape) == (N, POLICY_DIM), "batched observation space")
    for key in ("autoreset_mode", "observation_profile", "max_episode_steps", "privileged_info_keys"):
        _check(key in env.metadata, f"metadata has {key!r}")
    _check(set(PRIVILEGED_INFO_KEYS) & set(STEP_INFO) <= set(env.metadata["privileged_info_keys"]),
           "metadata labels the oracle-derived step info keys as privileged")
    report["clauses"].append("receipt/spaces/metadata")
    # full reset
    seeds = list(range(100, 100 + N)) if seeds is None else list(seeds)
    obs, info = env.reset(seeds=seeds)
    _check_obs(env, obs, "reset")
    for key, dtype in RESET_INFO.items():
        _check(key in info, f"reset info has {key!r}")
        _check_tensor(info[key], (N,), dtype, device, f"reset info[{key!r}]")
    _check(info["step_count"].tolist() == [0] * N, "full reset step_count == 0")
    _check(info["seed"].tolist() == seeds, "full reset seeds applied")
    episode0 = info["episode"].clone()
    report["clauses"].append("full reset")
    previous = torch.zeros(N, dtype=torch.int64, device=device)
    terminal_seen = 0
    for k in range(steps):
        obs, reward, terminated, truncated, info = env.step(contract_actions(N, k, device))
        _check_obs(env, obs, f"step {k}")
        _check_tensor(reward, (N,), "float32", device, "reward")
        _check(bool(torch.isfinite(reward).all()), "reward finite")
        _check_tensor(terminated, (N,), "bool", device, "terminated")
        _check_tensor(truncated, (N,), "bool", device, "truncated")
        for key, dtype in STEP_INFO.items():
            _check(key in info, f"step info has {key!r}")
            _check_tensor(info[key], (N,), dtype, device, f"step info[{key!r}]")
        reason = info["termination_reason"]
        _check(torch.equal(terminated, (reason & TERMINAL_BITS) != 0), "terminated == reason & TERMINAL_BITS")
        _check(torch.equal(truncated, (reason & TERM_TRUNCATED) != 0), "truncated == reason & TRUNCATED_HORIZON")
        _check(torch.equal(truncated, info["step_count"] >= env.max_episode_steps), "truncated at the horizon")
        _check(torch.equal(info["step_count"], previous + 1), f"step_count advanced by one at step {k}")
        terminal_seen += int(terminated.sum())
        previous = info["step_count"].clone()
        if k == reset_step and reset_ids:
            ids = list(reset_ids)
            obs, rinfo = env.reset(env_ids=ids, seeds=[7 + e for e in ids])
            _check_obs(env, obs, "partial reset")
            expected = previous.clone()
            expected[ids] = 0
            _check(torch.equal(rinfo["step_count"], expected), "partial reset zeroes only the reset envs' step_count")
            bumped = episode0.clone()
            bumped[ids] += 1
            _check(torch.equal(rinfo["episode"], bumped), "partial reset bumps only the reset envs' episode")
            _check([int(rinfo["seed"][e]) for e in ids] == [7 + e for e in ids], "partial reset seeds applied")
            previous = rinfo["step_count"].clone()
            report["clauses"].append("partial reset")
    report["clauses"].append("steps")
    report["final_step_count"] = previous.tolist()
    report["termination_reason"] = [decode_termination(v) for v in info["termination_reason"].tolist()]
    report["terminated_events"] = terminal_seen
    if evaluate:
        results = env.evaluate()
        _check([r["env"] for r in results] == list(range(N)), "evaluate() covers every env in order")
        stream = receipt["evaluation_stream"]
        for r in results:
            for key in EVALUATION_KEYS:
                _check(key in r, f"evaluation result has {key!r}")
            _check(r["evaluation_mode"] == receipt["evaluation_mode"], "evaluation mode matches the receipt")
            for key in ("verdict", "integrity", "outcome"):
                _check(key in r["result"], f"frozen evaluator result has {key!r}")
            s = int(r["step_count"])
            if stream is None:
                _check(r["frames"] == (2 if s > 0 else 1), "two-frame evaluation by default")
            else:
                every = int(stream["config"]["every"])
                expected = 1 + s // every + (1 if s % every else 0) if s > 0 else 1
                _check(r["frames"] == expected, f"stream frames {r['frames']} == {expected} at step {s}")
                _check(r["stream_dropped_samples"] == 0, "no stream sample dropped")
        report["evaluation"] = [dict(env=r["env"], scene_seed=r["scene_seed"], step_count=r["step_count"],
                                     frames=r["frames"], mode=r["evaluation_mode"],
                                     verdict=r["summary"]["verdict"], labelled=r["labelled"]) for r in results]
        report["clauses"].append("evaluate")
    report["status"] = "PASS"
    return report


def summarize_info_dtypes(info):
    """``{key: (shape, dtype)}`` of an info dict (receipt/diagnostics helper)."""
    return {k: (tuple(v.shape), str(v.dtype)) for k, v in info.items() if hasattr(v, "shape")}


__all__ = ["check_vector_env_contract", "contract_actions", "CONTRACT_VERSION", "STEP_INFO", "RESET_INFO",
           "RECEIPT_KEYS", "POLICY_DIM", "PRIVILEGED_INFO_KEYS", "summarize_info_dtypes"]
