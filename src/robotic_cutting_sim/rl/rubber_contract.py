"""The rubber vector-env contract ``rubber_vector_contract_v1`` (P2-6, RL-01; ADR 0023 §23.2 decision 7).

``check_rubber_vector_env_contract(env, ...)`` drives an open ``RubberVectorEnv`` through a full reset, steps (preload
included), a partial reset and ``evaluate()`` and asserts: observation keys / shapes / dtypes / device, the deployable
78-dim ``policy`` block with no privileged term name, the opt-in ``critic`` block (starts with the policy block),
spaces and metadata, ``info`` keys / shapes / dtypes, ``termination_reason`` <-> ``terminated`` / ``truncated``, the
partial-reset bookkeeping, the receipt keys (calibration status, fidelity, coupling, perception, envelope row) and the
evaluation keys. It raises ``AssertionError`` naming the first violated clause and returns a JSON-able report. The CPU
tests (parity mode) and the GPU runs call it with the same arguments.
"""
CONTRACT_VERSION = "rubber_vector_contract_v1"
POLICY_DIM = 78
CRITIC_DIM = 106
ACTION_DIM = 6
RESET_INFO = {"step_count": "int64", "seed": "int64", "episode": "int64", "drive": "int8"}
STEP_INFO = {"step_count": "int64", "action_status": "int32", "invalid_reason": "int32", "action_clamped": "bool",
             "action_out_of_box": "bool", "drive": "int8", "seed": "int64", "episode": "int64",
             "termination_reason": "int32", "tip_distance_m": "float64", "within_streak": "int64",
             "invalid_penalty_applied": "bool"}
RECEIPT_KEYS = ("version", "api_revision", "runtime", "backend", "num_envs", "termination", "termination_mapping",
                "evaluation_mode", "fidelity", "observation", "action_spec", "reward", "tasks", "episode", "versions",
                "calibration_status", "success_source", "profile", "contact_law", "envelope", "labels", "material")
EVALUATION_KEYS = ("env", "slot", "seed", "episode", "step_count", "evaluator", "success", "verdict",
                   "tip_distance_m", "within_streak", "termination_reason", "invalid_reason", "labels")
# F2: step-info keys computed from oracle material validity (labelled in env.metadata["privileged_info_keys"])
PRIVILEGED_STEP_INFO = ("invalid_reason", "termination_reason", "invalid_penalty_applied")
# the Phase 1 list (rl.vector_contract.PRIVILEGED_INFO_KEYS; a Torch-free constant, copied to keep this module light;
# tests/test_rubber_rl_contract_f2_f4.py asserts the copy equals the Phase 1 tuple)
PHASE1_PRIVILEGED_INFO_KEYS = ("invalid_reason", "termination_reason", "progress_delta", "off_target_delta",
                               "target_bond_progress_fraction", "off_target_bond_fraction", "severed_target_bonds",
                               "severed_substrate_bonds", "invalid_penalty_applied", "input_guard_reason",
                               "rigid_reason", "autoreset", "terminal_captured")
# substrings that mark a privileged (non-deployable) quantity in a policy term name
PRIVILEGED_MARKERS = ("force", "wrench", "stress", "strain", "jbar", "j_", "energy", "psi", "kinetic", "contact",
                      "penetration", "friction", "mu0", "density", "prony", "material", "invalid", "traction", "cfl")


def contract_actions(num_envs, step, device):
    """Small deterministic grip_delta_v1 actions (+x drag, alternating y, a small yaw), the same on every backend."""
    import torch
    a = torch.zeros((num_envs, ACTION_DIM), dtype=torch.float32, device=device)
    a[:, 0] = .5
    a[:, 1] = .3 if step % 2 == 0 else -.3
    a[:, 5] = .2
    return a


def _check(condition, clause):
    if not condition:
        raise AssertionError(f"rubber vector env contract: {clause}")


def _check_tensor(value, shape, dtype, device, clause):
    import torch
    _check(isinstance(value, torch.Tensor), f"{clause} is a torch tensor")
    _check(tuple(value.shape) == tuple(shape), f"{clause} shape {tuple(value.shape)} == {tuple(shape)}")
    _check(str(value.dtype).replace("torch.", "") == dtype, f"{clause} dtype {value.dtype} == {dtype}")
    _check(value.device == device, f"{clause} on {device} (got {value.device})")


def privileged_term(name):
    n = name.lower()
    return any(m in n for m in PRIVILEGED_MARKERS)


def _check_obs(env, obs, clause):
    import torch
    N, device = env.num_envs, env.torch_device
    privileged = env.observation_profile == "privileged_state"
    _check(set(obs) == ({"policy", "critic"} if privileged else {"policy"}), f"{clause} observation keys {sorted(obs)}")
    _check_tensor(obs["policy"], (N, POLICY_DIM), "float32", device, f"{clause} obs['policy']")
    _check(bool(torch.isfinite(obs["policy"]).all()), f"{clause} policy finite")
    meta = env.observation_metadata()
    _check(meta["policy"]["deployable"] is True, "policy block is deployable")
    names = [t["name"] for t in meta["policy"]["terms"]]
    _check(sum(t["dim"] for t in meta["policy"]["terms"]) == POLICY_DIM, "policy term dims sum to 78")
    _check(not any(privileged_term(n) for n in names), f"policy terms carry no privileged names ({names})")
    _check("invalid_reason" in meta.get("info_privileged", ()), "info['invalid_reason'] listed as privileged")
    if privileged:
        _check_tensor(obs["critic"], (N, CRITIC_DIM), "float32", device, f"{clause} obs['critic']")
        _check(torch.equal(obs["critic"][:, :POLICY_DIM], obs["policy"]), "critic starts with the policy block")
        _check(meta["critic"]["deployable"] is False, "critic block is not deployable")
    else:
        _check("critic" not in meta, "no critic block under the deployable 'state' profile")


def check_rubber_vector_env_contract(env, *, steps=None, reset_ids=(1,), reset_step=None, seeds=None, evaluate=True):
    """Run the contract on an open env (module docstring); returns a report dict. ``steps`` defaults to the preload
    plus 2 policy steps; the partial reset happens after step ``reset_step`` (default: the last preload step)."""
    import torch
    from .termination import TERM_MATERIAL_INVALID, TERM_TRUNCATED, TERMINAL_BITS, decode_termination
    N, device = env.num_envs, env.torch_device
    pre = env.family.preload_steps
    steps = pre+2 if steps is None else int(steps)
    reset_step = max(pre-1, 0) if reset_step is None else int(reset_step)
    reset_ids = tuple(e for e in reset_ids if e < N)
    report = {"contract": CONTRACT_VERSION, "num_envs": N, "steps": steps, "clauses": []}
    receipt = env.receipt()
    for key in RECEIPT_KEYS:
        _check(key in receipt, f"receipt has {key!r}")
    _check(receipt["runtime"] == "rl_gpu_headless" and receipt["backend"] == "kinematic_grip_batched", "runtime/backend")
    _check(receipt["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED", "calibration status")
    fid = receipt["fidelity"]
    _check(fid["coupling"].startswith("kinematic_grip_ideal_tracking") and fid["perception"] == "ideal_marker_tracking",
           "fidelity labels: coupling and perception")
    _check("NOT_IMPLEMENTED_V1" in fid["self_contact"], "self-contact label")
    for n_t, row in receipt["envelope"].items():
        _check(row["status"] == "ESTABLISHED" and row["sliding_mu_max"] is not None, f"envelope row n_t {n_t}")
    _check(set(receipt["termination"]["bits"].values()) >= {"MATERIAL_INVALID", "TRUNCATED_HORIZON",
                                                             "INFRASTRUCTURE_ERROR"}, "termination table")
    report.update(backend=receipt["backend"], runtime=receipt["runtime"], api_revision=receipt["api_revision"],
                  observation_profile=env.observation_profile, test_configuration=fid["test_configuration"],
                  envelope={k: v["sliding_mu_max"] for k, v in receipt["envelope"].items()})
    _check(tuple(env.single_action_space.shape) == (ACTION_DIM,), "single action space")
    _check(tuple(env.action_space.shape) == (N, ACTION_DIM), "batched action space")
    _check(tuple(env.single_observation_space["policy"].shape) == (POLICY_DIM,), "single observation space")
    _check(tuple(env.observation_space["policy"].shape) == (N, POLICY_DIM), "batched observation space")
    for key in ("autoreset_mode", "observation_profile", "max_episode_steps"):
        _check(key in env.metadata, f"metadata has {key!r}")
    # F2 (P6.7 review, additive clause): oracle-derived info keys are labelled as on the Phase 1 backends
    _check(set(PRIVILEGED_STEP_INFO) <= set(env.metadata.get("privileged_info_keys", ())),
           "metadata['privileged_info_keys'] lists the oracle-derived info keys")
    report["clauses"].append("receipt/spaces/metadata")
    seeds = list(range(100, 100+N)) if seeds is None else list(seeds)
    obs, info = env.reset(seeds=seeds)
    _check_obs(env, obs, "reset")
    for key, dtype in RESET_INFO.items():
        _check(key in info, f"reset info has {key!r}")
        _check_tensor(info[key], (N,), dtype, device, f"reset info[{key!r}]")
    _check(info["step_count"].tolist() == [0]*N, "full reset step_count == 0")
    _check(info["seed"].tolist() == seeds, "full reset seeds applied")
    _check(bool((obs["policy"][:, -1] == (1.0 if pre > 0 else 0.0)).all()), "preload flag after reset")
    episode0 = info["episode"].clone()
    report["clauses"].append("full reset")
    previous = torch.zeros(N, dtype=torch.int64, device=device)
    terminal_seen = 0
    statuses = []
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
        _check(set(info) & set(PHASE1_PRIVILEGED_INFO_KEYS) <= set(env.metadata["privileged_info_keys"]),
               "every Phase 1 privileged info key the step carries is labelled")
        reason = info["termination_reason"]
        _check(torch.equal(terminated, (reason & TERMINAL_BITS) != 0), "terminated == reason & TERMINAL_BITS")
        _check(torch.equal(truncated, (reason & TERM_TRUNCATED) != 0), "truncated == reason & TRUNCATED_HORIZON")
        _check(torch.equal(truncated, info["step_count"] >= env.max_episode_steps), "truncated at the horizon")
        _check(bool(((info["invalid_reason"] == 0) | ((reason & TERM_MATERIAL_INVALID) != 0)).all()),
               "every rubber invalid bit sets MATERIAL_INVALID")
        _check(torch.equal(info["step_count"], previous+1), f"step_count advanced by one at step {k}")
        in_preload = previous < pre
        _check(bool((info["action_status"][in_preload] == 3).all()), "preload steps report action_status 3")
        _check(bool((reward[in_preload] == 0).all()), "reward 0 during preload")
        statuses.append(info["action_status"].tolist())
        terminal_seen += int(terminated.sum())
        previous = info["step_count"].clone()
        if k == reset_step and reset_ids:
            ids = list(reset_ids)
            obs, rinfo = env.reset(env_ids=ids, seeds=[7+e for e in ids])
            _check_obs(env, obs, "partial reset")
            expected = previous.clone()
            expected[ids] = 0
            _check(torch.equal(rinfo["step_count"], expected), "partial reset zeroes only the reset envs' step_count")
            bumped = episode0.clone()
            bumped[ids] += 1
            _check(torch.equal(rinfo["episode"], bumped), "partial reset bumps only the reset envs' episode")
            _check([int(rinfo["seed"][e]) for e in ids] == [7+e for e in ids], "partial reset seeds applied")
            previous = rinfo["step_count"].clone()
            report["clauses"].append("partial reset")
    report["clauses"].append("steps")
    report["action_status"] = statuses
    report["final_step_count"] = previous.tolist()
    report["termination_reason"] = [decode_termination(v) for v in info["termination_reason"].tolist()]
    report["terminated_events"] = terminal_seen
    if evaluate:
        results = env.evaluate()
        _check([r["env"] for r in results] == list(range(N)), "evaluate() covers every env in order")
        for r in results:
            for key in EVALUATION_KEYS:
                _check(key in r, f"evaluation result has {key!r}")
            _check(r["evaluator"] == receipt["evaluation_mode"], "evaluator id matches the receipt")
            _check(r["labels"]["calibration_status"] == "SIMULATOR_ONLY_UNCALIBRATED", "evaluation label")
            _check(isinstance(r["success"], bool), "success is a bool from the evaluator")
        report["evaluation"] = [{k: r[k] for k in ("env", "slot", "step_count", "success", "tip_distance_m")}
                                for r in results]
        report["clauses"].append("evaluate")
    report["status"] = "PASS"
    return report


__all__ = ["check_rubber_vector_env_contract", "contract_actions", "CONTRACT_VERSION", "STEP_INFO", "RESET_INFO",
           "RECEIPT_KEYS", "EVALUATION_KEYS", "POLICY_DIM", "CRITIC_DIM", "privileged_term"]
