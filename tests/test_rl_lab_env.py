"""ADR 0022 §8 T1 / T3 (+ Lab-level isolation) on the Isaac Lab kit-less runtime, Warp CPU device.

Runs only in the Isaac Lab venv (``.deps/isaaclab-venv``, with the D-09 pytest overlay); in ``.venv`` the module is
skipped = NOT_RUN. Every Isaac Lab build runs in its own process (one simulation context per process) through
``scripts/rl_ready/probe_lab_batched_cpu.py``; the tests assert on what it writes.

T1  model receipt per world: bodies / DOFs (B1: 23 / 18, B2: 23 / 24), no global bodies, ``neq``, MuJoCo actuator
    arrays and per-joint Newton drive arrays equal to the frozen single Lab env (``make_lab_cfg``, physical grasp);
    after ``reset_envs(mask)`` the Newton manager's world and FK reset masks hold exactly those worlds.
T3  coupling contract, every env and step of a 50-step N=4 run (synthetic test coupling, NOT material): the
    coupling input equals the edge-kernel output and the wrench read in every substep of interval k+1 equals the
    coupling output of step k, bitwise.
Lab isolation (CPU analogue of G1 a-c): N=1 equals world 0 of N=3; perturbing env 2 and a masked reset of env 1
    leave the other worlds bitwise unchanged.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("isaaclab", reason="T tests need the Isaac Lab venv (NOT_RUN in .venv)")

ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT/"scripts/rl_ready/probe_lab_batched_cpu.py"
TIMEOUT_S = 1800


def run_worker(tmp_path_factory, name, *args):
    out = tmp_path_factory.mktemp(name)/"run"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONPATH": str(ROOT/"src")}
    proc = subprocess.run([sys.executable, str(WORKER), *map(str, args), "--out", str(out)], env=env, cwd=ROOT,
                          capture_output=True, text=True, timeout=TIMEOUT_S)
    if proc.returncode != 0:
        raise AssertionError(f"worker {args} failed ({proc.returncode}):\n{proc.stderr[-4000:]}")
    return out


@pytest.fixture(scope="module")
def single_receipt(tmp_path_factory):
    return json.loads((run_worker(tmp_path_factory, "single", "--mode", "single")/"receipt.json").read_text())


@pytest.fixture(scope="module", params=["rigid_mount_v1", "physical_grasp_v1"])
def batched_receipt(request, tmp_path_factory):
    out = run_worker(tmp_path_factory, request.param, "--mode", "receipt", "--num-envs", 3, "--tool-mount", request.param)
    return request.param, json.loads((out/"receipt.json").read_text())


def _arm_joints(joints):
    return {k: v for k, v in joints.items() if k.startswith(("Franka_0:", "Franka_1:")) and v["type"] in (0, 1)}  # prismatic, revolute


def test_t1_model_receipt_per_world(batched_receipt, single_receipt):
    mount, r = batched_receipt
    bodies, dofs = (23, 18) if mount == "rigid_mount_v1" else (23, 24)
    assert r["nworld"] == 3 and r["separate_worlds"] is True
    for e in ("0", "1", "2"):
        w = r["newton"][e]
        assert (w["bodies"], w["dofs"], w["global_bodies"]) == (bodies, dofs, 0), (mount, e, w["bodies"], w["dofs"])
        assert w["joints"] == r["newton"]["0"]["joints"]
    single = single_receipt["newton"]
    assert (single["bodies"], single["dofs"]) == (23, 24), "frozen single env: 2 Frankas + free tool"
    ours, theirs = _arm_joints(r["newton"]["0"]["joints"]), _arm_joints(single["joints"])
    assert set(ours) == set(theirs) and len(ours) == 18
    for key in ours:
        assert ours[key] == theirs[key], f"{mount}: drive/limit arrays of {key} differ from the single Lab env"
    mj, smj = r["mujoco"], single_receipt["mujoco"]
    assert mj["neq"] == smj["neq"] == 2 and mj["eq_type"] == smj["eq_type"]
    assert mj["nu"] == smj["nu"]
    for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_gear", "actuator_trnid"):
        assert mj[name] == smj[name], f"{mount}: MuJoCo {name} differs from the single Lab env"
    assert mj["dof_armature"][:18] == smj["dof_armature"][:18]
    tables = r["env"]["index_tables"] if "env" in r else None
    if tables:
        assert len(set(tables["tool"])) == 3
    assert "CuttingTool" in r["body_names_franka0"] if mount == "rigid_mount_v1" else True


def test_t1_partial_reset_marks_exactly_the_reset_worlds(tmp_path_factory):
    out = run_worker(tmp_path_factory, "mask", "--mode", "mask_audit", "--num-envs", 3, "--reset-env", 1)
    audit = json.loads((out/"mask_audit.json").read_text())
    assert audit["world_reset_mask"] == [False, True, False, False]
    world = np.asarray(audit["articulation_world"])
    assert audit["fk_reset_mask"] == (world == 1).tolist()


def test_t3_coupling_contract_bitwise(tmp_path_factory):
    out = run_worker(tmp_path_factory, "coupling", "--mode", "coupling", "--num-envs", 4, "--steps", 50)
    d = np.load(out/"coupling.npz")
    steps = d["edges"].shape[0]
    assert steps == 50 and d["edges"].shape[1] == 4
    assert np.array_equal(d["inputs"], d["edges"]), "coupling input must be the edge-kernel output, bitwise"
    applied, index = d["applied"], d["applied_step"]
    for k in range(steps):
        rows = applied[index == k]
        assert len(rows) == 10, "one wrench application per 2 ms substep"
        expected = np.zeros((4, 6)) if k == 0 else d["pending"][k-1]
        assert all(np.array_equal(row, expected) for row in rows), f"interval {k} wrench != coupling output {k-1}"
    assert np.abs(d["pending"]).max() > 0 and np.isfinite(d["boxes"]).all()


@pytest.fixture(scope="module")
def rollouts(tmp_path_factory):
    def go(name, *args):
        return np.load(run_worker(tmp_path_factory, name, "--mode", "rollout", "--steps", 10, *args)/"rollout.npz")
    return {"n1": go("n1", "--num-envs", 1), "n3": go("n3", "--num-envs", 3),
            "perturb": go("perturb", "--num-envs", 3, "--perturb-env", 2),
            "reset": go("reset", "--num-envs", 3, "--reset-env", 1, "--reset-step", 5)}


def _world(d, e):
    cs = d["coord_start"]
    bodies = np.flatnonzero(d["body_world"] == e)
    return {"q": d["joint_q"][:, cs[e]:cs[e+1]], "body_q": d["body_q"][:, bodies], "edge": d["edge"][:, e],
            "commanded": d["commanded"][:, e], "obs": d["obs"][:, e], "status": d["status"][:, e]}


def _equal(a, b, label):
    for key in a:
        assert np.array_equal(a[key], b[key]), f"{label}: {key} differs (max {np.abs(a[key]-b[key]).max()})"


def test_lab_n1_equals_world0_of_n3(rollouts):
    _equal(_world(rollouts["n1"], 0), _world(rollouts["n3"], 0), "N=1 vs world 0 of N=3")
    for e in (1, 2):
        _equal(_world(rollouts["n3"], 0), _world(rollouts["n3"], e), f"identical copies world {e}")
    assert min(rollouts["n3"]["nacon"]) > 0


def test_lab_perturbing_env2_leaves_others(rollouts):
    for e in (0, 1):
        _equal(_world(rollouts["perturb"], e), _world(rollouts["n3"], e), f"perturb env 2, world {e}")
    assert not np.array_equal(_world(rollouts["perturb"], 2)["q"], _world(rollouts["n3"], 2)["q"])


def test_lab_masked_reset_of_env1_leaves_others(rollouts):
    for e in (0, 2):
        _equal(_world(rollouts["reset"], e), _world(rollouts["n3"], e), f"masked reset env 1, world {e}")
    reset, base = _world(rollouts["reset"], 1), _world(rollouts["n3"], 1)
    assert np.array_equal(reset["q"][:5], base["q"][:5]) and not np.array_equal(reset["q"][5:], base["q"][5:])
