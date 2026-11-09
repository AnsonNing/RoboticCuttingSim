"""Workload B stage S0 worker: one Isaac Lab kit-less build per process on the Warp CPU device (ADR 0022 §8 T1/T3).

Isaac Lab owns one process-wide simulation context, so every build runs in its own process; the T tests
(``tests/test_rl_lab_env.py``) call this script and assert on the JSON/NPZ it writes. Run in the Isaac Lab venv:

    CUDA_VISIBLE_DEVICES='' PYTHONPATH=src .deps/isaaclab-venv/bin/python scripts/rl_ready/probe_lab_batched_cpu.py \
        --mode receipt --num-envs 3 --tool-mount rigid_mount_v1 --out OUT_DIR

Modes:
  receipt      batched env receipt: per-world model counts, actuator/MuJoCo arrays of world 0, no global bodies
  single       the frozen single Lab env's physics scene (``isaac_lab_env.make_lab_cfg``, physical grasp, MJWarp on
               the Warp CPU device) in a bare DirectRLEnv: the same receipt fields of its world 0
  mask_audit   N=3: reset, 3 steps, reset_envs([F,T,F]); the masks right after the writes
  rollout      scripted edge_delta_v1 actions for --steps; per-step joint_q / body_q / edge (bitwise comparisons
               across runs: --perturb-env k scales env k's action, --reset-env k --reset-step s masked reset)
  coupling     T3: synthetic test coupling (NOT material) over --steps; material input vs edge-kernel output and
               the wrench applied in every substep of interval k+1 vs the coupling output of step k
No GPU is used; the script refuses to run if CUDA is visible.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np


def _task(seed):
    from robotic_cutting_sim.scene_config import compile_task
    from robotic_cutting_sim.seeded_trim import generate_scene, goal_config
    return compile_task(generate_scene(seed), goal_config(), seed=42, particle_spacing_m=.0025)


def _mjw_receipt(solver, world):
    m = solver.mjw_model
    out = {"nu": int(m.nu), "neq": int(m.neq), "nv": int(m.nv), "nq": int(m.nq), "nbody": int(m.nbody)}
    for name in ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange", "actuator_trnid", "actuator_gear",
                 "dof_armature", "dof_damping", "eq_type", "eq_data", "jnt_actgravcomp", "body_gravcomp"):
        value = getattr(m, name, None)
        if value is None:
            continue
        array = value.numpy()
        nworld = int(solver.mjw_data.nworld)
        if array.ndim >= 2 and array.shape[0] in (1, nworld) and name not in ("actuator_trnid", "eq_type",
                                                                           "jnt_actgravcomp"):
            array = array[world if array.shape[0] == nworld else 0]
        out[name] = np.asarray(array).tolist()
    return out


def _newton_world_receipt(model, world):
    bw, jw = model.body_world.numpy(), model.joint_world.numpy()
    joints = np.flatnonzero(jw == world)
    dofs = []
    qd_start = model.joint_qd_start.numpy()
    labels = list(model.joint_label)
    rows = {}
    for j in joints:
        start, end = qd_start[j], qd_start[j+1] if j+1 < len(qd_start) else model.joint_dof_count
        for d in range(start, end):
            dofs.append(d)
        name = labels[j].split("/World/")[-1]
        arm = "Franka_0" if "Franka_0/" in labels[j] else "Franka_1" if "Franka_1/" in labels[j] else "other"
        import re
        key = f"{arm}:{labels[j].split(arm+'/', 1)[-1] if arm != 'other' else re.sub(r'/World/envs/env_[0-9]+/', '', labels[j])}"
        rows[key] = {field: model.__dict__[field].numpy()[start:end].tolist() for field in
                     ("joint_target_ke", "joint_target_kd", "joint_target_mode", "joint_effort_limit", "joint_armature",
                      "joint_limit_lower", "joint_limit_upper")}
        rows[key]["type"] = int(model.joint_type.numpy()[j])
    return {"bodies": int(np.sum(bw == world)), "dofs": len(dofs), "global_bodies": int(np.sum(bw < 0)),
            "joints": rows, "body_count": int(model.body_count), "world_count": int(model.world_count)}


def _launch(cfg):
    from isaaclab.app.sim_launcher import launch_simulation
    return launch_simulation(cfg, {"headless": True, "visualizer": [], "device": "cpu"})


def _options(args):
    from robotic_cutting_sim.batched.lab_scene import BatchedLabOptions
    return BatchedLabOptions(num_envs=args.num_envs, tool_mount=args.tool_mount, device="cpu", cpu_test_mode=True,
                             seed=42)


def mode_single(args, out):
    import torch
    from isaaclab.envs import DirectRLEnv
    from robotic_cutting_sim.isaac_lab_env import LabCuttingConfig, make_lab_cfg
    options = LabCuttingConfig(device="cpu", material_device="cpu", camera=False, task_id="TRIM-01-R",
                               tool_attachment_mode="physical_grasp")
    cfg = make_lab_cfg(options)
    cfg.sim.physics.solver_cfg.use_mujoco_cpu = False   # MJWarp on the Warp CPU device, as the batch

    class Bare(DirectRLEnv):
        def _setup_scene(self): pass
        def _pre_physics_step(self, actions): pass
        def _apply_action(self): pass
        def _get_observations(self): return {"policy": torch.zeros(1, 1)}
        def _get_rewards(self): return torch.zeros(1)
        def _get_dones(self):
            z = torch.zeros(1, dtype=torch.bool)
            return z, z

    with _launch(cfg):
        from isaaclab_newton.physics import NewtonManager
        env = Bare(cfg)
        model = NewtonManager.get_model()
        receipt = {"newton": _newton_world_receipt(model, -1 if (model.body_world.numpy() < 0).all() else 0),
                   "mujoco": _mjw_receipt(env.sim.physics_manager._solver, 0), "source": "isaac_lab_env.make_lab_cfg",
                   "joint_names": [list(env.scene[f"franka_{i}"].joint_names) for i in range(2)]}
        (out/"receipt.json").write_text(json.dumps(receipt))


def _build(args):
    from robotic_cutting_sim.batched import lab_scene
    from robotic_cutting_sim.rl.lab_batched_env import IsaacLabBatchedCuttingEnv
    task = _task(args.scene_seed)
    options = _options(args)
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    return cfg, options, task, IsaacLabBatchedCuttingEnv


def mode_receipt(args, out):
    cfg, options, task, Env = _build(args)
    with _launch(cfg):
        from isaaclab_newton.physics import NewtonManager
        t0 = time.perf_counter()
        env = Env(cfg, options, task)
        build_s = time.perf_counter()-t0
        if options.tool_mount == "rigid_mount_v1":
            env.reset()   # the B2 settle + snapshot reset is stage S2 (the model receipt needs no reset)
        model = NewtonManager.get_model()
        receipt = {"newton": {e: _newton_world_receipt(model, e) for e in range(args.num_envs)},
                   "mujoco": _mjw_receipt(env.solver, 0), "env": env.receipt(), "build_s": build_s,
                   "separate_worlds": bool(getattr(env.solver, "separate_worlds", getattr(env.solver, "_separate_worlds", True))),
                   "nworld": int(env.solver.mjw_data.nworld), "naconmax": int(env.solver.mjw_data.naconmax),
                   "joint_names": [list(env.scene[f"franka_{i}"].joint_names) for i in range(2)],
                   "body_names_franka0": list(env.robots[0].body_names),
                   "home_commanded": env.home_commanded.numpy().tolist()}
        (out/"receipt.json").write_text(json.dumps(receipt, default=str))


def mode_mask_audit(args, out):
    import torch
    cfg, options, task, Env = _build(args)
    with _launch(cfg):
        env = Env(cfg, options, task)
        env.reset()
        actions = torch.zeros((args.num_envs, 8))
        actions[:, 0] = .5
        for _ in range(3):
            env.step(actions)
        mask = torch.zeros(args.num_envs, dtype=torch.bool)
        mask[args.reset_env] = True
        env.reset_envs(mask)
        audit = env.last_reset_mask_audit
        result = {"mask": mask.tolist(), "world_reset_mask": audit["world_reset_mask"].numpy().astype(bool).tolist(),
                  "fk_reset_mask": audit["fk_reset_mask"].numpy().astype(bool).tolist(),
                  "articulation_world": env.model.articulation_world.numpy().tolist()
                  if getattr(env.model, "articulation_world", None) is not None else None}
        env.step(actions)
        (out/"mask_audit.json").write_text(json.dumps(result))


def scripted_actions(num_envs, step, perturb_env=None):
    import torch
    a = torch.zeros((num_envs, 8))
    phase = step*.35
    a[:, 0] = .6*np.sin(phase)
    a[:, 1] = .4*np.cos(.7*phase)
    a[:, 2] = -.5 if step < 6 else .3
    a[:, 5] = .2*np.sin(.5*phase)
    a[:, 6] = 1.
    if perturb_env is not None:
        a[perturb_env, :3] = torch.tensor([-1., 1., -1.])
    return a


def mode_rollout(args, out):
    import torch
    from isaaclab_newton.physics import NewtonManager
    cfg, options, task, Env = _build(args)
    with _launch(cfg):
        env = Env(cfg, options, task)
        env.reset()
        rows = {"joint_q": [], "joint_qd": [], "body_q": [], "edge": [], "status": [], "commanded": [], "obs": [],
                "nacon": []}
        for step in range(args.steps):
            if args.reset_env is not None and step == args.reset_step:
                mask = torch.zeros(args.num_envs, dtype=torch.bool)
                mask[args.reset_env] = True
                env.reset_envs(mask)
            obs, *_ = env.step(scripted_actions(args.num_envs, step, args.perturb_env))
            state = NewtonManager.get_state_0()
            rows["joint_q"].append(state.joint_q.numpy().copy())
            rows["joint_qd"].append(state.joint_qd.numpy().copy())
            rows["body_q"].append(state.body_q.numpy().copy())
            rows["edge"].append(env.frames.measured_edge.numpy().copy())
            rows["status"].append(env.action_status.numpy().copy())
            rows["commanded"].append(env.commanded.numpy().copy())
            rows["obs"].append(obs["policy"].numpy().copy())
            rows["nacon"].append(int(env.solver.mjw_data.nacon.numpy()[0]))
        np.savez_compressed(out/"rollout.npz", **{k: np.asarray(v) for k, v in rows.items()},
                            body_world=env.model.body_world.numpy(), joint_world=env.model.joint_world.numpy(),
                            joint_q_start=env.model.joint_q_start.numpy(), joint_qd_start=env.model.joint_qd_start.numpy(),
                            coord_start=env.coord_start.numpy())


def _synthetic_coupling_class():
    """Test-only stand-in with the ``MaterialCoupling`` surface. It is NOT a material model: its 'reaction' is a
    deterministic small function of the measured edge so that the coupling plumbing can be checked bitwise."""
    import warp as wp
    from robotic_cutting_sim.rl.lab_batched_env import MaterialCoupling

    @wp.kernel
    def _probe(pose: wp.array2d(dtype=wp.float64), step: int, pending: wp.array2d(dtype=wp.float64),
               pad: wp.array3d(dtype=wp.float64)):
        e = wp.tid()
        s = wp.float64(step)
        for k in range(6):
            pending[e, k] = wp.float64(.05)*wp.sin(wp.float64(37.0)*pose[e, k % 3]+s+wp.float64(k+e))
        for b in range(3):
            for k in range(6):
                pad[e, b, k] = wp.float64(0.0)
                if b > 0:
                    pad[e, b, k] = wp.float64(.01)*wp.cos(wp.float64(11.0)*pose[e, 2]+s+wp.float64(b*7+k))

    class SyntheticCouplingProbe(MaterialCoupling):
        name = "synthetic_test_probe_not_material"

        def __init__(self, num_envs, device, boxes=3):
            with wp.ScopedDevice(device):
                self.pending_wrench = wp.zeros((num_envs, 6), dtype=wp.float64)
                self.pad_wrench = wp.zeros((num_envs, boxes, 6), dtype=wp.float64)
                self.box_centers = wp.zeros(num_envs*boxes, dtype=wp.vec3)
                self.box_halves = wp.zeros(num_envs*boxes, dtype=wp.vec3)
                self.measured_pose = wp.zeros((num_envs, 7), dtype=wp.float64)
                self.cut_input = wp.zeros(num_envs, dtype=wp.int32)
            self.num_envs, self.device, self.step = num_envs, device, 0
            self.inputs, self.outputs = [], []

        def set_measured_poses(self, measured_pose, cut):
            wp.copy(self.measured_pose, measured_pose)
            wp.copy(self.cut_input, cut)

        def advance(self):
            self.inputs.append(self.measured_pose.numpy().copy())
            wp.launch(_probe, self.num_envs, device=self.device,
                      inputs=[self.measured_pose, self.step, self.pending_wrench, self.pad_wrench])
            self.outputs.append((self.pending_wrench.numpy().copy(), self.pad_wrench.numpy().copy()))
            self.step += 1

        def reset_masked(self, mask):
            return None

    return SyntheticCouplingProbe


def mode_coupling(args, out):
    from isaaclab_newton.physics import NewtonManager
    cfg, options, task, Env = _build(args)
    Probe = _synthetic_coupling_class()
    with _launch(cfg):
        env = Env(cfg, options, task, coupling=Probe)
        applied = []
        NewtonManager.register_state_force_callback(
            lambda state: applied.append((env.step_index, env.frames.applied_material.numpy().copy(),
                                          env.frames.applied_com.numpy().copy())))
        env.reset()
        applied.clear()
        edges, boxes = [], []
        for step in range(args.steps):
            env.step(scripted_actions(args.num_envs, step))
            edges.append(env.frames.measured_edge.numpy().copy())
            boxes.append(np.stack([env.coupling.box_centers.numpy().copy(), env.coupling.box_halves.numpy().copy()]))
        coupling = env.coupling
        np.savez_compressed(out/"coupling.npz", edges=np.asarray(edges), inputs=np.asarray(coupling.inputs),
                            pending=np.asarray([o[0] for o in coupling.outputs]),
                            pad=np.asarray([o[1] for o in coupling.outputs]),
                            applied_step=np.asarray([a[0] for a in applied]),
                            applied=np.asarray([a[1] for a in applied]), boxes=np.asarray(boxes))


def mode_grasp(args, out):
    """B2 build-time settle (2 s, all worlds) with per-world gates and snapshot; steps; masked restore of one env."""
    import torch
    from isaaclab_newton.physics import NewtonManager
    args.tool_mount = "physical_grasp_v1"
    cfg, options, task, Env = _build(args)
    with _launch(cfg):
        env = Env(cfg, options, task)
        t0 = time.perf_counter()
        env.reset()
        settle_s = time.perf_counter()-t0
        state = NewtonManager.get_state_0()
        settled = {"body_q": state.body_q.numpy().copy(), "joint_q": state.joint_q.numpy().copy()}
        rows = []
        for step in range(args.steps):
            env.step(scripted_actions(args.num_envs, step) * .3)
            rows.append({"grasp_status": env.grasp_status.numpy().tolist(), "slip_m": env.grasp_slip.numpy().tolist(),
                         "angle_rad": env.grasp_angle.numpy().tolist(),
                         "bilateral": (env.contact_integral.numpy()[:, 8]/np.maximum(env.contact_integral.numpy()[:, 9], 1)).tolist()})
        state = NewtonManager.get_state_0()
        before = {"body_q": state.body_q.numpy().copy(), "joint_q": state.joint_q.numpy().copy()}
        mask = torch.zeros(args.num_envs, dtype=torch.bool)
        mask[args.reset_env] = True
        env.reset_envs(mask)
        audit = env.last_reset_mask_audit
        env.scene.write_data_to_sim()
        env.sim.forward()
        state = NewtonManager.get_state_0()
        restored = {"body_q": state.body_q.numpy().copy(), "joint_q": state.joint_q.numpy().copy()}
        result = {"settle_wall_s": settle_s, "grasp_initialization": env.grasp_initialization,
                  "snapshot": env.snapshot.receipt(), "steps": rows,
                  "world_reset_mask": audit["world_reset_mask"].numpy().astype(bool).tolist(),
                  "fk_reset_mask": audit["fk_reset_mask"].numpy().astype(bool).tolist(),
                  "articulation_world": env.model.articulation_world.numpy().tolist()}
        (out/"grasp.json").write_text(json.dumps(result, default=float))
        np.savez_compressed(out/"grasp.npz", settled_body_q=settled["body_q"], settled_joint_q=settled["joint_q"],
                            restored_body_q=restored["body_q"], restored_joint_q=restored["joint_q"],
                            before_body_q=before["body_q"], before_joint_q=before["joint_q"],
                            body_world=env.model.body_world.numpy(), coord_start=env.coord_start.numpy())


def unreachable_pose(task):
    """HOME edge rotated 150 deg about its local x: kinematically unreachable from HOME (C2 scan: 0.79 m error)."""
    from scipy.spatial.transform import Rotation
    home = np.asarray(task.reference_edge_pose_world(0.), float)
    home[3:] = (Rotation.from_quat(home[3:])*Rotation.from_euler("x", 150, degrees=True)).as_quat()
    return home


def mode_ik_hold(args, out):
    """IK-rejection hold in the env: env ``--reset-env`` gets an unreachable commanded pose at step 2."""
    import torch
    from isaaclab_newton.physics import NewtonManager
    cfg, options, task, Env = _build(args)
    with _launch(cfg):
        env = Env(cfg, options, task)
        env.reset()
        rows = {k: [] for k in ("status", "commanded", "targets", "joint_q", "previous_action", "armed", "cut_digit",
                                "ik_streak", "body_q")}
        for step in range(args.steps):
            if args.reset_env is not None and step == 2:
                mask = torch.zeros(args.num_envs, dtype=torch.bool)
                mask[args.reset_env] = True
                pose = env.commanded.clone()
                pose[args.reset_env] = torch.as_tensor(unreachable_pose(task))
                env.debug_set_commanded(mask, pose)
            env.step(scripted_actions(args.num_envs, step))
            state = NewtonManager.get_state_0()
            rows["status"].append(env.action_status.numpy().copy())
            rows["commanded"].append(env.commanded.numpy().copy())
            rows["targets"].append(env.last_targets.numpy().copy())
            rows["joint_q"].append(state.joint_q.numpy().copy())
            rows["body_q"].append(state.body_q.numpy().copy())
            rows["previous_action"].append(env.previous_action.numpy().copy())
            rows["armed"].append(env.armed.numpy().copy())
            rows["cut_digit"].append(env.cut_digit.numpy().copy())
            rows["ik_streak"].append(env.ik_streak.numpy().copy())
        np.savez_compressed(out/"ik_hold.npz", **{k: np.asarray(v) for k, v in rows.items()},
                            unreachable=unreachable_pose(task), reject_code=env.reject_ik,
                            body_world=env.model.body_world.numpy(), coord_start=env.coord_start.numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", required=True, choices=("receipt", "single", "mask_audit", "rollout", "coupling",
                                                               "grasp", "ik_hold"))
    parser.add_argument("--num-envs", type=int, default=3)
    parser.add_argument("--tool-mount", default="rigid_mount_v1")
    parser.add_argument("--scene-seed", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--perturb-env", type=int, default=None)
    parser.add_argument("--reset-env", type=int, default=None, help="reset / perturb-target env (rollout, grasp, ik_hold)")
    parser.add_argument("--reset-step", type=int, default=5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", None) != "":
        raise SystemExit("S0 worker is CPU only: run with CUDA_VISIBLE_DEVICES=''")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    {"receipt": mode_receipt, "single": mode_single, "mask_audit": mode_mask_audit, "rollout": mode_rollout,
     "coupling": mode_coupling, "grasp": mode_grasp, "ik_hold": mode_ik_hold}[args.mode](args, out)
    (out/"done.json").write_text(json.dumps({"mode": args.mode, "wall_s": time.perf_counter()-start,
                                             "num_envs": args.num_envs, "tool_mount": args.tool_mount}))


if __name__ == "__main__":
    main()
