"""P1-1b B2 grasp follow-up: robot-only diagnosis of the physical grasp under the recorded TRIM commands.

One Isaac Lab kit-less build per process (Warp CPU device with ``--device cpu`` = ``cpu_test_mode``, or one GPU).
Workload B, ``physical_grasp_v1``, no material (robot-only), driver ``recorded_command_policy_v1`` (the WP-17 Native
commanded edge targets, the S3 input) or ``hold``. Eager stepping (no CUDA graph) so that every control step can
be read on the host: this is a DIAGNOSTIC probe, not a timing row.

Per control step (after the 10th substep) it records for every env:

* the tool pose relative to the hand, minus the settled reference: slip vector and rotation vector, both in the
  HAND frame (the twist axis is visible), and the grasp-monitor slip / angle / status;
* finger joint positions and velocities, hand and tool linear / angular velocity (world);
* every MuJoCo-Warp contact of the tool: owner (left / right finger, other), the two geoms, distance, position in
  the TOOL frame, the contact-frame force (normal, tangential 1/2, torsional, rolling for condim 6), the friction
  coefficients, the elliptic-cone usage ``sqrt(sum (f_i/mu_i)^2) / f_n``;
* per finger: contact count, normal force, tangential force, torque about the grip centre around the pad normal;
* the gravity torque of the tool about the grasp centre (hand frame).

Example (CPU, no GPU lock needed)::

    CUDA_VISIBLE_DEVICES='' TMPDIR=/mnt/data/ningan/tmp PYTHONPATH=src .deps/isaaclab-venv/bin/python \
        scripts/rl_ready/probe_lab_b2_grasp.py --device cpu --scene-seeds 512 --steps 250 \
        --out artifacts/rl_ready/p1_1b_b2/diag_cpu_point_001

``--grip-profile`` selects a versioned grip contact profile (``lab_scene.GRIP_PROFILES``); ``--solver-set k=v``
overrides one MJWarp solver option for a DIAGNOSTIC sweep only (recorded in the row, never a profile).
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _commit():
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no", "--", "src", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return {"sha": sha, "tracked_src_or_scripts_modified": bool(dirty)}
    except Exception:
        return None


def _parse_value(text):
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(text.lower(), text)


def _quat_to_matrix(q):
    from scipy.spatial.transform import Rotation
    return Rotation.from_quat(np.asarray(q, float)).as_matrix()


def _rotvec(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_rotvec()


class ContactReader:
    """Host-side reader of the MuJoCo-Warp contact buffers of the Newton solver (diagnostic, reads per call)."""

    def __init__(self, env):
        import mujoco
        self.env = env
        solver = env.solver
        self.m, self.d = solver.mjw_model, solver.mjw_data
        self.geom_to_shape = solver.mjc_geom_to_newton_shape.numpy()      # [nworld, ngeom]
        self.shape_body = env.model.shape_body.numpy()
        self.shape_label = list(env.model.shape_label)
        self.cone = int(self.m.opt.cone)
        self.mj_model = solver.mj_model
        self.geom_names = [mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
                           for g in range(self.mj_model.ngeom)]
        self.tool = env.tables["tool"]
        self.finger = env.tables["finger"]
        self.hand = env.tables["hand"]

    def geom_receipt(self):
        """condim / friction / solref / solimp / margin / gap / size / type of every tool and finger geom (world 0)."""
        rows = []
        m = self.mj_model
        bodies = {int(self.tool[0]): "tool", int(self.finger[0, 0]): "left_finger", int(self.finger[0, 1]): "right_finger"}
        for g in range(m.ngeom):
            shape = int(self.geom_to_shape[0, g])
            if shape < 0:
                continue
            owner = bodies.get(int(self.shape_body[shape]))
            if owner is None:
                continue
            rows.append({"geom": g, "name": self.geom_names[g], "shape": shape, "label": self.shape_label[shape],
                         "owner": owner, "type": int(m.geom_type[g]), "size": m.geom_size[g].tolist(),
                         "condim": int(m.geom_condim[g]), "friction": m.geom_friction[g].tolist(),
                         "solref": m.geom_solref[g].tolist(), "solimp": m.geom_solimp[g].tolist(),
                         "solmix": float(m.geom_solmix[g]), "margin": float(m.geom_margin[g]),
                         "gap": float(m.geom_gap[g]), "priority": int(m.geom_priority[g]),
                         "contype": int(m.geom_contype[g]), "conaffinity": int(m.geom_conaffinity[g])})
        opt = m.opt
        options = {"cone": int(opt.cone), "impratio": float(opt.impratio), "iterations": int(opt.iterations),
                   "ls_iterations": int(opt.ls_iterations), "tolerance": float(opt.tolerance),
                   "noslip_iterations": int(opt.noslip_iterations), "timestep": float(opt.timestep),
                   "integrator": int(opt.integrator), "solver": int(opt.solver),
                   "disableflags": int(opt.disableflags), "enableflags": int(opt.enableflags),
                   "mjw_impratio_invsqrt": self.m.opt.impratio_invsqrt.numpy().tolist()
                   if hasattr(self.m.opt, "impratio_invsqrt") else None}
        return {"geoms": rows, "options": options}

    def read(self, body_q):
        d = self.d
        nacon = int(d.nacon.numpy()[0])
        c = d.contact
        geom = c.geom.numpy()[:nacon]
        worldid = c.worldid.numpy()[:nacon]
        pos = c.pos.numpy()[:nacon]
        frame = c.frame.numpy()[:nacon]
        dist = c.dist.numpy()[:nacon]
        dim = c.dim.numpy()[:nacon]
        friction = c.friction.numpy()[:nacon]
        addr = c.efc_address.numpy()[:nacon]
        efc_force = d.efc.force.numpy()
        out = [dict(contacts=[], fingers={"left_finger": None, "right_finger": None}) for _ in range(len(self.tool))]
        for i in range(nacon):
            w = int(worldid[i])
            if w < 0 or w >= len(self.tool):
                continue
            g1, g2 = int(geom[i, 0]), int(geom[i, 1])
            s1, s2 = int(self.geom_to_shape[w, g1]), int(self.geom_to_shape[w, g2])
            b1 = int(self.shape_body[s1]) if s1 >= 0 else -1
            b2 = int(self.shape_body[s2]) if s2 >= 0 else -1
            tool = int(self.tool[w])
            if tool not in (b1, b2):
                continue
            other = b2 if b1 == tool else b1
            sign = 1. if b2 == tool else -1.   # MuJoCo: force on geom2 along +normal (normal points geom1 -> geom2)
            owner = ("left_finger" if other == int(self.finger[w, 0]) else
                     "right_finger" if other == int(self.finger[w, 1]) else "other")
            k = int(dim[i])
            f = np.zeros(6)
            for j in range(k):
                a = int(addr[i, j])
                if 0 <= a < efc_force.shape[1]:
                    f[j] = efc_force[w, a]
            mu = friction[i]
            fn = f[0]
            usage = None
            if fn > 1e-9 and k > 1:
                ratios = [f[j]/mu[j-1] for j in range(1, k) if mu[j-1] > 0]
                usage = float(np.sqrt(np.sum(np.square(ratios)))/fn) if ratios else None
            T = body_q[tool]
            R = _quat_to_matrix(T[3:7])
            p_tool = R.T @ (pos[i]-T[:3])
            n_world = frame[i][0]*sign      # normal pushing INTO the tool
            out[w]["contacts"].append(dict(owner=owner, geoms=[self.geom_names[g1], self.geom_names[g2]],
                                           dist=float(dist[i]), pos_tool=p_tool.tolist(),
                                           normal_into_tool_world=n_world.tolist(), dim=k,
                                           force_contact_frame=f[:k].tolist(), friction=mu[:max(k-1, 1)].tolist(),
                                           cone_usage=usage))
        for w, row in enumerate(out):
            for name in ("left_finger", "right_finger"):
                cs = [c for c in row["contacts"] if c["owner"] == name]
                row["fingers"][name] = {
                    "count": len(cs), "normal_n": float(sum(c["force_contact_frame"][0] for c in cs)),
                    "tangential_n": float(sum(np.linalg.norm(c["force_contact_frame"][1:3]) for c in cs)),
                    "torsional_nm": float(sum(c["force_contact_frame"][3] if c["dim"] >= 4 else 0. for c in cs)),
                    "max_cone_usage": max([c["cone_usage"] for c in cs if c["cone_usage"] is not None], default=None),
                    "min_dist": min([c["dist"] for c in cs], default=None),
                    "geoms": sorted({g for c in cs for g in c["geoms"]})}
            row["other_count"] = sum(1 for c in row["contacts"] if c["owner"] == "other")
        return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu (cpu_test_mode) or cuda")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--scene-seeds", type=int, nargs="*", default=[512])
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--driver", default="recorded", choices=("recorded", "hold"))
    parser.add_argument("--grip-profile", default="legacy_point_contact")
    parser.add_argument("--solver-set", action="append", default=[], help="DIAGNOSTIC MJWarp option override k=v")
    parser.add_argument("--mjw-disable", action="append", default=[],
                        help="DIAGNOSTIC: set a MuJoCo-Warp DisableBit (e.g. NATIVECCD) on the built model before reset")
    parser.add_argument("--non-strict", action="store_true")
    parser.add_argument("--contacts-every", type=int, default=1, help="record full contact lists every k steps")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cpu = args.device == "cpu"
    if cpu and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "":
        raise SystemExit("CPU diagnosis: set CUDA_VISIBLE_DEVICES='' (no GPU is used)")
    if not cpu and os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise SystemExit("GPU diagnosis: set CUDA_VISIBLE_DEVICES to the GPU this job holds the flock for")
    t_start = time.perf_counter()
    import torch
    import warp as wp
    from isaaclab.app.sim_launcher import launch_simulation
    from probe_lab_batched_gpu import RecordedDriver, _task
    from robotic_cutting_sim.batched import lab_scene
    from robotic_cutting_sim.rl.lab_batched_env import IsaacLabBatchedCuttingEnv
    from robotic_cutting_sim.physical_tool import ToolGraspConfig  # noqa: F401
    device = "cpu" if cpu else "cuda:0"
    seeds = [args.scene_seeds[e % len(args.scene_seeds)] for e in range(args.num_envs)]
    compiled = {seed: _task(seed) for seed in dict.fromkeys(seeds)}
    task = compiled[seeds[0]]
    env_tasks = [compiled[s] for s in seeds]
    options = lab_scene.BatchedLabOptions(num_envs=args.num_envs, tool_mount="physical_grasp_v1", device=device,
                                          cpu_test_mode=cpu, use_cuda_graph=False, strict=not args.non_strict,
                                          material=None, grip_contact_profile=args.grip_profile)
    cfg = lab_scene.make_batched_lab_cfg(options, task)
    overrides = {}
    for item in args.solver_set:
        key, value = item.split("=", 1)
        if not hasattr(cfg.sim.physics.solver_cfg, key):
            raise SystemExit(f"unknown MJWarpSolverCfg field {key!r}")
        setattr(cfg.sim.physics.solver_cfg, key, _parse_value(value))
        overrides[key] = _parse_value(value)
    row = {"probe": "b2_grasp_diag", "device": device, "N": args.num_envs, "scene_seeds": seeds, "driver": args.driver,
           "steps": args.steps, "grip_profile": args.grip_profile, "diagnostic_solver_overrides": overrides,
           "use_cuda_graph": False, "material": "none (robot-only)", "commit": _commit(),
           "fidelity": options.fidelity(), "label": "DIAGNOSTIC (eager, host reads every step; not a timing row)",
           "command": " ".join(sys.argv)}
    if not cpu:
        from probe_lab_batched_gpu import _gpu_row
        row["gpu"] = _gpu_row(args.gpu_index)
        row["gpu_sharing"] = "SHARED_GPU"
    trace = {k: [] for k in ("slip_hand", "rotvec_hand", "monitor_slip", "monitor_angle", "monitor_status",
                             "finger_q", "finger_qd", "hand_v", "hand_w", "tool_v", "tool_w", "count_left",
                             "count_right", "count_other", "fn_left", "fn_right", "ft_left", "ft_right",
                             "tors_left", "tors_right", "usage_left", "usage_right", "gravity_torque_hand",
                             "commanded", "edge", "action_status", "hand_pose", "tool_pose")}
    contact_dump = []
    with launch_simulation(cfg, {"headless": True, "visualizer": [], "device": device}):
        from isaaclab_newton.physics import NewtonManager
        env = IsaacLabBatchedCuttingEnv(cfg, options, task, tasks=env_tasks)
        if args.mjw_disable:
            from mujoco_warp._src.types import DisableBit
            for name in args.mjw_disable:
                env.solver.mjw_model.opt.disableflags |= int(getattr(DisableBit, name))
            overrides["mjw_disable"] = list(args.mjw_disable)
        reader = ContactReader(env)
        row["geom_receipt"] = reader.geom_receipt()
        row["geom_receipt"]["options"]["mjw_disableflags"] = int(env.solver.mjw_model.opt.disableflags)
        row["grip_material"] = getattr(env, "grip_material_receipt", None)
        row["contact_solver"] = getattr(env, "contact_solver_receipt", None)
        t0 = time.perf_counter()
        env.reset()
        row["reset_s"] = time.perf_counter()-t0
        row["grasp_initialization"] = env.grasp_initialization
        model = env.model
        mass = model.body_mass.numpy()
        com_local = model.body_com.numpy()
        ref_R = env.reference_R.numpy()
        ref_p = env.reference_p.numpy()
        driver = RecordedDriver(env, seeds, device) if args.driver == "recorded" else None
        if driver is not None:
            row["recorded_policy"] = driver.reports
        names = list(env.robots[0].joint_names)
        fcols = [names.index("panda_finger_joint1"), names.index("panda_finger_joint2")]
        settle_contacts = reader.read(NewtonManager.get_state_0().body_q.numpy())
        row["settled_contacts"] = settle_contacts
        t_loop = time.perf_counter()
        error = None
        for step in range(1, args.steps+1):
            if driver is not None:
                action, _ = driver()
            else:
                action = torch.zeros((args.num_envs, 8), device=device)
                action[:, 6] = -1.
            try:
                env.step(action)
            except Exception as exc:     # strict-mode raise: recorded, the trace so far is kept
                error = f"step {step}: {type(exc).__name__}: {exc}"[:2000]
                break
            state = NewtonManager.get_state_0()
            body_q = state.body_q.numpy()
            body_qd = state.body_qd.numpy()
            contacts = reader.read(body_q)
            jp = env.robots[0].data.joint_pos.torch.cpu().numpy()
            jv = env.robots[0].data.joint_vel.torch.cpu().numpy()
            per = {k: [] for k in trace}
            for e in range(args.num_envs):
                h, t = int(env.tables["hand"][e]), int(env.tables["tool"][e])
                Rh, Rt = _quat_to_matrix(body_q[h, 3:]), _quat_to_matrix(body_q[t, 3:])
                R_rel = Rh.T @ Rt
                p_rel = Rh.T @ (body_q[t, :3]-body_q[h, :3])
                dR = ref_R[e].T @ R_rel
                # slip and rotation since settling, expressed in the HAND frame
                per["slip_hand"].append((p_rel-ref_p[e]).tolist())
                per["rotvec_hand"].append((ref_R[e] @ _rotvec(dR)).tolist())
                per["finger_q"].append(jp[e, fcols].tolist())
                per["finger_qd"].append(jv[e, fcols].tolist())
                # Newton body_qd: (linear, angular) world
                per["hand_v"].append(body_qd[h, :3].tolist())
                per["hand_w"].append(body_qd[h, 3:].tolist())
                per["tool_v"].append(body_qd[t, :3].tolist())
                per["tool_w"].append(body_qd[t, 3:].tolist())
                f = contacts[e]["fingers"]
                for side, key in (("left_finger", "left"), ("right_finger", "right")):
                    per[f"count_{key}"].append(f[side]["count"])
                    per[f"fn_{key}"].append(f[side]["normal_n"])
                    per[f"ft_{key}"].append(f[side]["tangential_n"])
                    per[f"tors_{key}"].append(f[side]["torsional_nm"])
                    per[f"usage_{key}"].append(np.nan if f[side]["max_cone_usage"] is None else f[side]["max_cone_usage"])
                per["count_other"].append(contacts[e]["other_count"])
                # gravity torque about the hand-frame grip centre (tool origin = handle centre)
                com_w = body_q[t, :3]+Rt @ com_local[t]
                g = np.array([0., 0., -9.81])*mass[t]
                per["gravity_torque_hand"].append((Rh.T @ np.cross(com_w-body_q[t, :3], g)).tolist())
                per["hand_pose"].append(body_q[h].tolist())
                per["tool_pose"].append(body_q[t].tolist())
            per["monitor_slip"] = env.grasp_slip.numpy().tolist()
            per["monitor_angle"] = env.grasp_angle.numpy().tolist()
            per["monitor_status"] = env.grasp_status.numpy().tolist()
            per["commanded"] = env.commanded.cpu().numpy().tolist()
            per["edge"] = env.frames.measured_edge.numpy().tolist()
            per["action_status"] = env.action_status.cpu().numpy().tolist()
            for k in trace:
                trace[k].append(per[k])
            if args.contacts_every and (step % args.contacts_every == 0 or step < 5):
                contact_dump.append({"step": step, "worlds": [c["contacts"] for c in contacts]})
        row["loop_s"] = time.perf_counter()-t_loop
        row["steps_completed"] = len(trace["slip_hand"])
        row["error"] = error
        row["termination_reason"] = env.termination_reason.cpu().tolist()
        row["tool_mass_kg"] = float(mass[int(env.tables["tool"][0])])
        row["tool_com_local_m"] = com_local[int(env.tables["tool"][0])].tolist()
        row["tool_inertia_local"] = model.body_inertia.numpy()[int(env.tables["tool"][0])].tolist()
    arrays = {k: np.asarray(v, float) for k, v in trace.items() if v}
    np.savez_compressed(out/"diag_trace.npz", **arrays)
    (out/"contacts.json").write_text(json.dumps(contact_dump))
    if arrays:
        slip = np.linalg.norm(arrays["slip_hand"], axis=-1)
        rot = np.linalg.norm(arrays["rotvec_hand"], axis=-1)
        status = arrays["monitor_status"]
        row["summary"] = {
            "max_slip_m": slip.max(0).tolist(), "max_rotation_rad": rot.max(0).tolist(),
            "slip_at": {str(s): slip[min(s, len(slip))-1].tolist() for s in (20, 50, 100, 150, 200, 250) if s <= len(slip)},
            "rotation_at": {str(s): rot[min(s, len(rot))-1].tolist() for s in (20, 50, 100, 150, 200, 250) if s <= len(rot)},
            "first_slipping_step": [int(np.argmax(status[:, e] >= 1))+1 if (status[:, e] >= 1).any() else None
                                    for e in range(status.shape[1])],
            "first_lost_step": [int(np.argmax(status[:, e] == 2))+1 if (status[:, e] == 2).any() else None
                                for e in range(status.shape[1])],
            "final_status": status[-1].tolist(),
            "mean_count_left": arrays["count_left"].mean(0).tolist(), "mean_count_right": arrays["count_right"].mean(0).tolist(),
            "min_count_left": arrays["count_left"].min(0).tolist(), "min_count_right": arrays["count_right"].min(0).tolist(),
            "mean_fn_left": arrays["fn_left"].mean(0).tolist(), "mean_fn_right": arrays["fn_right"].mean(0).tolist(),
            "max_usage_left": np.nanmax(arrays["usage_left"], 0).tolist() if np.isfinite(arrays["usage_left"]).any() else None,
            "max_usage_right": np.nanmax(arrays["usage_right"], 0).tolist() if np.isfinite(arrays["usage_right"]).any() else None,
            "finger_q_range": [arrays["finger_q"].min(0).tolist(), arrays["finger_q"].max(0).tolist()],
            "max_other_contacts": arrays["count_other"].max(0).tolist(),
        }
    row["wall_s_total_process"] = time.perf_counter()-t_start
    (out/"row.json").write_text(json.dumps(row, indent=1, default=str))
    print(json.dumps({k: row.get(k) for k in ("grip_profile", "diagnostic_solver_overrides", "grasp_initialization",
                                              "steps_completed", "error", "termination_reason", "summary")},
                     indent=1, default=str))


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
