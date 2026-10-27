"""Standalone dynamic dual-Panda control; no material-force coupling is implied.

Official Isaac 6.2 USD inertias, joints and convex collision meshes are imported
by Newton. The knife is a shape rigidly attached to the first hand, so its pose
always follows measured articulation state. MuJoCo resolves articulated dynamics
and rigid contacts; MPM and RTX remain separate integration concerns.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .recording import ROOT
from .workcell import ASSETS, PARK_JOINTS


@dataclass(frozen=True)
class RobotConfig:
    backend: str = "mujoco_cpu"
    device: str = "cpu"
    control_dt: float = 1 / 60
    substeps: int = 8
    asset_path: str | None = None
    tool_offset_m: tuple = (0., 0., .18)
    tool_half_extents_m: tuple = (.001, .05, .015)
    tool_rotation_xyzw: tuple = (0., 0., 0., 1.)
    tool_shape: str = "box"
    wedge_length_m: float = .012
    wedge_height_m: float = .004
    wedge_half_width_m: float = .025

    def __post_init__(self):
        if self.backend not in ("mujoco_cpu", "mujoco_warp"):
            raise ValueError("Robot backend must be mujoco_cpu or mujoco_warp")
        if not np.isfinite(self.control_dt) or self.control_dt <= 0 or not isinstance(self.substeps, int) or self.substeps < 1:
            raise ValueError("Positive control_dt and substeps are required")
        if len(self.tool_offset_m) != 3 or not np.all(np.isfinite(self.tool_offset_m)):
            raise ValueError("tool_offset_m must contain three finite values")
        if (len(self.tool_half_extents_m) != 3 or not np.all(np.isfinite(self.tool_half_extents_m))
                or not np.all(np.asarray(self.tool_half_extents_m) > 0)):
            raise ValueError("tool_half_extents_m must be positive")
        if self.tool_shape not in ("box", "cut_wedge"):
            raise ValueError("tool_shape must be box or cut_wedge")
        validate_pose([0, 0, 0, *self.tool_rotation_xyzw])
        if min(self.wedge_length_m, self.wedge_height_m, self.wedge_half_width_m) <= 0:
            raise ValueError("Wedge dimensions must be positive")


def validate_pose(pose):
    pose = np.asarray(pose, dtype=np.float64)
    if pose.shape != (7,) or not np.all(np.isfinite(pose)):
        raise ValueError("Expected finite XYZ + XYZW pose")
    norm = np.linalg.norm(pose[3:])
    if norm < 1e-8:
        raise ValueError("Quaternion must be nonzero")
    pose = pose.copy()
    pose[3:] /= norm
    return pose


def compose_tool_pose(hand_pose, offset, rotation_xyzw=None):
    """Compose a fixed hand-to-tool translation, preserving XYZW orientation."""
    hand = validate_pose(hand_pose)
    rot = Rotation.from_quat(hand[3:])
    orientation = hand[3:] if rotation_xyzw is None else (rot * Rotation.from_quat(rotation_xyzw)).as_quat()
    return np.r_[hand[:3] + rot.apply(offset), orientation]


def wedge_mesh(length=.012, height=.004, half_width=.025):
    """Closed triangular extrusion matching TLMPM._wedge_sdf exactly."""
    vertices = np.array([[0, -half_width, 0], [-length, -half_width, height],
                         [-length, -half_width, -.0004], [0, half_width, 0],
                         [-length, half_width, height], [-length, half_width, -.0004]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [3, 4, 5], [0, 1, 4], [0, 4, 3],
                      [1, 2, 5], [1, 5, 4], [2, 0, 3], [2, 3, 5]], dtype=np.int32)
    return vertices, faces[:, ::-1].copy()


class DualFrankaController:
    """Joint drives + Newton analytic-Jacobian IK, with measured proprioception.

    ``step_joints`` receives 7 cutting-arm joint positions. ``step_cartesian``
    receives the desired world knife-center pose, not the hand/gripper pose.
    The second arm always receives its park targets. No force enters the public
    proprioception dictionary. Diagnostic contacts are explicitly separate.
    """

    def __init__(self, config=None):
        import warp as wp
        import newton
        import newton.ik as ik
        from newton.solvers import SolverMuJoCo
        from pxr import Usd

        self.config = config or RobotConfig()
        self.wp, self.newton = wp, newton
        wp.init()
        self.device = wp.get_device(self.config.device)
        path = Path(self.config.asset_path) if self.config.asset_path else (
            ROOT / "artifacts/workcell_asset_cache" / ASSETS["franka"])
        if not path.is_file():
            raise FileNotFoundError(f"Official Franka USD cache missing: {path}. Run the workcell asset fetch first.")
        self.asset_path = path.resolve()
        start = time.perf_counter()
        with wp.ScopedDevice(self.device):
            single = newton.ModelBuilder()
            single.rigid_gap = .001
            SolverMuJoCo.register_custom_attributes(single)
            # Session edits leave the cached official asset and its hashes intact.
            stage = Usd.Stage.Open(str(path))
            stage.SetEditTarget(stage.GetSessionLayer())
            root = stage.GetDefaultPrim()
            root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("mujoco")
            root.GetVariantSets().GetVariantSet("Colliders").SetVariantSelection("convex_hulls")
            root.GetAttribute("newton:selfCollisionEnabled").Set(True)
            single.add_usd(stage, floating=False, enable_self_collisions=True,
                           collapse_fixed_joints=False, load_visual_shapes=False,
                           skip_mesh_approximation=True)
            if single.joint_coord_count != 9 or single.joint_dof_count != 9:
                raise RuntimeError("Expected 7 arm + 2 finger DOFs in the official Panda asset")
            self.hand_local = next(i for i, label in enumerate(single.body_label) if label.endswith("/panda_hand"))
            self.bodies_per_arm = single.body_count
            single.joint_q[:] = list(PARK_JOINTS) + [.04, .04]
            single.joint_target_q[:] = single.joint_q
            single.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)] * 9
            single.joint_target_ke[:] = [1000.] * 4 + [600.] * 3 + [350., 350.]
            single.joint_target_kd[:] = [80.] * 4 + [40.] * 3 + [10., 10.]
            single.joint_effort_limit[:] = [87.] * 4 + [12.] * 3 + [20., 20.]
            single.joint_armature[:] = [.2] * 4 + [.07] * 3 + [.1, .1]
            single.shape_gap[:] = [.001] * len(single.shape_gap)
            # Gravity compensation is an explicit low-level controller setting.
            single.custom_attributes["mujoco:gravcomp"].values = {i: 1. for i in range(single.body_count)}
            single.custom_attributes["mujoco:jnt_actgravcomp"].values = {i: True for i in range(9)}
            builder = newton.ModelBuilder()
            SolverMuJoCo.register_custom_attributes(builder)
            builder.add_builder(single, xform=wp.transform(wp.vec3(-.46, .12, -.01)), label_prefix="arm_0")
            builder.add_builder(single, xform=wp.transform(wp.vec3(.46, .12, -.01),
                                wp.quat_from_axis_angle(wp.vec3(0, 0, 1), wp.pi)), label_prefix="arm_1")
            builder.joint_q[9] = -np.pi / 2  # park toward +Y, away from the cutting arm
            builder.joint_target_q[9] = -np.pi / 2
            cfg = newton.ModelBuilder.ShapeConfig(density=7800., mu=.35, gap=.001)
            mount = wp.transform(wp.vec3(*self.config.tool_offset_m), wp.quat(*self.config.tool_rotation_xyzw))
            if self.config.tool_shape == "cut_wedge":
                vertices, faces = wedge_mesh(self.config.wedge_length_m, self.config.wedge_height_m,
                                             self.config.wedge_half_width_m)
                self.tool_shape = builder.add_shape_convex_hull(self.hand_local, xform=mount,
                    mesh=newton.Mesh(vertices, faces.ravel()), cfg=cfg, label="knife_fixed_to_arm_0_hand")
            else:
                self.tool_shape = builder.add_shape_box(self.hand_local, xform=mount,
                    hx=self.config.tool_half_extents_m[0], hy=self.config.tool_half_extents_m[1],
                    hz=self.config.tool_half_extents_m[2], cfg=cfg, label="knife_fixed_to_arm_0_hand")
            # Shape-on-link attachment is exactly rigid, including added inertia.
            self.table_shape = builder.add_shape_box(-1, xform=wp.transform(wp.vec3(0, 0, -.035)),
                hx=.9, hy=.6, hz=.025, label="table")
            self.model = builder.finalize(device=self.device)
            self.initial_q = np.asarray(builder.joint_q, dtype=np.float32)
            self.lower = self.model.joint_limit_lower.numpy()
            self.upper = self.model.joint_limit_upper.numpy()
            self.state = self.model.state()
            self.next_state = self.model.state()
            self.control = self.model.control()
            self.ik_state = self.model.state()
            self.ik_q = wp.clone(self.model.joint_q).reshape((1, 18))
            self.position_objective = ik.IKObjectivePosition(link_index=self.hand_local,
                link_offset=wp.vec3(*self.config.tool_offset_m),
                target_positions=wp.array([wp.vec3(0, 0, .1)], dtype=wp.vec3, device=self.device))
            self.rotation_objective = ik.IKObjectiveRotation(link_index=self.hand_local,
                link_offset_rotation=wp.quat(*self.config.tool_rotation_xyzw),
                target_rotations=wp.array([wp.vec4(1, 0, 0, 0)], dtype=wp.vec4, device=self.device))
            limit_objective = ik.IKObjectiveJointLimit(self.model.joint_limit_lower,
                                                      self.model.joint_limit_upper, weight=10.)
            self.ik = ik.IKSolver(model=self.model, n_problems=1,
                objectives=[self.position_objective, self.rotation_objective, limit_objective],
                lambda_initial=.01, jacobian_mode=ik.IKJacobianType.ANALYTIC)
            self.solver = SolverMuJoCo(self.model, use_mujoco_cpu=self.config.backend == "mujoco_cpu",
                separate_worlds=False, iterations=50, nconmax=256, njmax=1024,
                integrator="implicitfast", cone="elliptic")
            self.reset()
        self.construction_wall_s = time.perf_counter() - start

    def reset(self, joints=None):
        q = self.initial_q.copy()
        if joints is not None:
            q[:7] = self._validate_joints(joints)
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.solver.reset(self.state)
        self.state.joint_q.assign(q)
        self.state.joint_qd.zero_()
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.control.joint_target_q.assign(q)
        self.targets = q.copy()
        self.time_s = 0.
        return self.proprioception()

    def _validate_joints(self, joints):
        q = np.asarray(joints, dtype=np.float32)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise ValueError("Expected seven finite joint positions")
        if np.any(q < self.lower[:7]) or np.any(q > self.upper[:7]):
            raise ValueError("Joint command violates Panda joint limits")
        return q

    def inverse_kinematics(self, tool_pose, iterations=48):
        pose = validate_pose(tool_pose)
        self.ik_q.assign(self.state.joint_q.numpy().reshape(1, 18))
        self.position_objective.set_target_position(0, self.wp.vec3(*pose[:3]))
        self.rotation_objective.set_target_rotation(0, self.wp.vec4(*pose[3:]))
        self.ik.step(self.ik_q, self.ik_q, iterations=iterations)
        q = self.ik_q.numpy()[0]
        if not np.all(np.isfinite(q)):
            raise RuntimeError("IK produced non-finite joint positions")
        q = np.clip(q, self.lower, self.upper)
        self.ik_state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.ik_state.joint_q, self.model.joint_qd, self.ik_state)
        actual = compose_tool_pose(self.ik_state.body_q.numpy()[self.hand_local], self.config.tool_offset_m,
                                   self.config.tool_rotation_xyzw)
        position_error = float(np.linalg.norm(actual[:3] - pose[:3]))
        rotation_error = float((Rotation.from_quat(actual[3:]).inv() * Rotation.from_quat(pose[3:])).magnitude())
        return q[:7].copy(), {"position_error_m": position_error, "rotation_error_rad": rotation_error,
                             "converged": position_error < .001 and rotation_error < .01}

    def forward_kinematics(self, joints):
        q = self.state.joint_q.numpy().copy()
        q[:7] = self._validate_joints(joints)
        self.ik_state.joint_q.assign(q)
        self.newton.eval_fk(self.model, self.ik_state.joint_q, self.model.joint_qd, self.ik_state)
        return compose_tool_pose(self.ik_state.body_q.numpy()[self.hand_local], self.config.tool_offset_m,
                                 self.config.tool_rotation_xyzw)

    def step_joints(self, joints=None, tool_wrench_world=None):
        """Advance dynamics, optionally applying an internal diagnostic wrench.

        ``tool_wrench_world`` is [Fx,Fy,Fz,Tx,Ty,Tz] at the knife center in
        world coordinates. It is not an observation or a material-force claim.
        The wrench is shifted to the hand COM before the articulated solve.
        """
        if joints is not None:
            self.targets[:7] = self._validate_joints(joints)
        wrench = None if tool_wrench_world is None else np.asarray(tool_wrench_world, dtype=float)
        if wrench is not None and (wrench.shape != (6,) or not np.all(np.isfinite(wrench))):
            raise ValueError("Expected a finite world-frame force + torque 6-vector")
        self.targets[9:] = self.initial_q[9:]
        self.control.joint_target_q.assign(self.targets)
        start = time.perf_counter()
        wrench_work = 0.
        dt = self.config.control_dt / self.config.substeps
        for _ in range(self.config.substeps):
            self.state.clear_forces()
            if wrench is not None:
                self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
                hand = self.state.body_q.numpy()[self.hand_local]
                rot = Rotation.from_quat(hand[3:])
                center = hand[:3] + rot.apply(self.model.body_com.numpy()[self.hand_local])
                tool = compose_tool_pose(hand, self.config.tool_offset_m, self.config.tool_rotation_xyzw)
                forces = np.zeros((self.model.body_count, 6), dtype=np.float32)
                forces[self.hand_local, :3] = wrench[:3]
                forces[self.hand_local, 3:] = wrench[3:] + np.cross(tool[:3] - center, wrench[:3])
                self.state.body_f.assign(forces)
            self.solver.step(self.state, self.next_state, self.control, None, dt)
            self.state, self.next_state = self.next_state, self.state
            if wrench is not None:
                self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
                after = compose_tool_pose(self.state.body_q.numpy()[self.hand_local], self.config.tool_offset_m,
                                          self.config.tool_rotation_xyzw)
                rotation_delta = (Rotation.from_quat(after[3:]) * Rotation.from_quat(tool[3:]).inv()).as_rotvec()
                wrench_work += float(np.dot(wrench[:3], after[:3] - tool[:3]) + np.dot(wrench[3:], rotation_delta))
        # MuJoCo's cached xpos precedes its final integration. Derive published
        # link poses from the final joint state at the same timestamp.
        self.newton.eval_fk(self.model, self.state.joint_q, self.state.joint_qd, self.state)
        self.wp.synchronize_device(self.device)
        self.time_s += self.config.control_dt
        return self.proprioception(), {"physics_wall_s": time.perf_counter() - start,
                                      "applied_tool_wrench_work_j": wrench_work}

    def step_cartesian(self, tool_pose, tool_wrench_world=None):
        start = time.perf_counter()
        joints, info = self.inverse_kinematics(tool_pose)
        info["ik_wall_s"] = time.perf_counter() - start
        if not info["converged"]:
            raise ValueError(f"Knife pose is not reachable within the IK tolerance: {info}")
        state, step_info = self.step_joints(joints, tool_wrench_world=tool_wrench_world)
        return state, {**info, **step_info}

    def proprioception(self):
        q = self.state.joint_q.numpy().reshape(2, 9)
        qdot = self.state.joint_qd.numpy().reshape(2, 9)
        return {"q": q[:, :7].copy(), "qdot": qdot[:, :7].copy(),
                "tool_pose": compose_tool_pose(self.state.body_q.numpy()[self.hand_local],
                                                self.config.tool_offset_m, self.config.tool_rotation_xyzw),
                "timestamp_s": self.time_s}

    def diagnostic_contacts(self):
        """Rigid-solver contacts only; excluded from formal observations."""
        import mujoco
        if self.config.backend != "mujoco_cpu":
            return {"status": "NOT_RUN", "reason": "Contact export currently supports the CPU backend"}
        data = self.solver.mj_data
        records = []
        for i in range(data.ncon):
            contact = data.contact[i]
            wrench = np.zeros(6)
            if contact.efc_address >= 0:
                mujoco.mj_contactForce(self.solver.mj_model, data, i, wrench)
            names = [mujoco.mj_id2name(self.solver.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g))
                     for g in contact.geom]
            records.append({"geoms": names, "distance_m": float(contact.dist),
                            "active_constraint": bool(contact.efc_address >= 0),
                            "normal_force_N": float(wrench[0])})
        return {"status": "PASS", "count": len(records),
                "active_count": sum(r["active_constraint"] for r in records), "contacts": records}

    def description(self):
        return {"config": asdict(self.config), "asset": str(self.asset_path),
                "articulations": 2, "arm_dof": 7, "finger_dof": 2,
                "bodies": self.model.body_count, "collider_variant": "convex_hulls",
                "self_collision_enabled": True, "adjacent_links_filtered": True,
                "knife_attachment": "rigid shape on hand link, contributes mass and inertia",
                "gravity_compensation": True, "proprioception_source": "measured articulation state",
                "diagnostic_tool_wrench_input": "world-frame force then torque at knife center; shifted to hand COM",
                "material_coupling": "NOT_IMPLEMENTED", "bidirectional_material_force": False,
                "isaac_lab_articulation": "NOT_RUN"}
