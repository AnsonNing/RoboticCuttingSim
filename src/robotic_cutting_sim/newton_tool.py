"""Independent Newton tool, generic USD spawning and privileged contact ledger.

No runtime pose synchronization or hand wrench is performed here. The only pose
write is explicit reset of the free joint before physical grasp settling.
"""
import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp

from .physical_tool import measured_edge_pose
from .scene_frames import matrix_pose, pose_matrix
from .pad_contact import POINT_PROFILE, PATCH_PROFILE, patch_receipt


TOOL_PATH = "/World/Tools/CuttingTool_0"
MIN_LOAD_BEARING_NORMAL_N = .05


def author_newton_tool(stage, spec, world_pose, grasp, *, contact_profile=POINT_PROFILE):
    """Use shared geometry/mass, with no PhysxSchema dependency in kit-less Lab."""
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
    if contact_profile not in (POINT_PROFILE, PATCH_PROFILE):
        raise ValueError("Unknown Lab grip contact profile")
    patch = patch_receipt(spec.handle_half_extents_m, grasp.friction) if contact_profile == PATCH_PROFILE else None
    UsdGeom.Xform.Define(stage, "/World/Tools")
    body = UsdGeom.Xform.Define(stage, TOOL_PATH)
    body.AddTranslateOp().Set(Gf.Vec3d(*world_pose[:3]))
    body.AddOrientOp().Set(Gf.Quatf(float(world_pose[6]), Gf.Vec3f(*world_pose[3:6])))
    rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    rigid.CreateRigidBodyEnabledAttr(True)
    rigid.CreateKinematicEnabledAttr(False)
    rigid.CreateVelocityAttr(Gf.Vec3f(0))
    rigid.CreateAngularVelocityAttr(Gf.Vec3f(0))
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(float(spec.mass_kg))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*spec.com_tool_m))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec.principal_moments_kg_m2))
    q = spec.principal_axes_xyzw
    mass.CreatePrincipalAxesAttr(Gf.Quatf(float(q[3]), Gf.Vec3f(*q[:3])))
    def make_material(path):
        material = UsdShade.Material.Define(stage, path)
        physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        physics.CreateStaticFrictionAttr(float(grasp.friction))
        physics.CreateDynamicFrictionAttr(float(grasp.friction))
        physics.CreateRestitutionAttr(0.)
        if patch:
            for name in ("mjc:torsionalfriction", "newton:torsionalFriction"):
                material.GetPrim().CreateAttribute(name, Sdf.ValueTypeNames.Double).Set(patch["torsional_friction_length_m"])
        return material
    # Lab imports each asset subtree into a separate builder. Physics material
    # bindings must resolve inside that subtree rather than a sibling asset.
    material = make_material(TOOL_PATH+"/GripMaterial")
    finger_material = make_material("/World/Franka_0/PhysicalGripMaterial") if stage.GetPrimAtPath("/World/Franka_0") else None
    for shape in spec.shapes:
        if shape.role == "handle":
            # The shared handle IS an axis-aligned box. Native box-box contact
            # preserves its full face manifold; mesh GJK's single-point contact
            # is an inadequate approximation for this flat-pad grasp.
            vertices = np.asarray(shape.vertices_tool_m)
            lo, hi = vertices.min(axis=0), vertices.max(axis=0)
            if len(vertices) != 8 or not np.all(np.isclose(vertices, lo) | np.isclose(vertices, hi)):
                raise ValueError("Native box handle requires the exact shared rectangular geometry")
            shape_prim = UsdGeom.Cube.Define(stage, TOOL_PATH+"/"+shape.name)
            shape_prim.CreateSizeAttr(2.)
            shape_prim.AddTranslateOp().Set(Gf.Vec3d(*((lo+hi)/2)))
            shape_prim.AddScaleOp().Set(Gf.Vec3f(*((hi-lo)/2)))
        else:
            shape_prim = UsdGeom.Mesh.Define(stage, TOOL_PATH+"/"+shape.name)
            shape_prim.CreatePointsAttr(np.asarray(shape.vertices_tool_m).tolist())
            faces = np.asarray(shape.triangles)
            shape_prim.CreateFaceVertexCountsAttr([3]*len(faces))
            shape_prim.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            shape_prim.CreateSubdivisionSchemeAttr("none")
            UsdPhysics.MeshCollisionAPI.Apply(shape_prim.GetPrim()).CreateApproximationAttr("convexHull")
        shape_prim.CreateDisplayColorAttr([Gf.Vec3f(*((.12, .13, .16) if shape.role == "handle" else (.64, .68, .73)))])
        UsdPhysics.CollisionAPI.Apply(shape_prim.GetPrim())
        shape_prim.GetPrim().AddAppliedSchema("MjcCollisionAPI")
        # Same finite normal-contact profile as the official Panda pads, rather
        # than mixing them with the generic softer imported-tool default.
        shape_prim.GetPrim().CreateAttribute("mjc:solref", Sdf.ValueTypeNames.DoubleArray).Set([.01, 1.])
        shape_prim.GetPrim().CreateAttribute("mjc:solimp", Sdf.ValueTypeNames.DoubleArray).Set([.99, .995, .01, .5, 2.])
        UsdShade.MaterialBindingAPI.Apply(shape_prim.GetPrim()).Bind(material, materialPurpose="physics")
    # MuJoCo mixes the maximum geom coefficient. Apply the declared coefficient
    # to BOTH sides so the weak-friction control is an actual pair perturbation.
    for prim in stage.Traverse():
        if "Franka_0/" in str(prim.GetPath()) and any(
                name in str(prim.GetPath()) for name in ("panda_leftfinger", "panda_rightfinger")):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(finger_material, materialPurpose="physics")
                if patch and prim.GetName() in ("left_finger_pad", "right_finger_pad"):
                    prim.AddAppliedSchema("MjcCollisionAPI")
                    prim.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(4)
    body.GetPrim().SetCustomDataByKey("research:tool_attachment_mode", "physical_grasp")
    body.GetPrim().SetCustomDataByKey("research:grip_contact_profile", contact_profile)
    body.GetPrim().SetCustomDataByKey("research:contact_owner", "MuJoCo-Warp rigid contacts; TLMPM soft contact only")
    edge = UsdGeom.Xform.Define(stage, TOOL_PATH+"/CuttingEdge")
    p = spec.tool_from_edge_xyz_xyzw
    edge.AddTranslateOp().Set(Gf.Vec3d(*p[:3]))
    edge.AddOrientOp().Set(Gf.Quatf(float(p[6]), Gf.Vec3f(*p[3:6])))
    return body.GetPrim()


@wp.kernel
def apply_tool_wrench(body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                      wrench_edge: wp.array(dtype=wp.spatial_vector), tool_body: int,
                      edge_offset: wp.vec3, dt: float,
                      body_f: wp.array(dtype=wp.spatial_vector), impulse: wp.array(dtype=float),
                      applied_com: wp.array(dtype=wp.spatial_vector), applied_pose: wp.array(dtype=wp.transform),
                      hand_body: int, pre_hand_wrench: wp.array(dtype=wp.spatial_vector)):
    pose = body_q[tool_body]
    edge = wp.transform_point(pose, edge_offset)
    center = wp.transform_point(pose, body_com[tool_body])
    w = wrench_edge[0]
    force = wp.vec3(w[0], w[1], w[2])
    torque = wp.vec3(w[3], w[4], w[5])+wp.cross(edge-center, force)
    body_f[tool_body] = body_f[tool_body]+wp.spatial_vector(force, torque)
    applied_com[0] = wp.spatial_vector(force, torque)
    applied_pose[0] = pose
    pre_hand_wrench[0] = body_f[hand_body]
    for k in range(3):
        impulse[k] = impulse[k]+dt*force[k]
        impulse[k+3] = impulse[k+3]+dt*torque[k]


@wp.kernel
def collect_tool_contacts(count: wp.array(dtype=int), shape0: wp.array(dtype=int), shape1: wp.array(dtype=int),
                          forces: wp.array(dtype=wp.spatial_vector), normal: wp.array(dtype=wp.vec3),
                          world_point: wp.array(dtype=wp.vec3), distance: wp.array(dtype=float),
                          shape_body: wp.array(dtype=int), is_handle: wp.array(dtype=int),
                          body_q: wp.array(dtype=wp.transform), body_com: wp.array(dtype=wp.vec3),
                          tool: int, left: int, right: int, sample: wp.array(dtype=float)):
    i = wp.tid()
    if i < count[0]:
        a, b = shape0[i], shape1[i]
        if a >= 0 and b >= 0:
            ba, bb = shape_body[a], shape_body[b]
            penetration = wp.max(0., -distance[i])
            wp.atomic_max(sample, 13, float(count[0]))
            if ba == tool or bb == tool:
                sign = float(1.)
                other = bb+0
                tool_shape = a+0
                if bb == tool:
                    sign = -1.
                    other = ba+0
                    tool_shape = b+0
                w = forces[i]*sign
                f = wp.vec3(w[0], w[1], w[2])
                # Pinned contact_force_fn returns a world contact-frame moment.
                # Shift it explicitly to the ACTUAL tool COM with its lever arm.
                center = wp.transform_point(body_q[tool], body_com[tool])
                torque = wp.vec3(w[3], w[4], w[5])+wp.cross(world_point[i]-center, f)
                for k in range(3):
                    wp.atomic_add(sample, k, f[k])
                    wp.atomic_add(sample, k+3, torque[k])
                fn = wp.abs(wp.dot(f, normal[i]))
                if is_handle[tool_shape] != 0 and other == left:
                    wp.atomic_add(sample, 6, fn)
                    wp.atomic_add(sample, 14, 1.)
                elif is_handle[tool_shape] != 0 and other == right:
                    wp.atomic_add(sample, 7, fn)
                    wp.atomic_add(sample, 15, 1.)
                else:
                    wp.atomic_add(sample, 11, 1.)
                wp.atomic_max(sample, 10, penetration)
            else:
                wp.atomic_max(sample, 12, penetration)


@wp.kernel
def integrate_contact_sample(sample: wp.array(dtype=float), dt: float, integral: wp.array(dtype=float)):
    for k in range(8):
        integral[k] = integral[k]+sample[k]*dt
    if sample[6] > 0.05 and sample[7] > 0.05:
        integral[8] = integral[8]+1.
    integral[9] = integral[9]+1.
    integral[10] = wp.max(integral[10], sample[10])
    integral[11] = integral[11]+sample[11]
    integral[12] = wp.max(integral[12], sample[12])
    integral[13] = wp.max(integral[13], sample[13])
    integral[14] = integral[14]+sample[14]
    integral[15] = integral[15]+sample[15]


def relative_tool_pose(hand_pose, tool_pose):
    return matrix_pose(np.linalg.inv(pose_matrix(hand_pose)) @ pose_matrix(tool_pose))


def relative_pose_error(reference, measured):
    translation = float(np.linalg.norm(np.asarray(measured)[:3]-np.asarray(reference)[:3]))
    rotation = float((Rotation.from_quat(reference[3:]).inv()*Rotation.from_quat(measured[3:])).magnitude())
    return translation, rotation


class NewtonTool:
    """Measured free-body state and actual MJWarp contact forces; debug only."""
    def __init__(self, model, solver, spec, *, contact_profile=POINT_PROFILE):
        import newton
        self.model, self.solver, self.spec = model, solver, spec
        self.contact_profile = contact_profile
        self.body = list(model.body_label).index(TOOL_PATH)
        children = model.joint_child.numpy()
        self.joint = next(i for i, child in enumerate(children) if child == self.body)
        if model.joint_type.numpy()[self.joint] != int(newton.JointType.FREE):
            raise RuntimeError("Physical tool must have an actual independent FREE joint")
        if model.joint_parent.numpy()[self.joint] != -1:
            raise RuntimeError("Physical tool cannot be parented to a robot body")
        self.q_start = int(model.joint_q_start.numpy()[self.joint])
        self.qd_start = int(model.joint_qd_start.numpy()[self.joint])
        if not 0. < model.body_mass.numpy()[self.body] < 1.:
            raise RuntimeError("Physical tool mass is absent or outside declared compact-tool range")
        self.left = next(i for i, label in enumerate(model.body_label) if "Franka_0/" in label and label.endswith("/panda_leftfinger"))
        self.right = next(i for i, label in enumerate(model.body_label) if "Franka_0/" in label and label.endswith("/panda_rightfinger"))
        self.shapes = np.flatnonzero(model.shape_body.numpy() == self.body)
        self.handles = [i for i in self.shapes if any(s.role == "handle" and model.shape_label[i].endswith("/"+s.name) for s in spec.shapes)]
        if not self.handles:
            raise RuntimeError("Independent tool has no imported handle collider")
        mask = np.zeros(model.shape_count, np.int32);mask[self.handles] = 1
        self.handle_mask = wp.array(mask, dtype=int, device=model.device)
        self.contacts = newton.Contacts(rigid_contact_max=solver.mjw_data.naconmax, soft_contact_max=0,
            requested_attributes={"force"}, device=model.device)
        self.sample = wp.zeros(16, dtype=float, device=model.device)
        self.integral = wp.zeros(16, dtype=float, device=model.device)
        self.applied_impulse = wp.zeros(6, dtype=float, device=model.device)
        self.current_applied_com = wp.zeros(1, dtype=wp.spatial_vector, device=model.device)
        self.current_applied_pose = wp.zeros(1, dtype=wp.transform, device=model.device)
        self.current_pre_hand_wrench = wp.zeros(1, dtype=wp.spatial_vector, device=model.device)
        self.hand = next(i for i, label in enumerate(model.body_label) if "Franka_0/" in label and label.endswith("/panda_hand"))
        finger_joints = [next(i for i, label in enumerate(model.joint_label) if "Franka_0/" in label and label.endswith("/panda_finger_joint"+str(j))) for j in (1, 2)]
        self.finger_coords = model.joint_q_start.numpy()[finger_joints]
        self.finger_dofs = model.joint_qd_start.numpy()[finger_joints]
        self.raw_trace = []
        self.phase = "settling"
        self.rigid_time_s = 0.
        if contact_profile == PATCH_PROFILE:
            pads = [i for i, label in enumerate(model.shape_label) if label.endswith(("/left_finger_pad", "/right_finger_pad")) and "Franka_0/" in label]
            if len(pads) != 2 or not np.all(model.mujoco.condim.numpy()[pads] == 4):
                raise RuntimeError("Finite pad profile was not imported as actual condim4 pad shapes")

    def reset_pose(self, state, pose):
        q = state.joint_q.numpy();qd = state.joint_qd.numpy()
        q[self.q_start:self.q_start+7] = pose
        qd[self.qd_start:self.qd_start+6] = 0.
        state.joint_q.assign(q);state.joint_qd.assign(qd)
        self.raw_trace = []
        self.rigid_time_s = 0.
        self.phase = "settling"

    def begin_interval(self):
        self.integral.zero_();self.applied_impulse.zero_()

    def post_substep(self, state, dt):
        self.solver.update_contacts(self.contacts, state)
        self.sample.zero_()
        data = self.solver.mjw_data
        wp.launch(collect_tool_contacts, dim=data.naconmax,
            inputs=[self.contacts.rigid_contact_count, self.contacts.rigid_contact_shape0,
                self.contacts.rigid_contact_shape1, self.contacts.force, self.contacts.rigid_contact_normal,
                data.contact.pos, data.contact.dist, self.model.shape_body, self.handle_mask,
                state.body_q, self.model.body_com, self.body, self.left, self.right, self.sample], device=self.model.device)
        wp.launch(integrate_contact_sample, dim=1, inputs=[self.sample, dt, self.integral], device=self.model.device)
        # Privileged raw2ms evidence for independent contact/impulse checks.
        # Small host copies intentionally trade probe throughput for auditable
        # data; no contact force or free-tool truth enters formal observations.
        count = min(int(self.contacts.rigid_contact_count.numpy()[0]), data.naconmax)
        body = state.body_q.numpy()
        self.rigid_time_s += dt
        self.raw_trace.append({"rigid_time_s": self.rigid_time_s, "phase": self.phase,
            "tool_body_pose_xyzw": body[self.body].copy(), "hand_pose_xyzw": body[self.hand].copy(),
            "tool_velocity_com_world": state.body_qd.numpy()[self.body].copy(),
            "finger_q": state.joint_q.numpy()[self.finger_coords].copy(),
            "finger_qdot": state.joint_qd.numpy()[self.finger_dofs].copy(),
            "applied_wrench_com_world": self.current_applied_com.numpy()[0].copy(),
            "applied_at_tool_pose_xyzw": self.current_applied_pose.numpy()[0].copy(),
            "pre_step_hand_external_wrench_world": self.current_pre_hand_wrench.numpy()[0].copy(),
            "shape0": self.contacts.rigid_contact_shape0.numpy()[:count].copy(),
            "shape1": self.contacts.rigid_contact_shape1.numpy()[:count].copy(),
            "contact_wrench_on_shape0_world": self.contacts.force.numpy()[:count].copy(),
            "point_world": data.contact.pos.numpy()[:count].copy(),
            "normal_world": self.contacts.rigid_contact_normal.numpy()[:count].copy(),
            "friction": data.contact.friction.numpy()[:count].copy(),
            "contact_dimension": data.contact.dim.numpy()[:count].copy(),
            "hand_external_wrench_world": state.body_f.numpy()[self.hand].copy(),
            "distance_m": data.contact.dist.numpy()[:count].copy()})

    def save_raw_trace(self, path):
        from pathlib import Path
        from .recording import write_json
        path = Path(path)
        rows = self.raw_trace
        if not rows:raise RuntimeError("No actual tool substeps to save")
        fixed = ["rigid_time_s", "tool_body_pose_xyzw", "hand_pose_xyzw", "tool_velocity_com_world",
                 "finger_q", "finger_qdot", "applied_wrench_com_world", "applied_at_tool_pose_xyzw", "hand_external_wrench_world"]
        fixed += ["pre_step_hand_external_wrench_world"]
        variable = ["shape0", "shape1", "contact_wrench_on_shape0_world", "point_world", "normal_world", "friction", "distance_m", "contact_dimension"]
        arrays = {key: np.asarray([r[key] for r in rows]) for key in fixed}
        arrays.update({key: np.concatenate([r[key] for r in rows], axis=0) for key in variable})
        arrays["contact_offsets"] = np.r_[0, np.cumsum([len(r["shape0"]) for r in rows])]
        arrays["phase"] = np.array([r["phase"] for r in rows])
        np.savez_compressed(path, **arrays)
        write_json(path.with_suffix(".json"), {"scope": "Privileged actual2ms MJWarp tool/contact/actuation samples",
            "tool_body_index": self.body, "left_finger_body_index": self.left, "right_finger_body_index": self.right,
            "shape_labels": list(self.model.shape_label), "shape_body": self.model.shape_body.numpy(),
            "handle_shape_indices": self.handles, "tool_mass_kg": self.model.body_mass.numpy()[self.body],
            "tool_com_local_m": self.model.body_com.numpy()[self.body],
            "tool_inertia_com_local_kg_m2": self.model.body_inertia.numpy()[self.body],
            "rigid_model_gravity_world_m_s2": self.model.gravity.numpy(),
            "tool_body_world_index": int(self.model.body_world.numpy()[self.body]),
            "native_mujoco_gravity_world_m_s2": self.solver.mjw_model.opt.gravity.numpy(),
            "shape_contact_dimension": self.model.mujoco.condim.numpy(),
            "grip_contact_profile": self.contact_profile,
            "dt_s": .002, "samples": len(rows), "runtime_tool_pose_writes": 0,
            "contact_wrench_reference": "world axes at contact point; force on shape0. Shift torque to COM with recorded lever arm.",
            "applied_wrench_reference": "actual tool COM, world axes, computed at recorded preintegration tool pose",
            "normal_contact_profile": {"mjc_solref": [.01, 1.], "mjc_solimp": [.99, .995, .01, .5, 2.],
                "source": "same finite profile as official Panda pads"},
            "handle_collision": "exact native box; shaft/blade exact convex hull meshes",
            "finite_pad_patch": patch_receipt(self.spec.handle_half_extents_m,
                float(self.model.shape_material_mu.numpy()[self.handles[0]])) if self.contact_profile == PATCH_PROFILE else None})

    def measured(self, state):
        pose = state.body_q.numpy()[self.body].astype(float)
        velocity = state.body_qd.numpy()[self.body].astype(float)
        center = pose_matrix(pose)[:3, :3] @ self.model.body_com.numpy()[self.body]+pose[:3]
        return {"tool_body_pose_xyzw": pose, "tool_pose": measured_edge_pose(pose, self.spec),
            "tool_linear_velocity_world_m_s": velocity[:3], "tool_angular_velocity_world_rad_s": velocity[3:],
            "tool_com_world_m": center}

    def diagnostics(self, state, hand_pose, initial_relative):
        measured = self.measured(state)
        relative = relative_tool_pose(hand_pose, measured["tool_body_pose_xyzw"])
        translation, rotation = relative_pose_error(initial_relative, relative)
        integral, sample = self.integral.numpy(), self.sample.numpy()
        count = min(int(self.contacts.rigid_contact_count.numpy()[0]), self.solver.mjw_data.naconmax)
        a, b = self.contacts.rigid_contact_shape0.numpy(), self.contacts.rigid_contact_shape1.numpy()
        f = self.contacts.force.numpy();distance = self.solver.mjw_data.contact.dist.numpy()
        shape_body = self.model.shape_body.numpy()
        pairs = []
        for i in range(count):
            if a[i] < 0 or b[i] < 0 or self.body not in (shape_body[a[i]], shape_body[b[i]]):continue
            sign = 1. if shape_body[a[i]] == self.body else -1.
            pairs.append({"shape0": str(self.model.shape_label[a[i]]), "shape1": str(self.model.shape_label[b[i]]),
                "force_on_tool_world_n": (sign*f[i, :3]).tolist(), "distance_m": float(distance[i])})
        return {**measured, "relative_tool_in_hand_xyzw": relative,
            "relative_translation_drift_m": translation, "relative_rotation_drift_rad": rotation,
            "left_handle_normal_n": float(sample[6]), "right_handle_normal_n": float(sample[7]),
            "bilateral_substep_fraction": float(integral[8]/integral[9]) if integral[9] else 0.,
            "observed_substeps": int(integral[9]), "max_tool_rigid_penetration_m": float(integral[10]),
            "unintended_tool_contact_slots": int(integral[11]), "max_non_tool_penetration_m": float(integral[12]),
            "contact_impulse_on_tool_com_world": integral[:6].astype(float),
            "applied_impulse_to_tool_com_world": self.applied_impulse.numpy().astype(float),
            "contact_pairs": pairs,
            "contact_source": "actual MJWarp solver.update_contacts/contact_force_fn after every2ms substep; force on tool, world axes",
            "minimum_load_bearing_normal_n": MIN_LOAD_BEARING_NORMAL_N,
            "torque_reference": "contact moment explicitly shifted to measured tool COM",
            "hand_wrench_injection": False, "runtime_tool_pose_writes": 0}
