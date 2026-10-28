"""Native PhysX contact evidence for an independent tool, outside observations.

Isaac modules are imported only when the monitor is constructed after Kit boots.
Raw sensor impulses retain their native body ordering; this module does not
invent an action/reaction sign or treat a bounding box as evidence of contact.
"""
from collections.abc import Mapping
import math

import numpy as np


def _under(path, root):
    return path == root or path.startswith(root.rstrip("/")+"/")


def _vector(value):
    result = np.asarray([value[k] for k in ("x", "y", "z")] if isinstance(value, Mapping) else value, float)
    if result.shape != (3,) or not np.isfinite(result).all():
        raise ValueError("Contact vectors must contain three finite values")
    return result


def _time_tolerance(now):
    # Raw sensor times use float32; permit representational error, not an old
    # rigid step. At an eight-second episode this is below four microseconds.
    return max(1e-7, 4*np.finfo(np.float32).eps*max(1., abs(now)))


def contact_records(raw, *, tool_path, role_paths, physics_time_s, dt,
                    decode_body=str, handle_bounds_tool_m=None, tool_pose_xyzw=None):
    """Copy actual contact reports, classify either body ordering and retain age.

    ``force_magnitude_n`` is |raw impulse| / the report's own positive dt.
    Its vector stays ``raw_force_world_n`` until the native impulse convention
    is independently measured. Handle attribution tests the *reported contact
    point*, never infers contact from nearby geometry.
    """
    if not math.isfinite(physics_time_s) or not math.isfinite(dt) or dt <= 0:
        raise ValueError("A finite native physics time and positive dt are required")
    world_from_tool = None
    if tool_pose_xyzw is not None:
        from scipy.spatial.transform import Rotation
        pose = np.asarray(tool_pose_xyzw, float)
        if pose.shape != (7,) or not np.isfinite(pose).all() or abs(np.linalg.norm(pose[3:])-1) > 1e-4:
            raise ValueError("Expected measured tool pose [xyz, quaternion XYZW]")
        world_from_tool = Rotation.from_quat(pose[3:]).as_matrix(), pose[:3]
    bounds = None if handle_bounds_tool_m is None else np.asarray(handle_bounds_tool_m, float)
    if bounds is not None and (bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0])):
        raise ValueError("Handle bounds must be finite lower/upper tool-frame points")
    records = []
    for item in raw:
        body0, body1 = (str(decode_body(int(item[k]))) for k in ("body0", "body1"))
        on0, on1 = _under(body0, tool_path), _under(body1, tool_path)
        other = body1 if on0 and not on1 else body0 if on1 and not on0 else None
        role = next((name for name, roots in role_paths.items() if other and any(_under(other, p) for p in roots)), "other")
        position, normal, impulse = (_vector(item[k]) for k in ("position", "normal", "impulse"))
        raw_dt, raw_time = float(item["dt"]), float(item["time"])
        if not math.isfinite(raw_dt) or raw_dt <= 0 or not math.isfinite(raw_time):
            raise ValueError("Actual raw contact time/dt is missing, nonfinite or nonpositive")
        age = physics_time_s-raw_time
        fresh = abs(age) <= _time_tolerance(physics_time_s) and abs(raw_dt-dt) <= max(1e-7, dt*1e-4)
        point_tool, in_handle = None, None
        if world_from_tool is not None:
            rotation, translation = world_from_tool
            point_tool = rotation.T@(position-translation)
            if bounds is not None:
                in_handle = bool(np.all(point_tool >= bounds[0]-1e-6) and np.all(point_tool <= bounds[1]+1e-6))
        force = impulse/raw_dt
        records.append({"body0": body0, "body1": body1, "other_body": other, "tool_is_body0": bool(on0),
            "tool_in_pair": other is not None, "role": role, "time_s": raw_time, "dt_s": raw_dt,
            "age_s": age, "fresh": bool(fresh), "position_world_m": position.tolist(),
            "position_tool_m": None if point_tool is None else point_tool.tolist(), "in_handle_region": in_handle,
            "normal_world": normal.tolist(), "raw_impulse_world_ns": impulse.tolist(),
            "raw_force_world_n": force.tolist(), "force_magnitude_n": float(np.linalg.norm(force))})
    return records


def summarize_contacts(records, *, valid, fresh, role_names=("left_finger", "right_finger", "palm", "table", "fixture", "other")):
    """Summaries use fresh, explicitly matched tool pairs only."""
    roles = {}
    for name in role_names:
        selected = [r for r in records if valid and fresh and r["fresh"] and r["tool_in_pair"] and r["role"] == name]
        roles[name] = {"contact_count": len(selected), "handle_contact_count": sum(r["in_handle_region"] is True for r in selected),
            "loaded_contact_count": sum(r["force_magnitude_n"] > 1e-6 for r in selected),
            "loaded_handle_contact_count": sum(r["in_handle_region"] is True and r["force_magnitude_n"] > 1e-6 for r in selected),
            "force_magnitude_sum_n": sum(r["force_magnitude_n"] for r in selected),
            "raw_force_vector_sum_n": np.sum([r["raw_force_world_n"] for r in selected], axis=0).tolist() if selected else [0., 0., 0.]}
    left, right = roles["left_finger"], roles["right_finger"]
    return {"roles": roles, "bilateral_tool_contact": bool(left["contact_count"] and right["contact_count"]),
        "bilateral_handle_contact": bool(left["handle_contact_count"] and right["handle_contact_count"]),
        "bilateral_loaded_tool_contact": bool(left["loaded_contact_count"] and right["loaded_contact_count"]),
        "bilateral_loaded_handle_contact": bool(left["loaded_handle_contact_count"] and right["loaded_handle_contact_count"]),
        "loaded_contact_minimum_n": 1e-6,
        "unintended_contact_count": sum(v["contact_count"] for k, v in roles.items() if k not in ("left_finger", "right_finger")),
        "unmatched_or_stale_reports": sum(not r["tool_in_pair"] or not r["fresh"] for r in records)}


class NativeToolContacts:
    """One native sensor on the independent rigid tool, sampled each rigid step.

    Construct before physics initialization. ``finger_paths`` is left/right
    paths or a mapping with keys ``left`` and ``right``. The optional handle
    bounds use the rigid tool frame; pass its measured pose to ``sample``.
    """
    def __init__(self, stage, tool_path, finger_paths, hand_path, dt, *,
                 handle_bounds_tool_m=None, role_paths=None, tensor_contact_forces=True):
        import omni.kit.app
        from pxr import PhysicsSchemaTools, UsdGeom, UsdPhysics
        manager = omni.kit.app.get_app().get_extension_manager()
        manager.set_extension_enabled_immediate("isaacsim.sensors.experimental.physics", True)
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.sensors.experimental.physics import Contact, ContactSensor
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be the positive native rigid substep")
        if abs(UsdGeom.GetStageMetersPerUnit(stage)-1.) > 1e-12:
            raise ValueError("Tool contact recording currently requires a metre-scale stage")
        prim = stage.GetPrimAtPath(tool_path)
        if not prim.IsValid() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError("Contact monitor requires an existing independent rigid tool body")
        if isinstance(finger_paths, Mapping):
            left, right = finger_paths["left"], finger_paths["right"]
        else:
            left, right = finger_paths
        self.tool_path, self.dt = str(tool_path), float(dt)
        self.handle_bounds_tool_m = handle_bounds_tool_m
        self.role_paths = {"left_finger": [str(left)], "right_finger": [str(right)], "palm": [str(hand_path)],
            "table": ["/World/Table"], "fixture": ["/World/TaskFixture", "/World/TrayFloor", "/World/TrayLeft",
                "/World/TrayRight", "/World/TrayFront", "/World/TrayBack"], **(role_paths or {})}
        self.role_paths = {name: [roots] if isinstance(roots, str) else list(roots) for name, roots in self.role_paths.items()}
        self.manager, self.decode_body = SimulationManager, PhysicsSchemaTools.intToSdfPath
        self.sensor_path = self.tool_path+"/NativeContactSensor"
        self.sensor = ContactSensor(Contact.create(self.sensor_path, min_threshold=1e-6, max_threshold=1e6, radius=-1))
        self.tensor = None
        self.tensor_filters = []
        if tensor_contact_forces:
            from isaacsim.core.experimental.prims import RigidPrim
            # Resolve actual body/static-collider paths. A fixture parent Xform
            # itself is not a contacting body. Finger/hand body filters come
            # first and must not also select descendant colliders twice.
            for candidate in stage.Traverse():
                path = str(candidate.GetPath())
                if not candidate.HasAPI(UsdPhysics.CollisionAPI) and not candidate.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                role = next((name for name, roots in self.role_paths.items() if any(_under(path, root) for root in roots)), None)
                if role is None or any(role == row["role"] and _under(path, row["path"]) for row in self.tensor_filters):
                    continue
                self.tensor_filters.append({"path": path, "role": role})
            if self.tensor_filters:
                self.tensor = RigidPrim(self.tool_path, contact_filter_paths=[row["path"] for row in self.tensor_filters], max_contact_count=2048)
        self._last_step = None

    def _tensor_sample(self, tool_pose_xyzw):
        if self.tensor is None:
            return {"status": "NOT_RECORDED", "filters": self.tensor_filters}
        normal, points, normals, distances, counts, starts = [a.numpy().copy() for a in self.tensor.get_contact_force_data(dt=self.dt)]
        friction, friction_points, friction_counts, friction_starts = [a.numpy().copy() for a in self.tensor.get_friction_data(dt=self.dt)]
        records, role_forces = [], {name: np.zeros(3) for name in self.role_paths}
        origin = None if tool_pose_xyzw is None else np.asarray(tool_pose_xyzw, float)[:3]
        for i, entry in enumerate(self.tensor_filters):
            for kind, number, first in (("normal", counts[0, i], starts[0, i]),
                                        ("friction", friction_counts[0, i], friction_starts[0, i])):
                for j in range(int(first), int(first)+int(number)):
                    force = float(normal[j, 0])*normals[j] if kind == "normal" else friction[j]
                    point = points[j] if kind == "normal" else friction_points[j]
                    role_forces[entry["role"]] += force
                    records.append({"kind": kind, **entry, "force_world_on_tool_n": force.tolist(),
                        "position_world_m": point.tolist(), "normal_world": normals[j].tolist() if kind == "normal" else None,
                        "separation_m": float(distances[j, 0]) if kind == "normal" else None,
                        "moment_about_tool_origin_world_nm": np.cross(point-origin, force).tolist() if origin is not None else None})
        finite = all(np.isfinite(row["force_world_on_tool_n"]).all() for row in records)
        return {"status": "VALID_NATIVE_NORMAL_AND_FRICTION_DATA" if finite else "FAIL_NONFINITE_TENSOR_FORCES",
            "filters": self.tensor_filters, "records": records, "dt_s": self.dt,
            "normal_contact_count": int(counts.sum()), "friction_contact_count": int(friction_counts.sum()),
            "capacity_may_be_reached": bool(counts.sum() >= 2048 or friction_counts.sum() >= 2048),
            "force_by_role_world_on_tool_n": {k: v.tolist() for k, v in role_forces.items()},
            "total_filtered_force_world_on_tool_n": sum(role_forces.values(), np.zeros(3)).tolist(),
            "scope": "Native tool contact-view forces on the declared filters; unlisted bodies are not included"}

    def sample(self, *, tool_pose_xyzw=None):
        """Read actual post-step data. Invalid/stale readings cannot certify grip."""
        now, step = self.manager.get_simulation_time(), self.manager.get_num_physics_steps()
        reading = self.sensor.get_sensor_reading()
        valid = bool(reading.is_valid)
        age = float(now)-float(reading.time)
        fresh = valid and abs(age) <= _time_tolerance(float(now))
        raw = self.sensor.get_raw_data() if valid else []
        records = contact_records(raw, tool_path=self.tool_path, role_paths=self.role_paths,
            physics_time_s=float(now), dt=self.dt, decode_body=self.decode_body,
            handle_bounds_tool_m=self.handle_bounds_tool_m, tool_pose_xyzw=tool_pose_xyzw)
        summary = summarize_contacts(records, valid=valid, fresh=fresh, role_names=tuple(self.role_paths)+("other",))
        result = {"source": "isaacsim.sensors.experimental.physics.ContactSensor", "sensor_path": self.sensor_path,
            "sensor_valid": valid, "sensor_fresh": bool(fresh), "native_time_s": float(now), "native_step": int(step),
            "new_physics_step": step != self._last_step, "sensor_time_s": float(reading.time), "sensor_age_s": age,
            "sensor_in_contact": bool(reading.in_contact), "sensor_force_magnitude_n": float(reading.value),
            "raw_contact_count": len(records), "contacts": records, **summary,
            "handle_region_classification": "MEASURED_CONTACT_POINT_IN_TOOL_FRAME" if tool_pose_xyzw is not None and self.handle_bounds_tool_m is not None else "NOT_CLASSIFIED",
            "impulse_sign_scope": "Raw native body ordering retained; no assumed signed tool wrench",
            "raw_impulse_scope": "Normal contact reports; friction is recorded separately by the native tensor contact view",
            "tensor_forces": self._tensor_sample(tool_pose_xyzw) if valid and fresh else {"status": "NOT_RUN_INVALID_OR_STALE_SENSOR"},
            "privileged_debug_only": True}
        self._last_step = step
        return result

    def close(self):
        self.sensor.reset()
        self.tensor = None
