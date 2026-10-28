#!/usr/bin/env python3
"""Read-only rigid-tool impulse balance with separately prefrozen budgets."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_physical_tool import digest, read_json, runtime_log_audit, source_audit, under

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/tool_momentum_audit_v1.json"


def angular_momentum(pose_xyzw, omega_world, inertia_body):
    rotation = Rotation.from_quat(np.asarray(pose_xyzw, float)[3:]).as_matrix()
    return rotation @ inertia_body @ rotation.T @ omega_world


def balance_step(*, mass, inertia, pose_before, pose_after, velocity_before,
                 velocity_after, gravity, applied_wrench, contact_wrenches,
                 contact_points, com_before, dt, limits):
    """All velocities/wrenches use world axes; contact torques are AT points."""
    wrenches = np.asarray(contact_wrenches, float).reshape(-1, 6)
    points = np.asarray(contact_points, float).reshape(-1, 3)
    if len(wrenches) != len(points) or dt <= 0 or mass <= 0:
        raise ValueError("Contact rows must align; mass and dt must be positive")
    force_impulses = np.vstack((wrenches[:, :3], applied_wrench[:3], mass * gravity)) * dt
    torques = wrenches[:, 3:] + np.cross(points - com_before, wrenches[:, :3])
    angular_impulses = np.vstack((torques, applied_wrench[3:])) * dt
    delta_p = mass * (velocity_after[:3] - velocity_before[:3])
    delta_l = (angular_momentum(pose_after, velocity_after[3:], inertia)
               - angular_momentum(pose_before, velocity_before[3:], inertia))
    output = {}
    for name, measured, impulses, absolute in (
            ("linear", delta_p, force_impulses, limits["linear_absolute_budget_n_s"]),
            ("angular", delta_l, angular_impulses, limits["angular_absolute_budget_n_m_s"])):
        predicted = impulses.sum(axis=0)
        residual = measured - predicted
        scale = float(np.linalg.norm(impulses, axis=1).sum())
        budget = absolute + limits["relative_budget"] * scale
        output[name] = {"momentum_change": measured.tolist(), "integrated_wrench": predicted.tolist(),
                        "residual": residual.tolist(), "residual_norm": float(np.linalg.norm(residual)),
                        "constituent_impulse_norm_sum": scale, "budget": float(budget),
                        "budget_usage": float(np.linalg.norm(residual) / budget),
                        "pass": bool(np.linalg.norm(residual) <= budget)}
    if not all(np.isfinite(row["residual"]).all() for row in output.values()):
        raise ValueError("Nonfinite impulse balance")
    return output


def summary(rows, key):
    if not rows:
        return {"status": "NOT_SCORED_NO_ROWS"}
    values = np.array([r[key]["residual_norm"] for r in rows])
    usage = np.array([r[key]["budget_usage"] for r in rows])
    failures = [i for i, r in enumerate(rows) if not r[key]["pass"]]
    return {"status": "FAIL_IMPULSE_BALANCE" if failures else "PASS_ENGINEERING_IMPULSE_BALANCE",
            "steps": len(rows), "failing_steps": len(failures), "first_failing_row_indices": failures[:10],
            "residual_norm_quantiles": dict(zip(("p50", "p95", "p99", "max"), np.quantile(values, [.5, .95, .99, 1]).tolist())),
            "maximum_budget_usage": float(usage.max()),
            "accumulated_vector_residual": np.sum([r[key]["residual"] for r in rows], axis=0).tolist(),
            "sum_residual_norms": float(values.sum()), "all_tails_retained": True}


def check_properties(mass, inertia):
    inertia = np.asarray(inertia, float).reshape(3, 3)
    if mass <= 0 or not np.isfinite(mass) or not np.isfinite(inertia).all():
        raise ValueError("Actual mass/inertia must be finite and positive")
    if not np.allclose(inertia, inertia.T, atol=1e-12, rtol=1e-6) or np.linalg.eigvalsh(inertia).min() <= 0:
        raise ValueError("Actual COM inertia must be symmetric positive definite")
    return inertia


def native_rows(episode, setup, limits):
    trace = episode / "physical_grasp_trace_00.jsonl"
    properties = setup.get("dynamic_tool_properties", {})
    if not properties or not trace.is_file():
        return [], ["Missing actual rigid mass/inertia or raw trace"], {}
    mass = float(np.asarray(properties["mass_kg"]).reshape(-1)[0])
    inertia = check_properties(mass, properties["inertia_native_flat_kg_m2"])
    gravity = np.asarray(properties["gravity_world_m_s2"], float)
    com = np.asarray(properties["com_pose_native_position"], float).reshape(-1, 3)[0]
    body = properties["path"]
    rows, errors = [], []
    with trace.open() as stream:
        raw = [json.loads(line) for line in stream]
    required = ("tool_body_pose_before_step_xyzw", "tool_linear_velocity_before_step_world_m_s",
                "tool_angular_velocity_before_step_world_rad_s", "applied_total_wrench_about_com_world")
    for index, row in enumerate(raw):
        if row.get("phase") != "coupled_episode":
            continue
        problem = []
        if any(key not in row for key in required):
            problem.append("Missing measured PRE state")
        contact = row.get("contacts", {})
        tensor = contact.get("tensor_forces", {})
        if not all(contact.get(k) is True for k in ("sensor_valid", "sensor_fresh", "new_physics_step")):
            problem.append("Stale/invalid current contact sensor")
        if tensor.get("status") != "VALID_NATIVE_NORMAL_AND_FRICTION_DATA" or tensor.get("capacity_may_be_reached", False):
            problem.append("Incomplete current normal/friction tensor view")
        if row.get("direct_hand_wrench_injection") is not False:
            problem.append("Direct hand wrench state unverified")
        dt = float(row.get("wrench_interval_dt_s", 0.))
        if abs(float(tensor.get("dt_s", 0.)) - dt) > 1e-12 or dt <= 0:
            problem.append("Force-to-impulse interval mismatch")
        if index == 0 or abs(row["native_time_s"] - raw[index - 1]["native_time_s"] - dt) > 1e-9:
            problem.append("PRE/POST times are not consecutive actual rigid steps")
        elif all(key in row for key in required):
            previous = raw[index - 1]
            for before, after in (("tool_body_pose_before_step_xyzw", "tool_body_pose_xyzw"),
                                  ("tool_linear_velocity_before_step_world_m_s", "tool_linear_velocity_world_m_s"),
                                  ("tool_angular_velocity_before_step_world_rad_s", "tool_angular_velocity_world_rad_s")):
                if not np.array_equal(row[before], previous[after]):
                    problem.append("Recorded PRE state does not equal preceding measured state: " + before)
        if (contact.get("contact_geometry_frame") != "PRE_STEP_MEASURED_TOOL_POSE" or
                not np.array_equal(contact.get("contact_geometry_tool_pose_xyzw"), row.get("tool_body_pose_before_step_xyzw")) or
                abs(float(contact.get("contact_geometry_time_s", -1)) - row["native_time_s"] + dt) > 1e-9):
            problem.append("Contact point event geometry is not the explicit measured PRE state")
        filters = [x["path"] for x in tensor.get("filters", [])]
        for pair in contact.get("contacts", []):
            if float(pair.get("force_magnitude_n", 0.)) <= 1e-6:
                continue
            other = pair.get("other_body", "")
            if not any(under(other, path) or under(path, other) for path in filters):
                problem.append("Loaded raw contact outside tensor force filters: " + other)
        if problem:
            errors.append({"trace_row": index, "native_time_s": row["native_time_s"], "reasons": problem})
            continue
        pose0, pose1 = np.asarray(row["tool_body_pose_before_step_xyzw"]), np.asarray(row["tool_body_pose_xyzw"])
        velocity0 = np.r_[row["tool_linear_velocity_before_step_world_m_s"], row["tool_angular_velocity_before_step_world_rad_s"]]
        velocity1 = np.r_[row["tool_linear_velocity_world_m_s"], row["tool_angular_velocity_world_rad_s"]]
        rotation = Rotation.from_quat(pose0[3:]).as_matrix()
        center = pose0[:3] + rotation @ com
        if not np.allclose(center, row["tool_com_before_step_world_m"], atol=1e-8, rtol=0):
            errors.append({"trace_row": index, "reasons": ["Actual PRE COM mismatch"]})
            continue
        forces, points = [], []
        for item in tensor.get("records", []):
            if item["kind"] not in ("normal", "friction"):
                raise ValueError("Unexpected tensor contact force kind")
            forces.append(np.r_[item["force_world_on_tool_n"], np.zeros(3)])
            points.append(item["position_world_m"])
        balanced = balance_step(mass=mass, inertia=inertia, pose_before=pose0, pose_after=pose1,
            velocity_before=velocity0, velocity_after=velocity1, gravity=gravity,
            applied_wrench=np.asarray(row["applied_total_wrench_about_com_world"]), contact_wrenches=forces,
            contact_points=points, com_before=center, dt=dt, limits=limits)
        rows.append({"trace_row": index, "time_s": row["native_time_s"], "dt_s": dt,
                     "normal_force_records": sum(x["kind"] == "normal" for x in tensor.get("records", [])),
                     "friction_force_records": sum(x["kind"] == "friction" for x in tensor.get("records", [])), **balanced})
    return rows, errors, {"mass_kg": mass, "inertia_body_com_kg_m2": inertia.tolist(), "gravity_world_m_s2": gravity.tolist(),
                          "property_source": properties.get("readback_source"), "body": body, "raw_sha256": digest(trace),
                          "velocity_convention": "Actual PhysX COM linear velocity and world angular velocity",
                          "inertia_convention": "Actual physics tensor COM inertia expressed in rigid-body prim frame"}


def lab_rows(episode, setup, limits):
    path = episode / "tool_substeps.npz"
    metadata = read_json(path.with_suffix(".json"), {})
    if not path.is_file() or not metadata:
        return [], ["Missing actual MJWarp substeps"], {}
    with np.load(path, allow_pickle=False) as archive:
        raw = {k: archive[k] for k in archive.files}
    mass = float(metadata["tool_mass_kg"])
    inertia = check_properties(mass, metadata["tool_inertia_com_local_kg_m2"])
    com = np.asarray(metadata["tool_com_local_m"])
    gravity = setup.get("actual_rigid_gravity_world_m_s2")
    # The legacy recorder retains TLMPM gravity, not a rigid-model readback.
    # Keep that distinction explicit; angular balance has no gravity torque.
    assumed_gravity = np.array([0., 0., -9.81]) if gravity is None else np.asarray(gravity)
    shape_body = np.asarray(metadata["shape_body"], int)
    offsets, body = raw["contact_offsets"], int(metadata["tool_body_index"])
    rows, errors = [], []
    for index in range(1, len(raw["rigid_time_s"])):
        if raw["phase"][index] != "formal_episode":
            continue
        dt = float(raw["rigid_time_s"][index] - raw["rigid_time_s"][index - 1])
        before = raw["tool_body_pose_xyzw"][index - 1]
        if (abs(dt - metadata["dt_s"]) > 1e-9 or
                not np.allclose(raw["applied_at_tool_pose_xyzw"][index], before, atol=1e-6, rtol=0) or
                np.max(np.abs(raw["hand_external_wrench_world"][index])) > 1e-8):
            errors.append({"raw_row": index, "reasons": ["Inconsistent PRE state, interval or direct hand force"]})
            continue
        center = before[:3] + Rotation.from_quat(before[3:]).as_matrix() @ com
        wrenches, points = [], []
        for slot in range(offsets[index], offsets[index + 1]):
            first, second = int(raw["shape0"][slot]), int(raw["shape1"][slot])
            a, b = shape_body[first] == body, shape_body[second] == body
            if a ^ b:
                wrenches.append(raw["contact_wrench_on_shape0_world"][slot] * (1 if a else -1))
                points.append(raw["point_world"][slot])
        balanced = balance_step(mass=mass, inertia=inertia, pose_before=before, pose_after=raw["tool_body_pose_xyzw"][index],
            velocity_before=raw["tool_velocity_com_world"][index - 1], velocity_after=raw["tool_velocity_com_world"][index],
            gravity=assumed_gravity, applied_wrench=raw["applied_wrench_com_world"][index], contact_wrenches=wrenches,
            contact_points=points, com_before=center, dt=dt, limits=limits)
        rows.append({"raw_row": index, "time_s": float(raw["rigid_time_s"][index]), "dt_s": dt,
                     "tool_contact_rows": len(wrenches), **balanced})
    return rows, errors, {"mass_kg": mass, "inertia_body_com_kg_m2": inertia.tolist(),
                          "gravity_world_m_s2": None if gravity is None else assumed_gravity.tolist(),
                          "diagnostic_assumed_gravity_world_m_s2": assumed_gravity.tolist(),
                          "linear_scoring_available": gravity is not None,
                          "angular_scoring_available": True, "body_index": body,
                          "raw_sha256": digest(path), "raw_metadata_sha256": digest(path.with_suffix(".json")),
                          "velocity_convention": "Recorded actual Newton body COM velocity, linear then angular",
                          "contact_convention": metadata["contact_wrench_reference"]}


def audit(episode, protocol):
    episode = Path(episode).resolve()
    setup, manifest = read_json(episode / "setup.json", {}), read_json(episode / "manifest.json", {})
    limits = protocol["per_step_acceptance"]
    is_lab = (episode / "tool_substeps.npz").is_file()
    rows, errors, properties = (lab_rows if is_lab else native_rows)(episode, setup, limits)
    source = source_audit(episode, manifest)
    logs = runtime_log_audit(episode)
    linear, angular = summary(rows, "linear"), summary(rows, "angular")
    if is_lab and not properties.get("linear_scoring_available", False):
        linear = {"status": "NOT_SCORED_RIGID_GRAVITY_READBACK_NOT_RECORDED", "assumed_gravity_diagnostic": linear}
    invalid = bool(errors) or not source["pass"] or logs["status"].startswith("FAIL")
    result = {"episode": str(episode), "runtime": "isaac_lab" if is_lab else "isaac_sim",
              "scope": protocol["scope"], "protocol_sha256": digest(PROTOCOL), "script_sha256": digest(__file__),
              "properties": properties, "source": source, "runtime_logs": logs,
              "rows_evaluated": len(rows), "bookkeeping_failures": errors, "linear": linear, "angular": angular,
              "input_sha256": {p.name: digest(p) for p in (episode / "setup.json", episode / "manifest.json") if p.is_file()},
              "material_interface_work_convergence": "NOT_SCORED_SEPARATE_AUDIT", "full_physics_acceptance": False}
    statuses = [linear["status"], angular["status"]]
    result["status"] = ("NOT_SCORED_INCOMPLETE_OR_INVALID_BOOKKEEPING" if invalid or not rows else
                         "FAIL_ENGINEERING_IMPULSE_BALANCE" if any(s.startswith("FAIL") for s in statuses) else
                         "PARTIAL_ENGINEERING_IMPULSE_AUDIT" if any(s.startswith("NOT_SCORED") for s in statuses) else
                         "PASS_ENGINEERING_IMPULSE_BALANCE")
    return result, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    protocol = read_json(PROTOCOL)
    result, rows = audit(args.episode, protocol)
    (args.output_dir / "protocol.json").write_text(PROTOCOL.read_text())
    with (args.output_dir / "substeps.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "result": str(args.output_dir / "result.json")}))
    return (0 if result["status"].startswith("PASS") else
            1 if result["status"].startswith("FAIL") else 2)


if __name__ == "__main__":
    raise SystemExit(main())
