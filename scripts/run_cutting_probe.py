#!/usr/bin/env python3
"""Run an experimental persistent-cut TLMPM probe; retain raw state and failures."""
import argparse
import csv
from dataclasses import asdict
from pathlib import Path
import time

import numpy as np
import warp as wp

from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.geometry import Specimen, sample_particles, specimen_dict
from robotic_cutting_sim.recording import manifest, new_output, write_json
from robotic_cutting_sim.tlmpm import TLMPM, TLMPMConfig


def tool_plan(times, case, z=.0155):
    """World edge poses; advance +X with a horizontal finite wedge, then withdraw."""
    knots = np.array([[0, -.030, .06], [.20, -.030, z], [.30, -.030, z],
                      [1.30, .045, z], [1.45, .080, .06], [3., .080, .06]])
    if case == "no_contact":
        knots[:, 2] += .1
    elif case in ("cut_recontact", "two_cuts", "cross_cuts", "supported_cross_cuts"):
        # A shorter sweep and vertical withdrawal leave the chip above the
        # substrate, where gravity produces genuine cut-face recontact.
        knots = np.array([[0, -.030, .06], [.20, -.030, z], [.30, -.030, z],
                          [1.30, .022, z], [1.45, .022, .06], [3., .022, .06]])
    elif case == "indent":
        knots = np.array([[0, 0., .06], [.3, 0., .06], [.55, 0., .018],
                          [.8, 0., .018], [1., 0., .06], [3., 0., .06]])
    position = np.column_stack([np.interp(times, knots[:, 0], knots[:, 1]),
                                np.zeros(len(times)), np.interp(times, knots[:, 0], knots[:, 2])])
    velocity = np.gradient(position, times, axis=0)
    poses = np.c_[position, np.tile([0, 0, 0, 1], (len(times), 1))].astype(np.float32)
    twists = np.c_[velocity, np.zeros_like(velocity)].astype(np.float32)
    active = ((times >= .30) & (times < 1.30) & (case in ("cut", "cut_recontact", "two_cuts", "cross_cuts", "supported_cross_cuts", "no_contact"))).astype(np.int32)-1
    if case in ("two_cuts", "cross_cuts", "supported_cross_cuts"):
        from scipy.spatial.transform import Rotation, Slerp
        second = times >= 1.7
        t = times[second]
        second_position = np.column_stack([np.interp(t, [1.7, 2., 4.5], [.022, 0., 0.]),
            np.interp(t, [1.7, 2., 2.4, 3.4, 4.5], [0., -.030, -.030, .030, .030]),
            np.interp(t, [1.7, 2., 2.3, 3.4, 3.55, 4.5], [.06, .06, .0125, .0125, .06, .06])])
        rotations = Slerp([1.7, 2., 4.5], Rotation.from_euler("z", np.array([0, 90, 90])[:, None], degrees=True))
        if case in ("cross_cuts", "supported_cross_cuts"):
            # Controlled free-specimen/zero-g test: the second finite edge spans
            # the full cross section. No rigid tray wall is traversed or ignored.
            second_position = np.column_stack([np.interp(t, [1.7, 2., 4.5], [.022, 0., 0.]),
                np.interp(t, [1.7, 2., 2.4, 3.4, 4.5], [0., -.055, -.055, .055, .055]),
                np.interp(t, [1.7, 2., 2.3, 3.4, 3.55, 4.5], [.06, .06, .025, .025, .08, .08])])
            vertical_edge = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
            rotations = Slerp([1.7, 2., 4.5], Rotation.from_matrix(np.array([np.eye(3), vertical_edge, vertical_edge])))
        poses[second, :3] = second_position
        poses[second, 3:] = rotations(t).as_quat()
        twists[second, :3] = np.gradient(second_position, t, axis=0)
        turning = (times >= 1.7) & (times < 2.)
        turn = rotations.rotations[1]*rotations.rotations[0].inv()
        twists[turning, 3:] = turn.as_rotvec()/.3
        active[(times >= 2.4) & (times < 3.4)] = 1
    # Active cut zero, inactive -1. Full history is retained after this interval.
    return poses, twists, active


def run(args, output):
    cut_count = 2 if args.case in ("two_cuts", "cross_cuts", "supported_cross_cuts") else 1
    material = TLMPMConfig(dx=args.dx, dt=args.dt, cuts=cut_count, pic=args.pic,
                           cutting_energy=args.cutting_energy, young=args.young,
                           yield_stress=args.yield_stress, tool=args.case != "settle",
                           floor=args.case != "cross_cuts",
                           fixture_walls=args.case != "supported_cross_cuts",
                           gravity=(0., 0., 0.) if args.case == "cross_cuts" else (0., 0., -9.81),
                           material_contact=not args.no_material_contact)
    spacing = args.dx/2
    spec = Specimen.generate(args.seed, args.variant, args.family)
    points, masses, lattice = sample_particles(spec, spacing, material.density)
    frames = int(round(args.duration/args.frame_dt))
    substeps = int(round(args.frame_dt/args.dt))
    if abs(substeps*args.dt-args.frame_dt) > 1e-9:
        raise ValueError("frame-dt must be an integer multiple of dt")
    times = np.arange(frames*substeps+1)*args.dt
    poses, twists, cuts = tool_plan(times, args.case, args.cut_height)
    np.savetxt(output/"tool_trajectory.csv", np.c_[times, poses, twists, cuts], delimiter=",",
               header="time_s,x_m,y_m,z_m,qx,qy,qz,qw,vx,vy,vz,wx,wy,wz,active_cut", comments="")
    write_json(output/"specimen.json", specimen_dict(spec, points, masses, spacing))
    config = {"solver": asdict(material), "case": args.case, "seed": args.seed,
              "variant": args.variant, "family": args.family, "device": args.device,
              "cut_height_m": args.cut_height, "duration_s": args.duration,
              "frame_dt_s": args.frame_dt, "implementation": "independent experimental TLMPM"}
    write_json(output/"manifest.json", manifest(config, "experimental_tlmpm_persistent_cut"))
    tick = time.perf_counter()
    bed = TLMPM(points, masses, spacing, material, args.device)
    bed.set_trajectory(poses, twists, cuts)
    wp.load_module(module="robotic_cutting_sim.tlmpm", device=args.device)
    with wp.ScopedCapture(device=args.device) as capture:
        for _ in range(substeps):
            bed.step()
    setup = time.perf_counter()-tick
    records = [{"positions": points.copy(), "velocities": np.zeros_like(points),
                "codes": np.zeros(len(points), np.int32)}]
    rows = []
    tick = time.perf_counter()
    failed = None
    for frame in range(frames):
        bed.reaction.zero_(); bed.stats.zero_()
        wp.capture_launch(capture.graph)
        wp.synchronize_device(args.device)
        q, v, code = bed.x.numpy(), bed.v.numpy(), bed.code.numpy()
        F = bed.F.numpy()
        determinant = np.linalg.det(F)
        stats, reaction = bed.stats.numpy(), bed.reaction.numpy()[0]
        speed = np.linalg.norm(v, axis=1)
        finite = bool(np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(F).all())
        row = {"time_s": (frame+1)*args.frame_dt, "mass_kg": float(masses.sum()),
               "max_speed_m_s": float(speed.max()), "min_z_m": float(q[:, 2].min()),
               "max_z_m": float(q[:, 2].max()), "min_detF": float(determinant.min()),
               "max_detF": float(determinant.max()), "unassigned": int((code == 0).sum()),
               "negative_side": int((code == 1).sum()), "positive_side": int((code == 2).sum()),
               "blade_fx_N": float(reaction[0]/args.frame_dt),
               "blade_fy_N": float(reaction[1]/args.frame_dt),
               "blade_fz_N": float(reaction[2]/args.frame_dt),
               "tool_work_j": float(stats[2]), "max_floor_penetration_m": float(stats[0]),
               "tool_contacts": int(stats[1]), "max_tool_penetration_m": float(stats[3]),
               "material_contacts": int(stats[4]), "material_contact_impulse_Ns": float(stats[5]),
               "max_material_overlap_m": float(stats[6]), "finite": finite}
        rows.append(row)
        records.append({"positions": q, "velocities": v, "codes": code})
        if (frame+1) % 25 == 0:
            print(f"frame={frame+1}/{frames} t={row['time_s']:.3f} speed={speed.max():.4f} "
                  f"sides={row['negative_side']}/{row['positive_side']} detF={determinant.min():.4f}", flush=True)
        if not finite or speed.max() > 5 or (material.floor and q[:, 2].min() < -.013) or determinant.min() < .05:
            failed = "Nonfinite, speed>5m/s, table penetration>3mm, or detF<.05 guard"
            break
    wall = time.perf_counter()-tick
    snapshots = {k: np.stack([r[k] for r in records]) for k in records[0]}
    final = bed.snapshot()
    components, component = mechanical_components(bed.nodes.numpy(), bed.weights.numpy(), final["codes"], cut_count,
                                                   gradients=bed.gradients.numpy())
    initial_components, _ = mechanical_components(bed.nodes.numpy(), bed.weights.numpy(), np.zeros(len(points), int), cut_count,
                                                   gradients=bed.gradients.numpy())
    np.savez_compressed(output/"physics_state.npz", **snapshots, mass_kg=masses,
                        reference_positions=points, reference_indices=lattice,
                        time_s=np.arange(len(records))*args.frame_dt,
                        tool_pose_xyzw=poses[np.arange(len(records))*substeps],
                        final_F=final["F"], final_Fpi=final["Fpi"], final_plastic_strain=final["plastic_strain"],
                        final_cut_progress=final["cut_progress"], final_contact_work_j=final["contact_work_j"],
                        component_ids=component,
                        component_provenance="final reference MPM stencil compatibility incl nonzero stress gradients")
    with (output/"trace.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    component_mass = np.bincount(component, weights=masses)
    labels = snapshots["codes"]
    nonzero_changed = False
    for cut in range(cut_count):
        digits = (labels//3**cut) % 3
        nonzero_changed |= bool(((digits[:-1] != 0) & (digits[1:] != digits[:-1])).any())
    withdrew = np.flatnonzero(np.arange(len(records))*args.frame_dt >= (3.55 if cut_count == 2 else 1.45))
    post = rows[withdrew[0]:] if len(withdrew) else []
    component_stats = []
    for index, mass in enumerate(component_mass):
        selected = component == index
        com = np.average(snapshots["positions"][:, selected], axis=1, weights=masses[selected])
        velocity = np.average(snapshots["velocities"][:, selected], axis=1, weights=masses[selected])
        component_stats.append({"id": index, "particles": int(selected.sum()), "mass_kg": float(mass),
                                "final_com_m": com[-1].tolist(), "final_velocity_m_s": velocity[-1].tolist(),
                                "initial_com_m": com[0].tolist(),
                                "withdraw_to_final_com_distance_m": float(np.linalg.norm(com[-1]-com[withdrew[0]])) if len(withdrew) else None})
    sanity = failed is None and max(r["max_speed_m_s"] for r in rows) < 2
    sanity = sanity and (not material.floor or min(r["min_z_m"] for r in rows) >= -.011)
    result = {"status": "PASS" if sanity else "FAIL", "scope": "experimental TLMPM numerical sanity",
              "case": args.case, "setup_wall_s": setup, "physics_wall_s": wall,
              "simulation_s": len(rows)*args.frame_dt, "particle_count": len(points),
              "initial_mass_kg": float(masses.sum()), "active_mass_error_relative": 0.0,
              "mass_ledger": "same immutable masses and particle IDs, no spawn/deletion",
              "initial_mechanical_components": initial_components, "final_mechanical_components": components,
              "component_provenance": "actual reference-node/stencil compatibility incl code0 bridges and stress gradients",
              "components": component_stats, "existing_side_digit_changed": nonzero_changed,
              "maximum_speed_m_s": max(r["max_speed_m_s"] for r in rows),
              "minimum_detF": min(r["min_detF"] for r in rows),
              "max_tool_penetration_m": max(r["max_tool_penetration_m"] for r in rows),
              "maximum_material_overlap_m": max(r["max_material_overlap_m"] for r in rows),
              "post_withdraw_material_contact_substep_pairs": sum(r["material_contacts"] for r in post),
              "guard_failure": failed, "force_scope": "equal-opposite wedge contact impulse/frame dt, diagnostic only",
              "physics_proven": False, "material_calibration": "uncalibrated",
              "maximum_accumulated_contact_work_j": float(final["contact_work_j"].max()),
              "total_accumulated_contact_work_j": float(final["contact_work_j"].sum()),
              "persistent_cutting_acceptance": "NOT_EVALUATED; mechanical components alone are insufficient",
              "limitations": ["Experimental independent solver, not BladeMaster release",
                              "Linear TL reference stencil; no hourglass stabilization",
                              "Dynamic Coulomb friction without tangential history",
                              "Kinematic tool, no two-way articulated coupling in this probe"]}
    write_json(output/"result.json", result)
    print(result, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--case", choices=["settle", "no_contact", "indent", "cut", "cut_recontact", "two_cuts", "cross_cuts", "supported_cross_cuts", "uncut_scrape"], default="cut")
    p.add_argument("--dx", type=float, default=.004)
    p.add_argument("--dt", type=float, default=.0001)
    p.add_argument("--frame-dt", type=float, default=.02)
    p.add_argument("--duration", type=float, default=2.5)
    p.add_argument("--cut-height", type=float, default=.0155)
    p.add_argument("--cutting-energy", type=float, default=.05)
    p.add_argument("--pic", type=float, default=.05)
    p.add_argument("--young", type=float, default=50000.)
    p.add_argument("--yield-stress", type=float, default=1000.)
    p.add_argument("--no-material-contact", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--variant", choices=["D0", "D1"], default="D0")
    p.add_argument("--family", default="dome")
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    result = run(args, new_output(args.output_dir))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
