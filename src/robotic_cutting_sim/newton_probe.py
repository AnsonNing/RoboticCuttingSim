"""Kinematic-tool Newton 1.6.1 implicit MPM testbed. No cutting claim."""
import csv
import time
from dataclasses import asdict

import numpy as np
import yaml

from .geometry import Specimen, sample_particles, specimen_dict
from .recording import ROOT, manifest, write_json
from .trajectory import Trajectory, reference_trajectory


class NewtonTestbed:
    def __init__(self, points, masses, spacing, voxel_size, case, material, device="cuda:0"):
        import newton
        import warp as wp
        from newton.solvers import SolverImplicitMPM

        if newton.__version__ != "1.6.1" or wp.__version__ != "1.17.0":
            raise RuntimeError("This testbed requires Newton 1.6.1 and Warp 1.17.0")
        wp.init()
        if not wp.get_device(device).is_cuda:
            raise RuntimeError("Implicit MPM requires a CUDA GPU; CPU geometry is available via scene")
        self.wp, self.device = wp, device
        self.half_extents = np.array([.010, .010, .005] if case == "indent_unload" else [.001, .025, .010])
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            SolverImplicitMPM.register_custom_attributes(builder)
            builder.add_particles(pos=points.tolist(), vel=np.zeros_like(points).tolist(),
                                  mass=masses.tolist(), radius=[spacing*.5]*len(points))
            # Finite tray floor and four rigid walls; no particle pinning.
            self.fixture_boxes = [([0,0,-.005], [.068,.048,.005]),
                                  ([-.064,0,.0075], [.004,.044,.0075]),
                                  ([.064,0,.0075], [.004,.044,.0075]),
                                  ([0,-.044,.0075], [.060,.004,.0075]),
                                  ([0,.044,.0075], [.060,.004,.0075])]
            cfg = newton.ModelBuilder.ShapeConfig(mu=material["friction"], density=0.0, margin=0.0)
            for center, half in self.fixture_boxes:
                builder.add_shape_box(body=-1, xform=wp.transform(wp.vec3(*center), wp.quat_identity()),
                                      hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
            self.blade = builder.add_body(is_kinematic=True, label="blade_proxy")
            builder.add_shape_box(body=self.blade, hx=self.half_extents[0], hy=self.half_extents[1],
                                  hz=self.half_extents[2], cfg=cfg, label="blade_collider")
            self.model = builder.finalize(device=device)
            self.model.set_gravity((0,0,-9.81))
            for name, value in {"young_modulus": material["young_modulus_pa"],
                                "poisson_ratio": material["poisson_ratio"],
                                "yield_stress": material["yield_stress_pa"],
                                "friction": 0.0, "tensile_yield_ratio": 1.0,
                                "damping": 0.001, "viscosity": 10.0}.items():
                getattr(self.model.mpm, name).fill_(value)
            options = SolverImplicitMPM.Config(voxel_size=voxel_size, max_iterations=100,
                                               tolerance=1e-5, grid_type="sparse",
                                               strain_basis="P0", velocity_basis="Q1",
                                               collider_basis="S2", critical_fraction=0.0)
            self.solver = SolverImplicitMPM(self.model, config=options)
            self.state, self.next_state = self.model.state(), self.model.state()
            self.collider_bodies = self.solver.collider_body_index.numpy()
        self.initial_mass = float(self.model.particle_mass.numpy().astype(float).sum())

    def step(self, dt, pose, twist):
        wp = self.wp
        with wp.ScopedDevice(self.device):
            self.state.body_q.assign(np.asarray([pose], dtype=np.float32))
            self.state.body_qd.assign(np.asarray([twist], dtype=np.float32))
            self.solver.step(self.state, self.next_state, None, None, dt)
            self.solver.project_outside(self.next_state, self.next_state, dt)
            self.state, self.next_state = self.next_state, self.state
            impulses, _, ids = self.solver.collect_collider_impulses(self.state)
            ids_np, imp_np = ids.numpy(), impulses.numpy()
            valid = (ids_np >= 0) & (ids_np < len(self.collider_bodies))
            selected = np.zeros(len(ids_np), dtype=bool)
            selected[valid] = self.collider_bodies[ids_np[valid]] == self.blade
            return imp_np[selected].sum(axis=0) if selected.any() else np.zeros(3)

    def snapshot(self):
        return self.state.particle_q.numpy(), self.state.particle_qd.numpy()


def run(args, output):
    spec = Specimen.generate(args.seed, args.variant)
    material = yaml.safe_load((ROOT/"configs/materials/plasticine_provisional.yaml").read_text())
    spacing = args.voxel_size / 2
    points, masses, _ = sample_particles(spec, spacing, material["density_kg_m3"])
    trajectory = Trajectory.load(args.trajectory) if args.trajectory else reference_trajectory(args.case)
    trajectory.save(output/"blade_trajectory.csv")
    write_json(output/"specimen.json", specimen_dict(spec, points, masses, spacing))
    config = {"specimen": asdict(spec), "case": args.case, "voxel_size_m": args.voxel_size,
              "particle_spacing_m": spacing, "dt_s": args.dt, "substeps": args.substeps,
              "frames": args.frames, "material": material, "device": args.device,
              "trajectory": trajectory.data.tolist(), "tool": "box collision proxy"}
    episode = manifest(config, "newton_implicit_mpm")
    write_json(output/"manifest.json", episode)
    wall_start = time.perf_counter()
    bed = NewtonTestbed(points, masses, spacing, args.voxel_size, args.case, material, args.device)
    setup_s = time.perf_counter()-wall_start
    frames, velocities, times, tool_poses = [points], [np.zeros_like(points)], [0.0], []
    rows = []
    sub_dt = args.dt / args.substeps
    start_sim = time.perf_counter()
    for frame in range(args.frames):
        start = time.perf_counter()
        impulse = np.zeros(3)
        for sub in range(args.substeps):
            t = frame*args.dt + sub*sub_dt
            pose, twist = trajectory.at(t)
            impulse += bed.step(sub_dt, pose, twist)
        t = (frame+1)*args.dt
        q, v = bed.snapshot()
        if not np.isfinite(q).all() or not np.isfinite(v).all():
            raise FloatingPointError(f"Nonfinite material state at frame {frame}")
        pose, _ = trajectory.at(t)
        from scipy.spatial.transform import Rotation
        local = Rotation.from_quat(pose[3:]).inv().apply(q-pose[:3])
        distances = bed.half_extents - np.abs(local)
        penetration = np.maximum(np.min(distances, axis=1), 0)
        force = impulse/args.dt
        jp = bed.state.mpm.particle_Jp.numpy()
        rows.append({"time_s": t, "frame_wall_s": time.perf_counter()-start,
                     "mass_kg": bed.initial_mass, "particle_count": len(q),
                     "mass_error_relative": 0.0, "min_z_m": float(q[:,2].min()),
                     "max_z_m": float(q[:,2].max()), "max_speed_m_s": float(np.linalg.norm(v,axis=1).max()),
                     "rms_displacement_m": float(np.sqrt(np.mean(np.sum((q-points)**2,axis=1)))),
                     "blade_particle_center_penetration_m": float(penetration.max()),
                     "max_abs_Jp_minus_one": float(np.max(np.abs(jp-1))),
                     "blade_fx_N": float(force[0]), "blade_fy_N": float(force[1]), "blade_fz_N": float(force[2]),
                     "blade_x_m": float(pose[0]), "blade_y_m": float(pose[1]), "blade_z_m": float(pose[2])})
        frames.append(q.copy()); velocities.append(v.copy()); times.append(t); tool_poses.append(pose)
        if (frame+1) % 15 == 0:
            print(f"frame={frame+1}/{args.frames} t={t:.4f}s max_speed={rows[-1]['max_speed_m_s']:.4f}", flush=True)
    sim_wall_s = time.perf_counter()-start_sim
    with open(output/"trace.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(output/"physics_state.npz", positions=np.stack(frames), velocities=np.stack(velocities),
                        time_s=np.array(times), mass_kg=bed.model.particle_mass.numpy(),
                        tool_pose_xyzw=np.array(tool_poses), tool_half_extents=bed.half_extents)
    # Constant particle masses are not a geometric chip/substrate budget.
    result = {"status": "PASS", "scope": "finite MPM contact/deformation execution only",
              "case": args.case, "particle_count": len(points), "initial_mass_kg": bed.initial_mass,
              "mass_error_relative": 0.0, "mass_budget_scope": "all unchanged active particles; no deletion/spawn",
              "setup_wall_s": setup_s, "physics_wall_s": sim_wall_s,
              "simulation_s": args.frames*args.dt, "sim_s_per_wall_s": args.frames*args.dt/sim_wall_s,
              "max_speed_m_s": max(r["max_speed_m_s"] for r in rows),
              "max_blade_particle_center_penetration_m": max(r["blade_particle_center_penetration_m"] for r in rows),
              "peak_blade_force_N": max(np.linalg.norm([r[k] for k in ["blade_fx_N","blade_fy_N","blade_fz_N"]]) for r in rows),
              "final_rms_displacement_m": rows[-1]["rms_displacement_m"],
              "max_abs_Jp_minus_one": max(r["max_abs_Jp_minus_one"] for r in rows),
              "solver_residual": None, "persistent_separation_valid": None,
              "physics_proven": False, "calibration": "uncalibrated"}
    write_json(output/"result.json", result)
    from .preview import make_preview
    make_preview(output)
    return result
