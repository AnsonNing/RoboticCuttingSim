"""Process-isolated CRESSim candidate; explicitly reports material/tool mismatch."""
import csv
import subprocess
import time

import numpy as np
import yaml

from .geometry import Specimen,sample_particles,specimen_dict
from .recording import ROOT,manifest,write_json
from .trajectory import Trajectory,reference_trajectory


def run(args,output):
    executable=ROOT/".deps/cressim-build/cressim_probe"
    if not executable.is_file():
        raise RuntimeError("Build the CRESSim adapter with bash scripts/build_cressim.sh")
    if args.case=="indent_unload":
        raise ValueError("CRESSim indentation punch has not been implemented; use settle or scraping")
    spec=Specimen.generate(args.seed,args.variant); spacing=args.voxel_size/2
    points,mass,_=sample_particles(spec,spacing,1500)
    trajectory=Trajectory.load(args.trajectory) if args.trajectory else reference_trajectory(args.case)
    trajectory.save(output/"blade_trajectory.csv")
    write_json(output/"specimen.json",specimen_dict(spec,points,mass,spacing))
    np.savetxt(output/"particles.txt",np.c_[points,mass],fmt="%.10g")
    config={"case":args.case,"voxel_size_m":args.voxel_size,"spacing_m":spacing,"frame_dt_s":args.dt,
            "subdt_s":args.dt/args.substeps,"frames":args.frames,
            "material":"co-rotational elastic, E=50000 Pa, nu=0.3; plasticity absent",
            "tool":"QuadSlicer XY plane at Newton blade lower edge; different collider semantics"}
    write_json(output/"manifest.json",manifest(config,"cressim_gpu_mls_mpm"))
    command=[str(executable),str(output/"particles.txt"),str(output/"blade_trajectory.csv"),str(output/"frames.bin"),
             str(args.frames),str(args.dt),str(args.dt/args.substeps),str(args.voxel_size),"slicer","50000"]
    tick=time.perf_counter()
    with open(output/"solver.log","w") as log:
        completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    elapsed=time.perf_counter()-tick
    if completed.returncode:
        raise RuntimeError(f"CRESSim exit={completed.returncode}; see solver.log")
    values=np.fromfile(output/"frames.bin",dtype=np.float32).reshape(args.frames+1,len(points),7)
    if not np.isfinite(values).all():
        raise FloatingPointError("CRESSim exported nonfinite state")
    positions=values[:,:,:3]; velocities=values[:,:,3:6]; masses=values[:,:,6]
    mass_error=np.max(np.abs(masses.sum(axis=1,dtype=float)-mass.sum())/mass.sum())
    poses=np.array([trajectory.at(i*args.dt)[0] for i in range(args.frames+1)])
    poses[:,2] -= .010 # export actual slicer plane, not Newton box center
    np.savez_compressed(output/"physics_state.npz",positions=positions,velocities=velocities,time_s=np.arange(args.frames+1)*args.dt,
                        mass_kg=masses[0],tool_pose_xyzw=poses,tool_half_extents=[.010,.025,.0005])
    with open(output/"trace.csv","w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=["time_s","mass_kg","max_z_m","max_speed_m_s","blade_fx_N","blade_fz_N"])
        writer.writeheader()
        for i in range(args.frames):
            writer.writerow({"time_s":(i+1)*args.dt,"mass_kg":float(masses[i+1].sum(dtype=float)),
                             "max_z_m":float(positions[i+1,:,2].max()),"max_speed_m_s":float(np.linalg.norm(velocities[i+1],axis=1).max()),
                             "blade_fx_N":"nan","blade_fz_N":"nan"})
    max_speed=float(np.linalg.norm(velocities,axis=2).max())
    # Broad engineering guard, not a calibrated physical acceptance threshold.
    stable=max_speed<2.0 and positions[:,:,:2].min()>-.15 and positions[:,:,:2].max()<.15
    result={"status":"PASS" if stable else "FAIL","scope":"GPU QuadSlicer execution with broad numerical sanity guard","particle_count":len(points),
            "wall_s":elapsed,"simulation_s":args.frames*args.dt,"sim_s_per_wall_s":args.frames*args.dt/elapsed,
            "mass_error_relative":float(mass_error),"max_speed_m_s":max_speed,
            "numerical_sanity_guard":{"max_speed_limit_m_s":2.0,"passed":bool(stable)},
            "persistent_separation_valid":None,"physics_proven":False,
            "limitations":["Only elastic co-rotational law, no equivalent Newton plasticity",
                           "QuadSlicer and Newton box differ; not a controlled quantitative ranking",
                           "No persistent cut labels exported; withdrawal/recontact gate remains open"]}
    write_json(output/"result.json",result)
    from .preview import make_preview
    make_preview(output)
    return result
