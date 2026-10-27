#!/usr/bin/env python3
"""Reproduce CRESSim math controls and optional GPU probes in a fresh source archive.

Never edits .deps source/build. Run with project Python; --gpu requires a CUDA
toolkit and explicit CUDA_VISIBLE_DEVICES. Large/raw evidence stays in artifacts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
PIN = "09aa5009b8580351f516b6df7660e87821fc5eb6"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def command(argv, log: Path, *, cwd=None, expected=(0,)):
    tick = time.perf_counter()
    result = subprocess.run(list(map(str, argv)), cwd=cwd, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log.write_text(result.stdout)
    record = {"argv": list(map(str, argv)), "cwd": str(cwd) if cwd else None,
              "returncode": result.returncode, "wall_s": time.perf_counter()-tick,
              "log": str(log.relative_to(log.parents[1]))}
    dump(log.with_suffix(".command.json"), record)
    if result.returncode not in expected:
        raise RuntimeError(f"Exit {result.returncode}; see {log}")
    return result


def cpu_controls(source: Path, output: Path, label: str):
    include = ["include/math", "include/core", "include/utils", "include/engine",
               "include/geometry_utils", "include/solver", "src/solver"]
    executable = output / f"{label}_controls"
    command(["g++", "-std=c++17", "-O2", ROOT/"scripts/cressim_low_level_verify.cpp",
             *[f"-I{source/p}" for p in include], "-o", executable],
            output/f"{label}_compile.log")
    result = command([executable], output/f"{label}_controls.jsonl", expected=(0, 1))
    return {"returncode": result.returncode,
            "checks": [json.loads(line) for line in result.stdout.splitlines()]}


def build_gpu(source: Path, output: Path, nvcc: str, arch: str) -> Path:
    build = output/"build"
    command(["cmake", "-S", source, "-B", build,
             f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
             "-DCMAKE_BUILD_TYPE=Release", "-DENGINE_STATIC=OFF",
             "-DENABLE_EXAMPLES=OFF", "-DENABLE_TESTS=OFF", "-DENABLE_DEBUG_LOGGER=ON"],
            output/"gpu_configure.log")
    command(["cmake", "--build", build, "-j", "4"], output/"gpu_build.log")
    include = ["include/engine", "include/solver", "include/core", "include/math",
               "include/utils", "include/geometry_utils"]
    executable = output/"cressim_probe"
    cuda_include = Path(nvcc).resolve().parents[1]/"include"
    command(["g++", "-std=c++17", "-O2", ROOT/"scripts/cressim_probe.cpp",
             f"-I{cuda_include}", *[f"-I{source/p}" for p in include],
             f"-L{build/'bin'}", f"-Wl,-rpath,{build/'bin'}", "-lcrmpm_engine",
             "-o", executable], output/"gpu_adapter_compile.log")
    return executable


def gpu_probe(executable: Path, output: Path, case: str):
    import numpy as np
    from robotic_cutting_sim.geometry import Specimen, sample_particles, specimen_dict
    from robotic_cutting_sim.trajectory import reference_trajectory

    output.mkdir()
    points, mass, _ = sample_particles(Specimen.generate(42, "D0"), .002, 1500)
    trajectory = reference_trajectory(case)
    trajectory.save(output/"blade_trajectory.csv")
    dump(output/"specimen.json", specimen_dict(Specimen.generate(42, "D0"), points, mass, .002))
    np.savetxt(output/"particles.txt", np.c_[points, mass], fmt="%.10g")
    frames, dt, subdt = 120, 1/60, 1/6000
    tick = time.perf_counter()
    device = os.environ["CUDA_VISIBLE_DEVICES"].split(",")[0]
    memory_command = ["nvidia-smi", "-i", device,
                      "--query-gpu=index,memory.used,utilization.gpu",
                      "--format=csv,noheader,nounits", "--loop-ms=100"]
    with (output/"gpu_memory.csv").open("w") as memory_log:
        sampler = subprocess.Popen(memory_command, stdout=memory_log, stderr=subprocess.STDOUT)
        try:
            command([executable, output/"particles.txt", output/"blade_trajectory.csv",
                     output/"frames.bin", frames, dt, subdt, .004, "slicer", 50000], output/"solver.log")
        finally:
            sampler.terminate()
            sampler.wait()
    elapsed = time.perf_counter()-tick
    memory_samples = []
    for line in (output/"gpu_memory.csv").read_text().splitlines():
        fields = line.split(",")
        if len(fields) == 3:
            try:
                memory_samples.append(float(fields[1]))
            except ValueError:
                pass
    values = np.fromfile(output/"frames.bin", np.float32).reshape(frames+1, len(points), 7)
    positions, velocities, masses = values[:, :, :3], values[:, :, 3:6], values[:, :, 6]
    finite = bool(np.isfinite(values).all())
    speed = np.linalg.norm(velocities, axis=2).max(axis=1)
    min_z = float(positions[:, :, 2].min())
    stable = finite and float(speed.max()) < 2 and min_z >= -.001
    stable = stable and positions[:, :, :2].min() > -.15 and positions[:, :, :2].max() < .15
    poses = np.array([trajectory.at(i*dt)[0] for i in range(frames+1)])
    poses[:, 2] -= .010
    np.savez_compressed(output/"physics_state.npz", positions=positions, velocities=velocities,
                        mass_kg=masses[0], time_s=np.arange(frames+1)*dt,
                        tool_pose_xyzw=poses, tool_half_extents=[.010, .025, .0005])
    with (output/"trace.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "mass_kg", "max_z_m",
                                              "max_speed_m_s", "blade_fx_N", "blade_fz_N"])
        writer.writeheader()
        for i in range(frames+1):
            writer.writerow({"time_s": i*dt, "mass_kg": float(masses[i].sum(dtype=float)),
                             "max_z_m": float(positions[i, :, 2].max()),
                             "max_speed_m_s": float(speed[i]),
                             "blade_fx_N": "nan", "blade_fz_N": "nan"})
    result = {"status": "PASS" if stable else "FAIL", "scope": "numerical_sanity_only",
              "seed": 42, "particle_count": len(points), "frames": frames,
              "dt_s": dt, "subdt_s": subdt, "dx_m": .004, "wall_s": elapsed,
              "max_speed_m_s": float(speed.max()), "min_particle_z_m": min_z,
              "finite": finite, "mass_error_relative":
              float(np.abs(masses.sum(axis=1, dtype=float)-mass.sum()).max()/mass.sum()),
              "sampled_device_peak_MiB": max(memory_samples) if memory_samples else None,
              "gpu_memory_scope": "entire selected physical device including baseline; 100ms sampling",
              "gpu_memory_command": memory_command,
              "material": "uncalibrated co-rotational elastic E=50000Pa nu=.3",
              "persistent_cutting_status": "UNSUPPORTED", "physics_proven": False,
              "limitations": ["No plasticity, damage or persistent multi-field cut history",
                              "Native kinematic reaction forces unavailable",
                              "Moving-slicer G2P reflection ignores relative tool velocity"]}
    dump(output/"result.json", result)
    from robotic_cutting_sim.preview import make_preview
    make_preview(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/".deps/CRESSim-MPM")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--nvcc", default=shutil.which("nvcc"))
    parser.add_argument("--cuda-arch", default="120")
    args = parser.parse_args()
    if args.gpu and (not args.nvcc or not os.environ.get("CUDA_VISIBLE_DEVICES")):
        parser.error("--gpu requires --nvcc and an explicit CUDA_VISIBLE_DEVICES")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = output/"source"
    source.mkdir()
    original_diff = subprocess.check_output(["git", "-C", args.source, "diff"])
    archive = subprocess.check_output(["git", "-C", args.source, "archive", PIN])
    (output/"upstream.tar").write_bytes(archive)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(source, filter="data")
    result = {"upstream_sha": PIN, "upstream_archive_sha256": hashlib.sha256(archive).hexdigest(),
              "shared_source_diff_sha256": hashlib.sha256(original_diff).hexdigest(),
              "cpu": {}, "patches": {}, "gpu": {"status": "NOT_RUN"}}
    result["cpu"]["upstream"] = cpu_controls(source, output, "upstream")
    # Use patch(1) because git apply inside an untracked nested archive can skip all paths.
    for label, name in [("four_patch", "cressim_si_mls.patch"),
                        ("math_layout", "cressim_math_layout.patch")]:
        patch = ROOT/"scripts/patches"/name
        result["patches"][name] = sha(patch)
        command(["patch", "--batch", "--forward", "-p1", "-i", patch],
                output/f"{label}_apply.log", cwd=source)
        result["cpu"][label] = cpu_controls(source, output, label)
    if args.gpu:
        executable = build_gpu(source, output, args.nvcc, args.cuda_arch)
        result["gpu"] = {case: gpu_probe(executable, output/case, case)
                         for case in ("settle", "through_bump_cut")}
    result["shared_source_unchanged"] = original_diff == subprocess.check_output(
        ["git", "-C", args.source, "diff"])
    gpu_pass = not args.gpu or all(value["status"] == "PASS" for value in result["gpu"].values())
    result["status"] = ("PASS" if result["cpu"]["math_layout"]["returncode"] == 0
                        and result["shared_source_unchanged"] and gpu_pass else "FAIL")
    result["scope"] = "isolated_math_and_numerical_controls_only; physical_cutting_unsupported"
    dump(output/"summary.json", result)
    print(json.dumps({"status": result["status"], "summary": str(output/"summary.json")}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
