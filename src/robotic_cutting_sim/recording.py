import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, Path):
            return str(value)
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(data, indent=2, default=convert, allow_nan=False)+"\n")


def command(args, timeout=30, cwd=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return {"command": args, "returncode": r.returncode, "stdout": r.stdout, "stderr": r.stderr}
    except (OSError, subprocess.TimeoutExpired) as e:
        def captured(name):
            value = getattr(e, name, None) or ""
            return value.decode(errors="replace") if isinstance(value, bytes) else value
        return {"command": args, "returncode": None, "stdout": captured("stdout"),
                "stderr": captured("stderr"), "error": str(e)}


def _successful_output(receipt):
    return receipt["stdout"] if receipt["returncode"] == 0 else None


def manifest(config, backend):
    packages = {}
    for name in ["newton", "warp-lang", "numpy", "scipy", "isaaclab", "isaacsim", "torch", "usd-core",
                 "mujoco", "mujoco-warp", "newton-usd-schemas", "ovrtx", "ovstage"]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    sources = {}
    for name in ["newton", "CRESSim-MPM", "IsaacLab"]:
        checkout = ROOT/".deps"/name
        if checkout.is_dir():
            checks = {"diff": command(["git", "diff", "HEAD"], cwd=checkout),
                      "revision": command(["git", "rev-parse", "HEAD"], cwd=checkout)}
            diff, revision = (_successful_output(checks[key]) for key in ("diff", "revision"))
            sources[name] = {"sha": revision.strip() if revision is not None else None,
                             "tracked_patch_sha256": hashlib.sha256(diff.encode()).hexdigest()
                                                     if diff is not None else None,
                             "dirty": bool(diff) if diff is not None else None}
            failed = {key: value for key, value in checks.items() if value["returncode"] != 0}
            if failed:
                sources[name]["command_errors"] = failed
    git = {"revision": command(["git", "rev-parse", "HEAD"], cwd=ROOT),
           "status": command(["git", "status", "--porcelain"], cwd=ROOT)}
    revision, status = (_successful_output(git[key]) for key in ("revision", "status"))
    git_errors = {key: value for key, value in git.items() if value["returncode"] != 0}
    return {"timestamp": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "command": sys.argv, "python": sys.version, "backend": backend,
            "git_sha": revision.strip() if revision is not None else None,
            "git_dirty": bool(status) if status is not None else None,
            **({"git_command_errors": git_errors} if git_errors else {}),
            "config": config, "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
            "packages": packages, "source_checkouts":sources,
            "local_code_sha256": {str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted((ROOT/"src").rglob("*.py"))},
            "physics_proven": False, "vision_proven": False}


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


PHYSICS_STREAM_SCHEMA = 'physical_state_stream_v1'
_PHYSICS_SERIES = ('time_s', 'positions', 'velocities', 'codes')


def _atomic_checkpoint(path, data):
    """Publish a checkpoint only after its original sample bytes are durable."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    write_json(temporary, data)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class PhysicsStreamWriter:
    """Opt-in exact-dtype append files; one committed prefix survives a hard kill.

    The four files may have an uncommitted tail after interruption. Readers use
    only ``sample_count`` from the atomic checkpoint. No material history or
    final stacking allocation is retained by this writer.
    """
    def __init__(self, output, reference, *, expected_duration_s):
        self.output = Path(output)
        self.checkpoint_path = self.output / 'physics_stream.json'
        self.particles = len(reference['mass_kg'])
        reference_path = self.output / 'physics_reference.npz'
        with reference_path.open('xb') as stream:
            np.savez_compressed(stream, **reference)
            stream.flush()
            os.fsync(stream.fileno())
        self.streams = {key: (self.output / f'physics_{key}.bin').open('xb') for key in _PHYSICS_SERIES}
        self.checkpoint = dict(schema=PHYSICS_STREAM_SCHEMA, reference_path=reference_path.name,
            reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            expected_duration_s=float(expected_duration_s), sample_count=0, first_time_s=None,
            last_time_s=None, finalized=False, complete=False, arrays={})
        _atomic_checkpoint(self.checkpoint_path, self.checkpoint)

    def append(self, time_s, state):
        if self.checkpoint['finalized']:
            raise RuntimeError('Physical recording is finalized')
        t = float(time_s)
        if not np.isfinite(t) or (self.checkpoint['sample_count'] and t <= self.checkpoint['last_time_s']):
            raise ValueError('Physical sample timestamps must increase strictly')
        values = dict(time_s=np.asarray(t, np.float64),
                      **{key: np.asarray(state[key]) for key in _PHYSICS_SERIES[1:]})
        shapes = dict(time_s=(), positions=(self.particles, 3), velocities=(self.particles, 3),
                      codes=(self.particles,))
        for key, array in values.items():
            layout = dict(path=f'physics_{key}.bin', dtype=array.dtype.str, shape=list(array.shape))
            if array.shape != shapes[key] or array.dtype.hasobject:
                raise ValueError(f'Invalid physical sample array {key}')
            if key in self.checkpoint['arrays'] and layout != self.checkpoint['arrays'][key]:
                raise ValueError('Physical sample dtype or shape changed')
            self.checkpoint['arrays'][key] = layout
        for key, array in values.items():
            self.streams[key].write(memoryview(np.ascontiguousarray(array)).cast('B'))
        for stream in self.streams.values():
            stream.flush()
            os.fsync(stream.fileno())
        self.checkpoint.update(sample_count=self.checkpoint['sample_count'] + 1,
            first_time_s=t if self.checkpoint['sample_count'] == 0 else self.checkpoint['first_time_s'],
            last_time_s=t)
        _atomic_checkpoint(self.checkpoint_path, self.checkpoint)

    def finalize(self):
        if self.checkpoint['finalized']:
            return self.checkpoint
        for stream in self.streams.values():
            stream.close()
        first, last = self.checkpoint['first_time_s'], self.checkpoint['last_time_s']
        complete = bool(self.checkpoint['sample_count'] >= 2 and first == 0.
            and last - first >= self.checkpoint['expected_duration_s'] - 1e-9)
        self.checkpoint.update(finalized=True, complete=complete)
        _atomic_checkpoint(self.checkpoint_path, self.checkpoint)
        return self.checkpoint


def physical_recording_paths(output):
    """Original physical input paths for hashes/media, without converting data."""
    output = Path(output)
    if (output / 'physics_stream.json').exists():
        paths = [output / 'physics_stream.json', output / 'physics_reference.npz',
                 *[output / f'physics_{key}.bin' for key in _PHYSICS_SERIES]]
        if (output / 'physics_phases.npz').exists():
            paths.append(output / 'physics_phases.npz')
        return paths
    return [output / 'physics_state.npz']


def load_physics_recording(output):
    """Read legacy NPZ or a committed streamed prefix, without writing to it.

    Streamed arrays are read-only memory maps of the recorded original dtype.
    The completion receipt is separate from the unchanged mechanical evaluator.
    """
    output = Path(output)
    checkpoint_path = output / 'physics_stream.json'
    if not checkpoint_path.exists():
        with np.load(output / 'physics_state.npz', allow_pickle=False) as source:
            return {key: source[key] for key in source.files}, None
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint['schema'] != PHYSICS_STREAM_SCHEMA or checkpoint['reference_path'] != 'physics_reference.npz':
        raise ValueError('Unsupported physical stream checkpoint')
    reference = output / 'physics_reference.npz'
    if hashlib.sha256(reference.read_bytes()).hexdigest() != checkpoint['reference_sha256']:
        raise ValueError('Physical stream reference hash mismatch')
    with np.load(reference, allow_pickle=False) as source:
        data = {key: source[key] for key in source.files}
    if (output / 'physics_phases.npz').exists():
        with np.load(output / 'physics_phases.npz', allow_pickle=False) as source:
            data.update({key: source[key] for key in source.files})
    count = checkpoint['sample_count']
    if isinstance(count, bool) or not isinstance(count, int) or count < 0:
        raise ValueError('Invalid physical stream sample count')
    if not count:
        return {**data, 'time_s': np.empty(0)}, checkpoint
    for key in _PHYSICS_SERIES:
        layout = checkpoint['arrays'][key]
        dtype = np.dtype(layout['dtype'])
        path = output / f'physics_{key}.bin'
        if layout['path'] != path.name or dtype.hasobject:
            raise ValueError('Invalid physical stream array layout')
        shape = (count, *layout['shape'])
        required = math_prod(shape) * dtype.itemsize
        if path.stat().st_size < required or (checkpoint['finalized'] and path.stat().st_size != required):
            raise ValueError('Physical stream committed bytes are missing or final size changed')
        data[key] = np.memmap(path, dtype=dtype, mode='r', shape=shape)
    if (float(data['time_s'][0]) != checkpoint['first_time_s']
            or float(data['time_s'][-1]) != checkpoint['last_time_s']):
        raise ValueError('Physical stream timestamp differs from its checkpoint')
    return data, checkpoint


def math_prod(shape):
    result = 1
    for value in shape:
        if not isinstance(value, int) or value < 0:
            raise ValueError('Invalid physical stream shape')
        result *= value
    return result
