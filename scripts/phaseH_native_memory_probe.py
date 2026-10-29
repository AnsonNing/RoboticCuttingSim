#!/usr/bin/env python3
"""WP-11 Phase H1 diagnostic: attribute the Native (Isaac Sim) per-reset host RSS growth.

Opens the ``cutting_research_v1`` env on Isaac Sim Native (same scene/goal/planner
options as the Phase G reliability run), then runs ``--resets`` consecutive short
episodes (``--steps`` zero-delta HOLD steps each, cutting disarmed) in one process.
Per reset it records:

* host RSS at reset sub-stages (stage open, camera creation, material build/reset,
  grasp settle, PhysX initialize) via read-only wrappers around the runner methods;
* host RSS after the steps, after ``gc.collect()`` and after ``malloc_trim(0)``;
* the number of loaded ``Sdf.Layer`` objects and any layer from an earlier reset
  that is still loaded (stage retention);
* gc-tracked Python object counts by type (top growth vs the previous reset);
* optionally (``--tracemalloc``) the Python-heap growth by allocation site.

The wrappers only observe; they never change arguments or results. ``--camera off``
and ``--stage-reset``/``--mode`` select diagnostic variants. Output: ``<out>/probe.json`` (never
overwritten). Diagnostic only: not a reliability run, no task verdict.

  OMNI_KIT_ACCEPT_EULA=YES TMPDIR=/mnt/data/ningan/tmp .deps/isaacsim-venv/bin/python \
      scripts/phaseH_native_memory_probe.py --gpu 0 --resets 6 --steps 5 --out artifacts/wp11/phaseH/<new_dir>
"""
import argparse
import collections
import ctypes
import functools
import gc
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

HOLD = {'type': 'edge_delta', 'target': [0.]*6, 'frame': 'world', 'cut_enabled': False}


def rss_mib():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):
            return float(line.split()[1])/1024.
    return None


def type_counts():
    counts = collections.Counter()
    for obj in gc.get_objects():
        kind = type(obj)
        counts[f'{kind.__module__}.{kind.__qualname__}'] += 1
    return counts


class Marks:
    def __init__(self):
        self.reset_index, self.rows = None, []

    def mark(self, label):
        self.rows.append(dict(reset=self.reset_index, label=label, rss_mib=rss_mib(), t=time.monotonic()))


def wrap_method(owner, name, marks, label):
    original = getattr(owner, name)

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        marks.mark(f'{label}:before')
        try:
            return original(*args, **kwargs)
        finally:
            marks.mark(f'{label}:after')
    setattr(owner, name, wrapper)


def install_marks(marks):
    from robotic_cutting_sim import isaac_sim_runner as runner
    from robotic_cutting_sim import live_material, policy_env
    from isaacsim.core.simulation_manager import SimulationManager
    original_wait = runner.wait_for_workcell_stage

    def wait(*args, **kwargs):
        marks.mark('open_stage:returned')
        result = original_wait(*args, **kwargs)
        marks.mark('stage_loaded')
        return result
    runner.wait_for_workcell_stage = wait
    for name in ('_create_cameras', '_settle_grasp', '_author_task_fixture', '_update_surface', 'capture'):
        wrap_method(runner.LiveIsaacSimCutting, name, marks, name)
    wrap_method(live_material.LiveCuttingMaterial, '__init__', marks, 'material_init')
    wrap_method(live_material.LiveCuttingMaterial, 'reset', marks, 'material_reset')
    wrap_method(SimulationManager, 'initialize_physics', marks, 'initialize_physics')
    wrap_method(policy_env.PolicyReadyEnv, 'reset', marks, 'policy_env_reset')


def render_state():
    """Replicator / SyntheticData registries that outlive a stage reopen (counts only)."""
    out = {}
    try:
        from omni.replicator.core.scripts.utils.viewport_manager import ViewportManager
        textures = list(ViewportManager._hydra_textures.hydra_textures)
        out['replicator_hydra_textures'] = len(textures)
    except Exception as error:  # diagnostic only
        out['replicator_hydra_textures_error'] = repr(error)
    try:
        from omni.syntheticdata import SyntheticData
        out['syntheticdata_node_graphs'] = len(SyntheticData.Get()._nodeGraphs)
    except Exception as error:
        out['syntheticdata_node_graphs_error'] = repr(error)
    try:
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        out['render_product_prims'] = sum(1 for prim in stage.Traverse() if prim.GetTypeName() == 'RenderProduct')
    except Exception as error:
        out['render_product_prims_error'] = repr(error)
    try:
        import omni.graph.core as og
        out['omnigraph_graphs'] = len(og.get_all_graphs())
    except Exception as error:
        out['omnigraph_graphs_error'] = repr(error)
    return out


def loaded_layers():
    from pxr import Sdf
    return sorted(str(layer.identifier) for layer in Sdf.Layer.GetLoadedLayers())


def stage_loop(a, env, record, libc, start):
    """Isolate the stage export / open cost: no physics, cameras or material after the first reset."""
    from argparse import Namespace
    import omni.usd
    from robotic_cutting_sim import isaac_sim_runner as runner
    from robotic_cutting_sim.recording import write_json
    from robotic_cutting_sim.workcell import export
    env.reset(seed=42)
    backend = env.backend
    c = backend.config
    context = omni.usd.get_context()
    base = a.out/'stage_loop'
    base.mkdir()
    first = base/'workcell_000'
    first.mkdir()
    export(Namespace(variant='D0', family='dome', seed=c.seed, spacing=c.dx/2, asset_root=None), first)
    for k in range(a.resets):
        rss0 = rss_mib()
        source = first/'workcell.usda'
        if a.mode in ('stage_new_path', 'export_only'):
            directory = base/f'workcell_{k+1:03d}'
            directory.mkdir()
            export(Namespace(variant='D0', family='dome', seed=c.seed, spacing=c.dx/2, asset_root=None), directory)
            source = directory/'workcell.usda'
        rss_export = rss_mib()
        if a.mode != 'export_only':
            import omni.timeline
            omni.timeline.get_timeline_interface().stop()
            if a.mode == 'stage_close_open':
                context.close_stage()
                for _ in range(10):
                    backend.app.update()
            if a.mode == 'stage_reload':
                stage = context.get_stage()
                stage.GetSessionLayer().Clear()
                stage.GetRootLayer().Reload(force=True)
            elif a.mode == 'stage_new_empty':
                context.new_stage()
            else:
                if not context.open_stage(str(source)):
                    raise RuntimeError('open_stage failed')
                runner.wait_for_workcell_stage(backend.app, context, base/f'stage_load_{k:03d}.json')
            for _ in range(10):
                backend.app.update()
        rss_open = rss_mib()
        gc.collect()
        libc.malloc_trim(0)
        rss_trim = rss_mib()
        row = dict(reset=k, rss_before=rss0, rss_after_export=rss_export, rss_after_open=rss_open,
                   rss_after_malloc_trim=rss_trim, loaded_layers=len(loaded_layers()), render_state=render_state())
        record['resets'].append(row)
        print(f'[probe] {a.mode} {k}: rss {rss0:.0f} export={rss_export:.0f} open={rss_open:.0f} trim={rss_trim:.0f} '
              f'layers={row["loaded_layers"]}', flush=True)
        write_json(a.out/'probe.json', record)
    import numpy as np
    ys = np.asarray([r['rss_after_malloc_trim'] for r in record['resets'][1:]])
    record['slope_rss_after_malloc_trim_mib_per_reset'] = float(np.polyfit(np.arange(len(ys)), ys, 1)[0])
    record['wall_s'] = time.monotonic()-start
    write_json(a.out/'probe.json', record)
    print(json.dumps({k: v for k, v in record.items() if k.startswith('slope')}), flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--resets', type=int, default=6)
    p.add_argument('--steps', type=int, default=5)
    p.add_argument('--scene', default='trim01_wp11_workcell')
    p.add_argument('--goal', default='trim_bump_rho10')
    p.add_argument('--planner-options', default='{"max_rotation_deg": 1.0, "max_translation_m": 0.001}')
    p.add_argument('--camera', choices=('on', 'off'), default='on')
    p.add_argument('--camera-every', type=int, default=5)
    p.add_argument('--tracemalloc', type=int, default=0, help='Frames to keep (0: off)')
    p.add_argument('--stage-reset', choices=('reopen', 'reload_in_place'),
                   help='IsaacSimCuttingConfig.stage_reset (omitted: the runtime default)')
    p.add_argument('--mode', choices=('episodes', 'stage_new_path', 'stage_same_path', 'export_only',
                                           'stage_close_open', 'stage_reload', 'stage_new_empty'), default='episodes',
                   help="episodes: full env resets; stage_*: after one env reset, only re-export and/or reopen "
                        "the initial workcell stage (Kit/USD stage cost in isolation; the env is unusable afterwards)")
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=False)
    from robotic_cutting_sim.live_env import open_cutting_env
    from robotic_cutting_sim.recording import manifest, write_json
    marks = Marks()
    libc = ctypes.CDLL('libc.so.6')
    record = dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, resets=[])
    record['manifest'] = {k: v for k, v in manifest({}, 'phaseH_native_memory_probe').items()
                          if k in ('timestamp', 'command', 'git_sha', 'git_dirty', 'packages', 'python')}
    options = dict(gpu=a.gpu, device=f'cuda:{a.gpu}', material_dt=1e-4, camera=a.camera == 'on')
    if a.stage_reset is not None:
        options['stage_reset'] = a.stage_reset
    start = time.monotonic()
    with open_cutting_env(runtime='isaac_sim', output_dir=a.out/'runtime', seed=42, preset='cutting_research_v1',
                          scene=a.scene, goal=a.goal, planner_options=json.loads(a.planner_options),
                          preset_overrides=dict(camera_every=a.camera_every), **options) as env:
        record['launch_wall_s'] = time.monotonic()-start
        install_marks(marks)   # Kit modules import only after the app launched; no reset has run yet
        if a.tracemalloc:
            tracemalloc.start(a.tracemalloc)
        previous_counts, previous_snapshot, first_layers = None, None, None
        if a.mode != 'episodes':
            return stage_loop(a, env, record, libc, start)
        for k in range(a.resets):
            marks.reset_index = k
            t0, rss0 = time.monotonic(), rss_mib()
            env.reset(seed=42)
            reset_wall = time.monotonic()-t0
            rss_reset = rss_mib()
            for _ in range(a.steps):
                env.step(HOLD)
            rss_steps = rss_mib()
            gc.collect()
            rss_gc = rss_mib()
            libc.malloc_trim(0)
            rss_trim = rss_mib()
            layers = loaded_layers()
            first_layers = first_layers or set(layers)
            counts = type_counts()
            growth = ([] if previous_counts is None else
                      sorted(((name, counts[name]-previous_counts.get(name, 0)) for name in counts
                              if counts[name]-previous_counts.get(name, 0) > 0), key=lambda kv: -kv[1])[:25])
            row = dict(reset=k, reset_wall_s=reset_wall, rss_before_reset=rss0, rss_after_reset=rss_reset,
                       rss_after_steps=rss_steps, rss_after_gc=rss_gc, rss_after_malloc_trim=rss_trim,
                       loaded_layers=len(layers), new_layers_vs_reset0=sorted(set(layers)-first_layers),
                       initial_workcell_layers=[x for x in layers if 'initial_workcell_' in x],
                       gc_objects=sum(counts.values()), gc_type_growth=growth, render_state=render_state())
            if a.tracemalloc:
                snapshot = tracemalloc.take_snapshot()
                row['tracemalloc_traced_mib'] = tracemalloc.get_traced_memory()[0]/2**20
                if previous_snapshot is not None:
                    stats = snapshot.compare_to(previous_snapshot, 'lineno')
                    row['tracemalloc_top_growth'] = [dict(site=str(s.traceback), size_diff_kib=s.size_diff/1024.,
                                                          count_diff=s.count_diff) for s in stats[:20]]
                previous_snapshot = snapshot
            previous_counts = counts
            record['resets'].append(row)
            print(f'[probe] reset {k}: wall={reset_wall:.1f}s rss {rss0:.0f}->{rss_reset:.0f} steps={rss_steps:.0f} '
                  f'gc={rss_gc:.0f} trim={rss_trim:.0f} layers={len(layers)} gc_objects={row["gc_objects"]} '
                  f'render={row["render_state"]}',
                  flush=True)
            record['marks'] = marks.rows
            write_json(a.out/'probe.json', record)
    rows = record['resets']
    if len(rows) >= 3:
        import numpy as np
        xs = np.arange(1, len(rows))
        for key in ('rss_after_steps', 'rss_after_malloc_trim'):
            ys = np.asarray([r[key] for r in rows[1:]])
            record[f'slope_{key}_mib_per_reset'] = float(np.polyfit(xs, ys, 1)[0])
    record['wall_s'] = time.monotonic()-start
    write_json(a.out/'probe.json', record)
    print(json.dumps({k: v for k, v in record.items() if k.startswith('slope')}), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
