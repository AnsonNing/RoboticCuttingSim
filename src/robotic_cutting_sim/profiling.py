"""Opt-in per-stage wall-time profiling of the cutting loop (WP-11 Phase H2).

Nothing in the simulator imports this module. Profiling exists only while a
``StageProfiler`` is installed: ``install()`` wraps a declared list of methods
(``STAGES``) with timers and ``uninstall()`` restores the original attributes, so
with no profiler installed every code path is exactly the unprofiled one (no
flag checks, no overhead).

* Stage timers are inclusive wall times (``time.perf_counter``). The exclusive
  time of a stage is its inclusive time minus the stages nested inside it.
* ``sync_device``: CUDA device to synchronize at the boundaries of the material
  stages, so asynchronous GPU work is charged to the stage that launched it
  (``tlmpm_substeps_graph`` = ``warp.capture_launch`` of the substep graph plus
  the device wait). Every other stage is host wall time. No synchronization is
  issued while a device is capturing a graph.
* ``kernels=True`` (a separate, perturbing mode): the CUDA graph of the material
  is dropped after each material reset so the identical kernel sequence runs
  eagerly, and every ``warp.launch`` is timed with a device synchronization
  before and after (CPU launches are synchronous). The totals of this mode are
  inflated by the synchronizations; use it for the kernel split only, and the
  stage mode for the stage times.

Rows are aggregated per control step (``PolicyReadyEnv.step``): mean / max per
step and the share of the summed control-step time. Episode-level stages
(reset, evaluation, recording finish, return home) are reported per call.
"""
from collections import defaultdict
import functools
import importlib
import time

import numpy as np

# (stage name, "module:Class.method" or "module:function"); missing targets are skipped
# (Isaac classes exist only in the Isaac runtime).
STAGES = (
    ("control_step", "robotic_cutting_sim.policy_env:PolicyReadyEnv.step"),
    ("reset", "robotic_cutting_sim.policy_env:PolicyReadyEnv.reset"),
    ("reset_checks", "robotic_cutting_sim.policy_env:PolicyReadyEnv.reset_checks"),
    ("action_validation", "robotic_cutting_sim.policy_env:PolicyReadyEnv._normalize"),
    ("action_validation", "robotic_cutting_sim.policy_env:PolicyReadyEnv._resolve"),
    ("collision_check", "robotic_cutting_sim.policy_env:PolicyReadyEnv._collision"),
    ("observation", "robotic_cutting_sim.policy_env:PolicyReadyEnv.observe"),
    ("camera_policy_refresh", "robotic_cutting_sim.policy_env:PolicyReadyEnv._refresh_cameras"),
    ("evaluation_topology", "robotic_cutting_sim.live_env:CuttingTaskEnv.evaluate"),
    ("return_home", "robotic_cutting_sim.live_env:CuttingTaskEnv.return_home"),
    ("oracle_history", "robotic_cutting_sim.live_env:CuttingTaskEnv._record_oracle"),
    ("proprio_refresh", "robotic_cutting_sim.live_env:CuttingTaskEnv._refresh"),
    ("backend_step", "robotic_cutting_sim.kinematic_env:KinematicCuttingBackend.step"),
    ("backend_step", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting.step"),
    ("material_advance", "robotic_cutting_sim.live_material:LiveCuttingMaterial.advance"),
    ("material_reset", "robotic_cutting_sim.live_material:LiveCuttingMaterial.reset"),
    ("material_debug_state", "robotic_cutting_sim.live_material:LiveCuttingMaterial.debug_state"),
    ("material_history_state", "robotic_cutting_sim.live_material:LiveCuttingMaterial.history_state"),
    ("tlmpm_substep_eager", "robotic_cutting_sim.tlmpm_v4:TLMPM.step"),
    ("tlmpm_substep_eager", "robotic_cutting_sim.tlmpm_v5:TLMPM.step"),
    ("slot_check", "robotic_cutting_sim.tlmpm_v5:TLMPM.check_slots"),
    ("label_stroke_readback", "robotic_cutting_sim.tlmpm_v4:TLMPM.cutting_state"),
    ("label_stroke_readback", "robotic_cutting_sim.tlmpm_v5:TLMPM.cutting_state"),
    ("ik_controller", "robotic_cutting_sim.robot:DualFrankaController.inverse_kinematics"),
    ("physx_step", "isaacsim.core.simulation_manager:SimulationManager.step"),
    ("grasp_contact_sample", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting._grasp_sample"),
    ("proprioception_readback", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting.proprioception"),
    ("surface_extract_usd", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting._update_surface"),
    ("camera_extrinsics", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting._update_camera_extrinsics"),
    ("camera_render_readback", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting.capture"),
    ("native_reset", "robotic_cutting_sim.isaac_sim_runner:LiveIsaacSimCutting.reset"),
    ("policy_act", "robotic_cutting_sim.policies:WaypointPolicy.__call__"),
    ("recording_finish_io", "robotic_cutting_sim.policy_env:PolicyReadyEnv.finish_recording"),
    ("dataset_export_io", "robotic_cutting_sim.dataset:export_episode"),
)
# Stages whose device work is asynchronous: synchronize at their boundaries when sync_device is set.
SYNCED = {"material_advance", "tlmpm_substep_eager", "material_reset"}
KERNEL_GROUPS = (("p2g", ("_p2g",)), ("grid_update", ("grid_update",)), ("g2p", ("_g2p", "_musl", "_deformation")),
                 ("slot_claim", ("_claim_slots",)), ("stress", ("_stress",)),
                 ("contacts", ("_rigid_contacts", "_obstacle_contacts", "fixture_contacts", "_particle_contacts")),
                 ("label_stroke", ("_segment", "_update_stroke", "_assign", "_start_cut", "_remember_cut", "_advance",
                                   "_persist", "_stroke_end", "_commit", "_pending", "_mark_nodes", "_lead_closure")))


def kernel_group(key):
    name = str(key)
    for group, prefixes in KERNEL_GROUPS:
        if any(prefix in name for prefix in prefixes):
            return group
    return "other_kernels"


def _resolve(target):
    module_name, attribute = target.split(":")
    module = importlib.import_module(module_name)
    owner_name, _, method = attribute.rpartition(".")
    owner = getattr(module, owner_name) if owner_name else module
    return owner, method or attribute


class StageProfiler:
    """Install, run the loop, ``uninstall``; then ``summary()``. One profiler at a time."""

    _installed = None

    def __init__(self, *, sync_device=None, kernels=False, stages=STAGES):
        self.sync_device, self.kernels, self.stages = sync_device, bool(kernels), tuple(stages)
        self._patches, self._stack = [], []
        self.step_rows, self._step = [], None
        self.episode = defaultdict(list)      # stage -> inclusive durations outside control steps
        self.kernel_times = defaultdict(float)  # kernel key -> seconds (kernels mode)
        self.kernel_calls = defaultdict(int)
        self.skipped = []

    # -- installation
    def install(self):
        if StageProfiler._installed is not None:
            raise RuntimeError("A StageProfiler is already installed")
        for name, target in self.stages:
            try:
                owner, attribute = _resolve(target)
                if any(patch[0] is owner and patch[1] == attribute for patch in self._patches):
                    continue
                own = attribute in getattr(owner, "__dict__", {})
                original = owner.__dict__[attribute] if own else getattr(owner, attribute)
            except (ImportError, AttributeError, KeyError) as error:
                self.skipped.append({"stage": name, "target": target, "reason": repr(error)})
                continue
            self._wrap(owner, attribute, original, name, own)
        self._wrap_recorder()
        import warp
        self._wrap_graph_launch(warp)
        if self.kernels:
            self._wrap_kernels(warp)
            from . import live_material
            original_reset = live_material.LiveCuttingMaterial.__dict__["reset"]

            @functools.wraps(original_reset)
            def reset_without_graph(material, *args, **kwargs):
                out = original_reset(material, *args, **kwargs)   # already timed as material_reset
                material.graph = None   # same kernels, launched eagerly so each can be timed
                return out
            self._patches.append((live_material.LiveCuttingMaterial, "reset", original_reset, True))
            live_material.LiveCuttingMaterial.reset = reset_without_graph
        StageProfiler._installed = self
        return self

    def uninstall(self):
        for owner, attribute, original, own in reversed(self._patches):
            if own:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)   # it was inherited: restore the lookup, not a copy
        self._patches = []
        if StageProfiler._installed is self:
            StageProfiler._installed = None

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()

    def _wrap(self, owner, attribute, original, name, own=True):
        function = original.__func__ if isinstance(original, (staticmethod, classmethod)) else original
        timed = self._timed(function, name)
        if isinstance(original, staticmethod):
            timed = staticmethod(timed)
        elif isinstance(original, classmethod):
            timed = classmethod(timed)
        self._patches.append((owner, attribute, original, own))
        setattr(owner, attribute, timed)

    def _sync(self):
        if self.sync_device is not None:
            import warp
            device = warp.get_device(self.sync_device)
            if not device.is_capturing:
                warp.synchronize_device(device)

    def _timed(self, function, name):
        profiler = self
        synced = name in SYNCED

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            if synced:
                profiler._sync()
            outer = name == "control_step" and profiler._step is None
            if outer:
                profiler._step = defaultdict(lambda: [0., 0.])
            profiler._stack.append([name, 0.])
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                if synced:
                    profiler._sync()
                elapsed = time.perf_counter()-start
                _, children = profiler._stack.pop()
                if profiler._stack:
                    profiler._stack[-1][1] += elapsed
                profiler._add(name, elapsed, elapsed-children)
                if outer:
                    profiler.step_rows.append({key: tuple(value) for key, value in profiler._step.items()})
                    profiler._step = None
        return wrapper

    def _add(self, name, inclusive, exclusive):
        if self._step is not None:
            row = self._step[name]
            row[0] += inclusive
            row[1] += exclusive
        else:
            self.episode[name].append((inclusive, exclusive))

    def _wrap_recorder(self):
        """Time the per-step EpisodeRecorder.step of each recorder this env opens (instance attribute only)."""
        from . import policy_env
        profiler = self
        original = policy_env.PolicyReadyEnv.__dict__["_open_recorder"]

        @functools.wraps(original)
        def open_recorder(env, *args, **kwargs):
            out = original(env, *args, **kwargs)
            recorder = env._recorder
            recorder.step = profiler._timed(recorder.step, "recording_io")
            return out
        self._patches.append((policy_env.PolicyReadyEnv, "_open_recorder", original, True))
        policy_env.PolicyReadyEnv._open_recorder = open_recorder

    def _wrap_graph_launch(self, warp):
        profiler = self
        capture_launch = warp.capture_launch

        @functools.wraps(capture_launch)
        def timed_capture_launch(graph, *args, **kwargs):
            start = time.perf_counter()
            out = capture_launch(graph, *args, **kwargs)
            warp.synchronize()   # the launch is asynchronous: charge its GPU execution here
            elapsed = time.perf_counter()-start
            if profiler._stack:
                profiler._stack[-1][1] += elapsed
            profiler._add("tlmpm_substeps_graph", elapsed, elapsed)
            return out
        self._patches.append((warp, "capture_launch", capture_launch, True))
        warp.capture_launch = timed_capture_launch

    def _wrap_kernels(self, warp):
        profiler = self
        launch = warp.launch

        @functools.wraps(launch)
        def timed_launch(kernel, *args, **kwargs):
            device = kwargs.get("device")
            device = warp.get_device(device) if device is not None else warp.get_device()
            if device.is_capturing:
                return launch(kernel, *args, **kwargs)
            if device.is_cuda:
                warp.synchronize_device(device)
            start = time.perf_counter()
            out = launch(kernel, *args, **kwargs)
            if device.is_cuda:
                warp.synchronize_device(device)
            elapsed = time.perf_counter()-start
            key = getattr(kernel, "key", str(kernel))
            profiler.kernel_times[key] += elapsed
            profiler.kernel_calls[key] += 1
            return out
        self._patches.append((warp, "launch", launch, True))
        warp.launch = timed_launch

    # -- results
    def summary(self):
        steps = self.step_rows
        totals = np.asarray([row.get("control_step", (0., 0.))[0] for row in steps], float)
        total = float(totals.sum())
        names = sorted({name for row in steps for name in row})
        per_step = {}
        for name in names:
            inclusive = np.asarray([row.get(name, (0., 0.))[0] for row in steps], float)
            exclusive = np.asarray([row.get(name, (0., 0.))[1] for row in steps], float)
            per_step[name] = {"mean_ms": float(inclusive.mean()*1e3), "max_ms": float(inclusive.max()*1e3),
                              "exclusive_mean_ms": float(exclusive.mean()*1e3),
                              "share_of_control_step": float(inclusive.sum()/total) if total else None,
                              "exclusive_share_of_control_step": float(exclusive.sum()/total) if total else None,
                              "steps_with_stage": int(np.count_nonzero(inclusive))}
        episode = {name: {"calls": len(rows), "mean_s": float(np.mean([r[0] for r in rows])),
                          "max_s": float(np.max([r[0] for r in rows])), "total_s": float(np.sum([r[0] for r in rows])),
                          "exclusive_total_s": float(np.sum([r[1] for r in rows]))}
                   for name, rows in self.episode.items()}
        out = {"control_steps": len(steps), "control_step_total_s": total, "per_control_step": per_step,
               "episode_level": episode, "sync_device": self.sync_device, "kernels_mode": self.kernels,
               "skipped_targets": self.skipped}
        if self.kernels:
            kernel_total = float(sum(self.kernel_times.values()))
            groups = defaultdict(float)
            for key, seconds in self.kernel_times.items():
                groups[kernel_group(key)] += seconds
            out["kernels"] = {key: {"total_s": seconds, "calls": self.kernel_calls[key],
                                    "share_of_kernel_time": seconds/kernel_total if kernel_total else None}
                              for key, seconds in sorted(self.kernel_times.items(), key=lambda kv: -kv[1])}
            out["kernel_groups"] = {group: {"total_s": seconds,
                                            "share_of_kernel_time": seconds/kernel_total if kernel_total else None,
                                            "per_control_step_ms": seconds/max(len(steps), 1)*1e3}
                                    for group, seconds in sorted(groups.items(), key=lambda kv: -kv[1])}
            out["kernel_total_s"] = kernel_total
        return out
