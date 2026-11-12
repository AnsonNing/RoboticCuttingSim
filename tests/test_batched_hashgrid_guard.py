"""Process-level guard against a second live device HashGrid (Warp 1.17 static-descriptor hazard, P1-3a).

On CUDA the guard counts live CUDA ``wp.HashGrid`` objects. These CPU tests enable it on the Warp CPU device by
monkeypatching the two predicates to count only grids owned by batched materials, then check: a second batch
in the same process is refused with ``UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS``; ``rebuild()`` of the same material
is allowed (its own grid is exempt); ``release()`` frees the grid so a new batch may be built; the opt-out keeps
explicit diagnostic probes possible. No kernel source is involved."""
import weakref

import pytest
import warp as wp

from batched_fixtures import batch_material
from robotic_cutting_sim.batched import material_batch as MB
from robotic_cutting_sim.batched.errors import CODES, UnsupportedBatchConfiguration

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]
CONFIG = [(512, None, 42)]


@pytest.fixture
def cpu_guard(monkeypatch):
    owned = []
    original = MB.BatchedTLMPMv7_2.__init__

    def tracking_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        owned.append(weakref.ref(self.hash))      # weak: the test must not keep grids alive
    monkeypatch.setattr(MB.BatchedTLMPMv7_2, "__init__", tracking_init)
    monkeypatch.setattr(MB, "_guard_applies", lambda device: True)
    monkeypatch.setattr(MB, "_is_device_grid", lambda grid: any(grid is r() for r in owned))
    return owned


def test_code_is_declared():
    assert "UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS" in CODES


def test_second_batch_in_process_is_refused_until_the_first_is_released(cpu_guard):
    first = batch_material(CONFIG)
    assert first.device_hashgrid_guard == MB.DEVICE_HASHGRID_GUARD
    assert first.receipt()["device_hashgrid_guard"] == MB.DEVICE_HASHGRID_GUARD
    with pytest.raises(UnsupportedBatchConfiguration, match="^UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS"):
        batch_material(CONFIG)
    first.release()
    assert first.solver.hash is None and first.graph is None
    second = batch_material(CONFIG)           # allowed once the first grid is gone
    assert isinstance(second.solver.hash, wp.HashGrid)


def test_rebuild_of_the_same_material_is_allowed(cpu_guard):
    from batched_fixtures import start_pose, trim_material
    from robotic_cutting_sim.batched import layout
    material = batch_material(CONFIG)
    old_grid = material.solver.hash
    material.rebuild([layout.EnvSpec(trim_material(512), start_pose(512), 42)])
    assert material.solver.hash is not old_grid
    del old_grid
    assert not MB._REBUILD_EXEMPT_GRIDS


def test_opt_out_for_explicit_diagnostics(cpu_guard):
    first = batch_material(CONFIG)
    second = batch_material(CONFIG, allow_multiple_device_hashgrids=True)
    assert second.device_hashgrid_guard == "disabled" and first.solver.hash is not None


def test_guard_is_inactive_for_cpu_grids_without_the_test_predicates():
    first = batch_material(CONFIG)
    second = batch_material(CONFIG)           # host grids have no static descriptor: not counted
    assert first.device_hashgrid_guard == "not_applicable_host_grid" == second.device_hashgrid_guard


@pytest.mark.parametrize("options", [{}, {"speed_guard": "flag"}])
def test_alias_free_v1_owns_one_grid_released_and_rebuilt_under_the_guard(cpu_guard, options):
    """P1-3a ``alias_free_v1`` replaces the solver's frozen-dims grid by its alias-free grid: the guard sees one
    grid per material, ``release()`` frees it (also the contacts' reference) and ``rebuild()`` stays allowed."""
    from batched_fixtures import start_pose, trim_material
    from robotic_cutting_sim.batched import layout
    from robotic_cutting_sim.batched import particle_contacts as PC
    first = batch_material(CONFIG, particle_contact_mode=PC.ALIAS_FREE_MODE, particle_contact_options=options)
    assert first.solver.particle_contact.grid is first.solver.hash
    with pytest.raises(UnsupportedBatchConfiguration, match="^UNSUPPORTED_MULTIPLE_DEVICE_HASHGRIDS"):
        batch_material(CONFIG)
    first.rebuild([layout.EnvSpec(trim_material(512), start_pose(512), 42)])
    assert first.solver.particle_contact.grid is first.solver.hash and not MB._REBUILD_EXEMPT_GRIDS
    first.release()
    assert first.solver.hash is None and first.solver.particle_contact.grid is None
    second = batch_material(CONFIG, particle_contact_mode=PC.ALIAS_FREE_MODE, particle_contact_options=options)
    assert isinstance(second.solver.hash, wp.HashGrid)


# ----------------------------------------------------------------------------- review fix F1 (step-time refresh)

def _foreign_grid(owned, points=64):
    """A grid built on the host AFTER the batch (stands in for a frozen single LiveCuttingMaterial on CUDA)."""
    import numpy as np
    grid = wp.HashGrid(8, 8, 8, device="cpu")
    owned.append(weakref.ref(grid))           # counts as a device grid for the monkeypatched predicates
    x = wp.array(np.random.default_rng(0).uniform(1., 2., (points, 3)).astype(np.float32), dtype=wp.vec3,
                 device="cpu")
    grid.build(x, .01)
    return grid


def _last_built():
    ref = MB._LAST_HOST_BUILD["grid"]
    return None if ref is None else ref()


def _spy(material, monkeypatch, graph):
    """Record the launch path of ``advance`` without running the 200 substeps: ``refresh`` (with the grid the host
    built last right after it), ``launch`` (the graph replay, with the last host-built grid at replay time = the
    descriptor Warp 1.17 would copy) or ``step`` (eager substeps)."""
    events = []
    s = material.solver
    refresh = material.refresh_device_hashgrid

    def spy_refresh():
        refresh()
        events.append(("refresh", _last_built() is material.own_hashgrid))
    monkeypatch.setattr(material, "refresh_device_hashgrid", spy_refresh)
    monkeypatch.setattr(s, "step", lambda advance=True: events.append(("step", _last_built() is material.own_hashgrid)))
    if graph:
        material.graph = object()             # stands in for the captured CUDA graph (CPU has no capture)
        monkeypatch.setattr(MB.wp, "capture_launch",
                            lambda g: events.append(("launch", _last_built() is material.own_hashgrid)))
    return events


@pytest.mark.parametrize("mode", ["frozen_v1", "alias_free_v1"])
@pytest.mark.parametrize("graph", [True, False], ids=["graph", "eager"])
def test_own_grid_is_rebuilt_on_the_host_before_every_launch(cpu_guard, monkeypatch, mode, graph):
    """Every launch path (graph replay and eager substeps), both particle-contact modes: the batch's own grid is the
    last one built on the host before the launch, also after a foreign grid was built later than the batch."""
    material = batch_material(CONFIG, strict=False, particle_contact_mode=mode)
    events = _spy(material, monkeypatch, graph)
    S = material.solver.substeps
    material.advance(interpolation="preset")
    launch = [("launch", True)] if graph else [("step", True)]*S
    assert events == [("refresh", True)] + launch
    assert material.descriptor_refreshes == 1
    events.clear()
    foreign = _foreign_grid(cpu_guard)
    assert _last_built() is foreign               # without the refresh the replay would copy this descriptor
    material.advance(interpolation="preset")
    assert events == [("refresh", True)] + launch
    receipt = material.receipt()["device_hashgrid"]
    assert receipt["descriptor_refresh"] == MB.DEVICE_HASHGRID_REFRESH and receipt["descriptor_refreshes"] == 2
    del foreign


def test_foreign_grid_after_build_sets_sticky_flag_and_receipt(cpu_guard):
    material = batch_material(CONFIG, strict=False)
    material.advance(interpolation="preset")
    assert material.invalid_reason.numpy().tolist() == [0]
    assert material.receipt()["device_hashgrid"]["status"] == "OK"
    foreign = _foreign_grid(cpu_guard)
    material.advance(interpolation="preset")
    reasons = material.invalid_reason.numpy()
    assert MB.decode_reason(reasons[0]) == ["FOREIGN_DEVICE_HASHGRID"]
    receipt = material.receipt()["device_hashgrid"]
    assert receipt["status"] == "FOREIGN_DEVICE_HASHGRID_OBSERVED"
    assert receipt["foreign"]["first_control_step"] == 1 and receipt["foreign"]["max_foreign_live_grids"] == 1
    assert receipt["foreign"]["foreign_last_host_build_steps"] == 1 and receipt["foreign"]["invalid_bit_set"]
    material.strict = True                     # strict mode raises with the reason name
    with pytest.raises(MB.InvalidBatchEnvs, match="FOREIGN_DEVICE_HASHGRID"):
        material.check_valid()
    # the bit is sticky until reset; the per-batch record is sticky for the batch's life
    del foreign
    material.reset()
    material.strict = False
    material.advance(interpolation="preset")
    assert material.invalid_reason.numpy().tolist() == [0]
    receipt = material.receipt()["device_hashgrid"]
    assert receipt["status"] == "FOREIGN_DEVICE_HASHGRID_OBSERVED" and receipt["foreign"]["control_steps_flagged"] == 1


def test_step_time_census_needs_no_gc_scan_and_no_readback(cpu_guard, monkeypatch):
    import gc
    material = batch_material(CONFIG, strict=False)
    foreign = _foreign_grid(cpu_guard)
    reads = []
    original = wp.array.numpy
    monkeypatch.setattr(gc, "collect", lambda *a: (_ for _ in ()).throw(AssertionError("gc scan at step time")))
    monkeypatch.setattr(wp.array, "numpy", lambda self: reads.append(1) or original(self))
    material.advance(interpolation="preset")
    monkeypatch.undo()
    assert not reads
    assert material.foreign_hashgrid is not None
    del foreign


def test_opt_out_records_but_does_not_flag_envs(cpu_guard):
    material = batch_material(CONFIG, strict=False, allow_multiple_device_hashgrids=True)
    other = batch_material(CONFIG, strict=False, allow_multiple_device_hashgrids=True)
    material.advance(interpolation="preset")
    assert material.invalid_reason.numpy().tolist() == [0]
    record = material.receipt()["device_hashgrid"]["foreign"]
    assert record["max_foreign_live_grids"] == 1 and record["invalid_bit_set"] is False
    del other


def test_census_record_is_sticky_across_rebuild(cpu_guard):
    from batched_fixtures import start_pose, trim_material
    from robotic_cutting_sim.batched import layout
    material = batch_material(CONFIG, strict=False)
    foreign = _foreign_grid(cpu_guard)
    material.advance(interpolation="preset")
    del foreign
    material.rebuild([layout.EnvSpec(trim_material(512), start_pose(512), 42)])
    assert material.receipt()["device_hashgrid"]["status"] == "FOREIGN_DEVICE_HASHGRID_OBSERVED"


def test_no_census_on_host_grids_without_the_test_predicates():
    material = batch_material(CONFIG, strict=False)
    grid = wp.HashGrid(8, 8, 8, device="cpu")           # host grids have no static descriptor
    material.advance(interpolation="preset")
    receipt = material.receipt()["device_hashgrid"]
    assert receipt["census"] == "not_applicable_host_grid" and receipt["status"] == "OK"
    assert receipt["descriptor_refreshes"] == 1         # the refresh itself runs on every device
    del grid
