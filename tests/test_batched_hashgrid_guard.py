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
