"""RUBBER-VISC-09 with the DEFAULT solver options (review R4, P2_PREREQ_P2_3_REVIEW.md).

``tests/test_rubber_visco_solver.py`` runs VISC-09 with the full option set (F8 + ``kick_average_v1`` + F9 + F10).
``TLMPMRubberVisco.step()`` duplicates the frozen launch sequence, including the default branches (lumped clamp,
float32 state, particle-velocity traction work, end-velocity constraint work); a ``LiveRubberMaterial`` with Prony terms
and the default options builds exactly that path. Here, with the default options, the stable-step guard on, a table
plane (strips), a moving grip with rotation and traction (case A / C) or an affine group (block), for 5 control steps:

- M = 0 (``TLMPMRubberVisco`` with no Prony term) and
- M = 2 with every g = 0 (``visco_energy`` default ``off_v1``)

are byte-identical to the frozen ``tlmpm_rubber_v1.TLMPMRubber`` in every device array of the frozen solver, the
wrench and the diagnostics summary. Test only; no protocol, threshold or solver change. Warp CPU device.
"""
import numpy as np
import pytest

from test_rubber_visco_solver import ZERO, _inputs, _live, _same_bytes

from robotic_cutting_sim.tlmpm_rubber_v1 import TLMPMRubber
from robotic_cutting_sim.tlmpm_rubber_visco_v1 import TLMPMRubberVisco

DEFAULT_OPTIONS = {}


def _default_live(case, **kw):
    if case != "block":
        kw["table_plane"] = (0.0, 0.0, 1.0, 0.0)
    return _live(case, options=DEFAULT_OPTIONS, **kw)


@pytest.mark.parametrize("variant", ["M0", "M2_zero_g"])
@pytest.mark.parametrize("case", ["A", "C", "block"])
def test_visc09_default_options_byte_identical_to_the_frozen_solver(case, variant):
    ref = _default_live(case)
    assert type(ref.solver) is TLMPMRubber                       # the frozen class itself (defaults)
    if variant == "M0":                                          # visco_energy != off_v1 selects the visco class
        vis = _default_live(case, prony=(), energy="branch_work_v1")
        assert vis.solver.M == 0 and not vis.solver.visco_energy_on
    else:
        vis = _default_live(case, prony=ZERO)
        assert vis.solver.M == 2 and vis.solver.visco_energy == "off_v1"
    assert type(vis.solver) is TLMPMRubberVisco
    s = vis.solver
    assert (s.dirichlet, s.traction_work, s.state_precision, s.constraint_work) == (
        "lumped_v1", "particle_velocity_v1", "float32_v1", "end_velocity_v1")
    assert s.cfl_guard_limit == ref.solver.cfl_guard_limit == 1.0          # stable-step guard on
    for k in range(5):
        motions, traction = _inputs(ref, k)
        ref.advance(motions, traction=traction)
        vis.advance(motions, traction=traction)
        sa, sb = ref.solver.device_arrays(), vis.solver.device_arrays()
        assert set(sa) <= set(sb)
        for name, arr in sa.items():
            assert _same_bytes(arr.numpy(), sb[name].numpy()), (case, variant, k, name)
        assert _same_bytes(ref.last_wrench, vis.last_wrench), (case, variant, k)
        assert _same_bytes(ref.solver.diagnostics.summary.numpy(), vis.solver.diagnostics.summary.numpy())
        if variant == "M2_zero_g":
            assert not np.any(s.visco_q.numpy())
    assert ref.step_count == vis.step_count == 5
    assert float(np.max(np.abs(ref.solver.x.numpy()-ref.scene.X))) > 0     # the scene moved
