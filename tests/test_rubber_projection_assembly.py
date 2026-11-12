"""``projection_assembly_v2`` (P2-6 reset cost): the batch rebuilds an env's F8 block inverses after a density change
with an order-preserving vectorised accumulation; the blocks must be byte-identical to the single solver's frozen
``tlmpm_rubber_v1.projection_blocks`` (same nodes, same matrices, same inverses). Warp CPU only, no Torch."""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_scene as rs
from robotic_cutting_sim import tlmpm_rubber_v1 as T
from robotic_cutting_sim.batched.rubber_material_batch import projection_blocks_v2, projection_structure

SCENES = {"case_A_P16_t4": dict(length=0.016, width=0.006, thickness=0.004, n_t=4, dx=0.001, case="A"),
          "case_A_L125_t4": dict(length=0.125, width=0.016, thickness=0.004, n_t=4, dx=0.001, case="A"),
          "case_A_L150_t6": dict(length=0.150, width=0.016, thickness=0.006, n_t=6, dx=0.001, case="A"),
          "case_C_x": dict(length=0.012, width=0.004, thickness=0.002, n_t=2, case="C", end_mask="x"),
          "case_C_xyz": dict(length=0.010, width=0.004, thickness=0.003, n_t=3, case="C", end_mask="xyz")}


def _same(a, b):
    a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


@pytest.mark.parametrize("name", sorted(SCENES))
def test_blocks_are_bytewise_the_frozen_assembly(name):
    scene = rs.compile_strip(rs.RubberStripSpec(**SCENES[name]))
    cache = {}
    for rho in (1000.0, 1150.0, 1287.3456):
        m = np.float32(rho*scene.spacing**3)
        ref = T.projection_blocks(scene, m)
        new = projection_blocks_v2(scene, m, cache, 0)
        assert len(ref) == len(new) > 0
        for a, b in zip(ref, new):
            assert (a["group"], a["comp"]) == (b["group"], b["comp"])
            assert _same(a["nodes"], b["nodes"]) and _same(a["inv"], b["inv"]) and a["cond"] == b["cond"]
    assert list(cache) == [0]
