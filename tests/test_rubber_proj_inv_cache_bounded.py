"""F1 (P6.7 review): the F8 block-inverse cache of ``BatchedRubberMaterial`` is a bounded LRU
(``max(PROJ_INV_CACHE_MIN, 2 N)`` entries). Before the fix every reset that resampled the density added one entry
and nothing was evicted (+2.2 MB host RSS per reset on the declared task). The cache only memoises a deterministic
function of (scene, float32 mass): after any number of resets the staged inverses stay byte-identical to the frozen
single-solver ``tlmpm_rubber_v1.projection_blocks``. Warp CPU only, no Torch."""
import numpy as np

from robotic_cutting_sim import tlmpm_rubber_v1 as T
from robotic_cutting_sim.batched import rubber_material_batch as rmb
from robotic_cutting_sim.rl.rubber_family import RubberStripTaskFamily


def _batch(slots):
    fam = RubberStripTaskFamily(slots, variant="cpu_parity_v1")
    lives = [fam.build_life(s) for s in fam.env_slots]
    return rmb.BatchedRubberMaterial(lives, device="cpu", parity_mode=True, ranges=fam.batch_ranges())


def _frozen_inv(batch, e, mass):
    blocks = T.projection_blocks(batch._proj_scenes[e], np.float32(mass))
    return np.concatenate([b["inv"].reshape(-1) for b in blocks])


def test_cache_is_bounded_and_results_stay_bytewise():
    batch = _batch(["P16_t4", "P12_t6"])
    assert batch.solver.projection
    cap = batch.proj_inv_cache_max
    assert cap == max(rmb.PROJ_INV_CACHE_MIN, 2*batch.num_envs)
    io = batch.packed.env["proj_inv_env_offset"]
    sizes = []
    for k in range(3*cap+5):
        ids = [k % batch.num_envs]
        batch.reset(ids, seeds=[9000+k], rng_ranges=True)
        sizes.append(len(batch._proj_inv_cache))
        e = ids[0]
        mass = batch.stage_mass.numpy()[e]
        staged = batch.solver.proj_inv.numpy()[io[e]:io[e+1]]
        ref = (batch._proj_inv_initial[io[e]:io[e+1]] if mass == batch.initial_mass.numpy()[e]
               else _frozen_inv(batch, e, mass))
        assert staged.dtype == ref.dtype and staged.tobytes() == ref.tobytes(), k
    assert max(sizes) == cap and sizes[-1] == cap       # grew to the cap, never beyond (densities resampled)


def test_eviction_then_recompute_is_bytewise():
    batch = _batch(["P16_t4"])
    m0 = np.float32(1100.0*batch.solver.spacing**3)
    first = batch._projection_inverse(0, m0).copy()
    for k in range(batch.proj_inv_cache_max+2):         # push m0 out of the LRU
        batch._projection_inverse(0, np.float32((1001.0+k)*batch.solver.spacing**3))
    assert (0, m0.tobytes()) not in batch._proj_inv_cache
    again = batch._projection_inverse(0, m0)
    assert again.tobytes() == first.tobytes() == _frozen_inv(batch, 0, m0).tobytes()
    assert len(batch._proj_inv_cache) == batch.proj_inv_cache_max
