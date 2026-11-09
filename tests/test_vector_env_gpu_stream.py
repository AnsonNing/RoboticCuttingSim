"""ADR 0021 §7 single-stream policy of ``open_vector_env`` on CUDA (GPU test; skipped -> NOT_RUN without CUDA).

Run on the shared machine only under the RCS GPU lock, e.g.::

    flock /mnt/data/ningan/lab_records/rcs_rl_runner/gpu1.lock env CUDA_VISIBLE_DEVICES=1 \\
        PYTHONPATH=/home/ningan/rcs_rl_data/pytest_overlay_py312:src .deps/isaacsim-venv/bin/python -m pytest \\
        -p no:cacheprovider tests/test_vector_env_gpu_stream.py

Coarse, shortened task (2.5 mm particles, 3-step episodes): reduced fidelity, API check only."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available (NOT_RUN)", allow_module_level=True)

from batched_fixtures import start_pose  # noqa: E402
from robotic_cutting_sim.rl import TaskFamily, open_vector_env  # noqa: E402

pytestmark = [pytest.mark.filterwarnings("ignore::DeprecationWarning")]


def _family(seeds=(512, 15)):
    return TaskFamily(scene_seeds=seeds, particle_spacing_m=.0025, max_episode_steps=3,
                      initial_edge_poses=tuple(tuple(start_pose(s)) for s in seeds))


def _push(n):
    a = torch.zeros((n, 8), dtype=torch.float32, device="cuda")
    a[:, 0], a[:, 6], a[:, 7] = 1., 1., -.5
    return a


def test_warp_work_runs_on_the_torch_current_stream():
    with open_vector_env(2, task=_family(), device="cuda", debug_sync=True) as env:
        env.reset()
        default = torch.cuda.current_stream()
        env.step(_push(2))
        assert env.last_warp_stream_handle == default.cuda_stream
        side = torch.cuda.Stream()
        side.wait_stream(default)
        with torch.cuda.stream(side):
            actions = _push(2)                 # produced on the side stream, consumed in order on it
            obs, reward, term, trunc, info = env.step(actions)
            assert env.last_warp_stream_handle == side.cuda_stream
            total = obs["policy"].sum() + reward.sum()
        side.synchronize()
        assert bool(torch.isfinite(total)) and info["step_count"].tolist() == [2, 2]
        assert env.receipt()["debug_sync"] is True


def test_side_stream_without_debug_sync_matches_codes_monotone_and_finite():
    with open_vector_env(2, task=_family(), device="cuda") as env:
        env.reset()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        labelled = []
        with torch.cuda.stream(side):
            for _ in range(3):
                obs, reward, term, trunc, info = env.step(_push(2))
                labelled.append((env._codes != 0).sum())
        side.synchronize()
        counts = [int(c) for c in labelled]
        assert counts == sorted(counts) and counts[-1] > 0
        assert bool(trunc.all()) and not bool(term.any())
        assert np.isfinite(env.material.solver.x.numpy()).all()
