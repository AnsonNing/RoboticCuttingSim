import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from robotic_cutting_sim.tasks import make_task


ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("trim_diagnostic",ROOT/"scripts/run_trim_sweep_diagnostic.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
protocol=json.loads((ROOT/"configs/trim_sweep_diagnostic_v1.json").read_text())


def test_second_pass_preserves_the_entire_original_first_episode():
    task=make_task("TRIM-01")
    for time_s in np.arange(401)*task.control_dt_s:
        np.testing.assert_allclose(module.reference_pose(task,protocol["second_pass"],time_s),
                                   task.reference_edge_pose_world(time_s),rtol=0,atol=1e-12)
    assert protocol["second_pass"]["active_cut_intervals_s"][0]==list(task.active_cut_interval_s)


def test_faster_sweep_keeps_target_height_and_declares_its_new_speed():
    task=make_task("TRIM-01")
    first=module.reference_pose(task,protocol["fast_sweep"],1.6)
    last=module.reference_pose(task,protocol["fast_sweep"],5.6)
    np.testing.assert_allclose((last[0]-first[0])/4,.04375,atol=1e-12)
    np.testing.assert_allclose([first[2],last[2]],[.0202,.0202],atol=1e-12)
    assert protocol["candidate_checks"]["minimum_detached_reference_above_plane_mass_fraction"]==.9


def test_coarser_carryover_preserves_geometry_and_original_entire_path():
    control=json.loads((ROOT/"configs/trim_particle_spacing_control_v1.json").read_text())
    task=make_task("TRIM-01")
    coarse=replace(task,particle_spacing_m=.002)
    assert coarse.geometry==task.geometry and coarse.material==task.material
    assert coarse.grid_dx_m==task.grid_dx_m==.004
    for time_s in np.arange(401)*task.control_dt_s:
        np.testing.assert_allclose(module.reference_pose(coarse,control["coarse_particles"],time_s),
                                   task.reference_edge_pose_world(time_s),rtol=0,atol=1e-12)
