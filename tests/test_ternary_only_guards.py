"""Ternary-only consumers reject packed tlmpm_v5 codes with max_cuts >= 2 (WP-11 review finding 11).

Ternary inputs, and packed K = 1 (equal to ternary), keep their behaviour.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from robotic_cutting_sim.cut_topology import mechanical_components
from robotic_cutting_sim.live_acceptance import cut_history_metrics, graph_audit
from robotic_cutting_sim.reward import CuttingReward, require_ternary_codes


def test_reward_guard():
    for material in (SimpleNamespace(code_encoding="packed2", max_cuts=2), SimpleNamespace(code_encoding="packed2", max_cuts=5)):
        with pytest.raises(ValueError, match="ternary cut codes"):
            require_ternary_codes(material)
    for material in (SimpleNamespace(code_encoding="packed2", max_cuts=1), SimpleNamespace(code_encoding="ternary", max_cuts=2),
                     SimpleNamespace()):
        require_ternary_codes(material)   # no error


def test_reward_start_rejects_packed_multi_digit_material_before_reading_codes():
    task = SimpleNamespace(grid_dx_m=.004, particle_spacing_m=.001)
    env = SimpleNamespace(backend=SimpleNamespace(material=SimpleNamespace(code_encoding="packed2", max_cuts=3)))
    with pytest.raises(ValueError, match="max_cuts 3 >= 2"):
        CuttingReward(task)._start(env)


def _stencil():
    # Three particles on one node; particle 2 also on its own node.
    nodes = np.array([[0, 1], [0, 2], [0, 3]], np.int64)
    weights = np.full((3, 2), .5)
    return {"reference_stencil_nodes": nodes, "reference_stencil_weights": weights,
            "reference_stencil_gradients": np.ones((3, 2, 3))}


def test_graph_audit_ternary_unchanged_and_packed_not_applicable():
    physics = {**_stencil(), "codes": np.array([[0, 0, 0], [1, 2, 0]], np.int64)}
    graph, labels = graph_audit(physics, 1)
    count, expected = mechanical_components(physics["reference_stencil_nodes"], physics["reference_stencil_weights"],
                                            physics["codes"][-1], 1, gradients=physics["reference_stencil_gradients"])
    assert graph["status"] == "PASS" and graph["final_components"] == count and np.array_equal(labels, expected)
    for cuts, recorded in ((2, None), (1, 5)):
        packed = {**physics, "code_encoding": np.array("packed2"), **({} if recorded is None else {"max_cuts": np.array(recorded)})}
        graph, labels = graph_audit(packed, cuts)
        assert graph["status"] == "NOT_APPLICABLE" and labels is None and graph["max_cuts"] == max(cuts, recorded or 0)
    # One packed digit equals its ternary code: still audited.
    graph, labels = graph_audit({**physics, "code_encoding": np.array("packed2"), "max_cuts": np.array(None, dtype=object)}, 1)
    assert graph["status"] == "PASS" and graph["final_components"] == count


def test_cut_history_metrics_guard():
    codes = np.array([[0, 0], [1, 3]], np.int64)
    assert cut_history_metrics(codes, 2) == cut_history_metrics(codes, 2, encoding="ternary")
    with pytest.raises(ValueError, match="ternary codes"):
        cut_history_metrics(codes, 2, encoding="packed2")
    assert cut_history_metrics(np.array([[0], [1]]), 1, encoding="packed2")["nonzero_digit_rewrites"] == 0
