import copy
import numpy as np
import pytest
from robotic_cutting_sim.multi_cut_evaluation import MultiCutGoal, evaluate_multi_cut


def fixture(codes):
    return dict(codes=np.asarray(codes, np.int32), mass_kg=np.ones(3),
                reference_positions=np.array([[-.02, 0., 0.], [0., 0., 0.], [.02, 0., 0.]]),
                reference_indices=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]),
                reference_stencil_nodes=np.zeros((3, 1), int),
                reference_stencil_weights=np.ones((3, 1)), reference_stencil_gradients=np.zeros((3, 1, 3)))


GOAL = MultiCutGoal((((-.01, 0., 0.), (1., 0., 0.)), ((.01, 0., 0.), (1., 0., 0.))))


def test_two_actual_digits_and_read_only_source():
    data = fixture([4, 5, 8])
    before = copy.deepcopy(data)
    result = evaluate_multi_cut(data, GOAL)
    assert result['status'] == 'PASS_MULTI_CUT_OUTCOME'
    assert result['metrics']['mechanical_components'] == 3
    for key in data:
        assert np.array_equal(data[key], before[key])


@pytest.mark.parametrize('codes', [[1, 1, 1], [4, 5, 5], [0, 0, 0]])
def test_one_sided_labels_or_one_cut_cannot_pass(codes):
    assert evaluate_multi_cut(fixture(codes), GOAL)['status'] == 'FAIL_MULTI_CUT_OUTCOME'


def test_missing_stress_gradient_and_invalid_digit_cannot_pass():
    data = fixture([4, 5, 8]); data['reference_stencil_gradients'][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate_multi_cut(data, GOAL)
    data = fixture([4, 5, 9])
    with pytest.raises(ValueError):
        evaluate_multi_cut(data, GOAL)
