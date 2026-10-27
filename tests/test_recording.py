import json

import numpy as np
import pytest

from robotic_cutting_sim.recording import write_json


def test_numeric_audit_receipt_preserves_boolean_types_and_rejects_nan(tmp_path):
    output = tmp_path/"receipt.json"
    write_json(output, {"pass": np.bool_(True), "n": np.int64(2),
                        "error": np.float32(.5), "values": np.array([1, 2])})
    assert json.loads(output.read_text()) == {"pass": True, "n": 2, "error": .5, "values": [1, 2]}
    with pytest.raises(ValueError, match="Out of range"):
        write_json(output, {"error": np.float64(np.nan)})
