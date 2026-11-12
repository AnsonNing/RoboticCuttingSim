"""P1-2b ``rl.termination``: bitfield table and infrastructure-vs-task classification (no Torch needed)."""
import signal

from robotic_cutting_sim.rl import termination as T


def test_bits_are_distinct_and_masks_consistent():
    bits = list(T.TERMINATION_REASONS)
    assert len(set(bits)) == len(bits) and all(b & (b - 1) == 0 for b in bits)
    assert (T.TERM_RIGID_INVALID, T.TERM_MATERIAL_INVALID, T.TERM_GRASP_LOST, T.TERM_IK_STREAK) == (1, 2, 4, 8)
    assert not T.TERMINAL_BITS & T.TERM_TRUNCATED and T.TERMINAL_BITS & T.TERM_INFRASTRUCTURE
    assert not T.TASK_FAILURE_BITS & T.TERM_INFRASTRUCTURE
    assert T.decode_termination(T.TERM_GRASP_LOST | T.TERM_TRUNCATED) == ["GRASP_LOST", "TRUNCATED_HORIZON"]
    table = T.termination_table()
    assert table["bits"]["32"] == "MATERIAL_INPUT_DOMAIN_GUARD" and table["infrastructure_bit"] == 64


def test_exception_classification():
    assert T.classify_exception(RuntimeError("CUDA error: an illegal memory access was encountered")) == (
        "infrastructure", "CUDA_ERROR")
    assert T.classify_exception(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) == ("infrastructure", "OOM")
    assert T.classify_exception(MemoryError()) == ("infrastructure", "OOM")
    assert T.classify_exception(FloatingPointError("Invalid rigid worlds [1]: {1: ['RIGID_NONFINITE_STATE']}")) == (
        "task", "RIGID_INVALID")
    assert T.classify_exception(FloatingPointError("Invalid rigid worlds [0]: {0: ['CONTACT_POOL_OVERFLOW']}"))[0] == \
        "infrastructure"
    assert T.classify_exception(ValueError("actions must be [N, 8]"))[0] == "caller"
    error = T.VectorEnvInfrastructureError("OOM", "x")
    assert T.classify_exception(error) == ("infrastructure", "OOM") and str(error).startswith("INFRASTRUCTURE_ERROR[OOM]")


def test_process_exit_classification():
    assert T.classify_process_exit(0)["category"] == "ok"
    crash = T.classify_process_exit(-signal.SIGSEGV)
    assert crash == dict(category="infrastructure", kind="CRASH_SIGSEGV", detail="terminated by SIGSEGV")
    assert T.classify_process_exit(139)["kind"] == "CRASH_SIGSEGV"
    assert T.classify_process_exit(1, "Fatal Python error: Segmentation fault\n")["category"] == "infrastructure"
    assert T.classify_process_exit(1, "robotic_cutting_sim...InvalidBatchEnvs: envs [1]")["category"] == "task"
    assert T.classify_process_exit(1, "INFRASTRUCTURE_ERROR[CUDA_ERROR]: boom")["kind"] == "CUDA_ERROR"
    assert T.classify_process_exit(1, "torch.OutOfMemoryError: CUDA out of memory")["kind"] == "OOM"
    assert T.classify_process_exit(2, "usage: error")["category"] == "unknown"


# ----------------------------------------------------------------------------- material invalid_reason mapping (P1 consolidation)

def test_every_material_invalid_bit_maps_to_a_termination_bit():
    """Iterates ``material_batch.INVALID_REASONS``: a future material bit cannot stay unmapped or undecoded."""
    import numpy as np
    from robotic_cutting_sim.batched import material_batch as MB
    from robotic_cutting_sim.rl import supervision as S
    assert T.MATERIAL_INVALID_REASONS == MB.INVALID_REASONS          # Warp-free copy is complete and current
    assert S.INVALID_REASONS == MB.INVALID_REASONS                   # the supervisor decodes every bit as well
    table = T.termination_table()
    assert set(table["bits"]) == {str(b) for b in T.TERMINATION_REASONS}
    assert set(table["material_invalid"]["bits"]) == {str(b) for b in MB.INVALID_REASONS}
    for bit, name in MB.INVALID_REASONS.items():
        assert name in T.MATERIAL_REASON_TERMINATION, f"material bit {bit} {name} has no termination mapping"
        term = T.material_termination_bits(bit)
        assert term & T.TERM_MATERIAL_INVALID, name                  # every material bit -> MATERIAL_INVALID (2)
        expected = T.TERM_MATERIAL_INVALID | (T.TERM_INFRASTRUCTURE if bit == MB.INVALID_FOREIGN_DEVICE_HASHGRID else 0)
        assert term == expected, (name, term)
        assert table["material_invalid"]["bits"][str(bit)]["termination"] == T.decode_termination(expected)
        assert T.decode_material_invalid(bit) == [name] == MB.decode_reason(bit)
        assert int(T.material_termination_bits(np.array([bit]))[0]) == term
    # combinations, array path == int path, and no bit outside the table maps to anything
    values = np.arange(0, 2 * max(MB.INVALID_REASONS), dtype=np.int64)
    known = sum(MB.INVALID_REASONS)
    as_array = T.material_termination_bits(values)
    assert as_array.dtype == np.int32
    for v in values:
        assert int(as_array[v]) == T.material_termination_bits(int(v))
        assert (int(as_array[v]) != 0) == bool(v & known)
    assert T.MATERIAL_INFRASTRUCTURE_MASK == MB.INVALID_FOREIGN_DEVICE_HASHGRID


def test_input_out_of_domain_is_material_invalid_and_decodable():
    from robotic_cutting_sim.batched import material_batch as MB
    term = T.material_termination_bits(MB.INVALID_INPUT_DOMAIN)
    assert term == T.TERM_MATERIAL_INVALID and not term & T.TERM_INPUT_DOMAIN     # not the RL-side guard bit 32
    detail = T.decode_termination_detail(term, MB.INVALID_INPUT_DOMAIN)
    assert detail == {"termination": ["MATERIAL_INVALID"], "material_invalid": ["INPUT_OUT_OF_DOMAIN"],
                      "infrastructure": False}
    other = T.decode_termination_detail(T.material_termination_bits(MB.INVALID_NONFINITE_STATE), MB.INVALID_NONFINITE_STATE)
    assert other["termination"] == detail["termination"] and other["material_invalid"] == ["NONFINITE_STATE"]
    foreign = T.decode_termination_detail(T.material_termination_bits(MB.INVALID_FOREIGN_DEVICE_HASHGRID),
                                          MB.INVALID_FOREIGN_DEVICE_HASHGRID)
    assert foreign == {"termination": ["MATERIAL_INVALID", "INFRASTRUCTURE_ERROR"],
                       "material_invalid": ["FOREIGN_DEVICE_HASHGRID"], "infrastructure": True}


def test_foreign_hashgrid_strict_raise_is_classified_infrastructure():
    from robotic_cutting_sim.batched.material_batch import InvalidBatchEnvs
    from robotic_cutting_sim.rl import supervision as S
    foreign = InvalidBatchEnvs([0, 1], {0: ["FOREIGN_DEVICE_HASHGRID"], 1: ["FOREIGN_DEVICE_HASHGRID"]})
    assert T.classify_exception(foreign) == ("infrastructure", "FOREIGN_DEVICE_HASHGRID")
    assert T.classify_exception(InvalidBatchEnvs([1], {1: ["INPUT_OUT_OF_DOMAIN"]})) == ("task", "MATERIAL_INVALID")
    text = f"robotic_cutting_sim.batched.material_batch.InvalidBatchEnvs: {foreign}"
    assert T.classify_process_exit(1, text)["kind"] == "FOREIGN_DEVICE_HASHGRID"
    assert T.classify_process_exit(1, "InvalidBatchEnvs: Invalid batched envs [1]: {1: ['NONFINITE_STATE']}")[
        "category"] == "task"
    assert S.VECTOR_ENV_INFRA_KINDS["FOREIGN_DEVICE_HASHGRID"] in S.INFRA_CLASSES
    status = S.step_status({"invalid_reason": [0, 256, 128]}, num_envs=3)      # legacy info path
    # review finding 10: the derived INFRASTRUCTURE bit of FOREIGN_DEVICE_HASHGRID is an infra flag here too
    assert status.termination_reason.tolist() == [0, 66, 2] and status.infra.tolist() == [0, 1, 0]
    status = S.step_status({"invalid_reason": [0, 256, 128], "termination_reason": [0, 66, 2]}, num_envs=3)
    assert status.infra.tolist() == [0, 1, 0] and status.terminated.tolist() == [False, True, True]


# ----------------------------------------------------------------------------- review finding 10

def test_contact_pool_overflow_exit_is_infrastructure_before_the_task_signature():
    pool = "FloatingPointError: Invalid rigid worlds [0, 2]: {0: ['CONTACT_POOL_OVERFLOW'], 2: ['CONTACT_POOL_OVERFLOW']}"
    out = T.classify_process_exit(1, "Traceback (most recent call last):\n" + pool)
    assert (out["category"], out["kind"]) == ("infrastructure", "CONTACT_POOL_OVERFLOW")
    # the same exception object: classify_exception agrees
    assert T.classify_exception(FloatingPointError(pool.split(": ", 1)[1])) == ("infrastructure",
                                                                                 "CONTACT_POOL_OVERFLOW")
    task = T.classify_process_exit(1, "FloatingPointError: Invalid rigid worlds [1]: {1: ['RIGID_NONFINITE_STATE']}")
    assert (task["category"], task["kind"]) == ("task", "STRICT_INVALID_ENV")
