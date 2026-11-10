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
