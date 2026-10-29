"""Optional CUDA probe recording; no physical state is modified."""
import warp as wp


@wp.kernel
def record_reaction(reaction: wp.array(dtype=wp.spatial_vector), forces: wp.array2d(dtype=float),
                    index: int, dt: float):
    for axis in range(3):
        forces[index, axis] = reaction[0][axis] / dt
