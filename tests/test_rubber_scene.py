"""R1.3: rubber_strip_v1 scene compiler: counts, lattice alignment, BC node sets, marker CSR, cell CSR."""
import numpy as np
import pytest

from robotic_cutting_sim import rubber_scene as rs


def test_counts_and_lattice_alignment_case_A():
    spec = rs.RubberStripSpec(length=0.03, width=0.004, thickness=0.003, n_t=3)
    sc = rs.compile_strip(spec)
    dx, s = 0.001, 0.0005
    assert sc.dx == pytest.approx(dx) and sc.spacing == pytest.approx(s)
    assert sc.lattice == (64, 8, 6)          # (L + 2 dx) / s, W / s, t / s
    assert sc.n == 64*8*6 == 8*(0.032*0.004*0.003)/dx**3 + 1e-6 // 1
    assert np.allclose(sc.X.min(0), [-0.002+s/2, s/2, s/2]) and np.allclose(sc.X.max(0), [0.03-s/2, 0.004-s/2, 0.003-s/2])
    # minimum corner on a grid line, weights a partition of unity with exact lattice fractions
    assert np.allclose((sc.corner-sc.grid_origin)/dx, 2.0)
    assert np.allclose(sc.weights.sum(1), 1.0, atol=1e-7)
    assert set(np.unique(sc.weights).round(8)) <= {0.015625, 0.046875, 0.140625, 0.421875}
    assert np.allclose(sc.gradients.sum(1), 0.0, atol=1e-3)
    assert sc.receipt()["cells_through_thickness"] == pytest.approx(3.0)
    rs.validate_scene_receipt(sc.receipt())


def test_stencil_matches_the_frozen_formula():
    """Same nodes/weights/gradients as the formula of tlmpm_v5.TLMPM.__init__ applied to the float64 lattice."""
    spec = rs.RubberStripSpec(length=0.01, width=0.003, thickness=0.003, n_t=3)
    sc = rs.compile_strip(spec)
    dx = sc.dx
    pts = sc.corner+(sc.lattice_index+0.5)*sc.spacing
    origin = sc.grid_origin
    cells = np.floor((pts-origin)/dx+1e-9).astype(int)
    fraction = (pts-origin)/dx-cells
    for corner in range(8):
        bit = np.array([(corner >> j) & 1 for j in range(3)])
        xyz = cells+bit
        assert np.array_equal(sc.nodes[:, corner], (xyz[:, 0]*sc.grid_dims[1]+xyz[:, 1])*sc.grid_dims[2]+xyz[:, 2])
        f = np.where(bit, fraction, 1-fraction)
        assert np.allclose(sc.weights[:, corner], f.prod(1), atol=1e-7)
        for axis in range(3):
            g = (2*bit[axis]-1)*f[:, (axis+1) % 3]*f[:, (axis+2) % 3]/dx
            assert np.allclose(sc.gradients[:, corner, axis], g, rtol=1e-6)


def test_cell_csr_is_a_partition_in_particle_order():
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.02, width=0.006, thickness=0.004, n_t=4))
    assert sc.cell_offsets[0] == 0 and sc.cell_offsets[-1] == sc.n
    assert np.array_equal(np.sort(sc.cell_particles), np.arange(sc.n))
    for c in range(len(sc.cell_node)):
        members = sc.cell_particles[sc.cell_offsets[c]:sc.cell_offsets[c+1]]
        assert np.all(np.diff(members) > 0)
        assert np.all(sc.nodes[members, 0] == sc.cell_node[c])
    assert np.all(sc.cell_count == 8)     # 2 particles per cell per axis, t multiple of dx
    first = [np.flatnonzero(sc.nodes[:, 0] == c)[0] for c in sc.cell_node]
    assert np.all(np.diff(first) > 0)


def test_clamp_group_case_A():
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.03, width=0.003, thickness=0.003, n_t=3))
    clamp = np.flatnonzero(sc.node_group == 0)
    assert np.all(sc.node_X[clamp, 0] <= 1e-12) and np.all(sc.node_has_mass[clamp])
    assert np.all(sc.node_mask[clamp] == rs.MASK_XYZ)
    free = np.flatnonzero(sc.node_has_mass & (sc.node_X[:, 0] > 1e-12))
    assert np.all(sc.node_group[free] == -1)
    assert sc.groups[0]["nodes"] == len(clamp)


def test_end_faces_and_pins_case_C():
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.06, width=0.01, thickness=0.004, n_t=4, case="C"))
    g0, g1 = np.flatnonzero(sc.node_group == 0), np.flatnonzero(sc.node_group == 1)
    assert np.allclose(sc.node_X[g0, 0], 0.0) and np.allclose(sc.node_X[g1, 0], 0.06)
    assert len(g0) == len(g1) == 11*5
    for g in (g0, g1):
        centre = g[np.isclose(sc.node_X[g, 1], 0.005) & np.isclose(sc.node_X[g, 2], 0.002)]
        low = g[np.isclose(sc.node_X[g, 1], 0.005) & np.isclose(sc.node_X[g, 2], 0.0)]
        assert len(centre) == 1 and sc.node_mask[centre[0]] == rs.MASK_XYZ
        assert len(low) == 1 and sc.node_mask[low[0]] == rs.MASK_X | rs.MASK_Y
        assert np.count_nonzero(sc.node_mask[g] == rs.MASK_X) == len(g)-2


def test_markers_and_surface():
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.04, width=0.004, thickness=0.004, n_t=4))
    assert len(sc.marker_offsets) == rs.MARKERS+1
    ref_x = sc.marker_reference[:, 0]
    s = sc.spacing
    assert ref_x[0] == pytest.approx(s/2) and ref_x[-1] == pytest.approx(0.04-s/2)
    assert np.all(np.abs(ref_x-0.005*np.arange(9)) <= s/2+1e-12)
    for k in range(rs.MARKERS):
        ids = sc.marker_particles[sc.marker_offsets[k]:sc.marker_offsets[k+1]]
        assert len(ids) == 8*8 and np.allclose(sc.X[ids, 0], ref_x[k])
    assert np.array_equal(sc.tip_particles, sc.marker_particles[sc.marker_offsets[-2]:])
    nx, ny, nz = sc.lattice
    assert sc.surface.sum() == nx*ny*nz-(nx-2)*(ny-2)*(nz-2)


def test_block_case_and_interior_nodes():
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.006, width=0.006, thickness=0.006, dx=0.001, case="block"))
    assert sc.n == 12**3 and np.allclose(sc.X.mean(0), 0.0, atol=1e-9)
    assert sc.groups[0]["kind"] == "affine" and np.all(sc.node_group[sc.node_has_mass] == 0)
    assert len(sc.interior_nodes()) == 5**3


def test_refusals():
    with pytest.raises(ValueError, match="lattice"):
        rs.compile_strip(rs.RubberStripSpec(length=0.0303, width=0.003, thickness=0.003))
    with pytest.raises(ValueError, match="case"):
        rs.compile_strip(rs.RubberStripSpec(length=0.03, width=0.003, thickness=0.003, case="Z"))
    with pytest.raises(ValueError, match="2 particles"):
        rs.compile_strip(rs.RubberStripSpec(length=0.03, width=0.003, thickness=0.003, ppc=4))


def test_ppc3_quadrature_check_lattice():
    """ADR 0023 §14.4: 3 particles per cell per axis only as the refinement study's quadrature check; the exact
    stencil keeps the partition of unity and zero gradient sum, 27 particles per cell."""
    sc = rs.compile_strip(rs.RubberStripSpec(length=0.03, width=0.003, thickness=0.003, ppc=3))
    assert sc.spacing == pytest.approx(0.001/3) and set(sc.cell_count.tolist()) == {27}
    assert np.allclose(sc.weights.sum(1), 1.0, atol=1e-6)
    assert np.abs(sc.gradients.sum(1)).max() < 1e-3


def test_traction_region():
    spec = rs.RubberStripSpec(length=0.03, width=0.003, thickness=0.003,
                              traction_regions=(("tip", (0.029, -1, -1), (1, 1, 1)),))
    sc = rs.compile_strip(spec)
    sel = sc.particle_traction == 0
    assert sel.sum() == sc.traction_groups[0]["particles"] == 2*6*6
