"""Persistent, online cut history and conservative logical-grid coupling.

Pure NumPy reference routines used to test the GPU implementation. No surface is
presegmented: every particle starts with code zero, and a moving finite edge
assigns previously unset ternary digits. Labels change mechanical compatibility,
not particle position, mass, velocity, or visibility.
"""
import numpy as np


def conflicting(a, b, cuts=2):
    a, b = np.asarray(a), np.asarray(b)
    result = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=bool)
    for k in range(cuts):
        da, db = (a // 3**k) % 3, (b // 3**k) % 3
        result |= ((da == 1) & (db == 2)) | ((da == 2) & (db == 1))
    return result


def compatibility(cuts=2):
    codes = np.arange(3**cuts)
    return ~conflicting(codes[:, None], codes[None, :], cuts)


def couple(momentum, mass, compatible):
    """Pairwise antisymmetric exchange, including unassigned material.

    All inputs refer to one reference-grid node. The mass of each logical field
    is retained; both total momentum and a common input velocity are preserved.
    """
    p, m = np.asarray(momentum, float), np.asarray(mass, float)
    total = m.sum()
    if total == 0:
        return p.copy()
    exchange = m[:, None, None]*p[None, :, :] - m[None, :, None]*p[:, None, :]
    return p + (exchange*compatible[:, :, None]).sum(axis=1)/total


def sweep_labels(positions, codes, edge, rotation, displacement, *, cut=0,
                 half_width=.024, edge_half_length=.025, rear=.004):
    """Reference implementation for an advancing edge (columns d,e,n in R)."""
    if cut not in (0, 1):
        raise ValueError("This prototype supports two sequential cuts")
    local = (np.asarray(positions)-edge) @ np.asarray(rotation)
    x, y, z = local.T
    factor = 3**cut
    eligible = (((codes//factor) % 3) == 0) & (np.abs(z) <= half_width)
    eligible &= (np.abs(y) <= edge_half_length) & (x <= 0)
    eligible &= x >= -rear-max(float(displacement), 0)
    result = np.asarray(codes).copy()
    result[eligible] += np.where(z[eligible] < 0, 1, 2)*factor
    return result


def reference_components(points, lattice_indices, codes, cuts=2):
    """Six-neighbor material connectivity after online incompatible-side cuts.

    This is an oracle diagnostic, never an observation or rendering mask. It
    records reference bonds severed by the simulated blade history; it does not
    decide separation from a visual gap alone.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    lookup = {tuple(v): i for i, v in enumerate(lattice_indices)}
    rows, cols = [], []
    for i, ijk in enumerate(lattice_indices):
        for axis in range(3):
            neighbor = ijk.copy(); neighbor[axis] += 1
            j = lookup.get(tuple(neighbor))
            if j is not None and not conflicting(codes[i], codes[j], cuts):
                rows.extend((i, j)); cols.extend((j, i))
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points),)*2).tocsr()
    count, component = connected_components(graph, directed=False)
    return count, component


def mechanical_components(nodes, weights, codes, cuts=2, *, gradients=None):
    """Components of the actual positive-weight, compatible MPM stencils.

    Unlike six-neighbor lattice bonds, this also detects diagonal and unassigned
    field bridges. A component may contact another component without becoming
    connected by this continuum transfer graph.
    """
    codes = np.asarray(codes)
    parent = np.arange(len(codes))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        a, b = find(i), find(j)
        if a != b:
            parent[b] = a

    valid = np.asarray(weights).reshape(-1) > 0
    if gradients is not None:
        # Zero shape weight need not imply zero stress/gradient coupling. Include
        # those stencil entries conservatively in the connectivity audit too.
        valid |= np.linalg.norm(np.asarray(gradients), axis=-1).reshape(-1) > 0
    particle = np.repeat(np.arange(len(codes)), nodes.shape[1])[valid]
    node = np.asarray(nodes).reshape(-1)[valid]
    order = np.argsort(node, kind="stable")
    node, particle = node[order], particle[order]
    groups = np.split(particle, np.flatnonzero(np.diff(node))+1)
    table = compatibility(cuts)
    for group in groups:
        representatives = {}
        for p in group:
            c = codes[p]
            if c in representatives:
                union(p, representatives[c])
            else:
                representatives[c] = p
        occupied = list(representatives)
        for i, a in enumerate(occupied):
            for b in occupied[i+1:]:
                if table[a, b]:
                    union(representatives[a], representatives[b])
    roots = np.array([find(i) for i in range(len(codes))])
    unique, labels = np.unique(roots, return_inverse=True)
    return len(unique), labels
