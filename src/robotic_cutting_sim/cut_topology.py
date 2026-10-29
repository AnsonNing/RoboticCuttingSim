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


# ----------------------------------------------------------------------------- packed codes (ADR 0014, tlmpm_v5)
#
# One int64/int32 code holds 2 bits per cut: d_k = (code >> 2k) & 3, with
# 0 unassigned, 1 negative side, 2 positive side and 3 invalid. The semantics
# are exactly the ternary digits above; only the encoding differs.

PACKED_MAX_CUTS = 15
_LOW_BITS = 0x55555555


def _as_codes(codes):
    codes = np.asarray(codes)
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError('Packed cut codes must be integers')
    return codes.astype(np.int64, copy=False)


def validate_packed(codes, cuts):
    """True when every code uses only digits 0/1/2 of cuts 0..cuts-1."""
    codes = _as_codes(codes)
    if not 1 <= int(cuts) <= PACKED_MAX_CUTS:
        raise ValueError(f'Packed codes keep 1..{PACKED_MAX_CUTS} cuts')
    low, high = codes & _LOW_BITS, (codes >> 1) & _LOW_BITS
    return bool(np.all(codes >= 0) and np.all(codes < (1 << (2*int(cuts)))) and not np.any(low & high))


def decode_digits(codes, cuts):
    """(..., cuts) int8 digits of packed codes."""
    codes = _as_codes(codes)
    shifts = 2*np.arange(int(cuts))
    return ((codes[..., None] >> shifts) & 3).astype(np.int8)


def pack_digits(digits):
    """Packed codes from (..., K) digits in {0, 1, 2}."""
    digits = np.asarray(digits, np.int64)
    if digits.shape[-1] > PACKED_MAX_CUTS or np.any((digits < 0) | (digits > 2)):
        raise ValueError(f'Digits must be 0, 1 or 2 for at most {PACKED_MAX_CUTS} cuts')
    return (digits << (2*np.arange(digits.shape[-1]))).sum(-1)


def packed_to_ternary(codes, cuts):
    """Same history in v1-v4 ternary encoding (sum d_k 3^k); exact bijection."""
    return (decode_digits(codes, cuts).astype(np.int64)*3**np.arange(int(cuts), dtype=np.int64)).sum(-1)


def ternary_to_packed(codes, cuts):
    codes = _as_codes(codes)
    digits = (codes[..., None]//3**np.arange(int(cuts), dtype=np.int64)) % 3
    return pack_digits(digits)


def side_masks(codes):
    """(A1, A2): bit 2k set where digit k is 1 (negative) / 2 (positive)."""
    codes = _as_codes(codes)
    low, high = codes & _LOW_BITS, (codes >> 1) & _LOW_BITS
    return low & ~high, high & ~low


def conflicting_packed(a, b):
    """Opposite sides of some cut: (A1 & B2) | (A2 & B1) != 0 (ADR 0014)."""
    a1, a2 = side_masks(a)
    b1, b2 = side_masks(b)
    return ((a1 & b2) | (a2 & b1)) != 0


def mechanical_components_packed(nodes, weights, codes, cuts=None, *, gradients=None):
    """mechanical_components for packed codes: the same actual stencil graph.

    Particles sharing a code at a reference node are connected, and so are two
    codes at one node that do not conflict (bitmask test, no 3^K table). Valid
    stencil entries are the same as mechanical_components (weight > 0, or
    |gradient| > 0 when gradients are given). Vectorized: particles and
    (node, code) groups form one sparse graph. Labels are numbered by the first
    particle of each component.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    codes = _as_codes(codes)
    nodes = np.asarray(nodes)
    if cuts is not None and not validate_packed(codes, cuts):
        raise ValueError(f'Codes are not valid packed codes for {cuts} cuts')
    n = len(codes)
    valid = np.asarray(weights).reshape(-1) > 0
    if gradients is not None:
        valid |= np.linalg.norm(np.asarray(gradients), axis=-1).reshape(-1) > 0
    particle = np.repeat(np.arange(n), nodes.shape[1])[valid]
    node = nodes.reshape(-1).astype(np.int64)[valid]
    code = codes[particle]
    pairs, group = np.unique(np.stack([node, code], 1), axis=0, return_inverse=True)
    group = group.reshape(-1)
    rows, cols = [particle], [n+group]
    # pairs are sorted by node then code: compare each group with later groups of its node.
    for offset in range(1, len(pairs)):
        same = pairs[offset:, 0] == pairs[:-offset, 0]
        if not same.any():
            break
        first = np.flatnonzero(same)
        compatible = ~conflicting_packed(pairs[first, 1], pairs[first+offset, 1])
        rows.append(n+first[compatible]); cols.append(n+first[compatible]+offset)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    size = n+len(pairs)
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(size, size)).tocsr()
    _, labels = connected_components(graph, directed=False)
    _, first_index, labels = np.unique(labels[:n], return_index=True, return_inverse=True)
    order = np.argsort(np.argsort(first_index))
    labels = order[labels.reshape(-1)]
    return int(labels.max())+1 if n else 0, labels


def reference_components_packed(points, lattice_indices, codes):
    """reference_components (six-neighbor reference bonds) for packed codes."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    codes = _as_codes(codes)
    lattice = np.asarray(lattice_indices)
    lookup = {tuple(v): i for i, v in enumerate(lattice)}
    pairs = [(i, lookup[tuple(v+step)]) for i, v in enumerate(lattice)
             for step in np.eye(3, dtype=int) if tuple(v+step) in lookup]
    a, b = np.asarray(pairs, int).reshape(-1, 2).T
    keep = ~conflicting_packed(codes[a], codes[b])
    rows, cols = np.r_[a[keep], b[keep]], np.r_[b[keep], a[keep]]
    graph = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(points),)*2).tocsr()
    return connected_components(graph, directed=False)
