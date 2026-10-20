"""Mesh simplification and surface fairing: quadric-error decimation, Taubin smoothing.

    mesh = shape.to_mesh(cell=0.005)          # dense, uniform SDF mesh
    mesh.smooth_taubin(10)                    # remove grid ripples without shrinking
    mesh.decimate(12000, max_edge=0.03)       # adaptive: detail stays where curvature is

Decimation is Garland-Heckbert edge collapse (quadric error metric) with optimal vertex
placement, guarded against fold-overs, non-manifold collapses and slivers. Material borders are
preserved (penalty planes), `max_edge` bounds the edge length (a float, or fn(points)->(N,)
lengths for a sizing field) so long thin triangles do not form on cylinders, which matters for
skinned limbs that must bend.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from . import math3d as m3


def taubin_smooth(mesh, iterations: int = 10, lam: float = 0.5, mu: float = -0.53, verts=None,
                  keep_boundary: bool = True):
    """Volume-preserving (Taubin lambda|mu) smoothing: a shrink step followed by an inflate step,
    so bumps and grid ripples fade while the overall shape keeps its size."""
    mesh.normals = None
    edges = mesh.edges()
    nv = len(mesh.vertices)
    deg = np.bincount(edges.reshape(-1), minlength=nv).astype(np.float64)
    mask = np.ones(nv, dtype=bool)
    if verts is not None:
        mask[:] = False
        mask[np.asarray(verts)] = True
    if keep_boundary:
        be = mesh.boundary_edges()
        mask[be.reshape(-1)] = False
    upd = mask & (deg > 0)
    V = mesh.vertices
    for _ in range(int(iterations)):
        for f in (lam, mu):
            acc = np.zeros_like(V)
            np.add.at(acc, edges[:, 0], V[edges[:, 1]])
            np.add.at(acc, edges[:, 1], V[edges[:, 0]])
            avg = acc / np.maximum(deg, 1)[:, None]
            V[upd] += f * (avg[upd] - V[upd])
    return mesh


def _cross(a, b):
    """np.cross for (..., 3) arrays without its axis bookkeeping (hot loop)."""
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return np.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def _face_planes(V, F):
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    area2 = np.linalg.norm(n, axis=1)
    n = n / np.maximum(area2, 1e-30)[:, None]
    d = -np.einsum("ij,ij->i", n, a)
    return n, d, area2 * 0.5


def _plane_quadrics(n, d, w):
    p = np.concatenate([n, d[:, None]], axis=1)
    return w[:, None, None] * p[:, :, None] * p[:, None, :]


def _optimal(Q, a, b):
    """Batched optimal collapse positions and costs for quadrics Q (k,4,4) and edges a-b."""
    A = Q[:, :3, :3]
    rhs = -Q[:, :3, 3]
    mid = 0.5 * (a + b)
    # Tikhonov pull toward the edge midpoint keeps flat / cylindrical regions well-posed
    tr = np.trace(A, axis1=1, axis2=2) / 3.0
    eps = (1e-4 * tr + 1e-12)[:, None]
    x = np.linalg.solve(A + eps[:, :, None] * np.eye(3)[None], (rhs + eps * mid)[:, :, None])[:, :, 0]
    cands = np.stack([x, a, b, mid], axis=1)  # (k,4,3)
    h = np.concatenate([cands, np.ones(cands.shape[:2] + (1,))], axis=2)
    cost = np.einsum("kci,kij,kcj->kc", h, Q, h)
    best = np.argmin(cost, axis=1)
    return cands[np.arange(len(x)), best], cost[np.arange(len(x)), best]


def decimate(mesh, target: int | float, max_edge=None, keep_boundary: bool = True,
             border_weight: float = 1000.0, min_quality: float = 0.06, max_normal_change: float = 75.0):
    """Collapse edges (quadric error) until the mesh has at most `target` triangles
    (an int, or a float in (0, 1) = fraction of the current triangle count).

    max_edge         : longest allowed edge (meters) or fn(points (N,3)) -> (N,) lengths
    keep_boundary    : open borders stay where they are
    border_weight    : strength of the planes that keep material borders in place
    min_quality      : reject collapses that create triangles thinner than this (0..1, 1 = equilateral)
    max_normal_change: reject collapses that turn a neighbouring face more than this (degrees)

    The mesh is triangulated; vertex colors / skin weights of surviving vertices are kept
    (colors interpolated along the collapsed edge); UVs and shape keys are dropped.
    """
    mesh.triangulate()
    V = mesh.vertices.copy()
    F = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 3)
    FM = np.asarray(mesh.face_materials, dtype=np.int64)
    nv, nf = len(V), len(F)
    if isinstance(target, float) and target < 1.0:
        target = int(nf * target)
    target = int(target)
    if nf <= target:
        return mesh
    cos_limit = math.cos(math.radians(max_normal_change))

    # quadrics
    n, d, area = _face_planes(V, F)
    Kf = _plane_quadrics(n, d, area)
    Q = np.zeros((nv, 4, 4))
    for k in range(3):
        np.add.at(Q, F[:, k], Kf)
    # border planes (open boundary and material borders)
    e_all = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    f_all = np.tile(np.arange(nf), 3)
    key = np.sort(e_all, axis=1)
    uniq, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    boundary_v = np.zeros(nv, dtype=bool)
    border_e = cnt[inv] == 1
    boundary_v[e_all[border_e].reshape(-1)] = True
    if len(np.unique(FM)) > 1:
        order = np.argsort(inv, kind="stable")
        si = inv[order]
        first = np.flatnonzero(np.r_[True, si[1:] != si[:-1]])
        pair = first[cnt[si[first]] == 2]
        fa, fb = f_all[order[pair]], f_all[order[pair + 1]]
        mb = FM[fa] != FM[fb]
        border_e = border_e | np.isin(inv, si[pair[mb]])
    if border_e.any():
        ea = e_all[border_e]
        fn = n[f_all[border_e]]
        ev = V[ea[:, 1]] - V[ea[:, 0]]
        pn = m3.normalize(np.cross(ev, fn))
        pd = -np.einsum("ij,ij->i", pn, V[ea[:, 0]])
        Kb = _plane_quadrics(pn, pd, border_weight * np.einsum("ij,ij->i", ev, ev))
        np.add.at(Q, ea[:, 0], Kb)
        np.add.at(Q, ea[:, 1], Kb)
    locked = boundary_v if keep_boundary else np.zeros(nv, dtype=bool)

    def edge_limit(p):
        if max_edge is None:
            return np.full(len(p), np.inf)
        if callable(max_edge):
            return np.asarray(max_edge(p), dtype=np.float64).reshape(-1)
        return np.full(len(p), float(max_edge))

    # adjacency (python sets for fast incremental updates)
    vf = [set() for _ in range(nv)]
    for fi, (a, b, c) in enumerate(F.tolist()):
        vf[a].add(fi)
        vf[b].add(fi)
        vf[c].add(fi)
    Fl = F.tolist()
    falive = np.ones(nf, dtype=bool)
    valive = np.ones(nv, dtype=bool)
    stamp = np.zeros(nv, dtype=np.int64)
    colors = mesh.colors.copy() if mesh.colors is not None else None
    joints = mesh.joints.copy() if mesh.joints is not None else None
    weights = mesh.weights.copy() if mesh.weights is not None else None

    heap = []
    counter = [0]

    def push(pairs):
        if not pairs:
            return
        P = np.asarray(pairs, dtype=np.int64)
        P = P[~(locked[P[:, 0]] | locked[P[:, 1]])]
        if not len(P):
            return
        x, c = _optimal(Q[P[:, 0]] + Q[P[:, 1]], V[P[:, 0]], V[P[:, 1]])
        for (a, b), cc, xx in zip(P.tolist(), c.tolist(), x):
            counter[0] += 1
            heapq.heappush(heap, (cc, counter[0], a, b, int(stamp[a]), int(stamp[b]), xx))

    push(uniq.tolist())
    faces_left = nf

    def tri_quality(p0, p1, p2):
        e1, e2, e3 = p1 - p0, p2 - p1, p0 - p2
        ar = np.sqrt((_cross(e1, -e3) ** 2).sum(-1))
        ss = (e1 * e1).sum(-1) + (e2 * e2).sum(-1) + (e3 * e3).sum(-1)
        return 2.0 * math.sqrt(3.0) * ar / np.maximum(ss, 1e-30)

    rejected = {"link": 0, "flip": 0, "quality": 0, "edge": 0}
    while heap and faces_left > target:
        cost, _, a, b, sa, sb, x = heapq.heappop(heap)
        if not (valive[a] and valive[b]) or stamp[a] != sa or stamp[b] != sb:
            continue
        shared = vf[a] & vf[b]
        if not shared:
            continue
        # link condition: common neighbours == opposite vertices of the shared faces
        na = {v for f in vf[a] for v in Fl[f]}
        nb = {v for f in vf[b] for v in Fl[f]}
        common = (na & nb) - {a, b}
        opp = {v for f in shared for v in Fl[f]} - {a, b}
        if common != opp:
            rejected["link"] += 1
            continue
        ring = list((vf[a] | vf[b]) - shared)
        if not ring:
            continue
        T = np.asarray([Fl[f] for f in ring], dtype=np.int64)
        P0 = V[T]
        P1 = P0.copy()
        hit = (T == a) | (T == b)
        P1[hit] = x
        n0 = _cross(P0[:, 1] - P0[:, 0], P0[:, 2] - P0[:, 0])
        n1 = _cross(P1[:, 1] - P1[:, 0], P1[:, 2] - P1[:, 0])
        l0 = np.sqrt((n0 * n0).sum(1))
        l1 = np.sqrt((n1 * n1).sum(1))
        if (l1 < 1e-14).any():
            continue
        if ((n0 * n1).sum(1) < cos_limit * l0 * l1).any():
            rejected["flip"] += 1
            continue
        q1 = tri_quality(P1[:, 0], P1[:, 1], P1[:, 2])
        if (q1 < min_quality).any():
            q0 = tri_quality(P0[:, 0], P0[:, 1], P0[:, 2])
            if (q1 < np.minimum(min_quality, q0) - 1e-9).any():
                rejected["quality"] += 1
                continue
        nbrs = (na | nb) - {a, b}
        if max_edge is not None:
            nl = np.asarray(list(nbrs), dtype=np.int64)
            lim = edge_limit(x[None])[0]
            dv = V[nl] - x
            if ((dv * dv).sum(1) > lim * lim).any():
                rejected["edge"] += 1
                continue
        # collapse b into a at x
        if colors is not None or joints is not None:
            ab = V[b] - V[a]
            t = float(np.clip(np.dot(x - V[a], ab) / max(float(ab @ ab), 1e-30), 0, 1))
            if colors is not None:
                colors[a] = colors[a] * (1 - t) + colors[b] * t
            if joints is not None and t > 0.5:
                joints[a], weights[a] = joints[b], weights[b]
        V[a] = x
        Q[a] += Q[b]
        valive[b] = False
        for f in shared:
            falive[f] = False
            for v in Fl[f]:
                vf[v].discard(f)
            faces_left -= 1
        for f in vf[b]:
            fv = Fl[f]
            Fl[f] = [a if v == b else v for v in fv]
            vf[a].add(f)
        vf[b] = set()
        stamp[a] += 1
        push([(a, w) for w in nbrs if valive[w]])

    mesh.extras["decimate_rejected"] = rejected
    keep = np.flatnonzero(falive)
    Fk = np.asarray([Fl[f] for f in keep], dtype=np.int64).reshape(-1, 3)
    used = np.unique(Fk)
    remap = -np.ones(nv, dtype=np.int64)
    remap[used] = np.arange(len(used))
    mesh.vertices = V[used]
    mesh.faces = [tuple(r) for r in remap[Fk].tolist()]
    mesh.face_materials = FM[keep]
    mesh.uvs = None
    mesh.face_uvs = None
    mesh.colors = colors[used] if colors is not None else None
    mesh.joints = joints[used] if joints is not None else None
    mesh.weights = weights[used] if weights is not None else None
    mesh.shape_keys = {}
    mesh.normals = None
    return mesh
