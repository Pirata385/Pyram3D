"""SDF -> smooth game mesh: fine surface nets, Taubin fairing, quadric decimation, analytic normals.

The SDF is meshed on a fine uniform grid (every vertex on the true surface), faired with a few
volume-preserving Taubin passes (removes the grid's stair-step jitter), re-projected, then
decimated with the quadric error metric to the triangle budget: triangles concentrate where the
surface curves (face, hands, hems) and stay large on broad forms. `max_edge` caps edge lengths so
limbs keep enough rings to bend at the joints. Normals come from the SDF gradient (exactly smooth)
and are stored as frozen normals, so the glTF carries them.
"""
from __future__ import annotations

import time

import numpy as np

import pyram3d as p3


def drop_faces(m: p3.Mesh, vert_mask: np.ndarray, rings: int = 0) -> int:
    """Delete the faces whose vertices are all flagged (e.g. hidden under a garment, or outside a
    region). rings > 0 first shrinks the flagged area by that many vertex rings (so a margin of
    faces survives along its border). Returns the number of faces deleted."""
    if not len(m.faces):
        return 0
    F = np.asarray(m.faces, dtype=np.int64)
    keep_v = ~np.asarray(vert_mask, bool)
    for _ in range(rings):
        touch = keep_v[F].any(axis=1)
        grown = keep_v.copy()
        grown[F[touch].reshape(-1)] = True
        keep_v = grown
    kill = np.flatnonzero(~keep_v[F].any(axis=1))
    if len(kill):
        m.delete_faces(kill)
    return len(kill)


def mesh_sdf(shape, cell: float, target: int, max_edge=None, name: str = "Mesh", taubin: int = 4,
             verbose: bool = True, colors: bool | None = None, bounds=None, outside=None, hidden=None,
             rings: int = 4) -> p3.Mesh:
    """Mesh `shape` (optionally only inside `bounds` = (lo, hi): the cut is left open), fair it,
    decimate it to `target` triangles and give it analytic normals.

    outside: fn(points) -> bool, vertices outside the part's region (their faces are dropped
             before decimation; the open border must lie where something covers it)
    hidden:  fn(points) -> bool, vertices hidden under something else (garments, hair, the body):
             their faces are dropped (deep ones before decimation, the rest after it)
    """
    t0 = time.time()
    m = shape.to_mesh(cell=cell, name=name, colors=colors, band=1.5, bounds=bounds,
                      padding=0.0 if bounds is not None else 0.05)
    n0 = m.num_triangles
    if taubin:
        m.smooth_taubin(taubin)
        shape.project(m, iterations=2, max_step=cell)
    if outside is not None:
        drop_faces(m, outside(m.vertices))
    if hidden is not None:
        drop_faces(m, hidden(m.vertices), rings=rings)
    n1 = m.num_triangles
    t1 = time.time()
    if target and m.num_triangles > target:
        m.decimate(target, max_edge=max_edge)
    shape.project(m, iterations=2, max_step=cell)
    if hidden is not None:
        drop_faces(m, hidden(m.vertices))
    m.normals = shape.gradient(m.vertices, eps=cell * 0.25)
    if verbose:
        print(f"  {name}: {n0} -> {n1} -> {m.num_triangles} tris (mesh {t1 - t0:.1f}s, decimate {time.time() - t1:.1f}s)")
    return m


def covered_by(shapes, margin: float = 0.003, where=None):
    """hidden() test: points deeper than `margin` inside any of `shapes` (and where(points))."""
    def f(P):
        h = np.zeros(len(P), bool)
        for s in shapes:
            h |= s.dist(P) < -margin
        if where is not None:
            h &= where(P)
        return h
    return f


def sizing(regions, default: float):
    """max_edge field: regions = [(fn(points)->bool mask, max edge length), ...] (first match wins)."""
    def f(P):
        P = np.atleast_2d(P)
        out = np.full(len(P), float(default))
        done = np.zeros(len(P), bool)
        for test, L in regions:
            m = np.asarray(test(P), bool) & ~done
            out[m] = L
            done |= m
        return out
    return f
