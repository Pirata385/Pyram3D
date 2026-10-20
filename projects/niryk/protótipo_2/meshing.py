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


def mesh_sdf(shape, cell: float, target: int, max_edge=None, name: str = "Mesh", taubin: int = 4,
             verbose: bool = True, colors: bool | None = None) -> p3.Mesh:
    t0 = time.time()
    m = shape.to_mesh(cell=cell, name=name, colors=colors, band=1.5)
    n0 = m.num_triangles
    if taubin:
        m.smooth_taubin(taubin)
        shape.project(m, iterations=2, max_step=cell)
    t1 = time.time()
    if target and m.num_triangles > target:
        m.decimate(target, max_edge=max_edge)
    shape.project(m, iterations=2, max_step=cell)
    m.normals = shape.gradient(m.vertices, eps=cell * 0.25)
    if verbose:
        print(f"  {name}: {n0} -> {m.num_triangles} tris (mesh {t1 - t0:.1f}s, decimate {time.time() - t1:.1f}s)")
    return m


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
