"""Shared helpers for exporters: triangulated, vertex-split render data."""
from __future__ import annotations

import numpy as np

from ..mesh import Mesh, triangulate_corners


def mesh_primitives(mesh: Mesh, flip_v: bool = False) -> list[dict]:
    """Split a Mesh into GPU-style primitives (one per material).

    Each primitive dict has: material (index into mesh.materials or -1 for none),
    positions, normals, uvs (or None), colors (or None), joints/weights (or None),
    targets {name: position deltas}, indices (uint32), source_vertex (per export vertex).
    """
    if not mesh.faces:
        return []
    cv, cf, _, _ = mesh._corners()
    cuv = mesh._corner_uvs()
    cn = mesh.corner_normals()
    tris, tface = triangulate_corners(mesh)
    qn = np.round(cn * 2048).astype(np.int64)
    key = np.column_stack([cv, cuv if cuv is not None else np.zeros_like(cv), qn])
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    fm = mesh.face_materials[tface] if len(mesh.face_materials) else np.zeros(len(tface), dtype=np.int64)
    prims = []
    mats = np.unique(fm)
    for m in mats:
        sel = tris[fm == m]
        if len(sel) == 0:
            continue
        ex = inv[sel]                      # export-vertex ids per triangle corner
        used, local = np.unique(ex.reshape(-1), return_inverse=True)
        corner_of = first[used]            # representative corner for each export vertex
        verts = cv[corner_of]
        prim = {
            "material": int(m) if mesh.materials else -1,
            "positions": mesh.vertices[verts].astype(np.float32),
            "normals": cn[corner_of].astype(np.float32),
            "uvs": None,
            "colors": None,
            "joints": None,
            "weights": None,
            "targets": {},
            "indices": local.reshape(-1).astype(np.uint32),
            "source_vertex": verts,
        }
        if cuv is not None:
            uv = mesh.uvs[cuv[corner_of]].copy()
            if flip_v:
                uv[:, 1] = 1.0 - uv[:, 1]
            prim["uvs"] = uv.astype(np.float32)
        if mesh.colors is not None:
            prim["colors"] = np.clip(mesh.colors[verts], 0, 1).astype(np.float32)
        if mesh.joints is not None:
            prim["joints"] = mesh.joints[verts].astype(np.uint16)
            w = mesh.weights[verts].astype(np.float32)
            w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-8)
            prim["weights"] = w
        for name, sk in mesh.shape_keys.items():
            prim["targets"][name] = (sk[verts] - mesh.vertices[verts]).astype(np.float32)
        prims.append(prim)
    return prims


def compute_tangents(positions: np.ndarray, normals: np.ndarray, uvs: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Per-vertex tangents (xyz + handedness w) from UV derivatives, for normal mapping."""
    tri = indices.reshape(-1, 3).astype(np.int64)
    p0, p1, p2 = (positions[tri[:, k]].astype(np.float64) for k in range(3))
    w0, w1, w2 = (uvs[tri[:, k]].astype(np.float64) for k in range(3))
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = w1 - w0, w2 - w0
    r = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(r) < 1e-12, 1e-12, r)
    sdir = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) / r[:, None]
    tdir = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) / r[:, None]
    tan1 = np.zeros((len(positions), 3))
    tan2 = np.zeros((len(positions), 3))
    for k in range(3):
        np.add.at(tan1, tri[:, k], sdir)
        np.add.at(tan2, tri[:, k], tdir)
    n = normals.astype(np.float64)
    t = tan1 - n * np.einsum("ij,ij->i", n, tan1)[:, None]
    ln = np.linalg.norm(t, axis=1)
    bad = ln < 1e-9
    # fallback: any vector perpendicular to the normal
    alt = np.cross(n, np.where(np.abs(n[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
    t[bad] = alt[bad]
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-12)[:, None]
    w = np.where(np.einsum("ij,ij->i", np.cross(n, t), tan2) < 0.0, -1.0, 1.0)
    return np.column_stack([t, w]).astype(np.float32)
