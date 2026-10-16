"""Digital sculpting brushes for dense meshes (SDF meshes or subdivided meshes).

Each brush acts on vertices within `radius` of a center point with a smooth
falloff. Strokes apply a brush repeatedly along a path. `symmetry="x"` mirrors
the stroke across x=0 (characters, faces, vehicles).

    m = p3.sphere(0.5, 64, 32)
    sculpt.draw(m, center=(0, 0.3, 0.4), radius=0.15, strength=0.05)   # bump
    sculpt.grab(m, center=(0, -0.5, 0), radius=0.3, offset=(0, -0.2, 0)) # pull
    sculpt.stroke(m, "crease", path=[(..), (..)], radius=0.05, strength=0.01)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from . import math3d as m3
from .mesh import Mesh


def _falloff(d: np.ndarray, radius: float, kind: str = "smooth") -> np.ndarray:
    t = np.clip(1.0 - d / max(radius, 1e-12), 0, 1)
    if kind == "linear":
        return t
    if kind == "sharp":
        return t * t
    if kind == "constant":
        return (t > 0).astype(np.float64)
    return t * t * (3 - 2 * t)  # smooth


def _mirror_centers(center, symmetry: str | None):
    c = m3.vec3(center)
    out = [c]
    if symmetry:
        for ax in symmetry.lower():
            i = "xyz".index(ax)
            for p in list(out):
                q = p.copy()
                q[i] = -q[i]
                if not np.allclose(q, p):
                    out.append(q)
    return out


def _mirror_vector(v, center_src, center_dst):
    v = m3.vec3(v)
    out = v.copy()
    for i in range(3):
        if abs(center_src[i] + center_dst[i]) < 1e-9 and abs(center_src[i]) > 1e-9:
            out[i] = -out[i]
    return out


def _region(mesh: Mesh, center, radius, falloff):
    d = np.linalg.norm(mesh.vertices - center, axis=1)
    w = _falloff(d, radius, falloff)
    idx = np.flatnonzero(w > 0)
    return idx, w[idx]


def draw(mesh: Mesh, center, radius: float = 0.1, strength: float = 0.02, falloff: str = "smooth",
         symmetry: str | None = None) -> Mesh:
    """Raise (strength > 0) or carve (< 0) the surface along its normals."""
    vn = mesh.vertex_normals()
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, falloff)
        if len(idx) == 0:
            continue
        avg_n = m3.normalize((vn[idx] * w[:, None]).sum(axis=0))
        mesh.vertices[idx] += avg_n * (w * strength)[:, None]
    return mesh


def inflate(mesh: Mesh, center, radius: float = 0.1, strength: float = 0.02, falloff: str = "smooth",
            symmetry: str | None = None) -> Mesh:
    """Push each vertex along its own normal (puffs up / deflates with negative strength)."""
    vn = mesh.vertex_normals()
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, falloff)
        mesh.vertices[idx] += vn[idx] * (w * strength)[:, None]
    return mesh


def grab(mesh: Mesh, center, radius: float = 0.2, offset=(0, 0.1, 0), falloff: str = "smooth",
         symmetry: str | None = None) -> Mesh:
    """Move a region by `offset` (pull a nose, stretch a snout, bend an ear)."""
    c0 = m3.vec3(center)
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, falloff)
        off = _mirror_vector(offset, c0, c) if not np.allclose(c, c0) else m3.vec3(offset)
        mesh.vertices[idx] += off[None, :] * w[:, None]
    return mesh


def smooth(mesh: Mesh, center, radius: float = 0.1, strength: float = 0.5, iterations: int = 3,
           symmetry: str | None = None) -> Mesh:
    """Relax the surface locally."""
    edges = mesh.edges()
    nv = len(mesh.vertices)
    deg = np.bincount(edges.reshape(-1), minlength=nv).astype(np.float64)
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, "smooth")
        wfull = np.zeros(nv)
        wfull[idx] = w * strength
        for _ in range(iterations):
            acc = np.zeros_like(mesh.vertices)
            np.add.at(acc, edges[:, 0], mesh.vertices[edges[:, 1]])
            np.add.at(acc, edges[:, 1], mesh.vertices[edges[:, 0]])
            avg = acc / np.maximum(deg, 1)[:, None]
            mesh.vertices += (avg - mesh.vertices) * wfull[:, None]
    return mesh


def flatten(mesh: Mesh, center, radius: float = 0.1, strength: float = 1.0, symmetry: str | None = None) -> Mesh:
    """Pull vertices toward the average plane of the region."""
    vn = mesh.vertex_normals()
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, "smooth")
        if len(idx) < 3:
            continue
        p0 = (mesh.vertices[idx] * w[:, None]).sum(axis=0) / w.sum()
        n = m3.normalize((vn[idx] * w[:, None]).sum(axis=0))
        dist = (mesh.vertices[idx] - p0) @ n
        mesh.vertices[idx] -= np.outer(dist * w * strength, n)
    return mesh


def pinch(mesh: Mesh, center, radius: float = 0.1, strength: float = 0.5, symmetry: str | None = None) -> Mesh:
    """Pull vertices toward the brush center (sharpens creases and ridges)."""
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, "smooth")
        mesh.vertices[idx] += (c - mesh.vertices[idx]) * (w * strength)[:, None] * 0.5
    return mesh


def crease(mesh: Mesh, center, radius: float = 0.05, strength: float = 0.01, symmetry: str | None = None) -> Mesh:
    """Carve a sharp groove (wrinkles, lips, eyelids, panel lines)."""
    draw(mesh, center, radius, -abs(strength), "sharp", symmetry)
    return pinch(mesh, center, radius, 0.3, symmetry)


def noise(mesh: Mesh, center, radius: float = 0.2, strength: float = 0.01, scale: float = 20.0, seed: int = 0,
          symmetry: str | None = None) -> Mesh:
    """Local noisy displacement (skin pores, rock grit)."""
    from .noise import fbm3
    vn = mesh.vertex_normals()
    for c in _mirror_centers(center, symmetry):
        idx, w = _region(mesh, c, radius, "smooth")
        h = fbm3(mesh.vertices[idx] * scale, 4, seed=seed)
        mesh.vertices[idx] += vn[idx] * (h * w * strength)[:, None]
    return mesh


_BRUSHES = {"draw": draw, "inflate": inflate, "grab": grab, "smooth": smooth, "flatten": flatten,
            "pinch": pinch, "crease": crease, "noise": noise}


def stroke(mesh: Mesh, brush: str, path: Sequence[Sequence[float]], radius: float = 0.05, spacing: float = 0.5,
           **kw) -> Mesh:
    """Apply a brush repeatedly along a path. spacing is relative to radius."""
    fn = _BRUSHES[brush]
    pts = np.asarray(path, dtype=np.float64)
    if len(pts) > 1:
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        L = np.concatenate([[0], np.cumsum(seg)])
        n = max(int(L[-1] / max(radius * spacing, 1e-9)), 1)
        s = np.linspace(0, L[-1], n + 1)
        pts = np.stack([np.interp(s, L, pts[:, k]) for k in range(3)], axis=1)
    if brush == "grab":
        kw.setdefault("offset", (0, 0, 0))
        kw["offset"] = m3.vec3(kw["offset"]) / max(len(pts), 1)
    for p in pts:
        fn(mesh, p, radius, **kw)
    return mesh


@dataclass
class Brush:
    """Reusable brush settings: Brush("draw", radius=0.1, strength=0.02, symmetry="x").apply(mesh, point)."""
    kind: str = "draw"
    radius: float = 0.1
    strength: float = 0.02
    symmetry: str | None = None

    def apply(self, mesh: Mesh, center, **kw) -> Mesh:
        fn = _BRUSHES[self.kind]
        args = dict(radius=self.radius, symmetry=self.symmetry)
        if self.kind != "grab":
            args["strength"] = self.strength
        args.update(kw)
        return fn(mesh, center, **args)

    def stroke(self, mesh: Mesh, path, **kw) -> Mesh:
        args = dict(radius=self.radius, symmetry=self.symmetry)
        if self.kind != "grab":
            args["strength"] = self.strength
        args.update(kw)
        return stroke(mesh, self.kind, path, **args)
