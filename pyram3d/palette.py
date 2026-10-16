"""Palette texturing for pixel-art / stylized games.

`palettize` turns vertex-colored meshes into one small palette texture shared by
all of them: every face gets exactly one palette color (flat "pixel" look, no
blending between colors), and the texture uses nearest-neighbour filtering.

Why: many engine default shaders (e.g. Unity Lit) ignore vertex colors, while a
texture always works. A palette texture also enables palette swaps — recolor the
tiny PNG to make skin/costume variants without touching the model.
"""
from __future__ import annotations

import numpy as np

from . import math3d as m3
from .material import Material, Texture
from .mesh import Mesh


def face_colors(mesh: Mesh) -> np.ndarray:
    """Average vertex color (linear RGBA) of each face."""
    cv, cf, _, sizes = mesh._corners()
    acc = np.zeros((len(mesh.faces), 4))
    np.add.at(acc, cf, mesh.colors[cv])
    return acc / sizes[:, None]


def _kmeans(x: np.ndarray, k: int, iters: int = 20, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uniq = np.unique(np.round(x, 4), axis=0)
    if len(uniq) <= k:
        return uniq
    # k-means++ init
    c = [uniq[rng.integers(len(uniq))]]
    for _ in range(k - 1):
        d = np.min(((x[:, None, :] - np.array(c)[None]) ** 2).sum(-1), axis=1)
        c.append(x[rng.choice(len(x), p=d / d.sum())] if d.sum() > 0 else x[rng.integers(len(x))])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(axis=0)
    return c


def palettize(meshes: list[Mesh], palette=None, max_colors: int = 32, swatch: int = 4,
              name: str = "Palette", roughness: float = 0.8, material: Material | None = None,
              keep_vertex_colors: bool = False, mode: str = "majority") -> tuple[Material, list]:
    """Bake vertex colors of `meshes` into a shared palette texture (flat color per face).

    palette   : optional list of sRGB colors to snap to (exact art-directed palette);
                otherwise up to `max_colors` are chosen automatically (k-means).
    swatch    : pixels per palette entry in the texture.
    mode      : "majority" (crisp details, default) or "average" (smoother color regions).
    Returns (material, palette_srgb_list). Each mesh gets UVs and the material.
    """
    fcs = []
    for m in meshes:
        if m.colors is None:
            m.colors = np.ones((len(m.vertices), 4))
        fcs.append(m3.linear_to_srgb(np.clip(face_colors(m)[:, :3], 0, 1)))
    allc = np.concatenate(fcs)
    if palette is not None:
        pal = np.array([m3.as_color(c)[:3] for c in palette])
    else:
        pal = _kmeans(allc, max_colors)
    n = len(pal)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    # square power-of-two texture (mipmap/compression friendly); unused area stays white
    size = int(2 ** np.ceil(np.log2(max(cols, rows) * swatch)))
    img = np.ones((size, size, 4))
    for i, c in enumerate(pal):
        r, k = divmod(i, cols)
        img[r * swatch:(r + 1) * swatch, k * swatch:(k + 1) * swatch, :3] = c
    # uv of swatch centers (v up, row 0 at the top of the image)
    centers = np.array([[((i % cols) + 0.5) * swatch / size, 1 - ((i // cols) + 0.5) * swatch / size]
                        for i in range(n)])
    tex = Texture(img, name=name, srgb=True, filter="nearest", wrap="clamp")
    mat = material or Material(name, "#ffffff", 0.0, roughness)
    mat.base_color_texture = tex
    w = np.array([0.3, 0.59, 0.11]) ** 0.5
    lum = pal @ np.array([0.3, 0.59, 0.11])
    for m, fc in zip(meshes, fcs):
        if mode == "average":
            # nearest palette color to the average face color
            idx = np.argmin((((fc[:, None, :] - pal[None]) * w) ** 2).sum(-1), axis=1)
        else:
            # snap every vertex, then each face takes its most common corner color (ties: darker wins),
            # which keeps thin painted details (stripes, lips, brows) crisp
            vc = m3.linear_to_srgb(np.clip(m.colors[:, :3], 0, 1))
            vidx = np.argmin((((vc[:, None, :] - pal[None]) * w) ** 2).sum(-1), axis=1)
            idx = np.empty(len(m.faces), dtype=np.int64)
            for fi, f in enumerate(m.faces):
                cnt = np.bincount(vidx[list(f)], minlength=len(pal))
                best = np.flatnonzero(cnt == cnt.max())
                idx[fi] = best[np.argmin(lum[best])]
        m.uvs = centers.copy()
        m.face_uvs = [tuple([int(i)] * len(f)) for i, f in zip(idx, m.faces)]
        m.set_material(mat)
        if not keep_vertex_colors:
            m.colors = None
    hexes = ["#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c) for c in pal]
    return mat, hexes
