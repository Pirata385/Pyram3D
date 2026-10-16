"""Terrain generation, object scattering and water."""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import math3d as m3
from . import noise as nz
from .mesh import Mesh
from .primitives import plane


def terrain(size: float | tuple = 50.0, resolution: int = 128, height: float = 8.0, scale: float = 0.04,
            octaves: int = 6, ridged: float = 0.3, seed: int = 0, island: float = 0.0, terraces: int = 0,
            erosion: int = 0, heightmap: np.ndarray | None = None,
            height_func: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
            colors: dict | bool = True, name: str = "Terrain") -> Mesh:
    """Generate a terrain grid in the XZ plane centered on the origin.

    size      : meters (scalar or (x, z))
    height    : max elevation in meters
    scale     : noise frequency (smaller = broader hills)
    ridged    : 0..1 mix of ridged noise (mountain crests)
    island    : 0..1 radial falloff to make an island / bowl
    terraces  : number of terrace steps (0 = off)
    erosion   : smoothing passes that soften valleys (cheap erosion)
    heightmap : optional (H, W) array in [0, 1] instead of noise
    height_func: optional f(x, z) -> heights (meters), overrides noise
    colors    : True = automatic grass/rock/snow/sand vertex colors; dict to customize
                {"sand": "#..", "grass": "#..", "rock": "#..", "snow": "#..", "water_level": 0.0}
    """
    m = plane(size, resolution, name=name)
    x = m.vertices[:, 0]
    z = m.vertices[:, 2]
    if height_func is not None:
        h = np.asarray(height_func(x, z), dtype=np.float64)
    elif heightmap is not None:
        hm = np.asarray(heightmap, dtype=np.float64)
        sz = np.broadcast_to(np.asarray(size, dtype=np.float64), (2,))
        u = np.clip((x / sz[0] + 0.5) * (hm.shape[1] - 1), 0, hm.shape[1] - 1)
        v = np.clip((0.5 - z / sz[1]) * (hm.shape[0] - 1), 0, hm.shape[0] - 1)
        h = hm[np.round(v).astype(int), np.round(u).astype(int)] * height
    else:
        p = np.stack([x * scale, z * scale, np.full_like(x, seed * 0.123)], axis=1)
        base = 0.5 + 0.5 * nz.fbm3(p, octaves, seed=seed)
        rid = nz.ridged3(p * 0.7, octaves, seed=seed + 5)
        h01 = np.clip(base * (1 - ridged) + rid * ridged, 0, 1)
        if island > 0:
            sz = np.broadcast_to(np.asarray(size, dtype=np.float64), (2,))
            r = np.hypot(x / (sz[0] / 2), z / (sz[1] / 2))
            h01 = h01 * (1 - island * m3.smoothstep(0.4, 1.0, r)) - island * 0.15 * m3.smoothstep(0.7, 1.0, r)
        if terraces > 0:
            k = h01 * terraces
            h01 = (np.floor(k) + m3.smoothstep(0.35, 0.65, k - np.floor(k))) / terraces
        h = h01 * height
    m.vertices[:, 1] = h
    if erosion > 0:
        m.smooth(erosion, 0.5, keep_boundary=False)
    m.smooth_angle = 180.0
    if colors:
        cfg = {"sand": "#c9b27c", "grass": "#5b8c3a", "dark_grass": "#3f6b2a", "rock": "#7a736b",
               "snow": "#f4f6f8", "water_level": 0.0}
        if isinstance(colors, dict):
            cfg.update(colors)
        paint_terrain(m, height, cfg)
        m.extras["terrain_paint"] = (height, cfg)
    return m


def paint_terrain(m: Mesh, height: float, cfg: dict) -> Mesh:
    h = m.vertices[:, 1]
    vn = m.vertex_normals()
    slope = 1.0 - np.clip(vn[:, 1], 0, 1)  # 0 flat .. 1 vertical
    rel = (h - cfg.get("water_level", 0.0)) / max(height, 1e-9)

    def lin(c):
        a = m3.as_color(c)
        a[:3] = m3.srgb_to_linear(a[:3])
        return a

    sand, grass, dgrass, rock, snow = (lin(cfg[k]) for k in ("sand", "grass", "dark_grass", "rock", "snow"))
    n = 0.5 + 0.5 * nz.fbm3(m.vertices * 0.15, 3, seed=7)
    col = grass * (1 - n[:, None]) + dgrass * n[:, None]
    col = _blend(col, sand, 1 - m3.smoothstep(0.02, 0.07, rel))
    col = _blend(col, rock, m3.smoothstep(0.25, 0.45, slope))
    col = _blend(col, snow, m3.smoothstep(0.72, 0.85, rel) * (1 - m3.smoothstep(0.4, 0.6, slope)))
    m.colors = col
    return m


def _blend(a, b, t):
    t = np.asarray(t)[:, None]
    return a * (1 - t) + b * t


def scatter(surface: Mesh, count: int = 50, seed: int = 0, max_slope: float = 30.0, min_height: float | None = None,
            max_height: float | None = None, min_distance: float = 0.0, scale_range=(0.8, 1.2),
            align_to_normal: bool = False, mask: Callable[[np.ndarray], np.ndarray] | None = None) -> list[dict]:
    """Pick random placements on a surface (trees, rocks, grass...).

    Returns a list of dicts {position, rotation (quat), scale} ready for
    scene.add(mesh, position=..., rotation=..., scale=...). Reusing the same
    Mesh object for every placement keeps exported files small (instancing).
    """
    rng = np.random.default_rng(seed)
    tri = surface.copy().triangulate()
    V = tri.vertices
    F = np.array(tri.faces)
    a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    n = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(n, axis=1)
    nrm = m3.normalize(n)
    ok = nrm[:, 1] >= math.cos(math.radians(max_slope))
    prob = np.where(ok, area, 0.0)
    if prob.sum() <= 0:
        return []
    prob /= prob.sum()
    out = []
    tries = 0
    pts = []
    while len(out) < count and tries < count * 30:
        tries += 1
        fi = rng.choice(len(F), p=prob)
        r1, r2 = rng.random(), rng.random()
        if r1 + r2 > 1:
            r1, r2 = 1 - r1, 1 - r2
        p = a[fi] + (b[fi] - a[fi]) * r1 + (c[fi] - a[fi]) * r2
        if min_height is not None and p[1] < min_height:
            continue
        if max_height is not None and p[1] > max_height:
            continue
        if mask is not None and not bool(np.asarray(mask(p[None, :])).reshape(-1)[0]):
            continue
        if min_distance > 0 and pts and np.min(np.linalg.norm(np.array(pts) - p, axis=1)) < min_distance:
            continue
        yaw = m3.quat_from_euler((0, rng.uniform(0, 360), 0))
        rot = m3.quat_mul(m3.quat_between((0, 1, 0), nrm[fi]), yaw) if align_to_normal else yaw
        s = rng.uniform(*scale_range)
        out.append({"position": p, "rotation": rot, "scale": (s, s, s)})
        pts.append(p)
    return out


def water_plane(size: float = 50.0, level: float = 0.0, material=None, name: str = "Water") -> Mesh:
    from .material import presets
    m = plane(size, 1, name=name).translate((0, level, 0))
    m.set_material(material or presets.water())
    return m


def height_at(surface: Mesh, x: float, z: float) -> float:
    """Terrain height (y) under the point (x, z), interpolated from the nearest vertices."""
    d = np.hypot(surface.vertices[:, 0] - x, surface.vertices[:, 2] - z)
    idx = np.argpartition(d, 3)[:3]
    w = 1.0 / np.maximum(d[idx], 1e-6) ** 2
    return float((surface.vertices[idx, 1] * w).sum() / w.sum())


def flatten(surface: Mesh, center, radius: float, height: float | None = None, falloff: float | None = None,
            repaint: bool = True) -> float:
    """Level the terrain around center=(x, z) (building plots, roads, arenas). Returns the plateau height."""
    cx, cz = float(center[0]), float(center[-1])
    h = height_at(surface, cx, cz) if height is None else float(height)
    fall = radius * 0.6 if falloff is None else falloff
    d = np.hypot(surface.vertices[:, 0] - cx, surface.vertices[:, 2] - cz)
    w = 1.0 - m3.smoothstep(radius, radius + fall, d)
    surface.vertices[:, 1] = surface.vertices[:, 1] * (1 - w) + h * w
    if repaint and surface.colors is not None and "terrain_paint" in surface.extras:
        paint_terrain(surface, *surface.extras["terrain_paint"])
    return h
