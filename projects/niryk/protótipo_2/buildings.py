"""Niryk prototype 2: the village of Rondoria (houses, well, market stall, notice board, lamp posts,
crates, barrels, fences, benches, flower beds).

Cozy cartoon medieval for the client's toon shader with ink outlines: stone plinths, timber framing,
warm plaster, chunky stepped roof tiles (terracotta, slate) or a thick thatch, flower boxes,
shutters and chimneys. Every piece is a soft, chunky shape (bevelled boxes and beams with rounded
long edges, low-segment lathes); the color is in the vertex colors (materials are white and only
name the part: Stone, Wood, Plaster, Roof, Glass, Plants, Metal, Cloth, Paper, Produce, Water,
Soil, Glow). Hidden faces (inside walls, under roofs) are dropped to stay in budget.

Each model matches the footprint the game computes its colliders from (STRUCTURES in the game's
src/shared/world/village.ts): origin at the base center, standing on y = 0 (plinths and footings
reach below it for slopes), front (doors, counters, papers, the bench seat) facing +Z.
The client loads them by file name (src/client/world/structureModels.ts): house_a, house_b,
house_c, well, stall, notice_board, lamp_post, crates, barrels, fence, bench, flower_bed.

    python projects/niryk/protótipo_2/buildings.py [name ...] [--no-review]

Writes exports/buildings/<name>.glb and review/buildings/*.png (+ stats.json).
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyram3d").is_dir())))  # the Pyram3D root
sys.path.insert(0, str(HERE))

import pyram3d as p3  # noqa: E402
from pyram3d import math3d as m3  # noqa: E402
from pyram3d.noise import fbm3  # noqa: E402

OUT = HERE / "exports" / "buildings"
REVIEW = HERE / "review" / "buildings"

# footprints of the game's colliders (village.ts STRUCTURES): w (x), d (z), h
FOOTPRINTS = {
    "house_a": (6.2, 5.2, 6.4), "house_b": (7.6, 6.2, 7.4), "house_c": (5.2, 4.6, 5.6),
    "well": (2.4, 2.4, 2.9), "stall": (3.1, 1.9, 2.6), "notice_board": (1.6, 0.45, 2.3),
    "lamp_post": (0.4, 0.4, 3.1), "crates": (1.6, 0.9, 1.3), "barrels": (1.45, 0.9, 1.0),
    "fence": (3.0, 0.25, 1.0), "bench": (1.6, 0.55, 0.9), "flower_bed": (1.5, 1.5, 0.5),
}
ROUND = {"well", "lamp_post", "flower_bed"}
BUDGET = {"house_a": 3000, "house_b": 3000, "house_c": 3000, "well": 1600, "stall": 1600}
SMALL_BUDGET = 600

# ----------------------------------------------------------------------------- palette (sRGB)
STONE = "#a39c93"
STONE_WARM = "#b3a693"
STONE_COOL = "#8f8b8c"
MORTAR = "#7d766f"
TIMBER = "#6a4630"
TIMBER_DARK = "#523523"
WOOD = "#9a6a42"
WOOD_LIGHT = "#b98a58"
IRON = "#3d3a44"
GLASS_DARK = "#2c3b58"
GLASS_LIGHT = "#7d9cc0"
LEAF = "#4f9a42"
LEAF_DARK = "#3c7a37"
FLOWERS = ("#e8506a", "#f2d14a", "#f4f0ea", "#b070e0", "#f08a3a", "#e83e3e")
SOIL = "#5e4632"


def rgb(c) -> np.ndarray:
    return np.asarray(m3.as_color(c)[:3], float) if isinstance(c, str) else np.asarray(c, float)


def mix(a, b, t):
    a, b = rgb(a), rgb(b)
    t = np.asarray(t, float)
    if t.ndim:
        t = t[..., None]
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def noise(P, scale=1.0, seed=0, octaves=2):
    return np.clip(fbm3(np.asarray(P, float) * scale, octaves=octaves, seed=seed) * 1.6, -1, 1)


def unit(v) -> np.ndarray:
    v = np.asarray(v, float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-12, 1.0, n)


def rot(x=0.0, y=0.0, z=0.0) -> np.ndarray:
    """3x3 rotation, Euler degrees applied X, then Y, then Z (the Pyram3D convention)."""
    return m3.quat_to_matrix3(m3.quat_from_euler((x, y, z)))


def frame(z_dir, y_hint=(0, 1, 0)) -> np.ndarray:
    """Rotation whose local +Z is z_dir and local +Y is as close as possible to y_hint."""
    z = unit(z_dir)
    y = np.asarray(y_hint, float)
    y = y - np.dot(y, z) * z
    if np.linalg.norm(y) < 1e-6:
        y = np.array([1.0, 0, 0]) - z[0] * z
    y = unit(y)
    return np.column_stack([np.cross(y, z), y, z])


UP = np.array([0.0, 1.0, 0.0])

# ----------------------------------------------------------------------------- materials

_MATS: dict[str, p3.Material] = {}


def material(name: str) -> p3.Material:
    """White materials (the vertex colors carry the color); Glow (lantern glass) is emissive."""
    if name not in _MATS:
        if name == "Glow":
            _MATS[name] = p3.Material(name, "#ffffff", roughness=1.0, emissive="#ffc45e", emissive_strength=1.0)
        else:
            _MATS[name] = p3.Material(name, "#ffffff", metallic=0.0, roughness=0.85)
    return _MATS[name]


# ----------------------------------------------------------------------------- model (polygon soup)


def newell(P) -> np.ndarray:
    n = np.zeros(3)
    k = len(P)
    for i in range(k):
        a, b = P[i], P[(i + 1) % k]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    return n


class Model:
    """Unwelded polygons (each with its own vertices, normals and sRGB colors) grouped by material.
    Polygons are oriented to agree with their normals, so generators only need correct normals."""

    def __init__(self, name: str, seed: int = 0):
        self.name = name
        self.rng = np.random.default_rng(seed)
        self.polys: list[list] = []  # [mat, P, N, C]

    def add(self, mat, polys, color, R=None, t=(0, 0, 0), jitter=0.0, poly_jitter=0.0, s=1.0):
        """polys: [(P (k,3), N (k,3))] in local space; color: sRGB, (k,3) per vertex, or
        fn(P, N) -> (k,3) in world space. jitter: random tone of the whole piece, poly_jitter: per polygon."""
        R = np.eye(3) if R is None else np.asarray(R, float)
        t = np.asarray(t, float)
        S = np.broadcast_to(np.asarray(s, float), (3,))
        k_piece = 1.0 + self.rng.uniform(-jitter, jitter) if jitter else 1.0
        for P, N in polys:
            P = np.asarray(P, float) * S
            N = unit(np.asarray(N, float) / S)
            Pw = P @ R.T + t
            Nw = unit(N @ R.T)
            nn = newell(Pw)
            if np.linalg.norm(nn) < 1e-12:
                continue
            if np.dot(nn, Nw.sum(0)) < 0:
                Pw, Nw = Pw[::-1].copy(), Nw[::-1].copy()
            if callable(color):
                C = np.asarray(color(Pw, Nw), float)
            else:
                C = np.asarray(color, float) if not isinstance(color, str) else rgb(color)
                C = np.broadcast_to(C, Pw.shape).copy() if C.ndim == 1 else C[::1].copy()
            k = k_piece * (1.0 + self.rng.uniform(-poly_jitter, poly_jitter)) if poly_jitter else k_piece
            self.polys.append([mat, Pw, Nw, C * k])
        return self

    def carve(self, inside, mats=None, margin=0.0):
        """Drop polygons whose vertices are all inside(P) (hidden inside walls, roofs, the ground)."""
        keep = []
        for poly in self.polys:
            if (mats is None or poly[0] in mats) and inside(poly[1]).all():
                continue
            keep.append(poly)
        self.polys = keep

    @property
    def tris(self) -> int:
        return int(sum(len(p[1]) - 2 for p in self.polys))

    def tris_by_mat(self) -> dict:
        out: dict[str, int] = {}
        for p in self.polys:
            out[p[0]] = out.get(p[0], 0) + len(p[1]) - 2
        return dict(sorted(out.items()))

    def mesh(self) -> p3.Mesh:
        mats = sorted({p[0] for p in self.polys})
        V, N, C, F, FM = [], [], [], [], []
        off = 0
        for mi, m in enumerate(mats):
            for _, P, Nn, Cc in (p for p in self.polys if p[0] == m):
                k = len(P)
                V.append(P)
                N.append(Nn)
                # contact shade near the ground and a faint top light (the toon bands do the rest)
                ao = 0.8 + 0.2 * smoothstep(-0.15, 0.75, P[:, 1])
                lift = 1.0 + 0.04 * np.clip(Nn[:, 1], 0, 1)
                C.append(Cc * (ao * lift)[:, None])
                F.append(tuple(range(off, off + k)))
                FM.append(mi)
                off += k
        Cs = np.clip(np.concatenate(C), 0, 1)
        colors = np.concatenate([m3.srgb_to_linear(Cs), np.ones((len(Cs), 1))], axis=1)
        mesh = p3.Mesh(np.concatenate(V), F, name=self.name, materials=[material(m) for m in mats],
                       face_materials=FM, colors=colors, smooth_angle=180.0)
        mesh.normals = unit(np.concatenate(N))
        return mesh


# ----------------------------------------------------------------------------- shape generators
# All return [(P (k,3), N (k,3))] polygons in local space.


def box(size, center=(0, 0, 0), skip=()):
    """Sharp box, flat faces. skip: faces to leave out, e.g. ('-z', '+y')."""
    h = np.asarray(size, float) / 2
    c = np.asarray(center, float)
    polys = []
    for a in range(3):
        for s in (-1, 1):
            if f"{'-+'[s > 0]}{'xyz'[a]}" in skip:
                continue
            b, cc = (a + 1) % 3, (a + 2) % 3
            pts = []
            for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[a], p[b], p[cc] = s * h[a], sb * h[b], sc * h[cc]
                pts.append(p + c)
            n = np.zeros(3)
            n[a] = s
            polys.append((np.array(pts), np.tile(n, (4, 1))))
    return polys


def bevel_box(size, r=0.04, center=(0, 0, 0), skip=()):
    """Box with soft chamfered edges: flat faces, rounded normals across the chamfers (24 verts,
    44 triangles). skip: big faces to leave out ('-y' ...) together with nothing else."""
    h = np.asarray(size, float) / 2
    r = float(min(r, h.min() * 0.48))
    if r < 1e-4:
        return box(size, center, skip)
    inner = h - r
    c = np.asarray(center, float)
    E = np.eye(3)

    def vert(s, a):
        return s * inner + r * s[a] * E[a] + c, s[a] * E[a]

    def poly(vs):
        return np.array([v[0] for v in vs]), np.array([v[1] for v in vs])

    polys = []
    for a in range(3):
        b, cc = (a + 1) % 3, (a + 2) % 3
        for sa in (-1, 1):
            if f"{'-+'[sa > 0]}{'xyz'[a]}" in skip:
                continue
            vs = []
            for sb, sc in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                s = np.zeros(3)
                s[a], s[b], s[cc] = sa, sb, sc
                vs.append(vert(s, a))
            polys.append(poly(vs))
    for cax in range(3):
        a, b = (cax + 1) % 3, (cax + 2) % 3
        for sa in (-1, 1):
            for sb in (-1, 1):
                s0 = np.zeros(3)
                s0[a], s0[b], s0[cax] = sa, sb, -1
                s1 = s0.copy()
                s1[cax] = 1
                polys.append(poly([vert(s0, a), vert(s0, b), vert(s1, b), vert(s1, a)]))
    for s in itertools.product((-1.0, 1.0), repeat=3):
        s = np.array(s)
        polys.append(poly([vert(s, 0), vert(s, 1), vert(s, 2)]))
    return polys


def prism(section, z0, z1, caps=(True, True)):
    """Extrude a closed 2D section [(x, y, nx, ny), ...] along z. Repeat a point with another
    normal to make a hard edge. Caps are flat (convex sections)."""
    S = [(np.array(p[:2], float), unit([p[2], p[3]])) for p in section]
    polys = []
    m = len(S)
    for i in range(m):
        (a, na), (b, nb) = S[i], S[(i + 1) % m]
        if np.allclose(a, b):
            continue
        P = np.array([[a[0], a[1], z0], [b[0], b[1], z0], [b[0], b[1], z1], [a[0], a[1], z1]])
        N = np.array([[na[0], na[1], 0], [nb[0], nb[1], 0], [nb[0], nb[1], 0], [na[0], na[1], 0]])
        polys.append((P, N))
    pts = []
    for a, _ in S:
        if not pts or not np.allclose(a, pts[-1]):
            pts.append(a)
    if np.allclose(pts[0], pts[-1]):
        pts.pop()
    for z, cap, nz in ((z0, caps[0], -1.0), (z1, caps[1], 1.0)):
        if cap:
            P = np.array([[p[0], p[1], z] for p in pts])
            polys.append((P, np.tile([0, 0, nz], (len(P), 1))))
    return polys


def oct_section(w, t, c=None):
    """Rectangle w x t (centered) with chamfered corners, rounded normals."""
    c = min(w, t) * 0.22 if c is None else c
    x, y = w / 2, t / 2
    return [(x, -y + c, 1, 0), (x, y - c, 1, 0), (x - c, y, 0, 1), (-x + c, y, 0, 1),
            (-x, y - c, -1, 0), (-x, -y + c, -1, 0), (-x + c, -y, 0, -1), (x - c, -y, 0, -1)]


def wall_section(w, t, back=0.03, c=None):
    """Beam on a wall: chamfered front (+y, protruding t), flat back embedded `back` in the wall."""
    c = min(w, t) * 0.35 if c is None else c
    x = w / 2
    return [(x, -back, 1, 0), (x, t - c, 1, 0), (x - c, t, 0, 1), (-x + c, t, 0, 1), (-x, t - c, -1, 0),
            (-x, -back, -1, 0), (-x, -back, 0, -1), (x, -back, 0, -1)]


def lathe(profile, segs=12, phase=0.0, y_scale=None):
    """Surface of revolution around +Y. profile [(r, y, nr, ny), ...] bottom to top; r = 0 is a pole;
    repeat a point with another normal for a hard edge."""
    th = phase + np.arange(segs) * (2 * np.pi / segs)
    cs, sn = np.cos(th), np.sin(th)
    polys = []
    pr = [(float(p[0]), float(p[1]), unit([p[2], p[3]])) for p in profile]
    for (r0, y0, n0), (r1, y1, n1) in zip(pr[:-1], pr[1:]):
        if abs(r0 - r1) < 1e-9 and abs(y0 - y1) < 1e-9:
            continue
        for j in range(segs):
            k = (j + 1) % segs
            vs = []
            for r, y, n, idx in ((r0, y0, n0, j), (r0, y0, n0, k), (r1, y1, n1, k), (r1, y1, n1, j)):
                if r < 1e-9 and vs and vs[-1][2] == "pole" and abs(vs[-1][0][1] - y) < 1e-9:
                    continue
                P = np.array([r * cs[idx], y, r * sn[idx]])
                N = unit([n[0] * cs[idx], n[1], n[0] * sn[idx]])
                vs.append((P, N, "pole" if r < 1e-9 else idx))
            # collapse duplicate pole vertices
            out = []
            for v in vs:
                if out and v[2] == "pole" and out[-1][2] == "pole":
                    continue
                out.append(v)
            if len(out) > 1 and out[0][2] == "pole" and out[-1][2] == "pole":
                out.pop()
            if len(out) >= 3:
                polys.append((np.array([v[0] for v in out]), np.array([v[1] for v in out])))
    return polys


def cyl_profile(r, y0, y1, bevel=0.0, top=True, bottom=True, r_top=None):
    """Profile of a (tapered) cylinder with soft bevelled rims and flat caps."""
    rt = r if r_top is None else r_top
    b = bevel
    pr = []
    if bottom:
        pr += [(0, y0, 0, -1), (r - b, y0, 0, -1)]
    pr += [(r, y0 + b, 1, 0) if b else (r, y0, 1, 0)]
    if not b and bottom:
        pr.insert(len(pr) - 1, (r, y0, 0, -1))
    pr += [(rt, y1 - b, 1, 0) if b else (rt, y1, 1, 0)]
    if top:
        if not b:
            pr += [(rt, y1, 0, 1)]
        pr += [(rt - b, y1, 0, 1), (0, y1, 0, 1)] if b else [(0, y1, 0, 1)]
    return pr


def blob(radii, n=2, center=(0, 0, 0), flat=1.0):
    """Smooth ellipsoid from a cube with n x n quads per side (n=1: 12 triangles, n=2: 48).
    flat < 1 squashes the lower half."""
    b = p3.box(2.0, segments=n).weld()
    v = np.tan(np.clip(b.vertices, -1, 1) * (np.pi / 4))
    v = unit(v)
    R = np.asarray(radii, float) * np.ones(3)
    v2 = v.copy()
    v2[:, 1] = np.where(v2[:, 1] < 0, v2[:, 1] * flat, v2[:, 1])
    P = v2 * R + np.asarray(center, float)
    N = unit(v / R)
    return [(P[list(f)], N[list(f)]) for f in b.faces]


def tf(polys, R=None, t=(0, 0, 0), s=1.0):
    R = np.eye(3) if R is None else np.asarray(R, float)
    S = np.broadcast_to(np.asarray(s, float), (3,))
    return [((P * S) @ R.T + t, unit((N / S) @ R.T)) for P, N in polys]


def beam(M: Model, mat, a, b, w, t, out=UP, color=TIMBER, section="oct", back=0.03, jitter=0.06, c=None,
         caps=(True, True), poly_jitter=0.0):
    """A beam from a to b: section w (across) x t (along `out`); 'oct' is centered on the a-b line,
    'wall' sits on a wall whose outward normal is `out` (protrudes t, back embedded)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = np.linalg.norm(b - a)
    R = frame(b - a, out)
    sec = oct_section(w, t, c) if section == "oct" else wall_section(w, t, back, c)
    M.add(mat, prism(sec, 0.0, L, caps), color, R=R, t=a, jitter=jitter, poly_jitter=poly_jitter)


# ----------------------------------------------------------------------------- walls and facades


class Wall:
    """A planar wall: local (s, y, d) -> world center + s*u + y*up + d*n, n = outward normal,
    u = to the right when facing the wall from outside."""

    def __init__(self, center, n):
        self.c = np.asarray(center, float)
        self.n = unit(n)
        self.u = unit(np.cross(UP, self.n))
        self.R = np.column_stack([self.u, UP, self.n])

    def at(self, s, y, d=0.0):
        return self.c + s * self.u + y * UP + d * self.n

    def put(self, M, mat, polys, color, s=0.0, y=0.0, d=0.0, **kw):
        M.add(mat, polys, color, R=self.R, t=self.at(s, y, d), **kw)


def clip_rect(poly, s0, s1, y0, y1):
    """Sutherland-Hodgman: convex polygon [(s, y)] clipped to a rectangle."""
    pts = [np.asarray(p, float) for p in poly]
    for axis, val, greater in ((0, s0, True), (0, s1, False), (1, y0, True), (1, y1, False)):
        out = []
        for i in range(len(pts)):
            a, b = pts[i], pts[(i + 1) % len(pts)]
            ia = a[axis] >= val - 1e-9 if greater else a[axis] <= val + 1e-9
            ib = b[axis] >= val - 1e-9 if greater else b[axis] <= val + 1e-9
            if ia:
                out.append(a)
            if ia != ib:
                t = (val - a[axis]) / (b[axis] - a[axis])
                out.append(a + t * (b - a))
        pts = out
        if len(pts) < 3:
            return []
    clean = []
    for p in pts:
        if not clean or np.linalg.norm(p - clean[-1]) > 1e-6:
            clean.append(p)
    if len(clean) > 2 and np.linalg.norm(clean[0] - clean[-1]) < 1e-6:
        clean.pop()
    return clean if len(clean) >= 3 else []


def grid_cells(s0, s1, y0, y1, cols, rows):
    """Rectangular cells: cols/rows are either counts or explicit cut lists."""
    sc = np.linspace(s0, s1, cols + 1) if np.isscalar(cols) else np.asarray(cols, float)
    yc = np.linspace(y0, y1, rows + 1) if np.isscalar(rows) else np.asarray(rows, float)
    return [(sc[i], sc[i + 1], yc[j], yc[j + 1]) for j in range(len(yc) - 1) for i in range(len(sc) - 1)]


def masonry_cells(s0, s1, y0, y1, course=0.3, block=(0.4, 0.75), rng=None, phase=0.0):
    """Staggered stone courses: [(s0, s1, y0, y1, tone)] covering the rectangle."""
    rng = rng or np.random.default_rng(0)
    n_rows = max(1, int(round((y1 - y0) / course)))
    ys = np.linspace(y0, y1, n_rows + 1)
    cells = []
    for j in range(n_rows):
        s = s0 - rng.uniform(0, block[1]) * (0.5 + 0.5 * ((j + phase) % 2))
        while s < s1:
            e = s + rng.uniform(*block)
            cells.append((max(s, s0), min(e, s1), ys[j], ys[j + 1], rng.uniform(-1, 1)))
            s = e
    return [c for c in cells if c[1] - c[0] > 0.04]


def facade(M: Model, mat, wall: Wall, poly, cells, paint, d=0.0, holes=()):
    """Fill a convex wall polygon [(s, y)] with cells; paint(sy (k,2), cell, P world) -> colors.
    Cells entirely inside a hole (s0, s1, y0, y1) (behind a window or door) are skipped."""
    for cell in cells:
        if any(cell[0] >= h[0] - 1e-6 and cell[1] <= h[1] + 1e-6 and cell[2] >= h[2] - 1e-6 and cell[3] <= h[3] + 1e-6
               for h in holes):
            continue
        q = clip_rect(poly, *cell[:4])
        if not q:
            continue
        q = np.array(q)
        P = wall.c + q[:, :1] * wall.u + q[:, 1:2] * UP + d * wall.n
        N = np.tile(wall.n, (len(P), 1))
        C = paint(q, cell, P)
        M.add(mat, [(P, N)], C)


def stone_paint(base=STONE, warm=STONE_WARM, cool=STONE_COOL, seed=0, contrast=1.0):
    """Stone blocks: a tone per block (warm/cool), lighter top-left, darker bottom-right (a painted
    bevel), so the courses read without mortar geometry."""
    def paint(q, cell, P):
        tone = cell[4] if len(cell) > 4 else 0.0
        c = mix(base, warm if tone > 0 else cool, abs(tone) * 0.55)
        w = max(cell[1] - cell[0], 1e-6)
        h = max(cell[3] - cell[2], 1e-6)
        u = (q[:, 0] - cell[0]) / w
        v = (q[:, 1] - cell[2]) / h
        k = 1.0 + contrast * (0.09 * v - 0.07 * u - 0.05) + 0.02 * noise(P, 3.0, seed)
        return c[None, :] * k[:, None]
    return paint


def plaster_paint(base, eave=None, ground=0.4, seed=0):
    """Warm plaster: a soft dirt band near the plinth, a shade under the eaves, painterly patches."""
    b = rgb(base)
    dirt = b * np.array([0.86, 0.82, 0.76])

    def paint(q, cell, P):
        y = P[:, 1]
        c = mix(dirt, b, smoothstep(ground, ground + 0.55, y))
        if eave is not None:
            c = c * (1.0 - 0.08 * smoothstep(eave - 0.7, eave, y) * (y <= eave + 0.05))[:, None]
        c = c * (1.0 + 0.035 * noise(P, 0.9, seed))[:, None]
        return c
    return paint


# ----------------------------------------------------------------------------- house parts


def plinth(M: Model, w, d, y0, y1, rng, ledge=0.07):
    """Stone plinth (footing) w x d from y0 (below ground) to y1: staggered stone courses on the
    sides and a capstone ring on top (the walls stand inside it)."""
    walls = [Wall((0, 0, d / 2), (0, 0, 1)), Wall((w / 2, 0, 0), (1, 0, 0)), Wall((0, 0, -d / 2), (0, 0, -1)),
             Wall((-w / 2, 0, 0), (-1, 0, 0))]
    paint = stone_paint(seed=3)
    for i, wl in enumerate(walls):
        half = (w if i % 2 == 0 else d) / 2
        poly = [(-half, y0), (half, y0), (half, y1), (-half, y1)]
        cells = [(-half, half, y0, -0.12, -0.4)] + masonry_cells(-half, half, -0.12, y1 - ledge, course=(y1 - ledge + 0.12) / 2,
                                                                  block=(0.45, 0.85), rng=rng, phase=i)
        facade(M, "Stone", wl, poly, cells, paint)
    # capstones: a soft ledge ring around the top
    hw, hd = w / 2 + 0.03, d / 2 + 0.03
    for (a, b) in (((-hw, hd), (hw, hd)), ((hw, -hd), (-hw, -hd))):
        beam(M, "Stone", (a[0] - 0.0, y1 - ledge / 2, a[1] - np.sign(a[1]) * 0.09), (b[0], y1 - ledge / 2, b[1] - np.sign(b[1]) * 0.09),
             0.18 + 0.0, ledge, out=UP, color=mix(STONE, "#c9c1b4", 0.35), jitter=0.03)
    for (a, b) in (((hw, hd - 0.18), (hw, -hd + 0.18)), ((-hw, -hd + 0.18), (-hw, hd - 0.18))):
        beam(M, "Stone", (a[0] - np.sign(a[0]) * 0.09, y1 - ledge / 2, a[1]), (b[0] - np.sign(b[0]) * 0.09, y1 - ledge / 2, b[1]),
             0.18, ledge, out=UP, color=mix(STONE, "#c9c1b4", 0.3), jitter=0.03)


def corner_posts(M, W, D, y0, y1, size=0.2, color=TIMBER):
    for sx, sz in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        beam(M, "Wood", (sx * W / 2, y0, sz * D / 2), (sx * W / 2, y1, sz * D / 2), size, size, out=(sx, 0, sz),
             color=color, c=0.045, caps=(False, False))


def frame_ring(w, h, bar, depth, recess=0.012, bars=None):
    """A window/door frame on a wall (local x right, y up, z out): the opening is x in [-w/2, w/2],
    y in [0, h]; bars (left, right, bottom, top) wide; front at z=depth, reveals back to `recess`."""
    bl, br, bb, bt = bars or (bar, bar, bar, bar)
    O = np.array([[-w / 2 - bl, -bb], [w / 2 + br, -bb], [w / 2 + br, h + bt], [-w / 2 - bl, h + bt]])
    I = np.array([[-w / 2, 0], [w / 2, 0], [w / 2, h], [-w / 2, h]])
    polys = []
    z = depth
    for i in range(4):
        j = (i + 1) % 4
        P = np.array([[*O[i], z], [*O[j], z], [*I[j], z], [*I[i], z]])
        polys.append((P, np.tile([0, 0, 1.0], (4, 1))))
        # outer side
        n_out = unit(np.append(np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])[i], 0.0))
        P = np.array([[*O[i], 0], [*O[j], 0], [*O[j], z], [*O[i], z]])
        polys.append((P, np.tile(n_out, (4, 1))))
        # inner reveal (faces the opening)
        P = np.array([[*I[i], z], [*I[j], z], [*I[j], recess], [*I[i], recess]])
        polys.append((P, np.tile(-n_out, (4, 1))))
    return polys


def plants(M, x0, x1, y, z, depth, rng, R=None, t=(0, 0, 0), flowers=4, hang=True, leaf=LEAF):
    """A mound of leaves with chunky flower puffs (flower boxes, beds)."""
    w = x1 - x0
    parts = []
    M.add("Plants", blob((w / 2, 0.09, depth / 2), 2, ((x0 + x1) / 2, y + 0.03, z), flat=0.5),
          lambda P, N: mix(LEAF_DARK, leaf, smoothstep(y - 0.02, y + 0.1, P[:, 1])), R=R, t=t)
    cols = list(FLOWERS)
    rng.shuffle(cols)
    for i in range(flowers):
        fx = x0 + w * (i + 0.5) / flowers + rng.uniform(-0.04, 0.04)
        fz = z + rng.uniform(-0.3, 0.3) * depth
        r = rng.uniform(0.045, 0.06)
        c = cols[i % len(cols)]
        M.add("Plants", blob((r, r * 0.8, r), 1, (fx, y + 0.1 + rng.uniform(0, 0.03), fz)),
              lambda P, N, c=c: mix(c, "#ffffff", 0.1 * np.clip(N[:, 1], 0, 1)), R=R, t=t)
    if hang:
        for fx in (x0 + w * 0.3,):
            M.add("Plants", blob((0.07, 0.11, 0.05), 1, (fx, y - 0.06, z + depth / 2 + 0.02)),
                  lambda P, N: mix(LEAF_DARK, leaf, 0.4), R=R, t=t)
    return parts


def window(M: Model, wall: Wall, s, y0, w, h, *, frame_col=TIMBER, shutter_col=None, flowers=True,
           muntin=True, sill=True, arch=False, bar=0.09, depth=0.07):
    """A window on a wall: frame with reveals, 2x2 panes (painted glass), muntins, a sill, optional
    open shutters and a flower box. (s, y0): bottom center of the glass on the wall."""
    rng = M.rng
    R, t = wall.R, wall.at(s, y0)
    M.add("Wood", frame_ring(w, h, bar, depth), frame_col, R=R, t=t, jitter=0.05, poly_jitter=0.04)
    # glass: four panes, lighter at the top-left (sky reflection)
    for i, j in itertools.product(range(2), range(2)):
        x0, x1 = -w / 2 + i * w / 2, -w / 2 + (i + 1) * w / 2
        yy0, yy1 = j * h / 2, (j + 1) * h / 2
        P = np.array([[x0, yy0, 0.012], [x1, yy0, 0.012], [x1, yy1, 0.012], [x0, yy1, 0.012]])
        v = (P[:, 1] / h) * 0.8 - (P[:, 0] / w) * 0.35 + 0.1 * j
        M.add("Glass", [(P, np.tile([0, 0, 1.0], (4, 1)))], mix(GLASS_DARK, GLASS_LIGHT, np.clip(v, 0, 1) * 0.75), R=R, t=t)
    if muntin:
        M.add("Wood", box((0.05, h, 0.03), (0, h / 2, 0.03), skip=("-z", "-y", "+y")), frame_col, R=R, t=t)
        M.add("Wood", box((w, 0.05, 0.03), (0, h / 2, 0.03), skip=("-z", "-x", "+x")), frame_col, R=R, t=t)
    if sill:
        sl = w / 2 + bar + 0.06
        beam(M, "Wood", wall.at(s - sl, y0 - bar - 0.035), wall.at(s + sl, y0 - bar - 0.035), 0.07, 0.15, out=wall.n,
             color=mix(frame_col, WOOD_LIGHT, 0.35), section="wall", back=0.0, c=0.025, jitter=0.04)
    if shutter_col is not None:
        sw = w / 2 + 0.03
        for sx in (-1, 1):
            cx = sx * (w / 2 + bar + sw / 2 + 0.015)
            hh = h + 2 * bar - 0.06
            beam(M, "Wood", wall.at(s + cx, y0 + h / 2 - hh / 2), wall.at(s + cx, y0 + h / 2 + hh / 2), sw, 0.05, out=wall.n,
                 color=lambda P, N: rgb(shutter_col) * (0.92 + 0.12 * smoothstep(y0, y0 + h, P[:, 1]))[:, None],
                 section="wall", back=0.0, c=0.018, jitter=0.04)
            # one batten across, darker
            dark = mix(shutter_col, "#20151a", 0.3)
            M.add("Wood", box((sw - 0.05, 0.07, 0.022), (cx, h * 0.5, 0.06), skip=("-z",)), dark, R=R, t=t)
    if flowers:
        bw = w + 2 * bar + 0.02
        by = -bar - 0.07 - 0.2
        beam(M, "Wood", wall.at(s - bw / 2, y0 + by + 0.1), wall.at(s + bw / 2, y0 + by + 0.1), 0.2, 0.24, out=wall.n,
             color=mix(WOOD, TIMBER, 0.2), section="wall", back=0.0, c=0.03, jitter=0.05)
        plants(M, -bw / 2 + 0.03, bw / 2 - 0.03, by + 0.2, 0.13, 0.2, rng, R=R, t=t, flowers=4)


def door(M: Model, wall: Wall, s, y0, w, h, *, color="#5a3a26", frame_col=TIMBER, arch=0.0, step=True,
         frame=True, planks=4, strap_col=IRON):
    """A plank door on a wall; arch > 0 rounds the top (arch = rise of the top / half width)."""
    R, t = wall.R, wall.at(s, y0)
    rise = arch * w / 2
    hs = h - rise  # straight part

    def top(x):
        if rise <= 0:
            return h
        return hs + rise * np.sqrt(max(0.0, 1 - (x / (w / 2)) ** 2))

    xs = np.linspace(-w / 2, w / 2, planks + 1)
    for i in range(planks):
        a, b = xs[i], xs[i + 1]
        n_arc = 3 if rise > 0 else 1
        arc = [(x, top(x)) for x in np.linspace(b, a, n_arc + 1)]
        pts = [(a, 0.0), (b, 0.0)] + arc
        P = np.array([[x, y, 0.025] for x, y in pts])
        tone = 1.0 + (0.06 if i % 2 else -0.04) + M.rng.uniform(-0.03, 0.03)
        M.add("Wood", [(P, np.tile([0, 0, 1.0], (len(P), 1)))],
              lambda P_, N_, tone=tone: rgb(color) * tone * (0.9 + 0.12 * smoothstep(y0, y0 + h, P_[:, 1]))[:, None],
              R=R, t=t)
    # iron straps and a ring handle
    for yy in (0.25 * h, 0.7 * hs):
        M.add("Wood", box((w * 0.86, 0.06, 0.02), (-w * 0.05, yy, 0.035), skip=("-z",)), strap_col, R=R, t=t)
    M.add("Wood", blob((0.04, 0.04, 0.025), 1, (w * 0.32, 0.47 * h, 0.05)), "#c9a24a", R=R, t=t)
    if frame:
        bar = 0.12
        if rise <= 0:
            M.add("Wood", frame_ring(w, h, bar, 0.08, recess=0.025, bars=(bar, bar, 0.0, bar + 0.04)), frame_col,
                  R=R, t=t, jitter=0.05, poly_jitter=0.04)
        else:
            # straight jambs and an arched frame of short beams
            for sx in (-1, 1):
                beam(M, "Wood", wall.at(s + sx * (w / 2 + bar / 2), y0), wall.at(s + sx * (w / 2 + bar / 2), y0 + hs),
                     bar, 0.08, out=wall.n, color=frame_col, section="wall", back=0.0)
            n = 5
            angs = np.linspace(np.pi, 0, n + 1)
            rx, ry = w / 2 + bar / 2, rise + bar / 2
            for k in range(n):
                p0 = (rx * np.cos(angs[k]), hs + ry * np.sin(angs[k]))
                p1 = (rx * np.cos(angs[k + 1]), hs + ry * np.sin(angs[k + 1]))
                ext = 0.03
                d = np.array(p1) - np.array(p0)
                d = d / np.linalg.norm(d) * ext
                beam(M, "Wood", wall.at(s + p0[0] - d[0], y0 + p0[1] - d[1]), wall.at(s + p1[0] + d[0], y0 + p1[1] + d[1]),
                     bar, 0.08, out=wall.n, color=frame_col, section="wall", back=0.0, jitter=0.04)
    if step:
        M.add("Stone", bevel_box((w + 0.5, 0.2, 0.42), 0.04, (0, -y0 + 0.08 + 0.0, 0.2)), mix(STONE, "#c9c1b4", 0.25),
              R=R, t=t, jitter=0.03)


def roof_slopes(M: Model, *, W, D, ye, yr, o, ov, courses, th, lip, cols, palette, style="tile", soffit=TIMBER_DARK,
                seed=0, ridge_r=None):
    """A gable roof with the ridge along X: two slopes of stepped courses (each course a slab whose
    lower lip stands proud: chunky tiles, slates or thatch layers), a dark soffit and a ridge roll.
    ye: roof underside at the walls' outer faces (z = +-D/2), yr: underside at the ridge (z = 0),
    o: eave overhang (horizontal), ov: verge overhang beyond the gable walls.
    Returns roof_top(z) (the top surface height) for chimneys."""
    rng = np.random.default_rng(seed)
    X0, X1 = -W / 2 - ov, W / 2 + ov
    pal = [rgb(c) for c in palette]  # dark, mid, light
    for sgn in (1.0, -1.0):
        d = unit([0.0, ye - yr, sgn * D / 2])
        n = unit(np.cross(d, [sgn, 0.0, 0.0]))
        if n[1] < 0:
            n = -n
        L_wall = np.hypot(ye - yr, D / 2)
        S = L_wall * (1 + o / (D / 2))
        R0 = np.array([0.0, yr, 0.0])
        sk = np.linspace(0, S, courses + 1)

        def P(x, s, h):
            return R0 + s * d + h * n + np.array([x, 0.0, 0.0])

        for k in range(courses):
            s0, s1 = sk[k], sk[k + 1]
            last = k == courses - 1
            h0, h1 = th, th + lip * (1.25 if last and style == "thatch" else 1.0)
            tvec = unit(d * (s1 - s0) + n * (h1 - h0))
            ntop = unit(np.cross(tvec, [sgn, 0.0, 0.0]))
            if np.dot(ntop, n) < 0:
                ntop = -ntop
            nlip = unit(ntop + d * (2.2 if style == "thatch" else 0.9))
            # top face split into staggered patches along X
            n_cols = cols + (k % 2)
            xs = np.linspace(X0, X1, n_cols + 1)
            if k % 2:
                xs[1:-1] -= (X1 - X0) / n_cols / 2
            for j in range(n_cols):
                xa, xb = xs[j], xs[j + 1]
                Pq = np.array([P(xa, s0, h0), P(xb, s0, h0), P(xb, s1, h1), P(xa, s1, h1)])
                Nq = np.array([ntop, ntop, nlip, nlip])
                tone = rng.uniform(-1, 1)
                base = mix(pal[1], pal[2] if tone > 0 else pal[0], abs(tone) * 0.45)
                if style == "thatch":
                    # straw streaks: alternating column tones and a darker, combed top of each layer
                    base = mix(pal[1], pal[2] if tone > 0 else pal[0], abs(tone) * 0.25) * (1.0 + (0.05 if j % 2 else -0.04))
                grad = {"tile": [0.82, 0.82, 1.06, 1.06], "slate": [0.88, 0.88, 1.05, 1.05],
                        "thatch": [0.74, 0.74, 1.08, 1.08]}[style]
                grad = np.array(grad)
                M.add("Roof", [(Pq, Nq)], base[None, :] * grad[:, None])
            # the lip (riser) and, on the last course, the eave fascia down to the soffit
            hb = 0.0 if last else h0
            Pq = np.array([P(X0, s1, h1), P(X1, s1, h1), P(X1, s1, hb), P(X0, s1, hb)])
            Nq = np.array([nlip, nlip, d, d])
            M.add("Roof", [(Pq, Nq)], pal[0] * np.array([1.0, 1.0, 0.8, 0.8])[:, None] * 0.95)
            # gable-end caps of this course (verge)
            for xe, sx in ((X0, -1.0), (X1, 1.0)):
                Pq = np.array([P(xe, s0, 0.0), P(xe, s1, 0.0), P(xe, s1, h1), P(xe, s0, h0)])
                M.add("Roof", [(Pq, np.tile([sx, 0, 0], (4, 1)))], pal[0] * 0.92)
        # soffit
        Pq = np.array([P(X0, 0, 0), P(X1, 0, 0), P(X1, S, 0), P(X0, S, 0)])
        M.add("Wood", [(Pq, np.tile(-n, (4, 1)))], soffit)
    # ridge roll
    slope_n = unit(np.cross(unit([0.0, ye - yr, D / 2]), [1.0, 0, 0]))
    if slope_n[1] < 0:
        slope_n = -slope_n
    y_top = yr + th * slope_n[1] / 1.0
    rr = ridge_r or 0.13
    prof = cyl_profile(rr, X0 - 0.02, X1 + 0.02, bevel=rr * 0.5)
    ridge = tf(lathe(prof, 8, phase=np.pi / 8), R=rot(z=-90), t=(0, y_top + rr * 0.35, 0))
    M.add("Roof", ridge, lambda P_, N_: mix(pal[0], pal[1], 0.55 + 0.35 * np.clip(N_[:, 1], 0, 1)))

    pitch_dy = (yr - ye) / (D / 2)

    def roof_top(z):
        return yr + th / max(slope_n[1], 1e-3) - np.abs(z) * pitch_dy

    return roof_top


def chimney(M: Model, x, z, y0, y1, w=0.62, d=0.62, rng=None, pot=True):
    """Stone chimney stack with a capstone and a terracotta pot."""
    rng = rng or M.rng
    paint = stone_paint(seed=9)
    for i, (n, half) in enumerate((((0, 0, 1), w / 2), ((1, 0, 0), d / 2), ((0, 0, -1), w / 2), ((-1, 0, 0), d / 2))):
        n = np.array(n, float)
        off = d / 2 if i % 2 == 0 else w / 2
        wl = Wall((x + n[0] * off, 0, z + n[2] * off), n)
        poly = [(-half, y0), (half, y0), (half, y1), (-half, y1)]
        cells = masonry_cells(-half, half, y0, y1, course=0.34, block=(0.3, 0.55), rng=rng, phase=i)
        facade(M, "Stone", wl, poly, cells, paint)
    M.add("Stone", bevel_box((w + 0.14, 0.14, d + 0.14), 0.035, (x, y1 + 0.07, z)), mix(STONE, "#c9c1b4", 0.3))
    if pot:
        prof = [(0.0, 0.0, 0, -1), (0.13, 0.0, 0, -1), (0.13, 0.0, 1, 0.2), (0.11, 0.26, 1, 0.1), (0.14, 0.3, 0.6, 0.8),
                (0.14, 0.34, 0, 1), (0.08, 0.34, 0, 1), (0.08, 0.2, 0, 1), (0.0, 0.2, 0, 1)]
        M.add("Stone", lathe(prof, 6), lambda P_, N_: np.where((N_[:, 1] > 0.95)[:, None] & (P_[:, 1] < y1 + 0.14 + 0.33)[:, None],
                                                          rgb("#2a1e1c"), rgb("#b9654a")),
              t=(x + 0.1, y1 + 0.14, z))


def stone_patch(M, wl: Wall, s, y, rx=0.36, ry=0.24, rng=None):
    """A patch where the plaster fell off and the stones show."""
    rng = rng or M.rng
    a = np.linspace(0, 2 * np.pi, 9)[:-1] + rng.uniform(-0.2, 0.2, 8)
    k = rng.uniform(0.8, 1.15, 8)
    poly = [(s + rx * kk * np.cos(t), y + ry * kk * np.sin(t)) for t, kk in zip(a, k)]
    cells = masonry_cells(s - rx * 1.2, s + rx * 1.2, y - ry * 1.2, y + ry * 1.2, course=ry * 0.8, block=(0.16, 0.3), rng=rng)
    facade(M, "Stone", wl, poly, cells, stone_paint(base="#b4ab9e", seed=7, contrast=1.3), d=0.006)


def woodpile(M, wl: Wall, s0, s1, rows=3, r=0.075, depth=0.36, d0=0.12, rng=None):
    """Split logs stacked against a wall, cut ends facing out."""
    rng = rng or M.rng
    sec = [(r * np.cos(t), r * np.sin(t), np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi / 6]
    log = prism(sec, 0.0, depth, caps=(False, True))
    for j in range(rows):
        n = int((s1 - s0 - (j % 2) * r) / (2 * r))
        for i in range(n):
            ss = s0 + r + (j % 2) * r + i * 2 * r + rng.uniform(-0.01, 0.01)
            yy = r + j * r * 1.75
            dd = d0 + rng.uniform(-0.03, 0.03)
            bark = mix("#6b4a33", "#83603f", rng.uniform())
            cut = mix("#d9b47c", "#e8c995", rng.uniform())
            wl.put(M, "Wood", tf(log, R=rot(z=rng.uniform(0, 60))),
                   lambda P_, N_, bark=bark, cut=cut: np.where((np.abs(N_ @ wl.n) > 0.9)[:, None], cut, bark),
                   s=ss, y=yy, d=dd)


def climber(M, wl: Wall, s, y0, y1, rng=None, flowers=("#e85a7a", "#f08aa0", "#d8405e"), lean=0.25):
    """A climbing rose on a wall: leafy clumps up a gentle S and a few blooms."""
    rng = rng or M.rng
    n = 5
    for i in range(n):
        t = i / (n - 1)
        ss = s + lean * np.sin(t * 2.6) + rng.uniform(-0.05, 0.05)
        yy = y0 + (y1 - y0) * t
        rr = 0.24 - 0.08 * t
        wl.put(M, "Plants", blob((rr, rr * 0.9, 0.1), 2, (0, 0, 0)),
               lambda P_, N_: mix(LEAF_DARK, LEAF, smoothstep(-0.5, 0.8, N_[:, 1] + 0.3 * (N_ @ wl.n))), s=ss, y=yy, d=0.08)
        for k in range(2 if i < n - 1 else 1):
            fr_ = rng.uniform(0.045, 0.06)
            c = flowers[(i + k) % len(flowers)]
            wl.put(M, "Plants", blob((fr_, fr_, fr_ * 0.8), 1, (0, 0, 0)), c,
                   s=ss + rng.uniform(-rr, rr) * 0.7, y=yy + rng.uniform(-rr, rr) * 0.6, d=0.17)


def house_walls(W, D):
    return {"front": Wall((0, 0, D / 2), (0, 0, 1)), "back": Wall((0, 0, -D / 2), (0, 0, -1)),
            "right": Wall((W / 2, 0, 0), (1, 0, 0)), "left": Wall((-W / 2, 0, 0), (-1, 0, 0))}


def plaster_walls(M, walls, W, D, y0, ye, yr, color, holes, seed=0, gable=True):
    """Plaster on the four walls (front/back rectangles, gable pentagons on the sides)."""
    paint = plaster_paint(color, eave=ye, ground=y0, seed=seed)
    rows = [y0, y0 + 0.3, ye - 0.6, ye + 0.05, ye + 0.9, ye + 1.8, yr + 0.3]
    for key, wl in walls.items():
        half = (W if key in ("front", "back") else D) / 2
        if key in ("front", "back") or not gable:
            poly = [(-half, y0), (half, y0), (half, ye + 0.04), (-half, ye + 0.04)]
        else:
            poly = [(-half, y0), (half, y0), (half, ye), (0, yr + 0.03), (-half, ye)]
        cols = max(2, int(round(2 * half / 1.3)))
        cells = grid_cells(-half, half, y0, rows[-1], cols, [r for r in rows if r <= rows[-1]])
        facade(M, "Plaster", wl, poly, cells, paint, holes=holes.get(key, ()))


def wall_beam(M, wl: Wall, s0, y0, s1, y1, w=0.15, t=0.05, color=TIMBER, caps=False):
    """A timber on a wall. Ends are open by default: they butt into posts, plates or the roof."""
    beam(M, "Wood", wl.at(s0, y0), wl.at(s1, y1), w, t, out=wl.n, color=color, section="wall", jitter=0.07,
         caps=(caps, caps))


def brace_pair(M, wl, s_a, s_b, y0, y1, w=0.13, color=TIMBER):
    """An X (St Andrew's cross) between two posts."""
    wall_beam(M, wl, s_a, y0, s_b, y1, w, color=color)
    wall_beam(M, wl, s_b, y0, s_a, y1, w, color=color)


def interior(W, D, y_floor, roof_top, ov=0.0, o=0.0, yr=None, ye=None, e=0.012):
    """inside(P): strictly inside the walls under the roof top, or inside the roof slab."""
    def inside(P):
        x, y, z = P[:, 0], P[:, 1], P[:, 2]
        top = roof_top(z)
        house = (np.abs(x) < W / 2 - e) & (np.abs(z) < D / 2 - e) & (y > y_floor - 2) & (y < top - e)
        if yr is not None:
            under = yr - np.abs(z) * (yr - ye) / (D / 2)
            slab = (np.abs(x) < W / 2 + ov - e) & (np.abs(z) < D / 2 + o - e) & (y > under + e) & (y < top - e)
            house |= slab
        return house
    return inside


# ----------------------------------------------------------------------------- houses


def house_a():
    """Timber-framed cottage (footprint 6.2 x 5.2): stone plinth, cream plaster with dark Fachwerk
    (St Andrew's crosses under the windows), centered plank door under a little hood, two windows
    with blue shutters and flower boxes, terracotta tile roof, chimney, a round gable window."""
    M = Model("house_a", seed=11)
    rng = M.rng
    W, D = 5.7, 4.7
    y0, yp = -0.6, 0.42
    ye, yr = 3.25, 5.5
    o, ov = 0.42, 0.32
    plinth(M, W + 0.24, D + 0.24, y0, yp, rng)
    walls = house_walls(W, D)
    fr, bk, rt, lt = walls["front"], walls["back"], walls["right"], walls["left"]
    win_w, win_h, win_y = 0.86, 0.98, 1.35
    door_w, door_h = 1.0, 2.05
    holes = {"front": [(-door_w / 2, door_w / 2, yp, yp + door_h)] + [(sx * 1.75 - win_w / 2, sx * 1.75 + win_w / 2, win_y, win_y + win_h) for sx in (-1, 1)],
             "back": [(0.9 - win_w / 2, 0.9 + win_w / 2, win_y, win_y + win_h)],
             "right": [(-0.43, 0.43, win_y, win_y + win_h)], "left": [(-0.43, 0.43, win_y, win_y + win_h)]}
    plaster_walls(M, walls, W, D, yp, ye, yr, "#efe1c4", holes, seed=1)
    corner_posts(M, W, D, yp, ye + 0.02)
    # Fachwerk: sill, top plate, posts at the door, rails at the window sills and heads, crosses
    for wl, half in ((fr, W / 2), (bk, W / 2)):
        wall_beam(M, wl, -half, yp + 0.08, half, yp + 0.08, 0.16)
        wall_beam(M, wl, -half, ye - 0.09, half, ye - 0.09, 0.18)
    for wl, half in ((rt, D / 2), (lt, D / 2)):
        wall_beam(M, wl, -half, ye - 0.09, half, ye - 0.09, 0.18)
    for sx in (-1, 1):
        wall_beam(M, fr, sx * (door_w / 2 + 0.2), yp + 0.16, sx * (door_w / 2 + 0.2), ye - 0.18, 0.15)
        x_in, x_out = sx * (door_w / 2 + 0.28), sx * (W / 2 - 0.1)
        wall_beam(M, fr, x_in, win_y - 0.12, x_out, win_y - 0.12, 0.13)
        wall_beam(M, fr, x_in, win_y + win_h + 0.12, x_out, win_y + win_h + 0.12, 0.13)
        lo, hi = sorted((sx * 1.75 - 0.5, sx * 1.75 + 0.5))
        brace_pair(M, fr, lo, hi, yp + 0.16, win_y - 0.18)
    # back wall: a post and one cross; gables: king post and struts under the ridge
    wall_beam(M, bk, -0.2, yp + 0.16, -0.2, ye - 0.18, 0.15)
    brace_pair(M, bk, -2.5, -0.35, yp + 0.16, ye - 0.18, 0.13)
    for wl in (rt, lt):
        wall_beam(M, wl, -D / 2 + 0.1, ye + 0.02, -0.3, yr - 0.62, 0.12)
        wall_beam(M, wl, D / 2 - 0.1, ye + 0.02, 0.3, yr - 0.62, 0.12)
        wall_beam(M, wl, -D / 2 + 0.1, win_y - 0.12, D / 2 - 0.1, win_y - 0.12, 0.12)
    # openings
    door(M, fr, 0, yp, door_w, door_h, color="#5a3a26")
    for sx in (-1, 1):
        window(M, fr, sx * 1.75, win_y, win_w, win_h, shutter_col="#3f6a8a")
    window(M, bk, 0.9, win_y, win_w, win_h, shutter_col="#3f6a8a", flowers=False)
    window(M, rt, 0, win_y, win_w, win_h, flowers=False)
    window(M, lt, 0, win_y, win_w, win_h, flowers=False)
    # a round attic window on each gable
    for wl in (rt, lt):
        ring = lathe([(0.28, 0.0, 1, 0), (0.28, 0.07, 1, 0), (0.28, 0.07, 0, 1), (0.17, 0.07, 0, 1),
                      (0.17, 0.07, -1, 0), (0.17, 0.0, -1, 0)], 8, phase=np.pi / 8)
        wl.put(M, "Wood", tf(ring, R=rot(x=90)), TIMBER, s=0, y=ye + 1.0, d=0.0)
        disc = [(np.array([[0.17 * np.cos(a), 0.17 * np.sin(a), 0.01] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False) + np.pi / 8]),
                 np.tile([0, 0, 1.0], (8, 1)))]
        wl.put(M, "Glass", disc, lambda P_, N_: mix(GLASS_DARK, GLASS_LIGHT, 0.45), s=0, y=ye + 1.0)
    roof_top = roof_slopes(M, W=W, D=D, ye=ye, yr=yr, o=o, ov=ov, courses=6, th=0.12, lip=0.07, cols=12,
                           palette=("#8f3c2c", "#b4533b", "#cf6e4a"), seed=5)
    chimney(M, 1.55, -1.15, ye, 5.85, rng=rng)
    stone_patch(M, bk, -1.3, 2.3)
    stone_patch(M, fr, -2.3, 2.62, rx=0.22, ry=0.16)
    woodpile(M, rt, -1.95, -0.62, rows=3)
    M.carve(interior(W, D, yp, roof_top, ov, o, yr, ye), mats=("Stone", "Wood", "Plaster"))
    for p in M.polys:  # the walls sit a little left of center: the woodpile takes the right side
        p[1] = p[1] + np.array([-0.14, 0, 0])
    return M


def house_b():
    """Two-story house (footprint 7.6 x 6.2): stone ground floor with an arched door (left of
    center) and a big window, jettied timber-framed upper floor in peach plaster with red
    shutters and flower boxes, blue slate roof with a chimney, a hanging sign."""
    M = Model("house_b", seed=23)
    rng = M.rng
    W, D = 7.1, 5.7
    y0, yp = -0.6, 0.35
    y1 = 3.0  # first floor line (the jetty)
    jet = 0.16
    Wu, Du = W + 2 * jet, D + 2 * jet
    ye, yr = 5.35, 7.05
    o, ov = 0.45, 0.3
    low = house_walls(W, D)
    up = house_walls(Wu, Du)
    door_x, door_w, door_h = -1.25, 1.1, 2.15
    win_w, win_h = 0.95, 1.05
    holes_low = {"front": [(door_x - door_w / 2, door_x + door_w / 2, yp, yp + door_h - 0.55),
                           (1.6 - 0.75, 1.6 + 0.75, 1.15, 1.15 + 1.05)]}
    # ground floor: stone masonry
    paint = stone_paint(seed=4, contrast=1.1)
    for key, wl in low.items():
        half = (W if key in ("front", "back") else D) / 2
        poly = [(-half, y0), (half, y0), (half, y1), (-half, y1)]
        cells = [(-half, half, y0, -0.1, -0.5)] + masonry_cells(-half, half, -0.1, y1, course=0.52, block=(0.85, 1.5), rng=rng,
                                                                 phase=len(key))
        # quoins: the corner stones of every course are big and light
        cells = [c[:4] + ((1.6,) if (c[0] <= -half + 1e-6 or c[1] >= half - 1e-6) else c[4:]) for c in cells]
        facade(M, "Stone", wl, poly, cells, paint, holes=holes_low.get(key, ()))
    # upper floor: plaster + timber, jettied
    holes_up = {"front": [(sx - win_w / 2, sx + win_w / 2, y1 + 0.75, y1 + 0.75 + win_h) for sx in (-2.0, 0.0, 2.0)],
                "back": [(-0.5, 0.5, y1 + 0.75, y1 + 0.75 + win_h)],
                "right": [(-0.48, 0.48, y1 + 0.75, y1 + 0.75 + win_h)], "left": [(-0.48, 0.48, y1 + 0.75, y1 + 0.75 + win_h)]}
    plaster_walls(M, up, Wu, Du, y1, ye, yr, "#ecd3ae", holes_up, seed=2)
    # jetty: a heavy floor beam ring with joist ends under it
    for key, wl in up.items():
        half = (Wu if key in ("front", "back") else Du) / 2
        wall_beam(M, wl, -half, y1 + 0.1, half, y1 + 0.1, 0.24, 0.07)
        wall_beam(M, wl, -half, ye - 0.09, half, ye - 0.09, 0.18)
    for key in ("front", "back", "right", "left"):
        wl, half = low[key], (W if key in ("front", "back") else D) / 2
        # jetty underside (between the stone wall and the upper wall)
        Pq = np.array([wl.at(-half - jet, y1 - 0.02, jet + 0.0), wl.at(half + jet, y1 - 0.02, jet), wl.at(half, y1 - 0.02, 0.0),
                       wl.at(-half, y1 - 0.02, 0.0)])
        M.add("Wood", [(Pq, np.tile(-UP, (4, 1)))], TIMBER_DARK)
        if key == "front":
            for sx in np.linspace(-half + 0.4, half - 0.4, 6):
                wl.put(M, "Wood", box((0.13, 0.13, jet + 0.06), (0, 0, (jet + 0.06) / 2), skip=("-z",)), TIMBER_DARK,
                       s=sx, y=y1 - 0.08, d=0.0, jitter=0.05)
    corner_posts(M, Wu, Du, y1 + 0.2, ye + 0.02)
    fr = up["front"]
    for sx in (-1.0, 1.0):
        wall_beam(M, fr, sx * 1.0, y1 + 0.22, sx * 1.0, ye - 0.18, 0.15)
        wall_beam(M, fr, sx * 3.0, y1 + 0.22, sx * 3.0, ye - 0.18, 0.15)
        wall_beam(M, fr, sx * 1.08, y1 + 0.62, sx * 2.92, y1 + 0.62, 0.12)
        brace_pair(M, fr, sx * 3.08 if sx > 0 else -3.82, sx * 3.82 if sx > 0 else -3.08, y1 + 0.22, ye - 0.18, 0.12)
    for key in ("back", "right", "left"):
        wl = up[key]
        half = (Wu if key == "back" else Du) / 2
        for sx in (-1, 1):
            wall_beam(M, wl, sx * 0.75, y1 + 0.22, sx * 0.75, ye - 0.18, 0.15)
            if key != "back":
                wall_beam(M, wl, sx * 0.83, y1 + 0.22, sx * (half - 0.15), ye - 0.18, 0.12)
        if key != "back":
            wall_beam(M, wl, -half + 0.1, ye + 0.02, -0.12, yr - 0.45, 0.12)
            wall_beam(M, wl, half - 0.1, ye + 0.02, 0.12, yr - 0.45, 0.12)
    # ground floor openings: arched door in a stone surround, a wide window
    lf = low["front"]
    door(M, lf, door_x, yp, door_w, door_h, color="#6b3b2a", arch=0.9, frame=False)
    n = 7
    angs = np.linspace(np.pi, 0, n + 1)
    rise = 0.9 * door_w / 2
    hs = door_h - rise
    for k in range(n):
        a0, a1 = angs[k], angs[k + 1]
        am = (a0 + a1) / 2
        r_in, r_out = door_w / 2 + 0.01, door_w / 2 + 0.24
        pts = [(r_in * np.cos(a0), hs + rise / (door_w / 2) * r_in * np.sin(a0)),
               (r_out * np.cos(a0), hs + rise / (door_w / 2) * r_out * np.sin(a0)),
               (r_out * np.cos(a1), hs + rise / (door_w / 2) * r_out * np.sin(a1)),
               (r_in * np.cos(a1), hs + rise / (door_w / 2) * r_in * np.sin(a1))]
        sec_front = np.array([[x, y, 0.07] for x, y in pts])
        sec_back = np.array([[x, y, 0.0] for x, y in pts])
        polys = [(sec_front, np.tile([0, 0, 1.0], (4, 1)))]
        for i in range(4):
            j = (i + 1) % 4
            P = np.array([sec_back[i], sec_back[j], sec_front[j], sec_front[i]])
            mid = (P[0] + P[1]) / 2 - np.array([0, hs, 0])
            nrm = unit(np.cross(P[1] - P[0], [0, 0, 1.0]))
            polys.append((P, np.tile(nrm, (4, 1))))
        tone = mix(STONE, "#cfc7b9", 0.25 + 0.2 * (k % 2))
        lf.put(M, "Stone", polys, tone, s=door_x, y=yp, jitter=0.03)
    for sx in (-1, 1):
        for k in range(3):
            yy = yp + 0.25 + k * 0.42
            M.add("Stone", box((0.24, 0.38, 0.07), (0, 0, 0), skip=("-z",)), mix(STONE, "#cfc7b9", 0.2 + 0.15 * (k % 2)),
                  R=lf.R, t=lf.at(door_x + sx * (door_w / 2 + 0.13), yy + 0.0, 0.035), jitter=0.04)
    window(M, lf, 1.6, 1.15, 1.5, 1.05, shutter_col=None, flowers=True, frame_col=TIMBER_DARK)
    for sx in (-2.0, 0.0, 2.0):
        window(M, fr, sx, y1 + 0.75, win_w, win_h, shutter_col="#8a3b36", flowers=sx != 0.0)
    window(M, up["back"], 0, y1 + 0.75, win_w, win_h, shutter_col="#8a3b36", flowers=False)
    window(M, up["right"], 0, y1 + 0.75, win_w, win_h, flowers=False)
    window(M, up["left"], 0, y1 + 0.75, win_w, win_h, flowers=False)
    window(M, low["right"], 1.2, 1.1, 0.75, 0.85, flowers=False, muntin=True, frame_col=TIMBER_DARK)
    # hanging sign (a mug: the village tavern) on an iron bracket by the door
    sign_s = door_x - door_w / 2 - 0.55
    a = lf.at(sign_s, 2.75, 0.0)
    b = lf.at(sign_s, 2.75, 0.85)
    beam(M, "Wood", a, b, 0.06, 0.06, out=UP, color=IRON)
    beam(M, "Wood", lf.at(sign_s, 2.35, 0.0), lf.at(sign_s, 2.72, 0.6), 0.05, 0.05, out=UP, color=IRON)
    board = bevel_box((0.06, 0.48, 0.62), 0.02, (0, 0, 0))
    M.add("Wood", board, lambda P_, N_: mix("#7a4a2c", "#a8743f", smoothstep(-1, 1, P_[:, 1] - 2.2)),
          t=lf.at(sign_s, 2.38, 0.5), R=lf.R @ rot(y=90))
    # the emblem: a mug of ale with foam (both faces)
    for dz in (0.035, -0.035):
        sk = ("-z",) if dz > 0 else ("+z",)
        c = lf.at(sign_s, 2.36, 0.5 + dz)
        M.add("Wood", box((0.17, 0.2, 0.02), (0, 0, 0), skip=sk), "#e0a83a", t=c)
        M.add("Wood", box((0.2, 0.07, 0.025), (0, 0.12, 0), skip=sk), "#f6efe0", t=c)
        M.add("Wood", box((0.06, 0.13, 0.02), (0.12, 0.0, 0), skip=sk), "#c58a2c", t=c)
    roof_top = roof_slopes(M, W=Wu, D=Du, ye=ye, yr=yr, o=o, ov=ov, courses=7, th=0.12, lip=0.055, cols=8,
                           palette=("#45507a", "#5f6f9c", "#7f8fb8"), style="slate", seed=8)
    chimney(M, -2.2, -1.3, ye, 7.15, w=0.7, d=0.7, rng=rng, pot=False)
    M.add("Stone", box((0.36, 0.02, 0.36), (-2.2, 7.15 + 0.15, -1.3)), "#2a1e1c")
    upper = interior(Wu, Du, yp, roof_top, ov, o, yr, ye)
    e = 0.012

    def inside(P):
        lower = (np.abs(P[:, 0]) < W / 2 - e) & (np.abs(P[:, 2]) < D / 2 - e) & (P[:, 1] < y1 + 0.3)
        return lower | (upper(P) & (P[:, 1] > y1 - 0.03))
    M.carve(inside, mats=("Stone", "Wood", "Plaster"))
    return M


def house_c():
    """Small thatched cottage (footprint 5.2 x 4.6): low whitewashed walls on a stone plinth, a
    rounded plank door (right of center) under a thick thatch with rolled eaves, a stone chimney
    stack against the left gable, green shutters and a flower box."""
    M = Model("house_c", seed=37)
    rng = M.rng
    xo = 0.25  # the walls sit right of center: the chimney stack takes the left 0.5 m
    W, D = 4.5, 4.1
    y0, yp = -0.6, 0.38
    ye, yr = 2.7, 4.6
    o, ov = 0.45, 0.3
    door_x, door_w, door_h = 0.85 - xo, 0.95, 1.95
    win_w, win_h, win_y = 0.85, 0.85, 1.25
    walls = house_walls(W, D)
    fr, bk, rt, lt = walls["front"], walls["back"], walls["right"], walls["left"]
    holes = {"front": [(door_x - door_w / 2, door_x + door_w / 2, yp, yp + door_h - 0.5), (-1.3 - xo - win_w / 2, -1.3 - xo + win_w / 2, win_y, win_y + win_h)],
             "back": [(-0.4, 0.4, win_y, win_y + win_h)], "right": [(-0.4, 0.4, win_y, win_y + win_h)]}
    plinth(M, W + 0.24, D + 0.24, y0, yp, rng)
    plaster_walls(M, walls, W, D, yp, ye, yr, "#f4ecda", holes, seed=3)
    corner_posts(M, W, D, yp, ye + 0.02, size=0.18, color=TIMBER)
    for wl, half in ((fr, W / 2), (bk, W / 2), (rt, D / 2), (lt, D / 2)):
        wall_beam(M, wl, -half, ye - 0.08, half, ye - 0.08, 0.16)
    door(M, fr, door_x, yp, door_w, door_h, color="#4b6e3c", arch=0.75)
    window(M, fr, -1.3 - xo, win_y, win_w, win_h, shutter_col="#4f7a46")
    window(M, bk, 0, win_y, 0.8, 0.8, shutter_col="#4f7a46", flowers=False)
    window(M, rt, 0, win_y, 0.8, 0.8, flowers=False)
    climber(M, fr, door_x + door_w / 2 + 0.38, 0.55, 2.35)
    stone_patch(M, fr, -0.35 - xo, 0.95, rx=0.3, ry=0.2)
    stone_patch(M, rt, -1.2, 2.1)
    thatch = ("#86622c", "#b98a44", "#d6b46e")
    roof_top = roof_slopes(M, W=W, D=D, ye=ye, yr=yr, o=o, ov=ov, courses=3, th=0.16, lip=0.2, cols=22,
                           palette=thatch, style="thatch", seed=13, ridge_r=0.26)
    # rolled verges: a thick round of thatch along each gable rake
    for sgn in (1.0, -1.0):
        d = unit([0.0, ye - yr, sgn * D / 2])
        n = unit(np.cross(d, [sgn, 0.0, 0.0]))
        if n[1] < 0:
            n = -n
        S = np.hypot(ye - yr, D / 2) * (1 + o / (D / 2))
        for xe in (-W / 2 - ov + 0.12, W / 2 + ov - 0.12):
            a = np.array([xe, yr, 0.0]) + n * 0.2 - d * 0.05
            b = np.array([xe, yr, 0.0]) + d * (S + 0.04) + n * 0.2
            beam(M, "Roof", a, b, 0.32, 0.38, out=n, c=0.12, jitter=0.03,
                 color=lambda P_, N_: mix(thatch[0], thatch[1], 0.55 + 0.4 * np.clip(N_[:, 1], 0, 1)))
        # scalloped ridge band (the woven ridge of a thatch)
        sk1 = S / 3
        X0, X1 = -W / 2 - ov + 0.25, W / 2 + ov - 0.25
        teeth = 9
        pts = [(X0, 0.0, 0.0), (X1, 0.0, 0.0)]
        xs = np.linspace(X1, X0, 2 * teeth + 1)
        for i, x in enumerate(xs):
            pts.append((x, 0.62 if i % 2 else 0.42, 0))
        Pb = np.array([np.array([0.0, yr, 0.0]) + x * np.array([1.0, 0, 0]) + s_ * d
                       + n * (0.16 + 0.2 * s_ / sk1 + 0.035) for x, s_, _ in pts])
        M.add("Roof", [(Pb, np.tile(n, (len(Pb), 1)))], mix(thatch[0], thatch[1], 0.35))
    # chimney stack against the left gable, from the ground
    cx = -W / 2 - 0.3
    M.add("Stone", box((0.62, 0.7, 0.9), (cx, -0.25, 0.0)), STONE)
    for i, (n, half, off) in enumerate((((0, 0, 1), 0.3, 0.42), ((-1, 0, 0), 0.42, 0.3), ((0, 0, -1), 0.3, 0.42))):
        n = np.array(n, float)
        wl = Wall((cx + n[0] * off, 0, n[2] * off), n)
        hgt = 5.42
        poly = [(-half, -0.6), (half, -0.6), (half, hgt), (-half, hgt)]
        cells = masonry_cells(-half, half, -0.6, hgt, course=0.36, block=(0.25, 0.5), rng=rng, phase=i)
        facade(M, "Stone", wl, poly, cells, stone_paint(seed=5))
    M.add("Stone", bevel_box((0.76, 0.14, 0.98), 0.035, (cx + 0.04, 5.52, 0)), mix(STONE, "#c9c1b4", 0.3))
    M.add("Stone", box((0.4, 0.02, 0.6), (cx + 0.04, 5.6, 0)), "#2a1e1c")
    # whole model shifted right so the stack fits in the footprint
    for p in M.polys:
        p[1] = p[1] + np.array([xo, 0, 0])
    inside = interior(W, D, yp, roof_top, ov, o, yr, ye)
    M.carve(lambda P: inside(P - np.array([xo, 0, 0])), mats=("Stone", "Wood", "Plaster"))
    return M


# ----------------------------------------------------------------------------- export and review

MODELS: dict[str, callable] = {"house_a": house_a, "house_b": house_b, "house_c": house_c}


def check_footprint(name, mesh) -> dict:
    lo, hi = mesh.bounds()
    w, d, h = FOOTPRINTS[name]
    V = mesh.vertices
    low = V[(V[:, 1] > 0.05) & (V[:, 1] < 2.0)]  # what a walker could bump into
    info = {"bounds": [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]],
            "footprint": [w, d, h]}
    if name in ROUND:
        r = float(np.linalg.norm(low[:, [0, 2]], axis=1).max()) if len(low) else 0.0
        info["body_radius"] = round(r, 3)
        info["fits"] = bool(r <= w / 2 + 0.12)
    else:
        ex = float(np.abs(low[:, 0]).max() - w / 2) if len(low) else 0.0
        ez = float(np.abs(low[:, 2]).max() - d / 2) if len(low) else 0.0
        info["overhang_body"] = [round(ex, 3), round(ez, 3)]
        info["fits"] = bool(ex <= 0.12 and ez <= 0.3)
    info["top"] = round(float(hi[1]), 3)
    return info


def export(name: str, M: Model) -> dict:
    mesh = M.mesh()
    scene = p3.Scene(name)
    scene.add(mesh, name=name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.glb"
    scene.export(path)
    info = {"mesh": mesh, "tris": M.tris, "materials": M.tris_by_mat(), "bytes": path.stat().st_size}
    info.update(check_footprint(name, mesh))
    return info


# ----------------------------------------------------------------------------- review

SKY = "#bcd3e0"


def _toon(scene, path, **kw):
    import nature  # the nature set's review renderer: ink lines like the game's inverted hulls
    kw.setdefault("ground", False)
    return nature.toon(scene, path, **kw)


def _capsule():
    cap = p3.capsule(0.25, 1.8).translate((0, 0.9, 0))
    cap.set_material(p3.Material("Human", "#d08a63", roughness=0.8))
    return cap


def _ground(radius=40.0, color="#8fae5c"):
    g = p3.plane(radius * 2, 24)  # small triangles: the preview rasterizer drops ones crossing the camera plane
    g.set_material(p3.Material("Ground", color, roughness=1.0))
    return g


def _sun(scene, target=(0, 0, 0)):
    scene.add_light("directional", position=(-6, 9, 7), target=target, color="#fff1d6", intensity=3.0)
    scene.add_light("directional", position=(7, 4, -6), target=target, color="#9fb7e8", intensity=0.9,
                    cast_shadows=False)


def _load_all(results):
    meshes = {k: v["mesh"] for k, v in results.items()}
    for name in MODELS:
        if name not in meshes and (OUT / f"{name}.glb").exists():
            sc = p3.load(OUT / f"{name}.glb")
            ms = [n.mesh for n in sc.nodes() if getattr(n, "mesh", None) is not None] if hasattr(sc, "nodes") else []
            if ms:
                meshes[name] = ms[0]
    return meshes


def review(results):
    REVIEW.mkdir(parents=True, exist_ok=True)
    meshes = _load_all(results)
    cap = _capsule()
    # 1) the whole set: houses behind, props in front, a 1.8 m capsule
    scene = p3.Scene("VillageSheet")
    scene.add(_ground(60))
    x = -11.0
    for name in ("house_c", "house_a", "house_b"):
        if name in meshes:
            w = FOOTPRINTS[name][0]
            x += w / 2
            scene.add(meshes[name], position=(x, 0, -3.0))
            x += w / 2 + 1.2
    props = [n for n in MODELS if not n.startswith("house") and n in meshes]
    x = -9.5
    for name in props:
        w = FOOTPRINTS[name][0]
        x += w / 2
        scene.add(meshes[name], position=(x, 0, 2.6))
        x += w / 2 + 0.55
    scene.add(cap, position=(x + 0.3, 0, 2.6))
    _sun(scene)
    _toon(scene, REVIEW / "sheet.png", width=1200, height=640, view=(0.0, 0.32, 1.0), yfov=30,
          target=(0.5, 2.2, 0.0), distance=30, background=SKY)
    # 2) every house up close, 3/4 front, with the capsule by the door
    for name in ("house_a", "house_b", "house_c"):
        if name not in meshes:
            continue
        scene = p3.Scene(name)
        scene.add(_ground(30))
        scene.add(meshes[name])
        w, d, h = FOOTPRINTS[name]
        scene.add(cap, position=(-w / 2 - 0.3, 0, d / 2 + 1.0))
        _sun(scene)
        _toon(scene, REVIEW / f"{name}.png", width=1000, height=760, view=(0.75, 0.32, 1.0), yfov=32,
              target=(0, h * 0.42, 0), distance=h * 2.5, background=SKY)
        _toon(scene, REVIEW / f"{name}_back.png", width=700, height=520, view=(-0.8, 0.45, -1.0), yfov=32,
              target=(0, h * 0.42, 0), distance=h * 2.5, background=SKY)


def main(argv):
    names = [a for a in argv if not a.startswith("--")] or list(MODELS)
    results = {}
    for name in names:
        M = MODELS[name]()
        info = export(name, M)
        results[name] = info
        budget = BUDGET.get(name, SMALL_BUDGET)
        flag = "" if info["tris"] <= budget else f"  OVER BUDGET ({budget})"
        print(f"{name:13s} tris={info['tris']:5d}{flag} top={info['top']:.2f} fits={info['fits']} "
              f"{info.get('overhang_body', info.get('body_radius'))} mats={info['materials']} {info['bytes'] // 1024} KB")
    REVIEW.mkdir(parents=True, exist_ok=True)
    stats_path = REVIEW / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    for name, info in results.items():
        stats[name] = {k: v for k, v in info.items() if k != "mesh"}
    stats_path.write_text(json.dumps(dict(sorted(stats.items())), indent=1, ensure_ascii=False) + "\n")
    if "--no-review" not in argv:
        review(results)
    return results


if __name__ == "__main__":
    main(sys.argv[1:])
