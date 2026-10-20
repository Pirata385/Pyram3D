"""Niryk prototype 2: the 3D nature set of the clearing (oaks, pine, bush, rocks, grass, flowers, stump).

Stylized cartoon for a toon shader with ink outlines (the client): chunky silhouettes, canopies made
of a few smooth rounded lobes (cumulus-like, flatter underneath), clean trunks with taper and root
flare, faceted-but-smooth boulders (big flat planes, rounded edges), grass as a few thick tapered
blades (solid geometry, no alpha). Final colors are vertex colors (materials are white) with a
subtle top light / ambient occlusion; material names Bark, Leaves, Rock, Grass, Flower and Wood let
the client sway Leaves/Grass with wind. Every asset stands on y = 0 with its origin at the base
center, +Y up, and is instanced many times, so triangle budgets are tight.

    python projects/niryk/protótipo_2/nature.py [name ...] [--no-review]

Writes exports/nature/<name>.glb, exports/nature/nature.json and review/nature_*.png.
"""
from __future__ import annotations

import itertools
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyram3d").is_dir())))  # the Pyram3D root

import pyram3d as p3  # noqa: E402
from pyram3d import math3d as m3  # noqa: E402
from pyram3d.noise import fbm3  # noqa: E402

OUT = HERE / "exports" / "nature"
REVIEW = HERE / "review"

# ----------------------------------------------------------------------------- palette (sRGB albedo)
# dark / mid / light ramps; the toon shader adds the light bands on top of these
LEAF = ("#2d5634", "#4a8a3c", "#8ab950")
LEAF_2 = ("#2f5a30", "#56903a", "#9bc455")  # oak_2: a touch warmer and lighter
PINE = ("#173a34", "#285f45", "#5b9050")
BUSH = ("#2c5a38", "#468a42", "#7fb653")
BARK = ("#3a2722", "#694431", "#93694a")
WOOD = ("#a0703f", "#d4a56a", "#ecd09a")
ROCK = ("#4a4553", "#736c6a", "#9a9185")
MOSS = ("#41632f", "#5d8a3a")
GRASS = ("#3d7634", "#69a443", "#bddb6e")
FLOWER_SETS = {
    "flowers_1": ("#f7f3e8", "#e9e2f2", "#f2b52c"),  # white daisies, golden centers
    "flowers_2": ("#9c7fdc", "#b9a2ec", "#f6e08a"),  # lilac
    "flowers_3": ("#f2c33a", "#f6d867", "#c4642a"),  # yellow, rust centers
}


def rgb(c) -> np.ndarray:
    return np.asarray(m3.as_color(c)[:3], float) if isinstance(c, str) else np.asarray(c, float)


def mix(a, b, t):
    """Blend sRGB colors a -> b by t (scalar or (N,) array)."""
    a, b = rgb(a), rgb(b)
    t = np.asarray(t, float)
    if t.ndim:
        t = t[:, None]
    return a + (b - a) * t


def ramp(pal, t):
    """3-stop ramp: t in [0, 1] -> pal[0] .. pal[1] .. pal[2]."""
    t = np.clip(np.asarray(t, float), 0, 1)
    lo = mix(pal[0], pal[1], np.minimum(t, 0.5) * 2)
    hi = mix(pal[1], pal[2], np.maximum(t - 0.5, 0) * 2)
    return np.where((t < 0.5)[:, None], lo, hi)


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def noise(P, scale=1.0, seed=0, octaves=3):
    """Smooth value in ~[-1, 1] for painterly color variation."""
    return np.clip(fbm3(np.asarray(P, float) * scale, octaves=octaves, seed=seed) * 1.6, -1, 1)


# ----------------------------------------------------------------------------- materials

_MATS: dict[str, p3.Material] = {}


def material(name: str) -> p3.Material:
    """One shared white material per slot; the vertex colors carry the final color."""
    if name not in _MATS:
        _MATS[name] = p3.Material(name, "#ffffff", metallic=0.0, roughness=0.9)
    return _MATS[name]


# ----------------------------------------------------------------------------- parts


@dataclass
class Part:
    """A piece of an asset: positions, polygon faces (CCW = outside), one material, per-vertex
    sRGB colors and (optionally) custom per-vertex normals."""
    V: np.ndarray
    F: list
    mat: str
    C: np.ndarray | None = None
    N: np.ndarray | None = None
    tag: dict = field(default_factory=dict)

    def mesh(self) -> p3.Mesh:
        return p3.Mesh(self.V, self.F, smooth_angle=180.0)

    def normals(self) -> np.ndarray:
        if self.N is None:
            self.N = self.mesh().vertex_normals()
        return self.N

    def translate(self, t) -> "Part":
        self.V = self.V + np.asarray(t, float)
        return self

    def tris(self) -> int:
        return int(sum(len(f) - 2 for f in self.F))

    def closed(self) -> bool:
        """Watertight once coincident vertices (hard-edge splits) are welded."""
        return bool(self.mesh().weld(1e-6).stats()["watertight"])


def build_mesh(name: str, parts: list[Part]) -> p3.Mesh:
    """Merge parts into one Mesh (one primitive per material) with colors and normals."""
    order = sorted(parts, key=lambda p: p.mat)
    V, F, FM, C, N, mats = [], [], [], [], [], []
    off = 0
    for p in order:
        if p.mat not in mats:
            mats.append(p.mat)
        mi = mats.index(p.mat)
        V.append(p.V)
        F.extend(tuple(int(i) + off for i in f) for f in p.F)
        FM.extend([mi] * len(p.F))
        C.append(p.C if p.C is not None else np.ones((len(p.V), 3)))
        N.append(p.normals())
        off += len(p.V)
    Vc = np.concatenate(V)
    Cs = np.clip(np.concatenate(C), 0, 1)
    colors = np.concatenate([m3.srgb_to_linear(Cs), np.ones((len(Cs), 1))], axis=1)
    m = p3.Mesh(Vc, F, name=name, materials=[material(k) for k in mats], face_materials=FM, colors=colors,
                smooth_angle=180.0)
    m.normals = m3.normalize(np.concatenate(N))
    return m


def drop_faces(part: Part, keep) -> Part:
    """Keep only the faces flagged in `keep` and drop unused vertices (normals/colors follow)."""
    F = [f for f, k in zip(part.F, keep) if k]
    used = np.unique(np.fromiter(itertools.chain.from_iterable(F), int)) if F else np.zeros(0, int)
    remap = -np.ones(len(part.V), int)
    remap[used] = np.arange(len(used))
    part.F = [tuple(int(remap[i]) for i in f) for f in F]
    part.V = part.V[used]
    if part.N is not None:
        part.N = part.N[used]
    if part.C is not None:
        part.C = part.C[used]
    return part


def fix_winding(part: Part) -> Part:
    """Flip every face if the closed surface encloses a negative volume (points inward)."""
    V = part.V
    vol = 0.0
    for f in part.F:
        a = V[f[0]]
        for k in range(1, len(f) - 1):
            vol += float(np.dot(a, np.cross(V[f[k]], V[f[k + 1]])))
    if vol < 0:
        part.F = [tuple(reversed(f)) for f in part.F]
        part.N = None
    return part


def yaw_rot(v, deg):
    a = np.radians(deg)
    ca, sa = np.cos(a), np.sin(a)
    v = np.asarray(v, float)
    return np.stack([ca * v[:, 0] + sa * v[:, 2], v[:, 1], -sa * v[:, 0] + ca * v[:, 2]], 1)


# ----------------------------------------------------------------------------- shape builders


def cube_sphere(n: int) -> Part:
    """Unit sphere from a cube with n x n quads per side (equal-angle mapping: even quads)."""
    b = p3.box(2.0, segments=n).weld()
    v = np.tan(np.clip(b.vertices, -1, 1) * (np.pi / 4))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return fix_winding(Part(v, list(b.faces), "Leaves"))


@dataclass
class Lobe:
    """A cumulus-like canopy mass: an ellipsoid whose lower half is squashed by `flat`."""
    c: tuple
    r: tuple
    n: int = 4
    flat: float = 0.55
    yaw: float = 0.0
    tint: float = 0.0

    def local(self, P):
        q = yaw_rot(np.asarray(P, float) - np.asarray(self.c, float), -self.yaw) / np.asarray(self.r, float)
        q[:, 1] = np.where(q[:, 1] < 0, q[:, 1] / self.flat, q[:, 1])
        return q

    def inside(self, P, margin=0.0):
        """True where P is inside the lobe by at least `margin` meters (approximately)."""
        return np.linalg.norm(self.local(P), axis=1) < 1.0 - margin / min(self.r)

    def part(self, mat="Leaves") -> Part:
        p = cube_sphere(self.n)
        v = p.V.copy()
        v[:, 1] = np.where(v[:, 1] < 0, v[:, 1] * self.flat, v[:, 1])
        v *= np.asarray(self.r, float)
        p.V = yaw_rot(v, self.yaw) + np.asarray(self.c, float)
        p.mat = mat
        p.N = None
        p.normals()
        return p


def canopy(lobes: list[Lobe], center, radii, bend=0.45, margin=0.1, ground=None, mat="Leaves") -> list[Part]:
    """Overlapping closed lobes. Faces buried deep inside another lobe (or under the ground) are
    removed to save triangles; normals are bent toward the whole crown's ellipsoid so the toon
    bands sweep across the canopy as one volume while the outlines still draw every lobe."""
    parts = []
    for i, lb in enumerate(lobes):
        p = lb.part(mat)
        p.tag["lobe"] = lb
        P = p.V
        buried = np.zeros(len(P), bool)
        for j, other in enumerate(lobes):
            if j != i:
                buried |= other.inside(P, margin)
        if ground is not None:
            buried |= P[:, 1] < ground
        keep = [not buried[list(f)].all() for f in p.F]
        drop_faces(p, keep)
        g = m3.normalize((p.V - np.asarray(center, float)) / np.asarray(radii, float) ** 2)
        p.N = m3.normalize(p.N * (1 - bend) + g * bend)
        parts.append(p)
    return parts


def revolve(rings, segs: int, mod=None, close_bottom=True, close_top=True, phase=0.0, mat="Bark") -> Part:
    """Surface of revolution around +Y. rings: [(r, y), ...] bottom to top; r == 0 makes a pole.
    mod(i, theta) -> (radius factor, dy, dx, dz) shapes ring i around (roots, scallops, lean)."""
    th = phase + np.arange(segs) * (2 * np.pi / segs)
    V, F, ring_idx = [], [], []
    for i, (r, y) in enumerate(rings):
        if mod is not None:
            k, dy, dx, dz = (np.broadcast_to(np.asarray(v, float), th.shape) for v in mod(i, th))
        else:
            k = dy = dx = dz = np.zeros_like(th)
            k = np.ones_like(th)
        if r == 0:
            ring_idx.append([len(V)])
            V.append((dx.mean(), y + dy.mean(), dz.mean()))
            continue
        ring_idx.append(list(range(len(V), len(V) + segs)))
        for j in range(segs):
            rr = r * k[j]
            V.append((rr * np.cos(th[j]) + dx[j], y + dy[j], -rr * np.sin(th[j]) + dz[j]))
    for a, b in zip(ring_idx[:-1], ring_idx[1:]):
        if len(a) == 1 and len(b) == 1:
            continue
        if len(a) == 1:
            F.extend((a[0], b[j], b[(j + 1) % segs]) for j in range(segs))
        elif len(b) == 1:
            F.extend((a[j], b[0], a[(j + 1) % segs]) for j in range(segs))
        else:
            F.extend((a[j], b[j], b[(j + 1) % segs], a[(j + 1) % segs]) for j in range(segs))
    if close_bottom and len(ring_idx[0]) > 1:
        F.append(tuple(ring_idx[0][::-1]))
    if close_top and len(ring_idx[-1]) > 1:
        F.append(tuple(ring_idx[-1]))
    part = Part(np.array(V, float), F, mat)
    part.tag["rings"] = ring_idx
    # orientation: quads run (ring i, ring i+1) with theta CCW from above -> check by volume
    if close_bottom and close_top:
        fix_winding(part)
    else:
        m = part.mesh()
        if np.einsum("ij,ij->i", m.face_normals(), m.face_centers() - [0, m.face_centers()[:, 1].mean(), 0]).sum() < 0:
            part.F = [tuple(reversed(f)) for f in part.F]
    return part


def tube(path, radius, segs: int, mat="Bark") -> Part:
    """Swept circle along a polyline (branches), closed at both ends. radius: per-point list."""
    m = p3.sweep(np.asarray(path, float), radius=list(radius), segments=segs, caps=True)
    return fix_winding(Part(m.vertices.copy(), list(m.faces), mat))


def convex_polytope(normals, offsets):
    """Vertices and polygon faces of {x : n_i . x <= d_i} (bounded). Faces CCW from outside."""
    n = m3.normalize(np.asarray(normals, float))
    d = np.asarray(offsets, float)
    idx = np.array(list(itertools.combinations(range(len(n)), 3)))
    A = n[idx]
    ok = np.abs(np.linalg.det(A)) > 1e-10
    x = np.linalg.solve(A[ok], d[idx][ok][..., None])[..., 0]
    scale = np.abs(d).max()
    pts = x[(x @ n.T <= d + 1e-7 * scale).all(1)]
    merged: list[np.ndarray] = []  # merge points where more than 3 planes meet
    for p in pts:
        if not any(np.linalg.norm(p - q) < 1e-6 * scale for q in merged):
            merged.append(p)
    P = np.array(merged)
    faces = []
    for i in range(len(n)):
        on = np.flatnonzero(np.abs(P @ n[i] - d[i]) < 1e-6 * scale)
        if len(on) < 3:
            continue
        c = P[on].mean(0)
        u = m3.normalize(np.cross(n[i], [0.3, 1.0, 0.2] if abs(n[i][1]) < 0.9 else [1.0, 0.0, 0.3]))
        w = np.cross(n[i], u)
        q = P[on] - c
        order = on[np.argsort(np.arctan2(q @ w, q @ u))]
        f = tuple(int(k) for k in order)
        # CCW seen from outside (+n)
        if np.dot(np.cross(P[f[1]] - P[f[0]], P[f[2]] - P[f[0]]), n[i]) < 0:
            f = f[::-1]
        faces.append(f)
    return P, faces


def fib_dirs(count: int, seed=0, jitter=0.0) -> np.ndarray:
    i = np.arange(count) + 0.5
    phi = np.arccos(1 - 2 * i / count)
    th = np.pi * (1 + 5 ** 0.5) * i
    d = np.stack([np.cos(th) * np.sin(phi), np.cos(phi), np.sin(th) * np.sin(phi)], 1)
    if jitter:
        d += np.random.default_rng(seed).normal(0, jitter, d.shape)
    return m3.normalize(d)


def rounded_rock(radii, n_planes: int, round_r: float, seed: int, sink=0.06, top_bias=0.3,
                 cut=(0.85, 1.0), round_dirs=60, shear=(0.0, 0.0), bottom=0.5) -> Part:
    """Faceted-but-smooth boulder: a random convex polytope (big flat planes) Minkowski-rounded by
    a sphere of radius round_r (small extra planes chamfer every edge and corner). Area-weighted
    smooth normals keep the big planes flat and the edges soft."""
    rng = np.random.default_rng(seed)
    R = np.asarray(radii, float)
    dirs = fib_dirs(n_planes, seed, jitter=0.2)
    dirs[:, 1] += top_bias  # more, larger facets facing up (where the light falls)
    dirs = m3.normalize(dirs)
    dirs = np.vstack([dirs, [[0, -1, 0]]])
    supp = np.sqrt(((dirs * R) ** 2).sum(1))  # ellipsoid support function
    offs = supp * rng.uniform(*cut, len(dirs))
    offs[-1] = R[1] * bottom  # flat bottom
    P0, _ = convex_polytope(dirs, offs)
    U = fib_dirs(round_dirs, seed + 7)
    allN = np.vstack([dirs, U])
    allD = np.concatenate([offs, (P0 @ U.T).max(0)]) + round_r
    P, faces = convex_polytope(allN, allD)
    P[:, 0] += shear[0] * (P[:, 1] - P[:, 1].min())
    P[:, 2] += shear[1] * (P[:, 1] - P[:, 1].min())
    part = Part(P, faces, "Rock")
    foot = P[P[:, 1] < P[:, 1].min() + 0.15 * np.ptp(P[:, 1])]
    mid = (foot.min(0) + foot.max(0)) / 2  # origin at the center of the footprint
    part.V = part.V - [mid[0], part.V[:, 1].min() + sink, mid[2]]
    return split_facets(part)


def split_facets(part: Part, frac=0.015) -> Part:
    """Give every big facet (area > frac of the total) its own vertices with its exact normal: flat
    toon bands and crisp facet colors. The narrow chamfers keep shared vertices whose normals come
    from their largest neighbor, so light flows smoothly over the rounded edges. The positions are
    unchanged, so the surface stays closed (split only along the hard facet edges)."""
    m = part.mesh()
    fn, area = m.face_normals(), m.face_areas()
    shared_n = dominant_normals(part)
    big = area > frac * area.sum()
    V, N, F = list(part.V), list(shared_n), []
    for f, b, n in zip(part.F, big, fn):
        if not b:
            F.append(f)
            continue
        base = len(V)
        V.extend(part.V[list(f)])
        N.extend([n] * len(f))
        F.append(tuple(range(base, base + len(f))))
    out = Part(np.array(V), F, part.mat, N=np.array(N))
    used = np.unique(np.fromiter(itertools.chain.from_iterable(F), int))
    return drop_faces(out, [True] * len(F)) if len(used) < len(V) else out


def dominant_normals(part: Part, power=3.0) -> np.ndarray:
    """Vertex normals weighted by face area**power: every vertex takes the normal of its largest
    face, so big facets shade perfectly flat and only the small chamfers between them blend."""
    m = part.mesh()
    fn = m.face_normals()
    w = m.face_areas() ** power
    acc = np.zeros((len(part.V), 3))
    for f, n, wf in zip(part.F, fn, w):
        acc[list(f)] += n * wf
    return m3.normalize(acc)


def blade(base, azim, height, width, thick, lean, curl=0.25, mat="Grass") -> Part:
    """A thick tapered blade: V-shaped (keeled) triangular section, base -> mid -> pointed tip,
    leaning outward along azimuth `azim` (degrees) and curling further out near the tip."""
    a = np.radians(azim)
    d = np.array([np.cos(a), 0.0, -np.sin(a)])
    up = np.array([0.0, 1.0, 0.0])
    base = np.asarray(base, float)
    mid = base + up * height * 0.55 + d * height * lean * 0.45
    tip = base + up * height * (1 - 0.12 * curl) + d * height * (lean + curl * 0.6)
    w = m3.normalize(np.cross(up, d))

    def section(c, tangent, s):
        nrm = m3.normalize(np.cross(w, tangent))
        return [c - w * width * s / 2, c + w * width * s / 2, c - nrm * thick * s]

    t0 = m3.normalize(mid - base)
    t1 = m3.normalize(tip - base)
    V = section(base, t0, 1.0) + section(mid, t1, 0.72) + [tip]
    F = [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5), (3, 4, 6), (4, 5, 6), (5, 3, 6), (2, 1, 0)]
    return fix_winding(Part(np.array(V), F, mat))


# ----------------------------------------------------------------------------- color painters


def paint_canopy(parts: list[Part], y0, y1, center, pal, seed=0):
    """Leaves: darker and cooler underneath and inside the crown, lighter on top; each lobe gets a
    small value shift so the clusters read as separate masses."""
    for p in parts:
        P, N = p.V, p.normals()
        h = smoothstep(y0, y1, P[:, 1])
        out = m3.normalize(P - np.asarray(center, float))
        rim = np.clip(np.einsum("ij,ij->i", out, N), 0, 1)
        tint = p.tag["lobe"].tint if "lobe" in p.tag else 0.0
        t = 0.34 + 0.3 * h + 0.2 * N[:, 1] + 0.07 * noise(P, 0.7, seed) + tint
        p.C = ramp(pal, np.maximum(t * (0.8 + 0.2 * rim), 0.2))


def paint_bark(p: Part, y_shade, seed=0, pal=BARK):
    """Bark: dark at the roots (contact AO), lighter on the lit mid trunk, darker under the crown."""
    P, N = p.V, p.normals()
    ao = smoothstep(-0.05, 0.6, P[:, 1])
    shade = smoothstep(y_shade - 1.2, y_shade + 0.6, P[:, 1])
    t = 0.2 + 0.5 * ao - 0.22 * shade + 0.1 * N[:, 1] + 0.07 * noise(P, 2.5, seed)
    p.C = ramp(pal, t)
    return p


def paint_rock(p: Part, seed=0, moss=0.0, pal=ROCK):
    P, N = p.V, p.normals()
    H = P[:, 1].max()
    h = smoothstep(-0.02, H, P[:, 1])
    t = 0.28 + 0.32 * h + 0.2 * N[:, 1] + 0.1 * noise(P, 1.3, seed)
    C = ramp(pal, t)
    if moss:
        m = smoothstep(0.8, 0.88, N[:, 1] + 0.06 * noise(P, 1.1, seed + 3)) * smoothstep(H * 0.4, H * 0.55, P[:, 1])
        m *= moss
        C = C * (1 - m[:, None]) + mix(MOSS[0], MOSS[1], noise(P, 2.0, seed) * 0.5 + 0.5) * m[:, None]
    p.C = C
    return p


def paint_grass(p: Part, height, seed=0, pal=GRASS, up=0.65):
    """Grass: ground-matching base to light tips. Normals lean toward +Y so a tuft shades like the
    ground it grows from (no dark back sides on thin blades)."""
    P = p.V
    t = smoothstep(-0.02, height, P[:, 1]) * 0.85 + 0.12 + 0.06 * noise(P, 6.0, seed)
    p.C = ramp(pal, t)
    p.N = m3.normalize(p.normals() * (1 - up) + np.array([0.0, 1.0, 0.0]) * up)
    return p


# ----------------------------------------------------------------------------- assets

ASSETS: dict[str, dict] = {}


def asset(name, **meta):
    def deco(fn):
        ASSETS[name] = {"fn": fn, **meta}
        return fn
    return deco


def tree_trunk(r0, fork, flare, roots, branches, seed, segs=12, taper=0.28, lean=(0.0, 0.0)):
    """Tapered trunk with root buttresses up to the fork, plus swept limbs into the crown.
    branches: [(tip (x, y, z), base radius)]."""
    rng = np.random.default_rng(seed)
    ys = [-0.15, 0.0, 0.14, 0.38, 0.8, fork * 0.6, fork]
    ph = rng.uniform(0, 2 * np.pi)
    jit = rng.uniform(0.75, 1.25, roots)

    def mod(i, th):
        y = ys[i]
        lobe = np.cos(roots * (th + ph))
        bump = np.maximum(lobe, 0) ** 1.4 * jit[(((th + ph) * roots / (2 * np.pi)) + 0.5).astype(int) % roots]
        fl = flare * np.exp(-max(y, 0) / 0.3)
        k = 1 + fl * (0.3 + 0.95 * bump)
        s = max(y, 0) / fork
        return k, -0.05 * fl * bump * (y < 0.3), lean[0] * s * s * fork, lean[1] * s * s * fork

    rad = [r0 * (1 - taper * min(max(y, 0) / fork, 1) ** 0.8) for y in ys]
    r_top = rad[-1]
    # rounded top (a pole just above the fork) hidden inside the limb bases: no flat cap shows
    ys.append(fork + 0.6 * r_top)
    rad.append(0.0)
    trunk = revolve(list(zip(rad, ys)), segs, mod, phase=ph)
    parts = [trunk]
    top = np.array([lean[0] * fork, fork, lean[1] * fork])
    for tip, rb in branches:
        tip = np.asarray(tip, float)
        start = top + [0, -0.75, 0]
        dirv = tip - start
        p1 = start + [dirv[0] * 0.3, dirv[1] * 0.5, dirv[2] * 0.3]  # rises first, then reaches out
        path = m3.catmull_rom([start, p1, tip], samples_per_segment=2)
        n = len(path)
        radius = [max(rb, r_top * 0.8) * (1 - 0.6 * (k / (n - 1))) for k in range(n)]
        parts.append(tube(path, radius, 6))
    return parts


def ring_lobes(rng, count, dist, y, r, n, flat, phase=0.0, jitter=0.2, ydev=0.2, center=(0, 0)):
    out = []
    for k in range(count):
        a = phase + k * (2 * np.pi / count) + rng.uniform(-jitter, jitter)
        rr = dist * rng.uniform(0.9, 1.08)
        out.append(Lobe((center[0] + rr * np.cos(a), y + rng.uniform(-ydev, ydev), center[1] - rr * np.sin(a)),
                        tuple(np.array(r) * rng.uniform(0.9, 1.1)), n=n, flat=flat, yaw=np.degrees(a),
                        tint=rng.uniform(-0.07, 0.07)))
    return out


@asset("oak_1", kind="tree", collider="auto", scale=(0.85, 1.2), sway=0.1)
def oak_1():
    """Round broadleaf, ~7 m: a wide cumulus crown on a short thick trunk with a three-way fork."""
    seed = 11
    rng = np.random.default_rng(seed)
    fork = 2.4
    branches = [((-1.4, 4.1, 0.4), 0.26), ((1.25, 4.3, -0.5), 0.24), ((0.2, 4.8, 1.0), 0.2)]
    trunk = tree_trunk(0.42, fork, 0.6, 5, branches, seed)
    C, R = (0.0, 4.9, 0.0), (3.1, 2.3, 3.1)
    lobes = [Lobe((0, 4.85, 0), (2.3, 1.7, 2.3), n=3, flat=0.6)]
    lobes += ring_lobes(rng, 5, 1.85, 4.15, (1.5, 1.2, 1.5), 5, 0.5, phase=0.3)
    lobes += ring_lobes(rng, 3, 1.0, 5.65, (1.4, 1.15, 1.4), 5, 0.55, phase=1.2, ydev=0.12)
    leaves = canopy(lobes, C, R)
    paint_canopy(leaves, 3.0, 6.9, C, LEAF, seed)
    for p in trunk:
        paint_bark(p, 3.4, seed)
    return trunk + leaves, {"canopy": [2.9, 6.9]}


@asset("oak_2", kind="tree", collider="auto", scale=(0.85, 1.2), sway=0.1)
def oak_2():
    """Taller broadleaf, ~7.5 m: leaning trunk, a lower wide cluster and a higher offset cluster."""
    seed = 23
    rng = np.random.default_rng(seed)
    fork = 2.8
    lean = (0.3, -0.1)
    branches = [((-1.0, 4.3, 0.55), 0.24), ((1.35, 4.5, 0.1), 0.22), ((0.6, 5.6, -0.4), 0.2)]
    trunk = tree_trunk(0.38, fork, 0.55, 4, branches, seed, lean=lean)
    lobes = [Lobe((0.25, 5.0, -0.05), (1.8, 1.55, 1.8), n=3, flat=0.6)]
    lobes += ring_lobes(rng, 5, 1.55, 4.35, (1.3, 1.05, 1.3), 5, 0.5, phase=0.6, center=(0.2, -0.05))
    lobes += ring_lobes(rng, 3, 0.95, 5.8, (1.15, 0.98, 1.15), 4, 0.55, phase=0.2, center=(0.45, -0.25),
                        ydev=0.12)
    lobes.append(Lobe((0.6, 6.55, -0.35), (0.95, 0.9, 0.95), n=4, flat=0.6, tint=0.05))
    C, R = (0.35, 5.3, -0.15), (2.7, 2.5, 2.7)
    leaves = canopy(lobes, C, R)
    paint_canopy(leaves, 3.4, 7.4, C, LEAF_2, seed)
    for p in trunk:
        paint_bark(p, 3.7, seed)
    return trunk + leaves, {"canopy": [3.2, 7.5]}


@asset("pine_1", kind="tree", collider="auto", scale=(0.8, 1.2), sway=0.07)
def pine_1():
    """Stylized fir, ~9 m: five stacked scalloped tiers with drooping tips on a slim trunk."""
    seed = 5
    rng = np.random.default_rng(seed)
    trunk = revolve([(0.3, -0.12), (0.36, 0.0), (0.27, 0.3), (0.22, 2.0), (0.12, 6.5), (0.0, 7.6)], 8,
                    lambda i, th: (1 + (0.25 if i < 2 else 0.0) * np.maximum(np.cos(4 * th), 0), 0, 0, 0))
    paint_bark(trunk, 2.4, seed)
    tiers = [  # (base y, height, radius, segments, scallops)
        (1.75, 2.7, 2.35, 24, 8),
        (3.05, 2.5, 1.98, 24, 8),
        (4.3, 2.3, 1.6, 20, 7),
        (5.45, 2.1, 1.22, 18, 6),
        (6.5, 2.55, 0.82, 12, 5),
    ]
    parts = [trunk]
    for ti, (y0, h, R, segs, sc) in enumerate(tiers):
        ph = rng.uniform(0, 2 * np.pi)
        top = ti == len(tiers) - 1
        prof = [(0.0, 0.2 * h), (0.45 * R, 0.09 * h), (0.93 * R, 0.0), (R, 0.045 * h),
                (0.66 * R, 0.36 * h), (0.32 * R, 0.68 * h), (0.0, h)]
        amp = [0, 0.05, 0.16, 0.16, 0.08, 0.03, 0]
        droop = [0, 0.02, 0.1, 0.1, 0.03, 0, 0]

        def mod(i, th, amp=amp, droop=droop, sc=sc, ph=ph, h=h):
            c = np.cos(sc * th + ph)
            return 1 + amp[i] * c, -droop[i] * h * np.maximum(c, 0) ** 2, 0, 0

        tier = revolve([(r, y0 + y) for r, y in prof], segs, mod, mat="Leaves")
        P, N = tier.V, tier.normals()
        # crown normals: bend toward the cone of the whole tree so bands run across tiers
        axis_n = m3.normalize(np.column_stack([P[:, 0], np.full(len(P), 0.75), P[:, 2]]))
        tier.N = m3.normalize(N * 0.65 + axis_n * 0.35)
        hrel = (P[:, 1] - y0) / h
        under = N[:, 1] < -0.2
        t = 0.32 + 0.35 * np.clip(hrel, 0, 1) + 0.18 * (ti / 4) + 0.15 * N[:, 1] + 0.05 * noise(P, 1.0, seed)
        t = np.where(under, 0.2 + 0.08 * (ti / 4), t)
        tier.C = ramp(PINE, t)
        if top:
            tier.C = ramp(PINE, t + 0.08)
        parts.append(tier)
    return parts, {"canopy": [1.6, 9.05]}


@asset("bush_1", kind="bush", collider="auto", scale=(0.8, 1.3), sway=0.04)
def bush_1():
    """Round leafy bush, ~1 m: one big mass with four smaller ones around the base."""
    seed = 31
    rng = np.random.default_rng(seed)
    lobes = [Lobe((0, 0.5, 0), (0.62, 0.52, 0.62), n=4, flat=1.0, tint=0.04)]
    for k in range(3):
        a = 0.4 + k * (2 * np.pi / 3) + rng.uniform(-0.25, 0.25)
        rr = 0.45 * rng.uniform(0.92, 1.08)
        r = np.array((0.45, 0.38, 0.45)) * rng.uniform(0.92, 1.1)
        y = 0.3 * rng.uniform(0.9, 1.1)
        lobes.append(Lobe((rr * np.cos(a), y, -rr * np.sin(a)), tuple(r), n=4, flat=(y + 0.06) / r[1],
                          yaw=np.degrees(a), tint=rng.uniform(-0.08, 0.06)))
    C, R = (0, 0.35, 0), (0.85, 0.75, 0.85)
    leaves = canopy(lobes, C, R, ground=-0.03, margin=0.05)
    paint_canopy(leaves, -0.05, 1.05, C, BUSH, seed)
    return leaves, {}


@asset("rock_1", kind="rock", collider="auto", scale=(0.7, 1.4))
def rock_1():
    """Small stone, ~0.6 m wide."""
    p = rounded_rock((0.31, 0.24, 0.27), 14, 0.06, seed=9, sink=0.04, round_dirs=50, shear=(0.08, 0.0),
                     cut=(0.8, 1.0), top_bias=0.55, bottom=0.35)
    paint_rock(p, seed=9)
    return [p], {}


@asset("rock_2", kind="rock", collider="auto", scale=(0.8, 1.25))
def rock_2():
    """Boulder, ~1.5 m: wide base, a big slanted top plane with a bit of moss."""
    p = rounded_rock((0.8, 0.8, 0.72), 18, 0.15, seed=4, sink=0.08, shear=(0.1, -0.05), cut=(0.78, 1.0),
                     round_dirs=64, top_bias=0.5, bottom=0.35)
    paint_rock(p, seed=4, moss=1.0)
    return [p], {}


@asset("grass_1", kind="grass", collider=0.0, scale=(0.7, 1.35), sway=0.05)
def grass_1():
    """Grass tuft, ~0.35 m: seven thick blades fanning out from one clump."""
    rng = np.random.default_rng(41)
    parts = []
    n = 7
    for k in range(n):
        az = k * (360 / n) + rng.uniform(-18, 18)
        h = rng.uniform(0.22, 0.34) if k % 3 else rng.uniform(0.32, 0.38)
        off = rng.uniform(0.01, 0.05)
        a = np.radians(az)
        base = (off * np.cos(a), -0.02, -off * np.sin(a))
        parts.append(blade(base, az, h, rng.uniform(0.065, 0.08), 0.024, rng.uniform(0.2, 0.42),
                           curl=rng.uniform(0.15, 0.4)))
    for p in parts:
        paint_grass(p, 0.4, seed=41)
    return parts, {}


def blossom(center, R, petals, colors, facing=(0, 1, 0), thick=0.024, mat="Flower", seed=0) -> list[Part]:
    """Chunky daisy: a cupped lens-shaped petal disc with rounded petals (two shoulder vertices and a
    notch each) and a domed center button, turned to face `facing`. Two closed parts, 40 triangles."""
    rng = np.random.default_rng(seed)
    k = petals * 3
    step = 2 * np.pi / petals
    base = np.arange(petals) * step + rng.uniform(0, step)
    th = np.stack([base - 0.2 * step, base + 0.2 * step, base + 0.5 * step], 1).reshape(-1)
    notch = np.arange(k) % 3 == 2
    rr = np.where(notch, R * 0.5, R)
    lift = np.where(notch, 0.1, 0.3) * R  # petals cup up a little
    ring = np.column_stack([rr * np.cos(th), lift, -rr * np.sin(th)])
    V = np.vstack([ring, [[0, thick / 2, 0], [0, -thick / 2, 0]]])
    ct, cb = k, k + 1
    F = [(ct, j, (j + 1) % k) for j in range(k)] + [(cb, (j + 1) % k, j) for j in range(k)]
    petal = fix_winding(Part(V, F, mat))
    petal.C = np.vstack([np.where(notch[:, None], rgb(colors[1]), rgb(colors[0])), [rgb(colors[1])] * 2])
    dome = revolve([(0.0, thick / 2 - 0.006), (0.38 * R, thick / 2 + 0.002), (0.0, thick / 2 + 0.02)], 5, mat=mat)
    dome.C = np.tile(rgb(colors[2]), (len(dome.V), 1))
    dome.C[-1] = mix(colors[2], "#ffffff", 0.3)
    rot = m3.quat_to_matrix3(m3.quat_between((0, 1, 0), m3.normalize(np.asarray(facing, float))))
    for p in (petal, dome):
        p.V = p.V @ rot.T + np.asarray(center, float)
        p.N = None
    return [petal, dome]


def stem(base, top, r=0.013, pal=GRASS) -> Part:
    """3-sided closed prism (8 triangles)."""
    base, top = np.asarray(base, float), np.asarray(top, float)
    d = m3.normalize(top - base)
    s1 = m3.normalize(np.cross(d, [0.3, 0.1, 1.0]))
    s2 = np.cross(d, s1)
    sec = [s1 * r, (-s1 * 0.5 + s2 * 0.866) * r, (-s1 * 0.5 - s2 * 0.866) * r]
    V = np.array([base + s for s in sec] + [top + s * 0.75 for s in sec])
    p = fix_winding(Part(V, [(0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5), (2, 1, 0), (3, 4, 5)], "Grass"))
    p.C = ramp(pal, np.array([0.15] * 3 + [0.5] * 3))
    p.N = m3.normalize(p.normals() * 0.4 + np.array([0.0, 1.0, 0.0]) * 0.6)
    return p


@asset("flowers_1", kind="flowers", collider=0.0, scale=(0.8, 1.25), sway=0.05, wind=("Grass", "Flower"))
def flowers_1(colors=FLOWER_SETS["flowers_1"], seed=51):
    """Small cluster, ~0.3 m: three chunky daisies on stems with two leaves (under 200 triangles)."""
    parts = []
    heads = [((0.015, 0.29, 0.0), 0.072), ((-0.11, 0.21, 0.075), 0.066), ((0.1, 0.165, -0.075), 0.062)]
    for i, (hc, R) in enumerate(heads):
        hc = np.asarray(hc, float)
        base = np.array([hc[0] * 0.3, -0.03, hc[2] * 0.3])
        out = np.array([hc[0], 0.0, hc[2]])
        parts.append(stem(base, hc - [0, 0.006, 0]))
        parts += blossom(hc, R, 5, colors, facing=(0, 1, 0) + out * 4.0, seed=seed + i)
    for az, h in ((35, 0.17), (220, 0.14)):
        b = blade((0, -0.02, 0), az, h, 0.055, 0.016, 0.5, curl=0.35)
        paint_grass(b, 0.22, seed=seed)
        parts.append(b)
    return parts, {}


@asset("flowers_2", kind="flowers", collider=0.0, scale=(0.8, 1.25), sway=0.05, wind=("Grass", "Flower"))
def flowers_2():
    """The same cluster in lilac (vertex colors only)."""
    return flowers_1(FLOWER_SETS["flowers_2"], seed=57)


@asset("flowers_3", kind="flowers", collider=0.0, scale=(0.8, 1.25), sway=0.05, wind=("Grass", "Flower"))
def flowers_3():
    """The same cluster in yellow."""
    return flowers_1(FLOWER_SETS["flowers_3"], seed=59)


@asset("stump_1", kind="stump", collider="auto", scale=(0.85, 1.25))
def stump_1():
    """Cut stump, ~0.45 m: rooty bark sides and a tilted pale wood top with growth rings."""
    seed = 61
    rng = np.random.default_rng(seed)
    segs = 20
    ph = rng.uniform(0, 2 * np.pi)
    H = 0.42
    tilt = np.array([0.08, -0.05])  # top plane slope (dy/dx, dy/dz)
    ys = [-0.1, 0.0, 0.1, 0.24, H]
    rad = [0.31, 0.32, 0.3, 0.29, 0.285]
    jit = rng.uniform(0.6, 1.25, 5)

    def mod(i, th):
        y = ys[i]
        root = ((th - ph) * 5 / (2 * np.pi) + 0.5).astype(int) % 5
        bump = (0.5 + 0.5 * np.cos(5 * (th - ph))) ** 1.5 * jit[root]
        fl = 0.42 * np.exp(-max(y, 0) / 0.16)
        k = 1 + fl * (0.12 + 0.8 * bump)
        x, z = rad[i] * k * np.cos(th), -rad[i] * k * np.sin(th)
        dy = (x * tilt[0] + z * tilt[1]) if i == len(ys) - 1 else -0.03 * bump * fl
        return k, dy, 0, 0

    side = revolve(list(zip(rad, ys)), segs, mod, close_top=False, phase=ph)
    paint_bark(side, 2.0, seed)
    side.C = ramp(BARK, 0.25 + 0.45 * smoothstep(-0.05, 0.35, side.V[:, 1]) + 0.08 * noise(side.V, 4, seed))
    # wood top: rim ring (bark section) + rings
    rim = side.V[side.tag["rings"][-1]]
    c = np.array([0.0, H, 0.0])
    rings = [rim, c + (rim - c) * 0.88, c + (rim - c) * 0.58, c + (rim - c) * 0.28]
    V = np.vstack(rings + [c[None]])
    V[:, 1] = H + V[:, 0] * tilt[0] + V[:, 2] * tilt[1]
    F = []
    for a in range(3):
        for j in range(segs):
            F.append((a * segs + j, a * segs + (j + 1) % segs, (a + 1) * segs + (j + 1) % segs, (a + 1) * segs + j))
    ctr = 4 * segs
    F += [(3 * segs + j, 3 * segs + (j + 1) % segs, ctr) for j in range(segs)]
    top = Part(V, F, "Wood")
    if np.dot(top.mesh().face_normals().mean(0), [0, 1, 0]) < 0:
        top.F = [tuple(reversed(f)) for f in top.F]
    n_up = m3.normalize(np.array([-tilt[0], 1.0, -tilt[1]]))
    top.N = np.tile(n_up, (len(V), 1))
    tc = np.concatenate([np.full(segs, -0.4), np.full(segs, 0.95), np.full(segs, 0.45), np.full(segs, 0.9), [0.3]])
    top.C = np.where((tc < 0)[:, None], rgb(BARK[0]) * 1.1, ramp(WOOD, np.clip(tc, 0, 1)))
    return [side, top], {}


@asset("log_1", kind="log", collider=0.0, scale=(0.85, 1.2))
def log_1():
    """Fallen log, ~1.9 m long along X: bark sides with a couple of knots, pale wood ends with rings.
    Collide with the capsule in nature.json (a single base radius does not fit a long log)."""
    seed = 71
    rng = np.random.default_rng(seed)
    segs = 12
    L, R = 1.9, 0.2
    xs = np.linspace(-L / 2, L / 2, 6)
    rad = R * np.array([1.12, 1.02, 1.0, 0.95, 0.9, 0.88])
    knots = [(0.35, 1.2, 0.3), (-0.4, 4.0, 0.25)]  # (x, angle, strength)

    def mod(i, th):
        k = np.ones_like(th)
        for kx, ka, ks in knots:
            k = k + ks * np.exp(-((xs[i] - kx) / 0.3) ** 2) * np.maximum(np.cos(th - ka), 0) ** 4
        bend = 0.09 * np.sin(np.pi * (xs[i] + L / 2) / L)  # a gentle bow sideways
        return k, 0, 0, bend

    side = revolve(list(zip(rad, xs)), segs, mod, close_bottom=False, close_top=False)
    rings = side.tag["rings"]
    parts = [side]
    for ri, sgn in ((0, -1.0), (-1, 1.0)):  # wood end caps: rim (bark section) + rings + center
        rim = side.V[rings[ri]]
        c = np.array([0.0, xs[ri] + sgn * 0.012, 0.0])
        rs = [rim, c + (rim - c) * 0.86, c + (rim - c) * 0.5]
        V = np.vstack(rs + [c[None]])
        V[segs:, 1] = c[1]
        F = [(a * segs + j, a * segs + (j + 1) % segs, (a + 1) * segs + (j + 1) % segs, (a + 1) * segs + j)
             for a in range(2) for j in range(segs)]
        F += [(2 * segs + j, 2 * segs + (j + 1) % segs, 3 * segs) for j in range(segs)]
        cap = Part(V, F, "Wood")
        if np.dot(cap.mesh().face_normals().mean(0), [0, sgn, 0]) < 0:
            cap.F = [tuple(reversed(f)) for f in cap.F]
        cap.N = np.tile([0.0, sgn, 0.0], (len(V), 1))
        tc = np.concatenate([np.full(segs, -1.0), np.full(segs, 0.9), np.full(segs, 0.45), [0.75]])
        cap.C = np.where((tc < 0)[:, None], rgb(BARK[0]) * 1.1, ramp(WOOD, np.clip(tc, 0, 1)))
        parts.append(cap)
    # lie it down along X, resting on the ground (slightly sunk)
    rot = m3.rotation_matrix((0, 0, -90))[:3, :3]
    for p in parts:
        p.V = p.V @ rot.T
        if p.N is not None:
            p.N = p.N @ rot.T
    # broken branch stub (starts inside the log)
    a = np.array([0.3, 0.0, 0.07])
    d = m3.normalize(np.array([0.35, 0.8, 0.45]))
    stub = tube([a, a + d * 0.28, a + d * 0.5 + [0.04, 0, 0]], [0.075, 0.06, 0.04], 6)
    parts.append(stub)
    low = min(p.V[:, 1].min() for p in parts)
    for p in parts:
        p.translate((0, -low - 0.03, 0))
    stub.C = ramp(BARK, 0.5 + 0.2 * stub.normals()[:, 1])
    P = side.V
    side.C = ramp(BARK, 0.3 + 0.35 * smoothstep(0.0, 0.4, P[:, 1]) + 0.15 * side.normals()[:, 1]
                  + 0.08 * noise(P, 3, seed))
    _ = rng
    return parts, {"capsule": [[-0.9, 0.17, 0.0], [0.9, 0.17, 0.0], 0.2]}


# ----------------------------------------------------------------------------- export / review


def export(name: str, parts: list[Part]) -> dict:
    mesh = build_mesh(name, parts)
    scene = p3.Scene(name)
    scene.add(mesh, name=name)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.glb"
    scene.export(path)
    lo, hi = mesh.bounds()
    mats: dict[str, int] = {}
    for p in parts:
        mats[p.mat] = mats.get(p.mat, 0) + p.tris()
    return {"mesh": mesh, "tris": mesh.num_triangles, "height": round(float(hi[1]), 3),
            "radius": round(float(np.linalg.norm(mesh.vertices[:, [0, 2]], axis=1).max()), 3),
            "bounds": [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]],
            "materials": mats, "closed_parts": f"{sum(p.closed() for p in parts)}/{len(parts)}",
            "bytes": path.stat().st_size}


def main(argv):
    names = [a for a in argv if not a.startswith("--")] or list(ASSETS)
    results = {}
    for name in names:
        parts, extra = ASSETS[name]["fn"]()
        info = export(name, parts)
        info["extra"] = extra
        results[name] = info
        print(f"{name:10s} tris={info['tris']:5d} h={info['height']:.2f} r={info['radius']:.2f} "
              f"mats={info['materials']} closed={info['closed_parts']} {info['bytes'] // 1024} KB")
    write_manifest(results)
    if "--no-review" not in argv:
        review(results)
    return results


def auto_collider(kind: str, mesh: p3.Mesh, height: float) -> float:
    """Radius of the solid cylinder at the base, measured on the geometry: trunk at 0.3-1.0 m for
    trees (0.25-1.2 m), the base band of rocks and stumps, a soft core for bushes."""
    V = mesh.vertices
    r = np.linalg.norm(V[:, [0, 2]], axis=1)
    fm = np.zeros(len(V), int)
    for f, m in zip(mesh.faces, mesh.face_materials):
        fm[list(f)] = m
    bark = np.array([mesh.materials[k].name == "Bark" for k in fm])
    if kind == "tree":  # trunk rings between the root flare and the fork
        sel = bark & (V[:, 1] > 0.25) & (V[:, 1] < 1.2) & (r < 0.8)
        return round(float(r[sel].mean()), 2) if sel.any() else 0.3
    if kind == "stump":
        sel = bark & (V[:, 1] > 0.12) & (V[:, 1] < 0.35)
        return round(float(r[sel].mean()), 2)
    if kind == "rock":
        sel = V[:, 1] < 0.35 * height
        return round(float(r[sel].mean() * 0.9), 2)
    if kind == "bush":
        return round(float(r.max() * 0.5), 2)
    return 0.0


def write_manifest(results):
    path = OUT / "nature.json"
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault("assets", {})
    for name, info in results.items():
        meta = ASSETS[name]
        entry = {
            "file": f"{name}.glb",
            "kind": meta["kind"],
            "collider": (auto_collider(meta["kind"], info["mesh"], info["height"]) if meta["collider"] == "auto"
                         else meta["collider"]),
            "height": info["height"],
            "radius": info["radius"],
            "tris": info["tris"],
            "scale": list(meta["scale"]),
            "materials": sorted(info["materials"]),
        }
        if meta.get("sway"):
            entry["wind"] = {"sway": meta["sway"],
                             "materials": list(meta.get("wind", [m for m in info["materials"] if m in ("Leaves", "Grass")]))}
        entry.update(info["extra"])
        data["assets"][name] = entry
    data["about"] = ("Pyram3D nature set for Niryk prototype 2 (projects/niryk/protótipo_2/nature.py). Meters, +Y up, "
                     "origin at the base center. collider: radius of the solid cylinder at the base (0 = walk "
                     "through). radius: visual footprint. scale: suggested uniform random scale range. wind.sway: "
                     "suggested top displacement in meters for the listed materials, weighted by height above the "
                     "ground (vertex y / height). canopy: [min y, max y] of the leaves. Colors are vertex colors "
                     "(COLOR_0); materials are white.")
    data["assets"] = dict(sorted(data["assets"].items()))
    lines = ",\n".join(f'  "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in data["assets"].items())
    path.write_text('{\n "about": ' + json.dumps(data["about"], ensure_ascii=False) + ',\n "assets": {\n' + lines
                    + "\n }\n}\n")


def _ground(radius=40.0, color="#78a04a"):
    g = p3.disc(radius, 64)
    g.set_material(p3.Material("Ground", color, roughness=1.0))
    return g


INK = rgb("#1b1424")


def _ink_outline(img, lin_depth, objmap, footprint, ss):
    """Review-only replacement for the preview renderer's ink pass: lines where the depth has a
    kink (second difference), on silhouettes and between objects. Unlike a first-difference test it
    leaves flat surfaces seen at grazing angles clean, closer to the game's inverted-hull outline."""
    fin = np.where(np.isfinite(lin_depth), lin_depth, 1e4)
    obj = fin < 1e4
    edge = np.zeros(fin.shape, bool)
    for ax in (0, 1):
        a, c = np.roll(fin, 1, axis=ax), np.roll(fin, -1, axis=ax)
        d2 = np.abs(a + c - 2 * fin)
        e = d2 > 6.0 * footprint * fin
        e |= (np.roll(objmap, 1, axis=ax) != objmap) | (np.roll(objmap, -1, axis=ax) != objmap)
        edge |= e & obj
    grow = edge.copy()
    for s in range(1, max(ss, 1) + 1):
        grow |= np.roll(edge, s, axis=0) | np.roll(edge, s, axis=1)
    grow &= obj | np.roll(obj, 1, 0) | np.roll(obj, 1, 1)
    img = img.copy()
    img[grow] = img[grow] * 0.15 + INK * 0.85
    return img


def toon(scene, path, **kw):
    mod = sys.modules["pyram3d.render"]
    old = mod._outline
    mod._outline = _ink_outline
    try:
        return p3.render(scene, path, style="toon", **kw)
    finally:
        mod._outline = old


def review(results):
    REVIEW.mkdir(parents=True, exist_ok=True)
    cap = p3.capsule(0.25, 1.8).translate((0, 0.9, 0))
    cap.set_material(p3.Material("Human", "#d08a63", roughness=0.8))
    sky = "#bcd3e0"
    # 1) every asset in a row with a 1.8 m capsule
    scene = p3.Scene("NatureSheet")
    order = [n for n in ("oak_1", "oak_2", "pine_1", "bush_1", "rock_2", "rock_1", "stump_1", "log_1", "flowers_1",
                         "flowers_2", "flowers_3", "grass_1") if n in results]
    x, prev = 0.0, 0.0
    for name in order:
        r = min(results[name]["radius"], 2.4)
        x += prev + r + 0.3
        scene.add(results[name]["mesh"], position=(x, 0, 0))
        prev = r
    scene.add(cap, position=(x + prev + 0.7, 0, 0))
    scene.add_default_lights()
    toon(scene, REVIEW / "nature_sheet.png", width=1200, height=520, view=(0.0, 0.18, 1.0), yfov=24, background=sky)
    # 2) small assets close up with the capsule
    small = [n for n in ("bush_1", "rock_2", "rock_1", "stump_1", "log_1", "flowers_1", "flowers_2", "flowers_3",
                         "grass_1")
             if n in results]
    if small:
        scene = p3.Scene("NatureSmall")
        x, prev = 0.0, 0.0
        for name in small:
            r = results[name]["radius"]
            x += prev + r + 0.2
            scene.add(results[name]["mesh"], position=(x, 0, 0))
            prev = r
        scene.add(cap, position=(x + prev + 0.45, 0, 0))
        scene.add_default_lights()
        toon(scene, REVIEW / "nature_small.png", width=1200, height=520, view=(0.0, 0.5, 1.0), yfov=22, background=sky)
    if all(n in results for n in ("oak_1", "oak_2", "pine_1", "bush_1", "rock_1", "rock_2", "grass_1", "flowers_1")):
        clearing(results, cap, sky)
    if all(n in results for n in ("grass_1", "flowers_1", "flowers_2", "flowers_3", "rock_1", "stump_1")):
        ground_closeup(results, sky)


def ground_closeup(results, sky):
    """Grass, flowers, a stone and the stump at the player's feet (camera 1 m up, 1.6 m back)."""
    scene = p3.Scene("Ground")
    scene.add(_ground(6))
    spots = {"grass_1": [(-0.5, 0.2), (0.55, -0.1), (0.0, 0.9), (-0.9, 1.1)], "flowers_1": [(-0.15, 0.1)],
             "flowers_2": [(0.3, 0.5)], "flowers_3": [(-0.55, 0.7)], "rock_1": [(0.9, 0.9)], "stump_1": [(-1.3, 0.3)]}
    for name, ps in spots.items():
        for i, (x, z) in enumerate(ps):
            scene.add(results[name]["mesh"], position=(x, 0, z), rotation=(0, 40 * i + 15, 0))
    cam = scene.add_camera(position=(0.0, 1.0, -1.6), target=(-0.1, 0.1, 0.5), yfov=45)
    scene.add_light("directional", position=(-6, 9, -4), target=(0, 0, 0), color="#fff1d6", intensity=3.2)
    scene.add_light("directional", position=(6, 4, 10), target=(0, 0, 0), color="#9fb7e8", intensity=0.8,
                    cast_shadows=False)
    toon(scene, REVIEW / "nature_ground.png", width=1000, height=600, camera=cam, background=sky)


def clearing(results, cap, sky):
    """The clearing from the game's over-the-shoulder camera (0.45 m right, 1.6 m up, 2.6 m behind)."""
    rng = np.random.default_rng(3)
    scene = p3.Scene("Clearing")
    scene.add(_ground())
    M = {k: v["mesh"] for k, v in results.items()}
    place = [("oak_1", (-5.5, 9.0), 0, 1.0), ("oak_2", (6.5, 12.0), 40, 1.05), ("pine_1", (-1.5, 17.0), 0, 1.0),
             ("pine_1", (11.0, 20.0), 70, 0.9), ("oak_1", (-12.0, 18.0), 130, 1.15), ("bush_1", (-2.6, 5.2), 30, 1.0),
             ("bush_1", (3.8, 7.5), 200, 1.2), ("rock_2", (2.4, 4.4), 25, 1.0), ("rock_1", (-1.2, 3.4), 70, 1.0),
             ("rock_1", (3.6, 4.9), 180, 0.7)]
    if "stump_1" in M:
        place.append(("stump_1", (-3.6, 3.2), 10, 1.0))
    if "log_1" in M:
        place.append(("log_1", (-4.2, 7.0), 75, 1.0))
    for name, (x, z), yaw, s in place:
        scene.add(M[name], position=(x, 0, z), rotation=(0, yaw, 0), scale=s)
    flowers = [n for n in ("flowers_1", "flowers_2", "flowers_3") if n in M]
    for _ in range(90):
        x, z = rng.uniform(-7, 7), rng.uniform(0.8, 12)
        if abs(x) < 0.5 and z < 3.2:
            continue
        scene.add(M["grass_1"], position=(x, 0, z), rotation=(0, rng.uniform(0, 360), 0), scale=rng.uniform(0.75, 1.3))
    for k in range(26):
        x, z = rng.uniform(-6, 6), rng.uniform(0.8, 9)
        scene.add(M[flowers[k % len(flowers)]], position=(x, 0, z), rotation=(0, rng.uniform(0, 360), 0),
                  scale=rng.uniform(0.85, 1.2))
    scene.add(cap, position=(0, 0, 2.6))
    cam = scene.add_camera(position=(-0.45, 1.6, 0.0), target=(-0.45 + 0.1, 1.25, 6.0), yfov=55)
    scene.add_light("directional", position=(-6, 9, -4), target=(0, 0, 6), color="#fff1d6", intensity=3.2)
    scene.add_light("directional", position=(6, 4, 10), target=(0, 0, 6), color="#9fb7e8", intensity=0.8,
                    cast_shadows=False)
    toon(scene, REVIEW / "nature_clearing.png", width=1200, height=675, camera=cam, background=sky, fog=0.012)


if __name__ == "__main__":
    main(sys.argv[1:])
