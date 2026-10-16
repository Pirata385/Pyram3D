"""Signed-distance-field (SDF) modeling: organic shapes, smooth blends and booleans.

Build a shape by combining primitives, then call .to_mesh(). Ideal for
characters, animals, creatures, rocks, fruit, smooth stylized props and
any boolean operation (union / subtract / intersect).

    body = sdf.capsule((0, 0.9, 0), (0, 1.4, 0), 0.18)
    head = sdf.sphere((0, 1.65, 0), 0.13)
    char = body.smooth_union(head, 0.05)
    mesh = char.to_mesh(resolution=96)

Every primitive can carry a `color` (sRGB) and/or `material`; the mesh gets
vertex colors / per-face materials from the closest primitive.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from . import math3d as m3
from .mesh import Mesh


class SDF:
    """Base class: callable mapping points (N,3) -> signed distances (N,)."""

    color = None
    material = None
    bone = None

    def __call__(self, p: np.ndarray) -> np.ndarray:
        return self.dist(np.asarray(p, dtype=np.float64))

    def dist(self, p: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
        raise NotImplementedError

    # ---- attribute (color/material) query: returns (dist, leaf index) for the leaf primitives
    def leaves(self) -> list["SDF"]:
        return [self]

    def leaf_dists(self, p: np.ndarray) -> np.ndarray:
        """(N, L) distance of p to every leaf primitive (respecting transforms)."""
        return self.dist(p)[:, None]

    # ---- operators
    def union(self, *others: "SDF") -> "SDF":
        return Union([self, *others])

    def smooth_union(self, other: "SDF", k: float = 0.05) -> "SDF":
        return SmoothUnion(self, other, k)

    def subtract(self, other: "SDF") -> "SDF":
        return Subtract(self, other, 0.0)

    def smooth_subtract(self, other: "SDF", k: float = 0.05) -> "SDF":
        return Subtract(self, other, k)

    def intersect(self, other: "SDF") -> "SDF":
        return Intersect(self, other, 0.0)

    def smooth_intersect(self, other: "SDF", k: float = 0.05) -> "SDF":
        return Intersect(self, other, k)

    __or__ = union
    __sub__ = subtract
    __and__ = intersect

    def translate(self, offset) -> "SDF":
        return Transformed(self, m3.translation_matrix(offset))

    def rotate(self, rotation, pivot=(0, 0, 0)) -> "SDF":
        p = m3.vec3(pivot)
        return Transformed(self, m3.translation_matrix(p) @ m3.rotation_matrix(rotation) @ m3.translation_matrix(-p))

    def scale(self, factor: float) -> "SDF":
        return Transformed(self, m3.scale_matrix(factor))

    def mirror_x(self) -> "SDF":
        """Symmetric copy across x=0 (model one side of a character)."""
        return MirrorX(self)

    def round(self, radius: float) -> "SDF":
        return Rounded(self, radius)

    def shell(self, thickness: float) -> "SDF":
        """Hollow shell (onion) of the given thickness."""
        return Shell(self, thickness)

    def displace(self, amount: float = 0.02, scale: float = 4.0, octaves: int = 4, seed: int = 0,
                 func: Callable[[np.ndarray], np.ndarray] | None = None) -> "SDF":
        """Add noise (or func(points)->(N,)) to the surface: bumps, rock roughness, bark."""
        return Displaced(self, amount, scale, octaves, seed, func)

    def paint(self, color=None, material=None, bone=None) -> "SDF":
        """Set the color/material/bone of every leaf of this shape."""
        for leaf in self.leaves():
            if color is not None:
                leaf.color = color
            if material is not None:
                leaf.material = material
            if bone is not None:
                leaf.bone = bone
        return self

    # ---- meshing
    def to_mesh(self, resolution: int = 96, bounds=None, padding: float = 0.05, smooth: bool = True,
                name: str = "SDFMesh", project_iterations: int = 2, colors: bool | None = None,
                default_color="#cccccc") -> Mesh:
        """Polygonize with naive surface nets (quad mesh).

        resolution: grid cells along the longest axis (64 low, 96 default, 160+ detailed).
        """
        if bounds is None:
            lo, hi = self.bounds()
        else:
            lo, hi = (m3.vec3(b) for b in bounds)
        ext = hi - lo
        pad = ext.max() * padding + 1e-6
        lo = lo - pad
        hi = hi + pad
        h = (hi - lo).max() / resolution
        n = np.maximum(np.ceil((hi - lo) / h).astype(int) + 1, 2)
        mesh = surface_nets(self, lo, h, n, name=name, project_iterations=project_iterations)
        if not smooth:
            mesh.smooth_angle = 0.0
        self._assign_attributes(mesh, colors, default_color)
        return mesh

    def _assign_attributes(self, mesh: Mesh, colors, default_color):
        leaves = self.leaves()
        has_color = any(l.color is not None for l in leaves)
        has_mat = any(l.material is not None for l in leaves)
        has_bone = any(l.bone is not None for l in leaves)
        if not (has_color or has_mat or has_bone) or len(mesh.vertices) == 0:
            return
        if has_bone:
            # per-vertex membership to named bones: soft near seams, exact inside parts
            d = np.abs(self.leaf_dists(mesh.vertices))
            names = sorted({l.bone for l in leaves if l.bone is not None})
            k = max(float(mesh.size.max()) * 0.004, 1e-4)
            dmin = d.min(axis=1, keepdims=True)
            W = np.zeros((len(mesh.vertices), len(names)))
            for j, leaf in enumerate(leaves):
                if leaf.bone is not None:
                    W[:, names.index(leaf.bone)] = np.maximum(W[:, names.index(leaf.bone)],
                                                              np.exp(-(d[:, j] - dmin[:, 0]) / k))
            # vertices whose nearest part has no bone get no part weights (distance fallback in bind)
            unowned = np.ones(len(mesh.vertices), dtype=bool)
            for j, leaf in enumerate(leaves):
                if leaf.bone is not None:
                    unowned &= d[:, j] > dmin[:, 0] + 1e-9
            W[unowned] = 0.0
            mesh.extras["bone_parts"] = (names, W)
        if has_color and colors is not False:
            d = np.abs(self.leaf_dists(mesh.vertices))  # (N, L)
            # soft blend of leaf colors near seams
            k = max(float((mesh.size.max()) * 0.0015), 1e-4)
            w = np.exp(-(d - d.min(axis=1, keepdims=True)) / k)
            w /= w.sum(axis=1, keepdims=True)
            cols = []
            for leaf in leaves:
                c = m3.as_color(leaf.color if leaf.color is not None else default_color)
                c[:3] = m3.srgb_to_linear(c[:3])
                cols.append(c)
            mesh.colors = w @ np.array(cols)
        if has_mat:
            fc = mesh.face_centers()
            d = np.abs(self.leaf_dists(fc))
            nearest = np.argmin(d, axis=1)
            mats = []
            leaf_mat_idx = []
            for leaf in leaves:
                mt = leaf.material
                for j, ex in enumerate(mats):
                    if ex is mt:
                        leaf_mat_idx.append(j)
                        break
                else:
                    mats.append(mt)
                    leaf_mat_idx.append(len(mats) - 1)
            mesh.materials = mats
            mesh.face_materials = np.array(leaf_mat_idx, dtype=np.int64)[nearest]


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

class _Prim(SDF):
    def __init__(self, color=None, material=None, bone=None):
        self.color = color
        self.material = material
        self.bone = bone  # name of the skeleton bone this part follows (see Skeleton.bind)


class Sphere(_Prim):
    def __init__(self, center=(0, 0, 0), radius: float = 0.5, **kw):
        super().__init__(**kw)
        self.c = m3.vec3(center)
        self.r = float(radius)

    def dist(self, p):
        return np.linalg.norm(p - self.c, axis=1) - self.r

    def bounds(self):
        return self.c - self.r, self.c + self.r


class Ellipsoid(_Prim):
    def __init__(self, center=(0, 0, 0), radii=(0.5, 0.3, 0.4), **kw):
        super().__init__(**kw)
        self.c = m3.vec3(center)
        self.r = m3.vec3(radii)

    def dist(self, p):
        q = p - self.c
        k0 = np.linalg.norm(q / self.r, axis=1)
        k1 = np.linalg.norm(q / (self.r * self.r), axis=1)
        return k0 * (k0 - 1.0) / np.maximum(k1, 1e-12)

    def bounds(self):
        return self.c - self.r, self.c + self.r


class Box(_Prim):
    def __init__(self, center=(0, 0, 0), size=(1, 1, 1), radius: float = 0.0, **kw):
        super().__init__(**kw)
        self.c = m3.vec3(center)
        self.h = m3.vec3(size) / 2
        self.r = float(radius)

    def dist(self, p):
        q = np.abs(p - self.c) - (self.h - self.r)
        outside = np.linalg.norm(np.maximum(q, 0), axis=1)
        inside = np.minimum(q.max(axis=1), 0)
        return outside + inside - self.r

    def bounds(self):
        return self.c - self.h, self.c + self.h


class Capsule(_Prim):
    """Segment a-b with radius r (limbs, fingers, horns, branches)."""

    def __init__(self, a=(0, 0, 0), b=(0, 1, 0), radius: float = 0.1, **kw):
        super().__init__(**kw)
        self.a = m3.vec3(a)
        self.b = m3.vec3(b)
        self.r = float(radius)

    def dist(self, p):
        pa = p - self.a
        ba = self.b - self.a
        h = np.clip((pa @ ba) / max(ba @ ba, 1e-12), 0, 1)
        return np.linalg.norm(pa - h[:, None] * ba, axis=1) - self.r

    def bounds(self):
        return np.minimum(self.a, self.b) - self.r, np.maximum(self.a, self.b) + self.r


class RoundCone(_Prim):
    """Tapered capsule from a (radius r1) to b (radius r2): arms, legs, tails, necks, tentacles."""

    def __init__(self, a=(0, 0, 0), b=(0, 1, 0), r1: float = 0.2, r2: float = 0.1, **kw):
        super().__init__(**kw)
        self.a = m3.vec3(a)
        self.b = m3.vec3(b)
        self.r1 = float(r1)
        self.r2 = float(r2)

    def dist(self, p):
        ba = self.b - self.a
        l2 = float(ba @ ba)
        rr = self.r1 - self.r2
        a2 = l2 - rr * rr
        il2 = 1.0 / max(l2, 1e-12)
        pa = p - self.a
        y = pa @ ba
        z = y - l2
        xv = pa * l2 - y[:, None] * ba
        x2 = np.einsum("ij,ij->i", xv, xv)
        y2 = y * y * l2
        z2 = z * z * l2
        k = np.sign(rr) * rr * rr * x2
        res = (np.sqrt(np.maximum(x2 * a2 * il2, 0)) + y * rr) * il2 - self.r1
        c1 = np.sign(z) * a2 * z2 > k
        c2 = np.sign(y) * a2 * y2 < k
        res = np.where(c1, np.sqrt(np.maximum(x2 + z2, 0)) * il2 - self.r2, res)
        res = np.where(~c1 & c2, np.sqrt(np.maximum(x2 + y2, 0)) * il2 - self.r1, res)
        return res

    def bounds(self):
        r = max(self.r1, self.r2)
        return np.minimum(self.a, self.b) - r, np.maximum(self.a, self.b) + r


class Cylinder(_Prim):
    """Capped cylinder between a and b (optionally rounded edges)."""

    def __init__(self, a=(0, 0, 0), b=(0, 1, 0), radius: float = 0.2, rounding: float = 0.0, **kw):
        super().__init__(**kw)
        self.a = m3.vec3(a)
        self.b = m3.vec3(b)
        self.r = float(radius)
        self.round = float(rounding)

    def dist(self, p):
        ba = self.b - self.a
        L = np.linalg.norm(ba)
        axis = ba / max(L, 1e-12)
        pa = p - self.a
        y = pa @ axis - L / 2
        x = np.linalg.norm(pa - np.outer(pa @ axis, axis), axis=1)
        rr = self.round
        dx = x - (self.r - rr)
        dy = np.abs(y) - (L / 2 - rr)
        outside = np.hypot(np.maximum(dx, 0), np.maximum(dy, 0))
        return outside + np.minimum(np.maximum(dx, dy), 0) - rr

    def bounds(self):
        return np.minimum(self.a, self.b) - self.r, np.maximum(self.a, self.b) + self.r


class Torus(_Prim):
    """Torus around the Y axis."""

    def __init__(self, center=(0, 0, 0), major: float = 0.5, minor: float = 0.15, **kw):
        super().__init__(**kw)
        self.c = m3.vec3(center)
        self.R = float(major)
        self.r = float(minor)

    def dist(self, p):
        q = p - self.c
        xz = np.hypot(q[:, 0], q[:, 2]) - self.R
        return np.hypot(xz, q[:, 1]) - self.r

    def bounds(self):
        e = np.array([self.R + self.r, self.r, self.R + self.r])
        return self.c - e, self.c + e


class Plane(_Prim):
    """Half-space below the plane through `point` with `normal` (use with intersect to cut)."""

    def __init__(self, point=(0, 0, 0), normal=(0, 1, 0), extent: float = 10.0, **kw):
        super().__init__(**kw)
        self.p = m3.vec3(point)
        self.n = m3.normalize(m3.vec3(normal))
        self.extent = extent

    def dist(self, p):
        return (p - self.p) @ self.n

    def bounds(self):
        return self.p - self.extent, self.p + self.extent


class Custom(_Prim):
    """Wrap any function f(points)->(N,) as an SDF (you provide the bounds)."""

    def __init__(self, func, lo, hi, **kw):
        super().__init__(**kw)
        self.f = func
        self.lo, self.hi = m3.vec3(lo), m3.vec3(hi)

    def dist(self, p):
        return np.asarray(self.f(p), dtype=np.float64)

    def bounds(self):
        return self.lo, self.hi


# ---------------------------------------------------------------------------
# operators
# ---------------------------------------------------------------------------

def _smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def _smax(a, b, k):
    return -_smin(-a, -b, k)


class _Op(SDF):
    def __init__(self, children):
        self.children = children

    def leaves(self):
        out = []
        for c in self.children:
            out += c.leaves()
        return out

    def leaf_dists(self, p):
        return np.concatenate([c.leaf_dists(p) for c in self.children], axis=1)


class Union(_Op):
    def dist(self, p):
        d = self.children[0].dist(p)
        for c in self.children[1:]:
            d = np.minimum(d, c.dist(p))
        return d

    def bounds(self):
        bs = [c.bounds() for c in self.children]
        return np.min([b[0] for b in bs], axis=0), np.max([b[1] for b in bs], axis=0)


class SmoothUnion(_Op):
    def __init__(self, a, b, k):
        super().__init__([a, b])
        self.k = k

    def dist(self, p):
        return _smin(self.children[0].dist(p), self.children[1].dist(p), self.k)

    def bounds(self):
        (a0, a1), (b0, b1) = self.children[0].bounds(), self.children[1].bounds()
        return np.minimum(a0, b0) - self.k, np.maximum(a1, b1) + self.k


class Subtract(_Op):
    def __init__(self, a, b, k):
        super().__init__([a, b])
        self.k = k

    def dist(self, p):
        return _smax(self.children[0].dist(p), -self.children[1].dist(p), self.k)

    def leaf_dists(self, p):
        # material of the cut surface comes from the cutter
        return np.concatenate([self.children[0].leaf_dists(p), self.children[1].leaf_dists(p)], axis=1)

    def bounds(self):
        return self.children[0].bounds()


class Intersect(_Op):
    def __init__(self, a, b, k):
        super().__init__([a, b])
        self.k = k

    def dist(self, p):
        return _smax(self.children[0].dist(p), self.children[1].dist(p), self.k)

    def bounds(self):
        (a0, a1), (b0, b1) = self.children[0].bounds(), self.children[1].bounds()
        return np.maximum(a0, b0), np.minimum(a1, b1)


class Transformed(_Op):
    def __init__(self, child, matrix):
        super().__init__([child])
        self.m = np.asarray(matrix, dtype=np.float64)
        self.inv = np.linalg.inv(self.m)
        # uniform scale factor to keep distances metric
        self.s = float(np.cbrt(abs(np.linalg.det(self.m[:3, :3]))))

    def dist(self, p):
        return self.children[0].dist(m3.transform_points(self.inv, p)) * self.s

    def leaf_dists(self, p):
        return self.children[0].leaf_dists(m3.transform_points(self.inv, p)) * self.s

    def bounds(self):
        lo, hi = self.children[0].bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        c = m3.transform_points(self.m, corners)
        return c.min(axis=0), c.max(axis=0)


class MirrorX(_Op):
    def __init__(self, child):
        super().__init__([child])

    def _fold(self, p):
        q = p.copy()
        q[:, 0] = np.abs(q[:, 0])
        return q

    def dist(self, p):
        return self.children[0].dist(self._fold(p))

    def leaf_dists(self, p):
        return self.children[0].leaf_dists(self._fold(p))

    def bounds(self):
        lo, hi = self.children[0].bounds()
        m = max(abs(lo[0]), abs(hi[0]))
        return np.array([-m, lo[1], lo[2]]), np.array([m, hi[1], hi[2]])


class Rounded(_Op):
    def __init__(self, child, r):
        super().__init__([child])
        self.r = r

    def dist(self, p):
        return self.children[0].dist(p) - self.r

    def bounds(self):
        lo, hi = self.children[0].bounds()
        return lo - self.r, hi + self.r


class Shell(_Op):
    def __init__(self, child, t):
        super().__init__([child])
        self.t = t

    def dist(self, p):
        return np.abs(self.children[0].dist(p)) - self.t / 2

    def bounds(self):
        lo, hi = self.children[0].bounds()
        return lo - self.t, hi + self.t


class Displaced(_Op):
    def __init__(self, child, amount, scale, octaves, seed, func):
        super().__init__([child])
        self.amount, self.sc, self.oct, self.seed, self.func = amount, scale, octaves, seed, func

    def dist(self, p):
        from .noise import fbm3
        h = self.func(p) if self.func is not None else fbm3(p * self.sc, self.oct, seed=self.seed)
        return self.children[0].dist(p) + np.asarray(h) * self.amount

    def bounds(self):
        lo, hi = self.children[0].bounds()
        a = abs(self.amount) * 1.5
        return lo - a, hi + a


# ---------------------------------------------------------------------------
# convenience constructors (lowercase, what scripts should use)
# ---------------------------------------------------------------------------

def sphere(center=(0, 0, 0), radius=0.5, **kw) -> SDF:
    return Sphere(center, radius, **kw)


def ellipsoid(center=(0, 0, 0), radii=(0.5, 0.3, 0.4), **kw) -> SDF:
    return Ellipsoid(center, radii, **kw)


def box(center=(0, 0, 0), size=(1, 1, 1), radius=0.0, **kw) -> SDF:
    return Box(center, size, radius, **kw)


def capsule(a=(0, 0, 0), b=(0, 1, 0), radius=0.1, **kw) -> SDF:
    return Capsule(a, b, radius, **kw)


def round_cone(a=(0, 0, 0), b=(0, 1, 0), r1=0.2, r2=0.1, **kw) -> SDF:
    return RoundCone(a, b, r1, r2, **kw)


def cylinder(a=(0, 0, 0), b=(0, 1, 0), radius=0.2, rounding=0.0, **kw) -> SDF:
    return Cylinder(a, b, radius, rounding, **kw)


def torus(center=(0, 0, 0), major=0.5, minor=0.15, **kw) -> SDF:
    return Torus(center, major, minor, **kw)


def plane(point=(0, 0, 0), normal=(0, 1, 0), **kw) -> SDF:
    return Plane(point, normal, **kw)


def custom(func, lo, hi, **kw) -> SDF:
    return Custom(func, lo, hi, **kw)


def union(*shapes: SDF) -> SDF:
    return Union(list(shapes))


def smooth_union(*shapes: SDF, k: float = 0.05) -> SDF:
    out = shapes[0]
    for s in shapes[1:]:
        out = SmoothUnion(out, s, k)
    return out


def chain(points, radii, k: float = 0.02, **kw) -> SDF:
    """Smoothly joined round-cones through a list of points (tails, tentacles, spines, snakes)."""
    pts = [m3.vec3(p) for p in points]
    rs = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(pts),))
    parts = [RoundCone(pts[i], pts[i + 1], rs[i], rs[i + 1], **kw) for i in range(len(pts) - 1)]
    return smooth_union(*parts, k=k) if len(parts) > 1 else parts[0]


# ---------------------------------------------------------------------------
# surface nets
# ---------------------------------------------------------------------------

def surface_nets(f: SDF, lo: np.ndarray, h: float, n: np.ndarray, name: str = "SDFMesh",
                 project_iterations: int = 2, chunk: int = 2_000_000) -> Mesh:
    nx, ny, nz = (int(v) for v in n)
    xs = lo[0] + np.arange(nx) * h
    ys = lo[1] + np.arange(ny) * h
    zs = lo[2] + np.arange(nz) * h
    G = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    vals = np.empty(len(G))
    for s in range(0, len(G), chunk):
        vals[s:s + chunk] = f.dist(G[s:s + chunk])
    V = vals.reshape(nx, ny, nz)
    inside = V < 0

    # cells with a sign change
    c = np.zeros((nx - 1, ny - 1, nz - 1), dtype=np.int8)
    corners = [(i, j, k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    for (i, j, k) in corners:
        c += inside[i:nx - 1 + i, j:ny - 1 + j, k:nz - 1 + k]
    active = (c > 0) & (c < 8)
    idx = np.argwhere(active)
    if len(idx) == 0:
        return Mesh(name=name)
    vid = -np.ones(active.shape, dtype=np.int64)
    vid[active] = np.arange(len(idx))

    # vertex position = mean of edge crossings within the cell
    acc = np.zeros((len(idx), 3))
    cnt = np.zeros(len(idx))
    cell_edges = []
    for (a, b) in [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
                   ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
                   ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]:
        cell_edges.append((np.array(a), np.array(b)))
    for a, b in cell_edges:
        ia = idx + a
        ib = idx + b
        va = V[ia[:, 0], ia[:, 1], ia[:, 2]]
        vb = V[ib[:, 0], ib[:, 1], ib[:, 2]]
        cross = (va < 0) != (vb < 0)
        t = np.where(cross, va / np.where(np.abs(va - vb) < 1e-12, 1e-12, va - vb), 0.0)
        p = (ia + (ib - ia) * t[:, None]) * h + lo
        acc[cross] += p[cross]
        cnt[cross] += 1
    verts = acc / np.maximum(cnt, 1)[:, None]

    # quads: one per grid edge with a sign change, connecting the 4 cells around it
    faces = []
    # x-edges: between (i,j,k) and (i+1,j,k); cells (i, j-1..j, k-1..k)
    for axis in range(3):
        sl0 = [slice(None)] * 3
        sl1 = [slice(None)] * 3
        sl0[axis] = slice(0, -1)
        sl1[axis] = slice(1, None)
        e0 = inside[tuple(sl0)]
        e1 = inside[tuple(sl1)]
        sign = e0 != e1
        o = [k for k in range(3) if k != axis]
        # edges need neighbors at -1 in both other axes
        sel = np.argwhere(sign)
        sel = sel[(sel[:, o[0]] >= 1) & (sel[:, o[1]] >= 1)
                  & (sel[:, o[0]] <= active.shape[o[0]] - 1 + 0) & (sel[:, o[1]] <= active.shape[o[1]] - 1 + 0)]
        sel = sel[(sel[:, o[0]] < n[o[0]] - 1) & (sel[:, o[1]] < n[o[1]] - 1)]
        if len(sel) == 0:
            continue
        d0 = np.zeros(3, dtype=np.int64)
        d1 = np.zeros(3, dtype=np.int64)
        d0[o[0]] = 1
        d1[o[1]] = 1
        c00 = sel - d0 - d1
        c10 = sel - d1
        c11 = sel.copy()
        c01 = sel - d0
        q = np.stack([vid[tuple(cc.T)] for cc in (c00, c10, c11, c01)], axis=1)
        ok = (q >= 0).all(axis=1)
        q = q[ok]
        flip = e0[tuple(sel[ok].T)]  # inside at the low end -> outward normal along +axis
        # orientation: (c00,c10,c11,c01) normal is along o0 x o1 == +axis for axis order (x: y,z), (y: x,z -> -y), (z: x,y)
        handed = 1 if axis != 1 else -1
        want_pos = flip  # surface normal points toward outside = +axis when low end is inside
        rev = (want_pos & (handed < 0)) | (~want_pos & (handed > 0))
        q[rev] = q[rev][:, ::-1]
        faces.append(q)
    F = np.concatenate(faces) if faces else np.zeros((0, 4), dtype=np.int64)

    # project vertices onto the zero level set
    for _ in range(project_iterations):
        e = h * 0.25
        d = f.dist(verts)
        g = np.stack([f.dist(verts + [e, 0, 0]) - f.dist(verts - [e, 0, 0]),
                      f.dist(verts + [0, e, 0]) - f.dist(verts - [0, e, 0]),
                      f.dist(verts + [0, 0, e]) - f.dist(verts - [0, 0, e])], axis=1) / (2 * e)
        gl2 = np.maximum(np.einsum("ij,ij->i", g, g), 1e-12)
        step = (d / gl2)[:, None] * g
        step_len = np.linalg.norm(step, axis=1, keepdims=True)
        step = step * np.minimum(1.0, h / np.maximum(step_len, 1e-12))
        verts = verts - step

    mesh = Mesh(verts, [tuple(r) for r in F.tolist()], name=name, smooth_angle=180.0)
    return mesh
