"""Primitive and procedural shape builders.

Every function returns a new Mesh, centered on the origin unless stated
otherwise, with outward-facing normals and UVs.
Sizes are in meters. +Y is up.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from . import math3d as m3
from .mesh import Mesh


def _size3(size) -> np.ndarray:
    return m3.vec3(size)


# ----------------------------------------------------------------- boxes
def _grid_quads(nu: int, nv: int, offset: int = 0) -> list[tuple[int, int, int, int]]:
    faces = []
    for j in range(nv):
        for i in range(nu):
            a = offset + j * (nu + 1) + i
            faces.append((a, a + 1, a + nu + 2, a + nu + 1))
    return faces


def _box_from_coords(xs, ys, zs, name="Box") -> Mesh:
    verts, faces = [], []

    def face(fixed_axis, fixed_val, ua, ucoords, va, vcoords, flip):
        off = len(verts)
        for v in vcoords:
            for u in ucoords:
                p = [0.0, 0.0, 0.0]
                p[fixed_axis] = fixed_val
                p[ua] = u
                p[va] = v
                verts.append(p)
        quads = _grid_quads(len(ucoords) - 1, len(vcoords) - 1, off)
        if flip:
            quads = [tuple(reversed(q)) for q in quads]
        faces.extend(quads)

    # +X: u=z(-), v=y. For outward normal +X we need (u x v) = +X: z x y = -X -> flip
    face(0, xs[-1], 2, zs, 1, ys, True)
    face(0, xs[0], 2, zs, 1, ys, False)
    face(1, ys[-1], 0, xs, 2, zs, True)   # +Y : x x z = -Y -> flip
    face(1, ys[0], 0, xs, 2, zs, False)
    face(2, zs[-1], 0, xs, 1, ys, False)  # +Z : x x y = +Z
    face(2, zs[0], 0, xs, 1, ys, True)
    m = Mesh(np.array(verts), faces, name=name)
    m.weld(1e-7)
    return m


def box(size=1.0, segments: int | Sequence[int] = 1, name: str = "Box") -> Mesh:
    """Axis-aligned box. size: scalar or (width_x, height_y, depth_z)."""
    s = _size3(size) / 2
    seg = np.broadcast_to(np.asarray(segments, dtype=int), (3,))
    xs = np.linspace(-s[0], s[0], seg[0] + 1)
    ys = np.linspace(-s[1], s[1], seg[1] + 1)
    zs = np.linspace(-s[2], s[2], seg[2] + 1)
    m = _box_from_coords(xs, ys, zs, name)
    m.smooth_angle = 30.0
    return m.uv_box()


cube = box


def rounded_box(size=1.0, radius: float = 0.1, segments: int = 4, name: str = "RoundedBox") -> Mesh:
    """Box with rounded edges/corners (radius in meters, `segments` per rounded edge)."""
    s = _size3(size) / 2
    r = float(min(radius, s.min() * 0.999))

    def coords(h):
        if r <= 1e-9:
            return np.array([-h, h])
        phi = np.radians(np.linspace(45, 0, segments + 1))
        left = -h + r * (1 - np.tan(phi))
        mid_needed = h - r > -h + r + 1e-9
        right = -left[::-1]
        return np.concatenate([left, right]) if mid_needed else np.concatenate([left[:-1], right])

    m = _box_from_coords(coords(s[0]), coords(s[1]), coords(s[2]), name)
    if r > 1e-9:
        inner = np.clip(m.vertices, -(s - r), s - r)
        d = m.vertices - inner
        m.vertices = inner + m3.normalize(d) * r
        bad = np.linalg.norm(d, axis=1) < 1e-12
        m.vertices[bad] = inner[bad]
    m.smooth_angle = 50.0
    return m.uv_box()


# ---------------------------------------------------------------- planes
def plane(size=1.0, segments: int | Sequence[int] = 1, name: str = "Plane") -> Mesh:
    """Flat grid in the XZ plane facing +Y. size: scalar or (width_x, depth_z)."""
    sz = np.broadcast_to(np.asarray(size, dtype=np.float64), (2,))
    seg = np.broadcast_to(np.asarray(segments, dtype=int), (2,))
    nu, nv = int(seg[0]), int(seg[1])
    u = np.linspace(0, 1, nu + 1)
    v = np.linspace(0, 1, nv + 1)
    U, V = np.meshgrid(u, v)
    verts = np.stack([(U - 0.5) * sz[0], np.zeros_like(U), (0.5 - V) * sz[1]], axis=-1).reshape(-1, 3)
    faces = _grid_quads(nu, nv)
    uvs = np.stack([U, V], axis=-1).reshape(-1, 2)
    return Mesh(verts, faces, uvs=uvs, face_uvs=faces, name=name, smooth_angle=0.0)


grid = plane


def disc(radius: float = 0.5, segments: int = 32, name: str = "Disc") -> Mesh:
    """Flat circle (n-gon) in the XZ plane facing +Y."""
    a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    verts = np.stack([np.cos(a) * radius, np.zeros_like(a), -np.sin(a) * radius], axis=1)
    uvs = np.stack([0.5 + 0.5 * np.cos(a), 0.5 + 0.5 * np.sin(a)], axis=1)
    f = tuple(range(segments))
    return Mesh(verts, [f], uvs=uvs, face_uvs=[f], name=name, smooth_angle=0.0)


# --------------------------------------------------------------- spheres
def uv_sphere(radius: float = 0.5, segments: int = 32, rings: int = 16, name: str = "Sphere") -> Mesh:
    m = lathe([(math.sin(t) * radius, -math.cos(t) * radius) for t in np.linspace(0, math.pi, rings + 1)],
              segments=segments, name=name)
    m.smooth_angle = 180.0
    return m


sphere = uv_sphere


def ico_sphere(radius: float = 0.5, subdivisions: int = 2, name: str = "IcoSphere") -> Mesh:
    t = (1 + 5 ** 0.5) / 2
    v = np.array([[-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
                  [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]], dtype=np.float64)
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
         (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
         (8, 6, 7), (9, 8, 1)]
    verts = list(m3.normalize(v))
    faces = f
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def mid(a, b):
            key = (min(a, b), max(a, b))
            if key not in cache:
                verts.append(m3.normalize((verts[a] + verts[b]) / 2))
                cache[key] = len(verts) - 1
            return cache[key]
        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    m = Mesh(np.array(verts) * radius, faces, name=name, smooth_angle=180.0)
    return m.uv_sphere()


# ------------------------------------------------------------- revolved
def lathe(profile: Sequence[Sequence[float]], segments: int = 32, angle: float = 360.0,
          name: str = "Lathe", smooth_angle: float = 40.0) -> Mesh:
    """Revolve a 2D profile [(radius, y), ...] around the Y axis (vases, bottles, trunks, wheels).

    Points with radius == 0 become poles (closed ends). Order the profile from
    bottom to top so normals face outward.
    """
    prof = np.asarray(profile, dtype=np.float64)
    full = abs(angle - 360.0) < 1e-9
    nseg = segments
    ncol = nseg if full else nseg + 1
    th = np.radians(np.linspace(0, angle, nseg + 1))[:ncol]
    verts, faces, uvs, fuvs = [], [], [], []
    ring_index = []  # list of vertex index arrays per profile point
    # arc-length v coordinate
    seglen = np.linalg.norm(np.diff(prof, axis=0), axis=1)
    vcoord = np.concatenate([[0], np.cumsum(seglen)])
    vcoord = vcoord / max(vcoord[-1], 1e-12)
    for i, (r, y) in enumerate(prof):
        if abs(r) < 1e-9:
            ring_index.append(np.array([len(verts)]))
            verts.append([0.0, y, 0.0])
        else:
            idx = np.arange(len(verts), len(verts) + ncol)
            ring_index.append(idx)
            for t in th:
                verts.append([r * math.sin(t), y, r * math.cos(t)])
    # uv grid (always full grid with seam column)
    for i in range(len(prof)):
        for j in range(nseg + 1):
            uvs.append([j / nseg, vcoord[i]])

    def uvi(i, j):
        return i * (nseg + 1) + j

    for i in range(len(prof) - 1):
        a, b = ring_index[i], ring_index[i + 1]
        for j in range(nseg):
            j1 = (j + 1) % ncol if full else j + 1
            if len(a) == 1 and len(b) == 1:
                continue
            if len(a) == 1:
                faces.append((a[0], b[j], b[j1]))
                fuvs.append((uvi(i, j), uvi(i + 1, j), uvi(i + 1, j + 1)))
            elif len(b) == 1:
                faces.append((a[j], b[0], a[j1]))
                fuvs.append((uvi(i, j), uvi(i + 1, j), uvi(i, j + 1)))
            else:
                faces.append((a[j], b[j], b[j1], a[j1]))
                fuvs.append((uvi(i, j), uvi(i + 1, j), uvi(i + 1, j + 1), uvi(i, j + 1)))
    m = Mesh(np.array(verts), faces, uvs=np.array(uvs), face_uvs=fuvs, name=name, smooth_angle=smooth_angle)
    return m.flip()  # faces were built inward-facing


def cylinder(radius: float = 0.5, height: float = 1.0, segments: int = 32, rings: int = 1,
             radius_top: float | None = None, caps: bool = True, name: str = "Cylinder") -> Mesh:
    """Cylinder along Y centered at the origin. radius_top makes a truncated cone."""
    rt = radius if radius_top is None else radius_top
    h = height / 2
    prof = [(radius + (rt - radius) * t, -h + height * t) for t in np.linspace(0, 1, rings + 1)]
    if caps:
        prof = [(0.0, -h)] + prof + ([(0.0, h)] if rt > 1e-9 else [])
    m = lathe(prof, segments, name=name, smooth_angle=50.0)
    return m


def cone(radius: float = 0.5, height: float = 1.0, segments: int = 32, name: str = "Cone") -> Mesh:
    return cylinder(radius, height, segments, radius_top=0.0, name=name)


def capsule(radius: float = 0.25, height: float = 1.0, segments: int = 24, rings: int = 8,
            name: str = "Capsule") -> Mesh:
    """Capsule along Y. height is the total height including the hemispheres."""
    half = max(height / 2 - radius, 0.0)
    prof = []
    for t in np.linspace(0, math.pi / 2, rings + 1):
        prof.append((math.sin(t) * radius, -half - math.cos(t) * radius))
    for t in np.linspace(0, math.pi / 2, rings + 1):
        prof.append((math.cos(t) * radius, half + math.sin(t) * radius))
    m = lathe(prof, segments, name=name)
    m.smooth_angle = 180.0
    return m


def torus(major_radius: float = 0.5, minor_radius: float = 0.15, major_segments: int = 48,
          minor_segments: int = 16, name: str = "Torus") -> Mesh:
    """Torus lying in the XZ plane (ring around the Y axis)."""
    nu, nv = major_segments, minor_segments
    verts, uvs = [], []
    for i in range(nu):
        u = 2 * math.pi * i / nu
        for j in range(nv):
            v = 2 * math.pi * j / nv
            r = major_radius + minor_radius * math.cos(v)
            verts.append([r * math.sin(u), minor_radius * math.sin(v), r * math.cos(u)])
    for i in range(nu + 1):
        for j in range(nv + 1):
            uvs.append([i / nu, j / nv])
    faces, fuvs = [], []
    for i in range(nu):
        for j in range(nv):
            a = i * nv + j
            b = ((i + 1) % nu) * nv + j
            c = ((i + 1) % nu) * nv + (j + 1) % nv
            d = i * nv + (j + 1) % nv
            faces.append((a, b, c, d))
            ua = i * (nv + 1) + j
            fuvs.append((ua, ua + nv + 1, ua + nv + 2, ua + 1))
    return Mesh(np.array(verts), faces, uvs=np.array(uvs), face_uvs=fuvs, name=name, smooth_angle=180.0)


# ------------------------------------------------------------ sweeps
def sweep(path: Sequence[Sequence[float]], shape: Sequence[Sequence[float]] | None = None,
          radius: float | Sequence[float] | Callable[[float], float] = 0.1, segments: int = 12,
          caps: bool = True, closed: bool = False, twist: float = 0.0, up=(0, 1, 0),
          name: str = "Sweep") -> Mesh:
    """Sweep a cross-section along a 3D path (tubes, branches, cables, horns, rails).

    path:   (N, 3) points (use m3.catmull_rom / m3.bezier for smooth curves)
    shape:  optional closed 2D cross-section [(x, y), ...] (default: circle)
    radius: scalar, per-point list, or function of t in [0, 1] scaling the section
    twist:  total twist in degrees along the path
    """
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    if shape is None:
        a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        shape2 = np.stack([np.cos(a), np.sin(a)], axis=1)
    else:
        shape2 = np.asarray(shape, dtype=np.float64)
    ns = len(shape2)
    seglen = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0], np.cumsum(seglen)])
    tt = arc / max(arc[-1], 1e-12)
    if callable(radius):
        rad = np.array([radius(float(t)) for t in tt])
    else:
        rad = np.broadcast_to(np.asarray(radius, dtype=np.float64), (n,)).copy()
        if np.asarray(radius).size not in (1, n):
            rad = np.interp(tt, np.linspace(0, 1, np.asarray(radius).size), np.asarray(radius, dtype=np.float64))
    tang, nor, bin_ = m3.frames_along(path, up)
    verts = []
    for i in range(n):
        tw = math.radians(twist) * tt[i]
        c, s = math.cos(tw), math.sin(tw)
        for x, y in shape2:
            xr, yr = x * c - y * s, x * s + y * c
            verts.append(path[i] + (nor[i] * xr + bin_[i] * yr) * rad[i])
    verts = np.array(verts)
    faces, fuvs, uvs = [], [], []
    for i in range(n):
        for j in range(ns + 1):
            uvs.append([j / ns, arc[i]])
    rows = n if not closed else n + 1
    for i in range(rows - 1):
        i0, i1 = i % n, (i + 1) % n
        for j in range(ns):
            j1 = (j + 1) % ns
            faces.append((i0 * ns + j, i0 * ns + j1, i1 * ns + j1, i1 * ns + j))
            u0 = min(i, n - 1) * (ns + 1)
            u1 = min(i + 1, n - 1) * (ns + 1)
            fuvs.append((u0 + j, u0 + j + 1, u1 + j + 1, u1 + j))
    uvs = np.array(uvs)
    m = Mesh(verts, faces, uvs=uvs, face_uvs=fuvs, name=name, smooth_angle=60.0)
    # the section winding depends on frame handedness; fix normals to point outward
    fnorm = m.face_normals()
    out_dir = m.face_centers() - np.repeat(path[:-1] if not closed else path, ns, axis=0)[:len(m.faces)]
    if np.sum(np.einsum("ij,ij->i", fnorm, out_dir)) < 0:
        m.flip()
    if caps and not closed:
        if rad[0] > 1e-9:
            _add_cap(m, list(range(ns)), path[0] - path[1])
        if rad[-1] > 1e-9:
            _add_cap(m, list(range((n - 1) * ns, n * ns)), path[-1] - path[-2])
        m.weld(1e-9)
    return m


def _add_cap(m: Mesh, ring: list[int], outward):
    """Close a ring of vertices with an n-gon whose normal points along `outward`."""
    f = tuple(ring)
    test = Mesh(m.vertices, [f])
    if np.dot(test.face_normals()[0], outward) < 0:
        f = tuple(reversed(f))
    base = len(m.uvs)
    pts = m.vertices[list(f)]
    c = pts.mean(axis=0)
    d = pts - c
    rmax = max(np.linalg.norm(d, axis=1).max(), 1e-12)
    ax = np.argmax(np.abs(test.face_normals()[0]))
    o = [k for k in range(3) if k != ax]
    m.uvs = np.concatenate([m.uvs, 0.5 + 0.5 * d[:, o] / rmax])
    m.face_uvs.append(tuple(range(base, base + len(f))))
    m.faces.append(f)
    m.face_materials = np.append(m.face_materials, 0)


def tube(path, radius=0.05, segments: int = 12, caps: bool = True, name: str = "Tube", **kw) -> Mesh:
    return sweep(path, None, radius, segments, caps, name=name, **kw)


def loft(sections: Sequence[Sequence[Sequence[float]]], closed_sections: bool = True, caps: bool = True,
         name: str = "Loft", smooth_angle: float = 50.0) -> Mesh:
    """Skin a surface through cross-sections (each (M,3), same M). Great for hulls, car bodies, blades."""
    secs = [np.asarray(s, dtype=np.float64) for s in sections]
    m_ = len(secs[0])
    if any(len(s) != m_ for s in secs):
        raise ValueError("all sections must have the same number of points")
    verts = np.concatenate(secs)
    faces, fuvs, uvs = [], [], []
    ncols = m_ if closed_sections else m_ - 1
    for i in range(len(secs)):
        for j in range(m_ + 1):
            uvs.append([j / m_, i / max(len(secs) - 1, 1)])
    for i in range(len(secs) - 1):
        for j in range(ncols):
            j1 = (j + 1) % m_
            faces.append((i * m_ + j, i * m_ + j1, (i + 1) * m_ + j1, (i + 1) * m_ + j))
            fuvs.append((i * (m_ + 1) + j, i * (m_ + 1) + j + 1, (i + 1) * (m_ + 1) + j + 1, (i + 1) * (m_ + 1) + j))
    m = Mesh(verts, faces, uvs=np.array(uvs), face_uvs=fuvs, name=name, smooth_angle=smooth_angle)
    if closed_sections:
        centers = np.repeat(np.array([s.mean(axis=0) for s in secs[:-1]]), m_, axis=0)
        if np.sum(np.einsum("ij,ij->i", m.face_normals(), m.face_centers() - centers)) < 0:
            m.flip()
        if caps:
            _add_cap(m, list(range(m_)), secs[0].mean(axis=0) - secs[1].mean(axis=0))
            _add_cap(m, list(range((len(secs) - 1) * m_, len(secs) * m_)),
                     secs[-1].mean(axis=0) - secs[-2].mean(axis=0))
    return m


def extrude_shape(shape: Sequence[Sequence[float]], depth: float = 0.1, caps: bool = True,
                  name: str = "Extrusion") -> Mesh:
    """Extrude a closed 2D polygon [(x, y), ...] (in the XY plane) along Z, centered on z=0.

    Concave shapes are fine (caps are triangulated by ear clipping on export).
    Use shapes.* helpers (star, rounded_rect, circle...) or your own outline.
    """
    pts = np.asarray(shape, dtype=np.float64)
    area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area < 0:
        pts = pts[::-1]
    n = len(pts)
    h = depth / 2
    front = np.column_stack([pts, np.full(n, h)])
    back = np.column_stack([pts, np.full(n, -h)])
    verts = np.concatenate([back, front])
    faces = [(j, (j + 1) % n, n + (j + 1) % n, n + j) for j in range(n)]
    m = Mesh(verts, faces, name=name, smooth_angle=30.0)
    if caps:
        m.faces.append(tuple(range(n, 2 * n)))            # front (+Z), CCW
        m.faces.append(tuple(reversed(range(n))))         # back (-Z)
        m.face_materials = np.zeros(len(m.faces), dtype=np.int64)
    return m.uv_box()


# ------------------------------------------------------------- 2D shapes
class shapes:
    """2D outlines for extrude_shape / sweep (lists of (x, y))."""

    @staticmethod
    def circle(radius: float = 0.5, segments: int = 32):
        a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        return np.stack([np.cos(a) * radius, np.sin(a) * radius], axis=1)

    @staticmethod
    def ellipse(rx: float = 0.5, ry: float = 0.3, segments: int = 32):
        a = np.linspace(0, 2 * math.pi, segments, endpoint=False)
        return np.stack([np.cos(a) * rx, np.sin(a) * ry], axis=1)

    @staticmethod
    def rect(w: float = 1.0, h: float = 1.0):
        return np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])

    @staticmethod
    def rounded_rect(w: float = 1.0, h: float = 1.0, r: float = 0.1, segments: int = 6):
        r = min(r, w / 2, h / 2)
        pts = []
        for cx, cy, a0 in [(w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                           (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)]:
            for a in np.radians(np.linspace(a0, a0 + 90, segments + 1)):
                pts.append([cx + r * math.cos(a), cy + r * math.sin(a)])
        return np.array(pts)

    @staticmethod
    def star(points: int = 5, outer: float = 0.5, inner: float = 0.2):
        a = np.linspace(0, 2 * math.pi, points * 2, endpoint=False) + math.pi / 2
        r = np.where(np.arange(points * 2) % 2 == 0, outer, inner)
        return np.stack([np.cos(a) * r, np.sin(a) * r], axis=1)

    @staticmethod
    def polygon(sides: int = 6, radius: float = 0.5):
        a = np.linspace(0, 2 * math.pi, sides, endpoint=False) + math.pi / 2
        return np.stack([np.cos(a) * radius, np.sin(a) * radius], axis=1)

    @staticmethod
    def gear(teeth: int = 12, outer: float = 0.5, inner: float = 0.42, tooth_width: float = 0.5):
        pts = []
        step = 2 * math.pi / teeth
        for i in range(teeth):
            a = i * step
            w = step * tooth_width / 2
            pts += [[inner * math.cos(a - step / 2 + 0.01), inner * math.sin(a - step / 2 + 0.01)],
                    [inner * math.cos(a - w), inner * math.sin(a - w)],
                    [outer * math.cos(a - w * 0.7), outer * math.sin(a - w * 0.7)],
                    [outer * math.cos(a + w * 0.7), outer * math.sin(a + w * 0.7)],
                    [inner * math.cos(a + w), inner * math.sin(a + w)]]
        return np.array(pts)

    @staticmethod
    def from_curve(points, samples_per_segment: int = 8):
        """Smooth closed outline through control points."""
        return m3.catmull_rom(points, samples_per_segment, closed=True)
