"""Procedural plants, rocks and other natural props.

Every generator is deterministic for a given `seed` and returns a Mesh
(materials already assigned) standing on y = 0 and centered on x = z = 0.
"""
from __future__ import annotations

import math

import numpy as np

from .. import math3d as m3
from .. import sdf
from ..material import Material, Texture, presets
from ..mesh import Mesh
from ..primitives import cone, cylinder, ico_sphere, lathe, plane, sweep, uv_sphere


def _branch_path(start, direction, length, rng, bend=0.25, steps=6):
    d = m3.normalize(m3.vec3(direction))
    pts = [m3.vec3(start)]
    seg = length / steps
    for _ in range(steps):
        jitter = rng.normal(size=3) * bend * 0.35
        d = m3.normalize(d + jitter + np.array([0, 0.08, 0]))
        pts.append(pts[-1] + d * seg)
    return np.array(pts)


def tree(height: float = 5.0, trunk_radius: float | None = None, levels: int = 3, branches: int = 5,
         spread: float = 45.0, leaves: str = "blobs", leaf_color="#4f8a35", bark=None, foliage=None,
         leaf_size: float = 1.0, seed: int = 0, segments: int = 8, name: str = "Tree") -> Mesh:
    """Deciduous tree with recursive branches.

    leaves: "blobs" (stylized low-poly canopy clusters), "cards" (alpha leaf
            quads, more realistic), or "none" (bare winter tree)
    spread: branch angle from the parent in degrees
    """
    rng = np.random.default_rng(seed)
    tr = trunk_radius or height * 0.045
    bark = bark or presets.bark(seed=seed)
    foliage = foliage or presets.foliage(leaf_color)
    wood_parts: list[Mesh] = []
    tips: list[tuple[np.ndarray, np.ndarray, float]] = []

    def grow(start, direction, length, radius, level):
        path = _branch_path(start, direction, length, rng, bend=0.15 + 0.1 * level)
        r_end = radius * 0.45 if level < levels else radius * 0.2
        m = sweep(path, radius=lambda t: radius + (r_end - radius) * t, segments=max(segments - level * 2, 4),
                  caps=level == 0)
        wood_parts.append(m)
        if level >= levels:
            tips.append((path[-1], m3.normalize(path[-1] - path[-2]), length))
            return
        n_child = branches if level > 0 else branches + 1
        for i in range(n_child):
            t = rng.uniform(0.45, 0.95) if level == 0 else rng.uniform(0.3, 0.95)
            idx = min(int(t * (len(path) - 1)), len(path) - 2)
            base = path[idx] + (path[idx + 1] - path[idx]) * (t * (len(path) - 1) - idx)
            parent_dir = m3.normalize(path[idx + 1] - path[idx])
            az = 360.0 * i / n_child + rng.uniform(-25, 25)
            perp = m3.normalize(np.cross(parent_dir, [0.31, 0.0, 0.95]) if abs(parent_dir[1]) > 0.5
                                else np.cross(parent_dir, [0, 1, 0]))
            tilt = m3.quat_from_axis_angle(perp, spread + rng.uniform(-10, 10))
            roll = m3.quat_from_axis_angle(parent_dir, az)
            d = m3.quat_rotate(m3.quat_mul(roll, tilt), parent_dir)
            d = m3.normalize(d + np.array([0, 0.35, 0]))
            r_here = radius + (r_end - radius) * t
            grow(base, d, length * rng.uniform(0.5, 0.7), r_here * 0.65, level + 1)

    grow(np.zeros(3), (0, 1, 0), height * 0.6, tr, 0)
    wood = Mesh.join(wood_parts, name=name)
    wood.set_material(bark)
    wood.smooth_angle = 70.0
    if leaves != "none" and tips:
        leaf = _leaves(tips, leaves, foliage, height, leaf_size, rng)
        wood.merge(leaf)
    return wood.ground()


def _leaves(tips, kind, foliage, height, leaf_size, rng) -> Mesh:
    parts = []
    if kind == "blobs":
        for p, d, L in tips:
            r = height * 0.09 * leaf_size * rng.uniform(0.8, 1.25)
            b = ico_sphere(r, 1)
            b.displace(r * 0.25, scale=3.0 / r, seed=int(rng.integers(1e6)))
            b.scale((1.0, 0.8, 1.0)).translate(p + d * r * 0.4)
            b.smooth_angle = 0.0
            parts.append(b)
    else:
        tex = leaf_texture()
        mat = Material(foliage.name, foliage.base_color, 0.0, 0.7, base_color_texture=tex, alpha_mode="MASK",
                       alpha_cutoff=0.4, double_sided=True)
        for p, d, L in tips:
            for _ in range(10):
                s = height * 0.06 * leaf_size * rng.uniform(0.7, 1.3)
                q = plane((s, s), 1)
                q.rotate((rng.uniform(-60, 60), rng.uniform(0, 360), rng.uniform(-60, 60)))
                q.translate(p + rng.normal(size=3) * height * 0.05)
                parts.append(q)
        out = Mesh.join(parts, name="Leaves")
        out.set_material(mat)
        return out
    out = Mesh.join(parts, name="Leaves")
    out.set_material(foliage)
    return out


def leaf_texture(size: int = 256, color="#ffffff") -> Texture:
    """Leaf silhouette with a central vein, alpha-masked (use with alpha_mode='MASK')."""
    def f(u, v):
        x = (u - 0.5) * 2
        y = v * 2 - 1
        width = np.sqrt(np.clip(1 - y * y, 0, 1)) * 0.55 * (1 - 0.3 * y)
        inside = np.abs(x) < width
        vein = np.abs(x) < 0.03
        g = 0.75 + 0.25 * (1 - np.abs(x) / np.maximum(width, 1e-3))
        rgb = np.stack([g * 0.9, g, g * 0.85], axis=-1)
        rgb[vein] *= 1.15
        a = inside.astype(float)
        return np.concatenate([np.clip(rgb, 0, 1), a[..., None]], axis=-1)
    return Texture.from_function(f, size, "Leaf")


def pine(height: float = 6.0, layers: int = 5, radius: float | None = None, color="#2f5d34", trunk_color="#5a3d28",
         segments: int = 10, seed: int = 0, snow: bool = False, name: str = "Pine") -> Mesh:
    """Stylized conifer: stacked cones on a trunk."""
    rng = np.random.default_rng(seed)
    r = radius or height * 0.28
    trunk = cylinder(height * 0.05, height * 0.35, 8, radius_top=height * 0.035).translate((0, height * 0.175, 0))
    trunk.set_material(Material("Trunk", trunk_color, 0.0, 0.9))
    mat = Material("Needles", color, 0.0, 0.8)
    for i in range(layers):
        t = i / max(layers - 1, 1)
        rr = r * (1 - 0.75 * t) * rng.uniform(0.92, 1.08)
        hh = height * 0.38 * (1 - 0.3 * t)
        y = height * (0.2 + 0.62 * t)
        c = cone(rr, hh, segments).translate((0, y + hh / 2, 0))
        c.rotate((0, rng.uniform(0, 360), 0), pivot=(0, 0, 0))
        c.set_material(mat)
        if snow:
            c.set_color("#ffffff", verts=np.flatnonzero(c.vertices[:, 1] > y + hh * 0.75))
        trunk.merge(c)
    trunk.smooth_angle = 0.0
    trunk.name = name
    return trunk.ground()


def palm(height: float = 6.0, fronds: int = 8, lean: float = 15.0, seed: int = 0, name: str = "Palm") -> Mesh:
    rng = np.random.default_rng(seed)
    top = np.array([math.sin(math.radians(lean)) * height * 0.4, height, 0.0])
    path = m3.bezier([(0, 0, 0), (0, height * 0.5, 0), top], 16)
    trunk = sweep(path, radius=lambda t: height * (0.035 - 0.012 * t), segments=10)
    trunk.set_material(Material("PalmTrunk", "#8a6a48", 0.0, 0.9))
    frond_mat = Material("Frond", "#3f8a3a", 0.0, 0.7, double_sided=True)
    for i in range(fronds):
        az = 360 * i / fronds + rng.uniform(-10, 10)
        L = height * rng.uniform(0.35, 0.45)
        d = m3.quat_rotate(m3.quat_from_euler((0, az, 0)), (1, 0, 0))
        pts = [top + d * L * t + np.array([0, L * (0.25 * math.sin(t * math.pi * 0.9) - 0.35 * t * t), 0])
               for t in np.linspace(0, 1, 8)]
        leaf = sweep(np.array(pts), shape=[(-1, 0), (0, 0.02), (1, 0), (0, -0.02)],
                     radius=lambda t: height * 0.06 * math.sin(math.pi * min(t * 1.1, 1.0)) + 1e-3, caps=False,
                     up=(0, 1, 0))
        leaf.set_material(frond_mat)
        trunk.merge(leaf)
    trunk.name = name
    return trunk


def bush(size: float = 1.0, color="#4a7d34", seed: int = 0, berries: str | None = None, name: str = "Bush") -> Mesh:
    rng = np.random.default_rng(seed)
    parts = []
    for i in range(7):
        r = size * rng.uniform(0.25, 0.4)
        c = rng.normal(size=3) * size * 0.18
        c[1] = abs(c[1]) * 0.6 + r * 0.7
        b = ico_sphere(r, 1).displace(r * 0.2, scale=3 / r, seed=seed + i).translate(c)
        parts.append(b)
    m = Mesh.join(parts, name=name)
    m.set_material(presets.foliage(color))
    m.smooth_angle = 0.0
    if berries:
        bm = Material("Berries", berries, 0.0, 0.3)
        for _ in range(12):
            d = m3.normalize(rng.normal(size=3) * (1, 0.6, 1) + (0, 0.5, 0))
            p = d * size * 0.45 + np.array([0, size * 0.3, 0])
            m.merge(uv_sphere(size * 0.035, 8, 6).translate(p).set_material(bm))
    return m.ground()


def rock(size: float = 1.0, seed: int = 0, roughness: float = 0.25, flatness: float = 0.7, low_poly: bool = True,
         color="#8a857e", material=None, name: str = "Rock") -> Mesh:
    """Boulder. low_poly=True gives a faceted game-art rock; False a smooth, detailed one."""
    rng = np.random.default_rng(seed)
    if low_poly:
        m = ico_sphere(size / 2, 2)
        m.vertices *= rng.uniform(0.75, 1.25, size=3) * np.array([1, flatness, 1])
        m.displace(size * roughness * 0.5, scale=1.6 / size, octaves=3, seed=seed)
        m.smooth_angle = 0.0
    else:
        s = sdf.ellipsoid((0, 0, 0), np.array([0.5, 0.5 * flatness, 0.5]) * size * rng.uniform(0.8, 1.2, 3))
        s = s.displace(size * roughness * 0.25, scale=2.5 / size, octaves=5, seed=seed)
        m = s.to_mesh(64)
        m.smooth_angle = 180.0
    m.name = name
    m.set_material(material or presets.stone(color))
    m.uv_box(1.0 / size)
    lo, _ = m.bounds()
    return m.translate((0, -lo[1] - size * 0.08 * flatness, 0))


def grass_patch(size: float = 1.0, blades: int = 120, height: float = 0.25, color="#5f9a3a", seed: int = 0,
                name: str = "Grass") -> Mesh:
    """Clump of grass blades (thin triangles), good for scattering."""
    rng = np.random.default_rng(seed)
    verts, faces = [], []
    for i in range(blades):
        p = rng.uniform(-size / 2, size / 2, 2)
        h = height * rng.uniform(0.6, 1.3)
        w = height * 0.06
        a = rng.uniform(0, math.pi)
        lean = rng.normal(size=2) * h * 0.25
        dx, dz = math.cos(a) * w, math.sin(a) * w
        base = len(verts)
        verts += [[p[0] - dx, 0, p[1] - dz], [p[0] + dx, 0, p[1] + dz], [p[0] + lean[0], h, p[1] + lean[1]]]
        faces.append((base, base + 1, base + 2))
    m = Mesh(np.array(verts), faces, name=name, smooth_angle=0.0)
    m.set_color("#ffffff", func=lambda P: np.column_stack([0.4 + 0.6 * P[:, 1] / height] * 3))
    m.set_material(Material("Grass", color, 0.0, 0.8, double_sided=True))
    return m


def flower(height: float = 0.4, petals: int = 6, petal_color="#e84a7a", center_color="#f2c230", seed: int = 0,
           name: str = "Flower") -> Mesh:
    stem = sweep(m3.bezier([(0, 0, 0), (0.02, height * 0.5, 0), (0, height, 0)], 8), radius=height * 0.012,
                 segments=6)
    stem.set_material(Material("Stem", "#4a8a35", 0.0, 0.7))
    center = uv_sphere(height * 0.05, 10, 6).scale((1, 0.6, 1)).translate((0, height, 0))
    center.set_material(Material("FlowerCenter", center_color, 0.0, 0.6))
    pm = Material("Petal", petal_color, 0.0, 0.5, double_sided=True)
    stem.merge(center)
    for i in range(petals):
        p = uv_sphere(height * 0.07, 10, 6).scale((1, 0.18, 0.5)).translate((height * 0.08, 0, 0))
        p.rotate((0, 0, 12)).rotate((0, 360 * i / petals, 0)).translate((0, height, 0))
        p.set_material(pm)
        stem.merge(p)
    stem.name = name
    return stem


def mushroom(height: float = 0.25, cap_color="#c8322a", spots: bool = True, seed: int = 0,
             name: str = "Mushroom") -> Mesh:
    rng = np.random.default_rng(seed)
    stem = lathe([(0, 0), (height * 0.13, 0.0), (height * 0.11, height * 0.5), (height * 0.09, height * 0.8),
                  (0, height * 0.8)], 16)
    stem.set_material(Material("Stem", "#efe6d2", 0.0, 0.7))
    cap_prof = [(0, height * 0.72), (height * 0.42, height * 0.7)] + \
               [(height * 0.42 * math.cos(t), height * 0.7 + height * 0.35 * math.sin(t))
                for t in np.linspace(0.15, math.pi / 2, 8)]
    cap_prof[-1] = (0.0, cap_prof[-1][1])
    cap = lathe(cap_prof, 24)
    cap.set_material(Material("Cap", cap_color, 0.0, 0.4))
    stem.merge(cap)
    if spots:
        sm = Material("Spots", "#ffffff", 0.0, 0.6)
        for _ in range(9):
            a = rng.uniform(0, 2 * math.pi)
            t = rng.uniform(0.25, 1.2)
            p = np.array([height * 0.42 * math.cos(t) * math.cos(a), height * 0.7 + height * 0.35 * math.sin(t),
                          height * 0.42 * math.cos(t) * math.sin(a)])
            stem.merge(uv_sphere(height * 0.045, 8, 4).scale((1, 0.4, 1)).translate(p).set_material(sm))
    stem.name = name
    return stem


def cactus(height: float = 1.6, arms: int = 2, color="#4f8f4a", seed: int = 0, name: str = "Cactus") -> Mesh:
    rng = np.random.default_rng(seed)
    r = height * 0.09
    body = sdf.capsule((0, 0, 0), (0, height - r, 0), r)
    for i in range(arms):
        side = 1 if i % 2 == 0 else -1
        y0 = height * rng.uniform(0.3, 0.55)
        x1 = side * height * 0.22
        arm = sdf.chain([(0, y0, 0), (x1, y0, 0), (x1, y0 + height * rng.uniform(0.2, 0.35), 0)],
                        [r * 0.7, r * 0.65, r * 0.6], k=r * 0.4)
        body = body.smooth_union(arm, r * 0.3)
    m = body.to_mesh(80, name=name)
    ang = np.arctan2(m.vertices[:, 2], m.vertices[:, 0])
    m.vertices -= m.vertex_normals() * (0.006 * height * (0.5 + 0.5 * np.cos(ang * 12)))[:, None]
    m.set_material(Material("Cactus", color, 0.0, 0.6))
    return m.ground()
