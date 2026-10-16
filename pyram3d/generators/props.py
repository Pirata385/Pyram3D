"""Everyday props and furniture. Each returns a Mesh standing on y = 0, front = +Z."""
from __future__ import annotations

import math

import numpy as np

from ..material import Material, presets
from ..mesh import Mesh
from ..primitives import box, cylinder, lathe, rounded_box, torus, uv_sphere, extrude_shape, shapes


def crate(size: float = 1.0, color="#a8743f", name: str = "Crate") -> Mesh:
    """Wooden crate with framed edges."""
    m = box(size * 0.94, name=name).translate((0, size / 2, 0))
    m.set_material(presets.wood(name="CratePlanks"))
    frame = Material("CrateFrame", color, 0.0, 0.7)
    t = size * 0.08
    parts = []
    for ax in range(3):
        for a in (-1, 1):
            for b in (-1, 1):
                dims = [t, t, t]
                dims[ax] = size
                pos = [0.0, 0.0, 0.0]
                o = [k for k in range(3) if k != ax]
                pos[o[0]] = a * (size - t) / 2
                pos[o[1]] = b * (size - t) / 2
                pos[1] += size / 2
                parts.append(box(dims).translate(pos))
    f = Mesh.join(parts, name="Frame")
    f.set_material(frame)
    return m.merge(f)


def barrel(height: float = 1.0, radius: float = 0.32, name: str = "Barrel") -> Mesh:
    prof = [(0, 0)] + [(radius * (0.85 + 0.15 * math.sin(math.pi * t)), height * t)
                       for t in np.linspace(0, 1, 12)] + [(0, height)]
    m = lathe(prof, 24, name=name)
    m.set_material(presets.wood(name="Staves", seed=3))
    m.uv_cylinder()
    m.uv_transform(scale=(6, 1))
    metal = presets.metal("#3d3d40", 0.5)
    for y in (0.12, 0.88):
        r = radius * (0.85 + 0.15 * math.sin(math.pi * y)) + 0.01
        ring = torus(r, 0.015, 32, 6).scale((1, 2.5, 1)).translate((0, height * y, 0))
        ring.set_material(metal)
        m.merge(ring)
    return m


def table(width: float = 1.4, depth: float = 0.8, height: float = 0.75, material=None, name: str = "Table") -> Mesh:
    mat = material or presets.wood(name="TableWood")
    top = rounded_box((width, 0.05, depth), 0.015).translate((0, height - 0.025, 0))
    parts = [top]
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(box((0.06, height - 0.05, 0.06)).translate((sx * (width / 2 - 0.08), (height - 0.05) / 2,
                                                                    sz * (depth / 2 - 0.08))))
    m = Mesh.join(parts, name=name)
    m.set_material(mat)
    return m


def chair(seat_height: float = 0.45, width: float = 0.45, material=None, cushion=None, name: str = "Chair") -> Mesh:
    mat = material or presets.wood(name="ChairWood")
    w = width
    parts = [rounded_box((w, 0.04, w), 0.01).translate((0, seat_height, 0))]
    for sx in (-1, 1):
        for sz in (-1, 1):
            parts.append(box((0.04, seat_height, 0.04)).translate((sx * (w / 2 - 0.03), seat_height / 2,
                                                                   sz * (w / 2 - 0.03))))
        parts.append(box((0.04, 0.5, 0.04)).translate((sx * (w / 2 - 0.03), seat_height + 0.27, -w / 2 + 0.03)))
    parts.append(rounded_box((w, 0.18, 0.03), 0.01).translate((0, seat_height + 0.42, -w / 2 + 0.03)))
    m = Mesh.join(parts, name=name)
    m.set_material(mat)
    if cushion:
        c = rounded_box((w * 0.92, 0.05, w * 0.92), 0.02).translate((0, seat_height + 0.045, 0.0))
        c.set_material(presets.fabric(cushion))
        m.merge(c)
    return m


def sofa(width: float = 2.0, color="#5b6d8a", name: str = "Sofa") -> Mesh:
    fab = presets.fabric(color)
    parts = [rounded_box((width, 0.35, 0.9), 0.06).translate((0, 0.3, 0)),
             rounded_box((width, 0.5, 0.2), 0.08).translate((0, 0.65, -0.35))]
    for sx in (-1, 1):
        parts.append(rounded_box((0.2, 0.55, 0.9), 0.08).translate((sx * (width / 2 - 0.1), 0.4, 0)))
    for i in range(2):
        parts.append(rounded_box(((width - 0.45) / 2, 0.15, 0.65), 0.06).translate(
            ((i - 0.5) * (width - 0.4) / 2, 0.53, 0.08)))
    m = Mesh.join(parts, name=name)
    m.set_material(fab)
    for sx in (-1, 1):
        for sz in (-1, 1):
            leg = cylinder(0.025, 0.12, 8).translate((sx * (width / 2 - 0.1), 0.06, sz * 0.35))
            leg.set_material(presets.metal("#222222", 0.4))
            m.merge(leg)
    return m


def lamp(height: float = 1.5, shade_color="#f2e6c9", name: str = "FloorLamp") -> Mesh:
    base = cylinder(0.18, 0.04, 24).translate((0, 0.02, 0))
    pole = cylinder(0.015, height, 8).translate((0, height / 2, 0))
    metal = presets.metal("#2b2b2b", 0.35)
    base.set_material(metal)
    pole.set_material(metal)
    shade = cylinder(0.22, 0.3, 24, radius_top=0.15, caps=False).translate((0, height, 0))
    shade.set_material(Material("Shade", shade_color, 0.0, 0.8, double_sided=True,
                                emissive=shade_color, emissive_strength=0.6))
    bulb = uv_sphere(0.05, 12, 8).translate((0, height - 0.02, 0))
    bulb.set_material(presets.emissive("#fff3d0", 5.0))
    return Mesh.join([base, pole, shade, bulb], name=name)


def bookshelf(width: float = 1.0, height: float = 1.8, shelves: int = 5, seed: int = 0,
              name: str = "Bookshelf") -> Mesh:
    rng = np.random.default_rng(seed)
    wood = presets.wood(name="ShelfWood", seed=5)
    d = 0.3
    parts = [box((0.03, height, d)).translate((sx * (width / 2 - 0.015), height / 2, 0)) for sx in (-1, 1)]
    parts.append(box((width, height, 0.02)).translate((0, height / 2, -d / 2 + 0.01)))
    for i in range(shelves + 1):
        parts.append(box((width, 0.025, d)).translate((0, i * (height - 0.025) / shelves + 0.0125, 0)))
    m = Mesh.join(parts, name=name)
    m.set_material(wood)
    palette = ["#8c2f39", "#2f5d8c", "#d9a441", "#3c6e47", "#5e4b8b", "#c4642d", "#e6dcc8"]
    for i in range(shelves):
        y0 = i * (height - 0.025) / shelves + 0.025
        x = -width / 2 + 0.04
        while x < width / 2 - 0.08:
            bw = rng.uniform(0.025, 0.05)
            bh = (height / shelves - 0.05) * rng.uniform(0.65, 0.95)
            b = box((bw, bh, d * 0.8)).translate((x + bw / 2, y0 + bh / 2, 0.01))
            b.set_material(Material("Book", palette[rng.integers(len(palette))], 0.0, 0.7))
            m.merge(b)
            x += bw + 0.003
    return m


def sword(length: float = 1.0, blade_color="#d8dde3", grip_color="#4a2c1a", name: str = "Sword") -> Mesh:
    """Sword pointing up (+Y), guard at y = grip length."""
    grip_l = length * 0.22
    blade = extrude_shape([(-0.025, 0), (0.025, 0), (0.025, length * 0.68), (0, length * 0.76),
                           (-0.025, length * 0.68)], depth=0.008)
    blade.vertices[:, 0] *= length
    blade = blade.translate((0, grip_l + 0.03, 0))
    blade.set_material(presets.metal(blade_color, 0.15))
    guard = rounded_box((length * 0.22, 0.03, 0.04), 0.01).translate((0, grip_l + 0.015, 0))
    guard.set_material(presets.gold())
    grip = cylinder(0.018, grip_l, 12).translate((0, grip_l / 2, 0))
    grip.set_material(Material("Grip", grip_color, 0.0, 0.8))
    pommel = uv_sphere(0.03, 12, 8)
    pommel.set_material(presets.gold())
    return Mesh.join([blade, guard, grip, pommel], name=name).ground()


def treasure_chest(width: float = 0.8, name: str = "Chest") -> Mesh:
    d = width * 0.6
    h = width * 0.45
    body = box((width, h, d)).translate((0, h / 2, 0))
    body.set_material(presets.wood(name="ChestWood", seed=7))
    lid = extrude_shape(shapes.ellipse(d / 2, d * 0.35, 24)[:13], depth=width)  # half ellipse
    lid.rotate((0, 90, 0)).translate((0, h, 0))
    lid.set_material(presets.wood(name="ChestWood", seed=7))
    bands = []
    for x in (-width * 0.3, width * 0.3):
        bands.append(box((0.05, h + 0.01, d + 0.02)).translate((x, h / 2, 0)))
    b = Mesh.join(bands, name="Bands")
    b.set_material(presets.gold())
    lock = box((0.08, 0.1, 0.03)).translate((0, h * 0.9, d / 2 + 0.015))
    lock.set_material(presets.gold())
    return Mesh.join([body, lid, b, lock], name=name)
