"""Procedural buildings, houses and architectural props.

Generators return Meshes standing on y = 0, centered on x = z = 0, facing +Z
(the front door is on the +Z side).
"""
from __future__ import annotations

import numpy as np

from .. import math3d as m3
from ..material import Material, presets, textures
from ..mesh import Mesh
from ..primitives import box, cylinder, extrude_shape, lathe, plane, rounded_box


def _glass():
    return Material("WindowGlass", "#8fb4cc", 0.2, 0.08)


def building(width: float = 12.0, depth: float = 10.0, floors: int = 5, floor_height: float = 3.2,
             bays: int | None = None, wall="#c9c2b6", frame="#4a4a4a", glass=None, roof: str = "flat",
             window_ratio: float = 0.55, door: bool = True, name: str = "Building") -> Mesh:
    """Multi-storey building with an inset window in every bay of every floor.

    bays: windows per floor along the width (default: one per ~3 m)
    roof: "flat" (parapet + rooftop box) | "gable" | "hip"
    """
    bays = bays or max(int(round(width / 3.0)), 1)
    bays_d = max(int(round(depth / 3.0)), 1)
    height = floors * floor_height
    m = box((width, height, depth), segments=(bays, floors, bays_d), name=name).translate((0, height / 2, 0))
    wall_m = Material("Wall", wall, 0.0, 0.85)
    frame_m = Material("Frame", frame, 0.3, 0.5)
    glass_m = glass or _glass()
    m.set_material(wall_m)
    sides = m.faces_where(lambda c, n: np.abs(n[:, 1]) < 0.5)
    # the ground-floor center bay on the front becomes a door
    door_face = None
    if door:
        c = m.face_centers()
        front = [i for i in sides if m.face_normals()[i][2] > 0.9 and c[i][1] < floor_height]
        if front:
            door_face = min(front, key=lambda i: abs(c[i][0]))
    win = [i for i in sides if i != door_face]
    inset = (1 - window_ratio) * min(width / bays, floor_height) / 2
    m.inset(win, thickness=inset)
    m.set_material(frame_m, win)
    m.inset(win, thickness=0.06)
    m.extrude(win, distance=-0.12)
    m.set_material(glass_m, win)
    if door_face is not None:
        m.inset([door_face], thickness=0.35)
        # make the door start at the ground: move the bottom inner edge down
        f = m.faces[door_face]
        lo_y = min(m.vertices[v][1] for v in f)
        for v in f:
            if abs(m.vertices[v][1] - lo_y) < 1e-6:
                m.vertices[v][1] = 0.0
        m.extrude([door_face], distance=-0.2)
        m.set_material(Material("Door", "#5a3b26", 0.0, 0.6), [door_face])
    if roof == "flat":
        para = box((width + 0.3, 0.6, depth + 0.3)).translate((0, height + 0.3, 0))
        top = box((width - 0.3, 0.62, depth - 0.3)).translate((0, height + 0.32, 0))
        para.set_material(wall_m)
        hvac = box((width * 0.2, 1.2, depth * 0.25)).translate((width * 0.2, height + 1.2, -depth * 0.15))
        hvac.set_material(presets.metal("#9aa0a6", 0.5))
        m.merge(para, hvac)
        top.set_material(Material("Roof", "#555555", 0.0, 0.9))
        m.merge(top)
    else:
        m.merge(_roof(width + 0.6, depth + 0.6, height, max(width, depth) * 0.25, roof))
    m.smooth_angle = 0.0
    return m


def _roof(w, d, y0, h, kind, color="#7a3b2e") -> Mesh:
    mat = Material("RoofTiles", color, 0.0, 0.8)
    if kind == "gable":
        r = extrude_shape([(-w / 2, 0), (w / 2, 0), (0, h)], depth=d)
        r.translate((0, y0, 0))
    else:  # hip
        r = Mesh([(-w / 2, 0, -d / 2), (w / 2, 0, -d / 2), (w / 2, 0, d / 2), (-w / 2, 0, d / 2),
                  (-max(w - d, 0) / 2, h, 0), (max(w - d, 0) / 2, h, 0)],
                 [(0, 1, 2, 3)[::-1], (3, 2, 5, 4), (1, 0, 4, 5), (0, 3, 4), (2, 1, 5)], name="Roof")
        r.weld(1e-6)
        r.translate((0, y0, 0))
        r.uv_box()
    r.set_material(mat)
    return r


def house(width: float = 8.0, depth: float = 7.0, wall="#e8dcc6", roof_color="#8a3a2a", door_color="#5a3b26",
          chimney: bool = True, porch: bool = True, name: str = "House") -> Mesh:
    """Single-storey house with gable roof, door, windows, chimney and porch step."""
    h = 3.0
    m = box((width, h, depth), segments=(3, 1, 2), name=name).translate((0, h / 2, 0))
    m.set_material(Material("Wall", wall, 0.0, 0.9))
    sides = m.faces_where(lambda c, n: np.abs(n[:, 1]) < 0.5)
    c = m.face_centers()
    nrm = m.face_normals()
    front = [i for i in sides if nrm[i][2] > 0.9]
    door_face = min(front, key=lambda i: abs(c[i][0]))
    win = [i for i in sides if i != door_face]
    m.inset(win, thickness=0.75)
    m.set_material(Material("Frame", "#ffffff", 0.0, 0.5), win)
    m.inset(win, thickness=0.07)
    m.extrude(win, distance=-0.1)
    m.set_material(_glass(), win)
    m.inset([door_face], thickness=0.6)
    f = m.faces[door_face]
    lo_y = min(m.vertices[v][1] for v in f)
    for v in f:
        if abs(m.vertices[v][1] - lo_y) < 1e-6:
            m.vertices[v][1] = 0.0
    m.extrude([door_face], distance=-0.12)
    m.set_material(Material("Door", door_color, 0.0, 0.6), [door_face])
    rh = width * 0.3
    m.merge(_roof(width + 0.8, depth + 0.8, h, rh, "gable", roof_color))
    if chimney:
        ch = box((0.7, 2.2, 0.7)).translate((width * 0.25, h + rh * 0.6, -depth * 0.15))
        ch.set_material(presets.bricks())
        m.merge(ch)
    if porch:
        step = box((2.2, 0.2, 1.0)).translate((c[door_face][0], 0.1, depth / 2 + 0.5))
        step.set_material(Material("Concrete", "#a8a39b", 0.0, 0.95))
        m.merge(step)
    m.smooth_angle = 0.0
    return m


def wall_segment(length: float = 4.0, height: float = 2.0, thickness: float = 0.3, material=None,
                 name: str = "Wall") -> Mesh:
    m = box((length, height, thickness), name=name).translate((0, height / 2, 0))
    m.set_material(material or presets.bricks())
    m.uv_box(0.5)
    return m


def fence(length: float = 6.0, height: float = 1.0, posts: int | None = None, color="#c8b28f",
          name: str = "Fence") -> Mesh:
    posts = posts or max(int(length / 1.2) + 1, 2)
    mat = Material("FenceWood", color, 0.0, 0.8)
    xs = np.linspace(-length / 2, length / 2, posts)
    parts = [box((0.1, height, 0.1)).translate((x, height / 2, 0)) for x in xs]
    for y in (height * 0.35, height * 0.8):
        parts.append(box((length, 0.08, 0.04)).translate((0, y, 0.07)))
    m = Mesh.join(parts, name=name)
    m.set_material(mat)
    return m


def stairs(steps: int = 8, width: float = 1.2, rise: float = 0.18, run: float = 0.28, material=None,
           name: str = "Stairs") -> Mesh:
    parts = [box((width, rise * (i + 1), run)).translate((0, rise * (i + 1) / 2, -i * run)) for i in range(steps)]
    m = Mesh.join(parts, name=name)
    m.set_material(material or Material("Stone", "#9a958d", 0.0, 0.9))
    return m


def column(height: float = 4.0, radius: float = 0.3, flutes: int = 16, material=None, name: str = "Column") -> Mesh:
    """Classical column (base, fluted shaft, capital)."""
    prof = [(0, 0), (radius * 1.5, 0), (radius * 1.5, 0.15), (radius * 1.25, 0.25), (radius * 1.1, 0.3),
            (radius, 0.35), (radius * 0.9, height - 0.4), (radius * 1.1, height - 0.3), (radius * 1.5, height - 0.15),
            (radius * 1.5, height), (0, height)]
    m = lathe(prof, flutes * 2, name=name)
    ang = np.arctan2(m.vertices[:, 2], m.vertices[:, 0])
    shaft = (m.vertices[:, 1] > 0.4) & (m.vertices[:, 1] < height - 0.45)
    r = np.hypot(m.vertices[:, 0], m.vertices[:, 2])
    k = np.where(shaft, 1 - 0.06 * (0.5 + 0.5 * np.cos(ang * flutes)), 1.0)
    m.vertices[:, 0] = np.cos(ang) * r * k
    m.vertices[:, 2] = np.sin(ang) * r * k
    m.set_material(material or presets.marble())
    m.uv_cylinder()
    return m


def street_lamp(height: float = 4.0, name: str = "StreetLamp") -> Mesh:
    pole = cylinder(0.07, height, 12).translate((0, height / 2, 0))
    base = cylinder(0.18, 0.3, 12).translate((0, 0.15, 0))
    arm = box((0.9, 0.06, 0.06)).translate((0.4, height - 0.1, 0))
    head = rounded_box((0.5, 0.15, 0.3), 0.04).translate((0.8, height - 0.17, 0))
    metal = presets.metal("#2d3136", 0.5)
    for p in (pole, base, arm, head):
        p.set_material(metal)
    bulb = box((0.4, 0.04, 0.22)).translate((0.8, height - 0.26, 0))
    bulb.set_material(presets.emissive("#fff2c4", 4.0))
    return Mesh.join([pole, base, arm, head, bulb], name=name)


def road(length: float = 20.0, width: float = 7.0, lanes: int = 2, name: str = "Road") -> Mesh:
    m = plane((width, length), (1, 1), name=name)
    m.set_material(Material("Asphalt", "#3a3b3d", 0.0, 0.9))
    parts = [m]
    lw = width / lanes
    for i in range(1, lanes):
        for z in np.arange(-length / 2 + 1, length / 2, 3.0):
            mark = plane((0.15, 1.5), 1).translate((-width / 2 + i * lw, 0.005, z + 0.75))
            mark.set_material(Material("RoadPaint", "#f0f0e8", 0.0, 0.6))
            parts.append(mark)
    return Mesh.join(parts, name=name)
