"""Equipment for prototype 2: the iron sword.

Sword model space: the grip center is the origin and the hand closes around the Y axis; the
blade points along +Y, its edges along +-X, its flats face +-Z. Attached under Socket_Weapon_R
(no offset), the blade follows the right hand's thumb axis and the edges line up with the
knuckles (see common.grip_frame). Slots: Metal (blade, guard, pommel) and Leather (grip), both
tinted by the client; vertex colors carry the edge highlights and the leather wrap.
"""
from __future__ import annotations

import math

import numpy as np

from common import GRIP_RADIUS, slot

import pyram3d as p3

GRIP_HALF = 0.058  # grip length 11.6 cm (one big hand)
GUARD_Y = 0.066
BLADE_BASE = 0.082
BLADE_TIP = 0.86


def _section(w, t, bevel):
    """Flat hexagonal blade section: wide flats (+-Z), bevelled edges (+-X)."""
    e = min(bevel, w * 0.6)
    return [(-w, 0.0), (-w + e, -t), (w - e, -t), (w, 0.0), (w - e, t), (-w + e, t)]


def blade() -> p3.Mesh:
    secs = []
    for y, w, t in ((BLADE_BASE - 0.006, 0.031, 0.0072), (BLADE_BASE + 0.03, 0.032, 0.0072), (0.45, 0.028, 0.0064),
                    (0.74, 0.022, 0.0055), (0.81, 0.015, 0.0046), (0.845, 0.0065, 0.003), (BLADE_TIP, 0.0005, 0.0006)):
        secs.append([(x, y, z) for (x, z) in _section(w, t, 0.012)])
    m = p3.loft(secs, closed_sections=True, caps=True, name="Blade", smooth_angle=35.0)
    m.uvs = m.face_uvs = None
    m.set_material(slot("Metal"))
    # bright bevels (edge highlight), slightly darker flats
    V = m.vertices
    m.colors = np.ones((m.num_vertices, 4))
    w_at = np.interp(V[:, 1], [BLADE_BASE, 0.45, 0.74, 0.81, 0.845, BLADE_TIP], [0.032, 0.028, 0.022, 0.015, 0.0065, 0.0005])
    edge = np.abs(V[:, 0]) > w_at * 0.75
    m.colors[:, :3] = np.where(edge[:, None], 1.0, 0.8)
    return m


def guard() -> p3.Mesh:
    path = [(-0.1, GUARD_Y + 0.012, 0), (-0.05, GUARD_Y, 0), (0.05, GUARD_Y, 0), (0.1, GUARD_Y + 0.012, 0)]
    g = p3.sweep(p3.math3d.catmull_rom(path, 4), radius=lambda t: 0.011 + 0.004 * abs(t - 0.5) * 2, segments=8)
    g.uvs = g.face_uvs = None
    g.smooth_angle = 60.0
    ends = []
    for sx in (-1, 1):
        ends.append(p3.uv_sphere(0.016, 10, 6).translate((sx * 0.104, GUARD_Y + 0.014, 0)))
    block = p3.rounded_box((0.05, 0.026, 0.03), radius=0.007, segments=2).translate((0, GUARD_Y, 0))
    block.uvs = block.face_uvs = None
    for e in ends:
        e.uvs = e.face_uvs = None
    m = p3.Mesh.join([g, block] + ends, name="Guard").set_material(slot("Metal"))
    m.colors = np.ones((m.num_vertices, 4)) * 0.85
    m.colors[:, 3] = 1
    return m


def grip() -> p3.Mesh:
    prof = [(0.0, -GRIP_HALF - 0.004), (GRIP_RADIUS * 0.92, -GRIP_HALF), (GRIP_RADIUS, -GRIP_HALF * 0.6),
            (GRIP_RADIUS * 1.04, 0.0), (GRIP_RADIUS, GRIP_HALF * 0.6), (GRIP_RADIUS * 0.92, GRIP_HALF),
            (0.0, GRIP_HALF + 0.004)]
    m = p3.lathe(prof, 12, name="Grip")
    m.uvs = m.face_uvs = None
    m.set_material(slot("Leather"))
    # leather wrap: diagonal darker bands
    V = m.vertices
    ang = np.arctan2(V[:, 2], V[:, 0])
    band = np.sin((V[:, 1] / 0.016) * 2 * math.pi + ang) > 0.55
    m.colors = np.ones((m.num_vertices, 4))
    m.colors[band, :3] = 0.7
    m.smooth_angle = 180.0
    return m


def pommel() -> p3.Mesh:
    prof = [(0.0, -GRIP_HALF - 0.046), (0.012, -GRIP_HALF - 0.044), (0.021, -GRIP_HALF - 0.032),
            (0.022, -GRIP_HALF - 0.02), (0.015, -GRIP_HALF - 0.008), (0.011, -GRIP_HALF + 0.002), (0.0, -GRIP_HALF + 0.004)]
    m = p3.lathe(prof, 14, name="Pommel")
    m.uvs = m.face_uvs = None
    m.set_material(slot("Metal"))
    m.colors = np.ones((m.num_vertices, 4)) * 0.9
    m.colors[:, 3] = 1
    return m


def sword_iron() -> p3.Mesh:
    m = p3.Mesh.join([blade(), guard(), grip(), pommel()], name="Sword_Iron")
    return m
