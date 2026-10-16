"""Procedural vehicles. Each returns a Node hierarchy whose wheels are separate
child nodes (named "Wheel_FL", "Wheel_FR", ...) so they can be animated.

Vehicles face +Z and stand on y = 0.
"""
from __future__ import annotations

import math

import numpy as np

from .. import math3d as m3
from ..animation import Animation
from ..material import Material, presets
from ..mesh import Mesh
from ..primitives import box, cylinder, loft, rounded_box, shapes, torus, uv_sphere
from ..scene import Node


def wheel(radius: float = 0.34, width: float = 0.24, rim_color="#c8ccd0", spokes: int = 5) -> Mesh:
    """Tire + rim, axle along X."""
    tire = torus(radius - width * 0.35, width * 0.42, 32, 12).scale((1, width / (width * 0.84) * 0.9, 1))
    tire.rotate((0, 0, 90))
    tire.set_material(presets.rubber())
    rim = cylinder(radius * 0.62, width * 0.7, 24).rotate((0, 0, 90))
    rim.set_material(presets.metal(rim_color, 0.25))
    hub = cylinder(radius * 0.15, width * 0.8, 12).rotate((0, 0, 90))
    hub.set_material(presets.metal("#555a60", 0.4))
    m = Mesh.join([tire, rim, hub], name="Wheel")
    for i in range(spokes):
        s = box((width * 0.2, radius * 0.5, radius * 0.08)).translate((width * 0.3, radius * 0.3, 0))
        s.rotate((360 * i / spokes, 0, 0))
        s.set_material(presets.metal(rim_color, 0.25))
        m.merge(s)
    return m


def car(color="#c0392b", length: float = 4.3, width: float = 1.8, height: float = 1.4, style: str = "sedan",
        name: str = "Car") -> Node:
    """Stylized car. style: "sedan" | "hatchback" | "suv" | "pickup"."""
    L, W, H = length, width, height
    paint = Material("CarPaint", color, 0.6, 0.25)
    glass = Material("CarGlass", "#1d2a33", 0.3, 0.05)
    trim = presets.rubber("#151515")
    lights = presets.emissive("#fff6d8", 3.0, name="Headlight")
    tail = presets.emissive("#ff2a2a", 2.0, name="Taillight")
    clearance = 0.18 if style != "suv" else 0.28
    body_h = H * (0.42 if style != "suv" else 0.48)
    # lower body: loft along Z with rounded cross sections
    zs = np.linspace(-L / 2, L / 2, 9)
    secs = []
    for z in zs:
        t = (z + L / 2) / L
        nose = 1 - 0.25 * max(0.0, (t - 0.85) / 0.15) ** 2 - 0.2 * max(0.0, (0.1 - t) / 0.1) ** 2
        rr = shapes.rounded_rect(W * (0.96 + 0.04 * nose), body_h * nose, body_h * 0.25, 4)
        secs.append(np.column_stack([rr[:, 0], rr[:, 1] + clearance + body_h / 2 - (1 - nose) * body_h * 0.1,
                                     np.full(len(rr), z)]))
    body = loft(secs, name="Body")
    body.set_material(paint)
    # cabin (greenhouse)
    cab_len = {"sedan": 0.5, "hatchback": 0.55, "suv": 0.62, "pickup": 0.32}.get(style, 0.5)
    cab_back = {"sedan": -0.28, "hatchback": -0.45, "suv": -0.46, "pickup": -0.05}.get(style, -0.28)
    top_y = H
    base_y = clearance + body_h
    cl = L * cab_len
    cz0 = L * cab_back
    cz1 = cz0 + cl
    cab_shape = [(cz0, base_y), (cz1 + cl * 0.18, base_y), (cz1 - cl * 0.12, top_y), (cz0 + cl * 0.12, top_y)]
    if style in ("hatchback", "suv"):
        cab_shape = [(cz0, base_y), (cz1 + cl * 0.18, base_y), (cz1 - cl * 0.1, top_y), (cz0 + 0.02, top_y)]
    from ..primitives import extrude_shape
    cab = extrude_shape([(z, y) for z, y in cab_shape], depth=W * 0.82)
    cab.rotate((0, -90, 0))  # shape X -> world Z, extrusion -> X
    cab.set_material(glass)
    top_z0, top_z1 = cab_shape[3][0], cab_shape[2][0]
    roof = rounded_box((W * 0.84, 0.07, top_z1 - top_z0 + 0.04), 0.025).translate(
        (0, top_y, (top_z0 + top_z1) / 2))
    roof.set_material(paint)
    parts = [body, cab, roof]
    if style == "pickup":
        bed = box((W * 0.92, 0.35, L * 0.36)).translate((0, base_y + 0.17, -L * 0.27))
        bed.set_material(paint)
        inner = box((W * 0.8, 0.3, L * 0.33)).translate((0, base_y + 0.22, -L * 0.27))
        inner.set_material(trim)
        parts += [bed, inner]
    for sx in (1, -1):
        hl = rounded_box((W * 0.2, body_h * 0.18, 0.06), 0.02).translate((sx * W * 0.33, clearance + body_h * 0.68,
                                                                        L / 2 - 0.02))
        hl.set_material(lights)
        tl = rounded_box((W * 0.2, body_h * 0.15, 0.06), 0.02).translate((sx * W * 0.35, clearance + body_h * 0.7,
                                                                        -L / 2 + 0.02))
        tl.set_material(tail)
        mirror = rounded_box((0.08, 0.1, 0.16), 0.02).translate((sx * (W / 2 + 0.05), base_y + 0.12, cz1 + 0.05))
        mirror.set_material(paint)
        parts += [hl, tl, mirror]
    for z in (L / 2, -L / 2):
        bumper = rounded_box((W * 0.98, 0.16, 0.12), 0.05).translate((0, clearance + 0.12, z * 0.995))
        bumper.set_material(trim)
        parts.append(bumper)
    chassis = Mesh.join(parts, name=name + "_Body")
    chassis.smooth_angle = 35.0
    root = Node(name)
    root.add(chassis)
    r = clearance + body_h * 0.35
    wb = L * 0.32
    for label, sx, z in (("FL", 1, wb), ("FR", -1, wb), ("RL", 1, -wb), ("RR", -1, -wb)):
        wm = wheel(r, 0.24)
        if sx < 0:
            wm.rotate((0, 180, 0))
        root.add(wm, name=f"Wheel_{label}", position=(sx * (W / 2 - 0.12), r, z))
    # wheel arches (dark cut-outs)
    root.extras["wheel_radius"] = r
    return root


def drive(anim: Animation, vehicle: Node, distance: float = 10.0, duration: float = 3.0, steps: int = 12):
    """Move a vehicle forward along +Z (its facing) and spin its wheels accordingly."""
    r = vehicle.extras.get("wheel_radius", 0.35)
    start = vehicle.translation.copy()
    fwd = m3.quat_rotate(vehicle.rotation, (0, 0, 1))
    for k in range(steps + 1):
        t = duration * k / steps
        anim.key(vehicle, "position", t, start + fwd * distance * k / steps)
    turns = distance / (2 * math.pi * r)
    for n in vehicle.walk():
        if n.name.startswith("Wheel_"):
            base = n.rotation.copy()
            for k in range(steps + 1):
                ang = 360.0 * turns * k / steps
                anim.key(n, "rotation", duration * k / steps,
                         m3.quat_mul(base, m3.quat_from_axis_angle((1, 0, 0), ang if not _flipped(base) else -ang)))
    return anim


def _flipped(q) -> bool:
    return abs(m3.quat_rotate(q, (1, 0, 0))[0] + 1) < 1e-6


def boat(length: float = 5.0, color="#e8e2d6", hull_color="#2c4a6e", name: str = "Boat") -> Mesh:
    """Small motorboat hull built by lofting cross sections."""
    secs = []
    for t in np.linspace(0, 1, 11):
        z = (t - 0.5) * length
        bow = max(0.0, (t - 0.6) / 0.4)
        w = length * 0.2 * (1 - bow ** 1.8 * 0.95)
        d = length * 0.12 * (1 - bow * 0.3)
        keel = d * (0.2 + 0.6 * bow)
        pts = [(-w, d * 0.4), (-w * 0.9, 0), (-w * 0.4, -d * 0.4), (0, -keel), (w * 0.4, -d * 0.4), (w * 0.9, 0),
               (w, d * 0.4)]
        pts = [(x, y + d * bow * 0.6) for x, y in pts]
        secs.append([(x, y + length * 0.12, z) for x, y in pts] + [(0, length * 0.12 + d * 0.45 + d * bow * 0.6, z)])
    hull = loft(secs, name=name)
    hull.set_material(Material("Hull", hull_color, 0.1, 0.4))
    deck_y = hull.bounds()[1][1]
    cabin = rounded_box((length * 0.25, length * 0.12, length * 0.22), 0.05).translate((0, deck_y + length * 0.05, -0.1))
    cabin.set_material(Material("Cabin", color, 0.0, 0.4))
    hull.merge(cabin)
    return hull.ground()


def airplane(span: float = 10.0, color="#e8e8ec", accent="#c0392b", name: str = "Airplane") -> Mesh:
    """Stylized propeller airplane facing +Z."""
    L = span * 0.8
    fus = loft([np.column_stack([shapes.ellipse(r, r * 1.1, 16), np.full(16, z)])
                for z, r in ((-L / 2, 0.08), (-L * 0.3, 0.35), (0, 0.55), (L * 0.3, 0.55), (L / 2, 0.3))],
               name="Fuselage")
    fus.vertices += np.array([0, 1.5, 0])
    paint = Material("Paint", color, 0.3, 0.35)
    fus.set_material(paint)
    wing = rounded_box((span, 0.18, L * 0.18), 0.08).translate((0, 1.35, L * 0.08))
    tail_w = rounded_box((span * 0.35, 0.12, L * 0.1), 0.05).translate((0, 1.7, -L * 0.45))
    fin = rounded_box((0.12, 1.2, L * 0.12), 0.05).translate((0, 2.3, -L * 0.45))
    for p in (wing, tail_w):
        p.set_material(paint)
    fin.set_material(Material("Accent", accent, 0.2, 0.4))
    prop = Mesh.join([box((0.12, 1.8, 0.05)).translate((0, 1.5, L / 2 + 0.05)),
                      uv_sphere(0.18, 12, 8).translate((0, 1.5, L / 2 + 0.05))], name="Propeller")
    prop.set_material(presets.metal("#333333", 0.4))
    m = Mesh.join([fus, wing, tail_w, fin, prop], name=name)
    return m.ground()
