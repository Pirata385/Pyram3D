"""Equipment: the iron sword and the "Guarda de Rondoria" armor set.

The sword is a rigid model (grip at the origin, blade along +Y) attached to Socket_Weapon_R.
The armor is generated from each body's own SDF (dilated shells), so it fits all 4 variants, and
it is skinned on the same bone names (weights copied from the body), so the game re-binds it to
the character's skeleton. Harpin version: openings for the wings, greaves on the shins only (the
plumed feet and talons stay free).
"""
from __future__ import annotations

import math

import numpy as np

from body import torso
from common import BodySpec, joints

import pyram3d as p3
from pyram3d import sdf

STEEL = p3.Material("Metal_Steel", "#b9c0c8", metallic=0.7, roughness=0.35)
STEEL_DARK = p3.Material("Metal_SteelDark", "#6c7480", metallic=0.7, roughness=0.45)
BRASS = p3.Material("Metal_Brass", "#c9a24a", metallic=0.8, roughness=0.35)
GRIP = p3.Material("Leather_Grip", "#5a3a26", roughness=0.85)
STRAP = p3.Material("Leather_Strap", "#7a4e30", roughness=0.85)


def sword_iron() -> p3.Mesh:
    """~0.98 m arming sword: fullered diamond blade, curved crossguard, wrapped grip, pommel."""
    parts = []
    # blade: diamond section swept along +Y, tapering to a point
    L0, L1 = 0.075, 0.86
    shape = [(-1, 0), (-0.15, 0.22), (0, 0.24), (0.15, 0.22), (1, 0), (0.15, -0.22), (0, -0.24), (-0.15, -0.22)]
    blade = p3.sweep([(0, L0, 0), (0, (L0 + L1) / 2, 0), (0, L1 - 0.06, 0), (0, L1, 0)], shape=shape,
                     radius=lambda t: 0.03 * max(0.0, 1 - max(0.0, (t - 0.82) / 0.18)) ** 0.9 + 0.0015, segments=8, up=(0, 0, 1))
    blade.rotate((0, 90, 0))  # flat of the blade faces +-Z, edges along X like the crossguard
    blade.set_material(STEEL)
    parts.append(blade)
    # fuller: dark groove along the middle of the blade (thin inset strip, both faces)
    for sz in (1, -1):
        fuller = p3.box((0.009, 0.5, 0.002)).translate((0, L0 + 0.3, sz * 0.0062)).set_material(STEEL_DARK)
        parts.append(fuller)
    # crossguard: slightly curved bar with brass ends
    guard = p3.sweep([(-0.1, 0.075, 0), (-0.05, 0.065, 0), (0.05, 0.065, 0), (0.1, 0.075, 0)], radius=0.011, segments=8)
    parts.append(guard.set_material(STEEL_DARK))
    for sx in (-1, 1):
        parts.append(p3.uv_sphere(0.017, 10, 6).translate((sx * 0.104, 0.077, 0)).set_material(BRASS))
    parts.append(p3.box((0.04, 0.03, 0.026)).translate((0, 0.062, 0)).set_material(BRASS))
    # grip with leather wrap rings, pommel
    parts.append(p3.cylinder(0.0145, 0.13, 10, rings=6).translate((0, -0.01, 0)).set_material(GRIP))
    for k in range(5):
        parts.append(p3.torus(0.0152, 0.0028, 10, 4).translate((0, -0.06 + k * 0.025, 0)).set_material(STRAP))
    parts.append(p3.uv_sphere(0.024, 10, 8).scale((1, 0.8, 1)).translate((0, -0.088, 0)).set_material(BRASS))
    return p3.Mesh.join(parts, name="Sword_Iron")


# ----------------------------------------------------------------------------- armor

def _mat(shape, m: p3.Material):
    return shape.paint(color=(1, 1, 1), material=m)


def _slab(y0, y1, m):
    return _mat(sdf.plane((0, y1, 0), (0, 1, 0)).intersect(sdf.plane((0, y0, 0), (0, -1, 0))), m)


def _p(spec, x, y, z):
    return np.array([x, y * spec.s, z])


def cuirass(spec: BodySpec):
    s = spec.s
    body = sdf.smooth_union(*torso(spec), k=0.03)
    t = 0.045 if spec.gender == "m" else 0.04
    shell = _mat(body.round(t), STEEL).intersect(_slab(1.0 * s, 1.43 * s, STEEL))
    shell = shell.subtract(_mat(body.round(t - 0.012), STEEL_DARK))  # hollow plate (2 cm)
    # neck opening and arm holes
    shell = shell.subtract(_mat(sdf.ellipsoid(_p(spec, 0, 1.45, 0.02), (0.11, 0.07, 0.1)), STEEL_DARK))
    shell = shell.subtract(_mat(sdf.sphere(_p(spec, spec.shoulder + 0.02, 1.4, -0.02), 0.085), STEEL_DARK).mirror_x())
    if spec.harpin:  # openings for the wing roots
        shell = shell.subtract(_mat(sdf.ellipsoid(_p(spec, 0.085, 1.37, -0.14), (0.06, 0.11, 0.08)), STEEL_DARK).mirror_x())
    shell = shell.paint(color=(1, 1, 1), material=STEEL)  # one material for the whole plate
    # central ridge and a brass band at the bottom edge (separate shapes, painted after)
    ridge = _mat(sdf.capsule(_p(spec, 0, 1.06, 0.105 + t * 0.5), _p(spec, 0, 1.36, 0.12 + t * 0.5), 0.012), STEEL)
    hip_r = 0.15 if spec.gender == "m" else 0.155
    rim = _mat(sdf.ellipsoid(_p(spec, 0, 1.012, -0.004), (hip_r + t * 0.75, 0.05, 0.118 + t * 0.6)), BRASS)
    rim = rim.intersect(_slab(0.998 * s, 1.026 * s, BRASS))
    return shell.union(ridge).union(rim)


def pauldrons(spec: BodySpec):
    J = joints(spec)
    sh = J["LeftUpperArm"][0]
    parts = []
    for k, (r, dy, dx) in enumerate(((0.105, 0.035, 0.02), (0.088, -0.005, 0.06), (0.074, -0.04, 0.095))):
        c = sh + np.array([dx, dy, 0.0])
        dome = _mat(sdf.sphere(c, r), STEEL if k == 0 else STEEL_DARK)
        dome = dome.subtract(_mat(sdf.sphere(c - np.array([0, 0.03, 0]), r - 0.015), STEEL_DARK))
        dome = dome.intersect(_mat(sdf.plane(c - np.array([0, r * 0.25, 0]), (0, -1, 0)), STEEL))
        parts.append(dome.paint(color=(1, 1, 1), material=STEEL if k == 0 else STEEL_DARK))
    parts.append(_mat(sdf.sphere(sh + np.array([0.02, 0.105, 0.0]), 0.014), BRASS))  # rivet
    return sdf.union(*parts).mirror_x()


def bracers(spec: BodySpec):
    J = joints(spec)
    la0, la1 = J["LeftLowerArm"]
    a = la0 + (la1 - la0) * 0.38
    b = la0 + (la1 - la0) * 0.92
    L = spec.limb
    br = _mat(sdf.round_cone(a, b, 0.056 * L, 0.044 * L), STEEL)
    br = br.subtract(_mat(sdf.round_cone(a - (b - a) * 0.1, b + (b - a) * 0.1, 0.046 * L, 0.034 * L), STEEL)).paint(color=(1, 1, 1), material=STEEL)
    cuff = _mat(sdf.round_cone(b - (b - a) * 0.12, b, 0.05 * L, 0.048 * L), BRASS)
    return br.union(cuff).mirror_x()


def tassets(spec: BodySpec):
    """Three hanging plates over the hips (front, sides), weighted like the skirt."""
    s = spec.s
    parts = []
    hip_r = 0.19 if spec.gender == "f" else 0.18
    for ang in (-50, 0, 50):
        a = math.radians(ang)
        cx, cz = math.sin(a) * hip_r, math.cos(a) * hip_r
        plate = _mat(sdf.box(_p(spec, cx, 0.9, cz), (0.12, 0.16 * s, 0.02), 0.008), STEEL_DARK).rotate((8, ang, 0), tuple(_p(spec, cx, 0.9, cz)))
        parts.append(plate)
    belt = _mat(sdf.ellipsoid(_p(spec, 0, 0.995, -0.005), (hip_r - 0.005, 0.07, 0.15)), STRAP).intersect(_slab(0.965 * s, 1.02 * s, STRAP))
    belt = belt.subtract(_mat(sdf.ellipsoid(_p(spec, 0, 0.995, -0.005), (hip_r - 0.025, 0.07, 0.13)), STRAP))
    return sdf.union(belt, *parts)


def greaves(spec: BodySpec):
    J = joints(spec)
    c0, c1 = J["LeftLowerLeg"]
    L = spec.limb
    if spec.harpin:  # shin guard on the tibiotarsus, above the plumed ankle
        a = c0 + (c1 - c0) * 0.18
        b = c0 + (c1 - c0) * 0.8
        g = _mat(sdf.round_cone(a, b, 0.07 * L, 0.05 * L), STEEL)
        g = g.subtract(_mat(sdf.round_cone(a, b, 0.056 * L, 0.04 * L), STEEL_DARK))
        g = g.intersect(_mat(sdf.plane(a + (b - a) * 0.5 + np.array([0, 0, -0.01]), (0, 0, -1)), STEEL))  # front half
    else:
        a = c0 + (c1 - c0) * 0.06
        b = c0 + (c1 - c0) * 0.62
        g = _mat(sdf.round_cone(a, b, 0.078 * L, 0.07 * L), STEEL)
        g = g.subtract(_mat(sdf.round_cone(a, b, 0.064 * L, 0.058 * L), STEEL_DARK))
        g = g.intersect(_mat(sdf.plane(c0 + np.array([0, 0, -0.01]), (0, 0, -1)), STEEL))
    knee = _mat(sdf.sphere(c0 + np.array([0, 0.0, 0.05]), 0.045 * L), STEEL)
    knee = knee.subtract(_mat(sdf.sphere(c0 + np.array([0, 0.0, 0.02]), 0.05 * L), STEEL_DARK))
    return g.union(knee).mirror_x()


def armor_parts(spec: BodySpec):
    """[(name, sdf, cell, skirt?)]"""
    return [
        ("Armor_Cuirass", cuirass(spec), 0.012, False),
        ("Armor_Pauldrons", pauldrons(spec), 0.01, False),
        ("Armor_Bracers", bracers(spec), 0.009, False),
        ("Armor_Tassets", tassets(spec), 0.01, True),
        ("Armor_Greaves", greaves(spec), 0.01, False),
    ]
