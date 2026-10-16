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

# armor steel sits mid-gray so the toon light band doesn't turn the plates into a white shirt;
# narrow highlights (keel, blade edges) carry the shine
STEEL = p3.Material("Metal_Steel", "#8f98aa", metallic=0.7, roughness=0.35)
EDGE = p3.Material("Metal_Edge", "#e6ecf5", metallic=0.7, roughness=0.25)
BLADE = p3.Material("Metal_Blade", "#aeb8c8", metallic=0.7, roughness=0.3)
STEEL_DARK = p3.Material("Metal_SteelDark", "#6c7480", metallic=0.7, roughness=0.45)
BRASS = p3.Material("Metal_Brass", "#c9a24a", metallic=0.8, roughness=0.35)
GRIP = p3.Material("Leather_Grip", "#5a3a26", roughness=0.85)
STRAP = p3.Material("Leather_Strap", "#6b4a2e", roughness=0.85)


def sword_iron() -> p3.Mesh:
    """~0.98 m arming sword: fullered diamond blade, curved crossguard, wrapped grip, pommel."""
    parts = []
    # blade: flat hexagonal section (no inner ridge, so no ink crease inside the blade), 8 cm wide
    # at the base (stylized: >= 2 px at game scale), light bevels along both edges
    L0, L1 = 0.075, 0.86
    shape = [(-1, 0), (-0.55, 0.2), (0.55, 0.2), (1, 0), (0.55, -0.2), (-0.55, -0.2)]
    blade = p3.sweep([(0, L0, 0), (0, (L0 + L1) / 2, 0), (0, L1 - 0.06, 0), (0, L1, 0)], shape=shape,
                     radius=lambda t: 0.04 * max(0.0, 1 - max(0.0, (t - 0.8) / 0.2)) ** 0.85 + 0.0015, segments=8, up=(0, 0, 1))
    blade.rotate((0, 90, 0))  # flat of the blade faces +-Z, edges along X like the crossguard
    blade.set_material(BLADE)
    fn = blade.face_normals()
    blade.set_material(EDGE, np.flatnonzero(np.abs(fn[:, 0]) > 0.5))  # bevels catch the light
    parts.append(blade)
    # crossguard: slightly curved bar with brass ends
    guard = p3.sweep([(-0.12, 0.078, 0), (-0.06, 0.065, 0), (0.06, 0.065, 0), (0.12, 0.078, 0)], radius=0.013, segments=8)
    parts.append(guard.set_material(STEEL_DARK))
    for sx in (-1, 1):
        parts.append(p3.uv_sphere(0.019, 10, 6).translate((sx * 0.124, 0.08, 0)).set_material(BRASS))
    parts.append(p3.box((0.04, 0.03, 0.026)).translate((0, 0.062, 0)).set_material(BRASS))
    # grip with leather wrap rings, pommel
    parts.append(p3.cylinder(0.0145, 0.13, 10, rings=6).translate((0, -0.01, 0)).set_material(GRIP))
    for k in range(5):
        parts.append(p3.torus(0.0152, 0.0028, 10, 4).translate((0, -0.06 + k * 0.025, 0)).set_material(STRAP))
    parts.append(p3.uv_sphere(0.024, 10, 8).scale((1, 0.8, 1)).translate((0, -0.088, 0)).set_material(BRASS))
    # held with the flat of the blade facing out from the body, so the broad side shows in game
    return p3.Mesh.join(parts, name="Sword_Iron").rotate((0, 90, 0))


# ----------------------------------------------------------------------------- armor

def _mat(shape, m: p3.Material):
    return shape.paint(color=(1, 1, 1), material=m)


def _slab(y0, y1, m):
    return _mat(sdf.plane((0, y1, 0), (0, 1, 0)).intersect(sdf.plane((0, y0, 0), (0, -1, 0))), m)


def _p(spec, x, y, z):
    return np.array([x, y * spec.s, z])


def cuirass(spec: BodySpec):
    """Breastplate with a central keel, brass trim at the neck and arm holes and 3 fauld lames
    at the waist (each a step out from the one above, so the outline draws the joints)."""
    from clothes import a_line, skirt_radius

    s = spec.s
    t = 0.045 if spec.gender == "m" else 0.04

    # paint() recolors the leaves it reaches, so every use gets freshly built shapes
    def shell():
        body = sdf.smooth_union(*torso(spec), k=0.03)
        return body.round(t).intersect(_slab(1.0 * s, 1.43 * s, STEEL))

    def neck():
        return sdf.ellipsoid(_p(spec, 0, 1.45, 0.02), (0.11, 0.07, 0.1))

    def arm():
        return sdf.sphere(_p(spec, spec.shoulder + 0.02, 1.4, -0.02), 0.085).mirror_x()

    # solid plate: a 1-2 cm hollow shell is thinner than the meshing cell and comes out pitted
    plate = shell().subtract(neck()).subtract(arm())
    if spec.harpin:  # openings for the wing roots
        plate = plate.subtract(sdf.ellipsoid(_p(spec, 0.085, 1.37, -0.14), (0.06, 0.11, 0.08)).mirror_x())
    plate = _mat(plate, STEEL)
    # brass trim: a 1.2 cm band of the plate's skin around the neck and arm openings
    trim = shell().intersect(neck().round(0.012).union(arm().round(0.012))).subtract(neck().union(arm()))
    trim = _mat(trim.round(0.004), BRASS)
    keel = _mat(sdf.capsule(_p(spec, 0, 1.05, 0.105 + t * 0.5), _p(spec, 0, 1.37, 0.12 + t * 0.5), 0.014), EDGE)  # highlight
    # fauld: 3 lames below the plate, stepping out
    lames = []
    for k in range(3):
        y1 = 1.005 * s - k * 0.042 * s
        y0 = y1 - 0.046 * s
        rx, rz, zc = skirt_radius(spec, y1)
        g = 0.032 + 0.012 * k
        lames.append(_mat(a_line(y1, y0, (rx + g, rz + g, zc), (rx + g + 0.008, rz + g + 0.008, zc)), STEEL))
    return sdf.union(plate, trim, keel, *lames)


def pauldrons(spec: BodySpec):
    """Three stepped lames (flattened domes, ~3.5 cm drop between them) with a brass rim."""
    J = joints(spec)
    sh = J["LeftUpperArm"][0]
    parts = []
    for k, (r, dy, dx) in enumerate(((0.11, 0.05, 0.02), (0.096, 0.015, 0.065), (0.082, -0.02, 0.105))):
        c = sh + np.array([dx, dy, 0.0])
        h = r * 0.5  # flattened: height about half the width
        dome = sdf.ellipsoid(c, (r, h, r * 0.95))
        dome = dome.subtract(sdf.ellipsoid(c - np.array([0, 0.02, 0]), (r - 0.014, h - 0.014, r * 0.95 - 0.014)))
        dome = dome.intersect(sdf.plane(c - np.array([0, h * 0.45, 0]), (0, -1, 0)))
        parts.append(_mat(dome, STEEL if k != 1 else STEEL_DARK))
        if k == 2:  # brass rim on the lowest lame
            rim = sdf.ellipsoid(c, (r + 0.006, h + 0.006, r * 0.95 + 0.006))
            rim = rim.intersect(_slab(c[1] - h * 0.45, c[1] - h * 0.45 + 0.015, BRASS)).subtract(
                sdf.ellipsoid(c, (r - 0.012, h - 0.012, r * 0.95 - 0.012)))
            parts.append(_mat(rim, BRASS))
    parts.append(_mat(sdf.sphere(sh + np.array([0.02, 0.105, 0.0]), 0.014), BRASS))  # rivet
    return sdf.union(*parts).mirror_x()


def bracers(spec: BodySpec):
    J = joints(spec)
    la0, la1 = J["LeftLowerArm"]
    a = la0 + (la1 - la0) * 0.38
    b = la0 + (la1 - la0) * 0.92
    L = spec.limb
    br = _mat(sdf.round_cone(a, b, 0.056 * L, 0.044 * L), STEEL)
    br = br.paint(color=(1, 1, 1), material=STEEL)
    cuff = _mat(sdf.round_cone(b - (b - a) * 0.12, b, 0.05 * L, 0.048 * L), BRASS)
    return br.union(cuff).mirror_x()


def tassets(spec: BodySpec):
    """Leather belt with a brass buckle over the fauld, and three tassets of 2 lames (13 cm wide,
    slightly curved, riveted) hanging from it over the skirt; weighted like the skirt."""
    from clothes import a_line, skirt_radius

    s = spec.s
    by = 0.885 * s
    brx, brz, bzc = skirt_radius(spec, by)
    belt = _mat(a_line(by + 0.022, by - 0.022, (brx + 0.07, brz + 0.07, bzc), (brx + 0.072, brz + 0.072, bzc)), STRAP)
    buckle = _mat(sdf.box((0, by, bzc + brz + 0.078), (0.05, 0.05, 0.018), 0.006), BRASS)
    parts = [belt, buckle]
    for ang in (-55, 0, 55):
        a = math.radians(ang)
        for k in range(2):
            y = by - 0.045 - k * 0.075 * s
            rx, rz, zc = skirt_radius(spec, y)
            out = 0.075 + 0.012 * k
            cx, cz = math.sin(a) * (rx + out), zc + math.cos(a) * (rz + out)
            c = (cx, y, cz)
            # a short arc of an ellipse shell: curved plate
            plate = sdf.ellipsoid((0, y, zc), (rx + out + 0.012, 0.042 * s, rz + out + 0.012))
            plate = plate.subtract(sdf.ellipsoid((0, y + 0.01, zc), (rx + out - 0.008, 0.05 * s, rz + out - 0.008)))
            plate = plate.intersect(sdf.box(c, (0.24, 0.1, 0.24)).rotate((0, ang, 0), c))
            plate = plate.intersect(sdf.plane((0, 0, 0), (math.cos(a), 0, -math.sin(a)))
                                    .translate((0.065 * math.cos(a) + cx * 0, 0, -0.065 * math.sin(a))))
            plate = plate.intersect(sdf.plane((0, 0, 0), (-math.cos(a), 0, math.sin(a)))
                                    .translate((-0.065 * math.cos(a), 0, 0.065 * math.sin(a))))
            parts.append(_mat(plate, STEEL if k == 0 else STEEL_DARK))
            if k == 0:
                for side in (-1, 1):
                    rv = (cx + side * 0.045 * math.cos(a), y + 0.02, cz - side * 0.045 * math.sin(a))
                    rv = np.array(rv) + np.array([math.sin(a), 0, math.cos(a)]) * 0.012
                    parts.append(_mat(sdf.sphere(tuple(rv), 0.011), BRASS))
    return sdf.union(*parts)


def greaves(spec: BodySpec):
    J = joints(spec)
    c0, c1 = J["LeftLowerLeg"]
    L = spec.limb
    if spec.harpin:  # shin guard on the tibiotarsus, above the plumed ankle
        a = c0 + (c1 - c0) * 0.18
        b = c0 + (c1 - c0) * 0.8
        g = _mat(sdf.round_cone(a, b, 0.07 * L, 0.05 * L), STEEL)  # solid (thin shells mesh pitted)
        g = g.intersect(_mat(sdf.plane(a + (b - a) * 0.5 + np.array([0, 0, -0.01]), (0, 0, -1)), STEEL))  # front half
    else:
        a = c0 + (c1 - c0) * 0.06
        b = c0 + (c1 - c0) * 0.62
        g = _mat(sdf.round_cone(a, b, 0.078 * L, 0.07 * L), STEEL)
        g = g.intersect(_mat(sdf.plane(c0 + np.array([0, 0, -0.01]), (0, 0, -1)), STEEL))
    knee = _mat(sdf.sphere(c0 + np.array([0, 0.0, 0.05]), 0.045 * L), STEEL)
    knee = knee.subtract(_mat(sdf.sphere(c0 + np.array([0, 0.0, 0.02]), 0.05 * L), STEEL_DARK))
    # side wing of the knee cop (a flat fan on the outer side)
    wing = _mat(sdf.ellipsoid(c0 + np.array([0.05 * L, -0.005, 0.03]), (0.012, 0.045, 0.04)), STEEL_DARK)
    parts = [g, knee, wing]
    if not spec.harpin:  # greave flares a little over the boot at the ankle
        a = c0 + (c1 - c0) * 0.55
        b = c0 + (c1 - c0) * 0.66
        flare = _mat(sdf.round_cone(a, b, 0.07 * L, 0.082 * L), STEEL).intersect(
            _mat(sdf.plane(c0 + np.array([0, 0, -0.01]), (0, 0, -1)), STEEL))
        parts.append(flare)
    return sdf.union(*parts).mirror_x()


def armor_parts(spec: BodySpec):
    """[(name, sdf, cell, skirt?)]"""
    return [
        # cells sized so the set lands near the 4k-triangle budget after simplification
        ("Armor_Cuirass", cuirass(spec), 0.017, False),
        ("Armor_Pauldrons", pauldrons(spec), 0.013, False),
        ("Armor_Bracers", bracers(spec), 0.013, False),
        ("Armor_Tassets", tassets(spec), 0.013, True),
        ("Armor_Greaves", greaves(spec), 0.014, False),
    ]
