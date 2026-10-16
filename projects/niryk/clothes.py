"""Starter outfits (always worn) built as offset shells of each body, so they fit every variant.

Human: tunic (Cloth_A) with short sleeves, belt (Leather + Metal buckle), trousers or leggings
(Cloth_B) and boots (Leather). Harpin: sleeveless tunic open at the back for the wings, belt with a
pouch and leather wrist wraps — the plumed legs and talons stay free (ref. 06).
"""
from __future__ import annotations

import numpy as np

from body import torso
from common import BodySpec, joints, slot

from pyram3d import sdf


def mat(shape, name, shade=1.0):
    g = float(shade)
    return shape.paint(color=(g, g, g), material=slot(name))


def _p(spec, x, y, z):
    return np.array([x, y * spec.s, z])


def _slab(y0, y1):
    """Keep y0 < y < y1."""
    return sdf.plane((0, y1, 0), (0, 1, 0)).intersect(sdf.plane((0, y0, 0), (0, -1, 0)))


def tunic(spec: BodySpec):
    """Upper garment + skirt. Returns (sdf, skirt_top_y) — the skirt part is weighted to the thighs."""
    s = spec.s
    f = spec.gender == "f"
    J = joints(spec)
    t = 0.016
    body = sdf.smooth_union(*[p for p in torso(spec)], k=0.03)
    top_y = 1.47 * s
    hem = (0.6 if f else 0.7) * s if not spec.harpin else 0.72 * s
    upper = mat(body.round(t), "Cloth_A").intersect(mat(_slab(0.9 * s, top_y), "Cloth_A"))
    # neckline (cutters carry the garment material so the cut faces keep it)
    upper = upper.subtract(mat(sdf.ellipsoid(_p(spec, 0, 1.475, 0.035), (0.075, 0.06, 0.07)), "Cloth_A", 0.8))
    if not spec.harpin:  # short sleeves over the upper arm
        ua0, ua1 = J["LeftUpperArm"]
        sl_end = ua0 + (ua1 - ua0) * 0.42
        sleeve = mat(sdf.round_cone(ua0 - np.array([0.02, 0, 0]), sl_end, 0.075 * spec.limb, 0.058 * spec.limb), "Cloth_A")
        upper = upper.smooth_union(sleeve.mirror_x(), 0.02)
    else:  # open back for the wings + bare shoulders
        upper = upper.subtract(mat(sdf.box(_p(spec, 0, 1.36, -0.16), (0.24, 0.32 * s, 0.14), 0.03), "Cloth_A", 0.8))
        upper = upper.subtract(mat(sdf.sphere(_p(spec, spec.shoulder + 0.01, 1.44, -0.02), 0.07), "Cloth_A", 0.8).mirror_x())
    # skirt: flared cone around the hips and legs
    hip_r = (0.17 if f else 0.155) + (0.0 if not spec.harpin else 0.05)  # harpins: room for the thigh plumes
    flare = (0.08 if f else 0.05) + (0.05 if spec.harpin else 0.0)
    skirt = mat(sdf.round_cone(_p(spec, 0, 1.0, -0.005), _p(spec, 0, hem / s + 0.02, -0.01), hip_r, hip_r + flare), "Cloth_A")
    skirt = skirt.intersect(mat(_slab(hem, 1.02 * s), "Cloth_A", 0.72))
    # darker hem band and a lighter trim at the neckline are painted later (vertex colors)
    return upper.smooth_union(skirt, 0.03), hem


def belt(spec: BodySpec):
    s = spec.s
    y = 0.985 * s
    hip_r = 0.15 if spec.gender == "m" else 0.158
    band = mat(sdf.ellipsoid(_p(spec, 0, 0.985, -0.005), (hip_r + 0.012, 0.06, 0.112)), "Leather").intersect(mat(_slab(y - 0.026, y + 0.026), "Leather", 0.85))
    buckle = mat(sdf.box(_p(spec, 0, 0.985, 0.112), (0.05, 0.042, 0.016), 0.006), "Metal")
    parts = [band, buckle]
    # pouch on the right hip
    parts.append(mat(sdf.box(_p(spec, -0.12, 0.94, 0.06), (0.07, 0.08, 0.05), 0.018), "Leather", 0.85))
    if spec.harpin:  # strap and a second pouch, like the reference
        parts.append(mat(sdf.box(_p(spec, 0.13, 0.93, 0.02), (0.05, 0.07, 0.05), 0.016), "Leather", 0.8))
    return sdf.union(*parts)


def legwear(spec: BodySpec):
    """Trousers (m) or leggings (f) for humans, from the hips to the boots."""
    J = joints(spec)
    L = spec.limb * (1.06 if spec.gender == "f" else 1.0)
    t = 0.012 if spec.gender == "f" else 0.02
    t0, t1 = J["LeftUpperLeg"]
    c0, c1 = J["LeftLowerLeg"]
    thigh = sdf.round_cone(t0 + np.array([0, 0.05, 0]), t1, 0.09 * L + t, 0.053 * L + t)
    shin = sdf.round_cone(c0, c0 + (c1 - c0) * 0.75, 0.053 * L + t, 0.036 * L + t)
    leg = mat(sdf.smooth_union(thigh, shin, k=0.03), "Cloth_B").mirror_x()
    seat = mat(sdf.ellipsoid(_p(spec, 0, 0.94, -0.02), (0.15 + t, 0.08, 0.1 + t)), "Cloth_B")
    return leg.smooth_union(seat, 0.03).intersect(mat(_slab(0.2 * spec.s, 1.0 * spec.s), "Cloth_B"))


def boots(spec: BodySpec):
    J = joints(spec)
    L = spec.limb
    c0, c1 = J["LeftLowerLeg"]
    foot0 = J["LeftFoot"][0]
    toe1 = J["LeftToes"][1]
    top = c0 + (c1 - c0) * 0.55
    shaft = sdf.round_cone(top, c1, 0.06 * L, 0.046)
    cuff = sdf.round_cone(top + np.array([0, 0.02, 0]), top - np.array([0, 0.04, 0]), 0.068 * L, 0.064 * L)
    foot = sdf.round_cone(foot0 + np.array([0, -0.035, -0.02]), toe1 + np.array([0, 0.015, -0.005]), 0.05, 0.045)
    sole = sdf.box((J["LeftFoot"][0][0], 0.012, (foot0[2] + toe1[2]) / 2), (0.1, 0.024, toe1[2] - foot0[2] + 0.1), 0.01)
    b = sdf.smooth_union(mat(shaft, "Leather"), mat(foot, "Leather"), k=0.03)
    b = b.union(mat(cuff, "Leather", 0.82)).union(mat(sole, "Leather", 0.55))
    return b.mirror_x()


def wraps(spec: BodySpec):
    """Leather wrist wraps for the harpins."""
    J = joints(spec)
    la0, la1 = J["LeftLowerArm"]
    a = la0 + (la1 - la0) * 0.55
    b = la1 - (la1 - la0) * 0.05
    return mat(sdf.round_cone(a, b, 0.05 * spec.limb, 0.04 * spec.limb), "Leather").mirror_x()


def outfit_parts(spec: BodySpec):
    """[(name, sdf, cell)] garments; the tunic's skirt part is skinned with bind_skirt."""
    tun, hem = tunic(spec)
    parts = [("Tunic", tun, 0.014), ("Belt", belt(spec), 0.01)]
    if spec.harpin:
        parts.append(("Wraps", wraps(spec), 0.01))
    else:
        parts += [("Legwear", legwear(spec), 0.015), ("Boots", boots(spec), 0.012)]
    return parts, hem
