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


def a_line(top, hem, r_top, r_hem, jag=0.0, teeth=14):
    """Elliptic A-line frustum (skirt, tabard): sections (rx, rz, z_center) blend linearly from the
    waist (y=top) to the hem (y=hem). `jag` cuts a frayed, notched hem like the Harpin reference."""
    r_top = np.asarray(r_top, float)
    r_hem = np.asarray(r_hem, float)

    def f(p):
        y = p[:, 1]
        t = np.clip((top - y) / (top - hem), 0.0, 1.0)[:, None]
        rx, rz, zc = (r_top + (r_hem - r_top) * t).T
        e = np.sqrt((p[:, 0] / rx) ** 2 + ((p[:, 2] - zc) / rz) ** 2)
        side = (e - 1.0) * np.minimum(rx, rz)
        hem_y = hem
        if jag:
            ang = np.arctan2(p[:, 0], p[:, 2] - zc)
            hem_y = hem + jag * np.abs(np.sin(ang * teeth / 2)) ** 0.6
        return np.maximum(side, np.maximum(hem_y - y, y - top))

    m = float(max(r_hem[0], r_hem[1], r_top[0], r_top[1])) + 0.05
    zc = float(max(abs(r_hem[2]), abs(r_top[2])))
    return sdf.custom(f, (-m, hem - 0.05, -m - zc), (m, top + 0.05, m + zc))


def skirt_shape(spec: BodySpec):
    """(top_y, hem_y, r_top, r_hem) of the tunic skirt for this body: hugs the hips, flares over
    the thighs (harpins: thighs reach forward to the high knee, so the hem shifts forward)."""
    s = spec.s
    f = spec.gender == "f"
    if spec.harpin:
        if f:  # longer, flared
            return 1.0 * s, 0.64 * s, (0.2, 0.15, -0.01), (0.3, 0.23, 0.06)
        return 1.0 * s, 0.7 * s, (0.21, 0.155, -0.01), (0.26, 0.22, 0.06)
    if f:  # knee-length, flared: reads feminine at game scale next to the straight male tunic
        return 1.0 * s, 0.5 * s, (0.192, 0.138, -0.01), (0.3, 0.215, 0.0)
    return 1.0 * s, 0.7 * s, (0.19, 0.135, -0.01), (0.232, 0.172, 0.0)  # straight; clears the thighs


def skirt_radius(spec: BodySpec, y: float):
    """(rx, rz, zc) of the skirt section at height y."""
    top, hem, r0, r1 = skirt_shape(spec)
    t = float(np.clip((top - y) / (top - hem), 0, 1))
    return tuple(np.asarray(r0) + (np.asarray(r1) - np.asarray(r0)) * t)


def tunic(spec: BodySpec):
    """Upper garment + skirt. Returns (sdf, skirt_top_y) — the skirt part is weighted to the thighs."""
    s = spec.s
    f = spec.gender == "f"
    J = joints(spec)
    t = 0.02
    # a wide blend over the full torso: the cloth drapes over the muscles instead of hugging them
    body = sdf.smooth_union(*torso(spec), k=0.07)
    top_y = 1.47 * s
    top, hem, r_top, r_hem = skirt_shape(spec)
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
    # skirt: elliptic A-line from the waist (frayed hem on the harpins, like the reference)
    skirt = mat(a_line(top + 0.03 * s, hem, r_top, r_hem, jag=0.022 if spec.harpin else 0.0), "Cloth_A")
    # darker hem band and a lighter trim at the neckline are painted later (vertex colors)
    return upper.smooth_union(skirt, 0.03), hem


def belt(spec: BodySpec):
    """Leather belt worn over the tunic (sized from the skirt section), buckle and pouches."""
    s = spec.s
    y = 0.99 * s
    rx, rz, zc = skirt_radius(spec, y)
    band = mat(a_line(y + 0.026, y - 0.026, (rx + 0.014, rz + 0.014, zc), (rx + 0.016, rz + 0.016, zc)), "Leather")
    buckle = mat(sdf.box((0, y, zc + rz + 0.016), (0.05, 0.044, 0.018), 0.006), "Metal")
    parts = [band, buckle]
    # pouch hanging on the right hip, outside the skirt
    py = y - 0.07
    prx, _, _ = skirt_radius(spec, py)
    parts.append(mat(sdf.box((-(prx * 0.8 + 0.02), py, zc + rz * 0.55), (0.075, 0.085, 0.05), 0.018).rotate((0, -35, 0), (-(prx * 0.8 + 0.02), py, zc + rz * 0.55)), "Leather", 0.85))
    if spec.harpin:  # a second pouch and a hanging strap, like the reference
        parts.append(mat(sdf.box(((prx * 0.85 + 0.02), py, zc + rz * 0.4), (0.055, 0.075, 0.05), 0.016).rotate((0, 40, 0), ((prx * 0.85 + 0.02), py, zc + rz * 0.4)), "Leather", 0.8))
        parts.append(mat(sdf.box((0.05, y - 0.12, zc + rz + 0.035), (0.026, 0.2, 0.012), 0.004), "Leather", 0.75))
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
    knee = sdf.sphere(c0 + np.array([0, 0, 0.004]), 0.058 * L + t)  # the kneecap must not poke through
    calf = sdf.ellipsoid(c0 + np.array([0, -0.13 * spec.s, -0.03]), (0.048 * L + t, 0.1 * spec.s + t, 0.048 * L + t))  # body calf
    leg = mat(sdf.smooth_union(thigh, shin, knee, calf, k=0.03), "Cloth_B").mirror_x()
    seat = mat(sdf.ellipsoid(_p(spec, 0, 0.94, -0.02), (0.15 + t, 0.08, 0.1 + t)), "Cloth_B")
    # only below the tunic: wider trouser tops would poke through the skirt
    _, hem, _, _ = skirt_shape(spec)
    return leg.smooth_union(seat, 0.03).intersect(mat(_slab(0.2 * spec.s, hem + 0.08 * spec.s), "Cloth_B"))


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
