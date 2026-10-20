"""Starter outfit: tunic (Cloth_A), belt + bracers + boots (Leather), trousers (Cloth_B).

Every garment is a smooth offset shell of the body forms it covers (shapes.Offset), cut by planes
with rounded (smooth-intersected) hems, so edges are clean thick rolls and the cloth never shows
the body's small shapes. The tunic's skirt is an A-line tube enclosing both thighs (weighted like a
skirt). Garments are solid: the body faces they cover are deleted at build time.
"""
from __future__ import annotations

import numpy as np

import body as B
from common import FOREARM, SHOULDER, UPPER_ARM, arm_dir, arm_point, mirror_side, slot
from shapes import Offset, Tube

from pyram3d import math3d as m3
from pyram3d import sdf

TUNIC_HEM = 0.765
BELT_Y = (0.985, 1.04)
BOOT_TOP = 0.375
SLEEVE = 0.15  # sleeve length from the shoulder joint, along the arm


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _along_arm(p):
    """(distance along the nearest rest arm from its shoulder joint, side)."""
    side = np.where(p[:, 0] >= 0, 1, -1)
    out = np.empty(len(p))
    for s in (1, -1):
        m = side == s
        out[m] = (p[m] - mirror_side(SHOULDER, s)) @ arm_dir(s)
    return out, side


def _cut(shape, point, normal, k=0.006):
    """Keep the side of the plane opposite to `normal`, with a rounded edge."""
    return shape.smooth_intersect(sdf.plane(point, m3.normalize(m3.vec3(normal)), extent=2.0), k)


def _arms_upper():
    return sdf.union(B._arm_frame(B.arm_tube(), 1), B._arm_frame(B.arm_tube(), -1))


def tunic():
    upper_base = sdf.smooth_union(B.torso(), B.shoulders(), k=0.05).smooth_union(_arms_upper(), 0.035)

    def t_upper(p):
        d, _ = _along_arm(p)
        return 0.013 + 0.009 * _smoothstep(0.03, SLEEVE, d)  # sleeves flare a little

    upper = Offset(upper_base, t_upper, t_max=0.022)
    # A-line skirt from the waist to mid-thigh, over both legs
    y0, y1 = 1.06, TUNIC_HEM
    T = lambda y: (y0 - y) / (y0 - y1)  # noqa: E731
    skirt = Tube((0, y0, -0.006), (0, y1, -0.012), {T(1.06): 0.152, T(0.98): 0.172, T(0.9): 0.192, T(y1): 0.214},
                 {T(1.06): 0.112, T(0.97): 0.116, T(y1): 0.14}, k={T(1.06): 0.106, T(0.95): 0.128, T(y1): 0.15},
                 n=2.3, cap=0.012)
    shape = _cut(upper, (0, 0.99, 0), (0, -1, 0), 0.02).smooth_union(skirt, 0.03)
    # sleeves end at SLEEVE along each arm (the cut is limited to a cylinder around the arm, so it
    # never reaches the torso); round neckline
    for s in (1, -1):
        beyond = sdf.plane(arm_point(SLEEVE, s), -arm_dir(s), extent=1.0)
        cyl = sdf.capsule(arm_point(SLEEVE - 0.02, s), arm_point(UPPER_ARM + FOREARM + 0.3, s), 0.11)
        shape = shape.smooth_subtract(beyond.intersect(cyl), 0.006)
    neck = sdf.ellipsoid((0, 1.53, 0.012), (0.079, 0.075, 0.088))
    shape = shape.smooth_subtract(neck, 0.006)
    return shape.paint(color=(1, 1, 1), material=slot("Cloth_A"))


def belt():
    """Leather belt over the tunic waist, with a metal buckle."""
    base = sdf.smooth_union(B.torso(), sdf.ellipsoid((0, 1.0, -0.01), (0.16, 0.08, 0.125)), k=0.04)
    band = Offset(base, 0.028)
    band = _cut(_cut(band, (0, BELT_Y[1], 0), (0, 1, 0), 0.006), (0, BELT_Y[0], 0), (0, -1, 0), 0.006)
    band = band.paint(color=(1, 1, 1), material=slot("Leather"))
    yc = 0.5 * (BELT_Y[0] + BELT_Y[1])
    z = float(_front_z(base, yc)) + 0.028
    buckle = sdf.box((0, yc, z + 0.004), (0.05, 0.064, 0.014), radius=0.005)
    buckle = buckle.smooth_subtract(sdf.box((0, yc, z + 0.012), (0.026, 0.038, 0.02), radius=0.003), 0.002)
    buckle = buckle.paint(color=(1, 1, 1), material=slot("Metal"))
    return band.union(buckle)


def _front_z(shape, y, x=0.0):
    """z of the front surface of `shape` at (x, y) (bisection along +Z)."""
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if shape.dist(np.array([[x, y, mid]]))[0] < 0:
            lo = mid
        else:
            hi = mid
    return lo


def trousers():
    legs = sdf.union(B.leg_tube(1), B.leg_tube(-1))
    hips = sdf.smooth_union(B.torso(), legs, k=0.045)
    shape = Offset(hips, 0.01)
    shape = _cut(shape, (0, 1.02, 0), (0, 1, 0), 0.01)
    shape = _cut(shape, (0, BOOT_TOP - 0.06, 0), (0, -1, 0), 0.006)
    return shape.paint(color=(1, 1, 1), material=slot("Cloth_B"))


def boots():
    parts = []
    for s in (1, -1):
        leg = B.leg_tube(s).smooth_union(B.foot(s), 0.03)

        def t_boot(p):
            return 0.012 + 0.008 * _smoothstep(0.2, BOOT_TOP, p[:, 1])  # wider shaft at the top

        shaft = Offset(leg, t_boot, t_max=0.02)
        shaft = _cut(shaft, (0, BOOT_TOP, 0), (0, 1, 0), 0.008)
        shaft = shaft.smooth_intersect(sdf.plane((0, 0.0, 0), (0, -1, 0), extent=2.0), 0.006)
        # fold-over cuff at the top
        x = B.HIP[0] * s
        cuff = Tube((x, BOOT_TOP + 0.012, -0.006), (x, BOOT_TOP - 0.045, -0.008), {0: 0.06, 1: 0.058},
                    {0: 0.06, 1: 0.058}, k={0: 0.064, 1: 0.062}, cap=0.01)
        parts.append(shaft.smooth_union(cuff, 0.004))
    return sdf.union(*parts).paint(color=(1, 1, 1), material=slot("Leather"))


def bracers():
    parts = []
    for s in (1, -1):
        a = 0.4
        b = 0.97
        p0 = arm_point(UPPER_ARM + FOREARM * a, s)
        p1 = arm_point(UPPER_ARM + FOREARM * b, s)
        ax = arm_dir(s)
        shape = Offset(B._arm_frame(B.arm_tube(), s), 0.012)
        shape = _cut(_cut(shape, p1, ax, 0.005), p0, -ax, 0.005)
        parts.append(shape)
    return sdf.union(*parts).paint(color=(1, 1, 1), material=slot("Leather"))


GARMENTS = [  # (name, builder, cell scale, triangle target)
    ("Tunic", tunic, 1.0, 3800),
    ("Trousers", trousers, 1.2, 1800),
    ("Boots", boots, 1.0, 1800),
    ("Belt", belt, 0.8, 900),
    ("Bracers", bracers, 0.9, 900),
]
