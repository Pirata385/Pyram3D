"""Smooth heroic human male body (rest A-pose) as one SDF: few large forms, wide blends.

Torso, limbs and neck are profile tubes (see shapes.Tube): each is ONE continuous surface whose
silhouette comes from smooth radius curves (deltoid -> biceps -> elbow -> forearm -> wrist), so
there are no blobs to blend. Tubes meet at the shoulders, hips and neck through wide smooth
unions. The head is built from a cranium, an angular jaw wedge (rounded edges), a brow bar, a
wedge nose and cheek planes. Units: meters, faces +Z, left = +X.
"""
from __future__ import annotations

import numpy as np

from common import (ANKLE_Y, FOREARM, HAND, HIP, KNEE_Y, SHOULDER, UPPER_ARM, arm_rot, mirror_side, slot)
from shapes import Tube, wedge

from pyram3d import math3d as m3
from pyram3d import sdf

WHITE = (1.0, 1.0, 1.0)


def skin(shape, shade=1.0):
    g = float(np.clip(shade, 0, 1))
    return shape.paint(color=(g, g, g), material=slot("Skin"))


def _arm_frame(shape, side=1):
    """Place a shape modeled in the canonical left T-pose arm frame (shoulder joint at the
    origin, arm along +X, front +Z) into the rest pose."""
    q = arm_rot(1)
    M = m3.translation_matrix(SHOULDER) @ m3.rotation_matrix(q)
    out = sdf.Transformed(shape, M)
    return out if side == 1 else out.mirror_x()


# ----------------------------------------------------------------------------- torso & neck

def torso():
    y0, y1 = 0.83, 1.5
    T = lambda y: (y - y0) / (y1 - y0)  # noqa: E731
    w = {T(0.83): 0.12, T(0.88): 0.14, T(0.95): 0.152, T(1.02): 0.15, T(1.09): 0.138, T(1.17): 0.154,
         T(1.27): 0.18, T(1.35): 0.194, T(1.41): 0.186, T(1.465): 0.14, T(1.5): 0.085}
    f = {T(0.83): 0.07, T(0.9): 0.094, T(0.97): 0.098, T(1.09): 0.098, T(1.18): 0.11, T(1.29): 0.13,
         T(1.37): 0.124, T(1.43): 0.1, T(1.5): 0.055}
    k = {T(0.83): 0.08, T(0.9): 0.11, T(0.95): 0.116, T(1.03): 0.1, T(1.1): 0.09, T(1.2): 0.096,
         T(1.3): 0.104, T(1.4): 0.11, T(1.46): 0.102, T(1.5): 0.065}
    n = {0.0: 2.2, T(1.09): 2.2, T(1.3): 2.7, T(1.42): 2.5, 1.0: 2.2}
    cf = {0.0: 0.0, T(1.3): 0.0, T(1.45): -0.012, 1.0: -0.02}
    return skin(Tube((0, y0, -0.005), (0, y1, -0.005), w, f, k=k, n=n, cf=cf, cap=0.05))


def shoulders():
    """Shoulder girdle: a bar across the top of the chest that carries the trapezius slope out to
    the deltoids (front +Z; the bar's 'width' is its vertical half-height)."""
    a = mirror_side(SHOULDER, -1) + np.array([0.02, 0.0, 0.0])
    b = SHOULDER + np.array([-0.02, 0.0, 0.0])
    t = {0.0: 0.05, 0.2: 0.064, 0.5: 0.072, 0.8: 0.064, 1.0: 0.05}
    cl = {0.0: 0.004, 0.25: -0.012, 0.5: -0.026, 0.75: -0.012, 1.0: 0.004}  # l = -Y: negative raises
    return skin(Tube(a, b, t, {0.0: 0.066, 0.5: 0.074, 1.0: 0.066}, k={0.0: 0.07, 0.5: 0.084, 1.0: 0.07},
                     cl=cl, cap=0.045))


def neck():
    a = (0, 1.42, -0.03)
    b = (0, 1.6, -0.012)
    w = {0.0: 0.068, 0.3: 0.056, 0.7: 0.053, 1.0: 0.053}
    f = {0.0: 0.058, 0.3: 0.05, 1.0: 0.048}
    k = {0.0: 0.066, 0.3: 0.056, 1.0: 0.056}
    return skin(Tube(a, b, w, f, k=k, cap=0.03))


# ----------------------------------------------------------------------------- arms & hands

def arm_tube():
    """Canonical left arm (T-pose frame): shoulder joint at the origin, along +X, front +Z.
    The tube's lateral axis l = X x Z = -Y, so 'w' is the vertical half-thickness."""
    L0, L1 = -0.04, UPPER_ARM + FOREARM + 0.012
    T = lambda d: (d - L0) / (L1 - L0)  # noqa: E731
    E = UPPER_ARM
    w = {T(-0.04): 0.066, T(0.0): 0.077, T(0.06): 0.073, T(0.13): 0.06, T(0.19): 0.057, T(E - 0.04): 0.05,
         T(E): 0.045, T(E + 0.05): 0.048, T(E + 0.13): 0.039, T(E + FOREARM - 0.03): 0.026, T(L1): 0.024}
    f = {T(-0.04): 0.066, T(0.0): 0.073, T(0.07): 0.068, T(0.13): 0.06, T(0.19): 0.066, T(E - 0.04): 0.054,
         T(E): 0.046, T(E + 0.05): 0.058, T(E + 0.13): 0.05, T(E + FOREARM - 0.03): 0.038, T(L1): 0.036}
    k = {T(-0.04): 0.07, T(0.0): 0.076, T(0.07): 0.07, T(0.13): 0.06, T(0.2): 0.06, T(E - 0.04): 0.053,
         T(E): 0.049, T(E + 0.05): 0.054, T(E + 0.13): 0.046, T(E + FOREARM - 0.03): 0.035, T(L1): 0.033}
    cl = {T(-0.04): -0.006, T(0.03): -0.013, T(0.12): -0.004, T(0.2): 0.0, 1.0: 0.0}
    n = {0.0: 2.0, T(E): 2.0, T(E + FOREARM): 2.3, 1.0: 2.3}
    return Tube((L0, 0, 0), (L1, 0, 0), w, f, k=k, cl=cl, n=n, cap=0.02)


FINGERS = [  # (z at the knuckle, length, radius) for index, middle, ring, little (canonical left)
    (0.029, 0.084, 0.0108), (0.0095, 0.093, 0.0111), (-0.0095, 0.088, 0.0105), (-0.028, 0.071, 0.0096)]


def finger_chain(base, length, r, curls, spread=0.0, fold=0.0):
    """Points of a finger from its knuckle: 3 phalanges bending down (-Y) by `curls` (deg),
    `spread` turns it sideways (toward +Z), `fold` rolls the curl plane toward -Z (fist)."""
    segs = [length * 0.46, length * 0.31, length * 0.23]
    pts = [np.asarray(base, float)]
    ang = 0.0
    for seg, c in zip(segs, curls):
        ang += np.radians(c)
        d = np.array([np.cos(ang), -np.sin(ang), 0.0])
        d = d + np.array([0.0, 0.0, spread]) + np.array([0.0, 0.0, -fold * np.sin(ang)])
        pts.append(pts[-1] + seg * d / np.linalg.norm(d))
    return pts


def palm(wx):
    return Tube((wx - 0.012, 0, 0), (wx + 0.1, -0.003, 0), {0: 0.018, 0.6: 0.019, 1: 0.016},
                {0: 0.037, 0.45: 0.046, 1: 0.044}, k={0: 0.035, 0.45: 0.042, 1: 0.041}, n=3.0, cap=0.014)


def hand_relaxed():
    """Canonical left hand, palm down (-Y), fingers along +X from the wrist, thumb toward +Z.
    Fingers lie together (overlapping a little) so the blend leaves shallow grooves, never
    thin gaps the mesher could tear."""
    wx = UPPER_ARM + FOREARM
    fingers = []
    for i, (z, L, r) in enumerate(FINGERS):
        base = np.array([wx + 0.095 - abs(z) * 0.22, -0.003, z])
        pts = finger_chain(base, L, r, (14 + 2 * i, 24 + 2 * i, 14 + 2 * i), spread=0.03 * np.sign(z) * 0.5)
        fingers.append(sdf.chain(pts, [r, r * 0.96, r * 0.9, r * 0.82], k=0.004))
    t0 = np.array([wx + 0.014, -0.008, 0.026])
    t1 = t0 + np.array([0.034, -0.014, 0.03])
    t2 = t1 + np.array([0.027, -0.012, 0.014])
    t3 = t2 + np.array([0.021, -0.012, 0.004])
    thumb = sdf.chain([t0, t1, t2, t3], [0.018, 0.0135, 0.0118, 0.0102], k=0.006)
    thenar = sdf.ellipsoid((wx + 0.035, -0.012, 0.022), (0.03, 0.014, 0.02))
    fblock = sdf.smooth_union(*fingers, k=0.0065)
    return sdf.smooth_union(palm(wx), thenar, k=0.012).smooth_union(fblock, 0.01).smooth_union(thumb, 0.008)


def hand_fist():
    """Canonical left fist closed around a sword grip whose axis runs along Z through
    common.GRIP_OFFSET (radius ~1.5 cm): fingers straight down in front of the grip, then
    wrapping back under it; the thumb crosses over the index and middle fingers."""
    wx = UPPER_ARM + FOREARM
    curls = [(80, 52, 55), (87, 55, 58), (92, 56, 58), (97, 55, 55)]
    fingers = []
    for i, ((z, L, r), c) in enumerate(zip(FINGERS, curls)):
        base = np.array([wx + 0.1 - abs(z) * 0.2, -0.004, z * 0.95])
        pts = finger_chain(base, L, r, c)
        fingers.append(sdf.chain(pts, [r * 1.04, r, r * 0.93, r * 0.85], k=0.004))
    t0 = np.array([wx + 0.016, -0.012, 0.028])
    pts = [t0, np.array([wx + 0.046, -0.034, 0.042]), np.array([wx + 0.07, -0.052, 0.037]),
           np.array([wx + 0.088, -0.06, 0.022])]
    thumb = sdf.chain(pts, [0.0185, 0.0138, 0.012, 0.0102], k=0.006)
    thenar = sdf.ellipsoid((wx + 0.034, -0.014, 0.024), (0.03, 0.016, 0.02))
    fblock = sdf.smooth_union(*fingers, k=0.0065)
    return sdf.smooth_union(palm(wx), thenar, k=0.012).smooth_union(fblock, 0.01).smooth_union(thumb, 0.008)


def arm(side=1, fist=False):
    shape = arm_tube().smooth_union(hand_fist() if fist else hand_relaxed(), 0.012)
    return skin(_arm_frame(shape, side))


# ----------------------------------------------------------------------------- legs

def leg_tube(side=1):
    """Hip -> ankle along -Y (front +Z; l = -Y x Z = -X, so 'w' is the half-width in X and a
    negative 'cl' pushes the section outward on the left leg)."""
    x = HIP[0] * side
    y0, y1 = 1.0, 0.055
    T = lambda y: (y0 - y) / (y0 - y1)  # noqa: E731
    w = {T(1.0): 0.096, T(0.9): 0.099, T(0.78): 0.088, T(0.64): 0.072, T(KNEE_Y + 0.02): 0.058, T(0.45): 0.057,
         T(0.37): 0.06, T(0.24): 0.046, T(0.12): 0.035, T(0.055): 0.035}
    f = {T(1.0): 0.092, T(0.9): 0.094, T(0.78): 0.088, T(0.64): 0.072, T(KNEE_Y + 0.02): 0.06, T(0.45): 0.053,
         T(0.37): 0.05, T(0.24): 0.042, T(0.12): 0.036, T(0.055): 0.037}
    k = {T(1.0): 0.1, T(0.9): 0.092, T(0.78): 0.082, T(0.64): 0.066, T(KNEE_Y + 0.02): 0.052, T(0.45): 0.062,
         T(0.37): 0.072, T(0.24): 0.052, T(0.12): 0.037, T(0.055): 0.041}
    cl = {T(1.0): -0.008 * side, T(0.86): -0.008 * side, T(0.7): 0.0, 1.0: 0.0}
    cf = {0.0: 0.0, T(KNEE_Y): 0.008, T(0.3): -0.004, T(0.12): -0.012, 1.0: -0.016}
    return skin(Tube((x, y0, 0.0), (x, y1, -0.02), w, f, k=k, cl=cl, cf=cf, cap=0.03))


def foot(side=1):
    x = HIP[0] * side
    ft = Tube((x, 0.04, -0.07), (x + 0.012 * side, 0.032, 0.2), {0: 0.034, 0.3: 0.04, 0.75: 0.044, 1: 0.036},
              {0: 0.04, 0.35: 0.05, 0.75: 0.022, 1: 0.016}, k={0: 0.04, 1: 0.026}, n=2.4,
              front=(0, 1, 0), cap=0.02)
    return skin(ft.intersect(sdf.plane((0, 0.0, 0), (0, -1, 0), extent=1.0)))


# ----------------------------------------------------------------------------- head

EYE_C = np.array([0.034, 1.66, 0.064])  # eyeball center (left); radius EYE_R
EYE_R = 0.0145
MOUTH_Y = 1.583


def face_mass():
    """Face and jaw as one tube along Y (chin -> forehead): the width curve draws the front
    silhouette (cheekbones, jaw angles, square chin), the center offset and front/back depths draw
    the profile (the jaw underside rises from the chin back to the jaw angles); a boxier section
    low down makes the angular jaw, on smooth surfaces. The axis runs at z = -0.012."""
    y0, y1 = 1.53, 1.73
    T = lambda y: (y - y0) / (y1 - y0)  # noqa: E731
    w = {T(1.53): 0.024, T(1.545): 0.038, T(1.562): 0.055, T(1.582): 0.066, T(1.605): 0.07, T(1.635): 0.077,
         T(1.665): 0.078, T(1.7): 0.079, T(1.73): 0.076}
    cf = {T(1.53): 0.078, T(1.545): 0.07, T(1.562): 0.05, T(1.582): 0.032, T(1.605): 0.02, T(1.64): 0.01,
          T(1.73): 0.0}
    f = {T(1.53): 0.014, T(1.545): 0.026, T(1.562): 0.05, T(1.582): 0.073, T(1.605): 0.088, T(1.64): 0.087,
         T(1.68): 0.093, T(1.73): 0.085}
    k = {T(1.53): 0.012, T(1.545): 0.02, T(1.562): 0.05, T(1.582): 0.074, T(1.605): 0.086, T(1.64): 0.088,
         T(1.73): 0.07}
    n = {T(1.53): 3.0, T(1.562): 2.9, T(1.605): 2.6, T(1.66): 2.3, 1.0: 2.0}
    return Tube((0, y0, -0.012), (0, y1, -0.012), w, f, k=k, n=n, cf=cf, cap=0.012)


def head():
    cran = sdf.ellipsoid((0, 1.688, -0.016), (0.082, 0.097, 0.102))
    base = face_mass().smooth_union(cran, 0.03)
    # brow: two bars from the glabella to the temples
    brow = []
    for sx in (1, -1):
        brow.append(Tube((0, 1.681, 0.08), (sx * 0.066, 1.69, 0.056), {0: 0.0115, 0.6: 0.012, 1: 0.007},
                         {0: 0.0145, 0.5: 0.014, 1: 0.008}, k=0.02, front=(0, 0.2, 1), cap=0.007))
    base = base.smooth_union(sdf.union(*brow), 0.014)
    # nose: bridge -> tip, small wings
    nose = Tube((0, 1.672, 0.08), (0, 1.609, 0.121), {0: 0.008, 0.5: 0.0088, 0.85: 0.0112, 1: 0.0118},
                {0: 0.005, 0.5: 0.0075, 1: 0.0095}, k=0.02, front=(0, 0.6, 1), cap=0.0095)
    alae = sdf.sphere((0.0125, 1.609, 0.106), 0.0082).mirror_x()
    base = base.smooth_union(nose.smooth_union(alae, 0.006), 0.01)
    # eye sockets (the eyeballs sit in them) and upper lids over the eyeballs
    base = base.smooth_subtract(sdf.sphere(EYE_C + [0, 0.0, 0.014], 0.0165).mirror_x(), 0.006)
    lid = sdf.sphere(EYE_C, EYE_R + 0.0022).intersect(
        sdf.plane(EYE_C + [0, 0.0035, 0], (0, -1, -0.25), extent=0.05))
    base = base.smooth_union(lid.mirror_x(), 0.003)
    # mouth: a soft line (lips are painted)
    base = base.smooth_subtract(sdf.capsule((-0.019, MOUTH_Y, 0.1015), (0.019, MOUTH_Y, 0.1015), 0.002), 0.004)
    # ears
    ear = sdf.ellipsoid((0.077, 1.632, -0.02), (0.013, 0.03, 0.019))
    ear = ear.smooth_subtract(sdf.ellipsoid((0.088, 1.628, -0.016), (0.008, 0.018, 0.011)), 0.004)
    base = base.smooth_union(ear.mirror_x(), 0.008)
    return skin(base)


HEAD_SCALE = 1.06
HEAD_PIVOT = np.array([0.0, 1.53, 0.04])


def head_xf(p):
    """Rest position of a point modeled in head() space after the head scaling."""
    return HEAD_PIVOT + (np.asarray(p, float) - HEAD_PIVOT) * HEAD_SCALE


def head_scaled():
    """The head is modeled at 1.0 and scaled up about the chin (6.5-head heroic proportion)."""
    M = m3.translation_matrix(HEAD_PIVOT) @ m3.scale_matrix(HEAD_SCALE) @ m3.translation_matrix(-HEAD_PIVOT)
    return sdf.Transformed(head(), M)


# ----------------------------------------------------------------------------- assembly

def body_sdf():
    """The whole body (rest A-pose): left hand relaxed, right hand a fist around the sword grip."""
    core = sdf.smooth_union(torso(), shoulders(), k=0.05).smooth_union(neck(), 0.04)
    arms = arm(1).union(arm(-1, fist=True))
    core = core.smooth_union(arms, 0.035)
    legs = leg_tube(1).smooth_union(foot(1), 0.03).union(leg_tube(-1).smooth_union(foot(-1), 0.03))
    core = core.smooth_union(legs, 0.045)
    return core.smooth_union(head_scaled(), 0.025)


def upper_body_test():
    """Torso + neck + left arm only (approach test)."""
    core = sdf.smooth_union(torso(), shoulders(), k=0.05).smooth_union(neck(), 0.04)
    return core.smooth_union(arm(1), 0.035)


def head_test():
    return head().smooth_union(skin(neck()), 0.03)


def hand_center(side=1):
    """Rest-pose center of a hand (for mesh sizing fields)."""
    from common import arm_point
    return arm_point(UPPER_ARM + FOREARM + 0.07, side, up=-0.02)
