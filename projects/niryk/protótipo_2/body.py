"""Smooth anime-RPG human body (rest A-pose) as one SDF, male or female: few large forms, wide blends.

Torso, limbs and neck are profile tubes (see shapes.Tube): each is ONE continuous surface whose
silhouette comes from smooth radius curves (deltoid -> biceps -> elbow -> forearm -> wrist), so
there are no blobs to blend. Tubes meet at the shoulders, hips and neck through wide smooth
unions. Male: athletic (broad shoulders, tapered waist); female: slimmer, narrow shoulders, a
defined waist, wider hips and a bust. The head is head.py's. Units: meters, faces +Z, left = +X.
"""
from __future__ import annotations

import numpy as np

from common import (FEMALE, FOREARM, GRIP_OFFSET, GRIP_RADIUS, HIP, KNEE_Y, SHOULDER, UPPER_ARM, arm_rot,
                    mirror_side, slot)
from head import HEAD_PIVOT, HEAD_SCALE, head_scaled, head_xf, neck  # noqa: F401  (re-exported)
from shapes import Tube

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
    if FEMALE:
        w = {T(0.83): 0.122, T(0.88): 0.152, T(0.95): 0.166, T(1.01): 0.156, T(1.07): 0.124, T(1.11): 0.116,
             T(1.17): 0.124, T(1.25): 0.142, T(1.33): 0.152, T(1.39): 0.152, T(1.44): 0.13, T(1.48): 0.1, T(1.5): 0.07}
        f = {T(0.83): 0.068, T(0.9): 0.088, T(0.97): 0.09, T(1.06): 0.08, T(1.12): 0.08, T(1.2): 0.09, T(1.3): 0.098,
             T(1.38): 0.094, T(1.44): 0.08, T(1.5): 0.046}
        k = {T(0.83): 0.084, T(0.9): 0.116, T(0.95): 0.12, T(1.02): 0.1, T(1.09): 0.08, T(1.18): 0.086,
             T(1.28): 0.092, T(1.38): 0.096, T(1.45): 0.088, T(1.5): 0.058}
        n = {0.0: 2.2, T(1.09): 2.1, T(1.3): 2.4, T(1.42): 2.3, 1.0: 2.2}
    else:
        w = {T(0.83): 0.12, T(0.88): 0.14, T(0.95): 0.15, T(1.02): 0.146, T(1.09): 0.134, T(1.17): 0.15,
             T(1.27): 0.174, T(1.35): 0.188, T(1.41): 0.182, T(1.465): 0.138, T(1.5): 0.085}
        f = {T(0.83): 0.07, T(0.9): 0.092, T(0.97): 0.095, T(1.09): 0.094, T(1.18): 0.106, T(1.29): 0.124,
             T(1.37): 0.12, T(1.43): 0.098, T(1.5): 0.055}
        k = {T(0.83): 0.08, T(0.9): 0.108, T(0.95): 0.113, T(1.03): 0.098, T(1.1): 0.088, T(1.2): 0.094,
             T(1.3): 0.102, T(1.4): 0.108, T(1.46): 0.1, T(1.5): 0.065}
        n = {0.0: 2.2, T(1.09): 2.2, T(1.3): 2.6, T(1.42): 2.45, 1.0: 2.2}
    cf = {0.0: 0.0, T(1.3): 0.0, T(1.45): -0.012, 1.0: -0.02}
    body = Tube((0, y0, -0.005), (0, y1, -0.005), w, f, k=k, n=n, cf=cf, cap=0.05)
    if FEMALE:  # bust: two soft volumes blended into the chest
        bust = sdf.ellipsoid((0.048, 1.262, 0.06), (0.054, 0.05, 0.044)).mirror_x()
        body = body.smooth_union(bust, 0.04)
    return skin(body)


def shoulders():
    """Shoulder girdle: a bar across the top of the chest that carries the trapezius slope out to
    the deltoids (front +Z; the bar's 'width' is its vertical half-height)."""
    a = mirror_side(SHOULDER, -1) + np.array([-0.012, 0.0, 0.0])
    b = SHOULDER + np.array([0.012, 0.0, 0.0])
    g = 0.86 if FEMALE else 1.0
    t = {0.0: 0.056 * g, 0.2: 0.064 * g, 0.5: 0.072 * g, 0.8: 0.064 * g, 1.0: 0.056 * g}
    cl = {0.0: -0.006, 0.25: -0.02, 0.5: -0.028, 0.75: -0.02, 1.0: -0.006}  # l = -Y: negative raises
    return skin(Tube(a, b, t, {0.0: 0.066 * g, 0.5: 0.074 * g, 1.0: 0.066 * g},
                     k={0.0: 0.07 * g, 0.5: 0.084 * g, 1.0: 0.07 * g}, cl=cl, cap=0.045))


# ----------------------------------------------------------------------------- arms & hands

def arm_tube():
    """Canonical left arm (T-pose frame): shoulder joint at the origin, along +X, front +Z.
    The tube's lateral axis l = X x Z = -Y, so 'w' is the vertical half-thickness."""
    L0, L1 = -0.04, UPPER_ARM + FOREARM + 0.012
    T = lambda d: (d - L0) / (L1 - L0)  # noqa: E731
    E = UPPER_ARM
    w = {T(-0.04): 0.058, T(0.0): 0.065, T(0.06): 0.067, T(0.13): 0.06, T(0.19): 0.057, T(E - 0.04): 0.05,
         T(E): 0.045, T(E + 0.05): 0.048, T(E + 0.13): 0.039, T(E + FOREARM - 0.03): 0.024, T(L1): 0.021}
    f = {T(-0.04): 0.066, T(0.0): 0.073, T(0.07): 0.068, T(0.13): 0.06, T(0.19): 0.066, T(E - 0.04): 0.054,
         T(E): 0.046, T(E + 0.05): 0.058, T(E + 0.13): 0.05, T(E + FOREARM - 0.03): 0.035, T(L1): 0.032}
    k = {T(-0.04): 0.07, T(0.0): 0.076, T(0.07): 0.07, T(0.13): 0.06, T(0.2): 0.06, T(E - 0.04): 0.053,
         T(E): 0.049, T(E + 0.05): 0.054, T(E + 0.13): 0.046, T(E + FOREARM - 0.03): 0.032, T(L1): 0.03}
    cl = {T(-0.04): 0.0, T(0.03): -0.002, T(0.12): -0.002, T(0.2): 0.0, 1.0: 0.0}
    n = {0.0: 2.0, T(E): 2.0, T(E + FOREARM): 2.3, 1.0: 2.3}
    g = ARM_K
    w, f, k = ({t: v * g for t, v in d.items()} for d in (w, f, k))
    return Tube((L0, 0, 0), (L1, 0, 0), w, f, k=k, cl=cl, n=n, cap=0.02)


ARM_K = 0.84 if FEMALE else 0.94  # limb thickness relative to the old heroic male


WX = UPPER_ARM + FOREARM  # wrist, canonical left arm frame

# Stylized sturdy hand (canonical left frame: fingers along +X from the wrist, palm down (-Y),
# thumb toward +Z). Thick tapering fingers with blunt round tips; the middle, ring and little
# fingers overlap and blend into one grooved block, the index stands a little apart, the thumb
# is separate. Knuckle heads are small bumps on the back of the hand (big ones in the fist).
FINGERS = [  # (z at the knuckle, knuckle x from the wrist, length, base radius, tip radius)
    (0.0300, 0.094, 0.075, 0.0132, 0.0110),  # index
    (0.0098, 0.098, 0.082, 0.0136, 0.0112),  # middle
    (-0.0094, 0.095, 0.077, 0.0130, 0.0106),  # ring
    (-0.0270, 0.088, 0.062, 0.0118, 0.0096),  # little
]
PHALANX = (0.46, 0.31, 0.23)  # share of the finger length
# anime hands are slimmer than the old sturdy ones: finger/palm thickness, hand width and length
HAND_K, HAND_W, HAND_L = (0.8, 0.88, 0.93) if FEMALE else (0.88, 0.95, 1.0)
FINGERS = [(z * HAND_W, kx * HAND_L, L * HAND_L, r0 * HAND_K, r1 * HAND_K) for z, kx, L, r0, r1 in FINGERS]


def finger_shape(pts, r0, r1):
    """Tapered finger through [knuckle, PIP, DIP, tip]; joints a touch thicker than the bones."""
    rs = [r0, r0 * 0.94 + r1 * 0.06, r0 * 0.5 + r1 * 0.5, r1]
    return sdf.chain(pts, rs, k=0.003)


def finger_chain(base, length, curls, spread=0.0):
    """[knuckle, PIP, DIP, tip] of a finger bending down (-Y) by `curls` (deg per joint);
    `spread` leans it sideways (toward +Z)."""
    pts = [np.asarray(base, float)]
    ang = 0.0
    for share, c in zip(PHALANX, curls):
        ang += np.radians(c)
        d = np.array([np.cos(ang), -np.sin(ang), spread])
        pts.append(pts[-1] + length * share * d / np.linalg.norm(d))
    return pts


def _block(fingers):
    """Index apart (shallow groove), middle + ring + little blended into one grooved block."""
    group = sdf.smooth_union(*fingers[1:], k=0.0085)
    return fingers[0].smooth_union(group, 0.0055)


def palm():
    """Broad boxy palm (rounded square section), thickest at the heel, plus the thumb ball
    (thenar) and the pad on the little finger side."""
    K, Wd, Ln = HAND_K, HAND_W, HAND_L
    p = Tube((WX - 0.014, 0.0, 0.0), (WX + 0.095 * Ln, -0.002, 0.0), {0: 0.021 * K, 0.45: 0.0205 * K, 1: 0.0175 * K},
             {0: 0.035 * Wd, 0.45: 0.046 * Wd, 1: 0.047 * Wd}, k={0: 0.034 * Wd, 0.45: 0.044 * Wd, 1: 0.045 * Wd},
             n=3.4, cl={0: 0.0, 0.6: 0.001, 1: 0.0}, cap=0.015 * K)
    thenar = sdf.ellipsoid((WX + 0.03 * Ln, -0.013 * K, 0.027 * Wd), (0.033 * Ln, 0.017 * K, 0.021 * Wd))
    pad = sdf.ellipsoid((WX + 0.04 * Ln, -0.011 * K, -0.03 * Wd), (0.036 * Ln, 0.013 * K, 0.016 * Wd))
    return p.smooth_union(thenar, 0.014).smooth_union(pad, 0.012)


def knuckles(fist=False):
    """Knuckle heads on the back of the hand, at the finger roots."""
    out = []
    for z, kx, _, r0, _ in FINGERS:
        if fist:
            out.append(sdf.sphere((WX + kx - 0.001, 0.0035, z * 0.96), r0 * 1.02))
        else:
            out.append(sdf.ellipsoid((WX + kx - 0.006, 0.0075, z), (r0 * 0.9, r0 * 0.62, r0 * 0.78)))
    return sdf.union(*out)


def hand_relaxed():
    """Canonical left hand at rest: fingers gently curled (more toward the little finger), the
    thumb relaxed forward and down beside the index."""
    fingers = []
    curls = [(10, 18, 12), (15, 24, 15), (18, 27, 16), (21, 30, 18)]
    for (z, kx, L, r0, r1), c in zip(FINGERS, curls):
        base = np.array([WX + kx, -0.002, z])
        fingers.append(finger_shape(finger_chain(base, L, c, spread=0.04 * np.sign(z) * 0.5), r0, r1))
    S = np.array([HAND_L, HAND_K, HAND_W])
    t0 = np.array([WX, 0, 0]) + np.array([0.016, -0.01, 0.028]) * S
    t1 = t0 + np.array([0.032, -0.015, 0.016]) * S
    t2 = t1 + np.array([0.028, -0.004, 0.0]) * HAND_L
    t3 = t2 + np.array([0.022, 0.001, -0.005]) * HAND_L
    thumb = sdf.chain([t0, t1, t2, t3], [r * HAND_K for r in (0.0195, 0.0148, 0.0131, 0.0114)], k=0.005)
    hand = palm().smooth_union(knuckles(), 0.007).smooth_union(_block(fingers), 0.01)
    return hand.smooth_union(thumb, 0.009)


def _wrap(base, length, center, radius, theta0=12.0):
    """[knuckle, PIP, DIP, tip] of a finger that goes from its knuckle to the grip and wraps
    around it (circle of `radius` about the grip axis through `center`, clockwise in the X-Y
    plane: down the front, under, and back toward the palm)."""
    c = np.asarray(center, float)
    th = np.radians(np.arange(theta0, -260.0, -2.0))
    arc = np.stack([c[0] + radius * np.cos(th), c[1] + radius * np.sin(th), np.full_like(th, base[2])], axis=1)
    path = np.vstack([np.asarray(base, float)[None], arc])
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = [path[0]]
    acc = 0.0
    for share in PHALANX:
        acc += share * length
        out.append(np.array([np.interp(acc, cum, path[:, i]) for i in range(3)]))
    return out


def hand_fist():
    """Canonical left fist closed around a sword grip whose axis runs along Z through
    common.GRIP_OFFSET (radius GRIP_RADIUS): the fingers go straight down in front of the grip
    and wrap under it; the thumb wraps the other way, under the grip, and rests against the
    index finger. Big knuckles on top."""
    gx, gy = WX + GRIP_OFFSET[0], GRIP_OFFSET[1]
    fingers = []
    for z, kx, L, r0, r1 in FINGERS:
        base = np.array([WX + kx - 0.001, -0.001, z * 0.96])
        pts = _wrap(base, L, (gx, gy), GRIP_RADIUS + r0 * 0.82)
        fingers.append(finger_shape(pts, r0 * 1.02, r1))
    R = GRIP_RADIUS + 0.0105 * HAND_K
    ring = lambda deg, z: np.array([gx + R * np.cos(np.radians(deg)), gy + R * np.sin(np.radians(deg)), z])  # noqa: E731
    t0 = np.array([WX + 0.016 * HAND_L, -0.012 * HAND_K, 0.029 * HAND_W])
    t1 = np.array([WX + 0.043 * HAND_L, -0.03, 0.043 * HAND_W])
    thumb = sdf.chain([t0, t1, ring(-140, 0.042 * HAND_W), ring(-88, 0.037 * HAND_W)],
                      [r * HAND_K for r in (0.0195, 0.0148, 0.0131, 0.0114)], k=0.005)
    hand = palm().smooth_union(knuckles(True), 0.006).smooth_union(_block(fingers), 0.009)
    return hand.smooth_union(thumb, 0.008)


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
    if FEMALE:  # full thighs high up, slim knees, calves and ankles
        w = {T(1.0): 0.094, T(0.9): 0.096, T(0.78): 0.08, T(0.64): 0.064, T(KNEE_Y + 0.02): 0.05, T(0.45): 0.049,
             T(0.37): 0.051, T(0.24): 0.038, T(0.12): 0.029, T(0.055): 0.029}
        f = {T(1.0): 0.09, T(0.9): 0.09, T(0.78): 0.08, T(0.64): 0.064, T(KNEE_Y + 0.02): 0.052, T(0.45): 0.046,
             T(0.37): 0.043, T(0.24): 0.035, T(0.12): 0.03, T(0.055): 0.031}
        k = {T(1.0): 0.104, T(0.9): 0.09, T(0.78): 0.076, T(0.64): 0.06, T(KNEE_Y + 0.02): 0.046, T(0.45): 0.054,
             T(0.37): 0.062, T(0.24): 0.044, T(0.12): 0.031, T(0.055): 0.034}
        cl = {T(1.0): -0.008 * side, T(0.86): -0.006 * side, T(0.7): 0.0, 1.0: 0.0}
    else:
        w = {T(1.0): 0.086, T(0.9): 0.09, T(0.78): 0.083, T(0.64): 0.069, T(KNEE_Y + 0.02): 0.056, T(0.45): 0.055,
             T(0.37): 0.058, T(0.24): 0.044, T(0.12): 0.033, T(0.055): 0.033}
        f = {T(1.0): 0.089, T(0.9): 0.091, T(0.78): 0.085, T(0.64): 0.069, T(KNEE_Y + 0.02): 0.058, T(0.45): 0.051,
             T(0.37): 0.048, T(0.24): 0.04, T(0.12): 0.034, T(0.055): 0.035}
        k = {T(1.0): 0.097, T(0.9): 0.089, T(0.78): 0.079, T(0.64): 0.063, T(KNEE_Y + 0.02): 0.05, T(0.45): 0.06,
             T(0.37): 0.069, T(0.24): 0.05, T(0.12): 0.035, T(0.055): 0.039}
        cl = {T(1.0): -0.004 * side, T(0.86): -0.005 * side, T(0.7): 0.0, 1.0: 0.0}
    cf = {0.0: 0.0, T(KNEE_Y): 0.008, T(0.3): -0.004, T(0.12): -0.012, 1.0: -0.016}
    return skin(Tube((x, y0, 0.0), (x, y1, -0.02), w, f, k=k, cl=cl, cf=cf, cap=0.03))


def foot(side=1):
    x = HIP[0] * side
    g = 0.88 if FEMALE else 1.0
    ft = Tube((x, 0.04, -0.07), (x + 0.012 * side, 0.032, -0.07 + 0.27 * g),
              {0: 0.034 * g, 0.3: 0.04 * g, 0.75: 0.044 * g, 1: 0.036 * g},
              {0: 0.04, 0.35: 0.05, 0.75: 0.022, 1: 0.016}, k={0: 0.04, 1: 0.026}, n=2.4,
              front=(0, 1, 0), cap=0.02)
    return skin(ft.intersect(sdf.plane((0, 0.0, 0), (0, -1, 0), extent=1.0)))


# ----------------------------------------------------------------------------- assembly

def body_sdf():
    """The whole body (rest A-pose): left hand relaxed, right hand a fist around the sword grip."""
    core = sdf.smooth_union(torso(), shoulders(), k=0.05).smooth_union(neck(), 0.04)
    arms = arm(1).union(arm(-1, fist=True))
    core = core.smooth_union(arms, 0.035)
    legs = leg_tube(1).smooth_union(foot(1), 0.03).union(leg_tube(-1).smooth_union(foot(-1), 0.03))
    core = core.smooth_union(legs, 0.045)
    return core.smooth_union(head_scaled(), 0.014)


def hand_center(side=1):
    """Rest-pose center of a hand (for mesh sizing fields)."""
    from common import arm_point
    return arm_point(UPPER_ARM + FOREARM + 0.07, side, up=-0.02)
