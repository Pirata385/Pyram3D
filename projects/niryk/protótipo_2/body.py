"""Smooth heroic human male body (rest A-pose) as one SDF: few large forms, wide blends.

Torso, limbs and neck are profile tubes (see shapes.Tube): each is ONE continuous surface whose
silhouette comes from smooth radius curves (deltoid -> biceps -> elbow -> forearm -> wrist), so
there are no blobs to blend. Tubes meet at the shoulders, hips and neck through wide smooth
unions. The head is built from a cranium, an angular jaw wedge (rounded edges), a brow bar, a
wedge nose and cheek planes. Units: meters, faces +Z, left = +X.
"""
from __future__ import annotations

import numpy as np

from common import (ANKLE_Y, FOREARM, GRIP_OFFSET, GRIP_RADIUS, HIP, KNEE_Y, SHOULDER, UPPER_ARM, arm_rot,
                    mirror_side, slot)
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
    a = mirror_side(SHOULDER, -1) + np.array([-0.012, 0.0, 0.0])
    b = SHOULDER + np.array([0.012, 0.0, 0.0])
    t = {0.0: 0.056, 0.2: 0.064, 0.5: 0.072, 0.8: 0.064, 1.0: 0.056}
    cl = {0.0: -0.006, 0.25: -0.02, 0.5: -0.028, 0.75: -0.02, 1.0: -0.006}  # l = -Y: negative raises
    return skin(Tube(a, b, t, {0.0: 0.066, 0.5: 0.074, 1.0: 0.066}, k={0.0: 0.07, 0.5: 0.084, 1.0: 0.07},
                     cl=cl, cap=0.045))


def neck():
    a = (0, 1.42, -0.03)
    b = (0, 1.6, -0.012)
    w = {0.0: 0.066, 0.3: 0.053, 0.7: 0.05, 1.0: 0.05}
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
    w = {T(-0.04): 0.058, T(0.0): 0.065, T(0.06): 0.067, T(0.13): 0.06, T(0.19): 0.057, T(E - 0.04): 0.05,
         T(E): 0.045, T(E + 0.05): 0.048, T(E + 0.13): 0.039, T(E + FOREARM - 0.03): 0.024, T(L1): 0.021}
    f = {T(-0.04): 0.066, T(0.0): 0.073, T(0.07): 0.068, T(0.13): 0.06, T(0.19): 0.066, T(E - 0.04): 0.054,
         T(E): 0.046, T(E + 0.05): 0.058, T(E + 0.13): 0.05, T(E + FOREARM - 0.03): 0.035, T(L1): 0.032}
    k = {T(-0.04): 0.07, T(0.0): 0.076, T(0.07): 0.07, T(0.13): 0.06, T(0.2): 0.06, T(E - 0.04): 0.053,
         T(E): 0.049, T(E + 0.05): 0.054, T(E + 0.13): 0.046, T(E + FOREARM - 0.03): 0.032, T(L1): 0.03}
    cl = {T(-0.04): 0.0, T(0.03): -0.002, T(0.12): -0.002, T(0.2): 0.0, 1.0: 0.0}
    n = {0.0: 2.0, T(E): 2.0, T(E + FOREARM): 2.3, 1.0: 2.3}
    return Tube((L0, 0, 0), (L1, 0, 0), w, f, k=k, cl=cl, n=n, cap=0.02)


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
    p = Tube((WX - 0.014, 0.0, 0.0), (WX + 0.095, -0.002, 0.0), {0: 0.021, 0.45: 0.0205, 1: 0.0175},
             {0: 0.035, 0.45: 0.046, 1: 0.047}, k={0: 0.034, 0.45: 0.044, 1: 0.045}, n=3.4,
             cl={0: 0.0, 0.6: 0.001, 1: 0.0}, cap=0.015)
    thenar = sdf.ellipsoid((WX + 0.03, -0.013, 0.027), (0.033, 0.017, 0.021))
    pad = sdf.ellipsoid((WX + 0.04, -0.011, -0.03), (0.036, 0.013, 0.016))
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
    t0 = np.array([WX + 0.016, -0.01, 0.028])
    t1 = t0 + np.array([0.032, -0.015, 0.016])
    t2 = t1 + np.array([0.028, -0.004, 0.0])
    t3 = t2 + np.array([0.022, 0.001, -0.005])
    thumb = sdf.chain([t0, t1, t2, t3], [0.0195, 0.0148, 0.0131, 0.0114], k=0.005)
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
    R = GRIP_RADIUS + 0.0105
    ring = lambda deg, z: np.array([gx + R * np.cos(np.radians(deg)), gy + R * np.sin(np.radians(deg)), z])  # noqa: E731
    t0 = np.array([WX + 0.016, -0.012, 0.029])
    t1 = np.array([WX + 0.043, -0.03, 0.043])
    thumb = sdf.chain([t0, t1, ring(-140, 0.042), ring(-88, 0.037)], [0.0195, 0.0148, 0.0131, 0.0114], k=0.005)
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
    w = {T(1.0): 0.089, T(0.9): 0.093, T(0.78): 0.086, T(0.64): 0.072, T(KNEE_Y + 0.02): 0.058, T(0.45): 0.057,
         T(0.37): 0.06, T(0.24): 0.046, T(0.12): 0.035, T(0.055): 0.035}
    f = {T(1.0): 0.092, T(0.9): 0.094, T(0.78): 0.088, T(0.64): 0.072, T(KNEE_Y + 0.02): 0.06, T(0.45): 0.053,
         T(0.37): 0.05, T(0.24): 0.042, T(0.12): 0.036, T(0.055): 0.037}
    k = {T(1.0): 0.1, T(0.9): 0.092, T(0.78): 0.082, T(0.64): 0.066, T(KNEE_Y + 0.02): 0.052, T(0.45): 0.062,
         T(0.37): 0.072, T(0.24): 0.052, T(0.12): 0.037, T(0.055): 0.041}
    cl = {T(1.0): -0.004 * side, T(0.86): -0.005 * side, T(0.7): 0.0, 1.0: 0.0}
    cf = {0.0: 0.0, T(KNEE_Y): 0.008, T(0.3): -0.004, T(0.12): -0.012, 1.0: -0.016}
    return skin(Tube((x, y0, 0.0), (x, y1, -0.02), w, f, k=k, cl=cl, cf=cf, cap=0.03))


def foot(side=1):
    x = HIP[0] * side
    ft = Tube((x, 0.04, -0.07), (x + 0.012 * side, 0.032, 0.2), {0: 0.034, 0.3: 0.04, 0.75: 0.044, 1: 0.036},
              {0: 0.04, 0.35: 0.05, 0.75: 0.022, 1: 0.016}, k={0: 0.04, 1: 0.026}, n=2.4,
              front=(0, 1, 0), cap=0.02)
    return skin(ft.intersect(sdf.plane((0, 0.0, 0), (0, -1, 0), extent=1.0)))


# ----------------------------------------------------------------------------- head

EYE_C = np.array([0.0335, 1.661, 0.0635])  # eyeball center (left); radius EYE_R
EYE_R = 0.015
EYE_GAZE = np.array([0.0, -0.04, 1.0])  # the eyes look straight ahead, a touch down
LID_UP = 0.0066  # upper lid edge above the eye center (at the front of the eyeball)
LID_DN = -0.0078  # lower lid edge
UP_LID_N = (0.0, -1.0, -0.32)  # lid planes: normal (the lid is on the side opposite to it)
DN_LID_N = (0.0, 1.0, -0.2)


def lid_plane(upper=True):
    """(point, normal) of an eyelid's cutting plane (left eye) so that the lid edge crosses the
    front of the eyeball at LID_UP / LID_DN from its center."""
    n = np.array(UP_LID_N if upper else DN_LID_N)
    h = (LID_UP if upper else LID_DN) + n[2] / n[1] * EYE_R
    return EYE_C + np.array([0.0, h, 0.0]), n
MOUTH_Y = 1.5835
MOUTH_Z = 0.0985  # front of the lips' meeting line
MOUTH_W = 0.0235  # half width of the mouth line


def face_mass():
    """Face and jaw as one tube along Y (chin -> forehead): the width curve draws the front
    silhouette (cheekbones, jaw angles, square chin), the center offset and front/back depths draw
    the profile (the jaw underside rises from the chin back to the jaw angles); a boxy section
    low down makes the angular jaw and a squarer one at the cheeks gives the face a front plane
    and side planes, on smooth surfaces. The axis runs at z = -0.012."""
    y0, y1 = 1.53, 1.73
    T = lambda y: (y - y0) / (y1 - y0)  # noqa: E731
    w = {T(1.53): 0.022, T(1.545): 0.036, T(1.562): 0.05, T(1.58): 0.06, T(1.605): 0.066, T(1.635): 0.075,
         T(1.665): 0.077, T(1.7): 0.078, T(1.73): 0.076}
    cf = {T(1.53): 0.077, T(1.545): 0.069, T(1.562): 0.05, T(1.582): 0.032, T(1.605): 0.02, T(1.64): 0.01,
          T(1.73): 0.0}
    f = {T(1.53): 0.016, T(1.545): 0.028, T(1.562): 0.05, T(1.582): 0.072, T(1.605): 0.086, T(1.64): 0.086,
         T(1.68): 0.092, T(1.73): 0.085}
    k = {T(1.53): 0.012, T(1.545): 0.02, T(1.562): 0.05, T(1.582): 0.074, T(1.605): 0.086, T(1.64): 0.088,
         T(1.73): 0.07}
    n = {T(1.53): 3.4, T(1.562): 3.3, T(1.605): 3.0, T(1.645): 2.8, T(1.69): 2.3, 1.0: 2.0}
    return Tube((0, y0, -0.012), (0, y1, -0.012), w, f, k=k, n=n, cf=cf, cap=0.012)


def nose():
    """Straight bridge from the brow to a rounded tip (boxy section: flat sides, a clean ridge),
    small wings and a soft base."""
    bridge = Tube((0, 1.677, 0.082), (0, 1.615, 0.112), {0: 0.0072, 0.45: 0.0078, 0.85: 0.0096, 1: 0.0102},
                  {0: 0.0052, 0.5: 0.0066, 1: 0.0082}, k=0.018, n=2.7, front=(0, 0.5, 1), cap=0.008)
    tip = sdf.ellipsoid((0, 1.6135, 0.1105), (0.0108, 0.0094, 0.0098))
    wing = sdf.ellipsoid((0.0116, 1.6095, 0.1015), (0.008, 0.007, 0.008)).mirror_x()
    return bridge.smooth_union(tip, 0.008).smooth_union(wing, 0.006)


def head():
    cran = sdf.ellipsoid((0, 1.688, -0.016), (0.082, 0.097, 0.102))
    base = face_mass().smooth_union(cran, 0.03)
    # square chin
    base = base.smooth_union(sdf.box((0, 1.5435, 0.073), (0.04, 0.03, 0.032), radius=0.013), 0.012)
    # cheekbones: a clear plane change under the outer eye corners
    cheek = sdf.ellipsoid((0.05, 1.637, 0.054), (0.023, 0.011, 0.018)).mirror_x()
    base = base.smooth_union(cheek, 0.02)
    # brow ridge: two bars from the glabella to the temples, overhanging the eyes
    brow = []
    for sx in (1, -1):
        brow.append(Tube((0, 1.681, 0.083), (sx * 0.067, 1.689, 0.056), {0: 0.0125, 0.55: 0.0128, 1: 0.0075},
                         {0: 0.016, 0.5: 0.0165, 1: 0.009}, k=0.02, n=2.4, front=(0, 0.15, 1), cap=0.008))
    base = base.smooth_union(sdf.union(*brow), 0.017)
    base = base.smooth_union(nose(), 0.009)
    # eye sockets (the eyeballs sit in them), lids around an almond opening
    base = base.smooth_subtract(sdf.ellipsoid(EYE_C + [0.0005, -0.0005, 0.012], (0.0185, 0.0135, 0.0155)).mirror_x(),
                                0.005)
    up = sdf.sphere(EYE_C, EYE_R + 0.0021).intersect(sdf.plane(*lid_plane(True), extent=0.05))
    dn = sdf.sphere(EYE_C, EYE_R + 0.0014).intersect(sdf.plane(*lid_plane(False), extent=0.05))
    base = base.smooth_union(up.mirror_x(), 0.0028).smooth_union(dn.mirror_x(), 0.0035)
    # mouth: lips as soft volumes, a crisp meeting line with the corners turned down a touch,
    # a hollow under the lower lip above the chin
    lips = sdf.ellipsoid((0, MOUTH_Y + 0.0042, MOUTH_Z - 0.0045), (0.021, 0.0048, 0.0075)).smooth_union(
        sdf.ellipsoid((0, MOUTH_Y - 0.0052, MOUTH_Z - 0.0062), (0.0185, 0.0056, 0.0078)), 0.004)
    base = base.smooth_union(lips, 0.006)
    line = sdf.chain([(-MOUTH_W, MOUTH_Y - 0.0018, MOUTH_Z - 0.006), (0, MOUTH_Y, MOUTH_Z + 0.0005),
                      (MOUTH_W, MOUTH_Y - 0.0018, MOUTH_Z - 0.006)], 0.0021, k=0.004)
    base = base.smooth_subtract(line, 0.0028)
    base = base.smooth_subtract(sdf.ellipsoid((0, 1.5685, 0.097), (0.015, 0.0036, 0.007)), 0.006)
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
    return core.smooth_union(head_scaled(), 0.014)


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
