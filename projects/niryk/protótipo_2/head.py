"""Anime-RPG head (Genshin Impact / Tales of line) as one smooth SDF, male or female.

A big round cranium carries a small face: the face is a profile tube along Y (chin -> forehead)
whose width curve draws the front silhouette (narrow soft chin, smooth cheeks) and whose front /
back curves draw the profile (the jaw line rising from the chin to the ears). The eyes sit in
shallow dishes (the eye meshes are decals that follow the skin, see face.py); the nose is a small
bridge with a pointed tip; there are no lips (the mouth is a line). Ears are simple.

Modeled at scale 1 in "head space" (chin ~ y 1.53) and scaled about HEAD_PIVOT by HEAD_SCALE.
Units: meters, faces +Z, left = +X. Male: a little sharper jaw, smaller eyes; female: rounder
face, bigger eyes.
"""
from __future__ import annotations

import numpy as np

from common import FEMALE, HARPIN, slot
from shapes import Tube

from pyram3d import math3d as m3
from pyram3d import sdf

# ----------------------------------------------------------------------------- parameters
if FEMALE:
    P = dict(
        cran_c=(0.0, 1.676, -0.016), cran_r=(0.089, 0.104, 0.1),
        # face tube keys: y -> (half width, front z, back z, superellipse n)
        face={1.538: (0.004, 0.06, 0.05, 2.3), 1.546: (0.017, 0.068, 0.036, 2.3), 1.557: (0.031, 0.073, 0.014, 2.2),
              1.571: (0.044, 0.076, -0.014, 2.15), 1.587: (0.055, 0.078, -0.04, 2.1), 1.605: (0.063, 0.079, -0.054, 2.05),
              1.625: (0.069, 0.08, -0.06, 2.0), 1.65: (0.072, 0.079, -0.064, 2.0), 1.7: (0.074, 0.074, -0.064, 2.0)},
        eye_c=(0.0335, 1.646), eye_dish=(0.025, 0.022, 0.012), eye_depth=0.002,
        nose_tip=(1.6, 0.0875), nose_r=(0.0046, 0.0038, 0.0046), nose_root=(1.634, 0.081),
        ear=((0.081, 1.624, -0.022), (0.0105, 0.021, 0.014)),
        neck=dict(w=(0.044, 0.038, 0.037), f=(0.042, 0.037, 0.036), k=(0.05, 0.043, 0.042)),
        head_scale=1.07,
    )
else:
    P = dict(
        cran_c=(0.0, 1.674, -0.016), cran_r=(0.09, 0.106, 0.102),
        face={1.537: (0.007, 0.061, 0.052, 2.5), 1.545: (0.02, 0.07, 0.036, 2.5), 1.556: (0.031, 0.075, 0.012, 2.4),
              1.57: (0.045, 0.078, -0.016, 2.3), 1.586: (0.056, 0.08, -0.044, 2.2), 1.604: (0.064, 0.081, -0.058, 2.1),
              1.625: (0.07, 0.082, -0.064, 2.05), 1.65: (0.074, 0.081, -0.066, 2.0), 1.7: (0.077, 0.076, -0.066, 2.0)},
        eye_c=(0.0345, 1.647), eye_dish=(0.025, 0.02, 0.012), eye_depth=0.002,
        nose_tip=(1.599, 0.0905), nose_r=(0.005, 0.0042, 0.005), nose_root=(1.636, 0.0825),
        ear=((0.083, 1.624, -0.024), (0.011, 0.023, 0.015)),
        neck=dict(w=(0.054, 0.045, 0.044), f=(0.05, 0.043, 0.042), k=(0.058, 0.05, 0.05)),
        head_scale=1.03,
    )

# the lower face is drawn long above and compressed toward the eye line (short anime lower face)
P["compress"] = 0.84 if FEMALE else 0.87


def cy(y):
    """Head-space height of a lower-face feature authored at y (compressed toward the eyes)."""
    e = P["eye_c"][1]
    y = np.asarray(y, float)
    return np.where(y < e, e - (e - y) * P["compress"], y) if y.ndim else (e - (e - float(y)) * P["compress"] if y < e else float(y))


P["face"] = {float(cy(y)): v for y, v in P["face"].items()}
P["nose_tip"] = (float(cy(P["nose_tip"][0])), P["nose_tip"][1])
P["nose_root"] = (float(cy(P["nose_root"][0])), P["nose_root"][1])
P["ear"] = ((P["ear"][0][0], float(cy(P["ear"][0][1])), P["ear"][0][2]), P["ear"][1])

CRAN_C = np.array(P["cran_c"])
CRAN_R = np.array(P["cran_r"])
HEAD_SCALE = P["head_scale"]
HEAD_PIVOT = np.array([0.0, 1.53, 0.04])
EYE_X, EYE_Y = P["eye_c"]


def head_xf(p):
    """Rest (design-space) position of a point modeled in head space."""
    return HEAD_PIVOT + (np.asarray(p, float) - HEAD_PIVOT) * HEAD_SCALE


def head_matrix():
    return m3.translation_matrix(HEAD_PIVOT) @ m3.scale_matrix(HEAD_SCALE) @ m3.translation_matrix(-HEAD_PIVOT)


def face_front_z(y):
    """z of the face tube's front at height y (head space, center line)."""
    ys = sorted(P["face"])
    return float(np.interp(y, ys, [P["face"][k][1] for k in ys]))


def face_mass():
    keys = P["face"]
    ys = sorted(keys)
    y0, y1 = ys[0], ys[-1]
    T = lambda y: (y - y0) / (y1 - y0)  # noqa: E731
    z0 = -0.012
    w = {T(y): keys[y][0] for y in ys}
    cf = {T(y): 0.5 * (keys[y][1] + keys[y][2]) - z0 for y in ys}
    f = {T(y): 0.5 * (keys[y][1] - keys[y][2]) for y in ys}
    n = {T(y): keys[y][3] for y in ys}
    return Tube((0, y0, z0), (0, y1, z0), w, f, n=n, cf=cf, cap=0.004)


def nose():
    """A hint of a bridge from between the eyes down to a small pointed tip."""
    ry, rz = P["nose_root"]
    ty, tz = P["nose_tip"]
    bridge = Tube((0, ry, rz - 0.004), (0, ty + 0.002, tz - 0.003), {0: 0.0036, 1: 0.0044},
                  {0: 0.0015, 0.6: 0.0026, 1: 0.0036}, k=0.012, n=2.3, front=(0, 0.6, 1), cap=0.003)
    tip = sdf.ellipsoid((0, ty, tz - P["nose_r"][2]), P["nose_r"])
    return bridge.smooth_union(tip, 0.006)


def eye_dish(sx=1):
    """Shallow dish the eye decal sits in (left eye; mirrored by the caller)."""
    rx, ry, rz = P["eye_dish"]
    z = face_front_z(EYE_Y) - 0.002 + rz - P["eye_depth"]
    return sdf.ellipsoid((EYE_X * sx, EYE_Y + 0.001, z), (rx, ry, rz))


def head():
    cran = sdf.ellipsoid(CRAN_C, tuple(CRAN_R))
    base = face_mass().smooth_union(cran, 0.035)
    base = base.smooth_union(nose(), 0.006)
    base = base.smooth_subtract(eye_dish(1).union(eye_dish(-1)), 0.012)
    c, r = P["ear"]
    ear = sdf.ellipsoid(c, r)
    if HARPIN:  # pointed ear: a flat leaf sweeping back and up from the ear
        c = np.asarray(c, float)
        tip = c + np.array([0.016, 0.034, -0.03])
        leaf = Tube(c + np.array([-0.002, -0.006, 0.004]), tip, {0: r[1] * 0.75, 0.45: r[1] * 0.62, 1: 0.0012},
                    {0: r[0] * 0.55, 1: 0.0012}, front=(1.0, 0.0, 0.35), cap=0.001)
        ear = ear.smooth_union(leaf, 0.006)
    ear = ear.smooth_subtract(sdf.ellipsoid((c[0] + r[0] * 0.95, c[1] - 0.002, c[2] + 0.003),
                                            (r[0] * 0.6, r[1] * 0.62, r[2] * 0.6)), 0.004)
    base = base.smooth_union(ear.mirror_x(), 0.008)
    return base.paint(color=(1, 1, 1), material=slot("Skin"))


def head_scaled():
    return sdf.Transformed(head(), head_matrix())


def neck():
    a = (0, 1.42, -0.03)
    b = (0, 1.6, -0.016)
    N = P["neck"]
    w = {0.0: N["w"][0], 0.3: N["w"][1], 1.0: N["w"][2]}
    f = {0.0: N["f"][0], 0.3: N["f"][1], 1.0: N["f"][2]}
    k = {0.0: N["k"][0], 0.3: N["k"][1], 1.0: N["k"][2]}
    return Tube(a, b, w, f, k=k, cap=0.03).paint(color=(1, 1, 1), material=slot("Skin"))


# ----------------------------------------------------------------------------- face normals

def face_ellipsoid():
    """(center, radii) in design space of the ellipsoid the face normals bend toward."""
    c = head_xf((0.0, 1.635, -0.008))
    r = np.array([0.083, 0.1, 0.092]) * HEAD_SCALE
    return c, r


def face_normal_weight(P_design):
    """How much a point's normal follows the face ellipsoid (1 on the face, 0 at the ears, the
    back of the head and down the neck)."""
    q = (np.asarray(P_design, float) - HEAD_PIVOT) / HEAD_SCALE + HEAD_PIVOT  # back to head space
    y, z = q[:, 1], q[:, 2]

    def ss(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    front = ss(-0.03, 0.03, z)
    vert = ss(1.505, 1.545, y) * (1 - ss(1.7, 1.74, y))
    return 0.92 * front * vert


def face_normals(P_design, N):
    c, r = face_ellipsoid()
    ne = (np.asarray(P_design, float) - c) / (r * r)
    ne /= np.linalg.norm(ne, axis=1, keepdims=True)
    w = face_normal_weight(P_design)[:, None]
    out = N * (1 - w) + ne * w
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
