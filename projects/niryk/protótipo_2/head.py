"""Anime-RPG head in the Genshin Impact line, as one smooth SDF, male or female.

A big round cranium carries a small face. The face is a profile tube along Y (chin -> forehead):
its width curve draws the front silhouette (small pointed chin, full soft cheeks), its front curve
draws the profile (chin, a flat mouth area, the nose base, the eye line set a little behind the
brow and the cheeks, the forehead) and its back curve the jaw line rising to the ears. Its
cross-section is a superellipse that is squarer at the eyes (a broad front that reads smooth) and
rounder at the chin. Soft cheek volumes and a small chin are blended in; the eyes sit in very
shallow dishes; the nose is a tiny tip on a faint bridge. There are no lips and no eye geometry:
eyes, brows and mouth are painted textures on decals that follow the skin (face.py).

Female: rounder and younger (fuller cheeks, short lower face). Male: a longer lower face and a
firmer jaw (straighter jaw line, a broader chin), still anime.

Modeled at scale 1 in "head space" (chin ~ y 1.53) and scaled about HEAD_PIVOT by HEAD_SCALE.
Units: meters, faces +Z, left = +X.
"""
from __future__ import annotations

import numpy as np

from common import FEMALE, HARPIN, slot
from shapes import Offset, Tube

from pyram3d import math3d as m3
from pyram3d import sdf

# ----------------------------------------------------------------------------- parameters
# face tube keys: y -> (half width, front z, back z, superellipse n)
if FEMALE:
    P = dict(
        cran_c=(0.0, 1.678, -0.024), cran_r=(0.089, 0.104, 0.1),
        face={1.530: (0.0050, 0.053, 0.045, 2.0), 1.538: (0.0165, 0.0605, 0.033, 2.0),
              1.548: (0.0300, 0.0645, 0.017, 2.05), 1.560: (0.0435, 0.0668, -0.004, 2.15),
              1.575: (0.0560, 0.0690, -0.026, 2.3), 1.592: (0.0660, 0.0712, -0.044, 2.5),
              1.610: (0.0730, 0.0730, -0.056, 2.7), 1.630: (0.0775, 0.0740, -0.063, 2.9),
              1.655: (0.0795, 0.0752, -0.066, 2.8), 1.685: (0.0790, 0.0752, -0.066, 2.5),
              1.720: (0.0700, 0.0660, -0.062, 2.2)},
        blend=0.006,
        # smooth displacement bumps (head space): (x, y), (rx, ry), amplitude (+ out, - in), mirrored
        sculpt=[((0.040, 1.638), (0.030, 0.022), -0.0012, True),   # eye area, very slightly recessed
                ((0.050, 1.594), (0.027, 0.026), 0.0014, True),     # full soft cheeks
                ((0.0, 1.537), (0.012, 0.0095), 0.0009, False),     # small pointed chin
                ((0.0, 1.552), (0.013, 0.0075), -0.0006, False)],   # soft hollow above the chin
        eye_c=(0.040, 1.636),
        nose_tip=(1.590, 0.0802), nose_r=(0.0042, 0.0036, 0.0046), nose_root=(1.628, 0.0740),
        ear=((0.081, 1.624, -0.022), (0.0105, 0.021, 0.014)),
        neck=dict(w=(0.044, 0.038, 0.037), f=(0.042, 0.037, 0.036), k=(0.05, 0.043, 0.042)),
        head_scale=1.07,
    )
else:
    P = dict(
        cran_c=(0.0, 1.676, -0.025), cran_r=(0.09, 0.106, 0.102),
        face={1.525: (0.0105, 0.055, 0.046, 2.5), 1.533: (0.0215, 0.0625, 0.031, 2.5),
              1.544: (0.0340, 0.0665, 0.013, 2.5), 1.557: (0.0470, 0.0695, -0.009, 2.55),
              1.572: (0.0580, 0.0715, -0.031, 2.6), 1.590: (0.0670, 0.0735, -0.047, 2.7),
              1.610: (0.0735, 0.0750, -0.058, 2.8), 1.632: (0.0785, 0.0758, -0.064, 2.9),
              1.655: (0.0810, 0.0770, -0.067, 2.8), 1.685: (0.0805, 0.0772, -0.068, 2.5),
              1.720: (0.0720, 0.0680, -0.064, 2.2)},
        blend=0.006,
        sculpt=[((0.040, 1.637), (0.030, 0.021), -0.0012, True),
                ((0.052, 1.588), (0.026, 0.026), 0.0007, True),
                ((0.0, 1.531), (0.016, 0.009), 0.0008, False),
                ((0.0, 1.546), (0.015, 0.007), -0.0005, False)],
        eye_c=(0.040, 1.635),
        nose_tip=(1.588, 0.0835), nose_r=(0.0047, 0.004, 0.005), nose_root=(1.628, 0.0758),
        ear=((0.083, 1.624, -0.024), (0.011, 0.023, 0.015)),
        neck=dict(w=(0.054, 0.045, 0.044), f=(0.05, 0.043, 0.042), k=(0.058, 0.05, 0.05)),
        head_scale=1.03,
    )

CRAN_C = np.array(P["cran_c"])
CRAN_R = np.array(P["cran_r"])
HEAD_SCALE = P["head_scale"]
HEAD_PIVOT = np.array([0.0, 1.53, 0.04])
EYE_X, EYE_Y = P["eye_c"]
CHIN_Y = min(P["face"])


def head_xf(p):
    """Rest (design-space) position of a point modeled in head space."""
    return HEAD_PIVOT + (np.asarray(p, float) - HEAD_PIVOT) * HEAD_SCALE


def to_head(p):
    """Design-space point(s) -> head space."""
    return HEAD_PIVOT + (np.asarray(p, float) - HEAD_PIVOT) / HEAD_SCALE


def head_matrix():
    return m3.translation_matrix(HEAD_PIVOT) @ m3.scale_matrix(HEAD_SCALE) @ m3.translation_matrix(-HEAD_PIVOT)


def face_front_z(y):
    """z of the face tube's front at height y (head space, center line)."""
    ys = sorted(P["face"])
    return float(np.interp(y, ys, [P["face"][k][1] for k in ys]))


def face_half_width(y):
    ys = sorted(P["face"])
    return np.interp(y, ys, [P["face"][k][0] for k in ys])


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
    """A faint bridge from between the eyes down to a tiny pointed tip."""
    ry, rz = P["nose_root"]
    ty, tz = P["nose_tip"]
    bridge = Tube((0, ry, rz - 0.003), (0, ty + 0.002, tz - 0.003), {0: 0.0034, 1: 0.004},
                  {0: 0.001, 0.6: 0.0022, 1: 0.0032}, k=0.01, n=2.2, front=(0, 0.6, 1), cap=0.003)
    tip = sdf.ellipsoid((0, ty, tz - P["nose_r"][2]), P["nose_r"])
    return bridge.smooth_union(tip, 0.006)


def sculpt_bumps(p):
    """Smooth displacement (meters, + outward) of the face: the very slightly recessed eye area,
    full cheeks, the small chin. Bumps are smooth (1 - r^2)^2 falloffs in (x, y), on the front only."""
    p = np.atleast_2d(p)
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    t = np.clip((z - 0.0) / 0.04, 0, 1)
    front = t * t * (3 - 2 * t)
    out = np.zeros(len(p))
    for (cx, cy_), (rx, ry), amp, sym in P["sculpt"]:
        for sx in ((1, -1) if sym else (1,)):
            r2 = ((x - sx * cx) / rx) ** 2 + ((y - cy_) / ry) ** 2
            out += amp * np.clip(1 - r2, 0, 1) ** 2
    return out * front


def ears():
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
    return ear.mirror_x()


def head_base():
    """The head without the ears and the nose (smooth proxy for normals and the face map)."""
    cran = sdf.ellipsoid(CRAN_C, tuple(CRAN_R))
    return Offset(face_mass().smooth_union(cran, P["blend"]), sculpt_bumps, t_max=0.003)


def head():
    base = head_base().smooth_union(nose(), 0.006)
    base = base.smooth_union(ears(), 0.008)
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
    q = to_head(P_design)
    y, z = q[:, 1], q[:, 2]

    def ss(e0, e1, x):
        t = np.clip((x - e0) / (e1 - e0), 0, 1)
        return t * t * (3 - 2 * t)
    front = ss(-0.03, 0.03, z)
    vert = ss(CHIN_Y - 0.025, CHIN_Y + 0.015, y) * (1 - ss(1.7, 1.74, y))
    return 0.6 * front * vert


def face_normals(P_design, N):
    c, r = face_ellipsoid()
    ne = (np.asarray(P_design, float) - c) / (r * r)
    ne /= np.linalg.norm(ne, axis=1, keepdims=True)
    w = face_normal_weight(P_design)[:, None]
    out = N * (1 - w) + ne * w
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


# ----------------------------------------------------------------------------- Head_Skin UVs
# One UV space for the head and neck skin (Head_Skin's base color and its face shadow threshold
# map): u is the azimuth around a vertical axis through UV_C (u = 0.5 on the face's center line,
# the character's left side at u > 0.5, the seam down the back of the neck), warped so the face
# front gets about half of the width; v is the elevation seen from UV_C (the head-neck junction),
# so the chin, the underside of the jaw and the neck never fold over each other. Mesh points are
# first projected onto a smooth proxy of the head (no ears, no nose, the notch under the jaw filled),
# which is star-shaped from UV_C: every (u, v) is one point of the proxy and the textures are painted
# from those points. Symmetric: u(-x) = 1 - u(x).
UV_C = np.array([0.0, 1.57, -0.015])
UV_WARP = 1.2
UV_EL = (-88.0, 88.0)
_PROXY = None


def uv_proxy():
    """Smooth head + neck in head space (the UV and face map reference surface)."""
    global _PROXY
    if _PROXY is None:
        neck_h = sdf.Transformed(neck(), np.linalg.inv(head_matrix()))
        _PROXY = head_base().smooth_union(neck_h, 0.04)
    return _PROXY


def _warp(t):
    a = np.abs(t)
    return np.sign(t) * a * (1 + UV_WARP) / (1 + UV_WARP * a)


def _unwarp(w):
    a = np.abs(w)
    return np.sign(w) * a / (1 + UV_WARP - UV_WARP * a)


def project_to_proxy(q, iters=8):
    """Head-space points -> the nearest points on the UV proxy (gradient steps)."""
    pr = uv_proxy()
    q = np.array(q, float)
    for _ in range(iters):
        d = pr.dist(q)
        if np.abs(d).max() < 1e-6:
            break
        q = q - pr.gradient(q, eps=1e-4) * d[:, None]
    return q


def uv_of_head_point(q):
    """Head-space points ON the proxy -> (u, v) (pyram3d convention: v up)."""
    d = np.asarray(q, float) - UV_C
    az = np.arctan2(d[:, 0], d[:, 2])
    el = np.degrees(np.arctan2(d[:, 1], np.hypot(d[:, 0], d[:, 2])))
    u = 0.5 + 0.5 * _warp(az / np.pi)
    v = (el - UV_EL[0]) / (UV_EL[1] - UV_EL[0])
    return np.column_stack([u, v])


def head_uv(P_design):
    """Per-vertex UVs of Head_Skin points (design space)."""
    return uv_of_head_point(project_to_proxy(to_head(P_design)))


def head_point_of_uv(u, v, iters=36):
    """(u, v) arrays -> head-space points on the proxy (ray from UV_C), and the azimuth (rad)."""
    u, v = np.broadcast_arrays(np.asarray(u, float), np.asarray(v, float))
    az = np.pi * _unwarp(2 * u.ravel() - 1)
    el = np.radians(UV_EL[0] + v.ravel() * (UV_EL[1] - UV_EL[0]))
    dirs = np.column_stack([np.cos(el) * np.sin(az), np.sin(el), np.cos(el) * np.cos(az)])
    pr = uv_proxy()
    lo = np.zeros(len(dirs))
    hi = np.full(len(dirs), 0.3)
    for _ in range(iters):  # bisection: inside at lo, outside at hi
        mid = 0.5 * (lo + hi)
        inside = pr.dist(UV_C + dirs * mid[:, None]) < 0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return (UV_C + dirs * lo[:, None]).reshape(u.shape + (3,)), az.reshape(u.shape)
