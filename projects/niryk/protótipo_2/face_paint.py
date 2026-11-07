"""Painted anime face (Genshin Impact line): the 2D design of the eyes, brows and mouth, the
procedural textures of the face decals, Head_Skin's base color and its face shadow threshold map.

Everything is painted with numpy as anti-aliased signed-distance shapes and gradients:

  eye white  (EyeWhite, RGB)  white, a cool shadow under the upper lid, darker corners
  iris       (Eye, gray)      dark top -> light bottom, a darker rim, a lighter ring, faint radial
                              streaks, the pupil, the lid's shadow (the client tints it with the eye color)
  eye shine  (EyeShine, RGBA) two or three white highlights (alpha mask)
  lashes     (Lash, RGBA)     the thick upper lash line that flares at the outer corner (and lashes
                              for the female), the eyelid crease, the lower lash hint (alpha mask)
  brows      (Brow, RGBA)     a thin tapered brow, gray (tinted by the hair color; alpha mask)
  mouth      (Mouth, RGBA)    a tiny mouth line (alpha mask)
  skin       (Skin, RGB)      near white with a soft blush and a hint under the nose tip
  face map   (Skin occlusion, linear R): the face shadow threshold map, see face_threshold()

The eye, brow and mouth shapes are designed in 2D frames in head space (meters): eyes in (u, v)
around the eye center, u toward the outer corner (the right eye mirrors the left), v up; the brows
in the same frame as the eyes; the mouth around (0, MOUTH_Y). The decal meshes (face.py) carry UVs
that map those frames onto the textures.
"""
from __future__ import annotations

import math

import numpy as np

import head as H
from common import FEMALE

MM = 1e-3


def ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _bump(t, peak):
    """sin(pi * t^k): 0 at t = 0 and 1, maximum 1 at t = peak."""
    k = math.log(0.5) / math.log(peak)
    return np.sin(np.pi * np.clip(t, 0, 1) ** k)


# ----------------------------------------------------------------------------- 2D design (mm)
if FEMALE:
    EYE = dict(u0=-20.5, u1=24.0, v_in=-2.0, v_out=4.8, h_up=21.0, p_up=0.42, h_dn=-16.6, p_dn=0.5,
               iris_c=(1.2, 1.0), iris_r=(13.6, 19.6), pupil_c=(1.2, 2.4), pupil_r=(5.2, 8.6),
               lash=(1.3, 3.7, 5.4), flick=(9.5, 6.6), spikes=True, crease=(6.2, 0.75), lower=1.15,
               shine=[((7.0, 8.8), (4.3, 5.9), 18.0), ((-6.0, -10.6), (2.3, 2.1), 0.0), ((9.0, -4.8), (1.2, 1.2), 0.0)])
    BROW = dict(u=(-19.5, 29.0), v=(28.5, 34.0, 30.0), th=(1.9, 0.55), peak=0.48)
    MOUTH = dict(y=1.562, w=5.2, curve=0.6, th=1.0)
else:
    EYE = dict(u0=-20.5, u1=23.5, v_in=-1.0, v_out=4.0, h_up=15.0, p_up=0.4, h_dn=-12.0, p_dn=0.56,
               iris_c=(1.0, 1.2), iris_r=(12.0, 15.0), pupil_c=(1.0, 2.0), pupil_r=(4.6, 7.0),
               lash=(1.4, 3.3, 4.6), flick=(6.2, 3.4), spikes=False, crease=(5.0, 0.7), lower=0.95,
               shine=[((5.8, 6.4), (3.6, 4.4), 18.0), ((-5.0, -7.6), (1.9, 1.6), 0.0)])
    BROW = dict(u=(-20.0, 29.5), v=(22.0, 26.0, 24.0), th=(3.4, 1.1), peak=0.42)
    MOUTH = dict(y=1.557, w=6.4, curve=0.15, th=1.05)

# texture frames (meters): (u0, u1, v0, v1)
EYE_FRAME = (-0.030, 0.036, -0.028, 0.038)  # eyes: both eyes share it, the right one mirrored
BROW_FRAME = (-0.025, 0.035, 0.014, 0.044)
MOUTH_FRAME = (-0.012, 0.012, -0.006, 0.006)
EYE_TEX, BROW_TEX, MOUTH_TEX, SKIN_TEX, MAP_TEX = 1024, (1024, 512), (1024, 512), 2048, 1024


def lid_up(u):
    """Upper lid (mm): inner corner -> a high arc a little inward of the center -> outer corner."""
    E = EYE
    t = (np.asarray(u, float) - E["u0"]) / (E["u1"] - E["u0"])
    return E["v_in"] + (E["v_out"] - E["v_in"]) * t + E["h_up"] * _bump(t, E["p_up"])


def lid_dn(u):
    E = EYE
    t = (np.asarray(u, float) - E["u0"]) / (E["u1"] - E["u0"])
    return E["v_in"] + (E["v_out"] - E["v_in"]) * t + E["h_dn"] * _bump(t, E["p_dn"])


# ----------------------------------------------------------------------------- 2D geometry

def convex_hull(P):
    """Counter-clockwise convex hull of 2D points (monotone chain)."""
    P = sorted(map(tuple, np.asarray(P, float)))

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in P:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(P):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def clip_convex(poly, clip):
    """Sutherland-Hodgman: polygon (N, k) whose first two columns are 2D coordinates (the rest are
    interpolated attributes) clipped by a convex CCW polygon (M, 2)."""
    out = np.asarray(poly, float)
    C = np.asarray(clip, float)
    for i in range(len(C)):
        if len(out) == 0:
            break
        a, b = C[i], C[(i + 1) % len(C)]
        e = b - a
        side = e[0] * (out[:, 1] - a[1]) - e[1] * (out[:, 0] - a[0])  # > 0 inside (left of a->b)
        res = []
        n = len(out)
        for j in range(n):
            p, q = out[j], out[(j + 1) % n]
            sp, sq = side[j], side[(j + 1) % n]
            if sp >= 0:
                res.append(p)
            if (sp >= 0) != (sq >= 0):
                t = sp / (sp - sq)
                res.append(p + (q - p) * t)
        out = np.array(res) if res else np.zeros((0, out.shape[1]))
    return out


def eye_outline(n=72):
    """The eye opening (meters, eye frame), convex CCW."""
    E = EYE
    u = np.linspace(E["u0"], E["u1"], n)
    pts = np.concatenate([np.column_stack([u, lid_dn(u)]), np.column_stack([u[::-1], lid_up(u[::-1])])])
    return convex_hull(pts * MM)


def ellipse_poly(c, r, n=96, rot=0.0):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    p = np.column_stack([r[0] * np.cos(t), r[1] * np.sin(t)])
    a = math.radians(rot)
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    return p @ R.T + np.asarray(c, float)


def iris_outline():
    """The iris cut by the eye opening (meters, eye frame), convex CCW."""
    E = EYE
    return clip_convex(ellipse_poly(np.array(E["iris_c"]) * MM, np.array(E["iris_r"]) * MM), eye_outline())


# ----------------------------------------------------------------------------- canvas

class Canvas:
    """Pixel centers of a texture covering frame (u0, u1, v0, v1) (row 0 = v1, the top)."""

    def __init__(self, frame, size):
        w, h = (size, size) if isinstance(size, int) else size
        u0, u1, v0, v1 = frame
        self.w, self.h = w, h
        self.U, self.V = np.meshgrid(u0 + (np.arange(w) + 0.5) / w * (u1 - u0),
                                     v1 - (np.arange(h) + 0.5) / h * (v1 - v0))
        self.px = (u1 - u0) / w  # pixel size (meters)

    def mm(self):
        return self.U / MM, self.V / MM


def sd_ellipse(U, V, c, r, rot=0.0):
    """Approximate signed distance to an ellipse (same units as the inputs)."""
    x, y = U - c[0], V - c[1]
    if rot:
        a = math.radians(rot)
        x, y = x * math.cos(a) + y * math.sin(a), -x * math.sin(a) + y * math.cos(a)
    k0 = np.sqrt((x / r[0]) ** 2 + (y / r[1]) ** 2)
    k1 = np.sqrt((x / r[0] ** 2) ** 2 + (y / r[1] ** 2) ** 2)
    return k0 * (k0 - 1) / np.maximum(k1, 1e-12)


def sd_stroke(U, V, pts, half):
    """Signed distance to a polyline stroke whose half thickness varies linearly along it."""
    pts = np.asarray(pts, float)
    half = np.asarray(half, float)
    d = np.full(U.shape, np.inf)
    lo = pts.min(axis=0) - half.max() - 1.0
    hi = pts.max(axis=0) + half.max() + 1.0
    box = (U > lo[0]) & (U < hi[0]) & (V > lo[1]) & (V < hi[1])
    if not box.any():
        return d
    u, v = U[box], V[box]
    best = np.full(u.shape, np.inf)
    for i in range(len(pts) - 1):
        a, b = pts[i], pts[i + 1]
        e = b - a
        L2 = max(float(e @ e), 1e-18)
        t = np.clip(((u - a[0]) * e[0] + (v - a[1]) * e[1]) / L2, 0, 1)
        dist = np.hypot(u - a[0] - t * e[0], v - a[1] - t * e[1]) - (half[i] + t * (half[i + 1] - half[i]))
        best = np.minimum(best, dist)
    d[box] = best
    return d


def cover(d, px):
    """Anti-aliased coverage of a shape from its signed distance (px = pixel size)."""
    return np.clip(0.5 - d / px, 0, 1)


def mask_alpha(d, px, ramp=1.6):
    """Alpha for an alpha-tested (MASK, cutoff 0.5) texture: a ramp of a few pixels across the edge,
    so the bilinear-filtered 0.5 contour is smooth when magnified."""
    return np.clip(0.5 - d / (ramp * px), 0, 1)


def rgba(rgb, alpha):
    out = np.concatenate([np.broadcast_to(np.asarray(rgb, float), alpha.shape + (3,)), alpha[..., None]], axis=-1)
    return np.clip(out, 0, 1)


# ----------------------------------------------------------------------------- eyes

def paint_eye_white():
    c = Canvas(EYE_FRAME, EYE_TEX)
    u, v = c.mm()
    E = EYE
    base = np.array([0.985, 0.975, 0.975])
    shade = np.array([0.70, 0.69, 0.82])
    below = lid_up(u) - v  # mm under the upper lid
    s = ss(0.0, 5.2, below) ** 0.8
    t = np.abs(u - 0.5 * (E["u0"] + E["u1"])) / (0.5 * (E["u1"] - E["u0"]))
    corner = 1 - 0.1 * ss(0.55, 1.0, t)
    col = (shade[None, None] * (1 - s[..., None]) + base[None, None] * s[..., None]) * corner[..., None]
    # a thin soft line along the lower lid (the white ends softly, no hard edge)
    low = v - lid_dn(u)
    col *= (1 - 0.08 * (1 - ss(0.0, 1.4, low)))[..., None]
    return np.clip(col, 0, 1)


def paint_iris():
    """Grayscale iris (the client multiplies the eye color)."""
    c = Canvas(EYE_FRAME, EYE_TEX)
    u, v = c.mm()
    E = EYE
    (cx, cy), (rx, ry) = E["iris_c"], E["iris_r"]
    x, y = (u - cx) / rx, (v - cy) / ry
    r = np.sqrt(x * x + y * y)
    ang = np.arctan2(y, x)
    g = 0.2 + 0.8 * ss(0.6, -0.8, y) ** 0.85  # dark top -> light bottom
    g += 0.08 * np.exp(-((r - 0.6) / 0.09) ** 2)  # a lighter ring
    g *= 1 + 0.06 * np.sin(ang * 22) * ss(0.25, 0.85, r) * (1 - ss(0.9, 1.0, r))  # faint radial streaks
    g *= 1 - 0.62 * ss(0.78, 1.0, r)  # darker rim
    # the pupil: a darker vertical oval
    pd = sd_ellipse(u, v, E["pupil_c"], E["pupil_r"])
    pc = cover(pd, c.px / MM)
    g = g * (1 - pc) + 0.08 * pc
    # the upper lid's shadow across the top of the iris
    below = lid_up(u) - v
    g *= 1 - 0.6 * (1 - ss(0.0, 5.5, below))
    # a soft bright crescent low in the iris
    g += 0.1 * np.exp(-((y + 0.62) / 0.16) ** 2) * (1 - ss(0.75, 0.95, np.abs(x))) * (1 - pc)
    return np.clip(np.repeat(g[..., None], 3, axis=-1), 0, 1)


def paint_eye_shine():
    c = Canvas(EYE_FRAME, EYE_TEX)
    u, v = c.mm()
    d = np.full(u.shape, np.inf)
    for cc, rr, rot in EYE["shine"]:
        d = np.minimum(d, sd_ellipse(u, v, cc, rr, rot))
    return rgba((1.0, 1.0, 1.0), mask_alpha(d, c.px / MM))


def lash_curves():
    """Polylines (mm) and half thicknesses of the upper lash line, its flick, the outer corner,
    the lashes, the crease and the lower lash hint: [(points, half, color), ...]."""
    E = EYE
    out = []
    lash_c = (0.055, 0.032, 0.042)
    # upper lash line: its lower edge sits ~0.6 mm under the lid line, thickest past the middle,
    # then the flick out and up beyond the outer corner
    n = 40
    u = np.linspace(E["u0"] - 0.8, E["u1"], n)
    t = (u - u[0]) / (u[-1] - u[0])
    th_in, th_mid, th_out = E["lash"]
    th = np.where(t < 0.55, th_in + (th_mid - th_in) * ss(0.0, 0.55, t), th_mid + (th_out - th_mid) * ss(0.55, 1.0, t))
    th = th * np.clip(t / 0.06, 0.25, 1)
    lid = lid_up(u)
    pts = np.column_stack([u, lid + 0.5 * th - 0.6])
    fu, fv = E["flick"]
    k = np.linspace(0, 1, 10)[1:]
    end = pts[-1]
    fl = np.column_stack([end[0] + fu * k, end[1] + fv * k ** 1.25 + 0.3 * k])
    fth = th[-1] * 1.05 * (1 - k) ** 0.75 + 0.05
    pts = np.vstack([pts, fl])
    half = np.concatenate([0.5 * th, 0.5 * fth])
    out.append((pts, half, lash_c))
    # outer corner: the line curls down to meet the lower lid
    a = np.array([E["u1"] - 0.6, float(lid_up(E["u1"])) - 0.2])
    b = np.array([E["u1"] - 2.6, float(lid_dn(E["u1"] - 2.6)) + 0.2])
    m = 0.5 * (a + b) + np.array([0.9, 0.0])
    cur = np.array([a, m, b])
    out.append((cur, np.array([0.75, 0.55, 0.15]), lash_c))
    if E["spikes"]:  # a few lashes at the outer corner
        for tt, ln, w, bend in ((0.8, 3.0, 0.62, 0.25), (0.9, 3.6, 0.7, 0.4), (0.99, 3.2, 0.6, 0.55)):
            i = int(tt * (n - 1))
            base = pts[i] + np.array([0.0, 0.5 * th[i] - 0.3])
            d = np.array([0.35 + bend, 1.0])
            d /= np.linalg.norm(d)
            kk = np.linspace(0, 1, 6)
            sp = base + np.outer(kk, d) * ln + np.outer(kk ** 2, [ln * 0.35, -ln * 0.08])
            out.append((sp, w * (1 - kk) ** 0.9 + 0.04, lash_c))
    # eyelid crease: a thin line above the lid, over the middle and outer part
    off, w = E["crease"]
    u = np.linspace(E["u0"] + 7.5, E["u1"] - 1.5, 24)
    t = (u - u[0]) / (u[-1] - u[0])
    pts = np.column_stack([u, lid_up(u) + off * (0.75 + 0.25 * np.sin(np.pi * t)) + 0.4 * t])
    out.append((pts, w * np.sin(np.pi * np.clip(t, 0.02, 0.98)) ** 0.7, (0.42, 0.25, 0.24)))
    # lower lash hint: a thin line on the outer half of the lower lid
    u = np.linspace(E["u0"] + 0.5 * (E["u1"] - E["u0"]), E["u1"] - 1.2, 18)
    t = (u - u[0]) / (u[-1] - u[0])
    lw = E["lower"] * np.sin(np.pi * np.clip(t * 0.85 + 0.12, 0, 1)) ** 0.6
    pts = np.column_stack([u, lid_dn(u) - 0.25 * lw])
    out.append((pts, 0.5 * lw, (0.28, 0.16, 0.17)))
    return out


def paint_lashes():
    c = Canvas(EYE_FRAME, EYE_TEX)
    u, v = c.mm()
    px = c.px / MM
    col = np.zeros(u.shape + (3,))
    col[:] = (0.055, 0.032, 0.042)
    alpha = np.zeros(u.shape)
    # paint back to front: crease and lower line first, the lash line on top
    for pts, half, rgb in reversed(lash_curves()):
        d = sd_stroke(u, v, pts, half)
        a = mask_alpha(d, px)
        cv = cover(d - 0.6 * px, px)[..., None]
        col = col * (1 - cv) + np.asarray(rgb)[None, None] * cv
        alpha = np.maximum(alpha, a)
    return np.clip(np.concatenate([col, alpha[..., None]], -1), 0, 1)


# ----------------------------------------------------------------------------- brows and mouth

def brow_curve(n=24):
    B = BROW
    t = np.linspace(0, 1, n)
    u = B["u"][0] + (B["u"][1] - B["u"][0]) * t
    v0, vm, v1 = B["v"]
    v = v0 + (v1 - v0) * t + (vm - 0.5 * (v0 + v1)) * _bump(t, B["peak"])
    th = (B["th"][0] + (B["th"][1] - B["th"][0]) * t) * np.clip(t / 0.08, 0.45, 1) * np.clip((1 - t) / 0.18, 0.05, 1) ** 0.7
    return np.column_stack([u, v]), 0.5 * th


def paint_brow():
    c = Canvas(BROW_FRAME, BROW_TEX)
    u, v = c.mm()
    pts, half = brow_curve()
    d = sd_stroke(u, v, pts, half)
    return rgba((0.5, 0.5, 0.5), mask_alpha(d, c.px / MM))


def mouth_curve(n=16):
    M = MOUTH
    t = np.linspace(0, 1, n)
    u = -M["w"] + 2 * M["w"] * t
    v = M["curve"] * (2 * t - 1) ** 2 - 0.15 * M["curve"]
    th = M["th"] * np.sin(np.pi * np.clip(t, 0.03, 0.97)) ** 0.55
    return np.column_stack([u, v]), 0.5 * th


def paint_mouth():
    c = Canvas(MOUTH_FRAME, MOUTH_TEX)
    u, v = c.mm()
    pts, half = mouth_curve()
    d = sd_stroke(u, v, pts, half)
    return rgba((0.46, 0.2, 0.21), mask_alpha(d, c.px / MM))


# ----------------------------------------------------------------------------- Head_Skin textures

def _resize(a, h, w):
    """Bilinear resize of (H, W, k) samples taken at the corners of a grid to an (h, w) pixel grid."""
    H0, W0 = a.shape[:2]
    y = (np.arange(h) + 0.5) / h * (H0 - 1)
    x = (np.arange(w) + 0.5) / w * (W0 - 1)
    y0 = np.clip(np.floor(y).astype(int), 0, H0 - 2)
    x0 = np.clip(np.floor(x).astype(int), 0, W0 - 2)
    fy = (y - y0)[:, None, None]
    fx = (x - x0)[None, :, None]
    a00 = a[y0][:, x0]
    a01 = a[y0][:, x0 + 1]
    a10 = a[y0 + 1][:, x0]
    a11 = a[y0 + 1][:, x0 + 1]
    return (a00 * (1 - fx) + a01 * fx) * (1 - fy) + (a10 * (1 - fx) + a11 * fx) * fy


_SURF = {}


def skin_surface(size, coarse=256):
    """Head-space proxy point and proxy normal of every texel of a size x size Head_Skin texture
    (row 0 = v near 1). Computed on a coarse grid and interpolated (the proxy is smooth)."""
    if size in _SURF:
        return _SURF[size]
    g = np.linspace(0, 1, coarse + 1)
    U, V = np.meshgrid(g, g[::-1])
    Q, _ = H.head_point_of_uv(U, V)
    N = H.uv_proxy().gradient(Q.reshape(-1, 3), eps=5e-4).reshape(Q.shape)
    q = _resize(Q, size, size)
    n = _resize(N, size, size)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    _SURF[size] = (q, n)
    return q, n


def skin_base_color(size=SKIN_TEX):
    """Near white (the client multiplies the skin color) with a soft blush on the cheeks and a warm
    hint under the nose tip."""
    q, n = skin_surface(size)
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    front = ss(0.02, 0.045, z)
    col = np.ones(q.shape[:2] + (3,))
    blush_c = np.array([1.0, 0.8, 0.8]) if FEMALE else np.array([1.0, 0.88, 0.86])
    amt = 0.62 if FEMALE else 0.3
    for sx in (1, -1):
        r2 = ((x - sx * (H.EYE_X + 0.005)) / 0.0145) ** 2 + ((y - (H.EYE_Y - 0.0195)) / 0.0068) ** 2
        k = np.clip(1 - r2, 0, 1) ** 1.6 * front * amt
        col = col * (1 - k[..., None]) + blush_c * k[..., None]
    # the nose: a small angled tick on its right side (male), a small soft curve under the tip
    # (female), as Genshin draws them
    ty, tz = H.P["nose_tip"]
    px = 0.12 * MM
    if FEMALE:
        pts = np.array([[-1.7, -1.5], [-0.5, -2.05], [0.8, -1.75]]) * MM + np.array([0.0, ty])
        d = sd_stroke(x, y, pts, np.array([0.18, 0.34, 0.16]) * MM)
        k = np.clip(0.5 - d / (3 * px), 0, 1) * front * 0.55
        tick_c = np.array([0.82, 0.6, 0.58])
    else:
        pts = np.array([[-1.0, 1.9], [-2.1, 0.2], [-2.9, -1.1]]) * MM + np.array([0.0, ty])
        d = sd_stroke(x, y, pts, np.array([0.12, 0.36, 0.2]) * MM)
        k = np.clip(0.5 - d / (2 * px), 0, 1) * front * 0.8
        tick_c = np.array([0.66, 0.45, 0.43])
    col = col * (1 - k[..., None]) + tick_c * k[..., None]
    return np.clip(col, 0, 1)


def _blur(a, sigma, wrap_u=True):
    """Separable gaussian blur (sigma in pixels); u wraps (the seam), v clamps."""
    if sigma <= 0:
        return a
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = a
    pad_u = np.concatenate([out[:, -r:], out, out[:, :r]], axis=1) if wrap_u else np.pad(out, ((0, 0), (r, r)), "edge")
    out = sum(k[i] * pad_u[:, i:i + a.shape[1]] for i in range(2 * r + 1))
    pad_v = np.pad(out, ((r, r), (0, 0)), "edge")
    return sum(k[i] * pad_v[i:i + a.shape[0]] for i in range(2 * r + 1))


def neck_shadow(size, coarse=256, elev=40.0):
    """1 where the neck is in the head's shadow (see face_threshold), anti-aliased: the neck below
    the jaw and the skull (proxy points off the head's own surface) and the faces turned away from
    a high light (elev degrees up) at the texel's azimuth (the underside of the jaw)."""
    g = np.linspace(0, 1, coarse + 1)
    U, V = np.meshgrid(g, g[::-1])
    Q, az = H.head_point_of_uv(U, V)
    Q = Q.reshape(-1, 3)
    az = az.reshape(-1)
    N = H.uv_proxy().gradient(Q, eps=5e-4)
    e = math.radians(elev)
    D = np.column_stack([np.sin(az) * math.cos(e), np.full(len(az), math.sin(e)), np.cos(az) * math.cos(e)])
    away = np.einsum("ij,ij->i", N, D) < 0
    off_head = H.head_base().dist(Q) > 0.0015
    low = Q[:, 1] < H.CHIN_Y + 0.06
    m = ((away | off_head) & low).astype(float).reshape(coarse + 1, coarse + 1)
    m = _resize(m[..., None], size, size)[..., 0]
    m = _blur(m, size / coarse * 1.2)
    return ss(0.42, 0.58, m)


# when parts of the face go dark (fraction of pi) for light from the left: the center line (nose,
# between the eyes) stays lit until ~112 deg, the eye sockets until ~110 deg, the far cheek's edge
# goes dark at ~22 deg; light from 30-60 deg leaves only a sliver on the far cheek
T_CENTER, T_EDGE, T_SOCKET = 0.62, 0.12, 0.61


def t_of_s(s, p, q=1.6):
    """Threshold of a point at normalized side position s (-1 = the right edge of the face or the
    right side of the head, 0 = the center line, +1 = the left edge, -2 / +2 = straight behind
    through the right / left). p > 1 keeps more of the right half lit (the boundary curves out);
    q > 1 makes the left half darken sooner once the light is behind (a lit crescent on the left
    cheek instead of half the face)."""
    s = np.asarray(s, float)
    right = T_CENTER - (T_CENTER - T_EDGE) * np.clip(-s, 0, 1) ** p
    behind = T_EDGE * np.clip(2 + s, 0, 1)
    left = T_CENTER + (1 - T_CENTER) * np.clip(s, 0, 1) ** q
    return np.where(s >= 0, left, np.where(s >= -1, right, behind))


def face_threshold(size=MAP_TEX):
    """Face shadow threshold map (Genshin's technique), in Head_Skin's UV space, authored for light
    from the character's LEFT (+X). Texel value t in [0, 1]: the texel is lit while the light's
    horizontal angle phi (head space, from the front +Z (0) through the left side to straight behind
    (pi)) is below t * pi. The client mirrors u (1 - u) for light from the right.

    As the light swings from the front to the side, the far (right) cheek darkens first along a
    curved boundary that hugs the cheekbone and the jaw (the normalized position s = x / w(y), w =
    the face's half width, with a per-height bias: the eye band and the forehead stay lit longer,
    the cheekbone darkens a little sooner); the eye sockets and the center of the face stay lit
    until the light is well past 90 deg; a small shadow beside the nose appears at mid angles.
    Off the face (sides, back of the head) s comes from the azimuth of a smooth proxy's normal. The
    neck under the jaw stays in shadow (an anime chin shadow with a crisp edge along the jaw)."""
    q, n = skin_surface(size)
    x, y, z = q[..., 0], q[..., 1], q[..., 2]
    a_n = _blur(np.arctan2(n[..., 0], n[..., 2]), 0.01 * size)
    t_n = t_of_s(np.clip(a_n / (np.pi / 2), -2, 2), 1.6)
    w = np.maximum(H.face_half_width(np.clip(y, H.CHIN_Y, 1.72)) * 1.04, 0.012)
    h = (y - H.CHIN_Y) / (H.EYE_Y - H.CHIN_Y)  # 0 at the chin, 1 at the eye line
    # boundary shape by height: jaw 1.6, cheekbone 1.25 (darkens sooner), eye band 2.4, forehead 2.0
    p = np.interp(h, [0.0, 0.3, 0.55, 0.85, 1.05, 1.35, 2.0], [1.6, 1.5, 1.25, 1.7, 2.4, 2.0, 2.0])
    # once the light is behind, the lit part shrinks to a crescent that hugs the left cheek
    qf = np.interp(h, [0.0, 0.3, 0.55, 0.85, 1.05, 1.35, 2.0], [1.8, 2.0, 2.3, 1.8, 1.35, 1.6, 1.6])
    t_f = t_of_s(np.clip(x / w, -1, 1), p, qf)
    on_face = ss(0.05, 0.4, n[..., 2]) * ss(H.CHIN_Y - 0.002, H.CHIN_Y + 0.012, y) * (1 - ss(1.72, 1.76, y))
    t = t_n * (1 - on_face) + t_f * on_face
    # the eye sockets stay lit until the light is well past 90 deg
    for sx in (1, -1):
        r = np.sqrt(((x - sx * H.EYE_X) / 0.031) ** 2 + ((y - H.EYE_Y - 0.002) / 0.025) ** 2)
        m = ss(1.0, 0.5, r) * ss(0.03, 0.05, z)
        t = t + (np.maximum(t, T_SOCKET) - t) * m
    t = _blur(t, 0.003 * size)
    # beside the nose (its right side, light from the left): a small triangle that appears at mid
    # angles (~43 deg) and grows away from the nose
    ty, tz = H.P["nose_tip"]
    pts = np.array([[-0.8, ty / MM + 1.4], [-3.4, ty / MM - 1.0], [-7.0, ty / MM - 3.0]]) * MM
    d = sd_stroke(x, y, pts, np.array([1.2, 1.5, 0.3]) * MM)
    inside = cover(d, 0.35 * MM) * ss(0.03, 0.05, z)
    wedge = 0.24 + 0.14 * ss(0.6, 7.0, -x / MM)
    t = np.minimum(t, t * (1 - inside) + wedge * inside)
    t = t * (1 - neck_shadow(size))
    return np.clip(t, 0, 1)


# ----------------------------------------------------------------------------- review helper

def lit_skin(base, thr, light_head, light_rgb, shadow_rgb, mid_rgb=None, mid=0.012):
    """Review: Head_Skin as the client's face map shading draws it, baked for one light direction
    (head space, toward the light): base color x (light | shadow) per texel."""
    lx, lz = float(light_head[0]), float(light_head[2])
    phi = math.atan2(abs(lx), lz) / math.pi  # 0 front .. 1 behind
    t = thr if lx >= 0 else thr[:, ::-1]  # light from the right: mirror u
    if t.shape != base.shape[:2]:
        t = _resize(np.pad(t, ((0, 1), (0, 1)), "edge")[..., None], base.shape[0], base.shape[1])[..., 0]
    lit = ss(phi - 0.004, phi + 0.004, t)[..., None]
    out = base * (np.asarray(shadow_rgb) * (1 - lit) + np.asarray(light_rgb) * lit)
    if mid_rgb is not None:
        band = (np.abs(t - phi) < mid)[..., None] * 0.0
        out = out * (1 - band) + base * np.asarray(mid_rgb) * band
    return np.clip(out, 0, 1)
