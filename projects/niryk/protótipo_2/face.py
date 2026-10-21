"""Anime face parts as decals: eyes, lash lines, brows and the mouth line.

Anime eyes are drawn, not sculpted: each part is a flat 2D shape (almond eye white, iris with a
dark rim and a lighter core, pupil, two highlights, a thick upper lash line with a flick, a thin
lower lash line, tapered brows, a short mouth line) designed in a 2D frame around the eye (u
toward the outer corner, v up), laid onto the skin (projected along -Z onto the head SDF) and
lifted off it in thin layers (a fraction of a millimeter each), so the parts stack cleanly like
paint. Their normals are the face's (bent toward the face ellipsoid, see head.face_normals), so
toon shading treats them like the skin around them.

Materials: EyeWhite and EyeShine (untinted; EyeShine is emissive, the highlights stay white),
Eye (iris: the client's eye color; the rim and the lid shadow are vertex-color luminance), Dark
(pupils, lash lines, mouth), Hair (brows: they follow the hair color). Nodes: Head_Eyes,
Head_Brows, Head_Mouth (rigid on Head, hidden in first person).
"""
from __future__ import annotations

import math

import numpy as np

import head as H
from common import FEMALE, slot

import pyram3d as p3


def _bump(t, peak):
    """sin(pi * t^k), 0 at t = 0 and 1, maximum 1 at t = peak."""
    k = math.log(0.5) / math.log(peak)
    return np.sin(np.pi * np.clip(t, 0, 1) ** k)


# ----------------------------------------------------------------------------- eye design (2D)
if FEMALE:
    EYE = dict(u0=-0.018, u1=0.0196, v_in=-0.0014, v_out=0.0034, h_up=0.0152, p_up=0.44, h_dn=-0.0124, p_dn=0.52,
               iris_c=(0.0008, -0.0008), iris_r=(0.0109, 0.0146), pupil_r=(0.0044, 0.0068),
               lash=0.0034, lash_in=0.001, flick=(0.0046, 0.0028), lower=0.0009, spikes=True)
    BROW = dict(u=(-0.0175, 0.0255), v=(0.0238, 0.0298, 0.0248), th=(0.0022, 0.0007), peak=0.55)
    MOUTH = dict(y=float(H.cy(1.5705)), w=0.0068, curve=0.0007, th=0.00105)
else:
    EYE = dict(u0=-0.0176, u1=0.0192, v_in=-0.0008, v_out=0.0036, h_up=0.0124, p_up=0.4, h_dn=-0.0104, p_dn=0.56,
               iris_c=(0.0006, -0.0006), iris_r=(0.0099, 0.0128), pupil_r=(0.0039, 0.006),
               lash=0.0031, lash_in=0.0011, flick=(0.0032, 0.0012), lower=0.0007, spikes=False)
    BROW = dict(u=(-0.019, 0.0265), v=(0.0188, 0.0236, 0.0196), th=(0.0046, 0.0014), peak=0.45)
    MOUTH = dict(y=float(H.cy(1.5675)), w=0.0088, curve=-0.0004, th=0.0012)


def lid_up(u):
    E = EYE
    t = (np.asarray(u, float) - E["u0"]) / (E["u1"] - E["u0"])
    return E["v_in"] + (E["v_out"] - E["v_in"]) * t + E["h_up"] * _bump(t, E["p_up"])


def lid_dn(u):
    E = EYE
    t = (np.asarray(u, float) - E["u0"]) / (E["u1"] - E["u0"])
    return E["v_in"] + (E["v_out"] - E["v_in"]) * t + E["h_dn"] * _bump(t, E["p_dn"])


def ellipse(c, r):
    """(lo(u), hi(u)) of an ellipse; NaN outside its u range."""
    def lo(u):
        q = 1 - ((np.asarray(u, float) - c[0]) / r[0]) ** 2
        return np.where(q > 0, c[1] - r[1] * np.sqrt(np.maximum(q, 0)), np.nan)

    def hi(u):
        q = 1 - ((np.asarray(u, float) - c[0]) / r[0]) ** 2
        return np.where(q > 0, c[1] + r[1] * np.sqrt(np.maximum(q, 0)), np.nan)
    return lo, hi


def in_eye(lo, hi, shrink=0.0):
    """Intersect a region with the eye opening (minus `shrink` under the upper lid)."""
    def L(u):
        return np.fmax(lo(u), lid_dn(u))

    def Hh(u):
        return np.fmin(hi(u), lid_up(u) - shrink)
    return L, Hh


# ----------------------------------------------------------------------------- 2D meshing

def region(lo, hi, u0, u1, cols=16, rows=4):
    """Mesh {u0 <= u <= u1, lo(u) <= v <= hi(u)} as columns of quads; the ends where the region
    closes become fans. The u range is trimmed to where the region is open."""
    us = np.linspace(u0, u1, 801)
    h = hi(us) - lo(us)
    ok = np.isfinite(h) & (h > 1e-6)
    if not ok.any():
        return np.zeros((0, 2)), []
    i0, i1 = np.flatnonzero(ok)[[0, -1]]
    a = us[max(i0 - 1, 0)]
    b = us[min(i1 + 1, len(us) - 1)]
    s = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, cols + 1))
    U = a + (b - a) * s
    verts, colidx = [], []
    for k, u in enumerate(U):
        lo_, hi_ = float(lo(u)), float(hi(u))
        if not (np.isfinite(lo_) and np.isfinite(hi_)) or hi_ - lo_ < 2e-5:
            # closed end: one vertex where the region pinches off
            m = lo_ if np.isfinite(lo_) else hi_
            if not np.isfinite(m):
                m = float(np.nanmean([lo(U[min(k + 1, cols)]), hi(U[min(k + 1, cols)]), lo(U[max(k - 1, 0)]),
                                      hi(U[max(k - 1, 0)])]))
            colidx.append([len(verts)])
            verts.append((u, m if np.isfinite(lo_) and np.isfinite(hi_) else m))
        else:
            ids = []
            for j in range(rows + 1):
                ids.append(len(verts))
                verts.append((u, lo_ + (hi_ - lo_) * j / rows))
            colidx.append(ids)
    faces = []
    for c0, c1 in zip(colidx, colidx[1:]):
        if len(c0) > 1 and len(c1) > 1:
            for j in range(rows):
                faces.append((c0[j], c1[j], c1[j + 1], c0[j + 1]))
        elif len(c0) == 1 and len(c1) > 1:
            for j in range(rows):
                faces.append((c0[0], c1[j], c1[j + 1]))
        elif len(c0) > 1 and len(c1) == 1:
            for j in range(rows):
                faces.append((c0[j], c1[0], c0[j + 1]))
    return np.array(verts), faces


def ribbon(curve, th, n=16, side=0.5):
    """Stroke along curve(t) -> (u, v) (t in [0, 1]) with thickness th(t); `side` = share of the
    thickness above the curve (left of its direction). Ends with th = 0 close to a point."""
    t = np.linspace(0, 1, n + 1)
    C = np.array([curve(x) for x in t])
    d = np.gradient(C, axis=0)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    nrm = np.stack([-d[:, 1], d[:, 0]], axis=1)
    T = np.array([th(x) for x in t])
    verts, cols = [], []
    for k in range(n + 1):
        if T[k] < 2e-5:
            cols.append([len(verts)])
            verts.append(tuple(C[k]))
        else:
            cols.append([len(verts), len(verts) + 1])
            verts.append(tuple(C[k] - nrm[k] * T[k] * (1 - side)))
            verts.append(tuple(C[k] + nrm[k] * T[k] * side))
    faces = []
    for c0, c1 in zip(cols, cols[1:]):
        if len(c0) == 2 and len(c1) == 2:
            faces.append((c0[0], c1[0], c1[1], c0[1]))
        elif len(c0) == 1 and len(c1) == 2:
            faces.append((c0[0], c1[0], c1[1]))
        elif len(c0) == 2 and len(c1) == 1:
            faces.append((c0[0], c1[0], c0[1]))
    return np.array(verts), faces


# ----------------------------------------------------------------------------- onto the skin

_HEAD = None


def _head():
    global _HEAD
    if _HEAD is None:
        _HEAD = H.head()
    return _HEAD


def lay(xy, lift):
    """Head-space (x, y) points -> design-space points on the skin lifted by `lift` meters along
    its normal (+ their face normals)."""
    hd = _head()
    xy = np.asarray(xy, float)
    lo = np.zeros(len(xy))
    hi = np.full(len(xy), 0.16)
    for _ in range(40):  # bisection along z: inside at lo, outside at hi
        mid = 0.5 * (lo + hi)
        d = hd.dist(np.column_stack([xy, mid]))
        inside = d < 0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    P = np.column_stack([xy, lo])
    n = hd.gradient(P, eps=2e-4)
    P = P + n * lift
    Pd = H.head_xf(P)
    return Pd, H.face_normals(Pd, n)


def decal(verts2d, faces, to_xy, lift, material, lum=None, name="Decal", flip=False):
    """2D shape -> mesh on the skin. to_xy maps (u, v) to head-space (x, y)."""
    xy = np.array([to_xy(u, v) for u, v in verts2d])
    P, N = lay(xy, lift)
    if flip:
        faces = [tuple(reversed(f)) for f in faces]
    m = p3.Mesh(P, faces, name=name)
    m.normals = N
    m.smooth_angle = 180.0
    m.set_material(slot(material))
    m.colors = np.ones((m.num_vertices, 4))
    if lum is not None:
        m.colors[:, :3] = np.clip(lum(np.asarray(verts2d)), 0, 1)[:, None]
    return m


def eye(sx):
    """One eye (sx = +1 the character's left, -1 the right) as a list of decal meshes."""
    E = EYE
    to_xy = lambda u, v: (sx * (H.EYE_X + u), H.EYE_Y + v)  # noqa: E731
    flip = sx < 0
    parts = []
    # eye white (shaded under the upper lid)
    v, f = region(lid_dn, lid_up, E["u0"], E["u1"], cols=18, rows=4)

    def white_lum(V):
        d = lid_up(V[:, 0]) - V[:, 1]
        return 1 - 0.3 * np.clip(1 - d / 0.0045, 0, 1)
    parts.append(decal(v, f, to_xy, 0.00025, "EyeWhite", white_lum, "White", flip))
    # iris: dark rim, then a lighter core darkened under the lid
    c, r = E["iris_c"], E["iris_r"]
    lo, hi = in_eye(*ellipse(c, r))
    v, f = region(lo, hi, c[0] - r[0], c[0] + r[0], cols=18, rows=4)
    parts.append(decal(v, f, to_xy, 0.0005, "Eye", lambda V: np.full(len(V), 0.42), "IrisRim", flip))
    r2 = (r[0] * 0.8, r[1] * 0.84)
    c2 = (c[0], c[1] - r[1] * 0.04)
    lo, hi = in_eye(*ellipse(c2, r2))
    v, f = region(lo, hi, c2[0] - r2[0], c2[0] + r2[0], cols=16, rows=5)

    def iris_lum(V):
        y = (V[:, 1] - c2[1]) / r2[1]  # -1 bottom .. 1 top
        under_lid = np.clip(1 - (lid_up(V[:, 0]) - V[:, 1]) / 0.006, 0, 1)
        return (0.62 + 0.38 * np.clip((0.55 - y) / 1.2, 0, 1)) * (1 - 0.35 * under_lid)
    parts.append(decal(v, f, to_xy, 0.0007, "Eye", iris_lum, "Iris", flip))
    # pupil
    pr = E["pupil_r"]
    pc = (c[0], c[1] + 0.0004)
    lo, hi = in_eye(*ellipse(pc, pr))
    v, f = region(lo, hi, pc[0] - pr[0], pc[0] + pr[0], cols=12, rows=3)
    parts.append(decal(v, f, to_xy, 0.0009, "Dark", None, "Pupil", flip))
    # highlights: a big one up toward the outer corner, a small one low on the inner side
    for hc, hr in (((c[0] + r[0] * 0.38, c[1] + r[1] * 0.3), (r[0] * 0.3, r[1] * 0.22)),
                   ((c[0] - r[0] * 0.42, c[1] - r[1] * 0.42), (r[0] * 0.15, r[1] * 0.1))):
        lo, hi = in_eye(*ellipse(hc, hr), shrink=0.0012)
        v, f = region(lo, hi, hc[0] - hr[0], hc[0] + hr[0], cols=10, rows=2)
        if len(f):
            parts.append(decal(v, f, to_xy, 0.0011, "EyeShine", None, "Shine", flip))
    # upper lash line: on the lid edge, thickest past the middle, ending in a flick
    u0, u1 = E["u0"] - 0.0006, E["u1"] + E["flick"][0]

    def up_curve(t):
        u = u0 + (u1 - u0) * t
        if u <= E["u1"]:
            return np.array([u, float(lid_up(u))])
        k = (u - E["u1"]) / E["flick"][0]
        return np.array([u, float(lid_up(E["u1"])) + E["flick"][1] * k ** 1.3 + 0.0006 * k])

    def up_th(t):
        rise = np.clip(t / 0.12, 0, 1) ** 0.7
        tip = np.clip((1 - t) / 0.16, 0, 1) ** 0.8
        return (E["lash_in"] + (E["lash"] - E["lash_in"]) * _bump(np.clip(t * 1.05, 0, 1), 0.72)) * rise * tip
    v, f = ribbon(up_curve, up_th, n=24, side=0.82)
    parts.append(decal(v, f, to_xy, 0.001, "Dark", None, "LashUp", flip))
    if E["spikes"]:  # a few lashes at the outer corner
        for a, ln, wd, k in ((0.86, 0.0045, 0.0011, 0.0), (0.74, 0.0036, 0.0009, 0.25), (0.97, 0.0032, 0.0009, -0.2)):
            base = up_curve(a)
            dirv = np.array([0.55 + k, 0.85])
            dirv /= np.linalg.norm(dirv)
            sp = lambda t, b=base, dv=dirv, L=ln: b + dv * L * t + np.array([L * 0.25 * t * t, 0.0])  # noqa: E731
            v, f = ribbon(sp, lambda t, w=wd: w * (1 - t) ** 0.9, n=5)
            parts.append(decal(v, f, to_xy, 0.00095, "Dark", None, "Lash", flip))
    # lower lash line: thin, on the outer half of the lower lid
    def dn_curve(t):
        u = E["u0"] + (E["u1"] - E["u0"]) * (0.42 + 0.6 * t)
        return np.array([u, float(lid_dn(u))])
    v, f = ribbon(dn_curve, lambda t: E["lower"] * _bump(t, 0.6) ** 0.7, n=12, side=0.3)
    parts.append(decal(v, f, to_xy, 0.0006, "Dark", None, "LashDn", flip))
    return parts


def eyes() -> p3.Mesh:
    parts = eye(1) + eye(-1)
    return p3.Mesh.join(parts, name="Head_Eyes")


def brows() -> p3.Mesh:
    B = BROW
    parts = []
    for sx in (1, -1):
        to_xy = lambda u, v, s=sx: (s * (H.EYE_X + u), H.EYE_Y + v)  # noqa: E731

        def curve(t):
            u = B["u"][0] + (B["u"][1] - B["u"][0]) * t
            v0, vm, v1 = B["v"]
            v = v0 + (v1 - v0) * t + (vm - 0.5 * (v0 + v1)) * _bump(t, B["peak"])
            return np.array([u, v])

        def th(t):
            return (B["th"][0] + (B["th"][1] - B["th"][0]) * t) * np.clip(t / 0.06, 0.35, 1) * np.clip((1 - t) / 0.12, 0, 1) ** 0.6
        v, f = ribbon(curve, th, n=16, side=0.5)
        parts.append(decal(v, f, to_xy, 0.0006, "Hair", None, "Brow", sx < 0))
    return p3.Mesh.join(parts, name="Head_Brows")


def mouth() -> p3.Mesh:
    M = MOUTH
    to_xy = lambda u, v: (u, M["y"] + v)  # noqa: E731

    def curve(t):
        u = -M["w"] + 2 * M["w"] * t
        return np.array([u, M["curve"] * (2 * t - 1) ** 2])

    def th(t):
        return M["th"] * np.sin(np.pi * t) ** 0.6
    v, f = ribbon(curve, th, n=12, side=0.5)
    return decal(v, f, to_xy, 0.0005, "Dark", None, "Head_Mouth")


def face_meshes():
    return [eyes(), brows(), mouth()]
