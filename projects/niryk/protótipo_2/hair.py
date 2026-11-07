"""Anime hair in the Genshin Impact line, as smooth explicit meshes: Hair_curto, Hair_rabo,
Hair_longo, Hair_coque (humans) and Hair_espetado, Hair_crista, Hair_plumado (Harpins).

Every style is a smooth volume cap over the scalp plus layered locks. A lock is a swept strip with
thickness: a lens section that is wide and thin, its edges curled toward the head, along a
Catmull-Rom path, growing from a buried root and tapering to a point. It is built as a coarse quad
cage and smoothed by one Catmull-Clark step; normals come from the smooth surface (then bent part of
the way toward a hull around the head, so toon bands draw big clean shapes). Bangs frame the face
(over the irises their tips stop above the upper lash line), side locks fall in front of the ears,
the back is a smooth mass (the cap and wide back locks).

Two-tone shading is baked in vertex colors (the client multiplies the hair color): a base tone and a
darker underside (surfaces facing down or toward the head), plus the "angel ring": a zigzag highlight
band around the top of the head, cut into the mesh so its edges are crisp.

Locks on the head are laid out in (azimuth, elevation, lift): azimuth 0 = front, +90 = the
character's left, elevation seen from the cranium center CRAN_C, lift = meters off the head surface.
Hanging parts (ponytail, long hair) use design-space points behind the back. Everything is built in
head space and scaled with the head (head.head_xf).
"""
from __future__ import annotations

import math

import numpy as np

import head as H
from common import FEMALE, HARPIN, slot

import pyram3d as p3
from pyram3d import math3d as m3

STYLES = ("espetado", "crista", "plumado") if HARPIN else ("curto", "rabo", "longo", "coque")
PART_M = 24.0  # male side part (azimuth, on his left)
CRAN_C, CRAN_R = H.CRAN_C, H.CRAN_R
BASE, UNDER, RING = 0.8, 0.6, 1.0  # vertex-color luminance: base tone, underside, angel ring


def _ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _dirs(az, el):
    a, e = np.radians(az), np.radians(el)
    return np.stack([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)], axis=-1)


def _surface_r(d):
    """Distance from CRAN_C to the head surface (no ears, no nose) along unit directions d."""
    hb = H.head_base()
    lo = np.zeros(len(d))
    hi = np.full(len(d), 0.25)
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        inside = hb.dist(CRAN_C + d * mid[:, None]) < 0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return lo


def scalp(az, el, lift=0.0):
    """Head-space point `lift` m above the head surface at azimuth/elevation (degrees)."""
    one = np.ndim(az) == 0
    az, el, lift = (np.atleast_1d(np.asarray(a, float)) for a in np.broadcast_arrays(az, el, lift))
    d = _dirs(az, el)
    out = CRAN_C + d * (_surface_r(d) + lift)[:, None]
    return out[0] if one else out


def to_head(p):
    """Design-space point -> head space (the hair is scaled with the head)."""
    return H.to_head(p)


def cap_t(az, el):
    """Cap thickness (volume of the hair over the scalp)."""
    top = _ss(10, 70, el)
    back = _ss(90, 160, np.abs(((np.asarray(az, float) + 180) % 360) - 180))
    base = 0.015 if FEMALE else 0.012
    return base + (0.011 if FEMALE else 0.012) * top + 0.004 * back


def P(az, el, extra=0.0):
    """(az, el, lift) just above the cap."""
    return (az, el, float(cap_t(az, el)) + extra)


def _pt(p):
    return np.asarray(p, float) if isinstance(p, np.ndarray) else scalp(*p)


# ----------------------------------------------------------------------------- locks

def lock(pts, w, th, K=4, root=0.75, peak=0.3, tip=0.0, thin=0.55, bend=0.35, rings=None, out=None,
         samples=10) -> p3.Mesh:
    """A pointed lock through `pts` ((az, el, lift) tuples or head-space points), root first.
    w: max half width (reached at `peak` of the length), th: half thickness, bend: how far the
    edges curl toward the head (fraction of the width), out: an outward direction (default: away
    from CRAN_C). Returns a smooth closed mesh (cage + one Catmull-Clark step)."""
    Q = np.array([_pt(p) for p in pts])
    C = m3.catmull_rom(Q, samples)
    seg = np.linalg.norm(np.diff(C, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    L = s[-1]
    n = rings or int(np.clip(round(L / 0.034) + 2, 3, 7))
    t = (np.arange(n) / n) ** 0.92  # rings (the tip vertex closes at t = 1)
    pos = np.column_stack([np.interp(t * L, s, C[:, k]) for k in range(3)])
    tip_p = C[-1]
    allp = np.vstack([pos, tip_p])
    T = np.gradient(allp, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    T = T[:n]
    if out is None:
        O = pos - CRAN_C
    else:
        O = np.broadcast_to(np.asarray(out, float), pos.shape).copy()
    O = O - T * np.einsum("ij,ij->i", O, T)[:, None]
    O /= np.linalg.norm(O, axis=1, keepdims=True)
    S = np.cross(T, O)
    grow = root + (1 - root) * _ss(0, peak, t)
    taper = 1 - (1 - tip) * np.clip((t - peak) / (1 - peak), 0, 1) ** 1.3
    W = 1.12 * w * grow * taper  # the Catmull-Clark step shrinks the cage
    Th = 1.15 * th * (1 - thin * _ss(0.4, 1.0, t))
    ang = np.linspace(0, 2 * np.pi, K, endpoint=False)
    cx, cy = np.cos(ang), np.sin(ang)
    V = []
    for i in range(n):
        x = W[i] * cx
        y = Th[i] * cy - bend * W[i] * cx * cx
        V.append(pos[i] + np.outer(x, S[i]) + np.outer(y, O[i]))
    V = np.vstack(V)
    tip_i = len(V)
    root_i = tip_i + 1
    V = np.vstack([V, tip_p, pos[0] - T[0] * 0.6 * Th[0]])
    F = []
    for i in range(n - 1):
        for k in range(K):
            a, b = i * K + k, i * K + (k + 1) % K
            F.append((a, b, b + K, a + K))
    for k in range(K):
        F.append(((n - 1) * K + k, (n - 1) * K + (k + 1) % K, tip_i))
        F.append(((k + 1) % K, k, root_i))
    m = p3.Mesh(V, F, name="lock")
    # outward winding: the first ring's quads must face away from the lock's axis
    fn = m.face_normals()[0]
    if np.dot(fn, m.face_centers()[0] - pos[0]) < 0:
        m.faces = [tuple(reversed(f)) for f in m.faces]
    m.subdivide(1)
    return m


def tube(pts, r, rings=None, K=6):
    """Round closed tube (a tie)."""
    return lock(pts, r, r, K=K, root=1.0, peak=0.01, tip=1.0, thin=0.0, bend=0.0, rings=rings)


# ----------------------------------------------------------------------------- cap

def hairline(lo_front=0.0):
    """Elevation (deg) of the hairline at each azimuth: forehead, temples, above and behind the
    ears, nape. lo_front lowers it at the front (styles with long bangs)."""
    az = np.array([0, 40, 65, 88, 110, 135, 160, 180])
    el = np.array([26 - lo_front, 18 - lo_front, 2, -8, -18, -34, -44, -46])
    return az, el


def cap(lo_front=0.0, n_az=20, n_rows=5) -> p3.Mesh:
    """Smooth volume cap from the crown to the hairline (its edge tucked under the skin)."""
    A, E = hairline(lo_front)
    azs = np.linspace(-180, 180, n_az, endpoint=False)
    rows = []
    for az in azs:
        eh = float(np.interp(abs(az), A, E))
        els = np.concatenate([np.linspace(84, eh, n_rows), [eh - 4]])
        lift = cap_t(az, els) * _ss(eh, eh + 14, els) ** 0.6
        lift[-1] = -0.002
        rows.append(scalp(np.full(len(els), az), els, lift))
    R = np.stack(rows, axis=1)  # (rows, az, 3)
    nr = R.shape[0]
    V = R.reshape(-1, 3)
    pole = scalp(0.0, 90.0, float(cap_t(0, 90)))
    V = np.vstack([V, pole])
    F = []
    for r in range(nr - 1):
        for k in range(n_az):
            a, b = r * n_az + k, r * n_az + (k + 1) % n_az
            F.append((a, a + n_az, b + n_az, b))
    for k in range(n_az):
        F.append(((k + 1) % n_az, k, len(V) - 1))
    m = p3.Mesh(V, F, name="cap")
    if np.dot(m.face_normals()[0], m.face_centers()[0] - CRAN_C) < 0:
        m.faces = [tuple(reversed(f)) for f in m.faces]
    m.subdivide(1)
    return m


def covered_by_all_styles(P_design, margin=5.0):
    """Head_Skin points under the cap of every style (deleted from the skin)."""
    q = to_head(P_design)
    d = q - CRAN_C
    az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
    el = np.degrees(np.arcsin(np.clip(d[:, 1] / np.maximum(np.linalg.norm(d, axis=1), 1e-9), -1, 1)))
    A, E = hairline(0.0)
    return el > np.interp(np.abs(az), A, E) + margin


# ----------------------------------------------------------------------------- shared parts

def bangs():
    """Fringe framing the face: locks of different widths and lengths that overlap, the outer ones
    curving outward, the middle ones toward the center, with a few thin pointed strands between
    them. Over the irises the tips stay above the upper lash line; between the eyes and at the
    outer corners they fall lower. Female: soft, side-swept from a parting on her right, a long lock
    sweeping across the forehead. Male: spikier, in two layers."""
    L = []

    part = -14 if FEMALE else 0  # the parting's azimuth (female: on her right)

    def fringe(r, t, e, w, th, curl=0.0, lift=0.0055, peak=0.4):
        # root in the cap in front of the crown -> over the forehead -> the tip (`curl` pushes its
        # azimuth outward or inward)
        m1 = P(part + (r - part) * 0.7, 60, -0.002)
        m2 = P(part + (r - part) * 0.9, 44, 0.006)
        m3_ = P((r + t) / 2, 27, 0.007)
        m4 = P(t - curl * 0.4, 11, 0.006)
        return lock([m1, m2, m3_, m4, (t + curl, e, lift)], w, th, peak=peak, bend=0.32)

    if FEMALE:
        # (root az, tip az, tip el, half width, curl)
        for r, t, e, w, c in ((-66, -62, -19, 0.019, -4), (-50, -47, -10, 0.022, -3), (-35, -32, -5, 0.025, -1),
                              (-20, -18, -7, 0.02, 1), (-6, -4, -19, 0.023, 2), (16, 15, -6, 0.022, -1),
                              (31, 30, -4, 0.025, 1), (47, 45, -10, 0.022, 3), (65, 61, -19, 0.019, 4)):
            L.append(fringe(r, t, e, w, 0.0024, c))
        for r, t, e, c in ((-42, -40, -8, -1), (5, 7, -15, -2), (39, 39, -8, 2)):  # thin pointed strands
            L.append(fringe(r, t, e, 0.0105, 0.0019, c, lift=0.0068, peak=0.3))
        L.append(lock([P(-22, 62, 0.0), P(-6, 50, 0.009), P(10, 36, 0.011), P(28, 18, 0.011), (40, -4, 0.0085)],
                      0.028, 0.0026, peak=0.36, bend=0.3))
    else:
        # swept from a side part on his left (az PART): most locks root along the part and sweep
        # across the forehead to his right in layers, a few fall to the left; pointed tips
        part = PART_M
        for el0, t, e, w in ((68, -4, -17, 0.027), (63, -20, -6, 0.028), (57, -36, -7, 0.026),
                             (51, -50, -11, 0.023), (45, -62, -15, 0.019)):
            L.append(lock([P(part - 3, el0, -0.002), P((part + t) / 2 + 6, (el0 + 18) / 2 + 8, 0.012),
                           P(t + 9, 15, 0.01), (t, e, 0.0065)], w, 0.0029, peak=0.36, bend=0.32))
        for el0, t, e, w in ((62, 31, -5, 0.025), (54, 47, -9, 0.022), (47, 60, -13, 0.019)):
            L.append(lock([P(part + 3, el0, -0.002), P((part + t) / 2 - 2, (el0 + 18) / 2 + 6, 0.011),
                           P(t - 4, 14, 0.009), (t, e, 0.0065)], w, 0.0028, peak=0.36, bend=0.32))
        for el0, t, e in ((66, -13, -11), (60, -44, -9), (58, 40, -7)):  # thin pointed strands
            L.append(lock([P(part, el0, -0.002), P((part + t) / 2 + 3, (el0 + 18) / 2 + 7, 0.013),
                           P(t + 5, 14, 0.011), (t, e, 0.0078)], 0.0105, 0.0021, peak=0.3, bend=0.3))
    return L


def sides(length="short"):
    """Locks in front of and over the ears. length: short (ear), jaw, long (collarbone)."""
    L = []
    for s in (1, -1):
        if length == "short":
            if FEMALE:
                L.append(lock([P(s * 68, 46, -0.002), P(s * 73, 30, 0.005), P(s * 77, 8, 0.008), (s * 78, -22, 0.008)],
                              0.025, 0.003, bend=0.32))
            else:  # sideburn in front of the ear
                L.append(lock([P(s * 66, 44, -0.002), P(s * 71, 26, 0.006), P(s * 74, 4, 0.008), (s * 76, -18, 0.007)],
                              0.02, 0.0028, bend=0.32))
            # swept back over the ear and behind it, the tips pointing back
            fl = 1.0 if FEMALE else 1.5  # male: the tips flare out (a spiky silhouette)
            L.append(lock([P(s * 74, 50, -0.002), P(s * 86, 28, 0.009), P(s * 98, 6, 0.017), (s * 112, -12, 0.022 * fl)],
                          0.027, 0.0034, bend=0.32))
            L.append(lock([P(s * 96, 50, -0.002), P(s * 106, 26, 0.008), P(s * 116, 2, 0.016), (s * 128, -20, 0.02 * fl)],
                          0.028, 0.0034, bend=0.32))
            L.append(lock([P(s * 120, 46, -0.002), P(s * 128, 22, 0.007), P(s * 134, -6, 0.014), (s * 140, -30, 0.018 * fl)],
                          0.028, 0.0034, bend=0.32))
        else:
            y_tip = {"jaw": 1.565, "long": 1.47}[length]
            # framing lock in front of the ear, falling past the cheek
            tip = to_head(np.array([s * 0.083, y_tip, 0.038]))
            mid = scalp(s * 78, -26, 0.014)
            L.append(lock([P(s * 64, 46, -0.002), P(s * 70, 30, 0.007), P(s * 76, 2, 0.009), mid, tip], 0.027, 0.003,
                          bend=0.3))
            tip2 = to_head(np.array([s * 0.098, y_tip + 0.02, -0.004]))
            L.append(lock([P(s * 97, 44, -0.002), P(s * 100, 24, 0.008), P(s * 103, -6, 0.022), scalp(s * 102, -30, 0.026),
                           tip2], 0.03, 0.0036, bend=0.32))
            tip3 = to_head(np.array([s * 0.094, y_tip + 0.035, -0.05]))
            L.append(lock([P(s * 124, 42, -0.002), P(s * 126, 22, 0.008), P(s * 128, -8, 0.014), scalp(s * 126, -34, 0.018),
                           tip3], 0.032, 0.0038, bend=0.32))
    return L


def top(back_az=180, back_el=10):
    """Broad layered locks from the crown whorl down the back toward (back_az, back_el) (the nape,
    a ponytail tie or a bun)."""
    L = []
    for az in (-150, 150, -118, 118):
        a1 = back_az + (az - math.copysign(180, az)) * 0.35
        L.append(lock([P(az * 0.3 + 126 * math.copysign(1, az), 82, 0.0), P(az, 55, 0.003),
                       P((az + a1) / 2, 30, 0.004), P(a1, back_el, 0.002)], 0.03, 0.0045, root=0.9, tip=0.4,
                      thin=0.3, bend=0.4))
    return L


def nape(length=0.0):
    """Pointed locks over the back of the head in two layers; `length` lowers the lower layer's tips."""
    L = []
    azs = (124, 152, 180, -152, -124) if not FEMALE else (118, 142, 166, -166, -142, -118)
    for i, az in enumerate(azs):
        el_tip = -46 - 5 * (i % 2)
        tip = scalp(az, el_tip, 0.013 if FEMALE else 0.026)
        if length:
            tip = tip + np.array([0, -length / H.HEAD_SCALE, -0.008])
        L.append(lock([P(az, 36, -0.002), P(az, 18, 0.004), P(az, -14, 0.007), scalp(az, -34, 0.014 if FEMALE else 0.02),
                       tip], 0.024 if FEMALE else 0.022, 0.004, bend=0.35))
    for i, az in enumerate(np.array(azs[:-1]) + (14 if not FEMALE else 12)):  # upper layer, shorter
        az = float((az + 180) % 360 - 180)
        L.append(lock([P(az, 44, -0.002), P(az, 30, 0.008), P(az, 4, 0.01),
                       (az, -16 - 6 * (i % 2), float(cap_t(az, -16)) + (0.018 if FEMALE else 0.03))],
                      0.022, 0.004, bend=0.35))
    return L


# ----------------------------------------------------------------------------- styles

def style_curto():
    if FEMALE:  # chin-length bob: side locks to the jaw, the back to the nape curling in
        return [cap(6)] + bangs() + top() + sides("jaw") + nape(0.03)
    # a Genshin male lead's short cut: swept bangs, volume at the crown, layered pointed clumps
    # standing off the back and the sides (no smooth bowl)
    crown = spikes((-160, 168, 136, -128), 84, 44, 0.04, w=0.033, th=0.0045)
    layers = spikes((-165, -135, -105, 105, 135, 165), 50, 2, 0.034, w=0.03, th=0.0045)
    return [cap()] + bangs() + sides("short") + nape() + crown + layers


def tie_point():
    return scalp(180, 34 if FEMALE else 30, 0.02)


def _ring(center, axis, r_major, r_minor, name="tie"):
    m = p3.torus(r_major, r_minor, 24, 10)
    ax = m3.normalize(np.asarray(axis, float))
    q = m3.quat_from_axis_angle(m3.normalize(np.cross([0, 1.0, 0], ax)) if abs(ax[1]) < 0.999 else (1, 0, 0),
                                math.degrees(math.acos(float(np.clip(ax[1], -1, 1)))))
    m.rotate(q).translate(center)
    m.set_material(slot("Leather"))
    m.name = name
    return m


def style_rabo():
    """High ponytail: everything combed back to a tie at the back of the crown; the tail falls
    behind the back to the shoulder blades (female: lower)."""
    L = [cap()] + bangs() + sides("jaw" if FEMALE else "short") + top(back_el=34)
    T = tie_point()
    end_y = 1.2 if FEMALE else 1.34
    Td = H.head_xf(T)
    axis = [Td + np.array([0, 0.004, -0.014]), np.array([0, Td[1] - 0.035, -0.205]), np.array([0, 1.56, -0.205]),
            np.array([0, 1.44, -0.19]), np.array([0, end_y, -0.17])]
    spread = [0.008, 0.024, 0.03, 0.027, 0.02]
    n_locks = 7 if FEMALE else 6
    for i in range(n_locks):
        th = 2 * math.pi * i / n_locks + 0.4
        pts = []
        for j, (A, r) in enumerate(zip(axis, spread)):
            off = np.array([math.cos(th) * r * 1.1, 0.0, math.sin(th) * r * 1.15])
            if j == len(axis) - 1:  # tips at different lengths, splaying a little
                off = off * 1.6 + np.array([0, 0.03 * ((i * 7) % 3) / 2, 0])
            pts.append(to_head(A + off) if j else T + (A + off - Td) / H.HEAD_SCALE)
        out = np.array([math.cos(th), 0.0, math.sin(th)])
        L.append(lock(pts, 0.022 if FEMALE else 0.02, 0.009, K=4, peak=0.35, root=0.7, thin=0.45, bend=0.25,
                      out=out, rings=8))
    L.append(_ring(T + np.array([0, 0.004, -0.012]), (0, 0.26, -1.0), 0.016, 0.0065))
    return L


def style_longo():
    """Long hair: a curtain of wide locks from the crown down the back (female: to mid-back), long
    framing locks at the sides."""
    L = [cap(6 if FEMALE else 0)] + bangs() + sides("long") + top(back_el=-10)
    end_y = 1.17 if FEMALE else 1.3
    azs = (128, 146, 164, 180, -164, -146, -128)
    for i, az in enumerate(azs):
        a = math.radians(az)
        x_end = 0.088 * math.sin(a) * 1.45
        mid = scalp(az, -38, 0.03)
        b = to_head(np.array([x_end * 0.95, 1.5, -0.165]))
        c = to_head(np.array([x_end, 1.4, -0.168]))
        d = to_head(np.array([x_end * 1.02, end_y + 0.035 * abs(i - 3) ** 0.8 - 0.012 * (i % 2), -0.158]))
        L.append(lock([P(az, 36, -0.002), P(az, 6, 0.004), mid, b, c, d], 0.042, 0.008, peak=0.4, root=0.85,
                      thin=0.5, bend=0.4, rings=8))
    return L


def style_coque():
    """Bun: everything combed back and up into a bun at the back of the head, a tie under it, a
    few loose strands at the nape."""
    L = [cap()] + bangs() + sides("jaw" if FEMALE else "short") + top(back_el=22)
    c = scalp(180, 24, 0.042 if FEMALE else 0.036)
    r = 0.039 if FEMALE else 0.034
    nb = m3.normalize(c - CRAN_C)  # the bun's axis (out of the head)
    e1 = m3.normalize(np.cross(nb, [1.0, 0, 0]))
    e2 = np.cross(nb, e1)
    bun = p3.uv_sphere(r * 0.86, 20, 12)
    bun.vertices = bun.vertices * np.array([1.0, 0.96, 0.96])
    q = m3.quat_from_axis_angle(m3.normalize(np.cross([0, 1.0, 0], nb)), math.degrees(math.acos(float(np.clip(nb[1], -1, 1)))))
    bun.rotate(q).translate(c)
    bun.name = "bun"
    L.append(bun)
    for k in range(4):  # spiral locks wrapping the bun from its base toward its pole
        pts = []
        for tt in np.linspace(0, 1, 7):
            pol = math.radians(85 - 70 * tt)
            az = math.radians(90 * k + 260 * tt)
            d = math.cos(pol) * nb + math.sin(pol) * (math.cos(az) * e1 + math.sin(az) * e2)
            pts.append(c + d * (r * 0.9))
        out_c = c  # outward = away from the bun's center
        Q = np.array(pts)
        L.append(lock(list(Q), 0.019 if FEMALE else 0.016, 0.006, K=4, root=0.9, peak=0.25, thin=0.3, bend=0.25,
                      out=Q[len(Q) // 2] - out_c, rings=8))
    L.append(_ring(c - nb * r * 0.62, nb, r * 0.66, 0.0055))
    for s in (1, -1):
        L.append(lock([P(s * 150, 10), scalp(s * 152, -20, 0.009), scalp(s * 150, -42, 0.009),
                       scalp(s * 146, -56, 0.005)], 0.012, 0.0025, bend=0.3))
    return L


# ----------------------------------------------------------------------------- Harpin styles

def spikes(azs, el0, el1, lift, w=0.022, th=0.004, bend=0.0):
    """Spiky locks standing off the head: rooted at elevation el0, going toward el1 while the
    tip lifts `lift` m off the cap (bend: azimuth drift of the tip, deg)."""
    L = []
    for i, az in enumerate(azs):
        e1 = el1 + 4 * ((i * 7) % 3 - 1)
        mid_el = el0 + (e1 - el0) * 0.5
        L.append(lock([P(az, el0), P(az + bend * 0.4, mid_el, lift * 0.35), (az + bend, e1, float(cap_t(az, e1)) + lift)],
                      w, th, peak=0.3, thin=0.5, bend=0.3))
    return L


def style_espetado():
    """Spiky: pointed bangs, locks standing up and back from the crown, short spiky sides and nape."""
    L = [cap()] + bangs() + sides("jaw" if FEMALE else "short")  # female: a longer, softer pixie
    L += spikes((-160, -120, -80, -40, 0, 40, 80, 120, 160), 64, 26, 0.05, w=0.03)
    L += spikes((-150, -105, 105, 150), 40, -6, 0.04, w=0.028)
    L += spikes((-30, 0, 30), 78, 60, 0.055, w=0.027)
    L += spikes((-158, 158, 180, -136, 136), 8, -40, 0.03, w=0.024)
    return L


def style_crista():
    """Crest: the sides combed flat and close, a crest of long pointed locks standing along the
    middle from the forehead to the nape (tallest over the crown), like a bird's crest."""
    L = [cap(8)]
    crest = [(0, 38, 66, 0.055), (0, 60, 84, 0.075), (180, 90, 68, 0.095), (180, 72, 40, 0.1), (180, 52, 14, 0.085),
             (180, 30, -12, 0.065), (180, 8, -36, 0.045)]
    for i, (az, el0, el1, lift) in enumerate(crest):
        for dx in ((-9, 9) if i else (0,)):
            a0 = az + dx * (1 if az == 180 else 0.5)
            L.append(lock([P(a0, el0), P(a0, (el0 + el1) / 2 + (8 if az == 0 else 0), lift * 0.45),
                           (a0 + dx * 0.3, el1, float(cap_t(a0, el1)) + lift)], 0.026, 0.005, peak=0.3, thin=0.45,
                          bend=0.25))
    for sgn in (1, -1):  # flat side locks combed back
        for az, el in ((60, 40), (90, 30), (120, 20), (150, 10)):
            L.append(lock([P(sgn * az * 0.7, el + 18), P(sgn * az, el, 0.002), (sgn * (az + 25), el - 22, float(cap_t(az, el)) + 0.004)],
                          0.026, 0.0028, tip=0.04, bend=0.35))
    for az in (-30, -10, 12, 32):  # short bangs
        L.append(lock([P(az * 0.5, 66), P(az, 44, 0.007), (az * 1.15, 14, 0.011)], 0.026, 0.0028, bend=0.3))
    if FEMALE:
        L += sides("long")
    return L


def style_plumado():
    """Long and plumed: bangs, long framing locks and a narrow curtain down the back between the
    wing roots, the ends pointed like plumes (feathers are mixed in, see harpin_hair_feathers)."""
    L = [cap(6 if FEMALE else 2)] + bangs() + sides("long") + top(back_el=-10)
    end_y = 1.17 if FEMALE else 1.27
    azs = (150, 165, 180, -165, -150)
    for i, az in enumerate(azs):
        a = math.radians(az)
        x_end = 0.05 * math.sin(a) * 1.4
        mid = scalp(az, -38, 0.022)
        b = to_head(np.array([x_end * 0.9, 1.48, -0.15]))
        c = to_head(np.array([x_end, 1.38, -0.17]))
        d = to_head(np.array([x_end * 1.1, end_y + 0.03 * abs(i - 2) - 0.012 * (i % 2), -0.18]))
        L.append(lock([P(az, 36, -0.002), P(az, 6, 0.004), mid, b, c, d], 0.036, 0.008, peak=0.4, root=0.85,
                      thin=0.5, bend=0.4, rings=8))
    return L


def harpin_hair_feathers(style):
    """Feathers mixed into a Harpin hair style (Hair slot, in design space): tucked behind the ears
    and, for the crest and the long style, standing in the crest or hanging among the locks."""
    import harpin as HP
    parts = []

    def put(base, d, ln, w, out_hint, lum=1.0, tilt=8.0):
        d = m3.normalize(np.asarray(d, float))
        out = m3.normalize(np.asarray(out_hint, float) - d * np.dot(out_hint, d))
        to3d = HP.plane_map(base, d, np.cross(out, d), out)
        parts.append(HP.feather((0.0, 0.0), (ln, 0.0), np.array([0.0, 1.0]), w, to3d, round_tip=0.2, asym=0.1,
                                curve=0.04, tilt=tilt, thick=0.0024, n=7, lum=lum, tip_dark=0.15))

    for sx in (1, -1):
        m = np.array([sx, 1.0, 1.0])
        if style == "espetado":
            base = H.head_xf(scalp(sx * 128, 2, 0.0))
            put(base, np.array([0.35, -0.25, -0.9]) * m, 0.12, 0.028, np.array([1, 0, 0.2]) * m)
            put(base + np.array([0, 0.012, 0]), np.array([0.3, 0.05, -0.95]) * m, 0.1, 0.026, np.array([1, 0, 0.2]) * m, 0.92)
        elif style == "crista":
            base = H.head_xf(scalp(sx * 172, 62, 0.0))
            put(base, np.array([0.15, 0.75, -0.65]) * m, 0.15, 0.03, np.array([1, 0, 0]) * m)
            base = H.head_xf(scalp(sx * 176, 30, 0.0))
            put(base, np.array([0.12, 0.45, -0.88]) * m, 0.13, 0.028, np.array([1, 0, 0]) * m, 0.92)
        else:
            base = H.head_xf(scalp(sx * 112, -20, 0.012))
            put(base, np.array([0.12, -0.97, -0.2]) * m, 0.17 if FEMALE else 0.14, 0.03, np.array([1, 0, 0.3]) * m)
            base = H.head_xf(scalp(sx * 140, -30, 0.016))
            put(base, np.array([0.1, -0.95, -0.3]) * m, 0.15, 0.028, np.array([1, 0, -0.2]) * m, 0.92)
    out = p3.Mesh.join(parts, name="HairFeathers")
    out.set_material(slot("Hair"))
    return out


BUILDERS = {"curto": style_curto, "rabo": style_rabo, "longo": style_longo, "coque": style_coque,
            "espetado": style_espetado, "crista": style_crista, "plumado": style_plumado}


# ----------------------------------------------------------------------------- shading

def ring_field(P, N):
    """Signed field of the angel ring (degrees, < 0 inside): a zigzag band across the top of the
    head (around an axis tilted toward the face), only on surfaces facing out, near the head."""
    d = P - CRAN_C
    dist = np.linalg.norm(d, axis=1)
    dn = d / np.maximum(dist, 1e-9)[:, None]
    axis = m3.normalize(np.array([0.0, 1.0, 0.36]))
    polar = np.degrees(np.arccos(np.clip(dn @ axis, -1, 1)))
    az = np.degrees(np.arctan2(dn[:, 0], dn[:, 2]))
    tri = np.abs(((az * 9.0 / 360.0) % 1.0) * 2 - 1)  # triangle wave, 9 teeth around
    center = 47 + 6.0 * (tri - 0.5)
    half = 4.2 - 1.6 * tri
    f = np.abs(polar - center) - half
    ok = (np.einsum("ij,ij->i", N, dn) > 0.3) & (dist < 0.135) & (np.abs(az) < 155)
    return np.where(ok, f, np.maximum(f, 0.5))


def cut_iso(m: p3.Mesh, f: np.ndarray):
    """Split the (triangle) faces of `m` along the zero line of the per-vertex field f; returns
    (mesh, inside-face mask). The new vertices are shared along the cut (then split by the caller)."""
    V = m.vertices
    N = m.normals
    F = np.asarray(m.faces, dtype=np.int64).reshape(-1, 3)
    neg = f < 0
    newV, newN, newF, inside = [V], [N], [], []
    edge_v = {}
    nxt = [len(V)]

    def cross(i, j):
        key = (min(i, j), max(i, j))
        if key not in edge_v:
            t = f[i] / (f[i] - f[j])
            t = min(max(t, 0.03), 0.97)
            newV.append((V[i] + (V[j] - V[i]) * t)[None])
            nn = N[i] + (N[j] - N[i]) * t
            newN.append((nn / np.linalg.norm(nn))[None])
            edge_v[key] = nxt[0]
            nxt[0] += 1
        return edge_v[key]

    for tri in F:
        s = neg[tri]
        if s.all() or not s.any():
            newF.append(tuple(tri))
            inside.append(bool(s[0]))
            continue
        k = int(np.flatnonzero(s if s.sum() == 1 else ~s)[0])  # the lone vertex
        a, b, c = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]
        pab, pac = cross(a, b), cross(a, c)
        lone_in = bool(neg[a])
        newF += [(a, pab, pac), (pab, b, c), (pab, c, pac)]
        inside += [lone_in, not lone_in, not lone_in]
    out = p3.Mesh(np.vstack(newV), newF, name=m.name)
    out.normals = np.vstack(newN)
    out.materials = list(m.materials)
    out.face_materials = np.zeros(len(newF), dtype=np.int64)
    return out, np.array(inside)


def hull_normals(V, N, strength=0.82, hang=0.6):
    """Shading normals: the lock's own normal bent most of the way toward a smooth hull, so toon
    bands draw big clean shapes across all the locks instead of streaks per lock. On the head the
    hull is an ellipsoid around the cranium; hanging hair (ponytail, long hair) uses a vertical
    cylinder around the back of the head."""
    r = CRAN_R + 0.025
    ne = (V - CRAN_C) / (r * r)
    ne /= np.linalg.norm(ne, axis=1, keepdims=True)
    nc = V - np.array([0.0, 0.0, CRAN_C[2] - 0.02])
    nc[:, 1] = 0.0
    nc /= np.maximum(np.linalg.norm(nc, axis=1, keepdims=True), 1e-9)
    on_head = _ss(CRAN_C[1] - 0.13, CRAN_C[1] - 0.06, V[:, 1])[:, None]
    nh = ne * on_head + nc * (1 - on_head)
    nh /= np.linalg.norm(nh, axis=1, keepdims=True)
    w = (strength * on_head + hang * (1 - on_head))
    n = N * (1 - w) + nh * w
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12), nh


def shade(m: p3.Mesh) -> p3.Mesh:
    """Bake the two tones and the angel ring into the vertex colors (head space mesh)."""
    m.triangulate()
    geo = m.vertex_normals()
    m.normals, _ = hull_normals(m.vertices, geo)
    f = ring_field(m.vertices, m.normals)
    m, inside = cut_iso(m, f)
    # split the vertices on the ring's edge: crisp color edge
    F = np.asarray(m.faces, dtype=np.int64)
    used_in = np.zeros(m.num_vertices, bool)
    used_in[F[inside].ravel()] = True
    used_out = np.zeros(m.num_vertices, bool)
    used_out[F[~inside].ravel()] = True
    both = np.flatnonzero(used_in & used_out)
    remap = np.arange(m.num_vertices)
    remap[both] = m.num_vertices + np.arange(len(both))
    m.vertices = np.vstack([m.vertices, m.vertices[both]])
    m.normals = np.vstack([m.normals, m.normals[both]])
    F[inside] = remap[F[inside]]
    m.faces = [tuple(f_) for f_ in F]
    ring_v = np.zeros(m.num_vertices, bool)
    ring_v[F[inside].ravel()] = True
    V, N = m.vertices, m.normals
    _, nh = hull_normals(V, N)
    # two tones in big shapes: the lower part of the hull (under the crown's curve, the hanging
    # hair's inside) in the darker tone, with a soft step
    under = _ss(-0.05, -0.45, nh[:, 1]) * 0.85 + 0.15 * _ss(0.1, -0.6, N[:, 1])
    lum = BASE - (BASE - UNDER) * np.clip(under, 0, 1)
    lum = np.where(ring_v, RING, lum)
    m.colors = np.ones((m.num_vertices, 4))
    m.colors[:, :3] = lum[:, None]
    return m


def build_style(style: str) -> p3.Mesh:
    """The style's mesh in design space (Hair material, ties in Leather), shaded. Harpin feathers
    are added by the caller."""
    parts = BUILDERS[style]()
    hair = [p for p in parts if p.name != "tie"]
    ties = [p for p in parts if p.name == "tie"]
    m = p3.Mesh.join(hair, name=f"Hair_{style}")
    m.set_material(slot("Hair"))
    m = shade(m)
    for t in ties:
        t.triangulate()
        t.normals = t.vertex_normals()
        t.colors = np.ones((t.num_vertices, 4))
        m.merge(t)
    m.vertices = H.head_xf(m.vertices)
    m.name = f"Hair_{style}"
    return m
