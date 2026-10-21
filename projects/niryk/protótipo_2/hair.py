"""Anime hair: four styles (Hair_curto, Hair_rabo, Hair_longo, Hair_coque) for each body.

Every style is a volume cap (an offset of the cranium cut along a hairline) covered by defined
locks (shapes.Strand: a smooth sweep with a lens section) that end in pointed tips: bangs framing
the face, side locks, and the style's back (nape locks, a ponytail, a long curtain or a bun).
Locks meet the cap and each other through small blends, so the grooves between them stay crisp
(they read as ink lines in the toon look). The "angel ring" highlight and the darker undersides
are vertex-color luminance (the client multiplies the hair color).

Locks on the head are laid out in (azimuth, elevation, lift) over the cranium (azimuth 0 = front,
+90 = the character's left; elevation 0 = the cranium center's height); hanging parts (ponytail,
long hair) are placed in design space, behind the back, and converted to head space.
"""
from __future__ import annotations

import math

import numpy as np

import head as H
from common import FEMALE, slot
from shapes import Offset, Strand

from pyram3d import math3d as m3
from pyram3d import sdf

STYLES = ("curto", "rabo", "longo", "coque")
CRAN_C, CRAN_R = H.CRAN_C, H.CRAN_R


def _ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def scalp(az, el, lift=0.0):
    """Head-space point `lift` m above the cranium at azimuth/elevation (degrees)."""
    a, e = math.radians(az), math.radians(el)
    d = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    r = 1.0 / np.linalg.norm(d / CRAN_R)
    return CRAN_C + d * (r + lift)


def to_head(p):
    """Design-space point -> head space (the hair is scaled with the head)."""
    return H.HEAD_PIVOT + (np.asarray(p, float) - H.HEAD_PIVOT) / H.HEAD_SCALE


def cap_t(az, el):
    """Cap thickness (volume of the hair over the scalp)."""
    top = _ss(10, 70, el)
    back = _ss(90, 160, abs(((az + 180) % 360) - 180))
    base = 0.011 if FEMALE else 0.009
    return base + (0.01 if FEMALE else 0.009) * top + 0.002 * back


def P(az, el, extra=0.0):
    """(az, el, lift) just above the cap."""
    return (az, el, cap_t(az, el) + extra)


def lock(pts, w, f, k=None, root=0.8, peak=0.3, tip=0.04, thin=0.7, samples=6):
    """Pointed lock through pts: (az, el, lift) tuples on the head or np.array head-space points.
    w: max half width (reached at `peak` of the length), f: outward half thickness, k: inward."""
    Q = np.array([p if isinstance(p, np.ndarray) else scalp(*p) for p in pts])
    Q = m3.catmull_rom(Q, samples)
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    grow = root + (1 - root) * _ss(0, peak, u)
    taper = 1 - (1 - tip) * np.clip((u - peak) / (1 - peak), 0, 1) ** 1.25
    W = w * grow * taper
    F = f * (1 - thin * np.clip((u - 0.45) / 0.55, 0, 1) ** 1.2)
    out = Q - CRAN_C
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return Strand(Q, out, W, F, F * (0.6 if k is None else k / f))


# ----------------------------------------------------------------------------- cap

def _hairline(lo_front):
    """Elevation (deg) of the hairline at each azimuth: forehead, temples, above and behind the
    ears, nape. lo_front lowers it at the front (styles with long bangs)."""
    az = np.array([0, 40, 65, 88, 110, 135, 160, 180])
    el = np.array([28 - lo_front, 20 - lo_front, 2, -8, -18, -34, -44, -46])
    return az, el


def cap(lo_front=0.0):
    cran = sdf.ellipsoid(CRAN_C, tuple(CRAN_R))

    def thick(p):
        d = np.atleast_2d(p) - CRAN_C
        az = np.degrees(np.arctan2(d[:, 0], d[:, 2]))
        el = np.degrees(np.arcsin(np.clip(d[:, 1] / np.maximum(np.linalg.norm(d, axis=1), 1e-9), -1, 1)))
        return cap_t(az, el)
    c = Offset(cran, thick, t_max=0.03)
    A, E = _hairline(lo_front)

    def line(p):
        d = np.atleast_2d(p) - CRAN_C
        az = np.abs(np.degrees(np.arctan2(d[:, 0], d[:, 2])))
        el = np.degrees(np.arcsin(np.clip(d[:, 1] / np.maximum(np.linalg.norm(d, axis=1), 1e-9), -1, 1)))
        return (np.interp(az, A, E) - el) * math.radians(1) * 0.1  # ~ meters below the line
    cut = sdf.custom(line, CRAN_C - 0.2, CRAN_C + 0.2)
    return c.smooth_intersect(cut, 0.008)


# ----------------------------------------------------------------------------- shared parts

def bangs():
    """Fringe: pointed locks from the front of the crown over the forehead; the longest falls
    between the eyes. Female: softer, longer, side-swept, with a lock framing each eye."""
    L = []
    if FEMALE:
        # side-swept from a parting on her right: long soft locks, one sweeping across the forehead
        spec = [(-66, -56, -2, 0.022), (-46, -40, -8, 0.026), (-26, -22, -13, 0.026), (-6, -4, -9, 0.024),
                (14, 14, -12, 0.026), (34, 32, -6, 0.026), (54, 48, -2, 0.024), (70, 60, 2, 0.02)]
        for r, t, e, w in spec:
            L.append(lock([P(r * 0.5 - 20, 76), P(r * 0.8 - 6, 58, 0.008), P((r + t) / 2, 38, 0.009), P(t, 16, 0.007),
                           (t, e, 0.006)], w, 0.0062, k=0.004, tip=0.02, peak=0.4))
        L.append(lock([P(-34, 78, 0.002), P(-14, 62, 0.012), P(10, 42, 0.013), (30, 22, cap_t(30, 22) + 0.012),
                       (40, 4, 0.01)], 0.03, 0.0065, k=0.004, tip=0.03, peak=0.35))
    else:
        spec = [(-60, -48, 7, 0.024), (-36, -28, -2, 0.032), (-12, -8, -12, 0.028), (12, 11, -4, 0.032),
                (35, 28, 4, 0.028), (58, 47, 9, 0.022)]
        for r, t, e, w in spec:
            L.append(lock([P(r * 0.6, 76), P(r * 0.85, 58, 0.009), P((r + t) / 2, 38, 0.01), P(t, 18, 0.008),
                           (t, e, 0.006)], w, 0.0075, k=0.005, tip=0.03))
        # a second, shorter layer between them breaks the comb look
        for r, t, e in ((-24, -18, 8), (0, 1, 3), (24, 20, 10)):
            L.append(lock([P(r * 0.6, 74, 0.002), P(r * 0.85, 56, 0.013), P((r + t) / 2, 38, 0.013), (t, 20, 0.014),
                           (t, e, 0.0105)], 0.02, 0.006, k=0.004, tip=0.03))
    return L


def sides(length="short"):
    """Locks in front of and over the ears. length: short (ear), jaw, long (collarbone)."""
    L = []
    for s in (1, -1):
        if length == "short":
            L.append(lock([P(s * 66, 58), P(s * 74, 34, 0.006), P(s * 79, 8, 0.006), (s * 80, -20, 0.006)],
                          0.022, 0.0065, tip=0.03))
            L.append(lock([P(s * 98, 52), P(s * 104, 26, 0.008), P(s * 107, 2, 0.013), (s * 109, -18, 0.018)],
                          0.026, 0.0075, tip=0.03))
            L.append(lock([P(s * 122, 50), P(s * 126, 22, 0.007), P(s * 128, -6, 0.01), (s * 130, -30, 0.016)],
                          0.026, 0.0075, tip=0.03))
        else:
            y_tip = {"jaw": 1.575, "long": 1.5}[length]
            # framing lock in front of the ear, falling past the cheek
            tip = to_head(np.array([s * 0.088, y_tip, 0.035]))
            mid = scalp(s * 80, -30, 0.012)
            L.append(lock([P(s * 64, 58), P(s * 74, 30, 0.007), P(s * 80, 0, 0.008), mid, tip], 0.024, 0.0068,
                          tip=0.03))
            tip2 = to_head(np.array([s * 0.1, y_tip + 0.02, -0.005]))
            L.append(lock([P(s * 96, 54), P(s * 102, 24, 0.008), P(s * 104, -6, 0.014), scalp(s * 102, -32, 0.02),
                           tip2], 0.026, 0.0072, tip=0.03))
            tip3 = to_head(np.array([s * 0.095, y_tip + 0.035, -0.05]))
            L.append(lock([P(s * 122, 52), P(s * 126, 22, 0.008), P(s * 128, -8, 0.012), scalp(s * 126, -34, 0.016),
                           tip3], 0.028, 0.0072, tip=0.03))
    return L


def top(back_az=180, back_el=10):
    """Broad locks from the crown whorl: forward into the bangs and down the back toward
    (back_az, back_el) (the nape, a ponytail tie or a bun)."""
    L = []
    for az in (-140, -165, 165, 140, -115, 115):
        a1 = back_az + (az - math.copysign(180, az)) * 0.35
        L.append(lock([P(az * 0.3 + 126 * math.copysign(1, az), 82, 0.0), P(az, 55, 0.004),
                       P((az + a1) / 2, 30, 0.004), P(a1, back_el, 0.002)], 0.026, 0.01, k=0.008, tip=0.5,
                      thin=0.25, root=0.9))
    return L


def nape(length=0.0):
    """Pointed locks over the back of the head in two layers (curto); `length` lowers the
    lower layer's tips (meters)."""
    L = []
    azs = (125, 145, 165, -175, -155, -135) if not FEMALE else (120, 138, 156, 174, -168, -150, -132)
    for i, az in enumerate(azs):
        el_tip = -46 - 5 * (i % 2)
        tip = scalp(az, el_tip, 0.012)
        if length:
            tip = tip + np.array([0, -length / H.HEAD_SCALE, -0.008])
        L.append(lock([P(az, 58), P(az, 24, 0.005), P(az, -14, 0.008), scalp(az, -34, 0.013), tip], 0.022, 0.011,
                      k=0.007, tip=0.03))
    for i, az in enumerate(np.array(azs[:-1]) + (10 if not FEMALE else 9)):  # upper layer, shorter
        az = float((az + 180) % 360 - 180)
        L.append(lock([P(az, 64, 0.003), P(az, 34, 0.009), P(az, 4, 0.011), (az, -16 - 6 * (i % 2), cap_t(az, -16) + 0.019)],
                      0.02, 0.0105, k=0.007, tip=0.03))
    return L


# ----------------------------------------------------------------------------- styles

def _hair(shape):
    return shape.paint(color=(1, 1, 1), material=slot("Hair"))


def style_curto():
    L = bangs() + top()
    if FEMALE:  # chin-length bob: side locks to the jaw, the back to the nape curling in
        L += sides("jaw") + nape(0.03)
        return _hair(cap(6).smooth_union(sdf.smooth_union(*L, k=0.003), 0.006))
    L += sides("short") + nape()
    return _hair(cap().smooth_union(sdf.smooth_union(*L, k=0.003), 0.006))


def tie_point():
    return scalp(180, 34 if FEMALE else 30, 0.02)


def style_rabo():
    """High ponytail: everything combed back to a tie at the back of the crown; the tail falls
    behind the back to the shoulder blades (female: lower)."""
    L = bangs() + sides("jaw" if FEMALE else "short") + top(back_el=34)
    T = tie_point()
    end_y = 1.2 if FEMALE else 1.34
    # the tail's axis in design space: out of the tie, arcing back, then falling behind the back
    Td = H.head_xf(T)
    axis = [Td + np.array([0, 0.004, -0.014]), np.array([0, Td[1] - 0.035, -0.205]), np.array([0, 1.56, -0.205]),
            np.array([0, 1.44, -0.19]), np.array([0, end_y, -0.17])]
    spread = [0.006, 0.017, 0.021, 0.019, 0.015]  # bundle radius along the tail
    tail = []
    n_locks = 6 if FEMALE else 5
    for i in range(n_locks):
        th = 2 * math.pi * i / n_locks + 0.4
        pts = []
        for j, (A, r) in enumerate(zip(axis, spread)):
            off = np.array([math.cos(th) * r * 1.1, 0.0, math.sin(th) * r * 1.15])
            if j == len(axis) - 1:  # tips at different lengths, splaying a little
                off = off * 1.6 + np.array([0, 0.03 * ((i * 7) % 3) / 2, 0])
            pts.append(to_head(A + off) if j else T + (A + off - Td) / H.HEAD_SCALE)
        tail.append(lock(pts, 0.016 if FEMALE else 0.015, 0.014, k=0.013, peak=0.35, tip=0.03, root=0.7, thin=0.45))
    tie = sdf.torus((0, 0, 0), 0.016, 0.0065)
    M = m3.translation_matrix(T + np.array([0, 0.004, -0.012])) @ m3.rotation_matrix(m3.quat_from_axis_angle((1, 0, 0), 75))
    tie = sdf.Transformed(tie, M).paint(color=(1, 1, 1), material=slot("Leather"))
    hair = cap().smooth_union(sdf.smooth_union(*(L + tail), k=0.003), 0.006)
    return _hair(hair).union(tie)


def style_longo():
    """Long hair: a curtain of locks from the crown down the back (female: to mid-back), long
    framing locks at the sides."""
    L = bangs() + sides("long") + top(back_el=-10)
    end_y = 1.17 if FEMALE else 1.3
    azs = (128, 146, 164, 180, -164, -146, -128)
    for i, az in enumerate(azs):
        a = math.radians(az)
        x_end = 0.088 * math.sin(a) * 1.25
        mid = scalp(az, -38, 0.022)
        b = to_head(np.array([x_end * 0.95, 1.5, -0.165 + 0.04 * abs(math.cos(a)) * 0]))
        c = to_head(np.array([x_end, 1.4, -0.168]))
        d = to_head(np.array([x_end * 1.02, end_y + 0.035 * abs(i - 3) ** 0.8 - 0.012 * (i % 2), -0.158]))
        L.append(lock([P(az, 40), P(az, 6, 0.004), mid, b, c, d], 0.034, 0.011, k=0.008, peak=0.4, tip=0.03,
                      root=0.85, thin=0.5))
    return _hair(cap(6 if FEMALE else 0).smooth_union(sdf.smooth_union(*L, k=0.003), 0.006))


def style_coque():
    """Bun: everything combed back and up into a bun at the back of the head, a tie under it, a
    few loose strands at the nape."""
    L = bangs() + sides("jaw" if FEMALE else "short") + top(back_el=22)
    c = scalp(180, 24, 0.042 if FEMALE else 0.036)
    r = 0.039 if FEMALE else 0.034
    nb = m3.normalize(c - CRAN_C)  # the bun's axis (out of the head)
    e1 = m3.normalize(np.cross(nb, [1.0, 0, 0]))
    e2 = np.cross(nb, e1)
    bun = sdf.ellipsoid(c, (r, r * 0.96, r * 0.96))
    # spiral locks wrapping the bun from its base toward its pole
    sw = []
    for k in range(4):
        pts = []
        for t in np.linspace(0, 1, 9):
            pol = math.radians(85 - 70 * t)
            az = math.radians(90 * k + 260 * t)
            d = math.cos(pol) * nb + math.sin(pol) * (math.cos(az) * e1 + math.sin(az) * e2)
            pts.append(c + d * (r * 0.97))
        Q = m3.catmull_rom(np.array(pts), 5)
        u = np.linspace(0, 1, len(Q))
        out = Q - c
        out /= np.linalg.norm(out, axis=1, keepdims=True)
        W = (0.019 if FEMALE else 0.016) * (0.8 + 0.2 * np.sin(np.pi * np.minimum(u, 0.5))) * (1 - 0.92 * u ** 1.4)
        sw.append(Strand(Q, out, W, 0.0095 * (1 - 0.45 * u), 0.007 * np.ones(len(Q))))
    bun = bun.smooth_union(sdf.smooth_union(*sw, k=0.002), 0.003)
    tie = sdf.torus((0, 0, 0), r * 0.66, 0.0055)
    q = m3.quat_from_axis_angle(m3.normalize(np.cross([0, 1.0, 0], nb)), math.degrees(math.acos(float(np.clip(nb[1], -1, 1)))))
    M = m3.translation_matrix(c - nb * r * 0.62) @ m3.rotation_matrix(q)
    tie = sdf.Transformed(tie, M).paint(color=(1, 1, 1), material=slot("Leather"))
    loose = []
    for s in (1, -1):
        loose.append(lock([P(s * 150, 10), scalp(s * 152, -20, 0.008), scalp(s * 150, -42, 0.008),
                           scalp(s * 146, -56, 0.004)], 0.012, 0.005, tip=0.03))
    hair = cap().smooth_union(sdf.smooth_union(*(L + loose), k=0.003), 0.006).smooth_union(bun, 0.012)
    return _hair(hair).union(tie)


BUILDERS = {"curto": style_curto, "rabo": style_rabo, "longo": style_longo, "coque": style_coque}


def hair_sizing(style: str):
    """max_edge field: finer on the bun and the ties, coarse on the broad cap."""
    from meshing import sizing
    regions = []
    if style == "coque":
        c = H.head_xf(scalp(180, 24, 0.042 if FEMALE else 0.036))
        regions.append((lambda P, c=c: np.linalg.norm(P - c, axis=1) < 0.06, 0.011))
    if style == "rabo":
        t = H.head_xf(tie_point())
        regions.append((lambda P, t=t: np.linalg.norm(P - t, axis=1) < 0.04, 0.012))
    regions.append((lambda P: P[:, 1] < H.head_xf(CRAN_C)[1] - 0.1, 0.022))  # hanging hair
    return sizing(regions, 0.03)


def hair_sdf(style: str):
    return sdf.Transformed(BUILDERS[style](), H.head_matrix())


def hair_head_sdf(style: str):
    """The style in head space (for coverage tests)."""
    return BUILDERS[style]()


# ----------------------------------------------------------------------------- paint

RING_TMP = None  # temporary material of the angel-ring faces (merged back into Hair)


def _ring_tmp():
    global RING_TMP
    if RING_TMP is None:
        import pyram3d as p3
        RING_TMP = p3.Material("HairRingTmp", "#ffffff")
    return RING_TMP


def ring_mask(P, N):
    """Points on the angel ring: a zigzag band across the top of the head (around an axis tilted
    toward the face), on surfaces facing out, near the head (not on hanging hair)."""
    c = H.head_xf(CRAN_C)
    d = P - c
    dist = np.linalg.norm(d, axis=1)
    dn = d / np.maximum(dist, 1e-9)[:, None]
    axis = m3.normalize(np.array([0.0, 1.0, 0.34]))
    polar = np.degrees(np.arccos(np.clip(dn @ axis, -1, 1)))
    az = np.degrees(np.arctan2(dn[:, 0], dn[:, 2]))
    tri = np.abs(((az * 9.0 / 360.0) % 1.0) * 2 - 1)  # triangle wave, 9 teeth around
    center = 46 + 7.0 * (tri - 0.5)
    half = 5.0 - 1.5 * tri
    out = np.einsum("ij,ij->i", N, dn) > 0.35
    return (np.abs(polar - center) < half) & out & (dist < 0.15) & (np.abs(az) < 150)


def mark_ring(m):
    """before_decimate hook: ring faces get a temporary material (the decimation keeps its
    zigzag border)."""
    fc = m.face_centers()
    sel = np.flatnonzero(ring_mask(fc, m.face_normals()))
    if len(sel):
        m.set_material(_ring_tmp(), sel)
    return m


def finish_ring(m):
    """Ring faces get their own vertices (a crisp edge in the vertex colors) and the Hair
    material back; returns the vertex mask of the ring."""
    mats = m.materials or []
    ring_idx = [i for i, mt in enumerate(mats) if mt is RING_TMP]
    if not ring_idx:
        m.extras["ring_v"] = np.zeros(m.num_vertices, bool)
        return m
    sel = np.flatnonzero(np.isin(m.face_materials, ring_idx))
    part = m.separate(sel, name="ring")
    m.delete_faces(sel)
    n0 = m.num_vertices
    part.set_material(slot("Hair"), np.arange(len(part.faces)))
    m.merge(part)
    m.materials = [mt for mt in m.materials]
    # drop the temporary material from the list
    keep = [i for i, mt in enumerate(m.materials) if mt is not RING_TMP]
    remap = {old: new for new, old in enumerate(keep)}
    m.face_materials = np.array([remap[i] for i in m.face_materials], dtype=np.int64)
    m.materials = [m.materials[i] for i in keep]
    mask = np.zeros(m.num_vertices, bool)
    mask[n0:] = True
    m.extras["ring_v"] = mask
    return m


def paint_hair(m, style: str):
    """Vertex-color luminance (the hair color is the tint): the "angel ring" highlight (its own
    vertices, see finish_ring), darker undersides, inner surfaces and grooves between locks."""
    from meshing import cavity
    V = m.vertices
    N = m.normals if m.normals is not None else m.vertex_normals()
    c = H.head_xf(CRAN_C)
    d = V - c
    dist = np.linalg.norm(d, axis=1)
    dn = d / np.maximum(dist, 1e-9)[:, None]
    ring = m.extras.get("ring_v", np.zeros(len(V), bool))
    lum = np.where(ring, 1.0, 0.66)
    near = _ss(0.16, 0.12, dist)
    under = _ss(0.25, -0.5, N[:, 1])
    lum *= 1 - 0.16 * under
    inner = _ss(0.1, -0.3, np.einsum("ij,ij->i", N, dn)) * near
    lum *= 1 - 0.18 * inner
    lum *= 1 - 0.38 * _ss(0.08, 0.35, cavity(m, smooth=4))  # grooves between locks
    if m.colors is None:
        m.colors = np.ones((m.num_vertices, 4))
    hair_mat = [i for i, mt in enumerate(m.materials or []) if mt is slot("Hair")]
    is_hair_v = np.zeros(m.num_vertices, bool)
    F = np.asarray(m.faces, dtype=np.int64).reshape(-1, 3)
    if hair_mat:
        is_hair_v[np.unique(F[np.isin(m.face_materials, hair_mat)])] = True
    m.colors[is_hair_v, :3] = lum[is_hair_v, None]
    return m


def hair_normals(m):
    """Bend the normals of the hair on the head halfway toward a smooth hull around it, so toon
    shading draws clean shapes (the grooves between locks still show as ink and ring breaks)."""
    c = H.head_xf(CRAN_C)
    r = (CRAN_R + 0.02) * H.HEAD_SCALE
    ne = (m.vertices - c) / (r * r)
    ne /= np.linalg.norm(ne, axis=1, keepdims=True)
    dist = np.linalg.norm(m.vertices - c, axis=1)
    top = _ss(c[1] - 0.04, c[1] + 0.06, m.vertices[:, 1])
    w = ((0.5 + 0.3 * top) * _ss(0.17, 0.13, dist))[:, None]
    n = m.normals * (1 - w) + ne * w
    m.normals = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return m
