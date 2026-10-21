"""Hair_curto: short stylized hair as a volume cap covered by chunky locks (Hair slot).

The cap is an offset of the cranium (thick on top, thin at the sides and nape) cut along a
hairline. Over it lie big clumps: a quiff that rises from the forehead and sweeps back, locks
combed back over the crown, short locks swept back over the temples and layered locks at the back
ending in blunt tips at the nape. Each clump is a tapering tube with a lens-shaped section laid
on the scalp; clumps meet the cap and each other through small blends, so the creases between
them stay crisp (they read as ink lines in the toon look) and the surface never shows big flat
patches. Modeled in head space (unscaled), then scaled with the head.
"""
from __future__ import annotations

import math

import numpy as np

import body as B
from common import slot
from shapes import Offset, Strand

from pyram3d import math3d as m3
from pyram3d import sdf

CRAN_C = np.array([0.0, 1.688, -0.016])
CRAN_R = np.array([0.082, 0.097, 0.102])


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def scalp(az, el, lift=0.0):
    """Point `lift` m above the cranium at azimuth `az` (deg around Y, 0 = front, 90 = the
    character's left) and elevation `el` (deg, 0 = level with the cranium center)."""
    a, e = math.radians(az), math.radians(el)
    d = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
    r = 1.0 / np.linalg.norm(d / CRAN_R)
    return CRAN_C + d * (r + lift)


def lock(path, width, thick, taper=0.86, inner=0.55, tip=None):
    """A clump along path [(az, el, lift), ...] (root ... tip) on the scalp, plus an optional
    free `tip` point: lens-shaped section `width` wide and `thick` thick outward (`inner` x thick
    toward the scalp), widest a third of the way, tapering to a blunt point."""
    P = np.array([scalp(*p) for p in path] + ([np.asarray(tip, float)] if tip is not None else []))
    P = m3.catmull_rom(P, 6)
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    w = width * (0.8 + 0.2 * np.sin(np.pi * np.minimum(u, 0.6) / 1.2)) * (1 - taper * np.clip((u - 0.4) / 0.6, 0, 1) ** 1.5)
    f = thick * (1 - 0.62 * np.clip((u - 0.45) / 0.55, 0, 1) ** 1.4)
    out = P - CRAN_C
    out /= np.linalg.norm(out, axis=1, keepdims=True)
    return Strand(P, out, w, f, f * inner)


def cap_thickness(p):
    p = np.atleast_2d(p)
    top = _smoothstep(1.66, 1.77, p[:, 1])
    front = _smoothstep(-0.02, 0.06, p[:, 2])
    return 0.006 + 0.012 * top + 0.004 * top * front


def cap():
    cran = sdf.ellipsoid(CRAN_C, tuple(CRAN_R))
    c = Offset(cran, cap_thickness, t_max=0.03)
    # hairline: forehead, temples/above the ears down to the nape
    c = c.smooth_intersect(sdf.plane((0, 1.748, 0.07), m3.normalize((0, -0.45, 1)), extent=1.0), 0.01)
    c = c.smooth_intersect(sdf.plane((0, 1.664, 0.0), m3.normalize((0, -1, 0.55)), extent=1.0), 0.012)
    return c


def on_cap(az, el, extra=0.0):
    """(az, el, lift) of a point `extra` m above the cap surface."""
    return (az, el, float(cap_thickness(scalp(az, el))[0]) + extra)


def locks():
    L = []
    # quiff: clumps rise from the forehead and sweep back and to the right, the middle one highest
    for az, w, lift, d in ((-32, 0.04, 0.017, -1), (-8, 0.044, 0.022, -1), (17, 0.046, 0.024, -1), (40, 0.04, 0.018, -1)):
        L.append(lock([(az, 31, 0.004), (az + d * 3, 42, lift), (az + d * 8, 56, lift * 1.1), (az + d * 14, 71, lift * 0.9),
                       (az + d * 20, 85, lift * 0.7)], w, 0.014))
    # nape: short layered clumps ending in blunt tips below the hairline
    for az, w in ((-150, 0.04), (-174, 0.042), (164, 0.042), (140, 0.04), (-126, 0.034), (116, 0.034)):
        L.append(lock([(az, 18, 0.007), (az, 2, 0.006), (az, -13, 0.005), (az, -23, 0.003)], w, 0.01))
    # crown and back: broad flat clumps fanning from the crown, overlapping like shingles
    for az, w in ((-146, 0.056), (180, 0.06), (146, 0.056)):
        a0 = az * 0.6 if abs(az) < 179 else 180
        L.append(lock([on_cap(a0, 84, -0.004), on_cap(az * 0.9 if abs(az) < 179 else 180, 66, -0.003),
                       on_cap(az, 44, -0.003), on_cap(az, 22, -0.003), on_cap(az, 6, -0.003)], w, 0.008, inner=1.0,
                      taper=0.8))
    # temples: swept back over the sides
    for s in (1, -1):
        L.append(lock([(s * 60, 50, 0.014), (s * 80, 40, 0.011), (s * 102, 30, 0.009), (s * 122, 18, 0.006)],
                      0.04, 0.011))
    return sdf.smooth_union(*L, k=0.0025)


def hair_sdf():
    shape = cap().smooth_union(locks(), 0.0065)
    shape = shape.paint(color=(1, 1, 1), material=slot("Hair"))
    M = m3.translation_matrix(B.HEAD_PIVOT) @ m3.scale_matrix(B.HEAD_SCALE) @ m3.translation_matrix(-B.HEAD_PIVOT)
    return sdf.Transformed(shape, M)
