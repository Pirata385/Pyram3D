"""Hair_curto: short stylized hair as a smooth cap with a few big locks (Hair slot).

The cap is an offset of the cranium (thicker at the crown) cut along a hairline (forehead,
temples, above the ears, nape). Locks are flattened tapering tubes lying on it: a swept-up front
over the forehead, locks combed back over the crown, short pointed locks at the back and sides.
They meet the cap through small blends, so each lock reads as a clean clump with a soft crease.
Modeled in head space (unscaled), then scaled with the head.
"""
from __future__ import annotations

import numpy as np

import body as B
from common import slot
from shapes import Offset, Tube

from pyram3d import math3d as m3
from pyram3d import sdf

CRAN_C = np.array([0.0, 1.688, -0.016])


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _onto(p, lift):
    """Point p moved radially (from the cranium center) to `lift` above the cranium surface."""
    p = np.asarray(p, float)
    d = p - CRAN_C
    r = np.array([0.082, 0.097, 0.102])
    k = np.linalg.norm(d / r)
    return CRAN_C + d / k + m3.normalize(d) * lift


def lock(points, width, thick, lift=0.012):
    """A clump of hair along points (root ... tip) laid over the skull `lift` m above it: rounded
    section, flattened against the head, widest a third of the way, tapering to a point."""
    P = [_onto(p, lift) for p in points[:-1]] + [np.asarray(points[-1], float)]
    seg = [np.linalg.norm(b - a) for a, b in zip(P, P[1:])]
    tot = sum(seg)
    parts, acc = [], 0.0
    for a, b, L in zip(P, P[1:], seg):
        u0, u1 = acc / tot, (acc + L) / tot
        acc += L
        prof_w = lambda u: width * (0.75 + 0.25 * np.sin(np.pi * min(u, 0.6) / 1.2)) * (1 - 0.85 * max(0, (u - 0.35) / 0.65) ** 1.3)  # noqa: E731,E501
        prof_t = lambda u: thick * (1 - 0.7 * max(0, (u - 0.35) / 0.65) ** 1.3)  # noqa: E731
        out = m3.normalize(0.5 * (a + b) - CRAN_C)
        parts.append(Tube(a, b, {0: prof_w(u0), 1: prof_w(u1)}, {0: prof_t(u0), 1: prof_t(u1)}, front=out,
                          cap=min(prof_t(u1), prof_w(u1)) * 0.9))
    return sdf.smooth_union(*parts, k=0.004) if len(parts) > 1 else parts[0]


def cap():
    cran = sdf.ellipsoid(CRAN_C, (0.082, 0.097, 0.102))

    def t(p):
        return 0.008 + 0.016 * _smoothstep(1.67, 1.775, p[:, 1])

    c = Offset(cran, t, t_max=0.026)
    # hairline: forehead, temples/above the ears down to the nape
    c = c.smooth_intersect(sdf.plane((0, 1.745, 0.07), m3.normalize((0, -0.5, 1)), extent=1.0), 0.012)
    c = c.smooth_intersect(sdf.plane((0, 1.664, 0.0), m3.normalize((0, -1, 0.55)), extent=1.0), 0.012)
    return c


def locks():
    L = []
    # fringe: from the top, forward and down over the hairline, pointed tips
    for x, tipy in ((-0.045, 1.73), (-0.016, 1.722), (0.014, 1.727), (0.043, 1.735)):
        L.append(lock([(x * 0.6, 1.79, -0.01), (x * 0.9, 1.785, 0.05), (x * 1.08, tipy, 0.1)], 0.024, 0.012))
    # crown: back and down
    for x in (-0.05, -0.017, 0.017, 0.05):
        L.append(lock([(x * 0.5, 1.79, 0.0), (x, 1.76, -0.08), (x * 1.08, 1.69, -0.118)], 0.026, 0.012))
    # back: down to pointed tips at the nape
    for x in (-0.05, -0.016, 0.018, 0.052):
        L.append(lock([(x, 1.72, -0.1), (x * 1.02, 1.66, -0.11), (x * 1.04, 1.618, -0.104)], 0.022, 0.01))
    # sides: back over the temples, above the ears
    for s in (1, -1):
        L.append(lock([(s * 0.05, 1.775, 0.03), (s * 0.08, 1.735, -0.01), (s * 0.088, 1.69, -0.06)], 0.024, 0.011))
        L.append(lock([(s * 0.065, 1.75, 0.045), (s * 0.08, 1.71, 0.03), (s * 0.081, 1.675, 0.026)], 0.014, 0.008))
    return sdf.union(*L)


def hair_sdf():
    shape = cap().smooth_union(locks(), 0.007)
    shape = shape.paint(color=(1, 1, 1), material=slot("Hair"))
    M = m3.translation_matrix(B.HEAD_PIVOT) @ m3.scale_matrix(B.HEAD_SCALE) @ m3.translation_matrix(-B.HEAD_PIVOT)
    return sdf.Transformed(shape, M)
