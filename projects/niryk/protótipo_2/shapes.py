"""Smooth SDF building blocks for the prototype 2 characters.

`Tube` is a generalized cylinder: a straight axis with a superellipse cross-section whose width,
front depth, back depth, roundness and center offsets follow smooth profiles along the axis.
One tube draws a whole limb (deltoid -> biceps -> elbow -> forearm -> wrist) or the torso
(hips -> waist -> ribcage) as a single continuous surface, so the silhouette comes from curves,
not from blobs blended together (prototype 1's lumps).

`slab` / `wedge` intersect half-spaces with a smooth (rounded) max, for angular-but-smooth forms
(jaw, nose, brow, boots soles).
"""
from __future__ import annotations

import numpy as np

from pyram3d import math3d as m3
from pyram3d import sdf


class Profile:
    """Smooth 1D curve through keys {t: value} (cubic Hermite, Catmull-Rom tangents, flat ends)."""

    def __init__(self, keys):
        if isinstance(keys, (int, float)):
            keys = {0.0: float(keys), 1.0: float(keys)}
        items = sorted((float(t), float(v)) for t, v in dict(keys).items())
        self.t = np.array([a for a, _ in items])
        self.v = np.array([b for _, b in items])
        n = len(self.t)
        m = np.zeros(n)
        for i in range(1, n - 1):
            m[i] = (self.v[i + 1] - self.v[i - 1]) / (self.t[i + 1] - self.t[i - 1])
        self.m = m

    def __call__(self, t):
        """(value, derivative) at t (array)."""
        t = np.asarray(t, float)
        if len(self.t) == 1:
            return np.full_like(t, self.v[0]), np.zeros_like(t)
        tc = np.clip(t, self.t[0], self.t[-1])
        i = np.clip(np.searchsorted(self.t, tc, side="right") - 1, 0, len(self.t) - 2)
        t0, t1 = self.t[i], self.t[i + 1]
        dt = t1 - t0
        u = (tc - t0) / dt
        v0, v1 = self.v[i], self.v[i + 1]
        m0, m1 = self.m[i] * dt, self.m[i + 1] * dt
        u2, u3 = u * u, u * u * u
        val = (2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * m0 + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * m1
        der = ((6 * u2 - 6 * u) * v0 + (3 * u2 - 4 * u + 1) * m0 + (-6 * u2 + 6 * u) * v1 + (3 * u2 - 2 * u) * m1) / dt
        der = np.where((t < self.t[0]) | (t > self.t[-1]), 0.0, der)
        return val, der


class Tube(sdf.SDF):
    """Straight generalized cylinder from a to b.

    Profiles (dicts {t: value}, t in [0, 1] from a to b, or constants):
      w  half width along the lateral axis (l = front x axis)
      f  half depth toward `front`;  k  half depth toward -front (defaults to f)
      n  superellipse exponent (2 = ellipse, 3-4 = boxier)
      cf center offset toward front;  cl center offset along l
    cap: rounding radius of the two ends (they are flat discs rounded by `cap`).
    """

    def __init__(self, a, b, w, f, k=None, n=2.0, cf=0.0, cl=0.0, front=(0, 0, 1), cap=0.02,
                 color=None, material=None, bone=None):
        self.color, self.material, self.bone = color, material, bone
        self.a = m3.vec3(a)
        self.b = m3.vec3(b)
        ax = self.b - self.a
        self.L = float(np.linalg.norm(ax))
        self.u = ax / self.L
        fr = m3.vec3(front)
        fr = fr - self.u * np.dot(fr, self.u)
        self.fw = fr / np.linalg.norm(fr)
        self.l = np.cross(self.u, self.fw)
        self.w, self.f = Profile(w), Profile(f)
        self.k = Profile(k if k is not None else f)
        self.n, self.cf, self.cl = Profile(n), Profile(cf), Profile(cl)
        self.cap = float(cap)

    def dist(self, p):
        q = np.asarray(p, float) - self.a
        s = q @ self.u
        t = s / self.L
        w, dw = self.w(t)
        f, df = self.f(t)
        k, dk = self.k(t)
        n, _ = self.n(t)
        cf, dcf = self.cf(t)
        cl, dcl = self.cl(t)
        x = q @ self.l - cl
        y = q @ self.fw - cf
        front = y > 0
        ry = np.where(front, f, k)
        dry = np.where(front, df, dk)
        rc = self.cap
        w = np.maximum(w, 1e-4)
        ry = np.maximum(ry, 1e-4)
        ax_, ay_ = np.abs(x) / w + 1e-12, np.abs(y) / ry + 1e-12
        pn = ax_ ** n + ay_ ** n
        K = pn ** (1.0 / n)
        c = K ** (1 - n)
        gx = c * ax_ ** (n - 1) * np.sign(x) / w
        gy = c * ay_ ** (n - 1) * np.sign(y) / ry
        # derivative along the axis (radii and center change with s); far from the surface it
        # grows with K, so it is scaled back there (exact to first order on the surface, K = 1)
        gs = -c * (ax_ ** n * dw / w + ay_ ** n * dry / ry) / self.L - (gx * dcl + gy * dcf) / self.L
        gs = gs / np.maximum(K, 1.0)
        gl = np.sqrt(gx * gx + gy * gy + gs * gs) + 1e-12
        d_sec = (K - 1.0) / gl
        # outside, (K - 1) * smallest radius bounds the distance from below (thin sections)
        d_sec = np.where(K > 1, np.maximum(d_sec, 0.8 * (K - 1) * np.minimum(w, ry)), d_sec)
        # flat end discs, edges rounded by the cap radius
        a_ = d_sec + rc
        b_ = np.maximum(-s, s - self.L) + rc
        o1 = np.maximum(a_, 0)
        o2 = np.maximum(b_, 0)
        return np.sqrt(o1 * o1 + o2 * o2) + np.minimum(np.maximum(a_, b_), 0) - rc

    def bounds(self):
        ts = np.linspace(0, 1, 33)
        r = max(float(np.max(self.w(ts)[0])), float(np.max(self.f(ts)[0])), float(np.max(self.k(ts)[0])))
        r += max(float(np.max(np.abs(self.cf(ts)[0]))), float(np.max(np.abs(self.cl(ts)[0]))))
        lo = np.minimum(self.a, self.b) - r
        hi = np.maximum(self.a, self.b) + r
        return lo, hi


def tube(a, b, w, f, **kw) -> Tube:
    return Tube(a, b, w, f, **kw)


def halfspace(point, normal, **kw):
    """Solid below the plane through `point` with outward `normal`."""
    return sdf.plane(point, normal, **kw)


def wedge(planes, k=0.01, **kw):
    """Smooth intersection of half-spaces [(point, outward normal), ...]: a convex block with
    rounded edges (radius ~k)."""
    out = None
    for pt, nr in planes:
        h = sdf.plane(pt, m3.normalize(m3.vec3(nr)), **kw)
        out = h if out is None else out.smooth_intersect(h, k)
    return out


class Offset(sdf.SDF):
    """Leaf shape: `child` grown by a thickness that may vary in space (t: float or fn(points)->(N,)).
    Used for garments: a smooth shell over the body with its own single material."""

    def __init__(self, child, t, t_max=None, color=None, material=None, bone=None):
        self.child = child
        self.t = t
        self.t_max = float(t if not callable(t) else (t_max if t_max is not None else 0.05))
        self.color, self.material, self.bone = color, material, bone

    def dist(self, p):
        tt = self.t(p) if callable(self.t) else self.t
        return self.child.dist(p) - tt

    def bounds(self):
        lo, hi = self.child.bounds()
        return lo - self.t_max, hi + self.t_max


class Strand(sdf.SDF):
    """Smooth swept clump (hair locks): a dense polyline with an outward direction per point and
    a lens/ellipse section whose half width `w`, outward half thickness `f` and inward half
    thickness `k` vary along it (arrays, one value per point). Unlike a chain of short tubes
    there are no necks between segments: the section frame and radii are interpolated along the
    nearest segment. The root end is closed by a round cap; the tip closes by tapering."""

    def __init__(self, points, outward, w, f, k, color=None, material=None, bone=None):
        self.color, self.material, self.bone = color, material, bone
        self.P = np.asarray(points, float)
        self.D = np.diff(self.P, axis=0)
        self.L2 = np.einsum("ij,ij->i", self.D, self.D)
        T = self.D / np.sqrt(self.L2)[:, None]
        tang = np.vstack([T[:1], 0.5 * (T[1:] + T[:-1]), T[-1:]])
        tang /= np.linalg.norm(tang, axis=1, keepdims=True)
        N = np.asarray(outward, float)
        N = N - tang * np.einsum("ij,ij->i", N, tang)[:, None]
        self.N = N / np.linalg.norm(N, axis=1, keepdims=True)
        self.tang = tang
        self.w, self.f, self.k = (np.asarray(a, float) for a in (w, f, k))
        r = np.maximum(self.w, np.maximum(self.f, self.k))
        self.lo = (self.P - r[:, None]).min(axis=0)
        self.hi = (self.P + r[:, None]).max(axis=0)

    def bounds(self):
        return self.lo, self.hi

    def dist(self, p):
        p = np.asarray(p, float)
        out = np.empty(len(p))
        # far points: distance to the bounding box (a lower bound) is enough
        q = np.maximum(self.lo - p, 0) + np.maximum(p - self.hi, 0)
        far = np.linalg.norm(q, axis=1)
        near = far < 0.006
        out[~near] = far[~near]  # a lower bound: safe for the mesher's narrow band
        idx = np.flatnonzero(near)
        for s in range(0, len(idx), 60000):
            ii = idx[s:s + 60000]
            out[ii] = self._dist(p[ii])
        return out

    def _dist(self, p):
        rel = p[:, None, :] - self.P[None, :-1, :]  # (n, m, 3)
        t = np.clip(np.einsum("nmj,mj->nm", rel, self.D) / self.L2[None], 0, 1)
        c = rel - t[..., None] * self.D[None]
        d2 = np.einsum("nmj,nmj->nm", c, c)
        j = np.argmin(d2, axis=1)
        n = len(p)
        ar = np.arange(n)
        tj = t[ar, j]
        v = c[ar, j]  # from the axis point to p
        lerp = lambda A: A[j] * (1 - tj)[:, None] + A[j + 1] * tj[:, None] if A.ndim == 2 else A[j] * (1 - tj) + A[j + 1] * tj  # noqa: E731,E501
        tg = lerp(self.tang)
        tg /= np.linalg.norm(tg, axis=1, keepdims=True)
        nr = lerp(self.N)
        nr = nr - tg * np.einsum("ij,ij->i", nr, tg)[:, None]
        nr /= np.linalg.norm(nr, axis=1, keepdims=True)
        bn = np.cross(tg, nr)
        w = np.maximum(lerp(self.w), 1e-4)
        y = np.einsum("ij,ij->i", v, nr)
        x = np.einsum("ij,ij->i", v, bn)
        ry = np.maximum(np.where(y > 0, lerp(self.f), lerp(self.k)), 1e-4)
        K = np.sqrt((x / w) ** 2 + (y / ry) ** 2) + 1e-12
        g = np.sqrt((x / (w * w)) ** 2 + (y / (ry * ry)) ** 2) / K + 1e-12
        d_sec = (K - 1) / g
        d_sec = np.where(K > 1, np.maximum(d_sec, 0.8 * (K - 1) * np.minimum(w, ry)), d_sec)
        # root cap: before the first point the section rounds off
        # end caps (rounded at the root, nearly pointed at the tapered tip)
        q4 = max(len(self.P) // 4, 1)
        e = np.full(n, -1.0)
        s0 = -((p - self.P[0]) @ self.tang[0])
        s1 = (p - self.P[-1]) @ self.tang[-1]
        e = np.where(j < q4, s0, e)
        e = np.where(j >= len(self.D) - q4, np.maximum(e, s1), e)
        rc = np.minimum(w, ry) * 0.9
        a_ = d_sec + rc
        b_ = e + rc
        o1, o2 = np.maximum(a_, 0), np.maximum(b_, 0)
        return np.sqrt(o1 * o1 + o2 * o2) + np.minimum(np.maximum(a_, b_), 0) - rc
