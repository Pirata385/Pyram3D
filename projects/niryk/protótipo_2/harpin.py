"""Harpin anatomy for the prototype 2 anime bodies: plumed eagle legs, scaled toes with talons, the
long plumed tail, the huge folded wings and the feather tufts at the temples.

Smooth forms with crisp feather shapes (the anime-RPG line of the humans):
  - legs and tail are SDF tubes covered by pointed plume locks (shapes.Strand, the hair's locks),
    meshed like the body (Taubin, decimation, analytic normals), slot Plume;
  - toes are swept tubes with painted scutes (slot Skin) ending in curved claws (slot Talon);
  - wing feathers are thin closed solids (lens section with a raised, darker rachis): they read
    from both sides with any material sidedness. A folded wing is louvered like roof shingles:
    every feather is tilted a few degrees about its own axis, so neighbors in a row overlap
    without touching and the rows (primaries, secondaries + tertials, greater coverts, lesser
    coverts) stack only a couple of centimeters apart. The leading edge is a smooth SDF arm.

Everything is modeled in design space (see common.xf_points), left side first and mirrored.
"""
from __future__ import annotations

import math

import numpy as np

from common import (H_BALL, H_HOCK, H_KNEE, H_TOE, HIP, FEMALE, TAIL_POINTS, WING_2D, WING_D, WING_ELBOW, WING_O,
                    WING_ROOT, WING_TIP, WING_WRIST, slot, wing_point)
from shapes import Strand, Tube

import pyram3d as p3
from pyram3d import math3d as m3
from pyram3d import sdf


def _ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _n(v):
    v = np.asarray(v, float)
    return v / max(float(np.linalg.norm(v)), 1e-12)


def plume(shape, shade=1.0):
    g = float(shade)
    return shape.paint(color=(g, g, g), material=slot("Plume"))


SLIM = 0.92 if FEMALE else 1.0  # female limbs


# ----------------------------------------------------------------------------- plume locks

def plume_lock(pts, outward, w, f, k=None, root=0.75, peak=0.3, tip=0.03, thin=0.6, samples=5):
    """Pointed plume (a hair lock): control points, outward directions (one per point or one for
    all), max half width w (at `peak` of the length), outward half thickness f, inward k."""
    P = np.asarray(pts, float)
    O = np.asarray(outward, float)
    if O.ndim == 1:
        O = np.repeat(O[None], len(P), axis=0)
    Q = m3.catmull_rom(P, samples)
    OQ = m3.catmull_rom(O, samples)
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    grow = root + (1 - root) * _ss(0, peak, u)
    taper = 1 - (1 - tip) * np.clip((u - peak) / (1 - peak), 0, 1) ** 1.25
    W = w * grow * taper
    F = f * (1 - thin * np.clip((u - 0.45) / 0.55, 0, 1) ** 1.2)
    K = F * (0.6 if k is None else k / f)
    return Strand(Q, OQ, W, F, K)


def _frame(axis):
    """Two unit vectors perpendicular to `axis`: a = 'front-ish', b = a x axis."""
    axis = _n(axis)
    ref = np.array([0.0, 0.0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    a = _n(ref - axis * np.dot(ref, axis))
    b = np.cross(axis, a)
    return a, b


def ring_locks(a, b, ra, rb, t0, t1, azimuths, w, f, lift=0.004, flare=0.012, reach=0.0, side=1):
    """Plume locks lying on the tube a -> b (radius ra -> rb): roots at t0, tips at t1 (fractions
    of the segment, t1 may exceed 1), at the given azimuths (deg, 0 = front (+Z side), 90 =
    outward on the left leg). `flare`: how far the tips stand off the surface."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ax = b - a
    fa, fb = _frame(ax)
    out = []
    for az in azimuths:
        th = math.radians(az)
        d = math.cos(th) * fa + side * math.sin(th) * (-fb)
        d = _n(d - _n(ax) * np.dot(d, _n(ax)))
        pts, nrm = [], []
        for j, t in enumerate(np.linspace(t0, t1, 4)):
            r = ra + (rb - ra) * min(max(t, 0.0), 1.0)
            s = j / 3
            pts.append(a + ax * t + d * (r + lift + flare * s ** 1.6) + _n(ax) * reach * s)
            nrm.append(d)
        out.append(plume_lock(pts, nrm, w, f, tip=0.03))
    return out


# ----------------------------------------------------------------------------- legs

def leg_points(side=1):
    s = np.array([side, 1.0, 1.0])
    return [np.array([HIP[0], 1.0, 0.0]) * s, H_KNEE * s, H_HOCK * s, H_BALL * s, H_TOE * s]


def leg_core(side=1):
    """Thigh (under the tunic), the full plumed shin ('feathered trousers') and the slim tarsus,
    as smooth tubes."""
    top, knee, hock, ball, _ = leg_points(side)
    g = SLIM
    thigh = Tube(top, knee, {0: 0.088 * g, 0.4: 0.082 * g, 1: 0.058 * g}, {0: 0.088 * g, 0.5: 0.078 * g, 1: 0.06 * g},
                 k={0: 0.1 * g, 0.4: 0.09 * g, 1: 0.06 * g}, cap=0.03)
    # the shin: a full plumed volume high up (the 'drumstick'), slimming toward the fringe
    shin = Tube(knee + np.array([0, 0.03, 0.01]), hock, {0: 0.06 * g, 0.28: 0.068 * g, 0.7: 0.046 * g, 1: 0.03 * g},
                {0: 0.06 * g, 0.3: 0.06 * g, 1: 0.03 * g}, k={0: 0.064 * g, 0.3: 0.074 * g, 0.7: 0.05 * g, 1: 0.032 * g},
                cap=0.02)
    hock_ball = sdf.sphere(hock, 0.03 * g)
    tarsus = Tube(hock, ball + np.array([0, 0.012, -0.004]), {0: 0.026 * g, 0.5: 0.021 * g, 1: 0.025 * g},
                  {0: 0.028 * g, 0.5: 0.022 * g, 1: 0.028 * g}, cap=0.016)
    leg = thigh.smooth_union(shin, 0.035).smooth_union(hock_ball, 0.02).smooth_union(tarsus, 0.02)
    return plume(leg)


def leg_plumes(side=1):
    """Two crisp fringes of pointed plumes: a bell flaring out above the hock (the bottom of the
    feathered trousers) and a cuff over the toes."""
    _, knee, hock, ball, _ = leg_points(side)
    g = SLIM
    L = []
    L += ring_locks(knee, hock, 0.06 * g, 0.032 * g, 0.12, 0.6, range(-150, 151, 60), 0.042 * g, 0.009,
                    lift=-0.006, flare=0.022, side=side)
    L += ring_locks(knee, hock, 0.056 * g, 0.032 * g, 0.48, 1.12, range(-157, 158, 45), 0.046 * g, 0.009,
                    lift=-0.004, flare=0.045, side=side)
    L += ring_locks(hock, ball, 0.026 * g, 0.024 * g, 0.55, 1.08, range(-144, 145, 72), 0.026 * g, 0.006,
                    lift=-0.002, flare=0.018, side=side)
    return plume(sdf.smooth_union(*L, k=0.0025))


def legs_sdf():
    """Both legs (core + plumes), for the body SDF."""
    out = []
    for s in (1, -1):
        out.append(leg_core(s).smooth_union(leg_plumes(s), 0.004))
    return sdf.union(*out)


# ----------------------------------------------------------------------------- toes & talons

def _sweep(path, radii, sides=8, name="Sweep"):
    m = p3.sweep(path, radius=list(radii), segments=sides, caps=True, name=name)
    m.uvs = None
    m.face_uvs = None
    return m


TOES = [  # (yaw deg, length, root radius) for the front toes; the hallux points back
    (-26.0, 0.1, 0.0165),
    (0.0, 0.12, 0.0175),
    (25.0, 0.104, 0.0165),
]


def _toe(ball, yaw, length, r0, back=False):
    """One toe: a swept tube from inside the cuff along the ground (knuckle pads), then a curved
    claw. Returns (toe mesh [Skin], claw mesh [Talon])."""
    a = math.radians(yaw)
    d = np.array([math.sin(a), 0.0, math.cos(a)]) * (-1 if back else 1)
    start = ball + np.array([0, 0.012, 0]) - d * 0.012
    knots = [start, ball + d * length * 0.3 + np.array([0, -0.012, 0]),
             ball + d * length * 0.68 + np.array([0, -0.016, 0]), ball + d * length + np.array([0, -0.017, 0])]
    path = m3.catmull_rom(np.array(knots), 2)
    path[:, 1] = np.maximum(path[:, 1], r0 * 0.78)  # resting on the ground
    t = np.linspace(0, 1, len(path))
    rad = r0 * (1.0 - 0.28 * t) * (1 + 0.07 * np.sin(t * math.pi * 3) ** 2)
    toe = _sweep(path, rad, sides=7, name="Toe")
    V = toe.vertices
    # scutes: darker rings across the top, lighter underside pads
    along = (V - start) @ d
    ring = 0.5 + 0.5 * np.cos(along / 0.011 * 2 * math.pi)
    top = _ss(-0.3, 0.6, (V[:, 1] - path[:, 1].mean()) / r0)
    lum = 0.92 - 0.2 * ring ** 6 * top
    toe.colors = np.ones((len(V), 4))
    toe.colors[:, :3] = lum[:, None]
    toe.set_material(slot("Skin"))
    # claw: an arc curving down to the ground from the toe tip
    tip = path[-1]
    c0 = tip - d * 0.004 + np.array([0, 0.002, 0])
    arc = [c0 + d * 0.02 * math.sin(math.radians(x)) + np.array([0, 0.012 * (math.cos(math.radians(x)) - 1), 0])
           for x in np.linspace(0, 95, 5)]
    cl = np.array(arc)
    cl[:, 1] = np.maximum(cl[:, 1], 0.0015)
    crad = r0 * 0.62 * (1 - np.linspace(0, 1, len(cl)) ** 1.2) + 0.0006
    claw = _sweep(cl, crad, sides=6, name="Claw")
    claw.colors = np.ones((claw.num_vertices, 4))
    claw.set_material(slot("Talon"))
    return toe, claw


def feet_meshes(side=1):
    """Toes (Skin) and claws (Talon) of one foot: three forward, the hallux back."""
    ball = H_BALL * np.array([side, 1, 1])
    parts = []
    for yaw, ln, r0 in TOES:
        parts += _toe(ball, yaw * side, ln * SLIM, r0 * SLIM)
    parts += _toe(ball + np.array([0, 0.0, -0.006]), -12 * side, 0.06 * SLIM, 0.015 * SLIM, back=True)
    m = p3.Mesh.join(parts, name="Feet")
    m.normals = None
    m.normals = m.vertex_normals()
    return m


# ----------------------------------------------------------------------------- tail

def tail_path(samples=8):
    return m3.catmull_rom(TAIL_POINTS, samples)


def tail_sdf():
    """Long mammal tail covered in plumes: a smooth spindle along the curve (slim shaft, a long
    brush, a pointed tip) wrapped in rows of pointed plumes whose tips lift off the surface at
    staggered places, so the outline reads as feathery zigzags."""
    Q = tail_path(10)
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    t = s / s[-1]
    ext = 0.2  # the brush's tip goes this far (meters) past the last tail joint
    tq = [0.0, 0.1, 0.4, 0.55, 0.72, 0.86, 1.0]
    rq = [0.044, 0.038, 0.03, 0.044, 0.068, 0.066, 0.046]

    def point(tt):
        """Point, tangent, (a, b) frame and core radius at a fraction of the curve (past 1: along
        the end tangent, the radius closing to the tip)."""
        if tt <= 1.0:
            i = int(np.clip(np.searchsorted(t, tt), 1, len(t) - 1))
            q, tg = Q[i], _n(Q[min(i + 1, len(Q) - 1)] - Q[i - 1])
            rr = float(np.interp(tt, tq, rq))
        else:
            tg = _n(Q[-1] - Q[-3])
            q = Q[-1] + tg * (tt - 1.0) * s[-1]
            e = min((tt - 1.0) * s[-1] / ext, 1.0)
            rr = rq[-1] * (1 - e ** 1.6) ** 0.8
        a = _n(np.array([0.0, 1.0, 0.0]) - tg * tg[1])
        return q, tg, (a, np.cross(tg, a)), rr * SLIM

    t_end = 1.0 + ext / s[-1]
    ts = np.concatenate([np.linspace(0, 1, 16), np.linspace(1, t_end, 6)[1:]])
    pts = [point(x) for x in ts]
    core = sdf.chain([tuple(p[0]) for p in pts], [max(p[3] * 0.8, 0.003) for p in pts], k=0.015)
    locks = []

    def lock_along(t0, t1, az, w, f, lift, splay=0.0, n=6, tipk=0.03):
        P_, N_ = [], []
        for j, tt in enumerate(np.linspace(t0, t1, n)):
            q, _, (a, b), rr = point(min(tt, t_end))
            d = math.cos(az) * a + math.sin(az) * b
            u = j / (n - 1)
            P_.append(q + d * (rr * 0.8 + lift(u)) + (d * splay * u ** 3 if splay else 0))
            N_.append(d)
        return plume_lock(P_, N_, w * SLIM, f, k=f * 0.85, tip=tipk, peak=0.4, thin=0.45, root=0.7)

    # shaft: two rows of long smooth plumes lying flat, overlapping (shingles)
    for ri, t0 in enumerate((0.06, 0.26)):
        for k in range(4):
            az = 2 * math.pi * (k + 0.5 * ri) / 4
            locks.append(lock_along(t0, t0 + 0.3, az, 0.04, 0.01, lambda u: 0.004 + 0.008 * u ** 2))
    # brush: big locks bulging out and converging to splayed points past the end
    n_b = 7
    for k in range(n_b):
        az = 2 * math.pi * k / n_b
        t0 = 0.44 + 0.03 * (k % 2)
        t1 = t_end - 0.05 * ((k * 3) % 4) / 3
        locks.append(lock_along(t0, t1, az, 0.062, 0.017,
                                lambda u: 0.006 + 0.03 * math.sin(math.pi * min(1.0, u * 1.08)) ** 0.9, splay=0.018,
                                n=8, tipk=0.02))
    # an outer layer starting the brush: shorter locks whose tips lift off (zigzag outline)
    for k in range(5):
        az = 2 * math.pi * (k + 0.5) / 5
        locks.append(lock_along(0.5, 0.78, az, 0.05, 0.012, lambda u: 0.01 + 0.03 * u ** 1.4, n=5))
    shape = core.smooth_union(sdf.smooth_union(*locks, k=0.003), 0.005)
    return plume(shape)


# ----------------------------------------------------------------------------- feathers

def feather(root, tip, across, width, to3d, *, d0=0.0, round_tip=0.4, asym=0.0, curve=0.0, tilt=6.0,
            thick=0.0026, n=8, lum=1.0, rachis=0.6, tip_dark=0.12, under=0.82, quill=0.16):
    """A thin closed feather solid, laid out in a 2D frame and placed by to3d(u, v, d) (d = off the
    frame's surface, outward). root/tip: 2D points; `across`: 2D direction toward the feather that
    lies on top of this one in its row (that side is tilted down by `tilt` degrees, the other side
    up, so shingled neighbors never touch); d0: the row's depth. width: full vane width; asym > 0
    widens the vane on the `across` side; curve: in-plane bow (fraction of the length) away from
    `across`. Vertex colors: vane `lum`, darker rachis, darker tip and underside."""
    root, tip = np.asarray(root, float), np.asarray(tip, float)
    ax = tip - root
    L = float(np.linalg.norm(ax))
    ax /= L
    ac = np.array([-ax[1], ax[0]])
    if np.dot(ac, across) < 0:
        ac = -ac
    tt = math.tan(math.radians(tilt))
    ts = np.linspace(0, 1, n + 1)[:-1] ** 0.9  # a little denser toward the tip
    U, D, C = [], [], []
    for t in ts:
        grow = quill + (1 - quill) * math.sin(math.pi / 2 * min(1.0, t / 0.28))
        end = 1.0
        if t > 0.62:
            e = (t - 0.62) / 0.38
            end = (1 - e ** (2.0 + 2.5 * round_tip)) ** (0.5 + 0.4 * (1 - round_tip))
        h = 0.5 * width * grow * end
        c = root + ax * L * t - ac * curve * L * math.sin(math.pi * t)
        wl, wr = h * (1 - asym), h * (1 + asym)  # -across / +across vanes
        top = thick * (1 - 0.55 * t)
        bot = top * 0.5
        r = min(0.0016 + 0.0012 * (1 - t), h * 0.4)
        cup = 0.06 * h
        for y, z in ((-wl, -cup), (-r, top), (-r, top), (r, top), (r, top), (wr, -cup), (0.0, -bot)):
            U.append(c + ac * y)
            D.append(d0 + z - y * tt)
        vane = lum * (1 - tip_dark * _ss(0.72, 1.0, t))
        C += [vane * 0.9, vane, vane * rachis, vane * rachis, vane, vane * 0.97, vane * under]
    tip_i = len(U)
    U.append(tip)
    D.append(d0)
    C.append(lum * (1 - tip_dark))
    root_i = len(U)
    U.append(root - ax * 0.002)
    D.append(d0)
    C.append(lum * rachis)
    U = np.array(U)
    V = to3d(U[:, 0], U[:, 1], np.array(D))
    F = []
    R = 7
    for i in range(len(ts) - 1):
        a, b = i * R, (i + 1) * R
        F += [(a + 0, b + 0, b + 1, a + 1), (a + 2, b + 2, b + 3, a + 3), (a + 4, b + 4, b + 5, a + 5),
              (a + 5, b + 5, b + 6, a + 6), (a + 6, b + 6, b + 0, a + 0)]
    a = (len(ts) - 1) * R
    F += [(a + 0, tip_i, a + 1), (a + 2, tip_i, a + 3), (a + 4, tip_i, a + 5), (a + 5, tip_i, a + 6),
          (a + 6, tip_i, a + 0)]
    F += [(0, 1, root_i), (2, 3, root_i), (4, 5, root_i), (5, 6, root_i), (6, 0, root_i)]
    m = p3.Mesh(V, F, name="Feather")
    if _signed_volume(m) < 0:  # outward winding
        m.flip()
    m.colors = np.ones((len(V), 4))
    m.colors[:, :3] = np.clip(np.array(C), 0, 1)[:, None]
    _weld_normals(m)
    return m


def _signed_volume(m):
    vol = 0.0
    P = m.vertices
    for f in m.faces:
        for i in range(1, len(f) - 1):
            vol += float(np.dot(P[f[0]], np.cross(P[f[i]], P[f[i + 1]])))
    return vol


def _weld_normals(m):
    """Smooth normals shared by coincident vertices (the rachis seams are split for the color)."""
    N = m.vertex_normals()
    key = np.round(m.vertices / 1e-6).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.reshape(-1)
    S = np.zeros((inv.max() + 1, 3))
    np.add.at(S, inv, N)
    N = S[inv]
    m.normals = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    m.smooth_angle = 180.0


def plane_map(origin, e1, e2, normal):
    """to3d for feathers on a flat frame: (u, v, d) -> origin + u e1 + v e2 + d normal."""
    o, a, b, c = (np.asarray(x, float) for x in (origin, e1, e2, normal))
    return lambda u, v, d: o + np.outer(u, a) + np.outer(v, b) + np.outer(d, c)


# ----------------------------------------------------------------------------- wings

def _lerp(a, b, t):
    return np.asarray(a, float) + (np.asarray(b, float) - np.asarray(a, float)) * t


# feather rows: depth of each row off the wing surface (outward = away from the body)
ROW_D = {"prim": 0.0, "sec": 0.02, "gcov": 0.04, "lcov": 0.055}
INWARD = np.array([-1.0, 0.0])  # rows overlap from the outside in: inner feathers lie on top


def wing_feathers():
    """Left wing feathers (wing coordinates -> design space) as [(mesh, {bone: weight})]."""
    R, E, W, T = (np.array(WING_2D[k]) for k in ("root", "elbow", "wrist", "tip"))
    g = SLIM
    out = []

    def fth(r2, t2, row, width, **kw):
        return feather(r2, t2, INWARD, width, wing_point, d0=ROW_D[row], **kw)

    # primaries: roots along the hand (from the wrist down toward the tip), tips stepping up from
    # the longest (outermost, bottom of the stack); bowed outward
    hand = _n(T - W)
    n_p = 9
    for k in range(n_p):
        f = k / (n_p - 1)
        r2 = W + hand * (0.2 * (1 - f) + 0.03)
        t2 = T + np.array([0.022 * k, 0.095 * k + 0.002 * k * k])
        m = fth(r2, t2, "prim", (0.115 + 0.01 * f) * g, round_tip=0.12 + 0.3 * f, asym=0.25, curve=0.05 - 0.02 * f,
                n=8, lum=0.88 + 0.07 * (k % 2), thick=0.003)
        w3 = 1.0 if k < 5 else 1.0 - 0.06 * (k - 4)
        out.append((m, {"Wing3": w3, "Wing2": 1 - w3}))
    # secondaries (forearm, wrist -> elbow) and tertials (humerus, elbow -> root): one row going in
    n_s, n_t = 8, 3
    v_of = lambda y: (y - WING_O[1]) / 0.992  # noqa: E731  (v of a design height, near enough)
    for k in range(n_s + n_t):
        if k < n_s:
            f = k / (n_s - 1)
            r2 = _lerp(W, E, 0.04 + 0.92 * f) - np.array([0.0, 0.02])
            t2 = np.array([r2[0] - 0.035 - 0.004 * k, v_of(1.0 + 0.016 * k)])
            bone = {"Wing2": 1.0} if k < n_s - 2 else {"Wing2": 0.65, "Wing1": 0.35}
            width, lum = 0.13 * g, 0.94 + 0.06 * (k % 2)
        else:
            j = k - n_s
            r2 = _lerp(E, R, 0.15 + 0.6 * (j + 1) / (n_t + 1)) - np.array([0.01, 0.0])
            t2 = np.array([r2[0] - 0.02, v_of(1.12 + 0.03 * j)])
            bone = {"Wing1": 1.0}
            width, lum = 0.12 * g, 1.0
        out.append((fth(r2, t2, "sec", width, round_tip=0.75, asym=0.12, curve=0.01, n=6, lum=lum, thick=0.0028),
                    bone))
    # coverts along the arm: hand (primary coverts) -> wrist -> elbow -> root
    arm = [W + hand * 0.17, W, E, R + (E - R) * 0.15]
    cum = np.concatenate([[0], np.cumsum([np.linalg.norm(arm[i + 1] - arm[i]) for i in range(3)])])

    def on_arm(x):
        x = x * cum[-1]
        i = min(int(np.searchsorted(cum, x, side="right") - 1), 2)
        return _lerp(arm[i], arm[i + 1], (x - cum[i]) / (cum[i + 1] - cum[i])), i

    def cov_bone(seg, near_elbow=False):
        return {"Wing3": 1.0} if seg == 0 else ({"Wing2": 1.0} if seg == 1 else {"Wing1": 0.8, "Wing2": 0.2})

    n_g = 9
    for k in range(n_g):
        f = k / (n_g - 1)
        r2, seg = on_arm(f)
        r2 = r2 - np.array([0.012, 0.03])
        ln = (0.3 - 0.08 * f) * g if seg > 0 else 0.3 * g
        dirv = _n(np.array([-0.18, -1.0])) if seg > 0 else _n(hand + np.array([-0.15, 0.0]))
        out.append((fth(r2, r2 + dirv * ln, "gcov", 0.11 * g, round_tip=0.85, asym=0.1, n=5, lum=1.0, thick=0.0028,
                        tip_dark=0.08), cov_bone(seg)))
    n_l = 6
    for k in range(n_l):
        f = 0.08 + 0.84 * k / (n_l - 1)
        r2, seg = on_arm(f)
        r2 = r2 + np.array([-0.01, 0.0])
        dirv = _n(np.array([-0.25, -1.0])) if seg > 0 else _n(hand + np.array([-0.2, 0.0]))
        out.append((fth(r2, r2 + dirv * 0.16 * g, "lcov", 0.1 * g, round_tip=0.95, n=4, lum=1.0, thick=0.0026,
                        tip_dark=0.06, rachis=0.8), cov_bone(seg)))
    return out


def wing_arm_sdf():
    """The left wing's leading edge: a smooth plumed arm root -> elbow -> wrist -> hand along the
    wing surface (the bones run through it)."""
    R, E, W, T = (np.array(WING_2D[k]) for k in ("root", "elbow", "wrist", "tip"))
    hand = _n(T - W)
    knots = [R - (E - R) * 0.3, R, E, W - hand * 0.01, W + hand * 0.13]
    rad = [0.028, 0.036, 0.04, 0.032, 0.021]
    pts, rr = [], []
    for i in range(len(knots) - 1):
        for t in np.linspace(0, 1, 4)[:-1]:
            pts.append(_lerp(knots[i], knots[i + 1], t))
            rr.append(rad[i] + (rad[i + 1] - rad[i]) * t)
    pts.append(knots[-1])
    rr.append(rad[-1])
    pts = np.array(pts)
    P3 = wing_point(pts[:, 0], pts[:, 1], np.full(len(pts), WING_D))
    arm = sdf.chain([tuple(p) for p in P3], [r * SLIM for r in rr], k=0.02)
    return plume(arm)


def wing_bone_weights(P, names):
    """Weights of points on the left wing arm: root (Chest) -> Wing1 -> Wing2 -> Wing3 by the
    nearest point along root -> elbow -> wrist -> hand."""
    R, E, W, T = WING_ROOT, WING_ELBOW, WING_WRIST, WING_TIP
    knots = [R, E, W, W + _n(T - W) * 0.2]
    best_d = np.full(len(P), np.inf)
    best_s = np.zeros(len(P))
    for i in range(3):
        a, b = knots[i], knots[i + 1]
        ab = b - a
        t = np.clip(((P - a) @ ab) / (ab @ ab), -0.3 if i == 0 else 0, 1)
        dd = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        better = dd < best_d
        best_d = np.where(better, dd, best_d)
        best_s = np.where(better, i + t, best_s)
    s = best_s
    W_ = np.zeros((len(P), len(names)))
    idx = {nm: i for i, nm in enumerate(names)}
    W_[:, idx["Chest"]] = 1 - _ss(-0.15, 0.2, s)
    W_[:, idx["LeftWing1"]] = _ss(-0.15, 0.2, s) * (1 - _ss(0.85, 1.15, s))
    W_[:, idx["LeftWing2"]] = _ss(0.85, 1.15, s) * (1 - _ss(1.88, 2.12, s))
    W_[:, idx["LeftWing3"]] = _ss(1.88, 2.12, s)
    return W_


def set_weights(m, weights, names):
    """Rigid weights for a whole mesh: {bone: weight}."""
    m.joints = np.zeros((m.num_vertices, 4), np.int64)
    m.weights = np.zeros((m.num_vertices, 4))
    for k, (b, w) in enumerate(sorted(weights.items(), key=lambda kv: -kv[1])[:4]):
        m.joints[:, k] = names.index(b)
        m.weights[:, k] = w
    m.weights /= m.weights.sum(axis=1, keepdims=True)
    return m


def mirror_wing(m, names):
    """Left wing mesh (skinned) -> the right one (X flip, winding flip, Left -> Right bones)."""
    r = m.copy()
    r.vertices = r.vertices * np.array([-1.0, 1.0, 1.0])
    if r.normals is not None:
        r.normals = r.normals * np.array([-1.0, 1.0, 1.0])
    r.flip()
    if r.joints is not None:
        remap = np.array([names.index(nm.replace("Left", "Right")) if nm.startswith("Left") else i
                          for i, nm in enumerate(names)])
        r.joints = remap[r.joints]
    return r


# ----------------------------------------------------------------------------- temple tufts

def temple_feathers():
    """Feather tufts at the temples, sweeping back and up like small wings (both sides), in design
    space (they follow the Head bone). Slot Plume."""
    import head as H
    parts = []
    for side in (1, -1):
        sx = np.array([side, 1.0, 1.0])
        base = H.head_xf(np.array([0.08, 1.69, -0.02])) * sx  # above the ear, at the temple
        spec = [  # (direction (left side), length, width, lum): a fan sweeping up and back like a small wing
            ((0.5, 0.78, -0.38), 0.15, 0.036, 1.0),
            ((0.62, 0.55, -0.56), 0.17, 0.038, 0.94),
            ((0.66, 0.28, -0.7), 0.145, 0.034, 1.0),
            ((0.6, 0.02, -0.8), 0.11, 0.03, 0.94),
        ]
        for k, (dv, ln, w, lum) in enumerate(spec):
            d = _n(np.array(dv) * sx)
            out = _n(np.array([1.0, 0.1, 0.1]) * sx)  # the tuft's outer face looks sideways
            out = _n(out - d * np.dot(out, d))
            e2 = np.cross(out, d)
            to3d = plane_map(base + np.array([0, 0.004 * k, 0]), d, e2, out)
            m = feather((0.004, 0.0), (ln * (0.94 if FEMALE else 1.0), 0.0), np.array([0.0, 1.0]), w, to3d,
                        round_tip=0.1, asym=0.1, curve=-0.05, tilt=10.0, thick=0.0022, n=6, lum=lum, tip_dark=0.1)
            parts.append(m)
    return p3.Mesh.join(parts, name="Head_Plumes")
