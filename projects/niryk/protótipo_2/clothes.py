"""Adventurer outfit (anime RPG), male and female, as smooth offset shells of the body.

  Tunic    Cloth_A body with Cloth_B trims: long sleeves (they end under the bracers), an A-line
           skirt (male: a coat-like tunic split at the front, trim along the hem and the slit
           and a placket down the chest; female: a fitted dress, its hem over a Cloth_B
           underskirt layer that shows below it)
  Collar   layered: a Cloth_B stand collar around the neck and a Cloth_A capelet over the
           shoulders with a Cloth_B edge (male: pointed at the front; female: rounded, with a
           Cloth_B bow)
  Belt     Leather belt (female: a wide waist cincher) with a Metal buckle and a Leather pouch
  Trousers Cloth_B trousers (female: leggings)
  Boots    Leather knee boots with a folded cuff
  Bracers  Leather bracers with straps, over the sleeve ends

Every garment is a smooth offset shell of the body forms it covers (shapes.Offset) cut by planes
with rounded (smooth-intersected) hems; trims are slightly thicker shells limited to a band, so
the material change has a little step that the toon outline picks up. Body faces under the
garments are deleted at build time.
"""
from __future__ import annotations

import math

import numpy as np

import body as B
from common import FEMALE, FOREARM, HARPIN, SHOULDER, UPPER_ARM, arm_dir, arm_point, mirror_side, slot
from shapes import Offset, Tube

from pyram3d import math3d as m3
from pyram3d import sdf

TUNIC_HEM = 0.755 if FEMALE else 0.72
UNDER_HEM = TUNIC_HEM - 0.03  # female underskirt
BELT_Y = (1.035, 1.13) if FEMALE else (0.985, 1.04)
BOOT_TOP = 0.445 if FEMALE else 0.43
SLEEVE = UPPER_ARM + FOREARM * 0.56  # sleeve end along the arm (under the bracer)
BRACER = (0.42, 0.97)  # share of the forearm
CAPE_EDGE = 1.345 if FEMALE else 1.338  # capelet's lowest point (front center)
SLIT_TOP = 0.96  # male tunic front slit


def _ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _along_arm(p):
    """(distance along the nearest rest arm from its shoulder joint, side)."""
    side = np.where(p[:, 0] >= 0, 1, -1)
    out = np.empty(len(p))
    for s in (1, -1):
        m = side == s
        out[m] = (p[m] - mirror_side(SHOULDER, s)) @ arm_dir(s)
    return out, side


def _cut(shape, point, normal, k=0.006):
    """Keep the side of the plane opposite to `normal`, with a rounded edge."""
    return shape.smooth_intersect(sdf.plane(point, m3.normalize(m3.vec3(normal)), extent=2.0), k)


def _slab(y0, y1):
    return sdf.plane((0, y1, 0), (0, 1, 0), extent=2.0).intersect(sdf.plane((0, y0, 0), (0, -1, 0), extent=2.0))


def _arms_upper():
    return sdf.union(B._arm_frame(B.arm_tube(), 1), B._arm_frame(B.arm_tube(), -1))


def _upper_base():
    return sdf.smooth_union(B.torso(), B.shoulders(), k=0.05).smooth_union(_arms_upper(), 0.035)


def _t_upper(p):
    d, _ = _along_arm(p)
    on_arm = _ss(0.02, 0.08, d)
    cuff = _ss(SLEEVE - 0.05, SLEEVE - 0.015, d) * on_arm  # sleeves flare a little at the cuff
    return 0.01 + 0.002 * on_arm + 0.005 * cuff


def _skirt(extra=0.0, hem=None):
    hem = TUNIC_HEM if hem is None else hem
    y0, y1 = 1.06, hem
    T = lambda y: (y0 - y) / (y0 - y1)  # noqa: E731
    if FEMALE:
        w = {T(1.06): 0.134, T(1.0): 0.172, T(0.95): 0.196, T(0.9): 0.21, T(0.84): 0.222, T(y1): 0.238}
        f = {T(1.06): 0.09, T(0.97): 0.112, T(y1): 0.14}
        k = {T(1.06): 0.092, T(0.97): 0.136, T(y1): 0.162}
    else:
        w = {T(1.06): 0.15, T(1.0): 0.184, T(0.96): 0.202, T(0.9): 0.214, T(0.84): 0.22, T(y1): 0.228}
        f = {T(1.06): 0.108, T(0.97): 0.12, T(y1): 0.138}
        k = {T(1.06): 0.106, T(0.97): 0.138, T(y1): 0.158}
    if HARPIN:  # room in front for the forward knees of the digitigrade legs
        f = {t: v + 0.028 * t ** 1.5 for t, v in f.items()}
    if extra:
        w, f, k = ({t: v + extra for t, v in d.items()} for d in (w, f, k))
    return Tube((0, y0, -0.006), (0, y1, -0.012), w, f, k=k, n=2.3, cap=0.012)


def hem_y(p):
    """Height of the tunic hem around the body: level for the humans; Harpins: shorter at the back
    (the tail and the folded wings) with a notch over the tail."""
    p = np.atleast_2d(p)
    if not HARPIN:
        return np.full(len(p), TUNIC_HEM)
    x, z = p[:, 0], p[:, 2]
    back = _ss(0.02, -0.12, z)
    y = TUNIC_HEM + 0.06 * back
    notch = 0.975 - 2.2 * np.abs(x)  # an inverted V over the tail
    return np.maximum(y, np.where(z < -0.02, notch, -1.0))


def _hem_cut(shape, offset=0.0, k=0.006):
    """Keep the part of `shape` above the hem curve (+ offset)."""
    lo = np.array([-0.5, 0.3, -0.5])
    hi = np.array([0.5, 1.3, 0.5])
    cut = sdf.custom(lambda p: (hem_y(p) + offset) - np.atleast_2d(p)[:, 1], lo, hi)
    return shape.smooth_intersect(cut, k)


def wing_slits():
    """Harpins: where the wing arms come out of the back (subtracted from tunic and capelet)."""
    import harpin
    from common import WING_ELBOW, WING_ROOT
    out = []
    for s in (1, -1):
        sx = np.array([s, 1.0, 1.0])
        a = (WING_ROOT - (WING_ELBOW - WING_ROOT) * 0.6) * sx
        b = (WING_ROOT + (WING_ELBOW - WING_ROOT) * 0.55) * sx
        out.append(sdf.capsule(a, b, 0.046 * harpin.SLIM))
    return sdf.union(*out)


def _slit():
    """Front slit of the male tunic skirt (a thin wedge at the center front)."""
    return sdf.box((0, (SLIT_TOP + TUNIC_HEM - 0.1) / 2, 0.2), (0.012, (SLIT_TOP - TUNIC_HEM + 0.1) / 2, 0.12),
                   radius=0.004)


# ----------------------------------------------------------------------------- tunic

def tunic():
    upper = Offset(_upper_base(), _t_upper, t_max=0.02)
    shape = _cut(upper, (0, 0.99, 0), (0, -1, 0), 0.02).smooth_union(_skirt(), 0.03)
    for s in (1, -1):  # sleeves end at SLEEVE along each arm (cut limited to a cylinder around it)
        beyond = sdf.plane(arm_point(SLEEVE, s), -arm_dir(s), extent=1.0)
        cyl = sdf.capsule(arm_point(SLEEVE - 0.02, s), arm_point(UPPER_ARM + FOREARM + 0.3, s), 0.11)
        shape = shape.smooth_subtract(beyond.intersect(cyl), 0.005)
    neck = sdf.ellipsoid((0, 1.53, 0.012), (0.075, 0.08, 0.085))
    shape = shape.smooth_subtract(neck, 0.006)
    if not FEMALE:
        shape = shape.smooth_subtract(_slit(), 0.006)
    if HARPIN:
        shape = _hem_cut(shape).smooth_subtract(wing_slits(), 0.006)
    body = shape.paint(color=(1, 1, 1), material=slot("Cloth_A"))
    return body.union(tunic_trim(shape))


def tunic_trim(tunic_shape):
    """Cloth_B bands, a few millimeters proud of the tunic: along the hem (and the slit and down
    the chest for the male)."""
    grow = Offset(tunic_shape, 0.0022, t_max=0.003)
    if HARPIN:  # the band follows the hem curve
        band = sdf.custom(lambda p: np.atleast_2d(p)[:, 1] - (hem_y(p) + 0.028), np.array([-0.5, 0.3, -0.5]),
                          np.array([0.5, 1.3, 0.5]))
    else:
        band = _slab(TUNIC_HEM - 0.01, TUNIC_HEM + 0.028)
    parts = [grow.intersect(band)]
    if not FEMALE:
        slit_band = sdf.box((0, (SLIT_TOP + TUNIC_HEM) / 2, 0.2), (0.03, (SLIT_TOP - TUNIC_HEM) / 2 + 0.025, 0.12))
        parts.append(grow.intersect(slit_band))
        placket = sdf.box((0, (BELT_Y[1] + 1.36) / 2, 0.2), (0.014, (1.36 - BELT_Y[1]) / 2, 0.12))
        parts.append(grow.intersect(placket))
    trim = sdf.union(*parts)
    return trim.paint(color=(1, 1, 1), material=slot("Cloth_B"))


def underskirt():
    """Female: a Cloth_B layer whose hem shows below the dress."""
    sk = _skirt(extra=-0.004, hem=UNDER_HEM)
    sk = _cut(sk, (0, TUNIC_HEM + 0.05, 0), (0, 1, 0), 0.01)
    if HARPIN:
        sk = _hem_cut(sk, offset=-0.03)
    return sk.paint(color=(1, 1, 1), material=slot("Cloth_B"))


# ----------------------------------------------------------------------------- collar

def _cape_edge(p):
    """Height of the capelet's lower edge around the body (male: a point at the front center;
    female: a rounded front)."""
    x, z = p[:, 0], p[:, 2]
    ang = np.degrees(np.arctan2(np.abs(x), z + 0.02))  # 0 front, 90 side, 180 back
    if FEMALE:
        front = CAPE_EDGE + 0.02 * _ss(0, 60, ang) ** 0.7
    else:
        front = CAPE_EDGE + 0.05 * np.clip(ang / 55, 0, 1) ** 0.8  # V point at the front
    side = 1.382 - 0.016 * _ss(80, 120, ang)
    back = 1.358
    out = np.where(ang < 70, front + (side - front) * _ss(40, 70, ang), side)
    out = np.where(ang > 110, side + (back - side) * _ss(110, 160, ang), out)
    # over the arms the edge runs around the deltoid
    return out


def collar():
    base = sdf.smooth_union(B.torso(), B.shoulders(), k=0.05).smooth_union(_arms_upper(), 0.035)
    cape = Offset(base, 0.018, t_max=0.03)

    def edge(p):
        p = np.atleast_2d(p)
        return _cape_edge(p) - p[:, 1]  # > 0 below the edge (outside)
    lo = sdf.custom(edge, np.array([-0.4, 1.2, -0.3]), np.array([0.4, 1.6, 0.3]))
    cape = cape.smooth_intersect(lo, 0.006)
    # around the arms: stop at the top of the deltoid
    for s in (1, -1):
        cape = cape.smooth_subtract(sdf.plane(arm_point(0.075, s), -arm_dir(s), extent=1.0)
                                    .intersect(sdf.capsule(arm_point(0.05, s), arm_point(0.6, s), 0.12)), 0.01)
    neck_hole = sdf.capsule((0, 1.4, -0.03), (0, 1.62, -0.02), 0.068 if FEMALE else 0.074)
    cape = cape.smooth_subtract(neck_hole, 0.012)
    if HARPIN:
        cape = cape.smooth_subtract(wing_slits(), 0.008)
    cape_a = cape.paint(color=(1, 1, 1), material=slot("Cloth_A"))
    # trim along the lower edge
    grow = Offset(cape, 0.002, t_max=0.003)

    def band(p):
        p = np.atleast_2d(p)
        return p[:, 1] - (_cape_edge(p) + 0.018)  # < 0 within 1.8 cm of the edge
    trim = grow.intersect(sdf.custom(band, np.array([-0.4, 1.2, -0.3]), np.array([0.4, 1.6, 0.3])))
    trim = trim.paint(color=(1, 1, 1), material=slot("Cloth_B"))
    # stand collar around the neck, higher at the back than under the chin: the skin of the neck
    # (and the slope of the shoulders) grown by 7.5 mm, cut to a band. It follows the skin exactly
    # (a tube of its own sat partly inside the neck base: brown smears and skin poking through);
    # its inside and the skin under it are dropped at build time, its foot hides under the capelet.
    g = 0.86 if FEMALE else 1.0
    skin = sdf.smooth_union(B.torso(), B.shoulders(), k=0.05).smooth_union(B.neck(), 0.04)
    stand = Offset(skin, 0.0075, t_max=0.01).intersect(sdf.capsule((0, 1.3, -0.028), (0, 1.75, -0.022), 0.09 * g))
    stand = _cut(stand, (0, 1.43, 0), (0, -1, 0), 0.004)
    stand = _cut(stand, (0, 1.548 if FEMALE else 1.553, 0.0), (0, 1.0, 0.62), 0.006)  # back higher
    stand = stand.paint(color=(1, 1, 1), material=slot("Cloth_B"))
    parts = [cape_a, trim, stand]
    if FEMALE:
        parts.append(bow())
    return sdf.union(*parts)


def bow():
    """Small Cloth_B bow at the front of the female collar."""
    c = np.array([0.0, CAPE_EDGE + 0.028, 0.0])
    c[2] = _front_z(B.torso(), c[1]) + 0.03
    knot = sdf.ellipsoid(c, (0.011, 0.01, 0.008))
    loops = sdf.ellipsoid(c + np.array([0.024, 0.004, -0.004]), (0.022, 0.013, 0.007))
    loops = loops.smooth_subtract(sdf.ellipsoid(c + np.array([0.026, 0.004, 0.003]), (0.011, 0.005, 0.006)), 0.002)
    tails = sdf.capsule(c + np.array([0.004, -0.006, 0.0]), c + np.array([0.018, -0.042, -0.006]), 0.006)
    shape = knot.smooth_union(loops.mirror_x(), 0.005).smooth_union(tails.mirror_x(), 0.004)
    return shape.paint(color=(1, 1, 1), material=slot("Cloth_B"))


# ----------------------------------------------------------------------------- belt & pouch

def _front_z(shape, y, x=0.0):
    """z of the front surface of `shape` at (x, y) (bisection along +Z)."""
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if shape.dist(np.array([[x, y, mid]]))[0] < 0:
            lo = mid
        else:
            hi = mid
    return lo


def belt():
    """Leather belt over the tunic waist (female: a wide waist cincher), a metal buckle and a
    pouch on the left hip."""
    if FEMALE:
        base = B.torso()
        band = Offset(base, 0.02)
    else:
        base = sdf.smooth_union(B.torso(), sdf.ellipsoid((0, 1.0, -0.01), (0.17, 0.08, 0.126)), k=0.04)
        band = Offset(base, 0.026)
    band = _cut(_cut(band, (0, BELT_Y[1], 0), (0, 1, 0), 0.006), (0, BELT_Y[0], 0), (0, -1, 0), 0.006)
    band = band.paint(color=(1, 1, 1), material=slot("Leather"))
    yc = 0.5 * (BELT_Y[0] + BELT_Y[1])
    z = float(_front_z(base, yc)) + (0.02 if FEMALE else 0.026)
    if FEMALE:
        buckle = sdf.box((0, yc, z + 0.003), (0.034, 0.05, 0.01), radius=0.004)
        buckle = buckle.smooth_subtract(sdf.box((0, yc, z + 0.009), (0.016, 0.03, 0.014), radius=0.003), 0.002)
    else:
        buckle = sdf.box((0, yc, z + 0.004), (0.046, 0.06, 0.014), radius=0.005)
        buckle = buckle.smooth_subtract(sdf.box((0, yc, z + 0.012), (0.024, 0.034, 0.02), radius=0.003), 0.002)
    buckle = buckle.paint(color=(1, 1, 1), material=slot("Metal"))
    return sdf.union(band, buckle, pouch(base))


def pouch(base):
    """A leather pouch hanging from the belt on the left hip, a flap and a metal button."""
    y = BELT_Y[0] - 0.03 if not FEMALE else BELT_Y[0] - 0.035
    x = 0.118 if FEMALE else 0.13
    # sit it on the side-front of the hips (skirt surface)
    ang = math.radians(66)
    zc = 0.0
    sk = _skirt()
    # find the skirt surface along the direction (sin, 0, cos) at this height
    d = np.array([math.sin(ang), 0.0, math.cos(ang)])
    lo, hi = 0.0, 0.35
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if sk.dist(np.array([[0, y, zc]]) + d * mid)[0] < 0:
            lo = mid
        else:
            hi = mid
    c = np.array([0, y, zc]) + d * (lo + 0.024)
    x = c[0]
    body = sdf.box(c, (0.036, 0.044, 0.022), radius=0.012)
    rot = m3.quat_from_axis_angle((0, 1, 0), 66)
    M = m3.translation_matrix(c) @ m3.rotation_matrix(rot) @ m3.translation_matrix(-c)
    body = sdf.Transformed(body, M)
    flap = sdf.Transformed(sdf.box(c + np.array([0, 0.018, 0.016]), (0.036, 0.024, 0.006), radius=0.005), M)
    button = sdf.sphere(c + d * 0.027 + np.array([0, 0.004, 0]), 0.0055)
    shape = body.smooth_union(flap, 0.003).paint(color=(1, 1, 1), material=slot("Leather"))
    _ = x
    return shape.union(button.paint(color=(1, 1, 1), material=slot("Metal")))


# ----------------------------------------------------------------------------- legs

def trousers():
    legs = sdf.union(B.leg_tube(1), B.leg_tube(-1))
    hips = sdf.smooth_union(B.torso(), legs, k=0.045)
    # thinner under the tunic skirt, so the thighs never show through it
    t0 = 0.0065 if FEMALE else 0.01
    shape = Offset(hips, lambda p: t0 - 0.003 * _ss(0.82, 0.9, p[:, 1]), t_max=0.01)
    shape = _cut(shape, (0, 1.02, 0), (0, 1, 0), 0.01)
    shape = _cut(shape, (0, BOOT_TOP - 0.06, 0), (0, -1, 0), 0.006)
    return shape.paint(color=(1, 1, 1), material=slot("Cloth_B"))


def boots():
    parts = []
    for s in (1, -1):
        leg = B.leg_tube(s).smooth_union(B.foot(s), 0.03)

        def t_boot(p):
            return 0.0085 + 0.004 * _ss(0.2, BOOT_TOP, p[:, 1])  # a little wider at the top

        shaft = Offset(leg, t_boot, t_max=0.02)
        shaft = _cut(shaft, (0, BOOT_TOP, 0), (0, 1, 0), 0.008)
        shaft = shaft.smooth_intersect(sdf.plane((0, 0.0, 0), (0, -1, 0), extent=2.0), 0.006)
        # folded cuff: a wider band turned down over the top of the shaft
        x = B.HIP[0] * s
        g = 0.9 if FEMALE else 1.0
        cuff = Tube((x, BOOT_TOP + 0.012, -0.004), (x, BOOT_TOP - 0.046, -0.008), {0: 0.058 * g, 1: 0.055 * g},
                    {0: 0.058 * g, 1: 0.055 * g}, k={0: 0.062 * g, 1: 0.059 * g}, cap=0.008)
        cuff = cuff.smooth_subtract(sdf.capsule((x, BOOT_TOP - 0.02, -0.006), (x, BOOT_TOP + 0.2, -0.006),
                                                0.044 * g), 0.004)
        parts.append(shaft.smooth_union(cuff, 0.004))
    return sdf.union(*parts).paint(color=(1, 1, 1), material=slot("Leather"))


def bracers():
    parts = []
    for s in (1, -1):
        p0 = arm_point(UPPER_ARM + FOREARM * BRACER[0], s)
        p1 = arm_point(UPPER_ARM + FOREARM * BRACER[1], s)
        ax = arm_dir(s)

        def t_br(p, s=s):
            d, _ = _along_arm(p)
            return 0.017 - 0.004 * _ss(UPPER_ARM + FOREARM * 0.55, UPPER_ARM + FOREARM * 0.95, d)
        shape = Offset(B._arm_frame(B.arm_tube(), s), t_br, t_max=0.02)
        shape = _cut(_cut(shape, p1, ax, 0.005), p0, -ax, 0.005)
        parts.append(shape)
    return sdf.union(*parts).paint(color=(1, 1, 1), material=slot("Leather"))


GARMENTS = [  # (name, builder); mesh budgets are in build_characters.BUDGET
    ("Tunic", tunic),
    ("Collar", collar),
    ("Trousers", trousers),
    ("Boots", boots),
    ("Belt", belt),
    ("Bracers", bracers),
] + ([("Underskirt", underskirt)] if FEMALE else [])
if HARPIN:  # plumed legs and taloned feet: no trousers, no boots
    GARMENTS = [g for g in GARMENTS if g[0] not in ("Trousers", "Boots")]
