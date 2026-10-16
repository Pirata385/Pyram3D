"""Harpin-specific anatomy: plumed eagle legs with talons, feathered tail, collar ruff and the
huge wings (feather cards on a plumed wing arm). Rest pose: wings folded behind the back,
wrists arching above the head, primaries reaching down to the ankles.
"""
from __future__ import annotations

import math

import numpy as np

from common import BodySpec, joints, mirror, slot, tail_points

import pyram3d as p3
from pyram3d import sdf
from pyram3d.noise import fbm3


def plume(shape, shade=1.0):
    g = float(shade)
    return shape.paint(color=(g, g, g), material=slot("Plume"))


def _tufts(amount, scale=26.0, seed=0):
    """Displacement making a surface read as overlapping plumes (soft lumps, mostly outward)."""
    def f(p):
        n = fbm3(p * scale, octaves=2, seed=seed)
        return -np.clip(n * 1.6 + 0.35, -0.4, 1.0)  # negative = grow outward
    return dict(amount=amount, func=f)


def legs(spec: BodySpec):
    J = joints(spec)
    L = spec.limb
    hip, knee = J["LeftUpperLeg"]
    ankle = J["LeftLowerLeg"][1]
    ball = J["LeftFoot"][1]
    toe = J["LeftToes"][1]
    parts = []
    # feathered thigh (mostly under the tunic: kept slim so the skirt stays an A-line, ref. 06)
    thigh = plume(sdf.round_cone(hip + np.array([0, -0.05, -0.005]), knee + np.array([0, -0.02, 0.0]), 0.086 * L, 0.068 * L))
    thigh = thigh.smooth_union(plume(sdf.ellipsoid((hip + knee) / 2 + np.array([0.008, -0.03, 0.0]),
                                                   (0.08 * L, 0.13 * spec.s, 0.085 * L))), 0.04)
    parts.append(thigh.displace(**_tufts(0.01, 24, 1)))
    # slim plumed shin and tarsus with a frayed cuff above the toes
    # shin: full eagle "trousers" (thick plumes) ending in a fan of feathers over the tarsus
    parts.append(plume(sdf.round_cone(knee, ankle, 0.074 * L, 0.05 * L), 0.95).displace(**_tufts(0.012, 30, 2)))
    down = (ankle - knee) / np.linalg.norm(ankle - knee)
    for k, ang in enumerate(np.linspace(-150, 150, 7)):
        a = math.radians(ang)
        out = np.array([math.sin(a), 0.0, math.cos(a)])
        root = ankle - down * 0.07 + out * 0.035 * L
        tip = root + down * 0.11 + out * 0.045
        parts.append(plume(sdf.round_cone(root, tip, 0.024 * L, 0.004), 0.9 + 0.06 * (k % 2)))
    parts.append(plume(sdf.sphere(ankle, 0.045 * L), 0.92))
    parts.append(plume(sdf.round_cone(ankle, ball + np.array([0, 0.035, 0]), 0.037 * L, 0.03 * L), 0.95).displace(**_tufts(0.007, 40, 3)))
    parts.append(plume(sdf.ellipsoid(ball + np.array([0, 0.05, -0.005]), (0.05 * L, 0.04, 0.048 * L)), 0.88).displace(**_tufts(0.014, 30, 4)))
    return [p.mirror_x() for p in parts]


def feet(spec: BodySpec):
    """Scaly toes (skin slot) and dark talons: 3 forward, 1 back."""
    J = joints(spec)
    ball = J["LeftFoot"][1]
    toe_len = np.linalg.norm(J["LeftToes"][1] - ball)
    parts = []
    skin = lambda s: s.paint(color=(0.86, 0.82, 0.7), material=slot("Skin"))  # noqa: E731
    talon = lambda s: s.paint(color=(1, 1, 1), material=slot("Talon"))  # noqa: E731
    for ang, ln in ((-24, 0.92), (0, 1.0), (24, 0.9)):
        a = math.radians(ang)
        d = np.array([math.sin(a), 0, math.cos(a)])
        tip = ball + d * toe_len * ln
        tip[1] = 0.018
        parts.append(skin(sdf.round_cone(ball + np.array([0, 0.01, 0]), tip, 0.018, 0.013)))
        parts.append(talon(sdf.round_cone(tip, tip + d * 0.035 + np.array([0, -0.014, 0]), 0.011, 0.002)))
    back = ball + np.array([0, -0.012, -0.065])
    parts.append(skin(sdf.round_cone(ball, back, 0.017, 0.012)))
    parts.append(talon(sdf.round_cone(back, back + np.array([0, -0.013, -0.03]), 0.01, 0.002)))
    return [p.mirror_x() for p in parts]


def tail(spec: BodySpec):
    """Long mammal tail covered in plumes: thick at the root (r 0.11), 0.08 in the middle, ending
    in a fox-like brush (r ~0.18, ~0.45 m) that rests on the ground and curls up."""
    pts = tail_points(spec)
    radii = [0.11, 0.095, 0.085, 0.08, 0.085, 0.09, 0.05]
    shape = plume(sdf.chain([tuple(p) for p in pts], radii, k=0.04), 0.8)
    a, b, c = pts[-3], pts[-2], pts[-1]
    d = (c - b) / np.linalg.norm(c - b)
    # fox brush (~0.45 m): widest (r 0.13) past the middle, tapering to a point, lighter tip
    brush = plume(sdf.chain([tuple(a + (b - a) * 0.3), tuple(b + (c - b) * 0.2), tuple(c + d * 0.04)],
                            [0.095, 0.13, 0.06], k=0.06), 0.88)
    tip = plume(sdf.round_cone(tuple(c), tuple(c + d * 0.17), 0.06, 0.008), 1.0)
    shape = shape.smooth_union(brush, 0.05).smooth_union(tip, 0.03)
    return shape.displace(**_tufts(0.012, 24, 7))


def ruff(spec: BodySpec):
    """Feather collar around the neck and across the shoulders (ref. 06)."""
    s = spec.s
    r = plume(sdf.ellipsoid((0, 1.44 * s, -0.012), (0.115, 0.045, 0.098)), 1.0)
    r = r.smooth_union(plume(sdf.ellipsoid((0.085, 1.425 * s, -0.01), (0.07, 0.04, 0.075)), 1.0).mirror_x(), 0.03)
    return r.displace(**_tufts(0.016, 30, 9))


def head_plumes(spec: BodySpec):
    """Feather tufts at the temples, sweeping back like small wings (ref. 06)."""
    s = spec.s
    parts = []
    for k, (dy, ln, w) in enumerate(((0.0, 0.13, 0.03), (0.03, 0.15, 0.028), (0.06, 0.12, 0.024))):
        a = np.array([0.075, 1.66 * s + dy, -0.0])
        b = a + np.array([0.06, 0.07 + dy * 0.8, -0.1]) * (ln / 0.13)
        parts.append(plume(sdf.round_cone(a, b, w, 0.004), 1.0 - 0.04 * k))
    return [p.mirror_x() for p in parts]


# ----------------------------------------------------------------------------- wings

def feather(root, tip, width, normal, cup=0.12, curl=0.0, asym=0.15, segs=6, tone=1.0, tip_band=0.72,
            rachis_dark=False):
    """One flat feather card (double sided material) with painted rachis and tip band."""
    root, tip, normal = (np.asarray(v, float) for v in (root, tip, normal))
    d = tip - root
    length = np.linalg.norm(d)
    d /= length
    side = np.cross(normal, d)
    side /= np.linalg.norm(side)
    nrm = np.cross(d, side)
    V, C, F = [], [], []
    for i in range(segs + 1):
        t = i / segs
        c = root + d * length * t + nrm * curl * length * math.sin(math.pi * t) * 0.5
        w = width * 0.5 * max(math.sin(math.pi * min(1.0, t ** 0.62)), 0.0) ** 0.75
        if i == 0:
            w = width * 0.12
        lft = c - side * w * (1 - asym) + nrm * cup * w
        rgt = c + side * w * (1 + asym) + nrm * cup * w
        band = tip_band if t > 0.8 else (0.93 if t > 0.62 else 1.0)
        V += [lft, c, rgt]
        # dark trailing edge, light vane: overlapping feathers read as stripes even at game scale
        mid = tone * band * 0.55 if rachis_dark else tone * min(1.0, band * 1.08)
        C += [[tone * band * 0.58] * 3 + [1.0], [mid] * 3 + [1.0], [tone * band * 0.94] * 3 + [1.0]]
    for i in range(segs):
        a, b = i * 3, (i + 1) * 3
        F += [(a, a + 1, b + 1), (a, b + 1, b), (a + 1, a + 2, b + 2), (a + 1, b + 2, b + 1)]
    m = p3.Mesh(np.array(V), F, colors=np.array(C), smooth_angle=180)
    return m


def _lerp(a, b, t):
    return a + (b - a) * t


def wing_points(spec: BodySpec):
    J = joints(spec)
    S, E = J["LeftWing1"]
    W = J["LeftWing2"][1]
    T = J["LeftWing3"][1]
    return S, E, W, T


def wing_left(spec: BodySpec, sk: p3.Skeleton):
    """Left wing, skinned to Chest/LeftWing1..3: a thick plumed arm and covert mass (a teardrop
    section 0.15-0.2 m deep, so the folded wing never reads as a flat board or a needle from
    the iso camera) with long flight-feather cards hanging from it."""
    S, E, W, T = wing_points(spec)
    normal = np.cross(W - S, T - S)
    normal /= np.linalg.norm(normal)
    if normal[2] < 0:
        normal = -normal
    down = T - W
    down /= np.linalg.norm(down)
    inward = np.cross(normal, down)  # in-plane, perpendicular to the primaries
    if inward[0] > 0:
        inward = -inward

    def in_plane(v):
        v = np.asarray(v, float) - normal * np.dot(v, normal)
        return v / np.linalg.norm(v)

    pieces = []  # (mesh, bone name)
    # primaries: 8 long cards from the hand, tips fanned around T (outer = longest)
    # tips stepped ~9 cm apart along the trailing edge so the silhouette reads serrated
    n_p = 8
    for k in range(n_p):
        t = k / (n_p - 1)
        root = W + down * 0.08 + inward * (0.02 + 0.07 * t) - normal * 0.015 * (1 - t)
        tip = T + inward * (0.3 * t) + np.array([0, 0.62 * spec.s * t ** 1.15, 0]) - normal * 0.008 * k
        pieces.append((feather(root, tip, 0.14 - 0.012 * t, normal, cup=0.1, curl=0.04, asym=-0.2,
                               tone=(0.74 if k % 2 else 0.86) * (0.94 + 0.06 * t), tip_band=0.6), "LeftWing3"))
    # secondaries: 8 cards hanging from the forearm, shorter toward the body
    arm = [S + (E - S) * 0.3, E, W]
    sec_dir = in_plane(down * 0.55 + np.array([0, -1.0, 0]) * 0.45)
    n_s = 8
    for k in range(n_s):
        t = k / (n_s - 1)
        root = _lerp(arm[1], arm[2], 1 - 2 * t) if t < 0.5 else _lerp(arm[0], arm[1], 2 - 2 * t)
        root = root - normal * 0.01 + sec_dir * 0.06
        ln = _lerp(0.62, 0.38, t) * spec.s  # short: below them only the primaries remain (taper)
        tip = root + sec_dir * ln + inward * 0.03 * t
        bone = "LeftWing2" if t < 0.4 else "LeftWing1"
        pieces.append((feather(root, tip, 0.14, normal, cup=0.12, curl=0.03, asym=0.1,
                               tone=(0.72 if k % 2 else 0.84) * (0.9 + 0.08 * t), tip_band=0.64), bone))

    meshes = []
    names = {b.name: i for i, b in enumerate(sk.bones)}
    for m, bone in pieces:
        m.joints = np.zeros((m.num_vertices, 4), np.int64)
        m.weights = np.zeros((m.num_vertices, 4))
        m.joints[:, 0] = names[bone]
        m.weights[:, 0] = 1.0
        meshes.append(m)
    feathers = p3.Mesh.join(meshes, name="Wing_L_Feathers")
    mat = p3.Material("Plume_Feather", "#ecebe6", roughness=0.85, double_sided=True)
    feathers.set_material(mat)

    # plumed arm (leading edge, thick "shoulder" at head height) and the covert mass: rows of
    # plumes rotated around the arm so the folded wing has a teardrop section
    # thin toward the wrist so the carpal joint reads as an angle, not a dome
    arm_pts = [S + (E - S) * 0.15, E, W, W + down * 0.1]
    parts = [plume(sdf.chain([tuple(p) for p in arm_pts], [0.065, 0.075, 0.045, 0.03], k=0.03))]
    for k in range(7):
        t = k / 6
        p = _lerp(arm[1], arm[2], 1 - 2 * t) if t < 0.5 else _lerp(arm[0], arm[1], 2 - 2 * t)
        for side, rot in ((1, 0.5), (-1, 0.3)):  # front and back faces flare apart (+-20..30 deg)
            d = in_plane(sec_dir) * np.cos(rot) + normal * side * np.sin(rot)
            ln = _lerp(0.3, 0.24, t)
            parts.append(plume(sdf.round_cone(tuple(p), tuple(p + d * ln), 0.055, 0.016), 1.0))
    # coverts one tone lighter than the flight feathers, scale rows painted (paint_plumes);
    # only a trace of displacement (lumps read as cotton)
    arm_sdf = sdf.smooth_union(*parts, k=0.04).displace(**_tufts(0.003, 26, 11))
    arm_mesh = arm_sdf.to_mesh(cell=0.02, name="Wing_L_Arm")
    wing_bones = [names["Chest"], names["LeftWing1"], names["LeftWing2"], names["LeftWing3"]]
    sk.bind(arm_mesh, smooth=4, falloff=6.0, bone_filter=lambda P, i: np.full(len(P), i in wing_bones))
    return feathers, arm_mesh


def mirror_skinned(mesh: p3.Mesh, sk: p3.Skeleton, name: str) -> p3.Mesh:
    """Mirror a left-side skinned mesh to the right side (X flip, winding flip, Left->Right bones)."""
    m = mesh.copy(name=name)
    m.vertices = m.vertices * np.array([-1, 1, 1])
    m.flip()
    names = [b.name for b in sk.bones]
    remap = np.array([names.index(n.replace("Left", "Right")) if n.startswith("Left") else i
                      for i, n in enumerate(names)])
    m.joints = remap[m.joints]
    return m
