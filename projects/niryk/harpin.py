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
    # feathered thigh "trousers": full and fluffy
    thigh = plume(sdf.round_cone(hip + np.array([0, 0.02, -0.01]), knee + np.array([0, -0.02, 0.0]), 0.108 * L, 0.075 * L))
    thigh = thigh.smooth_union(plume(sdf.ellipsoid((hip + knee) / 2 + np.array([0.012, -0.04, 0.0]),
                                                   (0.1 * L, 0.15 * spec.s, 0.1 * L))), 0.04)
    parts.append(thigh.displace(**_tufts(0.014, 24, 1)))
    # slim plumed shin and tarsus with a frayed cuff above the toes
    parts.append(plume(sdf.round_cone(knee, ankle, 0.056 * L, 0.036 * L), 0.95).displace(**_tufts(0.008, 34, 2)))
    parts.append(plume(sdf.sphere(ankle, 0.04 * L), 0.92))
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
    pts = tail_points(spec)
    radii = np.linspace(0.05, 0.03, len(pts))
    shape = plume(sdf.chain([tuple(p) for p in pts], list(radii), k=0.03), 0.95)
    # plume tuft at the tip: elongated leaf-shaped fluff
    a, b = pts[-2], pts[-1]
    d = (b - a) / np.linalg.norm(b - a)
    tip = plume(sdf.round_cone(b - d * 0.1, b + d * 0.2, 0.075, 0.012), 1.0)
    shape = shape.smooth_union(tip, 0.04)
    return shape.displace(**_tufts(0.013, 28, 7))


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

def feather(root, tip, width, normal, cup=0.12, curl=0.0, asym=0.15, segs=6, tone=1.0, tip_band=0.72):
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
        C += [[tone * band * 0.86] * 3 + [1.0], [tone * min(1.0, band * 1.08)] * 3 + [1.0], [tone * band * 0.86] * 3 + [1.0]]
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
    """Left wing mesh (feathers + plumed arm), skinned to Chest/LeftWing1..3."""
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
    pieces = []  # (mesh, bone name)

    # primaries: roots fan along the wrist/hand, tips fan around T (outer = longest)
    n_p = 10
    for k in range(n_p):
        t = k / (n_p - 1)
        root = W + down * 0.06 + inward * (0.02 + 0.07 * t) - normal * 0.01 * (1 - t)
        tip = T + inward * (0.26 * t) + np.array([0, 0.3 * spec.s * t ** 1.4, 0]) - normal * 0.006 * k
        pieces.append((feather(root, tip, 0.13 - 0.015 * t, normal, cup=0.1, curl=0.04, asym=-0.2,
                               tone=0.94 + 0.06 * t), "LeftWing3"))
    # secondaries: hang from the arm, shorter toward the body
    n_s = 11
    arm = [S + (E - S) * 0.25, E, W]
    for k in range(n_s):
        t = k / (n_s - 1)
        root = _lerp(arm[1], arm[2], 1 - 2 * t) if t < 0.5 else _lerp(arm[0], arm[1], 2 - 2 * t)
        root = root - normal * 0.004
        tip_y = _lerp(0.82, 1.12, t) * spec.s
        tip = np.array([root[0] + 0.03 - 0.02 * t, tip_y, root[2] - 0.02]) + inward * 0.02
        bone = "LeftWing2" if t < 0.4 else "LeftWing1"
        pieces.append((feather(root, tip, 0.13, normal, cup=0.12, curl=0.03, asym=0.1, tone=0.9 + 0.08 * t),
                       bone))
    # coverts: two rows of short feathers over the secondaries' roots (front and back faces)
    for row, (ln, off, nf) in enumerate(((0.36, 0.012, 12), (0.2, 0.022, 10))):
        for k in range(nf):
            t = k / (nf - 1)
            root = _lerp(arm[1], arm[2], 1 - 2 * t) if t < 0.5 else _lerp(arm[0], arm[1], 2 - 2 * t)
            for sgn in (1, -1):
                r = root + normal * off * sgn + np.array([0, -0.01, 0])
                tip = r + np.array([0.01, -ln * spec.s, 0]) + inward * 0.03
                bone = "LeftWing2" if t < 0.45 else "LeftWing1"
                pieces.append((feather(r, tip, 0.1 - 0.02 * row, normal * sgn, cup=0.15, asym=0.05,
                                       tone=0.84 + 0.08 * row, tip_band=0.82), bone))
    # alula / wrist tuft
    pieces.append((feather(W + np.array([0, 0.01, 0]), W + np.array([0.06, -0.16, 0.0]) + inward * 0.04, 0.08,
                           normal, tone=0.9), "LeftWing2"))

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

    # plumed wing arm (leading edge)
    arm_sdf = plume(sdf.chain([tuple(S), tuple(E), tuple(W), tuple(W + down * 0.1)], [0.055, 0.045, 0.034, 0.026], k=0.03))
    arm_sdf = arm_sdf.displace(**_tufts(0.01, 30, 11))
    arm_mesh = arm_sdf.to_mesh(cell=0.016, name="Wing_L_Arm")
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
