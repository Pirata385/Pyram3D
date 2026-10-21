"""Niryk prototype 2: the wolf (the only enemy), smooth stylized toon wildlife with its own rig and clips.

    python projects/niryk/protótipo_2/wolf.py [--fast] [--no-review] [--no-cache]

Writes exports/creatures/wolf.glb and review/creatures/wolf_*.png (+ wolf_stats.json).

Look: the same family as the anime characters and the toon nature set (Okami / Genshin wildlife):
a clean silhouette of few large smooth forms (profile tubes, shapes.Tube) with wide blends, an
angular-but-smooth head with a long squared muzzle, big pointed ears, a thick neck ruff and cheek
fluff made of pointed locks (shapes.Strand, like the hair), slim strong legs with big paws, a bushy
tail ending in a point. Gray coat with a darker saddle and mask, cream belly, throat, muzzle and
brows, a dark tail tip, yellow eyes. Colors are vertex colors on untinted white materials.

GLB contents (the client: src/client/character/WolfView.ts):
  WolfRig    armature (bones below) - its root is Hips
  Wolf       body, ruff, tail, teeth (one skinned mesh)          [Fur]  (inked by the client)
  Wolf_Eyes  eyes: yellow iris, dark pupil, white highlight       [Eye]  (no ink: name starts with Eye)
  clips      Idle (2.4 s loop), Walk (0.56 s loop, a trot at WOLF.walkSpeed 2.0 m/s), Run (0.4 s loop,
             a gallop at RUN_SPEED 5.0 m/s: the client plays it at speed / 5), Attack (0.75 s: crouch,
             lunge, the bite closes at 0.42 s = WOLF.windup), Hit (0.36 s), Death (1.2 s, rolls onto
             its right side and stays there)
Bones: Hips (root) > Spine > Chest > Neck > Head > Jaw, LeftEar, RightEar; Hips > Tail1..4;
Chest > Front{Left,Right}{Upper,Lower,Foot,Toes}; Hips > Back{Left,Right}{Upper,Lower,Foot,Toes}
(Upper = humerus / femur, Lower = forearm / shin, Foot = metacarpus / metatarsus, Toes).
Conventions: meters, +Y up, the nose points +Z, the wolf's left is +X, paws on y = 0, the origin
under the middle of the body. Gaits: planted paws move back at exactly the clip's speed (two-bone
IK to the wrist / hock, the paw rolling on its ball), so they never skate.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import common  # noqa: F401  (puts the Pyram3D root on sys.path)
from common import EXPORTS, REVIEW, review_lights
from meshing import mesh_sdf, sizing
from shapes import Strand, Tube

import pyram3d as p3
from pyram3d import math3d as m3
from pyram3d import sdf

FAST = "--fast" in sys.argv
CELL = 0.0058 if FAST else 0.0042
BUDGET = 8600  # body; + eyes (~0.8k) + teeth (~0.2k) <= 10k
FPS = 30
WALK_SPEED, RUN_SPEED = 2.0, 5.0  # src/shared/constants.ts: WOLF.walkSpeed; RUN_SPEED (the client's Run time scale)
DUR = {"Idle": 2.4, "Walk": 0.56, "Run": 0.4, "Attack": 0.75, "Hit": 0.36, "Death": 1.2}
LOOPING = ("Idle", "Walk", "Run")
BITE = 0.42  # WOLF.windup: the jaws close on the target
HERE = Path(__file__).resolve().parent
OUT_REVIEW = REVIEW / "creatures"

FUR = p3.Material("Fur", "#ffffff", roughness=0.9)
EYE = p3.Material("Eye", "#ffffff", roughness=0.35)


def ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def V(*a):
    return np.array(a, float)


def mx(p, s):
    """Point p on side s (+1 left, -1 right)."""
    p = np.array(p, float).copy()
    p[0] *= s
    return p


# ============================================================================= MODEL
# design joints (left side; mirrored for the right)
FRONT = {"Upper": V(0.11, 0.63, 0.33), "Lower": V(0.11, 0.42, 0.212), "Foot": V(0.11, 0.13, 0.255),
         "Toes": V(0.11, 0.045, 0.3), "tip": V(0.11, 0.02, 0.39)}
BACK = {"Upper": V(0.1, 0.66, -0.37), "Lower": V(0.105, 0.42, -0.25), "Foot": V(0.105, 0.19, -0.435),
        "Toes": V(0.105, 0.045, -0.375), "tip": V(0.105, 0.02, -0.295)}
HEAD_C = V(0, 0.915, 0.58)  # skull center
NECK_A, NECK_B = V(0, 0.71, 0.29), V(0, 0.885, 0.53)  # neck tube axis
SPINE = {"Hips": (V(0, 0.7, -0.34), V(0, 0.72, -0.12)), "Spine": (V(0, 0.72, -0.12), V(0, 0.74, 0.1)),
         "Chest": (V(0, 0.74, 0.1), V(0, 0.76, 0.32)), "Neck": (V(0, 0.76, 0.32), V(0, 0.875, 0.52)),
         "Head": (V(0, 0.875, 0.52), V(0, 0.885, 0.8))}
MOUTH_Y = HEAD_C[1] - 0.067  # the mouth slit (closed); the corner is at MOUTH_Z
MOUTH_Z = HEAD_C[2] + 0.085
SPINE["Jaw"] = (V(0, MOUTH_Y - 0.004, MOUTH_Z - 0.035), V(0, MOUTH_Y - 0.02, 0.83))
EAR = (V(0.058, 0.975, 0.548), V(0.1, 1.135, 0.522))  # left ear: base -> tip
TAIL = [V(0, 0.745, -0.49), V(0, 0.71, -0.58), V(0, 0.62, -0.68), V(0, 0.5, -0.755), V(0, 0.37, -0.79)]
TAIL_TIP = V(0, 0.27, -0.798)
TORSO = [  # z, back top, belly bottom, half width
    (-0.55, 0.72, 0.64, 0.05), (-0.47, 0.77, 0.575, 0.11), (-0.37, 0.79, 0.545, 0.134), (-0.25, 0.785, 0.545, 0.13),
    (-0.13, 0.775, 0.545, 0.126), (0.0, 0.782, 0.515, 0.132), (0.12, 0.805, 0.465, 0.148), (0.24, 0.825, 0.43, 0.16),
    (0.34, 0.815, 0.445, 0.152), (0.42, 0.775, 0.5, 0.122), (0.49, 0.72, 0.57, 0.06)]
TORSO_YC = 0.68


def H(x, y, z):
    """Head-space offset -> design point."""
    return HEAD_C + V(x, y, z)


def torso():
    """Rump -> loin (tuck-up) -> deep chest -> breast, one profile tube along +Z (front = up)."""
    z0, z1, yc = TORSO[0][0] - 0.02, TORSO[-1][0], TORSO_YC
    T = lambda z: (z - z0) / (z1 - z0)  # noqa: E731
    f = {T(z): top - yc for z, top, _, _ in TORSO}
    k = {T(z): yc - bot for z, _, bot, _ in TORSO}
    w = {T(z): hw for z, _, _, hw in TORSO}
    return Tube((0, yc, z0), (0, yc, z1), w, f, k=k, n=2.15, front=(0, 1, 0), cap=0.04)


def neck_frame():
    u = (NECK_B - NECK_A) / np.linalg.norm(NECK_B - NECK_A)
    nape = np.cross(u, (1, 0, 0))
    nape = nape if nape[1] > 0 else -nape
    return u, nape


def neck():
    u, nape = neck_frame()
    return Tube(NECK_A, NECK_B, {0: 0.112, 0.5: 0.095, 1: 0.08}, {0: 0.09, 1: 0.078},
                k={0: 0.13, 0.5: 0.1, 1: 0.078}, n=2.1, front=nape, cap=0.04)


def head():
    """Skull with cheekbones and brows, a squared muzzle (flat bridge) over a slimmer lower jaw, a
    rounded nose, the mouth slit between them (the jaw opens there)."""
    skull = sdf.ellipsoid(H(0, 0, -0.005), (0.094, 0.08, 0.104))
    cheeks = sdf.ellipsoid(H(0, -0.042, 0.016), (0.104, 0.058, 0.085))
    core = skull.smooth_union(cheeks, 0.03)
    brow = sdf.ellipsoid(H(0.045, 0.036, 0.072), (0.034, 0.017, 0.03)).mirror_x()
    core = core.smooth_union(brow, 0.02)
    muzzle = Tube(H(0, -0.012, 0.045), H(0, -0.038, 0.248), {0: 0.062, 0.5: 0.05, 1: 0.039},
                  {0: 0.05, 0.4: 0.04, 1: 0.032}, k={0: 0.04, 1: 0.027}, n={0: 2.4, 1: 2.9},
                  front=(0, 1, 0), cap=0.022)
    jaw = Tube(H(0, -0.084, 0.04), H(0, -0.084, 0.232), {0: 0.054, 1: 0.03}, {0: 0.02, 1: 0.019},
               k={0: 0.036, 0.6: 0.028, 1: 0.021}, n=2.3, front=(0, 1, 0), cap=0.016)
    nose = sdf.box(H(0, -0.03, 0.252), (0.05, 0.042, 0.042), radius=0.014)
    h = core.smooth_union(muzzle, 0.035).smooth_union(jaw, 0.02).smooth_union(nose, 0.008)
    slit = sdf.box((0, MOUTH_Y, MOUTH_Z + 0.15), (0.15, 0.0036, 0.3), radius=0.0015)
    return h.smooth_subtract(slit, 0.003)


def ear(s):
    """Big pointed ear (lens-section strand) with a hollow front."""
    a, b = mx(EAR[0], s), mx(EAR[1], s)
    mid = a + (b - a) * 0.5 + V(0, 0, 0.004)
    pts = m3.catmull_rom([a - (b - a) * 0.12, a, mid, b], 6)
    n = len(pts)
    u = np.linspace(0, 1, n)
    back = np.tile(V(0, 0.18, -1.0), (n, 1))
    w = 0.05 * (1 - (1 - 0.06) * ss(0.25, 1.0, u) ** 0.9) * (0.85 + 0.15 * ss(0, 0.25, u))
    f = 0.018 * (1 - 0.6 * u)
    shell = Strand(pts, back, w, f, f * 0.8)
    cav_pts = pts + V(0, 0.004, 0.013)
    cav = Strand(cav_pts, back, w * 0.66, f * 0.7, f * 0.7)
    cut = sdf.plane(a + V(0, 0.02, 0), (0, -1, 0), extent=1.0)  # keep the cavity off the base
    return shell.smooth_subtract(cav.intersect(cut), 0.004)


def lock(pts, outward, w, f, k=None, root=0.75, peak=0.3, tip=0.05, samples=6):
    """Pointed fur lock through pts (smooth); outward: one vector or one per point."""
    Q = m3.catmull_rom(np.asarray(pts, float), samples)
    seg = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    u = np.concatenate([[0.0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    grow = root + (1 - root) * ss(0, peak, u)
    taper = 1 - (1 - tip) * np.clip((u - peak) / (1 - peak), 0, 1) ** 1.2
    o = np.asarray(outward, float)
    if o.ndim == 1:
        o = np.tile(o, (len(Q), 1))
    else:
        o = np.array([np.interp(u, np.linspace(0, 1, len(o)), o[:, i]) for i in range(3)]).T
    F = f * (1 - 0.55 * np.clip((u - 0.4) / 0.6, 0, 1))
    return Strand(Q, o, w * grow * taper, F, F * ((k / f) if k else 0.7))


RUFF = [  # (theta deg around the neck: 0 top, 90 left side, 180 throat), root t along the neck, length
    (18, 0.9, 0.2), (55, 0.92, 0.2), (95, 0.9, 0.2), (135, 0.82, 0.21), (165, 0.7, 0.22),
    (35, 0.55, 0.19), (78, 0.55, 0.2), (118, 0.5, 0.2)]


def ruff():
    """Thick neck ruff and cheek fluff: two rings of pointed locks around the neck sweeping back over
    the shoulders (the throat ones hang down as a pointed bib), and fluff behind the jaw corners."""
    u, top = neck_frame()
    X = V(1, 0, 0)
    locks = []
    rings = [(th * s, t, L) for th, t, L in RUFF for s in (1, -1)] + [(180, 0.72, 0.24), (0, 0.6, 0.18)]
    for th, t, L in rings:
        a = math.radians(th)
        rad = top * math.cos(a) + X * math.sin(a)
        throat = ss(100, 175, abs(th))
        flow = -u * (1 - throat) + V(0, -1, -0.22) * throat
        flow = flow + V(0, -0.35, 0) * ss(40, 120, abs(th)) * (1 - throat)
        flow /= np.linalg.norm(flow)
        c = NECK_A + (NECK_B - NECK_A) * t
        r0 = 0.08 - 0.012 * throat
        root = c + rad * r0
        mid = root + flow * L * 0.5 + rad * 0.06
        tip = root + flow * L + rad * 0.075
        locks.append(lock([root, mid, tip], rad, 0.064, 0.034, k=0.03, root=0.75, peak=0.3))
    for s in (1, -1):  # cheek fluff behind the jaw corner
        locks.append(lock([H(s * 0.07, -0.05, 0.0), H(s * 0.108, -0.085, -0.055), H(s * 0.132, -0.115, -0.115)],
                          V(s, -0.25, 0.2), 0.038, 0.017))
        locks.append(lock([H(s * 0.07, -0.08, 0.01), H(s * 0.098, -0.115, -0.04), H(s * 0.118, -0.15, -0.09)],
                          V(s, -0.6, 0.3), 0.032, 0.015))
    for s in (1, -1):  # elbow tufts and the "breeches" at the back of the thighs
        el = mx(FRONT["Lower"], s)
        locks.append(lock([el + mx((0.012, 0.03, 0.0), s), el + mx((0.03, -0.0, -0.05), s), el + mx((0.032, -0.05, -0.085), s)],
                          mx((1, 0, -0.6), s), 0.034, 0.016))
        kn = mx(BACK["Lower"], s)
        locks.append(lock([kn + mx((0.0, 0.16, -0.16), s), kn + mx((0.008, 0.07, -0.2), s), kn + mx((0.004, -0.02, -0.2), s)],
                          mx((0.3, 0, -1), s), 0.05, 0.024))
    return sdf.union(*locks)


def leg_front(s):
    sh, el, wr, ball = (mx(FRONT[k], s) for k in ("Upper", "Lower", "Foot", "Toes"))
    upper = Tube(sh + V(0, 0.04, 0.01), el, {0: 0.075, 1: 0.05}, {0: 0.095, 1: 0.052}, k={0: 0.08, 1: 0.055},
                 front=(0, 0, 1), cap=0.03)
    elbow = sdf.ellipsoid(el + V(0, 0.0, -0.014), (0.04, 0.045, 0.038))
    fore = Tube(el, wr, {0: 0.046, 0.3: 0.041, 1: 0.031}, {0: 0.05, 0.3: 0.045, 1: 0.032},
                k={0: 0.047, 0.5: 0.037, 1: 0.03}, front=(0, 0, 1), cap=0.02)
    past = sdf.round_cone(wr, ball, 0.031, 0.029)
    leg = upper.smooth_union(elbow, 0.03).smooth_union(fore, 0.03).smooth_union(past, 0.022)
    return leg.smooth_union(paw(ball, s, 1.0), 0.016)


def leg_back(s):
    hip, kn, hk, ball = (mx(BACK[k], s) for k in ("Upper", "Lower", "Foot", "Toes"))
    thigh = Tube(hip + V(0, 0.03, 0.0), kn, {0: 0.085, 0.5: 0.07, 1: 0.047}, {0: 0.11, 0.45: 0.078, 1: 0.05},
                 k={0: 0.115, 0.5: 0.092, 1: 0.05}, front=(0, 0, 1), cap=0.03)
    shin = Tube(kn, hk, {0: 0.047, 1: 0.032}, {0: 0.055, 0.5: 0.044, 1: 0.033}, k={0: 0.052, 0.5: 0.047, 1: 0.037},
                front=(0, 0, 1), cap=0.02)
    hock = sdf.sphere(hk + V(0, 0, -0.008), 0.033)
    meta = sdf.round_cone(hk, ball, 0.03, 0.028)
    leg = thigh.smooth_union(shin, 0.035).smooth_union(hock, 0.02).smooth_union(meta, 0.02)
    return leg.smooth_union(paw(ball, s, 0.97), 0.016)


def paw(ball, s, g):
    """Big rounded paw sitting on y = 0 with four toe pads in front."""
    c = ball + V(0, -0.01, 0.03)
    pad = sdf.ellipsoid(c, (0.05 * g, 0.039, 0.064 * g))
    toes = []
    for dx, dz in ((0.014, 0.053), (0.037, 0.039)):
        for sx in (1, -1):
            toes.append(sdf.ellipsoid(c + V(sx * dx * g, -0.006, dz * g), (0.018 * g, 0.023, 0.022 * g)))
    p = pad.smooth_union(sdf.union(*toes), 0.012)
    return p.smooth_intersect(sdf.plane((0, 0.0, 0), (0, -1, 0), extent=1.0), 0.006)


def tail_frame():
    pts = TAIL + [TAIL_TIP]
    out = []
    for i in range(len(pts)):  # outward = away from the bend (up and back)
        j0, j1 = max(i - 1, 0), min(i + 1, len(pts) - 1)
        t = pts[j1] - pts[j0]
        t /= np.linalg.norm(t)
        o = np.cross((1, 0, 0), t)
        out.append(o if o[1] - o[2] > 0 else -o)
    return pts, np.array(out)


def tail():
    """Bushy tail: one fat lens strand that tapers to a point, plus two locks flaring at the brush."""
    pts, out = tail_frame()
    main = lock(pts, out, 0.088, 0.075, k=0.066, root=0.6, peak=0.5, tip=0.03, samples=8)
    side = []
    for s in (1, -1):
        side.append(lock([mx((0.02, 0.58, -0.735), s), mx((0.065, 0.45, -0.8), s), mx((0.058, 0.32, -0.84), s)],
                         mx((1, 0, -0.4), s), 0.048, 0.032, root=0.6, peak=0.35))
    return main.smooth_union(sdf.union(*side), 0.02)


def wolf_sdf():
    body = torso().smooth_union(neck(), 0.06)
    legs = sdf.union(leg_front(1), leg_front(-1))
    body = body.smooth_union(legs, 0.045)
    body = body.smooth_union(sdf.union(leg_back(1), leg_back(-1)), 0.05)
    body = body.smooth_union(head(), 0.04)
    body = body.smooth_union(sdf.union(ear(1), ear(-1)), 0.012)
    body = body.smooth_union(ruff(), 0.014)
    body = body.smooth_union(tail(), 0.03)
    return body


def body_sizing():
    face = lambda P: (P[:, 2] > 0.55) & (P[:, 1] > 0.79)  # noqa: E731
    joints = lambda P: (P[:, 1] < 0.48)  # noqa: E731
    return sizing([(face, 0.022), (joints, 0.034)], 0.056)


# ----------------------------------------------------------------------------- eyes & teeth

def head_surface(shape, origin, direction):
    """First point where the ray origin + t*direction leaves the shape (bisection)."""
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    lo, hi = 0.0, 0.3
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if shape.dist((origin + d * mid)[None])[0] < 0:
            lo = mid
        else:
            hi = mid
    return origin + d * lo


def eye_frame(shape, s):
    """Eye center on the head surface, outward normal, horizontal (toward the front) and vertical
    tangents (left: s = 1)."""
    p = head_surface(shape, H(s * 0.02, 0.014, 0.0), V(s * 0.62, 0.05, 1.0))
    n = shape.gradient(p[None], eps=0.002)[0]
    h = np.cross((0, 1, 0), n)
    h /= np.linalg.norm(h)
    if h[2] < 0:
        h = -h
    v = np.cross(n, h)
    if v[1] < 0:
        v = -v
    return p, n, h, v


EYE_R = (0.025, 0.0135, 0.0095)  # almond half sizes: along h, along v, along n
EYE_TILT = 16.0  # degrees: the outer (back) corner higher


def eye_axes(fr):
    p, n, h, v = fr
    a = math.radians(EYE_TILT)
    hh = h * math.cos(a) - v * math.sin(a)  # tilt: the back corner (-h) goes up
    vv = np.cross(n, hh)
    if vv[1] < 0:
        vv = -vv
    return p, n, hh, vv


def eyes(shape):
    """Two almond eye domes on the head (front halves only): iris, pupil and a highlight painted as
    vertex colors; material Eye (white, untinted)."""
    parts = []
    for s in (1, -1):
        p, n, hh, vv = eye_axes(eye_frame(shape, s))
        m = p3.sphere(1.0, segments=22, rings=12)
        L = m.vertices.copy()  # unit sphere: x -> hh, y -> vv, z -> n
        keep = np.flatnonzero(np.array([np.mean(L[list(f), 2]) for f in m.faces]) > -0.35)
        m = m.separate(keep)
        m.remove_unused_vertices()
        L = m.vertices.copy()
        R = np.stack([hh, vv, n], axis=1)
        m.vertices = p - n * 0.0045 + (L * np.array(EYE_R)) @ R.T
        x, y = L[:, 0], L[:, 1]
        col = np.tile(m3.as_color("#f4c23e")[:3], (len(L), 1))
        rim = ss(0.72, 0.95, np.sqrt(x * x + y * y))
        col = col * (1 - 0.35 * rim[:, None]) + m3.as_color("#c97a1c")[:3] * 0.35 * rim[:, None]
        px = (x - 0.12) / 0.34
        py = (y - 0.02) / 0.62
        col[(px * px + py * py) < 1.0] = m3.as_color("#1c1820")[:3]
        hl = ((x - 0.32) / 0.16) ** 2 + ((y - 0.38) / 0.2) ** 2 < 1.0
        col[hl] = (1.0, 0.99, 0.95)
        m.colors = np.ones((len(L), 4))
        m.colors[:, :3] = m3.srgb_to_linear(col)
        m.normals = None
        m.set_material(EYE)
        parts.append(m)
    out = p3.Mesh.join(parts, name="Wolf_Eyes")
    out.smooth_angle = 180
    return out


def teeth():
    """Fangs and a few cheek teeth on both jaws, inside the mouth (seen when it opens): little
    cones, cream (vertex colors on the Fur material). Returns (upper, lower)."""
    up, lo = [], []
    zf = HEAD_C[2] + 0.232
    for s in (1, -1):
        for dz, r, ln in ((0.0, 0.0058, 0.016), (-0.036, 0.0042, 0.0085), (-0.068, 0.0042, 0.0085)):
            c = p3.cone(r, ln, segments=6).rotate((180, 0, 0)).translate((0, ln / 2, 0))  # tip down at y=0
            x = 0.022 + dz * 0.07
            up.append(c.translate((s * x, MOUTH_Y + 0.0065 - ln, zf + dz)))
        for dz, r, ln in ((-0.014, 0.0052, 0.012), (-0.05, 0.004, 0.0085)):
            c = p3.cone(r, ln, segments=6).translate((0, ln / 2, 0))  # tip up
            x = 0.018 + dz * 0.08
            lo.append(c.translate((s * x, MOUTH_Y - 0.0065, zf + dz)))
    res = []
    for parts, name in ((up, "TeethUp"), (lo, "TeethLow")):
        m = p3.Mesh.join(parts, name=name)
        m.colors = np.ones((m.num_vertices, 4))
        m.colors[:, :3] = m3.srgb_to_linear(m3.as_color("#f1ead9")[:3])
        m.set_material(FUR)
        res.append(m)
    return res
# ============================================================================= END MODEL


# ----------------------------------------------------------------------------- paint

PAL = {
    "coat": "#706f7d", "saddle": "#3e3c4a", "cream": "#e3ddd2", "leg": "#868592", "paw": "#c4c0bc",
    "nose": "#25222a", "dark": "#2e2b35", "palate": "#3f1c24", "tongue": "#8a3843", "earin": "#d0c6bd",
    "tip": "#3a3845",
}


def _c(name):
    return m3.as_color(PAL[name])[:3]


def _mix(col, name, w):
    w = np.clip(np.asarray(w, float), 0, 1)[:, None]
    return col * (1 - w) + _c(name) * w


def paint(m: p3.Mesh):
    """sRGB painting by position (smooth masks), stored linear."""
    P = m.vertices
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    ax = np.abs(x)
    col = np.tile(_c("coat"), (len(P), 1))
    rows = np.array(TORSO)
    top = np.interp(z, rows[:, 0], rows[:, 1])
    bot = np.interp(z, rows[:, 0], rows[:, 2])
    vtop = (y - TORSO_YC) / np.maximum(top - TORSO_YC, 1e-3)
    vbot = (TORSO_YC - y) / np.maximum(TORSO_YC - bot, 1e-3)
    on_torso = ss(-0.7, -0.56, z) * ss(0.5, 0.4, z)
    # the saddle on the back (continuing up the nape), cream belly
    col = _mix(col, "saddle", ss(0.32, 0.66, vtop) * on_torso)
    u, nape_dir = neck_frame()
    q = P - NECK_A
    t = np.clip(q @ u / np.linalg.norm(NECK_B - NECK_A), 0, 1)
    rel = q - np.outer(q @ u, u)
    up = rel @ nape_dir / np.maximum(np.linalg.norm(rel, axis=1), 1e-6)
    on_neck = ss(0.25, 0.33, z) * ss(0.98, 0.9, y)
    col = _mix(col, "saddle", on_neck * ss(0.5, 0.85, up) * (1 - ss(0.75, 1.0, t)))
    col = _mix(col, "cream", ss(0.4, 0.78, vbot) * on_torso * ss(0.36, 0.26, z))
    # legs: lighter going down, cream inner sides and paws
    col = _mix(col, "leg", ss(0.5, 0.38, y))
    col = _mix(col, "cream", ss(0.42, 0.3, y) * ss(0.112, 0.09, ax) * 0.45)
    col = _mix(col, "paw", ss(0.11, 0.05, y))
    # throat, bib and the lower ruff
    col = _mix(col, "cream", ss(0.3, 0.42, z) * ss(0.8, 0.66, y) * ss(0.13, 0.06, ax))
    col = _mix(col, "cream", on_neck * ss(-0.3, -0.75, up) * ss(0.3, 0.4, z))
    # head: dark mask over the skull and down the bridge, cream cheeks, muzzle sides, chin, brows
    hx, hy, hz = x - HEAD_C[0], y - HEAD_C[1], z - HEAD_C[2]
    on_head = ss(-0.06, -0.02, hz) * ss(-0.14, -0.11, hy)
    # cream: the lower half of the muzzle, cheeks below the eyes, the chin and throat
    mline = -0.012 - 0.026 * np.clip((hz - 0.045) / 0.217, 0, 1) - 0.004  # muzzle center height
    cheek = ss(0.0, 0.015, (-0.03 - 0.12 * np.clip(hz, 0, 0.3)) - hy)
    col = _mix(col, "cream", on_head * ss(0.02, 0.07, hz) * ss(0.0, 0.01, mline - hy) * ss(0.25, 0.23, hz))
    col = _mix(col, "cream", on_head * cheek * ss(-0.04, 0.0, hz) * ss(0.1, 0.06, hz) * ss(0.06, 0.08, ax))
    col = _mix(col, "cream", ss(-0.06, -0.02, hz) * ss(-0.075, -0.1, hy) * ss(0.86, 0.95, y + 0.0))
    # dark mask over the skull and down the bridge
    col = _mix(col, "saddle", on_head * ss(0.035, 0.065, hy) * ss(0.08, 0.035, ax) * ss(0.12, 0.07, hz))
    ridge = mline + 0.026 - 0.004 * np.clip(hz / 0.25, 0, 1)
    col = _mix(col, "saddle", on_head * ss(0.0, 0.012, hy - ridge) * ss(0.03 - 0.015 * np.clip(hz / 0.25, 0, 1), 0.008, ax)
               * ss(0.24, 0.2, hz))
    for s in (1, -1):  # light "eyebrow" spots
        c = H(s * 0.05, 0.046, 0.062)
        r = np.linalg.norm((P - c) / np.array([1.0, 0.7, 1.2]), axis=1)
        col = _mix(col, "cream", ss(0.02, 0.011, r))
    # eye rims (anime liner) with a little wing at the back corner
    for s in (1, -1):
        p, n, hh, vv = eye_axes(EYE_FRAMES[s])
        d = P - p
        near = np.abs(d @ n) < 0.02
        e = np.sqrt((d @ hh / (EYE_R[0] + 0.006)) ** 2 + (d @ vv / (EYE_R[1] + 0.0045)) ** 2)
        col = _mix(col, "dark", ss(1.15, 0.92, e) * near)
        back = -(d @ hh)
        wing = np.abs((d @ vv) - (back - 0.018) * 0.3) < 0.0045
        col = _mix(col, "dark", wing * (back > 0.016) * (back < 0.036) * near)
    # nose, lip line, mouth inside
    nose = ss(0.224, 0.238, hz) * ss(MOUTH_Y + 0.016, MOUTH_Y + 0.022, y) * ss(0.034, 0.026, ax)
    col = _mix(col, "nose", nose)
    inside = (np.abs(y - MOUTH_Y) < 0.0038) & (z > MOUTH_Z - 0.01) & (ax < 0.05)
    roof = inside & (y > MOUTH_Y)
    col[roof] = _c("palate")
    col[inside & ~roof] = _c("palate")
    col = _mix(col, "tongue", inside * (~roof) * ss(0.026, 0.014, ax) * ss(0.2, 0.17, hz))
    lips = ss(0.009, 0.004, np.abs(y - MOUTH_Y)) * ss(MOUTH_Z - 0.025, MOUTH_Z - 0.005, z) * (~inside)
    col = _mix(col, "dark", lips)
    # ears: dark backs and tips, light inside
    for s in (1, -1):
        a, b = mx(EAR[0], s), mx(EAR[1], s)
        on = ss(0.975, 1.0, y) * (x * s > 0.02) * ss(0.43, 0.48, z)
        zf = a[2] + (b[2] - a[2]) * np.clip((y - a[1]) / (b[1] - a[1]), 0, 1)
        front = ss(0.0, 0.006, z - zf)
        col = _mix(col, "saddle", on * (1 - front))
        col = _mix(col, "earin", on * front * ss(0.995, 1.02, y))
        col = _mix(col, "tip", on * ss(1.09, 1.12, y))
    # tail: dark top, dark tip
    st, dt = polyline_param(P, TAIL + [TAIL_TIP])
    tl = ss(0.3, 0.9, st) * (dt < 0.13)
    col = _mix(col, "saddle", tl * 0.55)
    col = _mix(col, "tip", tl * ss(3.6, 4.4, st))
    m.colors = np.ones((len(P), 4))
    m.colors[:, :3] = m3.srgb_to_linear(np.clip(col, 0, 1))


EYE_FRAMES = {}


# ----------------------------------------------------------------------------- cache

CACHE = HERE / ".cache"


def _model_key():
    src = Path(__file__).read_text()
    model = src[src.index("# " + "=" * 77 + " MODEL"):src.index("# " + "=" * 77 + " END MODEL")]
    h = hashlib.sha1(f"{CELL} {BUDGET}".encode() + model.encode())
    for f in ("shapes.py", "meshing.py"):
        h.update((HERE / f).read_bytes())
    for f in ("sdf.py", "remesh.py", "mesh.py"):
        h.update((Path(p3.__file__).resolve().parent / f).read_bytes())
    return h.hexdigest()[:16]


def build_meshes():
    key = _model_key()
    path = CACHE / f"wolf_{key}.pkl"
    shape = wolf_sdf()
    for s in (1, -1):
        EYE_FRAMES[s] = eye_frame(shape, s)
    if path.exists() and "--no-cache" not in sys.argv:
        body = pickle.loads(path.read_bytes())
        print("wolf mesh from cache", path.name)
    else:
        body = mesh_sdf(shape, CELL, BUDGET, max_edge=body_sizing(), name="Wolf", colors=False)
        CACHE.mkdir(exist_ok=True)
        for old in CACHE.glob("wolf_*.pkl"):
            old.unlink()
        path.write_bytes(pickle.dumps(body))
    body.materials = []
    body.face_materials = None
    body.set_material(FUR)
    paint(body)
    up, low = teeth()
    return body, eyes(shape), up, low, shape



def polyline_param(P, knots):
    """Per point: (i + t of the nearest point on the polyline through knots, distance to it)."""
    knots = [np.asarray(k, float) for k in knots]
    best_d = np.full(len(P), np.inf)
    best_s = np.zeros(len(P))
    n = len(knots) - 1
    for i in range(n):
        a, b = knots[i], knots[i + 1]
        ab = b - a
        t = np.clip((P - a) @ ab / (ab @ ab), 0.0, 1.0)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, i + t, best_s)
    return best_s, best_d


# ----------------------------------------------------------------------------- rig & weights

LEGS = {"Front": FRONT, "Back": BACK}
SIDES = (("Left", 1), ("Right", -1))
LEG_PARTS = ("Upper", "Lower", "Foot", "Toes")
TAIL_BONES = ["Tail1", "Tail2", "Tail3", "Tail4"]
TAIL_JOINTS = [TAIL[0], TAIL[1], TAIL[2], TAIL[3], TAIL_TIP]
CORE = [V(0, 0.7, -0.56)] + [SPINE[b][0] for b in ("Hips", "Spine", "Chest", "Neck", "Head")] + [SPINE["Head"][1]]


def build_skeleton() -> p3.Skeleton:
    sk = p3.Skeleton("WolfRig")
    par = {"Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck", "Jaw": "Head"}
    for b in ("Hips", "Spine", "Chest", "Neck", "Head", "Jaw"):
        sk.add_bone(b, SPINE[b][0], SPINE[b][1], parent=par.get(b))
    for side, s in SIDES:
        sk.add_bone(side + "Ear", mx(EAR[0], s), mx(EAR[1], s), parent="Head")
    sk.add_chain(TAIL_BONES, TAIL_JOINTS, parent="Hips")
    for leg, J in LEGS.items():
        for side, s in SIDES:
            parent = "Chest" if leg == "Front" else "Hips"
            for i, part in enumerate(LEG_PARTS):
                a = J[part]
                b = J[LEG_PARTS[i + 1]] if i < 3 else J["tip"]
                parent = sk.add_bone(f"{leg}{side}{part}", mx(a, s), mx(b, s), parent=parent).name
    return sk


def weights(P: np.ndarray, names: list[str]) -> np.ndarray:
    """(N, bones) weights from rest positions: soft memberships (legs, tail, ears, jaw) and smooth
    blends along each chain (no rigid seams at the joints)."""
    idx = {n: i for i, n in enumerate(names)}
    W = np.zeros((len(P), len(names)))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    taken = np.zeros(len(P))
    # legs
    legw = []
    for leg, J in LEGS.items():
        for side, s in SIDES:
            knots = [mx(J[k], s) for k in (*LEG_PARTS, "tip")]
            sp, d = polyline_param(P, knots)
            if leg == "Front":
                m = ss(0.62, 0.48, y) * (1 - ss(0.1, 0.14, d)) * ss(0.012, 0.055, s * x)
            else:
                m = ss(0.74, 0.58, y) * (1 - ss(0.13, 0.17, d)) * ss(0.0, 0.05, s * x) * ss(-0.08, -0.16, z)
            m = np.where(y < 0.4, 1 - ss(0.09, 0.12, d), m) * ss(0.0, 0.03, s * x)
            ball = mx(J["Toes"], s)
            toes = ss(ball[2] - 0.004, ball[2] + 0.026, z) * ss(0.08, 0.05, y)
            ch = {"Upper": 1 - ss(0.8, 1.15, sp), "Lower": ss(0.8, 1.15, sp) - ss(1.82, 2.12, sp)}
            ankle = ss(1.82, 2.12, sp)
            ch["Foot"] = ankle * (1 - toes)
            ch["Toes"] = ankle * toes
            legw.append((leg + side, m, ch))
    lsum = sum(m for _, m, _ in legw)
    scale = np.where(lsum > 1, 1 / np.maximum(lsum, 1e-9), 1.0)
    for pre, m, ch in legw:
        for part, w in ch.items():
            W[:, idx[pre + part]] += m * scale * w
        taken += m * scale
    # tail
    st, dt = polyline_param(P, TAIL_JOINTS)
    mt = ss(0.05, 0.3, st) * (1 - ss(0.12, 0.16, dt)) * (1 - taken)
    c = np.arange(4) + 0.5
    hat = np.clip(1 - np.abs(np.minimum(st, 3.999)[:, None] - c[None]), 0, 1)
    hat[:, 0] = np.where(st < 0.5, 1.0, hat[:, 0])
    hat[:, -1] = np.where(st > c[-1], 1.0, hat[:, -1])
    hat /= np.maximum(hat.sum(axis=1, keepdims=True), 1e-9)
    for k, b in enumerate(TAIL_BONES):
        W[:, idx[b]] += mt * hat[:, k]
    taken += mt
    free = 1 - taken
    # core chain: Hips, Spine, Chest, Neck, Head (knots 1..5 are the bone heads)
    sc, _ = polyline_param(P, CORE)
    core = {"Hips": 1 - ss(1.8, 2.2, sc), "Spine": ss(1.8, 2.2, sc) - ss(2.8, 3.2, sc),
            "Chest": ss(2.8, 3.2, sc) - ss(3.75, 4.2, sc), "Neck": ss(3.75, 4.2, sc) - ss(4.75, 5.1, sc),
            "Head": ss(4.75, 5.1, sc)}
    head = core["Head"]
    jaw = ss(MOUTH_Y + 0.001, MOUTH_Y - 0.005, y) * ss(MOUTH_Z - 0.06, MOUTH_Z - 0.005, z) * (np.abs(x) < 0.075)
    ears = {}
    for side, s in SIDES:
        a, b = mx(EAR[0], s), mx(EAR[1], s)
        _, de = polyline_param(P, [a - (b - a) * 0.1, b])
        ears[side] = ss(0.982, 1.015, y) * (1 - ss(0.05, 0.07, de)) * (s * x > 0.01)
    core["Jaw"] = head * jaw
    core["LeftEar"] = head * (1 - jaw) * ears["Left"]
    core["RightEar"] = head * (1 - jaw) * ears["Right"]
    core["Head"] = head * (1 - jaw) * (1 - ears["Left"]) * (1 - ears["Right"])
    for b, w in core.items():
        W[:, idx[b]] += free * w
    W = np.clip(W, 0, None)
    tot = W.sum(axis=1, keepdims=True)
    W = np.where(tot > 1e-9, W / np.maximum(tot, 1e-9), 0)
    W[tot[:, 0] <= 1e-9, idx["Hips"]] = 1.0
    return W


def top4(W):
    order = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, order, axis=1)
    w[w < 0.004] = 0
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    return np.where(w > 0, order, 0).astype(np.int64), w


def skin(mesh, sk, rigid=None):
    names = [b.name for b in sk.bones]
    if rigid:
        mesh.joints = np.zeros((mesh.num_vertices, 4), np.int64)
        mesh.weights = np.zeros((mesh.num_vertices, 4))
        mesh.joints[:, 0] = names.index(rigid)
        mesh.weights[:, 0] = 1
    else:
        mesh.joints, mesh.weights = top4(weights(mesh.vertices, names))
    return mesh


# ----------------------------------------------------------------------------- pose solver

I4 = np.array([0.0, 0.0, 0.0, 1.0])
X = V(1, 0, 0)


def qaxis(axis, deg):
    return m3.quat_from_axis_angle(m3.normalize(np.asarray(axis, float)), deg)


def qx(d):
    return qaxis((1, 0, 0), d)


def qy(d):
    return qaxis((0, 1, 0), d)


def qz(d):
    return qaxis((0, 0, 1), d)


def qm(*qs):
    out = I4
    for q in qs:
        out = m3.quat_mul(out, q)
    return out


def rot(q, v):
    return m3.quat_rotate(q, np.asarray(v, float))


def nrm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def frame_align(a0, b0, a1, b1):
    """Rotation taking a0 -> a1 exactly and b0 (orthogonalized) -> b1 (orthogonalized)."""
    def basis(a, b):
        a = nrm(a)
        b = nrm(np.asarray(b, float) - a * np.dot(b, a))
        return np.stack([a, b, np.cross(a, b)], axis=1)
    return m3.matrix3_to_quat(basis(a1, b1) @ basis(a0, b0).T)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def keys(t, ks):
    """Piecewise smooth interpolation of {time: value}."""
    ts = sorted(ks)
    if t <= ts[0]:
        return np.asarray(ks[ts[0]], float)
    for a, b in zip(ts, ts[1:]):
        if t <= b:
            u = smooth((t - a) / (b - a))
            return np.asarray(ks[a], float) * (1 - u) + np.asarray(ks[b], float) * u
    return np.asarray(ks[ts[-1]], float)


class Pose(dict):
    def __init__(self):
        super().__init__()
        self.hips = np.zeros(3)


class Rig:
    def __init__(self, sk: p3.Skeleton):
        self.sk = sk
        self.parent = {}
        for b in sk.bones:
            n = b.parent
            self.parent[b.name] = n.name if (n is not None and n.extras.get("is_bone")) else None
        self.head = {b.name: sk.head(b) for b in sk.bones}
        self.tail = {b.name: sk.tail(b) for b in sk.bones}
        self.dir = {n: nrm(self.tail[n] - self.head[n]) for n in self.head}
        self.len = {n: float(np.linalg.norm(self.tail[n] - self.head[n])) for n in self.head}
        self.names = [b.name for b in sk.bones]
        self.samples = None

    def world_rot(self, pose, bone):
        chain = []
        while bone is not None:
            chain.append(bone)
            bone = self.parent[bone]
        q = I4
        for name in reversed(chain):
            q = qm(q, pose.get(name, I4))
        return q

    def world_head(self, pose, bone):
        chain = []
        b = bone
        while b is not None:
            chain.append(b)
            b = self.parent[b]
        chain.reverse()
        p = self.head[chain[0]] + pose.hips
        q = I4
        prev = chain[0]
        for name in chain[1:]:
            q = qm(q, pose.get(prev, I4))
            p = p + rot(q, self.head[name] - self.head[prev])
            prev = name
        return p

    def world_point(self, pose, bone, p_rest):
        """A rest-space point carried rigidly by `bone`."""
        return self.world_head(pose, bone) + rot(self.world_rot(pose, bone), np.asarray(p_rest) - self.head[bone])

    def pose_points(self, pose, pts, W):
        """Linear-blend skinning of rest points (dense weights W, bones in skeleton order)."""
        out = np.zeros_like(pts)
        for i, b in enumerate(self.names):
            m = W[:, i] > 0
            if not m.any():
                continue
            R = m3.quat_to_matrix3(self.world_rot(pose, b))
            out[m] += W[m, i, None] * (self.world_head(pose, b) + (pts[m] - self.head[b]) @ R.T)
        return out

    def ground(self, pose, settle=0.0):
        """Lift the body out of the ground (and, with settle in [0, 1], down onto it)."""
        if getattr(self, "samples", None) is None:
            return
        low = float(self.pose_points(pose, *self.samples)[:, 1].min())
        pose.hips[1] += max(-low, 0.0) * (1 - settle) + (-low) * settle

    def set_world(self, pose, bone, q_world):
        par = self.parent[bone]
        pw = self.world_rot(pose, par) if par else I4
        pose[bone] = qm(m3.quat_conj(pw), q_world)

    def body(self, pose, pitch=0.0, roll=0.0, yaw=0.0, offset=(0, 0, 0), pivot=(0, 0.7, -0.05)):
        """Turn the whole body about `pivot` (not the hips joint), then shift it by `offset`."""
        q = qm(qy(yaw), qz(roll), qx(pitch))
        h = self.head["Hips"]
        pv = np.asarray(pivot, float)
        pose["Hips"] = q
        pose.hips = pv - h - rot(q, pv - h) + np.asarray(offset, float)

    def leg(self, pose, pre, ball, pitch=0.0, toes=0.0, auto=True):
        """Paw of leg `pre` (e.g. "FrontLeft") with its ball at `ball` (character space); the
        metapodial pitched by `pitch` deg about the body's lateral axis (+ = its top, the wrist /
        hock, swings forward), toes bent by `toes` deg from flat. If the wrist / hock is out of
        reach the pitch is changed as little as needed (the paw rolls; the ball never moves)."""
        up, lo, ft, to = (pre + p for p in LEG_PARTS)
        par = self.parent[up]
        lat = rot(self.world_rot(pose, par), X)
        top = self.world_head(pose, up)
        L1, L2 = self.len[up], self.len[lo]
        fv0 = self.tail[ft] - self.head[ft]  # wrist -> ball (rest)
        ball = np.asarray(ball, float)

        def wrist_at(pt):
            return ball - rot(qaxis(lat, -pt), fv0)
        reach = (L1 + L2) * 0.997
        best = pitch
        if auto and np.linalg.norm(wrist_at(pitch) - top) > reach:
            for k in range(1, 121):
                cand = [pitch + k, pitch - k]
                ok = [c for c in cand if np.linalg.norm(wrist_at(c) - top) <= reach]
                if ok:
                    best = ok[0]
                    break
            else:
                best = min(range(int(pitch) - 120, int(pitch) + 121), key=lambda c: np.linalg.norm(wrist_at(c) - top))
        wr = wrist_at(best)
        d = wr - top
        D = float(np.linalg.norm(d))
        D = min(max(D, abs(L1 - L2) + 1e-4), (L1 + L2) * 0.9995)
        dn = nrm(d)
        a = (L1 * L1 - L2 * L2 + D * D) / (2 * D)
        h = math.sqrt(max(L1 * L1 - a * a, 0.0))
        bend = -1.0 if pre.startswith("Front") else 1.0  # the elbow points back, the knee forward
        pole_w = rot(self.world_rot(pose, par), (0, 0, bend))
        pp = nrm(pole_w - dn * np.dot(pole_w, dn))
        mid = top + dn * a + pp * h
        hinge = nrm(np.cross(dn, pp))
        if np.dot(hinge, lat) < 0:
            hinge = -hinge
        self.set_world(pose, up, frame_align(self.dir[up], X, nrm(mid - top), hinge))
        self.set_world(pose, lo, frame_align(self.dir[lo], X, nrm(top + dn * D - mid), hinge))
        self.set_world(pose, ft, frame_align(self.dir[ft], X, nrm(ball - wr), hinge))
        tdir = rot(qaxis(lat, toes), rot(self.world_rot(pose, "Hips"), self.dir[to]) * 0 + self.dir[to])
        self.set_world(pose, to, frame_align(self.dir[to], X, tdir, hinge))
        err = float(np.linalg.norm(self.world_head(pose, to) - ball))
        self.residual = max(getattr(self, "residual", 0.0), err)
        return best

    def ball_rest(self, pre):
        return self.head[pre + "Toes"].copy()


# ----------------------------------------------------------------------------- clips

LEG_NAMES = [f"{leg}{side}" for leg in ("Front", "Back") for side, _ in SIDES]


def stance_cycle(u, duty, a, lift, center=0.0, roll_off=25.0, fold=60.0, curl=30.0):
    """One leg over a gait cycle (u = 0: the paw lands). Returns (dz, dy, pitch, toes): the paw
    moves back linearly while planted (exact ground speed) and swings forward in the air."""
    if u < duty:
        s = u / duty
        dz = center + a - 2 * a * s
        pitch = -roll_off * ss(0.55, 1.0, s)  # the wrist / hock lowers as the leg trails (push-off)
        return dz, 0.0, float(pitch), float(-12 * ss(0.7, 1.0, s))
    s = (u - duty) / (1 - duty)
    dz = center - a + 2 * a * smooth(s)
    dy = lift * math.sin(math.pi * s) ** 1.2
    k = math.sin(math.pi * min(s * 1.15, 1.0))
    pitch = -roll_off * (1 - smooth(s / 0.25)) + fold * k  # the paw folds back under the leg
    return dz, dy, pitch, -curl * k


def tail_pose(P, lift=0.0, sway=0.0, wave=0.0, phase=0.0, curl=0.0):
    for i, b in enumerate(TAIL_BONES):
        lag = phase - 0.12 * i
        P[b] = qm(qy(sway * (0.6 + 0.25 * i) * math.sin(2 * math.pi * lag) if wave == 0 else
                     sway * math.sin(2 * math.pi * lag)),
                  qx((lift if i == 0 else lift * 0.25) + wave * math.sin(2 * math.pi * lag) + curl * (i / 3)))


def plant_all(rig, P, shift=None, pitch=None, toes=None):
    """Every paw planted at rest (optionally shifted), legs solved by IK."""
    for pre in LEG_NAMES:
        b = rig.ball_rest(pre) + (np.zeros(3) if shift is None else np.asarray(shift.get(pre, (0, 0, 0)), float))
        rig.leg(P, pre, b, 0.0 if pitch is None else pitch.get(pre, 0.0), 0.0 if toes is None else toes.get(pre, 0.0))


def clip_idle(rig, t):
    D = DUR["Idle"]
    ph = t / D
    P = Pose()
    br = math.sin(2 * math.pi * 2 * ph)
    rig.body(P, pitch=0.4 * br, offset=(0, 0.004 * br, 0))
    P["Spine"] = qx(-0.6 * br)
    P["Chest"] = qx(0.5 * br)
    look = math.sin(2 * math.pi * ph)
    P["Neck"] = qm(qy(7 * look), qx(-2 + 2.5 * math.sin(2 * math.pi * ph + 0.7)))
    P["Head"] = qm(qy(5 * look), qz(-3 * look), qx(2 * math.sin(4 * math.pi * ph + 1.0)))
    tw = lambda c: math.exp(-((ph - c) / 0.035) ** 2)  # noqa: E731
    P["LeftEar"] = qm(qx(-22 * tw(0.38)), qz(-8 * tw(0.38)))
    P["RightEar"] = qm(qx(-22 * tw(0.71)), qz(8 * tw(0.71)))
    tail_pose(P, lift=4 + 2 * br, sway=7, phase=ph)
    plant_all(rig, P)
    return P


def clip_walk(rig, t):
    """A light trot: diagonal pairs (front left with hind right), a short float between steps."""
    D, v = DUR["Walk"], WALK_SPEED
    ph = t / D
    duty = 0.42
    a = v * duty * D / 2
    P = Pose()
    bob = math.cos(4 * math.pi * (ph - 0.21))
    rig.body(P, pitch=1.0, roll=1.5 * math.sin(2 * math.pi * ph), yaw=1.5 * math.sin(2 * math.pi * ph),
             offset=(0, -0.012 + 0.012 * bob, 0))
    P["Spine"] = qy(-2 * math.sin(2 * math.pi * ph))
    P["Chest"] = qy(-2 * math.sin(2 * math.pi * ph))
    P["Neck"] = qx(4 - 2 * bob)
    P["Head"] = qx(-2 + 2 * bob)
    P["LeftEar"] = qx(-6)
    P["RightEar"] = qx(-6)
    tail_pose(P, lift=-2, sway=8, phase=ph)
    starts = {"FrontLeft": 0.0, "BackRight": 0.02, "FrontRight": 0.5, "BackLeft": 0.52}
    for pre, st in starts.items():
        u = (ph - st) % 1.0
        front = pre.startswith("Front")
        dz, dy, pitch, toes = stance_cycle(u, duty, a, 0.09 if front else 0.08, center=0.02 if front else -0.01,
                                           roll_off=22, fold=55 if front else 35)
        rig.leg(P, pre, rig.ball_rest(pre) + V(0, dy, dz), pitch, toes)
    return P


def clip_run(rig, t):
    """Gallop (transverse): hind left, hind right, front left, front right; a gathered float after
    the front push-off; the back flexes and extends, the neck reaches forward, ears back, tail out."""
    D, v = DUR["Run"], RUN_SPEED
    ph = t / D
    duty = 0.26
    a = v * duty * D / 2
    P = Pose()
    lift_y = 0.03 * math.cos(2 * math.pi * (ph - 0.88)) - 0.025
    pitch = -5.0 * math.cos(2 * math.pi * (ph - 0.15))
    flex = math.cos(2 * math.pi * (ph - 0.95))  # +1 gathered (arched back), -1 stretched
    rig.body(P, pitch=pitch - 3 * flex, offset=(0, lift_y, 0))
    P["Spine"] = qx(6 * flex)
    P["Chest"] = qx(4 * flex)
    P["Neck"] = qx(18 + 5 * math.cos(2 * math.pi * (ph - 0.6)))
    P["Head"] = qx(-14 - 4 * math.cos(2 * math.pi * (ph - 0.6)))
    P["Jaw"] = qx(9 + 3 * math.sin(2 * math.pi * ph))
    P["LeftEar"] = qm(qx(-32), qz(-6))
    P["RightEar"] = qm(qx(-32), qz(6))
    tail_pose(P, lift=34, wave=8, sway=4, phase=ph)
    starts = {"BackLeft": 0.0, "BackRight": 0.09, "FrontLeft": 0.38, "FrontRight": 0.47}
    for pre, st in starts.items():
        u = (ph - st) % 1.0
        front = pre.startswith("Front")
        dz, dy, pt, toes = stance_cycle(u, duty, a, 0.16 if front else 0.13, center=0.04 if front else 0.07,
                                        roll_off=30, fold=95 if front else 55, curl=35)
        rig.leg(P, pre, rig.ball_rest(pre) + V(0, dy, dz), pt, toes)
    return P


def clip_attack(rig, t):
    """Crouch and stare (0-0.26), lunge with the jaws wide (0.26-0.4), the bite closes at 0.42
    (WOLF.windup), a tug, then back to the stance (front paws step back)."""
    P = Pose()
    fwd = float(keys(t, {0: 0, 0.24: -0.07, 0.38: 0.24, 0.46: 0.28, 0.62: 0.12, 0.75: 0}))
    dy = float(keys(t, {0: 0, 0.24: -0.1, 0.34: 0.035, 0.44: -0.01, 0.6: -0.02, 0.75: 0}))
    pitch = float(keys(t, {0: 0, 0.24: 7, 0.34: -8, 0.44: 6, 0.6: 1, 0.75: 0}))
    rig.body(P, pitch=pitch, offset=(0, dy, fwd))
    P["Spine"] = qx(float(keys(t, {0: 0, 0.24: -5, 0.36: 3, 0.75: 0})))
    P["Neck"] = qx(float(keys(t, {0: 0, 0.24: 26, 0.36: 12, 0.44: 22, 0.55: 14, 0.75: 0})))
    P["Head"] = qm(qx(float(keys(t, {0: 0, 0.24: -22, 0.36: -14, 0.44: 6, 0.52: -2, 0.75: 0}))),
                   qz(float(keys(t, {0: 0, 0.44: 0, 0.48: 8, 0.54: -6, 0.6: 0}))))
    P["Jaw"] = qx(float(keys(t, {0: 0, 0.18: 6, 0.3: 34, 0.395: 38, 0.425: 0, 0.6: 0, 0.66: 4, 0.75: 0})))
    ear = float(keys(t, {0: 0, 0.2: -38, 0.6: -38, 0.75: 0}))
    P["LeftEar"] = qm(qx(ear), qz(ear * 0.15))
    P["RightEar"] = qm(qx(ear), qz(-ear * 0.15))
    tail_pose(P, lift=float(keys(t, {0: 2, 0.24: 26, 0.44: 20, 0.75: 2})), sway=2, phase=t / 0.75)
    # front paws: planted, then in the air during the lunge, landing 0.22 ahead; step back later
    fz = float(keys(t, {0.0: 0, 0.27: 0, 0.42: 0.22, 0.58: 0.22, 0.72: 0, 0.75: 0}))
    fy = float(keys(t, {0.0: 0, 0.27: 0, 0.34: 0.17, 0.42: 0, 0.58: 0, 0.65: 0.07, 0.72: 0, 0.75: 0}))
    fp = float(keys(t, {0.0: 0, 0.27: 0, 0.33: 70, 0.42: 0, 0.58: 0, 0.64: 40, 0.72: 0}))
    for side, _ in SIDES:
        lag = 0.02 if side == "Right" else 0.0
        fz_s = float(keys(t - lag, {0.0: 0, 0.27: 0, 0.42: 0.26, 0.58: 0.26, 0.72: 0, 0.75: 0}))
        _ = fz
        rig.leg(P, "Front" + side, rig.ball_rest("Front" + side) + V(0, fy, fz_s), fp, -0.4 * fp)
        rig.leg(P, "Back" + side, rig.ball_rest("Back" + side), float(keys(t, {0: 0, 0.24: -6, 0.4: 18, 0.6: 6, 0.75: 0})))
    return P


def clip_hit(rig, t):
    P = Pose()
    k = float(keys(t, {0: 0, 0.07: 1, 0.36: 0}))
    rig.body(P, pitch=-3 * k, roll=-5 * k, offset=(0.025 * k, -0.01 * k, -0.06 * k))
    P["Spine"] = qy(6 * k)
    P["Neck"] = qm(qy(10 * k), qx(-16 * k))
    P["Head"] = qm(qx(-12 * k), qz(-8 * k))
    P["Jaw"] = qx(float(keys(t, {0: 0, 0.06: 20, 0.2: 6, 0.36: 0})))
    P["LeftEar"] = qx(-40 * k)
    P["RightEar"] = qx(-40 * k)
    tail_pose(P, lift=-24 * k, sway=3, phase=t / 0.36)
    plant_all(rig, P)
    return P


LIMP = {  # local rotations of the legs lying on the side (relative to the body)
    "FrontLeftUpper": qm(qz(-26), qx(-38)), "FrontLeftLower": qx(42), "FrontLeftFoot": qx(46), "FrontLeftToes": qx(14),
    "FrontRightUpper": qm(qz(-6), qx(-52)), "FrontRightLower": qx(30), "FrontRightFoot": qx(34), "FrontRightToes": qx(10),
    "BackLeftUpper": qm(qz(-24), qx(-6)), "BackLeftLower": qx(-38), "BackLeftFoot": qx(44), "BackLeftToes": qx(14),
    "BackRightUpper": qm(qz(-6), qx(12)), "BackRightLower": qx(-26), "BackRightFoot": qx(30), "BackRightToes": qx(10),
}


def clip_death(rig, t):
    """The hind legs buckle, the body rolls onto its right side (legs toward +X), the head lies on
    the ground, the tail flops; the last frame holds (the client sinks the body later)."""
    P = Pose()
    k1 = smooth(t / 0.3)  # stagger, the hind end drops
    k2 = smooth((t - 0.22) / 0.5)  # roll onto the side
    k3 = smooth((t - 0.62) / 0.4)  # settle
    roll = 86 * k2 - 4 * math.sin(math.pi * k3) * (1 - k3)
    pitch = -6 * k1 * (1 - k2)
    rig.body(P, pitch=pitch, roll=roll, offset=(-0.12 * k2, -0.16 * k1 * (1 - k2) - 0.53 * k2 + 0.01 * math.sin(math.pi * k3),
                                               -0.03 * k1), pivot=(0, 0.66, -0.05))
    P["Neck"] = qm(qz(-14 * k2), qx(-14 * k1 * (1 - k2) + 4 * k2))
    P["Head"] = qm(qz(-12 * k2), qx(-10 * k1 * (1 - k2)))
    P["Jaw"] = qx(16 * k1 * (1 - k2) + 7 * k3)
    P["LeftEar"] = qx(-30 * k1)
    P["RightEar"] = qx(-30 * k1)
    tail_pose(P, lift=-14 * k1 * (1 - k2), curl=-10 * k3, sway=2 * (1 - k2), phase=t)
    # legs: planted IK while standing, then limp (blend)
    stand = Pose()
    stand.update(P)
    stand.hips = P.hips
    sh = {pre: V(0, 0, 0) for pre in LEG_NAMES}
    plant_all(rig, stand, pitch={pre: (-15 * k1 if pre.startswith("Back") else 0) for pre in LEG_NAMES})
    w = smooth((t - 0.2) / 0.45)
    for pre in LEG_NAMES:
        for part in LEG_PARTS:
            b = pre + part
            P[b] = m3.quat_slerp(stand.get(b, I4), LIMP[b], w)
    _ = sh
    if t > 0.2:
        rig.ground(P, settle=smooth((t - 0.5) / 0.35))
    return P


CLIPS = {"Idle": clip_idle, "Walk": clip_walk, "Run": clip_run, "Attack": clip_attack, "Hit": clip_hit,
         "Death": clip_death}


def bake(scene, rig, clips=None):
    hips = rig.sk.bone("Hips")
    hips_rest = hips.extras["rest_translation"]
    for clip in clips or list(CLIPS):
        anim = scene.animation(clip)
        anim.loop = clip in LOOPING
        dur = DUR[clip]
        n = int(round(dur * FPS))
        prev = {}
        for i in range(n + 1):
            t = i / FPS if i < n else dur
            P = CLIPS[clip](rig, t if (i < n or not anim.loop) else 0.0)
            for b in rig.sk.bones:
                q = np.asarray(P.get(b.name, I4), float)
                if b.name in prev and np.dot(prev[b.name], q) < 0:
                    q = -q
                prev[b.name] = q
                anim.key(b, "rotation", t, q, "LINEAR")
            anim.key(hips, "position", t, hips_rest + P.hips, "LINEAR")
    return scene


def skate_report(rig):
    """Paws are placed by IK on targets that move back at exactly the clip speed while planted: the
    largest IK miss (m) per clip measures any skating (0 = none), plus the lowest paw ball height."""
    out = {}
    for clip in CLIPS:
        D = DUR[clip]
        n = int(round(D * FPS * 2))
        rig.residual = 0.0
        low = 1.0
        for i in range(n + 1):
            P = CLIPS[clip](rig, D * i / n)
            for pre in LEG_NAMES:
                low = min(low, float(rig.world_head(P, pre + "Toes")[1]))
        out[clip] = {"ik_miss_m": round(rig.residual, 4), "lowest_ball_m": round(low, 4)}
    return out


# ----------------------------------------------------------------------------- build / review

def build():
    t0 = time.time()
    body, eye_m, up, low, shape = build_meshes()
    sk = build_skeleton()
    skin(body, sk)
    skin(up, sk, rigid="Head")
    skin(low, sk, rigid="Jaw")
    skin(eye_m, sk, rigid="Head")
    wolf = p3.Mesh.join([body, up, low], name="Wolf")
    wolf.set_material(FUR)
    for m in (wolf, eye_m):
        probs = m.validate()
        if probs:
            print("  validate", m.name, probs)
    scene = p3.Scene("Niryk_Wolf")
    scene.add(sk.root)
    nodes = {"Wolf": scene.add(p3.skinned_node(wolf, sk, "Wolf")),
             "Wolf_Eyes": scene.add(p3.skinned_node(eye_m, sk, "Wolf_Eyes"))}
    rig = Rig(sk)
    sel = np.arange(0, body.num_vertices, 7)
    rig.samples = (body.vertices[sel], weights(body.vertices[sel], rig.names))
    bake(scene, rig)
    out = EXPORTS / "creatures" / "wolf.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out)
    lo, hi = wolf.bounds()
    stats = {"Wolf": wolf.num_triangles, "Wolf_Eyes": eye_m.num_triangles,
             "total": wolf.num_triangles + eye_m.num_triangles, "bones": len(sk.bones),
             "length_m": round(float(hi[2] - lo[2]), 3), "height_m": round(float(hi[1]), 3),
             "withers_m": round(float(max(r[1] for r in TORSO)), 3), "width_m": round(float(hi[0] - lo[0]), 3),
             "clips": {c: DUR[c] for c in CLIPS}, "gait": skate_report(rig), "time_s": round(time.time() - t0, 1)}
    OUT_REVIEW.mkdir(parents=True, exist_ok=True)
    (OUT_REVIEW / "wolf_stats.json").write_text(json.dumps(stats, indent=1))
    print("wolf", json.dumps(stats), "->", out)
    return scene, sk, nodes, rig


def review(scene, sk, nodes, rig, clips=("Run", "Attack", "Death")):
    from pyram3d.image import grid, write_png
    review_lights(scene)
    p3.render_views(scene, OUT_REVIEW / "wolf_turnaround.png", views=("front", "three_quarter", "right", "back"),
                    size=520, cols=4, style="toon", time=0.0, animation="Idle")
    hc = rig.world_point(clip_idle(rig, 0.0), "Head", HEAD_C + V(0, -0.01, 0.07))
    hc = tuple(float(v) for v in hc)
    shots = [((0.75, 0.2, 1.0), "Idle", 0.0), ((1.0, 0.05, 0.02), "Idle", 0.0), ((0.0, 0.08, 1.0), "Idle", 0.0),
             ((-0.6, 0.35, 1.0), "Idle", 0.0)]
    imgs = [p3.render(scene, None, 480, 420, view=v, style="toon", target=hc, distance=0.85, yfov=30, time=t,
                      animation=c) for v, c, t in shots]
    P = clip_attack(rig, 0.37)
    hb = tuple(float(v) for v in rig.world_point(P, "Head", HEAD_C + V(0, -0.03, 0.1)))
    imgs.append(p3.render(scene, None, 480, 420, view=(0.9, 0.15, 1.0), style="toon", target=hb, distance=0.95,
                          yfov=30, time=0.37, animation="Attack"))
    imgs.append(p3.render(scene, None, 480, 420, view=(-0.3, 0.1, 1.0), style="toon", target=hb, distance=0.95,
                          yfov=30, time=0.37, animation="Attack"))
    write_png(OUT_REVIEW / "wolf_head.png", grid(imgs, 3))
    for clip in clips:
        view = (1.0, 0.2, 0.15) if clip in ("Run", "Walk") else ((0.6, 0.45, 1.0) if clip == "Death" else (1.0, 0.25, 0.5))
        frames = 8 if clip in ("Run", "Attack", "Death") else 6
        p3.render_frames(scene, OUT_REVIEW / f"wolf_{clip}.png", clip, frames=frames, size=300, cols=frames // 2 if frames > 6 else None,
                         style="toon", view=view)


if __name__ == "__main__":
    scene, sk, nodes, rig = build()
    if "--no-review" not in sys.argv:
        clips = ("Run", "Attack", "Death", "Walk", "Hit", "Idle") if "--all-clips" in sys.argv else ("Run", "Attack", "Death")
        review(scene, sk, nodes, rig, clips)
