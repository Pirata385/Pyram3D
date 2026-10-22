"""Niryk prototype 2: the Guard armor (Armadura do Guarda), fitted to each body.

    python projects/niryk/protótipo_2/armor.py [human_m] [human_f] [harpin_m] [harpin_f] [--fast] [--jobs 2]
           [--no-review]

With no body named, builds all four (one process per body: the body modules read the proportions at
import time, from NIRYK_P2_BODY). Writes exports/equipment/armor_guard_<body>.glb,
review/armor/armor_guard_<body>*.png and review/armor/armor_guard_<body>_stats.json.

Anime-RPG plate (the Genshin / Tales line): clean bevelled steel shells with gold trim lines.
  - breastplate (front and back) with a raised keel down the middle, a lame tucked under its lower
    edge (male), gold trim around the neck, the arm holes and the lower edge;
  - layered pauldrons: a dome over the shoulder and two lames stepping down the upper arm, each with
    a gold lower edge, and a leather strap around the arm;
  - vambraces over the leather bracers (a ridge on top, gold ends);
  - a leather belt with a big gold buckle; tassets (two lames each) over the front of the thighs, a
    back fauld (humans), a Cloth_B tabard panel down the front;
  - greaves (front shells with a ridge, gold top, leather straps behind the calf) and knee cops.
  Harpins: the back of the breastplate is open above the waist and the pauldrons stop behind the
  shoulder (the folded wings lie there and open in the clips); no back fauld (the tail); the greaves
  are shin plates over the plumed 'drumstick', the tarsus, toes and talons stay free.

How it is made: the breastplate, vambraces, belt, tassets, tabard and greaves are smooth offset
shells (shapes.Offset) of the body forms they cover, imported from the character pipeline (body.py,
clothes.py, harpin.py), cut by planes with rounded edges. Pauldrons and knee cops are clean hollow
shells (profile tubes / ellipsoids) fitted around the outfit by sampling it. Trim lines are bands
(functions of the position) where the shell swells by ~2 mm; their faces are given the trim material
on the fine mesh, so the decimation keeps the gold border crisp, and every material gets its own
vertex colors. Faces inside the body, the outfit or another armor piece are dropped
(as meshing.mesh_sdf does).

Rig: the GLB carries the body's own armature (common.build_skeleton: same bone names and rest pose),
the client re-binds the skinned meshes to the character's bones by name
(src/client/character/equipment.ts in the game). Weights: the plates take the weights of the skin
right under them (projection on the body SDF, then rig.weight_matrix), so they move like the
garments they cover; tassets, fauld and tabard are weighted like the tunic skirt.

Slots (white base colors; the client tints by material name): Metal (steel plates: vertex color
lum ~0.8, so big plates do not read white), a second Metal primitive for the gold trims (gold vertex
color times the steel tint reads bronze-gold), Leather (belt, straps), Cloth_B (tabard).
"""
from __future__ import annotations

import os
import subprocess
import sys

BODIES = ("human_m", "human_f", "harpin_m", "harpin_f")


def _dispatch():
    """One process per requested body (unless this process is already that body's)."""
    wanted = [a for a in sys.argv[1:] if a in BODIES] or list(BODIES)
    flags = [a for a in sys.argv[1:] if a not in BODIES]
    jobs = 1
    if "--jobs" in flags:
        i = flags.index("--jobs")
        jobs = int(flags[i + 1])
        del flags[i:i + 2]
    if len(wanted) == 1 and os.environ.get("NIRYK_P2_BODY") == wanted[0]:
        return
    procs, code = [], 0
    for b in wanted:
        env = dict(os.environ, NIRYK_P2_BODY=b)
        procs.append(subprocess.Popen([sys.executable, __file__, b] + flags, env=env))
        if len(procs) >= jobs:
            code |= procs.pop(0).wait()
    for p in procs:
        code |= p.wait()
    sys.exit(code)


if __name__ == "__main__":
    _dispatch()

import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import body as B  # noqa: E402
import clothes as C  # noqa: E402
import rig as RG  # noqa: E402
from common import (BODY, FEMALE, FOREARM, HARPIN, HERE, HIP, KNEE_Y, REVIEW, SHOULDER, UPPER_ARM,  # noqa: E402
                    arm_rot, build_skeleton, export_glb, slot, xf_mesh)
from meshing import covered_by, drop_faces  # noqa: E402
from shapes import Offset, Tube  # noqa: E402

import pyram3d as p3  # noqa: E402
from pyram3d import math3d as m3  # noqa: E402
from pyram3d import sdf  # noqa: E402

FAST = "--fast" in sys.argv
CELL = 0.0056 if FAST else 0.0042
NAME = f"armor_guard_{BODY}"

# ----------------------------------------------------------------------------- materials & paint
METAL = slot("Metal")
TRIM = p3.Material("Metal", "#ffffff", metallic=0.6, roughness=0.35)  # a second Metal: the gold trims
LEATHER = slot("Leather")
CLOTH = slot("Cloth_B")
COLORS = {  # sRGB vertex colors per material (they multiply the slot tint)
    id(METAL): (0.8, 0.8, 0.8),  # steel a little darker than the sword fittings: big plates in the light band
    id(TRIM): (1.0, 0.8, 0.42),  # gold x the steel tint = bronze-gold
    id(LEATHER): (1.0, 1.0, 1.0),
    id(CLOTH): (1.0, 1.0, 1.0),
}
LIFT = 0.0022  # trims swell the shell by this much


def steel(shape):
    return shape.paint(color=(1, 1, 1), material=METAL)


def gold(shape):
    return shape.paint(color=(1, 1, 1), material=TRIM)


def leather(shape):
    return shape.paint(color=(1, 1, 1), material=LEATHER)


def cloth(shape):
    return shape.paint(color=(1, 1, 1), material=CLOTH)


def ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def band(lo, hi, x, soft=0.002):
    """1 inside [lo, hi], smooth 0 outside (ramps `soft` wide)."""
    return ss(lo - soft, lo + soft, x) * (1 - ss(hi - soft, hi + soft, x))


# ----------------------------------------------------------------------------- frames & cuts
_ARM_Q = arm_rot(1)
_ARM_R = m3.quat_to_matrix3(_ARM_Q)  # canonical left-arm frame -> rest space
_ARM_M = m3.translation_matrix(SHOULDER) @ m3.rotation_matrix(_ARM_Q)


def to_arm(P):
    """Rest-space points -> the canonical LEFT arm frame (shoulder joint at the origin, +X along the
    arm, +Y over the top of the arm, +Z front). Right-side points are folded onto the left."""
    P = np.array(np.atleast_2d(P), float)
    P[:, 0] = np.abs(P[:, 0])
    return (P - SHOULDER) @ _ARM_R


def from_arm(Q):
    return np.asarray(SHOULDER) + np.atleast_2d(Q) @ _ARM_R.T


def in_arm(shape):
    """A shape modeled in the canonical left arm frame, placed on the left rest arm."""
    return sdf.Transformed(shape, _ARM_M)


def keep_below(point, normal, extent=1.0):
    """Half-space solid on the side opposite to `normal` (a cut keeps it)."""
    return sdf.plane(point, m3.normalize(m3.vec3(normal)), extent=extent)


def slab_x(x0, x1):
    """x0 < x < x1 (canonical frames: along the arm)."""
    return keep_below((x1, 0, 0), (1, 0, 0)).intersect(keep_below((x0, 0, 0), (-1, 0, 0)))


def slab_y(y0, y1):
    return keep_below((0, y1, 0), (0, 1, 0)).intersect(keep_below((0, y0, 0), (0, -1, 0)))


def cut(shape, region, k=0.006):
    return shape.smooth_intersect(region, k)


def seg_dist(P, a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    t = np.clip((P - a) @ ab / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


# ----------------------------------------------------------------------------- breastplate
if FEMALE:
    PLATE_BOT = 1.145  # the breastplate stops over the waist cincher (it shows below)
    LAME = None
    BELT = (0.952, 1.0)  # hip belt over the skirt
else:
    PLATE_BOT = 1.118
    LAME = (C.BELT_Y[0] + 0.02, PLATE_BOT + 0.016)  # tucked under the plate, its foot under the belt
    BELT = (C.BELT_Y[0] - 0.006, C.BELT_Y[1] + 0.008)
T_PLATE = 0.031 if not FEMALE else 0.029
KEEL_H, KEEL_W = 0.015, 0.05
NECK_A, NECK_B = np.array([0, 1.3, -0.026]), np.array([0, 1.8, -0.02])
NECK_R = 0.106 if not FEMALE else 0.094
ARM_HOLE = 0.097 * B.ARM_K / 0.94
_HOLE_A = np.asarray(SHOULDER) + _ARM_R @ np.array([-0.005, 0.0, 0.0])
_HOLE_B = np.asarray(SHOULDER) + _ARM_R @ np.array([0.55, 0.0, 0.0])
BACK_Z, WAIST_UP = -0.035, 0.075  # Harpins: the back is open above PLATE_BOT + WAIST_UP, behind BACK_Z


def _upper():
    return sdf.smooth_union(B.torso(), B.shoulders(), k=0.05)


def _keel(P):
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    prof = np.clip(1 - np.abs(x) / KEEL_W, 0, 1) ** 1.3
    return KEEL_H * prof * ss(0.0, 0.07, z) * ss(PLATE_BOT + 0.01, PLATE_BOT + 0.09, y) * (1 - ss(1.33, 1.42, y))


def breastplate_trim(P):
    """Gold bands: around the neck, the arm holes and the lower edge (and a line on the keel)."""
    Pm = np.array(P, float)
    Pm[:, 0] = np.abs(Pm[:, 0])
    neck = seg_dist(P, NECK_A, NECK_B) - NECK_R
    hole = seg_dist(Pm, _HOLE_A, _HOLE_B) - ARM_HOLE
    y = P[:, 1]
    b = np.maximum(band(-0.01, 0.016, neck) * ss(1.37, 1.4, y), band(-0.01, 0.015, hole))
    b = np.maximum(b, band(PLATE_BOT - 0.004, PLATE_BOT + 0.015, y))
    crest = band(-1, 0.0062, np.abs(P[:, 0]), 0.0015) * ss(0.04, 0.06, P[:, 2]) * band(PLATE_BOT + 0.03, 1.33, y)
    b = np.maximum(b, crest)
    if HARPIN:
        back = band(BACK_Z - 0.01, BACK_Z + 0.015, P[:, 2]) * ss(PLATE_BOT + WAIST_UP - 0.02, PLATE_BOT + WAIST_UP, y)
        top = band(PLATE_BOT + WAIST_UP - 0.015, PLATE_BOT + WAIST_UP + 0.01, y) * ss(BACK_Z + 0.005, BACK_Z - 0.005, P[:, 2])
        b = np.maximum(b, np.maximum(back, top))
    return b


def breastplate():
    base = _upper()
    plate = Offset(base, lambda P: T_PLATE + _keel(P) + LIFT * breastplate_trim(P), t_max=T_PLATE + KEEL_H + LIFT)
    plate = cut(plate, keep_below((0, PLATE_BOT, 0), (0, -1, 0)), 0.006)
    if HARPIN:  # open back above the waist: the folded wings lie on the shoulder blades
        back = keep_below((0, 0, BACK_Z), (0, 0, -1))  # keep z > BACK_Z
        waist = keep_below((0, PLATE_BOT + WAIST_UP, 0), (0, 1, 0))
        plate = cut(plate, back.union(waist), 0.012)
    plate = plate.smooth_subtract(sdf.capsule(NECK_A, NECK_B, NECK_R), 0.008)
    plate = plate.smooth_subtract(sdf.capsule(_HOLE_A, _HOLE_B, ARM_HOLE).mirror_x(), 0.008)
    # over the top of the shoulder the plate stops under the pauldron dome
    stop = sdf.box((SHOULDER[0] + 0.2 - 0.035, SHOULDER[1] + 0.2 - 0.03, 0.0), (0.2, 0.2, 0.3)).mirror_x()
    plate = plate.smooth_subtract(stop, 0.01)
    layers = [(steel(plate), 0.85)]
    if LAME is not None:
        layers.append((steel(cut(Offset(base, T_PLATE - 0.005, t_max=T_PLATE), slab_y(*LAME), 0.006)), 0.15))
    return layers


# ----------------------------------------------------------------------------- pauldrons
# lames along the arm (arm-frame x): (start, end, lowest local y (how far under the arm it wraps));
# radii are fitted around the outfit and the breastplate (fit_pauldrons)
LAMES = [(-0.078, 0.078, 0.008), (0.058, 0.128, -0.02), (0.106, 0.176, -0.036)]
P_THICK, P_CLEAR = 0.009, 0.009
_FIT: dict = {}


def fit_pauldrons(occupied):
    """Radius of `occupied` (outfit + body + breastplate) around the left arm axis per arm-frame x
    (rows) and azimuth around the arm (columns, degrees from the top of the arm, + = front). Cached."""
    if "r" in _FIT:
        return _FIT["xs"], _FIT["th"], _FIT["r"]
    xs = np.arange(-0.1, 0.2001, 0.01)
    th = np.arange(-120, 121, 6)
    rs = np.linspace(0.0, 0.26, 261)
    t = np.radians(th)
    D = np.stack([np.zeros_like(t), np.cos(t), np.sin(t)], axis=1)
    Q = xs[:, None, None, None] * np.array([1.0, 0, 0]) + rs[None, None, :, None] * D[None, :, None, :]
    inside = (occupied.dist(from_arm(Q.reshape(-1, 3))) < 0).reshape(len(xs), len(th), len(rs))
    last = np.argmax(np.cumsum(inside, axis=2) == inside.sum(axis=2, keepdims=True), axis=2)
    _FIT["xs"], _FIT["th"], _FIT["r"] = xs, th, np.where(inside.any(axis=2), rs[last], 0.0)
    return _FIT["xs"], _FIT["th"], _FIT["r"]


def _pauldron_radii(occupied):
    """Per lame (w over the top, f front, k back): an elliptical section around the arm axis that
    clears everything under it by P_CLEAR, each lame a step out over the one below."""
    xs, th, r = fit_pauldrons(occupied)
    out = []
    for d0, d1, y_lo in LAMES:
        reach = 90 + np.degrees(np.arcsin(-y_lo / 0.105)) + 3  # how far around the arm it wraps
        lo = -reach if not HARPIN else -30
        m = (xs >= d0 - 0.005) & (xs <= d1 + 0.005)
        a = (th >= lo) & (th <= reach)
        need = r[np.ix_(m, a)].max(axis=0) + P_CLEAR + P_THICK  # per azimuth
        t = np.radians(th[a])
        top = np.abs(th[a]) <= 24
        w = need[top].max()
        f = max(need[th[a] >= 24].max(initial=0), w * 0.9)
        k = max(need[th[a] <= -24].max(initial=0), w * 0.9)
        ry = np.where(t >= 0, f, k)
        rad = 1 / np.sqrt((np.cos(t) / w) ** 2 + (np.sin(t) / ry) ** 2)
        s = max(1.0, float(np.max(need / rad)))
        out.append(np.array([w, f, k]) * s)
    for i in (1, 0):  # each lame steps out over the one below it
        out[i] = np.maximum(out[i], out[i + 1] + P_THICK + 0.005)
    return out


def pauldron_trim(P, layer=None):
    """Gold on the lower edge of the dome (wide) and of the last lame (a thin line)."""
    x = to_arm(P)[:, 0]
    (d0, d1, _), (_, e1, _) = LAMES[0], LAMES[-1]
    dome, last = band(d1 - 0.016, d1 + 0.01, x), band(e1 - 0.009, e1 + 0.01, x)
    if layer is not None:  # each lame its own band (the dome's must not paint the lame under it)
        dome = dome if layer == 0 else 0 * x
        last = last if layer == len(LAMES) - 1 else 0 * x
    b = np.maximum(dome, last)
    if HARPIN:
        b = np.maximum(b, band(-0.05, -0.024, to_arm(P)[:, 2]))
    return b


def pauldron_weights(P, names):
    """Rigid-ish: the lames ride the upper arm, the dome blends from the shoulder (its inner end)
    into the upper arm (no torn steel when the arm goes overhead)."""
    W = np.zeros((len(P), len(names)))
    x = to_arm(P)[:, 0]
    ua = ss(-0.1, 0.09, x)
    for side, pre in ((1, "Left"), (-1, "Right")):
        on = (np.sign(P[:, 0]) == side).astype(float)
        W[:, names.index(pre + "UpperArm")] += on * ua
        W[:, names.index(pre + "Shoulder")] += on * (1 - ua)
    return W


def pauldrons(occupied):
    radii = _pauldron_radii(occupied)
    lames = []
    for (d0, d1, y_lo), (w, f, k) in zip(LAMES, radii):
        if d0 < 0:  # the dome: rounder, swelling over the joint
            sw = {0.0: 0.84, 0.3: 0.98, 0.65: 1.02, 1.0: 0.99}
            W, F, K = ({t: v * g for t, g in sw.items()} for v in (w, f, k))
            Wi, Fi, Ki = ({t: v - P_THICK for t, v in d.items()} for d in (W, F, K))
            outer = Tube((d0, 0, 0), (d1, 0, 0), W, F, k=K, front=(0, 0, 1), cap=0.022)
            inner = Tube((d0 + P_THICK, 0, 0), (d1 + 0.03, 0, 0), Wi, Fi, k=Ki, front=(0, 0, 1), cap=0.012)
        else:
            outer = Tube((d0, 0, 0), (d1, 0, 0), w, f, k=k, front=(0, 0, 1), cap=0.006)
            inner = Tube((d0 - 0.03, 0, 0), (d1 + 0.03, 0, 0), w - P_THICK, f - P_THICK, k=k - P_THICK,
                         front=(0, 0, 1), cap=0.004)
        region = keep_below((0, y_lo, 0), (0, -1, 0), extent=0.5)
        if HARPIN:  # stop behind the shoulder: the wings
            region = region.intersect(keep_below((0, 0, -0.035), (0, 0, -1), extent=0.5))
        lame = cut(outer, region, 0.008) - inner
        lames.append(lame)
    layers = []
    for i, (lame, share) in enumerate(zip(lames, (0.5, 0.2, 0.2))):
        trim = (lambda P, i=i: pauldron_trim(P, i))
        shell = Offset(in_arm(lame), lambda P, trim=trim: LIFT * trim(P), t_max=LIFT)
        layers.append((steel(shell).mirror_x(), share, trim))
    # leather strap around the arm under the lowest lame
    arm = in_arm(B.arm_tube())
    strap = cut(Offset(arm, 0.021, t_max=0.025), in_arm(slab_x(0.18, 0.203)), 0.003)
    layers.append((leather(strap).mirror_x(), 0.1))
    return layers


# ----------------------------------------------------------------------------- vambraces
VB = (UPPER_ARM + FOREARM * 0.36, UPPER_ARM + FOREARM * 0.9)


def vambrace_trim(P):
    x = to_arm(P)[:, 0]
    return np.maximum(band(VB[0] - 0.01, VB[0] + 0.014, x), band(VB[1] - 0.014, VB[1] + 0.01, x))


def vambraces():
    arm = in_arm(B.arm_tube())

    def t_fn(P):
        q = to_arm(P)
        ridge = 0.0065 * np.clip(1 - np.abs(q[:, 2]) / 0.022, 0, 1) * ss(0.0, 0.02, q[:, 1])
        flare = 0.005 * ss(VB[0] + 0.05, VB[0], q[:, 0])  # the elbow end opens a little
        return 0.025 + ridge + flare + LIFT * vambrace_trim(P)
    core = cut(Offset(arm, t_fn, t_max=0.04), in_arm(slab_x(*VB)), 0.005)
    return [(steel(core).mirror_x(), 1.0)]


# ----------------------------------------------------------------------------- belt, tassets, tabard

def _belt_base():
    if FEMALE:
        return sdf.smooth_union(B.torso(), C._skirt(), k=0.03)
    return sdf.smooth_union(B.torso(), sdf.ellipsoid((0, 1.0, -0.01), (0.17, 0.08, 0.126)), k=0.04)


def _front_z(shape, y, x=0.0):
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if shape.dist(np.array([[x, y, mid]]))[0] < 0:
            lo = mid
        else:
            hi = mid
    return lo


T_BELT = 0.043 if not FEMALE else 0.036


def belt():
    base = _belt_base()
    strap = cut(Offset(base, T_BELT, t_max=T_BELT), slab_y(*BELT), 0.006)
    yc = 0.5 * (BELT[0] + BELT[1])
    z = _front_z(base, yc) + T_BELT
    if FEMALE:
        size, hole = (0.044, 0.042, 0.011), (0.024, 0.022, 0.016)
    else:  # covers the outfit's buckle (it stands 4.4 cm off the waist, 0.95 to 1.07 m)
        size, hole = (0.06, 0.072, 0.012), (0.032, 0.042, 0.02)
    bz = z + size[2] - 0.006
    buckle = sdf.box((0, yc, bz), size, radius=0.009)
    buckle = buckle.smooth_subtract(sdf.box((0, yc, bz + size[2]), hole, radius=0.006), 0.003)
    gem = steel(sdf.ellipsoid((0, yc, bz + size[2] - 0.006), (hole[0] * 0.75, hole[1] * 0.75, 0.009)))
    return [(leather(strap), 0.6), (sdf.union(gold(buckle), gem), 0.4)]


def _sector(a0, a1):
    """Vertical wedge between azimuths a0 < a1 (degrees, 0 = front (+Z), 90 = the left side (+X))."""
    def plane_at(a, sign):
        r = np.radians(a)
        d = np.array([np.sin(r), 0.0, np.cos(r)])
        return keep_below((0, 0, 0), np.cross([0.0, 1.0, 0.0], d) * sign)
    mid = np.radians((a0 + a1) / 2)
    return plane_at(a0, -1.0).intersect(plane_at(a1, 1.0)).intersect(
        keep_below((0, 0, 0), (-np.sin(mid), 0, -np.cos(mid))))


Y_TOP = BELT[0] + 0.012  # tassets hang from under the belt
HEM = C.TUNIC_HEM
_Y_MID = Y_TOP - (Y_TOP - (HEM + 0.07)) * 0.5
TASSET_LAMES = [(_Y_MID - 0.012, Y_TOP, 0.03), (HEM + 0.062, _Y_MID + 0.012, 0.023)]  # (bottom, top, offset)
TASSET_GROUPS = [(7.0, 52.0)] + ([] if HARPIN else [(130.0, 200.0)])  # front-left; back (left half)


def tasset_trim(P):
    y = P[:, 1]
    d = C._skirt().dist(P)
    b = np.zeros(len(P))
    for y0, y1, t in TASSET_LAMES:
        b = np.maximum(b, band(y0 - 0.005, y0 + 0.015, y) * ss(t - 0.0035, t - 0.0015, d))
    return b


def tassets():
    """Two lames over the front of each thigh (the upper one overlapping the lower), gold lower edges;
    humans: a back fauld."""
    sk = C._skirt()
    layers = []
    for y0, y1, t in TASSET_LAMES:  # one layer per row of lames (the upper row overlaps the lower)
        out = []
        for a0, a1 in TASSET_GROUPS:
            plate = cut(Offset(sk, lambda P, t=t: t + LIFT * tasset_trim(P), t_max=t + LIFT), slab_y(y0, y1), 0.007)
            out.append(cut(plate, _sector(a0, a1), 0.014))
        layers.append((steel(sdf.union(*out)).mirror_x(), 0.5))
    return layers


def tabard():
    """Cloth_B panel down the front, between the tassets, a little longer than the tunic."""
    hem = HEM - 0.055
    sk = C._skirt(extra=0.013, hem=hem)
    panel = cut(sk, slab_y(hem + 0.004, Y_TOP + 0.01), 0.006)
    return [(cloth(cut(panel, _sector(-13.0, 13.0), 0.012)), 1.0)]


# ----------------------------------------------------------------------------- greaves & knee cops

def _leg_axis_z(y):
    """z of the human leg tube's axis at height y (leg_tube: (x, 1, 0) -> (x, 0.055, -0.02))."""
    return -0.02 * (1.0 - np.asarray(y, float)) / (1.0 - 0.055)


G_RANGE = (0.135, 0.448)
G_BACK = 0.032  # the greave wraps to this far behind the leg's axis
KC_C = np.array([HIP[0], KNEE_Y + 0.012, _leg_axis_z(KNEE_Y) + 0.03])  # knee cop: an ellipsoid shell
KC_R = np.array([0.07, 0.08, 0.066]) * (0.93 if FEMALE else 1.0)
KC_CUT = 0.01  # the cop is the part of the shell in front of z = center + KC_CUT


def greave_trim(P):
    Pm = np.array(P, float)
    Pm[:, 0] = np.abs(Pm[:, 0])
    y = Pm[:, 1]
    b = np.maximum(band(G_RANGE[1] - 0.015, G_RANGE[1] + 0.01, y), band(G_RANGE[0] - 0.01, G_RANGE[0] + 0.013, y))
    back = Pm[:, 2] - (_leg_axis_z(y) - G_BACK)
    # knee cop rim
    kc = Pm[:, 2] - (KC_C[2] + KC_CUT)
    b = np.maximum(b, band(-0.01, 0.013, kc) * (np.linalg.norm((Pm - KC_C) / KC_R, axis=1) < 1.25))
    return b


def greaves_human():
    leg = B.leg_tube(1)
    x = HIP[0]

    def t_fn(P):
        ridge = 0.008 * np.clip(1 - np.abs(P[:, 0] - x) / 0.022, 0, 1) * ss(0.0, 0.03, P[:, 2] - _leg_axis_z(P[:, 1]))
        return 0.028 + ridge + LIFT * greave_trim(P)
    plate = cut(Offset(leg, t_fn, t_max=0.04), slab_y(*G_RANGE), 0.006)
    behind = sdf.custom(lambda P: (_leg_axis_z(P[:, 1]) - G_BACK) - P[:, 2], (x - 0.2, 0.0, -0.3), (x + 0.2, 0.6, 0.3))
    plate = cut(plate, behind, 0.01)
    straps = [cut(Offset(leg, 0.024, t_max=0.03), slab_y(y - 0.011, y + 0.011), 0.003) for y in (0.2, 0.36)]
    # knee cop: an ellipsoid shell over the front of the knee
    cop = sdf.ellipsoid(KC_C, KC_R) - sdf.ellipsoid(KC_C, KC_R - 0.011)
    cop = cut(cop, keep_below(KC_C + np.array([0, 0, KC_CUT]), (0, 0, -1), extent=0.3), 0.006)
    cop = Offset(cop, lambda P: LIFT * greave_trim(P), t_max=LIFT)
    return [(steel(plate).mirror_x(), 0.55), (steel(cop).mirror_x(), 0.25),
            (leather(sdf.union(*straps)).mirror_x(), 0.2)]


def greaves_harpin():
    import harpin as HP
    from common import H_HOCK, H_KNEE
    knee = H_KNEE + np.array([0, 0.03, 0.01])
    ax = H_HOCK - knee
    L = float(np.linalg.norm(ax))
    u = ax / L
    fwd = np.array([0.0, 0.0, 1.0]) - u * u[2]
    fwd /= np.linalg.norm(fwd)
    core_leg = HP.leg_core(1)
    s0, s1 = 0.1, 0.5

    def trim(P):
        Pm = np.array(P, float)
        Pm[:, 0] = np.abs(Pm[:, 0])
        q = Pm - knee
        s = q @ u / L
        b = np.maximum(band(s0 - 0.03, s0 + 0.04, s), band(s1 - 0.04, s1 + 0.03, s))
        b = np.maximum(b, band(-0.03, -0.004, q @ fwd) * band(s0, s1, s))
        return b

    def t_fn(P):
        q = P - knee
        side = q - np.outer(q @ u, u)
        lat = np.abs(P[:, 0] - HIP[0])  # from the front line of the shin
        ridge = 0.007 * np.clip(1 - lat / 0.02, 0, 1) * ss(0.0, 0.03, side @ fwd)
        return 0.034 + ridge + LIFT * trim(P)
    sh = Offset(core_leg, t_fn, t_max=0.05)
    reg = keep_below(knee + u * L * s1, u).intersect(keep_below(knee + u * L * s0, -u))
    reg = reg.intersect(keep_below(knee - fwd * 0.015, -fwd))  # the front half and a bit
    plate = cut(sh, reg, 0.01)
    # knee cop on the forward knee, below the tunic hem
    c = H_KNEE + np.array([0.0, -0.012, -0.004])
    R = np.array([0.064, 0.07, 0.074])
    cop = sdf.ellipsoid(c, R) - sdf.ellipsoid(c, R - 0.011)
    cop = cut(cop, keep_below(c + np.array([0, 0, 0.012]), (0, 0, -1), extent=0.3), 0.006)
    cop = cut(cop, keep_below((0, C.TUNIC_HEM - 0.012, 0), (0, 1, 0)), 0.006)
    return [(steel(plate).mirror_x(), 0.7), (steel(cop).mirror_x(), 0.3)], trim


# ----------------------------------------------------------------------------- build

def project_on(shape, P, iters=4):
    Q = np.array(P, float)
    for _ in range(iters):
        d = shape.dist(Q)
        g = shape.gradient(Q, eps=0.0015)
        Q = Q - g * d[:, None]
    return Q


def mesh_piece(shape, target, max_edge, name, hidden, before_decimate=None, taubin=4, rings=4):
    """meshing.mesh_sdf for armor pieces, with the open borders free to simplify: a piece is cut
    open where it dives under the outfit (hidden faces dropped), and those long borders (straps
    around a sleeve, a dome's foot in the breastplate) would otherwise keep every fine-grid vertex."""
    t0 = time.time()
    m = shape.to_mesh(cell=CELL, name=name, band=1.5, padding=0.05)
    n0 = m.num_triangles
    m.smooth_taubin(taubin)
    shape.project(m, iterations=2, max_step=CELL)
    drop_faces(m, hidden(m.vertices), rings=rings)
    if before_decimate is not None:
        before_decimate(m)
    n1 = m.num_triangles
    if m.num_triangles > target:
        m.decimate(target, max_edge=max_edge, keep_boundary=False, border_weight=250)
    shape.project(m, iterations=2, max_step=CELL)
    drop_faces(m, hidden(m.vertices))
    m.normals = shape.gradient(m.vertices, eps=CELL * 0.25)
    print(f"  {name}: {n0} -> {n1} -> {m.num_triangles} tris ({time.time() - t0:.1f}s)")
    return m


def mark_trim(trim_fn, threshold=0.5):
    """before_decimate hook: Metal faces inside the trim bands become the trim material, then the
    vertices on the border slide (along the surface) onto the band's iso-line: a straight gold edge
    instead of the grid's staircase (the decimation keeps material borders)."""
    def f(m):
        if trim_fn is None or not len(m.faces):
            return
        mats = list(m.materials)
        if METAL not in mats:
            return
        if TRIM not in mats:
            mats.append(TRIM)
            m.materials = mats
        fm = np.asarray(m.face_materials).copy()
        hit = (fm == mats.index(METAL)) & (trim_fn(m.face_centers()) > threshold)
        fm[hit] = mats.index(TRIM)
        m.face_materials = fm
        F = np.asarray(m.faces, dtype=np.int64)
        if F.ndim != 2:
            return
        tv = np.zeros(m.num_vertices, bool)
        mv = np.zeros(m.num_vertices, bool)
        tv[F[fm == mats.index(TRIM)].ravel()] = True
        mv[F[fm == mats.index(METAL)].ravel()] = True
        border = np.flatnonzero(tv & mv)
        if not len(border):
            return
        N = m.vertex_normals()[border]
        P = m.vertices[border].copy()
        e = 0.0006
        for _ in range(4):
            g = np.stack([(trim_fn(P + d) - trim_fn(P - d)) / (2 * e) for d in np.eye(3) * e], axis=1)
            g -= N * np.einsum("ij,ij->i", g, N)[:, None]
            g2 = np.einsum("ij,ij->i", g, g)
            ok = g2 > 1e-6
            step = np.zeros_like(P)
            step[ok] = ((threshold - trim_fn(P))[ok] / g2[ok])[:, None] * g[ok]
            n = np.linalg.norm(step, axis=1, keepdims=True)
            step *= np.minimum(1.0, (CELL * 0.6) / np.maximum(n, 1e-12))
            P += step
        m.vertices[border] = P
    return f


def paint_by_material(m, extra=None):
    """Splits the mesh by material (border vertices duplicated) and colors each part flat."""
    parts = []
    fm = np.asarray(m.face_materials)
    for mi, mat in enumerate(m.materials):
        faces = np.flatnonzero(fm == mi)
        if not len(faces):
            continue
        part = m.separate(faces, name=m.name)
        part.remove_unused_vertices()
        c = np.ones((part.num_vertices, 4))
        c[:, :3] = m3.srgb_to_linear(np.asarray(COLORS[id(mat)], float))
        if extra is not None:
            c[:, :3] *= extra(part, mat)[:, None]
        part.colors = c
        parts.append(part)
    return p3.Mesh.join(parts, name=m.name)


def tabard_shade(part, mat):
    """Tabard: a darker band along the hem."""
    y = part.vertices[:, 1]
    return 1.0 - 0.3 * ss(HEM - 0.025, HEM - 0.045, y)


def build():
    t0 = time.time()
    body = B.body_sdf()
    garments = [make() for _, make in C.GARMENTS]
    plate = breastplate()
    occupied = sdf.union(body, *garments, *[lay[0] for lay in plate])
    if HARPIN:
        greaves, g_trim = greaves_harpin()
    else:
        greaves, g_trim = greaves_human(), greave_trim
    pieces = {  # name: (layers [(shape, budget share)], trim bands, triangle budget, max edge)
        "Breastplate": (plate, breastplate_trim, 1350, 0.06),
        "Pauldrons": (pauldrons(occupied), pauldron_trim, 1150, 0.05),
        "Vambraces": (vambraces(), vambrace_trim, 400, 0.045),
        "Belt": (belt(), None, 360, 0.06),
        "Tassets": (tassets(), tasset_trim, 700 if not HARPIN else 400, 0.045),
        "Tabard": (tabard(), None, 80, 0.06),
        "Greaves": (greaves, g_trim, 1100 if not HARPIN else 800, 0.05),
    }
    print("pauldron radii (w, f, k)", [np.round(r, 3).tolist() for r in _pauldron_radii(occupied)])
    meshes = {}
    for name, (layers, trim, budget, max_edge) in pieces.items():
        parts = []
        for i, (shape, share, *own) in enumerate(layers):
            # hidden under the body, the outfit, the other pieces and the other layers of this one
            others = [lay[0] for n, (ls, *_r) in pieces.items() if n not in (name, "Tabard") for lay in ls]
            others += [layer[0] for j, layer in enumerate(layers) if j != i]
            hidden = covered_by([body] + garments + others, margin=0.0025)
            m = mesh_piece(shape, max(int(budget * share), 40), max_edge, f"{name}{i}", hidden,
                           before_decimate=mark_trim(own[0] if own else trim))
            if m.num_triangles:
                parts.append(paint_by_material(m, tabard_shade if name == "Tabard" else None))
        meshes[name] = p3.Mesh.join(parts, name=name)
    sk = build_skeleton(final=True)
    names = [b.name for b in sk.bones]
    head = B.head_scaled()
    for name, m in meshes.items():
        if name in ("Tassets", "Tabard"):  # like the tunic skirt
            RG.skin_mesh(m, sk, head, skirt_mask=m.vertices[:, 1] < 1.05)
        elif name == "Belt":  # like the outfit's belt under it
            RG.skin_mesh(m, sk, head)
        elif name == "Pauldrons":
            m.joints, m.weights = RG.top4(pauldron_weights(m.vertices, names))
        else:  # the skin right under the plate (the breastplate: the torso's skin, never the arm's)
            W = RG.weight_matrix(project_on(_upper() if name == "Breastplate" else body, m.vertices), names, head)
            m.joints, m.weights = RG.top4(W)
    stats = {}
    for name, m in meshes.items():
        stats[name] = m.num_triangles
        xf_mesh(m)
    mesh = p3.Mesh.join(list(meshes.values()), name="Armor_Guard")
    probs = mesh.validate()
    if probs:
        print("  validate", probs)
    scene = p3.Scene(NAME)
    scene.add(sk.root)
    scene.add(p3.skinned_node(mesh, sk, "Armor_Guard"))
    out = export_glb(scene, NAME, sub="equipment")
    stats["total"] = mesh.num_triangles
    fm = np.asarray(mesh.face_materials)
    stats["materials"] = {mat.name + ("_trim" if mat is TRIM else ""): int(np.sum(fm == mi))
                          for mi, mat in enumerate(mesh.materials)}
    lo, hi = mesh.bounds()
    stats["bounds"] = [[round(float(v), 3) for v in lo], [round(float(v), 3) for v in hi]]
    stats["time_s"] = round(time.time() - t0, 1)
    rev = REVIEW / "armor"
    rev.mkdir(parents=True, exist_ok=True)
    (rev / f"{NAME}_stats.json").write_text(json.dumps(stats, indent=1))
    print(NAME, stats, "->", out)
    return mesh, sk


# ----------------------------------------------------------------------------- review

def review_scene(mesh, sk):
    """The body the game imports with the armor re-bound to its own bones by name (like the client),
    the sword in the hand, the default hair, review lights and tints."""
    from common import review_lights, tint_for_review
    sc = p3.load(HERE / "exports" / "characters" / f"{BODY}.glb")  # the body the game imports
    skinned = [n for n in sc.walk() if n.mesh is not None and n.skin is not None]
    skel = skinned[0].skin
    bnames = [b.name for b in skel.bones]
    mine = [b.name for b in sk.bones]
    missing = [n for n in mine if n not in bnames]
    if missing:
        raise SystemExit(f"{NAME}: bones missing on the body: {missing}")
    remap = np.array([bnames.index(n) for n in mine])
    m2 = mesh.copy()
    m2.joints = remap[mesh.joints]
    node = p3.Node("Armor_Guard", mesh=m2)
    node.skin = skel
    sc.add(node)
    default = ("plumado" if FEMALE else "espetado") if HARPIN else ("longo" if FEMALE else "curto")
    for n in sc.walk():
        if n.name.startswith("Hair_"):
            n.visible = n.name == f"Hair_{default}"
    sword_glb = HERE / "exports" / "equipment" / "sword_iron.glb"
    sock = next((n for n in sc.walk() if n.name == "Socket_Weapon_R"), None)
    if sock is not None and sword_glb.exists():
        sw = next(n for n in p3.load(sword_glb).walk() if n.mesh is not None)
        sock.add(p3.Node("Sword", mesh=sw.mesh))
    review_lights(sc)
    tint_for_review(sc)
    return sc


def review(mesh, sk):
    """Front / 3-4 / side / back at rest, Attack1 (wind-up, hit, follow-through) from the front and
    from behind, close-ups of the torso and the legs, a Run strip."""
    from pyram3d.image import grid, write_png
    sc = review_scene(mesh, sk)
    out = REVIEW / "armor"
    out.mkdir(parents=True, exist_ok=True)
    kw = dict(style="toon", yfov=30)
    views = [(0.0, 0.08, 1.0), (0.65, 0.15, 1.0), (1.0, 0.1, 0.0), (-0.55, 0.2, -1.0)]
    imgs = [p3.render(sc, None, 420, 640, view=v, target=(0, 0.95, 0), distance=3.6, time=0.0, animation="Idle", **kw)
            for v in views]
    write_png(out / f"{NAME}.png", grid(imgs, 4))
    imgs = []
    for v in [(0.55, 0.25, 1.0), (-0.6, 0.3, -1.0)]:
        for t in (0.08, 0.2, 0.32):
            imgs.append(p3.render(sc, None, 420, 520, view=v, target=(0, 1.05, 0), distance=3.4, time=t,
                                  animation="Attack1", **kw))
    write_png(out / f"{NAME}_Attack1.png", grid(imgs, 3))
    imgs = [p3.render(sc, None, 460, 460, view=v, target=(0, 1.22, 0), distance=1.5, time=0.0, animation="Idle", **kw)
            for v in [(0.0, 0.1, 1.0), (0.8, 0.25, 0.6), (-0.7, 0.3, -0.7)]]
    imgs += [p3.render(sc, None, 460, 460, view=v, target=(0, 0.55, 0), distance=1.7, time=0.0, animation="Idle", **kw)
             for v in [(0.35, 0.1, 1.0), (1.0, 0.1, -0.2), (-0.5, 0.2, -1.0)]]
    write_png(out / f"{NAME}_closeup.png", grid(imgs, 3))
    if "--quick" not in sys.argv:
        imgs = [p3.render(sc, None, 360, 520, view=(1.0, 0.25, 0.25), target=(0, 0.95, 0), distance=3.6, time=t,
                          animation="Run", **kw) for t in (0.0, 0.16, 0.32, 0.48)]
        write_png(out / f"{NAME}_Run.png", grid(imgs, 4))


if __name__ == "__main__":
    cache = HERE / ".cache" / f"{NAME}_last.pkl"  # the last build (--review-only re-renders it)
    if "--review-only" in sys.argv:
        mesh, sk = pickle.loads(cache.read_bytes()), build_skeleton(final=True)
    else:
        mesh, sk = build()
        cache.parent.mkdir(exist_ok=True)
        cache.write_bytes(pickle.dumps(mesh))
    if "--no-review" not in sys.argv:
        review(mesh, sk)
