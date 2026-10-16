"""The wolf of the Campos de Rondoria: stylized, angular (Hades-like), rigged and animated.

    python projects/niryk/wolf.py [--fast]

~1.65 m nose to tail tip, 0.8 m at the shoulder. Dark saddle, gray flanks, pale belly, chest
ruff and muzzle, yellow eyes; a jaw bone opens the mouth for the bite. Clips: Idle, Walk, Run
(gallop), Attack (crouch, lunge, bite), Hit, Death. Gaits use 2-bone IK so planted paws don't slide.
"""
from __future__ import annotations

import math
import sys

import numpy as np

import anim as A
from common import REVIEW, export_glb

import pyram3d as p3
from pyram3d import sdf
from pyram3d.noise import fbm3

FAST = "--fast" in sys.argv
FPS = 30
SPEED = {"walk": 2.0, "run": 6.5}
DUR = {"Idle": 2.0, "Walk": 0.9, "Run": 0.5, "Attack": 0.75, "Hit": 0.35, "Death": 1.1}
ATTACK_HIT = 0.42  # bite frame (seconds)

FUR = p3.Material("Wolf_Fur", "#ffffff", roughness=0.9)
EYE = p3.Material("Wolf_Eye", "#f2c94c", emissive="#f2a92c", emissive_strength=0.6, roughness=0.4)
MOUTH = p3.Material("Wolf_Mouth", "#3a2026", roughness=0.7)
DARK = p3.Material("Wolf_Dark", "#262028", roughness=0.6)
TEETH = p3.Material("Wolf_Teeth", "#efe8d8", roughness=0.5)

SADDLE = (0.25, 0.25, 0.3)
FLANK = (0.5, 0.51, 0.56)
PALE = (0.83, 0.8, 0.75)

# ----------------------------------------------------------------------------- rig

J = {
    "Hips": ((0, 0.72, -0.42), (0, 0.74, -0.2)),
    "Spine1": ((0, 0.74, -0.2), (0, 0.76, 0.04)),
    "Spine2": ((0, 0.76, 0.04), (0, 0.8, 0.26)),
    "Neck": ((0, 0.82, 0.3), (0, 0.95, 0.52)),
    "Head": ((0, 0.95, 0.52), (0, 0.92, 0.82)),
    "Jaw": ((0, 0.9, 0.58), (0, 0.86, 0.8)),
    "LeftEar": ((0.05, 1.03, 0.55), (0.075, 1.15, 0.52)),
    "FrontLeftUpper": ((0.13, 0.72, 0.28), (0.14, 0.45, 0.2)),
    "FrontLeftLower": ((0.14, 0.45, 0.2), (0.13, 0.13, 0.25)),
    "FrontLeftPaw": ((0.13, 0.13, 0.25), (0.13, 0.03, 0.31)),
    "FrontLeftToes": ((0.13, 0.03, 0.31), (0.13, 0.02, 0.38)),
    "BackLeftUpper": ((0.12, 0.7, -0.44), (0.13, 0.46, -0.32)),
    "BackLeftLower": ((0.13, 0.46, -0.32), (0.13, 0.21, -0.5)),
    "BackLeftHock": ((0.13, 0.21, -0.5), (0.13, 0.04, -0.45)),
    "BackLeftToes": ((0.13, 0.04, -0.45), (0.13, 0.02, -0.37)),
}
TAIL = [(0, 0.74, -0.56), (0, 0.69, -0.7), (0, 0.6, -0.83), (0, 0.49, -0.93), (0, 0.38, -0.99)]
PARENT = {"Spine1": "Hips", "Spine2": "Spine1", "Neck": "Spine2", "Head": "Neck", "Jaw": "Head",
          "Ear": "Head", "FrontUpper": "Spine2", "FrontLower": "FrontUpper", "FrontPaw": "FrontLower",
          "FrontToes": "FrontPaw", "BackUpper": "Hips", "BackLower": "BackUpper", "BackHock": "BackLower",
          "BackToes": "BackHock"}


def mirror(p):
    return (-p[0], p[1], p[2])


def build_rig():
    sk = p3.Skeleton("WolfRig")
    for n in ("Hips", "Spine1", "Spine2", "Neck", "Head", "Jaw"):
        a, b = J[n]
        sk.add_bone(n, a, b, parent=PARENT.get(n))
    for side in ("Left", "Right"):
        a, b = J["LeftEar"]
        if side == "Right":
            a, b = mirror(a), mirror(b)
        sk.add_bone(f"{side}Ear", a, b, parent="Head")
    for part in ("FrontUpper", "FrontLower", "FrontPaw", "FrontToes", "BackUpper", "BackLower", "BackHock", "BackToes"):
        front, rest = part[:4] if part.startswith("Back") else part[:5], part[4:] if part.startswith("Back") else part[5:]
        key = f"{front}Left{rest}"
        a, b = J[key]
        for side in ("Left", "Right"):
            pa, pb = (a, b) if side == "Left" else (mirror(a), mirror(b))
            par = PARENT[part]
            if par not in ("Hips", "Spine2"):
                pf = par[:4] if par.startswith("Back") else par[:5]
                par = f"{pf}{side}{par[len(pf):]}"
            sk.add_bone(f"{front}{side}{rest}", pa, pb, parent=par)
    sk.add_chain([f"Tail{i}" for i in range(1, 5)], TAIL, parent="Hips")
    return sk


# ----------------------------------------------------------------------------- body

def fur(shape, col, shade=1.0):
    c = tuple(np.clip(np.array(col) * shade, 0, 1))
    return shape.paint(color=c, material=FUR)


def _clumps(amount, scale=22.0, seed=0):
    """Angular fur clumps: sharpened noise, mostly outward."""
    def f(p):
        n = fbm3(p * scale, octaves=2, seed=seed)
        return -np.clip(np.abs(n) * 2.6 - 0.15, -0.3, 1.0)
    return dict(amount=amount, func=f)


def _spikes(base, direction, n, length, radius, spread, color, seed=0, shade=1.0):
    """A fan of angular fur tufts (thin cones) growing from `base` roughly along `direction`."""
    rng = np.random.default_rng(seed)
    d0 = np.asarray(direction, float)
    d0 /= np.linalg.norm(d0)
    out = []
    for i in range(n):
        off = np.asarray(spread, float) * (rng.random(3) * 2 - 1)
        a = np.asarray(base, float) + off
        d = d0 + rng.normal(0, 0.18, 3)
        d /= np.linalg.norm(d)
        ln = length * (0.75 + 0.5 * rng.random())
        out.append(fur(sdf.round_cone(tuple(a), tuple(a + d * ln), radius, 0.003), color, shade * (0.92 + 0.12 * rng.random())))
    return out


def wolf_sdf():
    parts = []
    # torso: heavy chest and shoulders, tucked waist, lean haunches (Hades-like wedge silhouette)
    parts.append(fur(sdf.ellipsoid((0, 0.8, 0.17), (0.165, 0.215, 0.23)), FLANK))  # chest
    parts.append(fur(sdf.ellipsoid((0, 0.87, 0.12), (0.14, 0.09, 0.17)), SADDLE, 1.05))  # shoulder hump
    parts.append(fur(sdf.ellipsoid((0, 0.76, -0.12), (0.11, 0.12, 0.25)), FLANK))  # waist
    parts.append(fur(sdf.ellipsoid((0, 0.75, -0.42), (0.125, 0.14, 0.16)), FLANK))  # haunches
    parts.append(fur(sdf.ellipsoid((0, 0.85, -0.1), (0.085, 0.06, 0.38)), SADDLE))  # dark saddle along the back
    parts.append(fur(sdf.ellipsoid((0, 0.65, 0.22), (0.12, 0.17, 0.15)), PALE))  # chest ruff
    parts.append(fur(sdf.ellipsoid((0, 0.66, -0.12), (0.075, 0.055, 0.2)), PALE, 0.95))  # belly
    # neck and an angular mane swept back over the shoulders
    parts.append(fur(sdf.round_cone((0, 0.82, 0.28), (0, 0.94, 0.48), 0.15, 0.105), FLANK))
    parts.append(fur(sdf.ellipsoid((0, 0.91, 0.36), (0.14, 0.12, 0.11)), SADDLE, 1.1))  # mane
    parts += _spikes((0, 0.95, 0.4), (0, 0.35, -1), 9, 0.17, 0.035, (0.09, 0.04, 0.05), SADDLE, seed=1, shade=1.05)
    parts += [p.mirror_x() for p in _spikes((0.11, 0.86, 0.4), (0.6, 0.05, -1), 5, 0.14, 0.03, (0.02, 0.06, 0.05), FLANK, seed=2)]
    parts += _spikes((0, 0.55, 0.24), (0, -0.6, 0.35), 6, 0.12, 0.03, (0.07, 0.02, 0.03), PALE, seed=3)  # chest tuft
    # head: bigger wedge skull, strong brow, long snout
    parts.append(fur(sdf.ellipsoid((0, 0.975, 0.58), (0.095, 0.085, 0.11)), FLANK))  # skull
    parts.append(fur(sdf.ellipsoid((0, 1.01, 0.6), (0.075, 0.035, 0.07)), SADDLE, 1.15))  # brow / forehead
    parts += [p.mirror_x() for p in _spikes((0.075, 0.93, 0.57), (0.7, -0.2, -0.6), 3, 0.09, 0.028, (0.01, 0.02, 0.02), PALE, seed=4)]  # cheek ruff
    parts.append(fur(sdf.round_cone((0, 0.955, 0.63), (0, 0.925, 0.83), 0.056, 0.03), FLANK, 1.05))  # snout top
    parts.append(fur(sdf.round_cone((0, 0.9, 0.63), (0, 0.885, 0.8), 0.046, 0.024), PALE))  # muzzle / jaw
    parts.append(sdf.sphere((0, 0.942, 0.845), 0.022).paint(color=(1, 1, 1), material=DARK))  # nose
    # ears: tall triangles
    ear = sdf.round_cone((0.055, 1.03, 0.56), (0.088, 1.18, 0.52), 0.04, 0.006)
    parts.append(fur(ear, SADDLE).mirror_x())
    # legs: muscular tops, gray shanks, pale only at the paws
    for (a, b, r1, r2, c) in [
        ((0.13, 0.76, 0.26), (0.14, 0.45, 0.2), 0.082, 0.046, FLANK),
        ((0.14, 0.45, 0.2), (0.13, 0.13, 0.25), 0.04, 0.03, FLANK),
        ((0.13, 0.13, 0.25), (0.13, 0.035, 0.33), 0.034, 0.032, PALE),
        ((0.12, 0.73, -0.42), (0.13, 0.46, -0.32), 0.095, 0.054, FLANK),
        ((0.13, 0.46, -0.32), (0.13, 0.21, -0.5), 0.045, 0.032, FLANK),
        ((0.13, 0.21, -0.5), (0.13, 0.04, -0.44), 0.03, 0.028, FLANK),
        ((0.13, 0.035, -0.45), (0.13, 0.03, -0.37), 0.034, 0.03, PALE),
    ]:
        parts.append(fur(sdf.round_cone(a, b, r1, r2), c).mirror_x())
    # elbow and heel tufts
    parts += [p.mirror_x() for p in _spikes((0.15, 0.47, 0.17), (0, -0.4, -1), 2, 0.07, 0.022, (0.01, 0.01, 0.01), FLANK, seed=5)]
    # tail: bushy, angular tufts, dark tip
    tail = sdf.chain(TAIL, [0.05, 0.08, 0.085, 0.072, 0.03], k=0.04)
    parts.append(fur(tail, FLANK))
    parts += _spikes(TAIL[2], (0, -0.3, -1), 6, 0.12, 0.03, (0.05, 0.08, 0.06), FLANK, seed=6)
    parts.append(fur(sdf.round_cone(TAIL[-2], (0, 0.3, -1.03), 0.05, 0.006), SADDLE, 0.8))
    body = sdf.smooth_union(*parts, k=0.03).displace(**_clumps(0.006, 10, 3))
    # mouth opening line (the jaw bone opens it)
    body = body.smooth_subtract(sdf.ellipsoid((0.0, 0.905, 0.76), (0.048, 0.006, 0.08)).paint(material=MOUTH, color=(1, 1, 1)), 0.004)
    return body


def eyes_mesh():
    parts = []
    for sx in (1, -1):
        e = p3.uv_sphere(0.02, 10, 6).scale((1, 0.62, 1)).rotate((0, sx * 35, 0)).translate((sx * 0.058, 0.99, 0.665))
        parts.append(e.set_material(EYE))
        pupil = p3.uv_sphere(0.008, 6, 4).scale((0.5, 1, 0.5)).translate((sx * 0.069, 0.99, 0.68))
        parts.append(pupil.set_material(DARK))
    teeth = []
    for sx in (1, -1):  # fangs, visible when biting
        teeth.append(p3.cone(0.007, 0.03, 6).rotate((180, 0, 0)).translate((sx * 0.025, 0.905, 0.78)).set_material(TEETH))
    return p3.Mesh.join(parts, name="WolfEyes"), p3.Mesh.join(teeth, name="WolfFangs")


# ----------------------------------------------------------------------------- animation

class Leg:
    def __init__(self, sk, names, bend_forward):
        self.names = names
        h = [sk.head(n) for n in names] + [sk.tail(names[-1])]
        self.L1 = float(np.linalg.norm((h[1] - h[0])[1:]))
        self.L2 = float(np.linalg.norm((h[2] - h[1])[1:]))
        self.L3 = float(np.linalg.norm((h[3] - h[2])[1:]))
        ang = lambda d: math.atan2(d[2], -d[1])  # noqa: E731
        self.a1, self.a2, self.a3, self.a4 = ang(h[1] - h[0]), ang(h[2] - h[1]), ang(h[3] - h[2]), ang(h[4] - h[3])
        self.root = h[0]
        self.paw = h[3]  # planted point (paw pad)
        self.bend = 1 if bend_forward else -1

    def solve(self, parent_pitch, root_off, paw_target, paw_angle):
        """Rotations (deg) so the paw pad lands on paw_target with the paw bone at paw_angle."""
        az = paw_target[2] - self.L3 * math.sin(paw_angle)
        ay = paw_target[1] + self.L3 * math.cos(paw_angle)
        z = az - (self.root[2] + root_off[2])
        y = ay - (self.root[1] + root_off[1])
        D = min(max(1e-4, math.hypot(z, y)), (self.L1 + self.L2) * 0.999)
        alpha = math.atan2(z, -y)
        beta = math.acos(max(-1.0, min(1.0, (self.L1 ** 2 + D ** 2 - self.L2 ** 2) / (2 * self.L1 * D))))
        a1 = alpha + self.bend * beta
        kz, ky = self.L1 * math.sin(a1), -self.L1 * math.cos(a1)
        a2 = math.atan2(z - kz, -(y - ky))
        r1 = -(a1 - self.a1) - parent_pitch
        r2 = -(a2 - self.a2) - (parent_pitch + r1)
        r3 = -(paw_angle - self.a3) - (parent_pitch + r1 + r2)
        r4 = -(self.a4 - self.a4) - (parent_pitch + r1 + r2 + r3) + 0.0
        return [math.degrees(v) for v in (r1, r2, r3, r4)]


def sample(sk, legs, clip, t):
    dur = DUR[clip]
    ph = t / dur
    tp = 2 * math.pi
    P = {}
    off = np.zeros(3)
    spine_pitch = 0.0
    head = [0.0, 0.0]  # pitch, yaw
    jaw = 0.0
    ears = 0.0
    tail_amp, tail_lift = 6.0, 0.0
    gait = None
    if clip == "Idle":
        b = math.sin(ph * tp)
        off = np.array([0, 0.004 * b, 0])
        head = [4 * math.sin(ph * tp * 0.5), 10 * math.sin(ph * tp * 0.5 + 1)]
        ears = 6 * max(0.0, math.sin(ph * tp * 2)) ** 8
        gait = ("stand", None)
    elif clip in ("Walk", "Run"):
        run = clip == "Run"
        S = SPEED["run" if run else "walk"] * dur
        if run:
            phases = {"BackLeft": 0.0, "BackRight": 0.08, "FrontLeft": 0.5, "FrontRight": 0.58}
            stance, lift = 0.32, 0.17
            off = np.array([0, 0.035 * math.sin(ph * tp - 0.6) - 0.02, 0])
            spine_pitch = 9 * math.sin(ph * tp)
            head = [6 * math.sin(ph * tp + 1.0) - 6, 0]
            tail_amp, tail_lift = 4.0, -12
            ears = -18
        else:
            phases = {"BackLeft": 0.0, "FrontLeft": 0.25, "BackRight": 0.5, "FrontRight": 0.75}
            stance, lift = 0.62, 0.07
            off = np.array([0, 0.008 * math.cos(ph * tp * 2), 0])
            head = [3 * math.sin(ph * tp * 2), 0]
        gait = ("move", (phases, stance, lift, S))
    elif clip == "Attack":
        crouch = A.smooth(t / 0.25) * (1 - A.smooth((t - 0.32) / 0.08))
        lunge = A.smooth((t - 0.25) / 0.17) * (1 - A.smooth((t - 0.48) / 0.27))
        off = np.array([0, -0.12 * crouch + 0.04 * lunge, 0.55 * lunge])
        spine_pitch = 8 * crouch - 6 * lunge
        head = [-14 * crouch + 18 * lunge, 0]
        jaw = 32 * A.smooth((t - 0.26) / 0.12) * (1 - A.smooth((t - ATTACK_HIT) / 0.06))
        ears = -25 * max(crouch, lunge)
        tail_lift = -10 * crouch
        gait = ("lunge", (crouch, lunge))
    elif clip == "Hit":
        k = math.sin(math.pi * min(1.0, ph))
        off = np.array([0, -0.03 * k, -0.12 * k])
        head = [-18 * k, 14 * k]
        ears = -25 * k
        gait = ("stand", None)
    elif clip == "Death":
        f = A.smooth(ph / 0.75)
        off = np.array([0.0, -0.5 * f, -0.05 * f])
        gait = ("dead", f)
        head = [10 * f, 20 * f]
        jaw = 12 * f
        tail_amp = 0
    hips_pitch = spine_pitch * 0.4
    P["Hips"] = A.qm(A.qz(80 * gait[1]) if gait[0] == "dead" else A.I, A.qx(hips_pitch))
    P["Spine1"] = A.qx(-spine_pitch * 0.4)
    P["Spine2"] = A.qx(-spine_pitch * 0.4)
    P["Neck"] = A.qm(A.qy(head[1] * 0.4), A.qx(-head[0] * 0.5 + spine_pitch * 0.4))
    P["Head"] = A.qm(A.qy(head[1] * 0.6), A.qx(-head[0] * 0.5))
    P["Jaw"] = A.qx(jaw)
    P["LeftEar"] = A.qm(A.qz(-6), A.qx(ears))
    P["RightEar"] = A.qm(A.qz(6), A.qx(ears))
    for i in range(1, 5):
        w = math.sin(ph * tp * (2 if clip in ("Walk", "Run") else 1) - i * 0.8)
        P[f"Tail{i}"] = A.qm(A.qy(tail_amp * w), A.qx(tail_lift / 4))
    front_pitch = math.radians(hips_pitch - spine_pitch * 0.8)
    for name, leg in legs.items():
        is_front = name.startswith("Front")
        pitch = front_pitch if is_front else math.radians(hips_pitch)
        root_off = off if not is_front else off
        rest_paw = leg.paw
        pa = leg.a3
        if gait[0] == "move":
            phases, stance, lift, S = gait[1]
            u = (ph + phases[name]) % 1.0
            a = S * stance / 2
            if u < stance:
                s = u / stance
                fz, fy = a - 2 * a * s, 0.0
                pa = leg.a3 - math.radians(25) * A.smooth((s - 0.7) / 0.3)
            else:
                s = (u - stance) / (1 - stance)
                fz = -a + 2 * a * A.smooth(s)
                fy = lift * math.sin(math.pi * s)
                pa = leg.a3 + math.radians(40 if is_front else -10) * math.sin(math.pi * s)
            target = (rest_paw[0], rest_paw[1] + fy, rest_paw[2] + fz)
        elif gait[0] == "lunge":
            crouch, lunge = gait[1]
            reach = 0.32 * lunge if is_front else 0.18 * lunge
            target = (rest_paw[0], rest_paw[1] + (0.12 * lunge if is_front else 0.0), rest_paw[2] + reach + 0.55 * lunge * (0 if is_front else 0.85))
        elif gait[0] == "dead":
            f = gait[1]
            r = [(-25 if is_front else 20) * f, (30 if is_front else -25) * f, 10 * f, 0]
            for nm, v in zip(leg.names, r):
                P[nm] = A.qx(v)
            continue
        else:
            target = tuple(rest_paw)
        r = leg.solve(pitch, root_off, target, pa)
        for nm, v in zip(leg.names, r):
            P[nm] = A.qx(v)
    return P, off


def bake(scene, sk, legs):
    rest = sk.bone("Hips").extras["rest_translation"]
    for clip, dur in DUR.items():
        anim = scene.animation(clip)
        loop = clip in ("Idle", "Walk", "Run")
        anim.loop = loop
        n = int(round(dur * FPS))
        for i in range(n + 1):
            t = i / FPS if i < n else dur
            P, off = sample(sk, legs, clip, t if (i < n or not loop) else 0.0)
            for b in sk.bones:
                anim.key(b, "rotation", t, P.get(b.name, A.I), "LINEAR")
            anim.key(sk.bone("Hips"), "position", t, rest + off, "LINEAR")


def build():
    sk = build_rig()
    body = wolf_sdf().to_mesh(cell=0.02 if FAST else 0.0125, name="WolfBody")
    names = [b.name for b in sk.bones]

    def filt(P, i):
        n = names[i]
        if "Left" in n:
            return P[:, 0] > -0.02
        if "Right" in n:
            return P[:, 0] < 0.02
        if n == "Jaw":
            return (P[:, 1] < 0.905) & (P[:, 2] > 0.6)
        return np.ones(len(P), bool)

    sk.bind(body, smooth=6, falloff=5.0, bone_filter=filt)
    eyes, fangs = eyes_mesh()
    rig_idx = {n: i for i, n in enumerate(names)}
    for m, bone in ((eyes, "Head"), (fangs, "Jaw")):
        m.joints = np.zeros((m.num_vertices, 4), np.int64)
        m.weights = np.zeros((m.num_vertices, 4))
        m.joints[:, 0] = rig_idx[bone]
        m.weights[:, 0] = 1
    scene = p3.Scene("Wolf")
    scene.add(sk.root)
    for m, nm in ((body, "WolfBody"), (eyes, "WolfEyes"), (fangs, "WolfFangs")):
        scene.add(p3.skinned_node(m, sk, nm))
    legs = {
        "FrontLeft": Leg(sk, ["FrontLeftUpper", "FrontLeftLower", "FrontLeftPaw", "FrontLeftToes"], False),
        "FrontRight": Leg(sk, ["FrontRightUpper", "FrontRightLower", "FrontRightPaw", "FrontRightToes"], False),
        "BackLeft": Leg(sk, ["BackLeftUpper", "BackLeftLower", "BackLeftHock", "BackLeftToes"], True),
        "BackRight": Leg(sk, ["BackRightUpper", "BackRightLower", "BackRightHock", "BackRightToes"], True),
    }
    bake(scene, sk, legs)
    out = export_glb(scene, "wolf", sub="creatures")
    print("wolf", body.num_triangles, "tris ->", out)
    rv = REVIEW / "creatures"
    rv.mkdir(parents=True, exist_ok=True)
    p3.render_views(scene, rv / "wolf_turnaround.png", views=("front", "three_quarter", "right", "back"), size=420, cols=4,
                    style="toon", time=0.0, animation="Idle")
    for clip in DUR:
        p3.render_frames(scene, rv / f"wolf_{clip}.png", clip, frames=6, size=300, style="toon", view="right")
    return scene


if __name__ == "__main__":
    build()
