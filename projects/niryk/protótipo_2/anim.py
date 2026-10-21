"""Baked animation clips for the prototype 2 humans (30 fps, every bone keyed LINEAR).

Poses are authored in character space (faces +Z, left = +X) as world directions and solved into
local bone rotations, so they don't depend on the rest pose (an A-pose here):
  - legs: 3D two-bone IK to ankle targets, knees toward a pole; feet roll on the heel or the ball
    (the pivot stays planted), so planted feet never slide: each clip's stride is exactly
    speed x duration (SPEEDS, matching src/shared/constants.ts of the game);
  - arms: upper-arm and forearm directions (elbow hinge solved from them), hand frames (fingers,
    thumb); the sword hand's thumb axis IS the blade direction (Socket_Weapon_R +Y);
  - spine/head: small Euler layers.
Loops are seamless (the last key repeats the first). Clips: Idle, Walk, Run, WalkBack, StrafeLeft,
StrafeRight, Attack1, Attack2, Hit, Death.
"""
from __future__ import annotations

import math

import numpy as np

import common  # noqa: F401  (puts the Pyram3D root on sys.path)
from common import FEMALE, HARPIN, TAIL_BONES, _WN
import pyram3d as p3
from pyram3d import math3d as m3

FPS = 30
WALK, RUN = 1.9, 5.0
BACK = WALK * 0.7  # BACKPEDAL_FACTOR
DUR = {"Idle": 2.4, "Walk": 1.0, "Run": 0.64, "WalkBack": 1.0, "StrafeLeft": 0.8, "StrafeRight": 0.8,
       "Attack1": 0.5, "Attack2": 0.55, "Hit": 0.36, "Death": 1.2}
if HARPIN:  # quick light steps: the digitigrade legs plant without stretching (no foot skating)
    DUR.update({"Walk": 0.76, "Run": 0.52, "WalkBack": 0.8, "StrafeLeft": 0.68, "StrafeRight": 0.68})
LOOPING = ("Idle", "Walk", "Run", "WalkBack", "StrafeLeft", "StrafeRight")
HIT_TIME = {"Attack1": 0.2, "Attack2": 0.24}

I = np.array([0.0, 0.0, 0.0, 1.0])


def qx(a):
    return m3.quat_from_axis_angle((1, 0, 0), a)


def qy(a):
    return m3.quat_from_axis_angle((0, 1, 0), a)


def qz(a):
    return m3.quat_from_axis_angle((0, 0, 1), a)


def qaxis(axis, a):
    return m3.quat_from_axis_angle(m3.normalize(np.asarray(axis, float)), a)


def qm(*qs):
    out = I
    for q in qs:
        out = m3.quat_mul(out, q)
    return out


def qinv(q):
    return m3.quat_conj(q)


def rot(q, v):
    return m3.quat_rotate(q, np.asarray(v, float))


def nrm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def frame_align(a0, b0, a1, b1):
    """Rotation taking a0 -> a1 exactly and b0 (orthogonalized) -> b1 (orthogonalized)."""
    def basis(a, b):
        a = nrm(a)
        b = np.asarray(b, float) - a * np.dot(b, a)
        b = nrm(b)
        return np.stack([a, b, np.cross(a, b)], axis=1)
    M = basis(a1, b1) @ basis(a0, b0).T
    return m3.matrix3_to_quat(M)


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def keyframes(t, keys):
    """Piecewise smooth interpolation of {time: value}."""
    ts = sorted(keys)
    if t <= ts[0]:
        return np.asarray(keys[ts[0]], float)
    for a, b in zip(ts, ts[1:]):
        if t <= b:
            u = smooth((t - a) / (b - a))
            return np.asarray(keys[a], float) * (1 - u) + np.asarray(keys[b], float) * u
    return np.asarray(keys[ts[-1]], float)


# ----------------------------------------------------------------------------- rig

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

    def world_rot(self, pose, bone):
        chain = []
        b = bone
        while b is not None:
            chain.append(b)
            b = self.parent[b]
        q = I
        for name in reversed(chain):
            q = qm(q, pose.get(name, I))
        return q

    def world_head(self, pose, bone):
        chain = []
        b = bone
        while b is not None:
            chain.append(b)
            b = self.parent[b]
        chain.reverse()
        p = self.head[chain[0]] + pose.hips
        q = I
        prev = chain[0]
        for name in chain[1:]:
            q = qm(q, pose.get(prev, I))
            p = p + rot(q, self.head[name] - self.head[prev])
            prev = name
        return p

    def set_world(self, pose, bone, q_world):
        par = self.parent[bone]
        pw = self.world_rot(pose, par) if par else I
        pose[bone] = qm(qinv(pw), q_world)

    # ------------------------------------------------------------------ legs
    def leg(self, pose, side, ankle, pole, foot_q, toes_q=None):
        """Two-bone IK: thigh + shin reach `ankle` (world), knee toward `pole`; feet set in world."""
        hip = self.world_head(pose, side + "UpperLeg")
        L1 = self.len[side + "UpperLeg"]
        L2 = self.len[side + "LowerLeg"]
        d = np.asarray(ankle, float) - hip
        D = float(np.linalg.norm(d))
        D = min(max(D, abs(L1 - L2) + 1e-4), (L1 + L2) * 0.9995)
        dn = nrm(d)
        a = (L1 * L1 - L2 * L2 + D * D) / (2 * D)
        h = math.sqrt(max(L1 * L1 - a * a, 0.0))
        pp = nrm(np.asarray(pole, float) - dn * np.dot(pole, dn))
        knee = hip + dn * a + pp * h
        ank = hip + dn * D
        th = nrm(knee - hip)
        sh = nrm(ank - knee)
        z = np.array([0.0, 0.0, 1.0])
        self.set_world(pose, side + "UpperLeg", frame_align(self.dir[side + "UpperLeg"], z, th, pp))
        self.set_world(pose, side + "LowerLeg", frame_align(self.dir[side + "LowerLeg"], z, sh, pp))
        self.set_world(pose, side + "Foot", foot_q)
        self.set_world(pose, side + "Toes", toes_q if toes_q is not None else foot_q)

    # ------------------------------------------------------------------ arms
    def arm(self, pose, side, U, F, fingers=None, thumb=None, twist_share=0.5):
        """Arm from world directions: upper arm U, forearm F (the elbow hinge follows), hand frame
        (fingers direction, thumb direction); by default the hand continues the forearm."""
        U, F = nrm(U), nrm(F)
        u0 = self.dir[side + "UpperArm"]
        h0 = nrm(np.cross(u0, [0, 0, 1]))  # elbow flexion swings the forearm toward the front
        c = float(np.clip(np.dot(U, F), -1, 1))
        theta = math.degrees(math.acos(c))
        if theta > 0.5:
            h = nrm(np.cross(U, F))
        else:  # straight: keep the hinge as horizontal as possible
            h = nrm(np.cross(U, [0, 0, 1]) if abs(U[2]) < 0.95 else np.cross(U, [1, 0, 0]))
        Rup = frame_align(u0, h0, U, h)
        Rfo = qm(qaxis(h, theta), Rup)
        if fingers is None:
            fingers = F
        if thumb is None:
            thumb = rot(Rfo, [0, 0, 1])
        Rh = frame_align(u0, [0, 0, 1], fingers, thumb)
        # share the wrist twist with the forearm (no candy-wrapper at the wrist)
        rel = qm(qinv(Rfo), Rh)
        ax = u0
        tw = np.array([*(ax * np.dot(rel[:3], ax)), rel[3]])
        tw = tw / max(np.linalg.norm(tw), 1e-9)
        ang = 2 * math.degrees(math.atan2(np.linalg.norm(tw[:3]), tw[3])) * (1 if np.dot(tw[:3], ax) >= 0 else -1)
        if ang > 180:
            ang -= 360
        if ang < -180:
            ang += 360
        Rfo = qm(Rfo, qaxis(ax, ang * twist_share))
        self.set_world(pose, side + "UpperArm", Rup)
        self.set_world(pose, side + "LowerArm", Rfo)
        self.set_world(pose, side + "Hand", Rh)

    def sword_arm(self, pose, ua, fa, blade):
        """Right arm holding the sword: the blade direction is exact, the fingers wrap
        perpendicular to it, as close to the forearm direction as possible."""
        b = nrm(blade)
        fa = nrm(fa)
        fingers = fa - b * np.dot(fa, b)
        if np.linalg.norm(fingers) < 1e-3:
            fingers = np.cross(b, [0, 1, 0])
        self.arm(pose, "Right", ua, fa, fingers=nrm(fingers), thumb=b)


def swing_dir(down_side, swing_deg, side, out_deg=0.0):
    """Arm hanging down, `out_deg` away from the body, swung forward (+) / back (-)."""
    a = math.radians(out_deg)
    v = np.array([side * math.sin(a), -math.cos(a), 0.0])
    return rot(qx(-swing_deg), v)


def relaxed_arm(rig, P, side, swing=0.0, out=10.0, elbow=15.0):
    sgn = 1 if side == "Left" else -1
    U = swing_dir(None, swing, sgn, out)
    a = math.radians(out * 0.6)
    F = rot(qx(-(swing + elbow)), np.array([sgn * math.sin(a), -math.cos(a), 0.0]))
    rig.arm(P, side, U, F)


# ----------------------------------------------------------------------------- feet & locomotion

class Feet:
    def __init__(self, rig: Rig):
        self.rig = rig
        self.A0 = {s: rig.head[s + "Foot"] for s in ("Left", "Right")}  # ankles
        self.ball = {s: rig.head[s + "Toes"] for s in ("Left", "Right")}
        self.heel = {s: self.A0[s] + np.array([0.0, -self.A0[s][1] + 0.012, -0.05]) for s in ("Left", "Right")}
        if HARPIN:  # digitigrade: the foot pivots on the ball both ways; the toes stay flat
            self.heel = dict(self.ball)

    def place(self, P, side, T, roll, yaw=0.0, pole=None, toes_flat=True, curl=0.0):
        """Foot of `side` translated by T (character space) from rest, pitched by `roll` deg
        (+ = heel up, pivot on the ball; - = toes up, pivot on the heel), turned by `yaw`.
        Harpins: the foot is the tarsus (its 'heel' is the hock), the toes stay flat on the ground
        unless `curl` (deg, the claws curling under) is given."""
        A0 = self.A0[side]
        piv = self.ball[side] if roll >= 0 else self.heel[side]
        Vp = piv - A0
        Ry = qy(yaw)
        Rf = qm(Ry, qx(roll))
        ank = A0 + np.asarray(T, float) + rot(Ry, Vp) - rot(Rf, Vp)
        pole = rot(Ry, [0, 0, 1]) if pole is None else pole
        if HARPIN:
            toes = qm(Ry, qx(curl))
        else:
            toes = Ry if (toes_flat and roll > 0) else Rf
        self.rig.leg(P, side, ank, pole, Rf, toes)


def locomotion(rig, feet, P, phase, travel, speed, dur, stance=0.62, lift=0.1, roll_off=30.0, roll_strike=-14.0,
               spread=None, foot_yaw=0.0, pole_yaw=0.0, curl=0.0):
    """Planted feet move backward along `travel` at exactly `speed` (stride = speed * dur)."""
    S = speed * dur
    travel = nrm(travel)
    for side, off in (("Left", 0.0), ("Right", 0.5)):
        u = (phase + off) % 1.0
        a = S * stance / 2
        if u < stance:
            s = u / stance
            along = a - 2 * a * s
            up = 0.0
            if s < 0.18:
                roll = roll_strike * (1 - smooth(s / 0.18))
            elif s > 0.6:
                roll = roll_off * smooth((s - 0.6) / 0.4)
            else:
                roll = 0.0
        else:
            s = (u - stance) / (1 - stance)
            along = -a + 2 * a * smooth(s)
            up = lift * math.sin(math.pi * s) ** 1.1
            roll = roll_off * (1 - smooth(s / 0.45)) + roll_strike * smooth((s - 0.6) / 0.4)
        T = travel * along + np.array([0.0, up, 0.0])
        if spread is not None:
            T = T + np.asarray(spread[side], float)
        fy = foot_yaw if side == "Left" else foot_yaw
        cu = curl * math.sin(math.pi * s) ** 1.5 if u >= stance else 0.0  # claws curl under in the swing
        feet.place(P, side, T, roll, yaw=fy, pole=rot(qy(pole_yaw), [0, 0, 1]), curl=cu)


SWORD_IDLE = dict(ua=(-0.22, -0.96, -0.1), fa=(-0.08, -0.85, 0.5), blade=(0.06, -0.2, 1.0))
SWORD_CARRY = dict(blade=(-0.32, -0.62, -0.7))


def sample(rig: Rig, feet: Feet, clip: str, t: float) -> Pose:
    dur = DUR[clip]
    phase = t / dur
    P = Pose()
    two_pi = 2 * math.pi
    if HARPIN and clip == "Death":
        harpin_death(rig, feet, P, t)
    elif clip in ("Walk", "Run", "WalkBack", "StrafeLeft", "StrafeRight"):
        run = clip == "Run"
        back = clip == "WalkBack"
        strafe = 0 if not clip.startswith("Strafe") else (1 if clip == "StrafeLeft" else -1)
        cyc = math.sin(phase * two_pi)
        bob_amp = 0.035 if run else 0.02
        crouch = (0.06 if run else 0.015) if not HARPIN else (0.11 if run else (0.095 if strafe else 0.075))
        P.hips = np.array([0.0, -bob_amp * 0.5 + bob_amp * 0.5 * math.cos(phase * two_pi * 2) * (-1 if run else 1)
                           - crouch, 0.0])
        lean = 14 if run else (6 if back else 4)
        yaw = (9 if run else 6) * cyc
        hip_turn = 28.0 * strafe  # strafing: the hips turn toward the travel, the chest stays front
        P["Hips"] = qm(qy(yaw + hip_turn), qx(lean * 0.5), qz((4.0 if FEMALE else 2.5) * cyc))
        P["Spine"] = qm(qy(-yaw * 0.5 - hip_turn * 0.5), qx(lean * 0.3))
        P["Chest"] = qm(qy(-yaw * 0.9 - hip_turn * 0.5), qx(lean * 0.2))
        P["Neck"] = qx(-lean * 0.5)
        P["Head"] = qm(qx(-lean * 0.4), qy(yaw * 0.3))
        # female: the feet land closer to the center line
        track = {"Left": (-0.02, 0.0, 0.0), "Right": (0.02, 0.0, 0.0)} if FEMALE else None
        if HARPIN:
            harpin_steps(rig, feet, P, phase, dur, run, back, strafe, track)
        elif run:
            locomotion(rig, feet, P, phase, (0, 0, 1), RUN, dur, stance=0.4, lift=0.2, roll_off=40, roll_strike=-8,
                       spread=track)
        elif back:
            # walking backward: the swing goes back; toes touch first, the heel lowers
            locomotion(rig, feet, P, phase, (0, 0, -1), BACK, dur, stance=0.62, lift=0.08, roll_off=-10,
                       roll_strike=14)
        elif strafe:
            # side steps along X: the leading leg reaches out, the feet pass one in front of the
            # other (crossing at different depths) so the legs never collide
            spread = {"Left": (0.0, 0.0, 0.05 * strafe), "Right": (0.0, 0.0, -0.05 * strafe)}
            locomotion(rig, feet, P, phase, (strafe, 0, 0), WALK, dur, stance=0.58, lift=0.09, roll_off=18,
                       roll_strike=-6, spread=spread, foot_yaw=hip_turn * 0.8, pole_yaw=hip_turn)
        else:
            locomotion(rig, feet, P, phase, (0, 0, 1), WALK, dur, spread=track)
        arm_sw = (48 if run else (14 if back else 22)) * cyc * (0.5 if strafe else 1.0)
        if HARPIN:  # shorter backswing: the folded wings hang behind the arms
            arm_sw = arm_sw * (0.7 if arm_sw < 0 else 0.85)
        relaxed_arm(rig, P, "Left", swing=arm_sw, out=(10 if run else 7) if FEMALE else (12 if run else 10),
                    elbow=75 if run else 18)
        # right hand carries the sword: blade down and back, the arm swinging opposite
        sw_r = math.radians(25 if run else 15) * math.sin(phase * two_pi + math.pi) * (0.5 if strafe else 1.0)
        ua = (-0.2, -math.cos(sw_r), math.sin(sw_r) + (0.12 if run else 0.04))
        fa = (-0.16, -0.86, 0.48 + (0.25 if run else 0.0) + 0.6 * math.sin(sw_r))
        rig.sword_arm(P, ua=ua, fa=fa, blade=(-0.3, -0.62 if not run else -0.5, -0.72))
    elif clip == "Idle":
        br = math.sin(phase * two_pi)
        P.hips = np.array([0.012 if FEMALE else 0.0, -0.01 + 0.004 * br, 0.0])
        P["Hips"] = qm(qy(-6), qz(3.5)) if FEMALE else qy(-6)  # female: weight on one leg
        P["Spine"] = qm(qx(1.5 * br), qy(2), qz(-2.0 if FEMALE else 0.0))
        P["Chest"] = qm(qx(1.5 * br), qy(4), qz(-1.5 if FEMALE else 0.0))
        P["Head"] = qm(qy(2 + 3 * math.sin(phase * two_pi * 0.5)), qx(-2), qz(2.5 if FEMALE else 0.0))
        stand(rig, feet, P, wide=0.02 if FEMALE else 0.05)
        relaxed_arm(rig, P, "Left", swing=4 + 2 * br, out=8 if FEMALE else 12, elbow=22 if FEMALE else 16)
        rig.sword_arm(P, **SWORD_IDLE)
    elif clip in ("Attack1", "Attack2"):
        attack(rig, feet, P, clip, t)
    elif clip == "Hit":
        k = math.exp(-((t - 0.08) / 0.09) ** 2) if t > 0.08 else smooth(t / 0.08)
        P.hips = np.array([0.0, -0.01 - 0.04 * k, -0.08 * k])
        P["Hips"] = qx(-8 * k)
        P["Spine"] = qx(-10 * k)
        P["Chest"] = qm(qx(-12 * k), qy(-10 * k))
        P["Neck"] = qx(-10 * k)
        P["Head"] = qm(qx(-15 * k), qz(6 * k))
        stand(rig, feet, P, wide=0.05)
        relaxed_arm(rig, P, "Left", swing=-15 * k, out=12 + 25 * k, elbow=25 + 45 * k)
        ua0, fa0, bl0 = (np.array(SWORD_IDLE[n]) for n in ("ua", "fa", "blade"))
        rig.sword_arm(P, ua=lerp(ua0, np.array([-0.45, -0.88, -0.15]), k), fa=lerp(fa0, np.array([-0.35, -0.9, 0.2]), k),
                      blade=lerp(bl0, np.array([-0.3, -0.85, 0.4]), k))
    elif clip == "Death":
        b = smooth(t / 0.3)  # knees buckle
        f = smooth((t - 0.22) / 0.5)  # heavy fall backward with a twist
        P.hips = np.array([0.04 * f, -0.22 * b - 0.62 * f, -0.3 * f])
        P["Hips"] = qm(qy(20 * f), qx(10 * b * (1 - f) - 88 * f), qz(10 * f))
        P["Spine"] = qx(12 * b * (1 - f) - 2 * f)
        P["Chest"] = qm(qx(10 * b * (1 - f) + 2 * f), qy(-10 * f))
        P["Neck"] = qx(12 * b - 6 * f)
        P["Head"] = qm(qx(15 * b - 5 * f), qy(35 * f), qz(10 * f))
        if f < 0.02:
            stand(rig, feet, P, wide=0.05)
        else:  # legs fold loosely as the body lands
            knees = 60 * b + 10 * f
            for side, sgn in (("Left", 1), ("Right", -1)):
                P[side + "UpperLeg"] = qm(qz(-sgn * (6 + 10 * f) * (1.4 if side == "Left" else 1)), qx(-knees * 0.55 - 30 * f))
                P[side + "LowerLeg"] = qx(knees * (1.1 if side == "Left" else 0.8))
                P[side + "Foot"] = qx(-20 * f)
        relaxed_arm(rig, P, "Left", swing=lerp(4, 25, b) - 45 * f, out=lerp(12, 40, f), elbow=lerp(16, 55, b))
        ua0, fa0, bl0 = (np.array(SWORD_IDLE[n]) for n in ("ua", "fa", "blade"))
        rig.sword_arm(P, ua=lerp(ua0, np.array([-0.8, -0.55, 0.2]), f), fa=lerp(fa0, np.array([-0.7, -0.3, 0.65]), f),
                      blade=lerp(bl0, np.array([-0.5, -0.8, 0.3]), f))
    if HARPIN:
        harpin_layers(rig, P, clip, t)
    return P


def stand(rig, feet, P, wide=0.05, fwd=(0.04, -0.03)):
    """Feet planted under the hips, a little apart and staggered; the IK bends the knees to the
    lowered hips."""
    for side, sgn, dz in (("Left", 1, fwd[0]), ("Right", -1, fwd[1])):
        feet.place(P, side, (sgn * wide * 0.5, 0.0, dz), 0.0, yaw=sgn * 6)


def attack(rig, feet, P, clip, t):
    """Attack1: horizontal slash right -> left. Attack2: diagonal backhand left -> right (combo)."""
    if clip == "Attack1":
        chest = {0.0: (0, 4, 0), 0.12: (-35, 8, 0), 0.2: (30, -4, 0), 0.3: (50, -8, 0), 0.4: (40, -2, 0), 0.5: (4, 4, 0)}
        ua = {0.0: (-0.25, -0.95, 0.25), 0.12: (-0.85, 0.3, -0.35), 0.2: (0.05, 0.05, 1.0), 0.3: (0.8, -0.3, 0.45),
              0.4: (0.55, -0.75, 0.3), 0.5: (-0.25, -0.95, 0.25)}
        fa = {0.0: (-0.1, -0.55, 0.85), 0.12: (-0.5, 0.6, -0.6), 0.2: (0.35, 0.0, 0.95), 0.3: (0.9, -0.25, -0.1),
              0.4: (0.6, -0.7, 0.25), 0.5: (-0.1, -0.55, 0.85)}
        blade = {0.0: (0.06, -0.2, 1.0), 0.12: (-0.5, 0.3, -0.8), 0.2: (0.95, -0.05, 0.3), 0.3: (0.4, -0.15, -0.9),
                 0.4: (0.3, -0.75, 0.4), 0.5: (0.06, -0.2, 1.0)}
        step = {0.0: 0.0, 0.12: -0.04, 0.2: 0.26, 0.3: 0.3, 0.4: 0.26, 0.5: 0.0}
        crouch = {0.0: 0.0, 0.12: 0.07, 0.2: 0.06, 0.3: 0.08, 0.5: 0.0}
    else:
        chest = {0.0: (45, -4, 0), 0.12: (55, 2, 0), 0.24: (-30, -2, 0), 0.36: (-50, -4, 0), 0.46: (-35, 0, 0), 0.55: (0, 4, 0)}
        ua = {0.0: (0.6, -0.6, 0.5), 0.12: (0.75, 0.55, -0.2), 0.24: (-0.25, 0.1, 0.97), 0.36: (-0.9, -0.3, 0.3),
              0.46: (-0.6, -0.7, 0.3), 0.55: (-0.25, -0.95, 0.25)}
        fa = {0.0: (0.7, -0.5, 0.4), 0.12: (0.5, 0.75, -0.4), 0.24: (-0.6, -0.15, 0.8), 0.36: (-0.75, -0.45, -0.45),
              0.46: (-0.5, -0.6, 0.6), 0.55: (-0.1, -0.55, 0.85)}
        blade = {0.0: (0.4, -0.6, -0.6), 0.12: (0.45, 0.6, -0.65), 0.24: (-0.9, -0.3, 0.3), 0.36: (-0.4, -0.55, -0.75),
                 0.46: (-0.3, -0.5, 0.7), 0.55: (0.06, -0.2, 1.0)}
        step = {0.0: 0.2, 0.12: 0.18, 0.24: 0.3, 0.36: 0.3, 0.46: 0.2, 0.55: 0.0}
        crouch = {0.0: 0.06, 0.12: 0.07, 0.24: 0.06, 0.36: 0.08, 0.55: 0.0}
    yaw, pitch, _ = keyframes(t, chest)
    st = float(keyframes(t, step))
    cr = float(keyframes(t, crouch))
    P.hips = np.array([0.0, -0.03 - cr, st])
    P["Hips"] = qy(yaw * 0.4)
    P["Spine"] = qm(qy(yaw * 0.3), qx(pitch * 0.5))
    P["Chest"] = qm(qy(yaw * 0.35), qx(pitch * 0.5))
    P["Head"] = qy(-yaw * 0.6)
    # lunge: left foot forward with the body, right foot back, both planted
    for side, sgn, dz in (("Left", 1, 0.1 + st), ("Right", -1, -0.2 + st * 0.35)):
        feet.place(P, side, (sgn * 0.03, 0.0, dz), 0.0, yaw=sgn * 8 + yaw * 0.2)
    # free arm tucked against the chest
    rig.arm(P, "Left", nrm((0.3, -0.85, 0.42)), nrm((-0.6, -0.15, 0.78)))
    rig.sword_arm(P, ua=keyframes(t, ua), fa=keyframes(t, fa), blade=keyframes(t, blade))


def sword_drop(rig: Rig, feet: Feet, sock_t, sock_q):
    """Death: Socket_Weapon_R local (translation, rotation) over time: the sword leaves the hand at
    0.45 s and ends flat on the ground beside it."""
    end = sample(rig, feet, "Death", DUR["Death"])
    R = rig.world_rot(end, "RightHand")
    H = rig.world_head(end, "RightHand")
    u = nrm(np.array([0.35, 0.0, 0.94]))
    X = np.array([H[0] - 0.05, 0.03, H[2] + 0.28])
    Q = frame_align(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]), u, np.array([0.0, 1.0, 0.0]))
    t_end = rot(qinv(R), X - H)
    q_end = qm(qinv(R), Q)

    def at(t):
        k = smooth((t - 0.45) / 0.35)
        return np.asarray(sock_t) * (1 - k) + t_end * k, m3.quat_slerp(np.asarray(sock_q), q_end, k)
    return at


def bake(scene: p3.Scene, rig: Rig, clips=None):
    feet = Feet(rig)
    clips = clips or list(DUR)
    hips_rest = rig.sk.bone("Hips").extras["rest_translation"]
    sock = next((c for c in rig.sk.bone("RightHand").children if c.name == "Socket_Weapon_R"), None)
    drop = sword_drop(rig, feet, sock.translation, sock.rotation) if sock is not None else None
    for clip in clips:
        anim = scene.animation(clip)
        anim.loop = clip in LOOPING
        dur = DUR[clip]
        n = int(round(dur * FPS))
        for i in range(n + 1):
            t = i / FPS if i < n else dur
            P = sample(rig, feet, clip, t if (i < n or not anim.loop) else 0.0)
            for b in rig.sk.bones:
                anim.key(b, "rotation", t, P.get(b.name, I), "LINEAR")
            anim.key(rig.sk.bone("Hips"), "position", t, hips_rest + P.hips, "LINEAR")
            if drop is not None:
                st, sq = drop(t) if clip == "Death" else (sock.translation, sock.rotation)
                anim.key(sock, "position", t, st, "LINEAR")
                anim.key(sock, "rotation", t, sq, "LINEAR")
    return scene


# ----------------------------------------------------------------------------- Harpins

def harpin_steps(rig, feet, P, phase, dur, run, back, strafe, track):
    """Digitigrade steps: the toes plant flat and stay put while the tarsus rocks over the ball
    (push-off), the claws curl under in the swing."""
    if run:
        locomotion(rig, feet, P, phase, (0, 0, 1), RUN, dur, stance=0.34, lift=0.17, roll_off=38, roll_strike=-6,
                   spread=track, curl=45)
    elif back:
        locomotion(rig, feet, P, phase, (0, 0, -1), BACK, dur, stance=0.6, lift=0.08, roll_off=-8, roll_strike=18,
                   curl=20)
    elif strafe:
        hip_turn = 28.0 * strafe
        spread = {"Left": (0.0, 0.0, 0.05 * strafe), "Right": (0.0, 0.0, -0.05 * strafe)}
        locomotion(rig, feet, P, phase, (strafe, 0, 0), WALK, dur, stance=0.56, lift=0.09, roll_off=14, roll_strike=0,
                   spread=spread, foot_yaw=hip_turn * 0.8, pole_yaw=hip_turn, curl=20)
    else:
        locomotion(rig, feet, P, phase, (0, 0, 1), WALK, dur, stance=0.58, lift=0.1, roll_off=28, roll_strike=-6,
                   spread=track, curl=30)


_WING_OPEN = {}


def wing_targets(rig):
    """Local rotations (Wing1, Wing2, Wing3) of the wings fully opened ('display' spread: arm out
    and up, primaries out and down, the outer side facing back), per side."""
    key = id(rig)
    if key in _WING_OPEN:
        return _WING_OPEN[key]
    out = {}
    for side, sx in (("Left", 1.0), ("Right", -1.0)):
        m = np.array([sx, 1.0, 1.0])
        n0 = _WN * m
        D = [nrm(np.array(v) * m) for v in ((0.62, 0.62, -0.48), (0.93, 0.25, -0.27), (0.72, -0.62, -0.32))]
        N = nrm(np.array([-0.1, 0.25, -0.96]) * m)
        Qw = [frame_align(rig.dir[f"{side}Wing{i + 1}"], n0, D[i], N) for i in range(3)]
        out[side] = (Qw[0], qm(qinv(Qw[0]), Qw[1]), qm(qinv(Qw[1]), Qw[2]))
    _WING_OPEN[key] = out
    return out


def set_wings(rig, P, open_=0.0, pitch=0.0, hand=None, droop=0.0):
    """Wings from the fold (open_ = 0) to the display spread (1); pitch (deg) swings the whole
    wing back from the root (tips away from the legs); hand: the primaries' own opening (default
    open_); droop (deg): the spread wing sags down at the shoulder."""
    T = wing_targets(rig)
    for side, sx in (("Left", 1), ("Right", -1)):
        L1, L2, L3 = T[side]
        o3 = open_ if hand is None else hand
        P[side + "Wing1"] = qm(qx(pitch), qz(-sx * droop), m3.quat_slerp(I, L1, open_))
        P[side + "Wing2"] = m3.quat_slerp(I, L2, open_)
        P[side + "Wing3"] = m3.quat_slerp(I, L3, o3)


def set_tail(P, sway=0.0, lift=0.0, phase=0.0, wave=0.0, curl=0.0, lag=0.75, side=0.0):
    """Tail: a sideways wave traveling to the tip (sway deg at phase, delayed `lag` rad per bone),
    lift (deg, + raises it back and up, mostly at the root), curl (deg per bone, + up), side (deg,
    a constant sideways bend)."""
    for i, b in enumerate(TAIL_BONES):
        k = i / (len(TAIL_BONES) - 1)
        w = sway * math.sin(phase - i * lag) * (0.6 + 0.6 * k) + side * (0.5 + k)
        P[b] = qm(qy(w), qx(lift * (1.0 - 0.7 * k) + curl * k + wave * math.sin(phase - i * lag) * k))


def harpin_layers(rig, P, clip, t):
    """Wings and tail on top of the body clips."""
    dur = DUR[clip]
    ph = 2 * math.pi * t / dur
    if clip == "Idle":
        set_wings(rig, P, open_=0.03 + 0.025 * math.sin(ph), pitch=2.0 + 1.5 * math.sin(ph))
        set_tail(P, sway=4.0, lift=2.0 + 2.0 * math.sin(ph), phase=ph, lag=0.6)
    elif clip in ("Walk", "WalkBack", "StrafeLeft", "StrafeRight"):
        set_wings(rig, P, open_=0.07 + 0.03 * math.sin(2 * ph), pitch=17.0 + 2.0 * math.sin(2 * ph))
        set_tail(P, sway=7.0, lift=4.0, phase=ph + math.pi, lag=0.7, wave=2.0)
    elif clip == "Run":
        set_wings(rig, P, open_=0.2 + 0.06 * math.sin(2 * ph), pitch=26.0 + 3.0 * math.sin(2 * ph))
        set_tail(P, sway=6.0, lift=16.0, phase=ph + math.pi, lag=0.8, wave=3.0, curl=-2.0)
    elif clip in ("Attack1", "Attack2"):
        k = math.exp(-((t - HIT_TIME[clip]) / 0.12) ** 2)
        set_wings(rig, P, open_=0.08 + 0.32 * k, pitch=8.0 + 10.0 * k)
        sgn = 1 if clip == "Attack1" else -1
        set_tail(P, sway=0.0, lift=6.0 * k, side=-sgn * 9.0 * math.sin(math.pi * min(t / dur, 1.0)))
    elif clip == "Hit":
        k = math.exp(-((t - 0.08) / 0.09) ** 2) if t > 0.08 else smooth(t / 0.08)
        set_wings(rig, P, open_=0.72 * k, pitch=6.0 + 12.0 * k, hand=0.5 * k)
        set_tail(P, sway=0.0, lift=14.0 * k, curl=4.0 * k, side=4.0 * k)
    elif clip == "Death":
        o = smooth((t - 0.15) / 0.55)
        set_wings(rig, P, open_=0.5 * o, pitch=-30.0 * o, hand=0.35 * o, droop=25.0 * o)
        set_tail(P, lift=-10.0 * o, side=10.0 * o, curl=-4.0 * o)


def harpin_death(rig, feet, P, t):
    """Knees buckle, then a fall forward onto the chest, the head turned aside; the wings end
    half open over the body (see harpin_layers)."""
    b = smooth(t / 0.3)
    f = smooth((t - 0.2) / 0.55)
    P.hips = np.array([0.03 * f, -0.18 * b - 0.66 * f, 0.36 * f])
    P["Hips"] = qm(qy(-12 * f), qx(10 * b * (1 - f) + 80 * f), qz(-6 * f))
    P["Spine"] = qx(8 * b * (1 - f) + 2 * f)
    P["Chest"] = qm(qx(6 * b * (1 - f) - 6 * f), qy(8 * f))
    P["Neck"] = qx(-14 * f)
    P["Head"] = qm(qy(55 * f), qx(-18 * f), qz(-12 * f))
    if f < 0.02:
        stand(rig, feet, P, wide=0.05)
    else:
        for side, sgn in (("Left", 1), ("Right", -1)):
            P[side + "UpperLeg"] = qm(qz(sgn * 8 * f), qx(-30 * b * (1 - f) - 70 * f))
            P[side + "LowerLeg"] = qx(40 * b * (1 - f) + 30 * f * (1.2 if side == "Left" else 0.8))
            P[side + "Foot"] = qx(-30 * b * (1 - f) - 40 * f)
            P[side + "Toes"] = qx(30 * f)
    # arms thrown forward as the body lands
    U = nrm(lerp(np.array([0.2, -0.95, 0.1]), np.array([0.65, 0.05, 0.75]), f))
    F = nrm(lerp(np.array([0.1, -0.85, 0.45]), np.array([0.35, -0.1, 0.95]), f))
    rig.arm(P, "Left", U, F)
    ua0, fa0, bl0 = (np.array(SWORD_IDLE[n]) for n in ("ua", "fa", "blade"))
    rig.sword_arm(P, ua=lerp(ua0, np.array([-0.7, 0.0, 0.7]), f), fa=lerp(fa0, np.array([-0.45, -0.15, 0.9]), f),
                  blade=lerp(bl0, np.array([-0.7, -0.3, 0.6]), f))
