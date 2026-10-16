"""Procedural, baked animation clips for the Niryk humanoids (Human and Harpin rigs).

Every clip is sampled at 30 fps from a pose function and keyed LINEAR on every bone, so clips
fully define the pose (clean cross-fades in three.js) and loops are seamless (first == last).

Locomotion uses 2-bone IK in the sagittal plane so planted feet do not slide: the stride per cycle
equals the in-game speed × cycle duration (see SPEEDS). Harpin legs are digitigrade: the toes and
the ball stay planted, the tarsus (Foot bone) rolls, and the IK solves thigh + shin.

Sword attacks are authored as world directions for the right upper arm, forearm and blade, then
converted to local bone rotations through forward kinematics.
"""
from __future__ import annotations

import math

import numpy as np

from common import BodySpec, joints, tail_points

import pyram3d as p3
from pyram3d import math3d as m3

FPS = 30
SPEEDS = {"walk": 1.9, "run": 5.0}  # m/s — must match src/shared constants in the game
DUR = {"Idle": 2.4, "Walk": 1.0, "Run": 0.64, "Attack1": 0.5, "Attack2": 0.55, "Hit": 0.36, "Death": 1.2}
HIT_TIME = {"Attack1": 0.2, "Attack2": 0.24}  # when the blade crosses the front (game hit frame)

I = np.array([0.0, 0.0, 0.0, 1.0])


def qx(a):
    return m3.quat_from_axis_angle((1, 0, 0), a)


def qy(a):
    return m3.quat_from_axis_angle((0, 1, 0), a)


def qz(a):
    return m3.quat_from_axis_angle((0, 0, 1), a)


def qm(*qs):
    out = I
    for q in qs:
        out = m3.quat_mul(out, q)
    return out


def qinv(q):
    return m3.quat_conj(q)


def rot(q, v):
    return m3.quat_rotate(q, np.asarray(v, float))


def arc(a, b):
    """Shortest rotation taking unit vector a to unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = np.cross(a, b)
    d = float(np.dot(a, b))
    if d < -0.9999:
        axis = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        return m3.quat_from_axis_angle(axis / np.linalg.norm(axis), 180)
    q = np.array([c[0], c[1], c[2], 1 + d])
    return q / np.linalg.norm(q)


def frame_quat(x_axis, z_axis):
    """Rotation whose local +X maps to x_axis and +Z (orthogonalized) to z_axis."""
    x = x_axis / np.linalg.norm(x_axis)
    z = z_axis - x * np.dot(z_axis, x)
    z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return m3.matrix3_to_quat(np.stack([x, y, z], axis=1))


def smooth(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def lerp(a, b, t):
    return a + (b - a) * t


def keyframes(t, keys):
    """Piecewise smooth interpolation of {time: value} (values: float or np arrays)."""
    ts = sorted(keys)
    if t <= ts[0]:
        return np.asarray(keys[ts[0]], float)
    for a, b in zip(ts, ts[1:]):
        if t <= b:
            u = smooth((t - a) / (b - a))
            return np.asarray(keys[a], float) * (1 - u) + np.asarray(keys[b], float) * u
    return np.asarray(keys[ts[-1]], float)


# ----------------------------------------------------------------------------- rig geometry

class Rig:
    def __init__(self, spec: BodySpec, sk: p3.Skeleton):
        self.spec = spec
        self.sk = sk
        self.J = joints(spec)
        self.parent = {}
        for b in sk.bones:
            par = None
            n = b.parent
            if n is not None and n.extras.get("is_bone"):
                par = n.name
            self.parent[b.name] = par
        self.head = {b.name: sk.head(b) for b in sk.bones}
        self.tail = {b.name: sk.tail(b) for b in sk.bones}
        self.harpin = spec.harpin
        hip, knee = self.J["LeftUpperLeg"]
        ankle = self.J["LeftLowerLeg"][1]
        ball = self.J["LeftFoot"][1]
        self.L1 = float(np.linalg.norm((knee - hip)[1:]))
        self.L2 = float(np.linalg.norm((ankle - knee)[1:]))
        self.L3 = float(np.linalg.norm((ball - ankle)[1:]))
        self.a_thigh = self._ang(knee - hip)
        self.a_shin = self._ang(ankle - knee)
        self.a_foot = self._ang(ball - ankle)
        self.a_toes = self._ang(self.J["LeftToes"][1] - ball)
        self.hip_y = hip[1]
        self.ankle_rest = ankle
        self.ball_rest = ball

    @staticmethod
    def _ang(d):
        """Sagittal angle of a vector: 0 = straight down, positive = forward (+Z)."""
        return math.atan2(d[2], -d[1])

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
        """World position of a bone's head for a pose (bones have identity rest rotations)."""
        chain = []
        b = bone
        while b is not None:
            chain.append(b)
            b = self.parent[b]
        chain.reverse()
        root = chain[0]
        p = self.head[root] + getattr(pose, "hips", np.zeros(3))
        q = I
        prev = root
        for name in chain[1:]:
            q = qm(q, pose.get(prev, I))
            p = p + rot(q, self.head[name] - self.head[prev])
            prev = name
        return p

    def aim(self, pose, bone, world_dir, twist_ref=None):
        """Local rotation so that `bone` points along world_dir (shortest arc from rest)."""
        par = self.parent[bone]
        pw = self.world_rot(pose, par) if par else I
        rest = self.tail[bone] - self.head[bone]
        local_target = rot(qinv(pw), world_dir)
        return arc(rest, local_target)


# ----------------------------------------------------------------------------- leg IK

def leg_ik(rig: Rig, side, hip_pitch, target_ankle_zy, foot_world_angle, toe_world_angle=None):
    """Returns local X rotations (deg) for UpperLeg, LowerLeg, Foot, Toes in the sagittal plane.

    target_ankle_zy is the ankle target relative to the hip joint (z forward, y up).
    Angles: 0 = down, positive = forward. Knee bends forward (human knee and bird knee).
    """
    z, y = target_ankle_zy
    D = max(1e-4, math.hypot(z, y))
    D = min(D, (rig.L1 + rig.L2) * 0.999)
    alpha = math.atan2(z, -y)
    cos_b = (rig.L1 ** 2 + D ** 2 - rig.L2 ** 2) / (2 * rig.L1 * D)
    beta = math.acos(max(-1.0, min(1.0, cos_b)))
    a_thigh = alpha + beta
    kz, ky = rig.L1 * math.sin(a_thigh), -rig.L1 * math.cos(a_thigh)
    a_shin = math.atan2(z - kz, -(y - ky))
    # local X rotation: positive rotation swings backward -> rot = -(angle change)
    r_thigh = -(a_thigh - rig.a_thigh) - hip_pitch
    r_shin = -(a_shin - rig.a_shin) - (hip_pitch + r_thigh)
    r_foot = -(foot_world_angle - rig.a_foot) - (hip_pitch + r_thigh + r_shin)
    r_toes = 0.0
    if toe_world_angle is not None:
        r_toes = -(toe_world_angle - rig.a_toes) - (hip_pitch + r_thigh + r_shin + r_foot)
    return [math.degrees(v) for v in (r_thigh, r_shin, r_foot, r_toes)]


def locomotion_legs(rig: Rig, phase, mode, pose, hips_off, hip_pitch_deg):
    """Feet trajectories for walk/run; writes leg bones into pose."""
    run = mode == "run"
    dur = DUR["Run" if run else "Walk"]
    S = SPEEDS[mode] * dur  # stride per cycle (both steps)
    stance = 0.4 if run else 0.62
    lift = 0.2 if run else 0.1
    hp = math.radians(hip_pitch_deg)
    for side, off in (("Left", 0.0), ("Right", 0.5)):
        u = (phase + off) % 1.0
        a = S * stance / 2
        if u < stance:  # planted: moves backward at exactly the walk speed
            s = u / stance
            fz = a - 2 * a * s
            fy = 0.0
            roll = smooth((s - 0.62) / 0.38) * (32 if not rig.harpin else 40) if s > 0.62 else 0.0
            strike = (1 - smooth(s / 0.15)) * (12 if not rig.harpin else 4)
        else:  # swing: smooth return, lifted
            s = (u - stance) / (1 - stance)
            fz = -a + 2 * a * smooth(s)
            fy = lift * math.sin(math.pi * s) ** (0.9 if run else 1.2)
            roll = (1 - smooth(s / 0.4)) * (32 if not rig.harpin else 40)
            strike = smooth((s - 0.7) / 0.3) * (12 if not rig.harpin else 4)
        hip = rig.J[f"{side}UpperLeg"][0] if side == "Left" else rig.J["LeftUpperLeg"][0] * np.array([-1, 1, 1])
        hip_y = hip[1] + hips_off[1]
        hip_z = hip[2] + hips_off[2]
        if rig.harpin:
            ball_rest = rig.ball_rest
            ball_z = ball_rest[2] + fz
            ball_y = ball_rest[1] + fy
            tars = rig.a_foot - math.radians(roll * 1.0) + math.radians(strike)
            if u >= stance:  # folded foot while swinging
                tars += math.radians(12 * math.sin(math.pi * s))
            ank_z = ball_z - rig.L3 * math.sin(tars)
            ank_y = ball_y + rig.L3 * math.cos(tars)
            toe_ang = rig.a_toes  # flat on the ground while planted
            if u >= stance:
                toe_ang = rig.a_toes - math.radians(35 * math.sin(math.pi * s))  # curled in the air
            r = leg_ik(rig, side, hp, (ank_z - hip_z, ank_y - hip_y), tars, toe_world_angle=toe_ang)
        else:
            ank_rest = rig.ankle_rest
            ank_z = ank_rest[2] + fz
            ank_y = ank_rest[1] + fy + 0.07 * math.sin(math.radians(roll))
            foot_ang = rig.a_foot - math.radians(roll) + math.radians(strike)
            r = leg_ik(rig, side, hp, (ank_z - hip_z, ank_y - hip_y), foot_ang, toe_world_angle=rig.a_toes)
        names = ["UpperLeg", "LowerLeg", "Foot", "Toes"]
        for n, v in zip(names, r):
            pose[side + n] = qx(v)


# ----------------------------------------------------------------------------- arms

def arm_pose(pose, side, down=74.0, swing=0.0, elbow=10.0, twist=0.0, out=0.0):
    """Arm hanging down (deg from T-pose), swinging forward (+) / back (-), elbow bent forward."""
    sgn = 1 if side == "Left" else -1
    pose[side + "UpperArm"] = qm(qx(-swing), qz(-sgn * (down - out)), qx(sgn * twist))
    pose[side + "LowerArm"] = qy(-sgn * elbow)
    pose[side + "Hand"] = qz(-sgn * 4)


# ----------------------------------------------------------------------------- clips

class PoseSet(dict):
    def __init__(self):
        super().__init__()
        self.hips = np.zeros(3)


def wings_tail(rig: Rig, pose, t, clip, phase, intensity=1.0):
    """Harpin layers: wing breathing / opening and a tail wave."""
    if not rig.harpin:
        return
    if clip == "Run":
        o = 0.55 + 0.25 * math.sin(phase * 2 * math.pi * 2)  # half open, pumping
    elif clip == "Walk":
        o = 0.08 + 0.05 * math.sin(phase * 2 * math.pi * 2)
    elif clip in ("Attack1", "Attack2"):
        k = HIT_TIME[clip] / DUR[clip]
        o = 0.15 + 0.85 * math.exp(-((phase - k) / 0.18) ** 2)  # flare at the strike
    elif clip == "Hit":
        o = 0.35 * math.sin(math.pi * phase)
    elif clip == "Death":  # flare as the knees give, then fold half over the body
        o = 0.45 * smooth(phase / 0.3) - 0.4 * smooth((phase - 0.35) / 0.4)
    else:  # Idle breathing (~3-5 deg at the shoulder)
        o = 0.07 + 0.05 * math.sin(phase * 2 * math.pi)
    for side, sgn in (("Left", 1), ("Right", -1)):
        # open: raise the arm outward and back, unfold the hand outward
        pose[side + "Wing1"] = qm(qz(-sgn * 38 * o), qy(sgn * 22 * o))
        pose[side + "Wing2"] = qz(-sgn * 30 * o)
        pose[side + "Wing3"] = qm(qz(sgn * 105 * o), qx(-10 * o))
    amp = {"Run": 11, "Walk": 10, "Idle": 4, "Death": 2}.get(clip, 5)  # +-10 deg sway, delayed down the tail
    for i in range(1, 7):
        w = math.sin(phase * 2 * math.pi * (2 if clip in ("Walk", "Run") else 1) - i * 0.7)
        lift = {"Run": -6}.get(clip, 0)
        pose[f"Tail{i}"] = qm(qy(amp * w), qx(lift + (2 * math.sin(phase * 2 * math.pi + i) if clip == "Idle" else 0)))


SWORD_IDLE = dict(ua=(-0.2, -0.96, -0.12), fa=(-0.05, -0.88, 0.45), blade=(0.08, -0.22, 1.0))


def right_arm_world(rig: Rig, pose, ua, fa, blade):
    """Right arm from world directions (character space: +Z front, +X character's left)."""
    ua, fa, blade = (np.asarray(v, float) for v in (ua, fa, blade))
    pose["RightUpperArm"] = rig.aim(pose, "RightUpperArm", ua)
    pose["RightLowerArm"] = rig.aim(pose, "RightLowerArm", fa)
    # hand: the blade (+Z of the hand) is exact; the wrist bends so the fingers stay
    # perpendicular to it (a fist around the grip), as close to the forearm as possible
    pw = rig.world_rot(pose, "RightLowerArm")
    b = blade / np.linalg.norm(blade)
    fingers = fa - b * np.dot(fa, b)
    if np.linalg.norm(fingers) < 1e-3:
        fingers = np.cross(b, [0, 1, 0])
    fingers /= np.linalg.norm(fingers)
    R = frame_quat(-fingers, b)  # hand +X = -fingers for the right hand
    pose["RightHand"] = qm(qinv(pw), R)


def sample_clip(rig: Rig, clip: str, t: float) -> PoseSet:
    dur = DUR[clip]
    phase = t / dur
    P = PoseSet()
    two_pi = 2 * math.pi
    if clip in ("Walk", "Run"):
        run = clip == "Run"
        bob = (0.035 if run else 0.022)
        P.hips = np.array([0.0, -bob * 0.5 + bob * math.cos(phase * two_pi * 2) * (-1 if run else 1) * 0.5 - (0.06 if run else 0.012), 0.0])
        lean = 14 if run else 4
        yaw = (9 if run else 6) * math.sin(phase * two_pi)
        P["Hips"] = qm(qy(yaw), qx(lean * 0.5), qz(2.5 * math.sin(phase * two_pi)))
        P["Spine"] = qm(qy(-yaw * 0.5), qx(lean * 0.3))
        P["Chest"] = qm(qy(-yaw * 0.9), qx(lean * 0.2))
        P["Neck"] = qx(-lean * 0.5)
        P["Head"] = qm(qx(-lean * 0.4), qy(yaw * 0.3))
        locomotion_legs(rig, phase, "run" if run else "walk", P, P.hips, lean * 0.5)
        sw = (48 if run else 22) * math.sin(phase * two_pi)
        arm_pose(P, "Left", down=70 if run else 76, swing=sw, elbow=78 if run else 18)
        # right arm carries the sword relaxed: elbow bent ~25 deg, blade 30-45 deg below the
        # horizontal pointing back, the whole arm swinging opposite to the left one
        sw_r = math.radians(25 if run else 15) * math.sin(phase * two_pi + math.pi)
        ua = (-0.2, -math.cos(sw_r), math.sin(sw_r) + (0.12 if run else 0.04))
        fa = (-0.16, -0.86, 0.48 + (0.25 if run else 0.0) + 0.6 * math.sin(sw_r))
        right_arm_world(rig, P, ua=ua, fa=fa, blade=(-0.3, -0.62 if not run else -0.5, -0.72))
    elif clip == "Idle":
        br = math.sin(phase * two_pi)
        P.hips = np.array([0.0, -0.008 + 0.004 * br, 0.0])
        P["Hips"] = qy(-6)
        P["Spine"] = qm(qx(1.5 * br), qy(2))
        P["Chest"] = qm(qx(1.5 * br), qy(4))
        P["Head"] = qm(qy(2 + 3 * math.sin(phase * two_pi * 0.5)), qx(-2))
        _stand_legs(rig, P, wide=0.06)
        arm_pose(P, "Left", down=74 + 2 * br, swing=4, elbow=14)
        right_arm_world(rig, P, **SWORD_IDLE)
    elif clip in ("Attack1", "Attack2"):
        _attack(rig, P, clip, t)
    elif clip == "Hit":
        # sharp recoil (peak at ~0.08 s), back to guard by ~0.3 s
        k = math.exp(-((t - 0.08) / 0.09) ** 2) if t > 0.08 else smooth(t / 0.08)
        P.hips = np.array([0.0, -0.04 * k, -0.08 * k])
        P["Hips"] = qx(-8 * k)
        P["Spine"] = qx(-10 * k)
        P["Chest"] = qm(qx(-12 * k), qy(-10 * k))
        P["Neck"] = qx(-10 * k)
        P["Head"] = qm(qx(-15 * k), qz(6 * k))
        _stand_legs(rig, P, wide=0.06, knees=15 * k)
        arm_pose(P, "Left", down=70 - 25 * k, swing=-15 * k, elbow=25 + 45 * k)
        ua0, fa0, bl0 = (np.array(SWORD_IDLE[n]) for n in ("ua", "fa", "blade"))
        right_arm_world(rig, P, ua=lerp(ua0, np.array([-0.45, -0.88, -0.15]), k),
                        fa=lerp(fa0, np.array([-0.35, -0.9, 0.2]), k),
                        blade=lerp(bl0, np.array([-0.3, -0.85, 0.4]), k))  # the sword arm drops
    elif clip == "Death":
        # knees give way first (0-0.3 s), then a heavy fall backward with a twist, spine curled,
        # limbs loose; the sword leaves the hand (sword_drop) and the wings fold over
        b = smooth(t / 0.3)  # buckle
        f = smooth((t - 0.22) / 0.5)  # fall
        P.hips = np.array([0.04 * f, -0.22 * rig.spec.s * b - 0.6 * rig.spec.s * f, -0.3 * f])
        P["Hips"] = qm(qy(20 * f), qx(10 * b * (1 - f) - 88 * f), qz(10 * f))
        P["Spine"] = qx(12 * b * (1 - f) - 2 * f)  # the back lies flat on the ground at the end
        P["Chest"] = qm(qx(10 * b * (1 - f) + 2 * f), qy(-10 * f))
        P["Neck"] = qx(12 * b - 6 * f)
        P["Head"] = qm(qx(15 * b - 5 * f), qy(35 * f), qz(10 * f))
        knees = 60 * b + 10 * f
        _stand_legs(rig, P, wide=0.07 + 0.06 * f, knees=knees, hip_pitch=-85 * f, planted=f < 0.02)
        if f >= 0.02:  # legs fold loosely as the body lands
            for side, sgn in (("Left", 1), ("Right", -1)):
                P[side + "UpperLeg"] = qm(qz(-sgn * (6 + 10 * f) * (1.4 if side == "Left" else 1)), qx(-knees * 0.55 - 30 * f))
                P[side + "LowerLeg"] = qx(knees * (1.1 if side == "Left" else 0.8))
        arm_pose(P, "Left", down=lerp(74, 58, f), swing=lerp(4, 25, b) - 45 * f, elbow=lerp(14, 55, b))
        ua0, fa0, bl0 = (np.array(SWORD_IDLE[n]) for n in ("ua", "fa", "blade"))
        right_arm_world(rig, P, ua=lerp(ua0, np.array([-0.8, -0.55, 0.2]), f),
                        fa=lerp(fa0, np.array([-0.7, -0.3, 0.65]), f),
                        blade=lerp(bl0, np.array([-0.5, -0.8, 0.3]), f))
    wings_tail(rig, P, t, clip, phase)
    return P


def _stand_legs(rig: Rig, P: PoseSet, wide=0.05, knees=0.0, hip_pitch=0.0, planted=True):
    """Relaxed stance: feet planted under the hips with a slight bend (IK with lowered hips)."""
    hp = math.radians(hip_pitch)
    for side, sgn in (("Left", 1), ("Right", -1)):
        if planted:
            if rig.harpin:
                ball = rig.ball_rest
                tars = rig.a_foot
                ank_z = ball[2] - rig.L3 * math.sin(tars) + (0.03 if side == "Left" else -0.03)
                ank_y = ball[1] + rig.L3 * math.cos(tars)
            else:
                ank_z = rig.ankle_rest[2] + (0.05 if side == "Left" else -0.04)
                ank_y = rig.ankle_rest[1]
                tars = rig.a_foot
            hip = rig.J["LeftUpperLeg"][0]
            r = leg_ik(rig, side, hp, (ank_z - (hip[2] + P.hips[2]), ank_y - (hip[1] + P.hips[1])), tars,
                       toe_world_angle=rig.a_toes)
        else:
            r = [-knees * 0.5, knees, -knees * 0.3, 0.0]
        P[side + "UpperLeg"] = qm(qz(-sgn * math.degrees(wide)), qx(r[0]))
        P[side + "LowerLeg"] = qx(r[1])
        P[side + "Foot"] = qm(qx(r[2]), qz(sgn * math.degrees(wide)))
        P[side + "Toes"] = qx(r[3])


def _attack(rig: Rig, P: PoseSet, clip: str, t: float):
    """Sword slashes. Attack1: horizontal slash right -> left. Attack2: rising backhand left -> right."""
    if clip == "Attack1":
        # anticipation (0-0.12: blade behind the right shoulder, torso coiled back, knees bent),
        # fast diagonal cut down to the left (0.12-0.21, ~160 deg arc) with a 0.3 m step and a
        # 50 deg torso turn, follow-through to 0.4, back to guard at 0.5
        chest = {0.0: (0, 4, 0), 0.12: (-35, 8, 0), 0.2: (30, -4, 0), 0.3: (50, -8, 0), 0.4: (40, -2, 0), 0.5: (4, 4, 0)}
        ua = {0.0: (-0.25, -0.95, 0.25), 0.12: (-0.75, 0.6, -0.3), 0.2: (0.05, -0.15, 1.0), 0.3: (0.75, -0.55, 0.4),
              0.4: (0.55, -0.8, 0.3), 0.5: (-0.25, -0.95, 0.25)}
        fa = {0.0: (-0.1, -0.55, 0.85), 0.12: (-0.25, 0.95, -0.35), 0.2: (0.3, -0.3, 0.9), 0.3: (0.85, -0.5, -0.1),
              0.4: (0.6, -0.75, 0.25), 0.5: (-0.1, -0.55, 0.85)}
        blade = {0.0: (-0.05, 0.25, 1.0), 0.12: (-0.35, 0.55, -0.8), 0.2: (0.85, -0.45, 0.35), 0.3: (0.45, -0.75, -0.5),
                 0.4: (0.3, -0.8, 0.4), 0.5: (-0.05, 0.25, 1.0)}
        step = {0.0: 0.0, 0.12: -0.04, 0.2: 0.28, 0.3: 0.3, 0.4: 0.26, 0.5: 0.0}
        crouch = {0.0: 0.0, 0.12: 0.08, 0.2: 0.06, 0.3: 0.08, 0.5: 0.0}
    else:
        # backhand: starts where Attack1 ended (blade low on the left), sweeps horizontally to the
        # right; the free arm stays tucked
        chest = {0.0: (45, -4, 0), 0.12: (55, 2, 0), 0.24: (-30, -2, 0), 0.36: (-50, -4, 0), 0.46: (-35, 0, 0), 0.55: (0, 4, 0)}
        ua = {0.0: (0.6, -0.6, 0.5), 0.12: (0.85, 0.05, -0.2), 0.24: (-0.25, -0.05, 0.97), 0.36: (-0.9, -0.2, 0.35),
              0.46: (-0.6, -0.7, 0.3), 0.55: (-0.25, -0.95, 0.25)}
        fa = {0.0: (0.7, -0.5, 0.4), 0.12: (0.6, 0.2, -0.7), 0.24: (-0.6, 0.0, 0.8), 0.36: (-0.75, -0.1, -0.6),
              0.46: (-0.5, -0.6, 0.6), 0.55: (-0.1, -0.55, 0.85)}
        blade = {0.0: (0.4, -0.6, -0.6), 0.12: (0.55, 0.1, -0.8), 0.24: (-0.95, 0.05, 0.3), 0.36: (-0.4, 0.0, -0.9),
                 0.46: (-0.3, -0.5, 0.7), 0.55: (-0.05, 0.25, 1.0)}
        step = {0.0: 0.2, 0.12: 0.18, 0.24: 0.3, 0.36: 0.3, 0.46: 0.2, 0.55: 0.0}
        crouch = {0.0: 0.06, 0.12: 0.08, 0.24: 0.06, 0.36: 0.08, 0.55: 0.0}
    yaw, pitch, _ = keyframes(t, chest)
    st = float(keyframes(t, step))
    cr = float(keyframes(t, crouch))
    P.hips = np.array([0.0, -0.04 - cr, st])
    P["Hips"] = qy(yaw * 0.4)
    P["Spine"] = qm(qy(yaw * 0.3), qx(pitch * 0.5))
    P["Chest"] = qm(qy(yaw * 0.35), qx(pitch * 0.5))
    P["Head"] = qy(-yaw * 0.6)
    # lunge: left foot forward, right foot back, both planted (IK)
    hp = 0.0
    # the front foot steps with the body (the step is real), the back foot slides a little
    for side, (dz, dx) in (("Left", (0.1 + st, 0.04)), ("Right", (-0.2 + st * 0.35, -0.03))):
        sgn = 1 if side == "Left" else -1
        hip = rig.J["LeftUpperLeg"][0]
        if rig.harpin:
            tars = rig.a_foot
            ank_z = rig.ball_rest[2] + dz - rig.L3 * math.sin(tars)
            ank_y = rig.ball_rest[1] + rig.L3 * math.cos(tars)
        else:
            tars = rig.a_foot
            ank_z = rig.ankle_rest[2] + dz
            ank_y = rig.ankle_rest[1]
        hip_yaw = math.radians(yaw * 0.35)
        r = leg_ik(rig, side, hp, (ank_z - (hip[2] + P.hips[2]), ank_y - (hip[1] + P.hips[1])), tars,
                   toe_world_angle=rig.a_toes)
        P[side + "UpperLeg"] = qm(qy(-math.degrees(hip_yaw)), qz(-sgn * 4), qx(r[0]))
        P[side + "LowerLeg"] = qx(r[1])
        P[side + "Foot"] = qm(qx(r[2]), qz(sgn * 4))
        P[side + "Toes"] = qx(r[3])
    # free arm tucked against the chest (no "airplane" arms)
    arm_pose(P, "Left", down=72, swing=float(30 + abs(yaw) * 0.2), elbow=105, twist=10)
    right_arm_world(rig, P, ua=keyframes(t, ua), fa=keyframes(t, fa), blade=keyframes(t, blade))


def sword_drop(rig: Rig, sock_t, sock_q):
    """Death: Socket_Weapon_R local (translation, rotation) over time. The sword stays in the hand
    until 0.45 s, then falls and ends lying flat on the ground beside the hand."""
    end = sample_clip(rig, "Death", DUR["Death"])
    R = rig.world_rot(end, "RightHand")
    H = rig.world_head(end, "RightHand")
    u = np.array([0.35, 0.0, 0.94])  # blade on the ground, at an angle to the limp arm
    X = np.array([H[0] - 0.05, 0.035, H[2] + 0.28])  # landed a little away from the open hand
    Q = arc(np.array([0.0, 1.0, 0.0]), u)  # model +Y (blade) along u, guard flat
    t_end = rot(qinv(R), X - H)
    q_end = qm(qinv(R), Q)

    def at(t):
        k = smooth((t - 0.45) / 0.35)
        return np.asarray(sock_t) * (1 - k) + t_end * k, m3.quat_slerp(np.asarray(sock_q), q_end, k)
    return at


def bake(scene: p3.Scene, rig: Rig, clips=None):
    clips = clips or list(DUR)
    hips_rest = rig.sk.bone("Hips").extras["rest_translation"]
    sock = next((c for c in rig.sk.bone("RightHand").children if c.name == "Socket_Weapon_R"), None)
    drop = sword_drop(rig, sock.translation, sock.rotation) if sock is not None else None
    for clip in clips:
        anim = scene.animation(clip)
        anim.loop = clip in ("Idle", "Walk", "Run")
        dur = DUR[clip]
        n = int(round(dur * FPS))
        for i in range(n + 1):
            t = i / FPS if i < n else dur
            P = sample_clip(rig, clip, t if (i < n or not anim.loop) else 0.0)
            for b in rig.sk.bones:
                anim.key(b, "rotation", t, P.get(b.name, I), "LINEAR")
            anim.key(rig.sk.bone("Hips"), "position", t, hips_rest + P.hips, "LINEAR")
            if drop is not None:  # every clip keys the socket, so the sword returns to the hand
                st, sq = drop(t) if clip == "Death" else (sock.translation, sock.rotation)
                anim.key(sock, "position", t, st, "LINEAR")
                anim.key(sock, "rotation", t, sq, "LINEAR")
    return scene
