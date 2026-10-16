"""Rigged characters: humanoid and quadruped templates + animation cycles.

    from pyram3d.generators import character as ch
    hero = ch.humanoid(height=1.75, shirt="#3b6fb6")
    hero.add_to(scene)
    ch.walk_cycle(scene.animation("Walk"), hero.skeleton)
    ch.run_cycle(scene.animation("Run"), hero.skeleton)

Bone names follow the Unity Humanoid convention (Hips, Spine, Chest, Neck,
Head, LeftUpperArm, LeftLowerArm, LeftHand, LeftUpperLeg, LeftLowerLeg,
LeftFoot, LeftToes, ...), so Unity/Unreal/Godot retargeting maps automatically.

The rest pose is a T-pose (arms along X), the character faces +Z and its
left side is +X.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import math3d as m3
from .. import sdf
from ..armature import Skeleton
from ..material import Material
from ..mesh import Mesh
from ..primitives import uv_sphere
from ..scene import Node


@dataclass
class Character:
    mesh: Mesh
    skeleton: Skeleton
    extras: list = field(default_factory=list)  # (Mesh, bone_name, local_position) rigid attachments
    name: str = "Character"
    node: Node | None = None

    def add_to(self, scene, position=None) -> Node:
        """Add skeleton + skinned body (+ attachments like eyes) to the scene."""
        if position is not None:
            self.skeleton.root.translation = m3.vec3(position)
        scene.add(self.skeleton)
        self.node = Node(self.name + "_Body", mesh=self.mesh)
        self.node.skin = self.skeleton
        scene.add(self.node)
        for mesh, bone, pos in self.extras:
            b = self.skeleton.bone(bone)
            b.add(mesh, position=m3.vec3(pos) - self.skeleton.head(b))
        return self.node


def _q(x=0.0, y=0.0, z=0.0):
    """Compose rotations explicitly: Z first, then Y, then X (in parent space)."""
    return m3.quat_mul(m3.quat_from_axis_angle((1, 0, 0), x),
                       m3.quat_mul(m3.quat_from_axis_angle((0, 1, 0), y), m3.quat_from_axis_angle((0, 0, 1), z)))


# ---------------------------------------------------------------------------
# humanoid
# ---------------------------------------------------------------------------

def humanoid_skeleton(height: float = 1.8, name: str = "Rig") -> Skeleton:
    s = height / 1.8
    P = lambda x, y, z=0.0: np.array([x, y, z]) * s  # noqa: E731
    sk = Skeleton(name)
    hips = sk.add_bone("Hips", P(0, 0.95), P(0, 1.05))
    spine = sk.add_bone("Spine", P(0, 1.05), P(0, 1.22), hips)
    chest = sk.add_bone("Chest", P(0, 1.22), P(0, 1.42), spine)
    neck = sk.add_bone("Neck", P(0, 1.48), P(0, 1.56), chest)
    sk.add_bone("Head", P(0, 1.56), P(0, 1.80), neck)
    for side, sx in (("Left", 1), ("Right", -1)):
        sh = sk.add_bone(f"{side}Shoulder", P(sx * 0.03, 1.45), P(sx * 0.17, 1.45), chest)
        ua = sk.add_bone(f"{side}UpperArm", P(sx * 0.17, 1.45), P(sx * 0.45, 1.45), sh)
        la = sk.add_bone(f"{side}LowerArm", P(sx * 0.45, 1.45), P(sx * 0.70, 1.45), ua)
        sk.add_bone(f"{side}Hand", P(sx * 0.70, 1.45), P(sx * 0.86, 1.45), la)
        ul = sk.add_bone(f"{side}UpperLeg", P(sx * 0.1, 0.92), P(sx * 0.1, 0.5), hips)
        ll = sk.add_bone(f"{side}LowerLeg", P(sx * 0.1, 0.5), P(sx * 0.1, 0.09), ul)
        ft = sk.add_bone(f"{side}Foot", P(sx * 0.1, 0.09), P(sx * 0.1, 0.03, 0.14), ll)
        sk.add_bone(f"{side}Toes", P(sx * 0.1, 0.03, 0.14), P(sx * 0.1, 0.03, 0.22), ft)
    return sk


def humanoid(height: float = 1.8, build: float = 1.0, skin="#e2b08f", shirt="#3b6fb6", pants="#2f3542",
             shoes="#3a2a20", hair="#3b2416", eyes: bool = True, resolution: int = 120,
             head_scale: float = 1.0, name: str = "Humanoid", material: Material | None = None) -> Character:
    """Stylized rigged humanoid in T-pose built from SDF parts.

    build      : 0.8 slim .. 1.0 average .. 1.3 heavy
    head_scale : >1 for cartoon / chibi proportions
    Colors are vertex colors (single material) -> lightweight, engine-friendly.
    """
    s = height / 1.8
    sk = humanoid_skeleton(height, name + "_Rig")
    H = lambda b: sk.head(b)  # noqa: E731
    T = lambda b: sk.tail(b)  # noqa: E731
    b = build
    parts = []
    # torso
    pelvis = sdf.ellipsoid(H("Hips") + np.array([0, 0.02, 0]) * s, np.array([0.17 * b, 0.12, 0.11 * b]) * s,
                           color=pants)
    belly = sdf.round_cone(H("Spine"), T("Spine"), 0.13 * b * s, 0.145 * b * s, color=shirt)
    chest = sdf.ellipsoid((H("Chest") + T("Chest")) / 2, np.array([0.19 * b, 0.17, 0.12 * b]) * s, color=shirt)
    torso = sdf.smooth_union(pelvis, belly, chest, k=0.06 * s)
    neck = sdf.capsule(H("Neck") - np.array([0, 0.04, 0]) * s, T("Neck"), 0.05 * s, color=skin)
    hs = head_scale
    head_c = H("Head") + np.array([0, 0.11 * hs, 0.01]) * s
    head = sdf.ellipsoid(head_c, np.array([0.095, 0.12, 0.105]) * s * hs, color=skin)
    jaw = sdf.ellipsoid(head_c + np.array([0, -0.05, 0.02]) * s * hs, np.array([0.075, 0.07, 0.08]) * s * hs,
                        color=skin)
    nose = sdf.ellipsoid(head_c + np.array([0, -0.005, 0.105]) * s * hs, np.array([0.016, 0.025, 0.02]) * s * hs,
                         color=skin)
    ears = sdf.ellipsoid(head_c + np.array([0.095, 0.0, 0.0]) * s * hs, np.array([0.015, 0.03, 0.022]) * s * hs,
                         color=skin).mirror_x()
    head_all = sdf.smooth_union(head, jaw, k=0.04 * s * hs).smooth_union(nose, 0.015 * s).smooth_union(ears, 0.01 * s)
    if hair:
        cap = sdf.ellipsoid(head_c + np.array([0, 0.03, -0.012]) * s * hs, np.array([0.104, 0.12, 0.113]) * s * hs,
                            color=hair)
        cut = sdf.plane(head_c + np.array([0, -0.01, 0.0]) * s * hs, (0, -1, 0.9))
        hair_sdf = cap.intersect(cut)
        hair_sdf.paint(color=hair)
        head_all = sdf.union(head_all, hair_sdf)
    parts += [torso, neck, head_all]
    for side in ("Left", "Right"):
        ua = sdf.round_cone(H(f"{side}UpperArm") - np.sign(H(f"{side}UpperArm")[0]) * np.array([0.04, 0, 0]) * s,
                            T(f"{side}UpperArm"), 0.055 * b * s, 0.042 * b * s, color=shirt)
        la = sdf.round_cone(H(f"{side}LowerArm"), T(f"{side}LowerArm"), 0.04 * b * s, 0.03 * b * s, color=skin)
        hand_c = (H(f"{side}Hand") * 0.4 + T(f"{side}Hand") * 0.6)
        hand = sdf.ellipsoid(hand_c, np.array([0.06, 0.022, 0.045]) * s, color=skin)
        thumb = sdf.capsule(hand_c + np.array([-np.sign(hand_c[0]) * 0.03, 0, 0.04]) * s,
                            hand_c + np.array([np.sign(hand_c[0]) * 0.01, -0.005, 0.07]) * s, 0.012 * s, color=skin)
        ul = sdf.round_cone(H(f"{side}UpperLeg") + np.array([0, 0.03, 0]) * s, T(f"{side}UpperLeg"),
                            0.085 * b * s, 0.06 * b * s, color=pants)
        ll = sdf.round_cone(H(f"{side}LowerLeg"), T(f"{side}LowerLeg") + np.array([0, 0.02, 0]) * s,
                            0.058 * b * s, 0.042 * s, color=pants)
        foot = sdf.box(H(f"{side}Foot") * np.array([1, 0, 1]) + np.array([0, 0.045, 0.06]) * s,
                       np.array([0.1, 0.08, 0.27]) * s, radius=0.035 * s, color=shoes)
        arm = sdf.smooth_union(ua, la, k=0.03 * s).smooth_union(sdf.smooth_union(hand, thumb, k=0.01 * s), 0.02 * s)
        leg = sdf.smooth_union(ul, ll, k=0.03 * s).smooth_union(foot, 0.03 * s)
        parts += [arm, leg]
    body = parts[0]
    for p_ in parts[1:]:
        body = body.smooth_union(p_, 0.035 * s)
    mesh = body.to_mesh(resolution, name=name)
    if material is not None:
        mesh.set_material(material)
    else:
        mesh.set_material(Material(name + "_Skin", "#ffffff", 0.0, 0.6))
    sk.bind(mesh, smooth=6, falloff=5.0)
    ch = Character(mesh, sk, name=name)
    if eyes:
        eye_mat = Material("Eye", "#f8f8f8", 0.0, 0.15)
        iris = Material("Iris", "#2b1d14", 0.0, 0.1)
        for sx in (1, -1):
            e = uv_sphere(0.017 * s * hs, 16, 10, name="Eye")
            e.set_material(eye_mat)
            pupil = uv_sphere(0.009 * s * hs, 12, 8, name="Pupil").translate((0, 0, 0.012 * s * hs))
            pupil.set_material(iris)
            e.merge(pupil)
            pos = head_c + np.array([sx * 0.04, 0.02, 0.088]) * s * hs
            ch.extras.append((e, "Head", pos))
    return ch


# ---------------------------------------------------------------------------
# humanoid animation cycles (in-place; loop seamlessly)
# ---------------------------------------------------------------------------

def _arms_down(side: str, swing: float = 0.0, down: float = 72.0, elbow: float = 10.0):
    """Upper-arm and forearm rotations for an arm hanging at the side.

    swing > 0 moves the arm forward. elbow > 0 bends the forearm forward.
    """
    sgn = 1 if side == "Left" else -1
    upper = _q(x=-swing, z=-sgn * down)
    lower = _q(y=-sgn * elbow)
    return upper, lower


def _leg(thigh: float = 0.0, knee: float = 0.0, foot: float = 0.0):
    """thigh > 0 swings the leg forward, knee > 0 bends it, foot > 0 points the toes down."""
    return _q(x=-thigh), _q(x=knee), _q(x=foot)


def _key_humanoid(anim, sk, t, legs, arms, hips_y=0.0, spine=(0, 0, 0), chest=(0, 0, 0), head=(0, 0, 0),
                  hips_rot=(0, 0, 0), interp="SMOOTH"):
    rest_hips = sk.bone("Hips").extras["rest_translation"]
    anim.key(sk.bone("Hips"), "position", t, rest_hips + np.array([0, hips_y, 0]), interp)
    anim.key(sk.bone("Hips"), "rotation", t, _q(*hips_rot), interp)
    anim.key(sk.bone("Spine"), "rotation", t, _q(*spine), interp)
    anim.key(sk.bone("Chest"), "rotation", t, _q(*chest), interp)
    anim.key(sk.bone("Head"), "rotation", t, _q(*head), interp)
    for side in ("Left", "Right"):
        th, kn, ft = _leg(*legs[side])
        anim.key(sk.bone(f"{side}UpperLeg"), "rotation", t, th, interp)
        anim.key(sk.bone(f"{side}LowerLeg"), "rotation", t, kn, interp)
        anim.key(sk.bone(f"{side}Foot"), "rotation", t, ft, interp)
        up, lo = _arms_down(side, *arms[side])
        anim.key(sk.bone(f"{side}UpperArm"), "rotation", t, up, interp)
        anim.key(sk.bone(f"{side}LowerArm"), "rotation", t, lo, interp)


def walk_cycle(anim, sk: Skeleton, duration: float = 1.0, stride: float = 1.0):
    """Natural in-place walk. stride scales the leg/arm swing."""
    h = sk.head("Hips")[1] / 0.95
    a = stride
    phases = [
        # (time, left leg, right leg, left arm, right arm, hips_y, hips yaw)
        (0.00, (25 * a, 3, -5), (-18 * a, 12, 10), (-18 * a, 72, 10), (18 * a, 72, 15), -0.008, -4),
        (0.25, (5 * a, 8, 0), (8 * a, 55, 15), (-3, 72, 10), (3, 72, 10), 0.02, 0),
        (0.50, (-18 * a, 12, 10), (25 * a, 3, -5), (18 * a, 72, 15), (-18 * a, 72, 10), -0.008, 4),
        (0.75, (8 * a, 55, 15), (5 * a, 8, 0), (3, 72, 10), (-3, 72, 10), 0.02, 0),
        (1.00, (25 * a, 3, -5), (-18 * a, 12, 10), (-18 * a, 72, 10), (18 * a, 72, 15), -0.008, -4),
    ]
    for ph, ll, rl, la, ra, hy, yaw in phases:
        _key_humanoid(anim, sk, ph * duration, {"Left": ll, "Right": rl}, {"Left": la, "Right": ra},
                      hips_y=hy * h, hips_rot=(0, yaw, 0), chest=(0, -yaw * 1.5, 0), spine=(3, 0, 0))
    return anim


def run_cycle(anim, sk: Skeleton, duration: float = 0.66, stride: float = 1.0):
    h = sk.head("Hips")[1] / 0.95
    a = stride
    phases = [
        (0.00, (45 * a, 15, -5), (-25 * a, 25, 20), (-40 * a, 70, 85), (40 * a, 70, 60), -0.04, -7),
        (0.25, (10 * a, 30, 5), (30 * a, 110, 30), (-5, 70, 80), (5, 70, 80), 0.05, 0),
        (0.50, (-25 * a, 25, 20), (45 * a, 15, -5), (40 * a, 70, 60), (-40 * a, 70, 85), -0.04, 7),
        (0.75, (30 * a, 110, 30), (10 * a, 30, 5), (5, 70, 80), (-5, 70, 80), 0.05, 0),
        (1.00, (45 * a, 15, -5), (-25 * a, 25, 20), (-40 * a, 70, 85), (40 * a, 70, 60), -0.04, -7),
    ]
    for ph, ll, rl, la, ra, hy, yaw in phases:
        _key_humanoid(anim, sk, ph * duration, {"Left": ll, "Right": rl}, {"Left": la, "Right": ra},
                      hips_y=hy * h, hips_rot=(0, yaw, 0), chest=(0, -yaw * 1.8, 0), spine=(12, 0, 0),
                      head=(-8, 0, 0))
    return anim


def idle(anim, sk: Skeleton, duration: float = 3.0):
    """Breathing idle with subtle weight shift."""
    h = sk.head("Hips")[1] / 0.95
    for ph, br, sway in ((0, 0, 0), (0.5, 1, 1), (1.0, 0, 0)):
        _key_humanoid(anim, sk, ph * duration, {"Left": (2, 4, 0), "Right": (-1, 3, 0)},
                      {"Left": (2 + br * 2, 74 - br * 2, 12), "Right": (2 + br * 2, 74 - br * 2, 12)},
                      hips_y=-0.008 * h * br, hips_rot=(0, 0, sway * 1.5), chest=(-br * 2.5, 0, -sway),
                      head=(br * 2, sway * 3, 0))
    return anim


def jump(anim, sk: Skeleton, duration: float = 1.2, height: float = 0.45):
    """Anticipation, take-off, airborne tuck, landing, recovery (not looping)."""
    anim.loop = False
    h = sk.head("Hips")[1] / 0.95
    keys = [
        (0.00, (0, 3, 0), (0, 3, 0), (2, 72, 10), (2, 72, 10), 0.0, (0, 0, 0)),
        (0.25, (55, 100, -40), (55, 100, -40), (-35, 60, 20), (-35, 60, 20), -0.22 * h, (25, 0, 0)),
        (0.38, (-5, 5, 35), (-5, 5, 35), (150, 40, 10), (150, 40, 10), 0.05 * h, (-5, 0, 0)),
        (0.55, (60, 90, 10), (40, 70, 10), (110, 50, 30), (110, 50, 30), height * h, (5, 0, 0)),
        (0.72, (20, 25, 20), (20, 25, 20), (60, 60, 20), (60, 60, 20), 0.05 * h, (0, 0, 0)),
        (0.82, (55, 95, -35), (55, 95, -35), (-20, 55, 30), (-20, 55, 30), -0.2 * h, (22, 0, 0)),
        (1.00, (0, 3, 0), (0, 3, 0), (2, 72, 10), (2, 72, 10), 0.0, (0, 0, 0)),
    ]
    for ph, ll, rl, la, ra, hy, sp in keys:
        _key_humanoid(anim, sk, ph * duration, {"Left": ll, "Right": rl}, {"Left": la, "Right": ra},
                      hips_y=hy, spine=sp)
    return anim


def wave(anim, sk: Skeleton, duration: float = 2.0, side: str = "Right"):
    """Raise one hand and wave."""
    other = "Left" if side == "Right" else "Right"
    sgn = 1 if side == "Left" else -1
    for ph in np.linspace(0, 1, 9):
        _key_humanoid(anim, sk, ph * duration, {"Left": (0, 3, 0), "Right": (0, 3, 0)},
                      {other: (3, 72, 10), side: (0, 72, 10)}, head=(0, -sgn * 8, 0))
        w = 25 * np.sin(ph * np.pi * 4)
        anim.key(sk.bone(f"{side}UpperArm"), "rotation", ph * duration, _q(x=0, z=sgn * 60), "SMOOTH")
        anim.key(sk.bone(f"{side}LowerArm"), "rotation", ph * duration, _q(z=sgn * (70 + w)), "SMOOTH")
    return anim


def t_pose(anim, sk: Skeleton, time: float = 0.0):
    for b in sk.bones:
        anim.key(b, "rotation", time, m3.quat_identity())
    return anim


# ---------------------------------------------------------------------------
# quadruped
# ---------------------------------------------------------------------------

def quadruped_skeleton(length: float = 1.0, height: float = 0.6, name: str = "QuadRig") -> Skeleton:
    """Four-legged rig facing +Z (dog/cat/horse/deer proportions via length/height)."""
    L, Hh = length, height
    sk = Skeleton(name)
    hips = sk.add_bone("Hips", (0, Hh, -0.32 * L), (0, Hh, -0.15 * L))
    spine = sk.add_bone("Spine", (0, Hh, -0.15 * L), (0, Hh * 1.02, 0.1 * L), hips)
    chest = sk.add_bone("Chest", (0, Hh * 1.02, 0.1 * L), (0, Hh * 1.05, 0.3 * L), spine)
    neck = sk.add_bone("Neck", (0, Hh * 1.05, 0.3 * L), (0, Hh * 1.3, 0.42 * L), chest)
    sk.add_bone("Head", (0, Hh * 1.3, 0.42 * L), (0, Hh * 1.3, 0.62 * L), neck)
    t1 = sk.add_bone("Tail1", (0, Hh * 1.02, -0.36 * L), (0, Hh * 0.98, -0.5 * L), hips)
    t2 = sk.add_bone("Tail2", (0, Hh * 0.98, -0.5 * L), (0, Hh * 0.9, -0.64 * L), t1)
    sk.add_bone("Tail3", (0, Hh * 0.9, -0.64 * L), (0, Hh * 0.8, -0.76 * L), t2)
    w = 0.12 * L
    for front, parent, z in (("Front", chest, 0.24 * L), ("Back", hips, -0.3 * L)):
        for side, sx in (("Left", 1), ("Right", -1)):
            up = sk.add_bone(f"{front}{side}UpperLeg", (sx * w, Hh * 0.92, z), (sx * w, Hh * 0.5, z), parent)
            lo = sk.add_bone(f"{front}{side}LowerLeg", (sx * w, Hh * 0.5, z), (sx * w, Hh * 0.08, z), up)
            sk.add_bone(f"{front}{side}Foot", (sx * w, Hh * 0.08, z), (sx * w, 0.02, z + 0.06 * L), lo)
    return sk


def quadruped(length: float = 1.0, height: float = 0.6, fur="#b07a45", belly="#e8d2b0", nose="#222222",
              ears: str = "pointy", resolution: int = 110, name: str = "Quadruped", eyes: bool = True) -> Character:
    """Stylized rigged four-legged animal (dog/fox/cat-like). Tweak length/height/colors."""
    L, Hh = length, height
    sk = quadruped_skeleton(L, Hh, name + "_Rig")
    H = lambda b: sk.head(b)  # noqa: E731
    T = lambda b: sk.tail(b)  # noqa: E731
    body = sdf.ellipsoid((0, Hh * 1.0, 0.0), (0.16 * L, 0.15 * L, 0.42 * L), color=fur)
    bel = sdf.ellipsoid((0, Hh * 0.9, 0.02 * L), (0.13 * L, 0.11 * L, 0.33 * L), color=belly)
    torso = sdf.smooth_union(body, bel, k=0.05 * L)
    neck = sdf.round_cone(H("Neck") - np.array([0, 0.03, 0.05]) * L, T("Neck"), 0.1 * L, 0.075 * L, color=fur)
    hc = H("Head") + np.array([0, 0.02, 0.06]) * L
    skull = sdf.ellipsoid(hc, (0.09 * L, 0.085 * L, 0.1 * L), color=fur)
    snout = sdf.round_cone(hc + np.array([0, -0.02, 0.04]) * L, hc + np.array([0, -0.035, 0.16]) * L,
                           0.06 * L, 0.035 * L, color=fur)
    nose_s = sdf.sphere(hc + np.array([0, -0.025, 0.19]) * L, 0.022 * L, color=nose)
    head = sdf.smooth_union(skull, snout, k=0.04 * L).smooth_union(nose_s, 0.01 * L)
    if ears == "pointy":
        ear = sdf.round_cone(hc + np.array([0.05, 0.06, -0.01]) * L, hc + np.array([0.075, 0.16, -0.03]) * L,
                             0.03 * L, 0.006 * L, color=fur).mirror_x()
    else:
        ear = sdf.ellipsoid(hc + np.array([0.085, 0.0, -0.02]) * L, (0.02 * L, 0.07 * L, 0.04 * L),
                            color=fur).mirror_x()
    head = head.smooth_union(ear, 0.015 * L)
    tail = sdf.chain([H("Tail1"), T("Tail1"), T("Tail2"), T("Tail3")],
                     [0.035 * L, 0.03 * L, 0.025 * L, 0.012 * L], k=0.01 * L, color=fur)
    shape = sdf.smooth_union(torso, neck, k=0.05 * L).smooth_union(head, 0.04 * L).smooth_union(tail, 0.03 * L)
    for front in ("Front", "Back"):
        for side in ("Left", "Right"):
            up = sdf.round_cone(H(f"{front}{side}UpperLeg") + np.array([0, 0.04, 0]) * L,
                                T(f"{front}{side}UpperLeg"), 0.065 * L, 0.04 * L, color=fur)
            lo = sdf.round_cone(H(f"{front}{side}LowerLeg"), T(f"{front}{side}LowerLeg"), 0.038 * L, 0.03 * L,
                                color=fur)
            paw = sdf.ellipsoid(H(f"{front}{side}Foot") * np.array([1, 0, 1]) + np.array([0, 0.03, 0.025]) * L,
                                (0.04 * L, 0.03 * L, 0.06 * L), color=belly)
            leg = sdf.smooth_union(up, lo, k=0.02 * L).smooth_union(paw, 0.02 * L)
            shape = shape.smooth_union(leg, 0.04 * L)
    mesh = shape.to_mesh(resolution, name=name)
    mesh.set_material(Material(name + "_Fur", "#ffffff", 0.0, 0.8))
    sk.bind(mesh, smooth=6, falloff=5.0)
    ch = Character(mesh, sk, name=name)
    if eyes:
        eye_mat = Material("Eye", "#1a1410", 0.0, 0.08)
        for sx in (1, -1):
            e = uv_sphere(0.018 * L, 12, 8, name="Eye").set_material(eye_mat)
            ch.extras.append((e, "Head", hc + np.array([sx * 0.055, 0.03, 0.07]) * L))
    return ch


def quadruped_walk(anim, sk: Skeleton, duration: float = 1.0, stride: float = 1.0):
    """Diagonal-pair walk/trot cycle for quadruped_skeleton rigs."""
    a = stride
    for ph in np.linspace(0, 1, 9):
        ang = ph * 2 * np.pi
        for front, phase0 in (("Front", 0.0), ("Back", np.pi)):
            for side, off in (("Left", 0.0), ("Right", np.pi)):
                th = 22 * a * np.sin(ang + phase0 + off)
                lift = max(0.0, np.cos(ang + phase0 + off))
                knee = (35 if front == "Back" else -35) * lift * a
                anim.key(sk.bone(f"{front}{side}UpperLeg"), "rotation", ph * duration, _q(x=-th), "SMOOTH")
                anim.key(sk.bone(f"{front}{side}LowerLeg"), "rotation", ph * duration, _q(x=knee), "SMOOTH")
        anim.key(sk.bone("Spine"), "rotation", ph * duration, _q(y=4 * np.sin(ang)), "SMOOTH")
        anim.key(sk.bone("Head"), "rotation", ph * duration, _q(x=3 * np.sin(2 * ang), y=-3 * np.sin(ang)), "SMOOTH")
        anim.key(sk.bone("Tail1"), "rotation", ph * duration, _q(y=15 * np.sin(ang), x=-10), "SMOOTH")
        anim.key(sk.bone("Tail2"), "rotation", ph * duration, _q(y=20 * np.sin(ang - 0.6)), "SMOOTH")
        rest = sk.bone("Hips").extras["rest_translation"]
        anim.key(sk.bone("Hips"), "position", ph * duration,
                 rest + np.array([0, 0.012 * np.cos(2 * ang), 0]), "SMOOTH")
    return anim
