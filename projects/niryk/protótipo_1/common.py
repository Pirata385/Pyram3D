"""Shared pieces for the Niryk characters: body specs, the rig, material slots, weights and export.

Conventions (glTF): meters, +Y up, characters face +Z, their left is +X, rest pose = T-pose.
Bone names follow Unity Humanoid; Harpins add LeftWing1..3/RightWing1..3 and Tail1..6.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyram3d").is_dir())))  # the Pyram3D root

import pyram3d as p3  # noqa: E402

EXPORTS = HERE / "exports"
REVIEW = HERE / "review"

# ----------------------------------------------------------------------------- material slots
# Tintable slots: the game multiplies the material color by the player's choice; vertex colors
# carry painted detail as luminance (white = full color). Base colors here are the defaults.
SLOT_DEFAULTS = {
    "Skin": "#e2ad86",
    "Hair": "#5b3a26",
    "Plume": "#ecebe6",
    "Eye": "#3f8a5c",
    "EyeWhite": "#f3efe6",
    "Cloth_A": "#5c8a44",
    "Cloth_B": "#7d6a55",
    "Leather": "#8a5c3a",
    "Metal": "#aab0b8",
    "Talon": "#2c2428",
    "Dark": "#1e1820",
}
_SLOT_CACHE: dict[str, p3.Material] = {}


def slot(name: str) -> p3.Material:
    """The single shared Material for a slot (so every part using it merges into one)."""
    if name not in _SLOT_CACHE:
        metal = name == "Metal"
        _SLOT_CACHE[name] = p3.Material(name, SLOT_DEFAULTS[name], metallic=0.6 if metal else 0.0,
                                        roughness=0.45 if metal else 0.85)
    return _SLOT_CACHE[name]


# ----------------------------------------------------------------------------- body specs

@dataclass
class BodySpec:
    name: str
    race: str  # "human" | "harpin"
    gender: str  # "m" | "f"
    height: float
    shoulder: float  # x of the upper-arm joint
    hip: float  # x of the hip joint
    limb: float = 1.0  # limb thickness multiplier
    bust: float = 0.0
    waist: float = 1.0
    hand: float = 1.0
    extras: dict = field(default_factory=dict)

    @property
    def s(self) -> float:  # vertical scale vs the 1.75 m reference
        return self.height / 1.75

    @property
    def harpin(self) -> bool:
        return self.race == "harpin"


SPECS = {
    # dimorphism must read at ~43 px: broad shoulders/thick limbs vs narrow waist/wide hips;
    # hands slightly oversized (stylized, readable on the sword grip)
    "human_m": BodySpec("human_m", "human", "m", 1.75, shoulder=0.228, hip=0.094, limb=1.2, waist=1.08, hand=1.32),
    "human_f": BodySpec("human_f", "human", "f", 1.68, shoulder=0.164, hip=0.104, limb=0.94, bust=1.0, waist=0.78,
                        hand=1.15),
    "harpin_m": BodySpec("harpin_m", "harpin", "m", 1.72, shoulder=0.202, hip=0.094, limb=1.06, waist=0.98,
                         hand=1.2),
    "harpin_f": BodySpec("harpin_f", "harpin", "f", 1.66, shoulder=0.158, hip=0.1, limb=0.88, bust=0.85,
                         waist=0.72, hand=1.1),
}


def joints(spec: BodySpec) -> dict[str, tuple]:
    """Joint positions (rest pose) for a body spec: {bone: (head, tail)} for the left side + spine."""
    s = spec.s
    sh = spec.shoulder
    J = {
        "Hips": ((0, 0.98 * s, 0), (0, 1.08 * s, 0)),
        "Spine": ((0, 1.08 * s, 0), (0, 1.25 * s, 0)),
        "Chest": ((0, 1.25 * s, 0), (0, 1.45 * s, 0)),
        "Neck": ((0, 1.455 * s, -0.012), (0, 1.555 * s, 0)),
        "Head": ((0, 1.555 * s, 0), (0, 1.75 * s, 0)),
        "LeftShoulder": ((0.035, 1.415 * s, -0.012), (sh, 1.425 * s, -0.02)),
        "LeftUpperArm": ((sh, 1.425 * s, -0.02), (sh + 0.28 * s, 1.425 * s, -0.03)),
        "LeftLowerArm": ((sh + 0.28 * s, 1.425 * s, -0.03), (sh + 0.53 * s, 1.425 * s, -0.012)),
        "LeftHand": ((sh + 0.53 * s, 1.425 * s, -0.012), (sh + 0.53 * s + 0.16 * s * spec.hand, 1.42 * s, 0.0)),
    }
    hx = spec.hip
    if spec.harpin:
        J.update({
            "LeftUpperLeg": ((hx, 0.97 * s, 0.0), (hx, 0.71 * s, 0.11 * s)),
            "LeftLowerLeg": ((hx, 0.71 * s, 0.11 * s), (hx, 0.31 * s, -0.1 * s)),
            "LeftFoot": ((hx, 0.31 * s, -0.1 * s), (hx, 0.045 * s, -0.015 * s)),
            "LeftToes": ((hx, 0.045 * s, -0.015 * s), (hx, 0.02, 0.14 * s)),
        })
    else:
        J.update({
            "LeftUpperLeg": ((hx, 0.95 * s, 0.0), (hx, 0.52 * s, 0.015)),
            "LeftLowerLeg": ((hx, 0.52 * s, 0.015), (hx, 0.085 * s, -0.025)),
            "LeftFoot": ((hx, 0.085 * s, -0.025), (hx, 0.025, 0.1 * s)),
            "LeftToes": ((hx, 0.025, 0.1 * s), (hx, 0.02, 0.16 * s)),
        })
    if spec.harpin:
        J.update({
            # folded on the back (~30 deg off the back plane): wrist a little above the head,
            # primaries reaching down behind the legs to ~0.12 m above the ground
            # wrist (carpal angle) beside and behind the head, clear of it
            "LeftWing1": ((0.075, 1.37 * s, -0.12), (0.27, 1.6 * s, -0.22)),
            "LeftWing2": ((0.27, 1.6 * s, -0.22), (0.37, 1.84 * s, -0.31)),
            "LeftWing3": ((0.37, 1.84 * s, -0.31), (0.2, 0.12, -0.46)),
        })
    return {k: (np.array(a, float), np.array(b, float)) for k, (a, b) in J.items()}


def mirror(p):
    p = np.array(p, float).copy()
    p[0] = -p[0]
    return p


# Harpin tail: an S from the coccyx (under the tunic hem) down behind the legs, the brush held
# ~0.4 m above the ground and curling up (~1.1 m along the curve). y and z scale with the body.
TAIL_POINTS = [(0, 0.93, -0.11), (0, 0.78, -0.22), (0, 0.6, -0.31), (0, 0.46, -0.44), (0.02, 0.4, -0.62),
               (0.08, 0.44, -0.79), (0.17, 0.55, -0.9)]  # lowest ~0.4 m; the brush curls up and ~25 deg aside


def tail_points(spec: BodySpec):
    return [np.array([x, y * spec.s, z * spec.s]) for (x, y, z) in TAIL_POINTS]


PARENT = {
    "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "Shoulder": "Chest", "UpperArm": "Shoulder", "LowerArm": "UpperArm", "Hand": "LowerArm",
    "UpperLeg": "Hips", "LowerLeg": "UpperLeg", "Foot": "LowerLeg", "Toes": "Foot",
    "Wing1": "Chest", "Wing2": "Wing1", "Wing3": "Wing2",
}
SIDED = ["Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes", "Wing1", "Wing2", "Wing3"]


def build_skeleton(spec: BodySpec) -> p3.Skeleton:
    J = joints(spec)
    sk = p3.Skeleton("Rig")
    for name in ("Hips", "Spine", "Chest", "Neck", "Head"):
        a, b = J[name]
        sk.add_bone(name, a, b, parent=PARENT.get(name))
    for part in SIDED:
        key = "Left" + part
        if key not in J:
            continue
        a, b = J[key]
        for side, (pa, pb) in (("Left", (a, b)), ("Right", (mirror(a), mirror(b)))):
            par = PARENT[part]
            par = par if par in ("Hips", "Chest") else side + par
            sk.add_bone(side + part, pa, pb, parent=par)
    if spec.harpin:
        pts = tail_points(spec)
        sk.add_chain([f"Tail{i}" for i in range(1, 7)], pts, parent="Hips")
    # attachment points
    hand_r = mirror(J["LeftHand"][0]) + np.array([-0.075 * spec.s * spec.hand, -0.005, 0.012])
    sk.add_socket("Socket_Weapon_R", "RightHand", hand_r, rotation=(90, 0, 0))
    sk.add_socket("Socket_Back", "Chest", (0, 1.3 * spec.s, -0.17), rotation=(0, 0, 0))
    sk.add_socket("Socket_Head", "Head", (0, 1.75 * spec.s, 0))
    return sk


def side_filter(sk: p3.Skeleton, margin=0.025):
    """bone_filter for bind(): left bones never grab vertices on the right side and vice versa."""
    names = [b.name for b in sk.bones]

    def f(P, i):
        n = names[i]
        if n.startswith("Left"):
            return P[:, 0] > -margin
        if n.startswith("Right"):
            return P[:, 0] < margin
        return np.ones(len(P), bool)

    return f


# ----------------------------------------------------------------------------- export / review

def export_glb(scene: p3.Scene, name: str, sub: str = "characters") -> Path:
    out = EXPORTS / sub / f"{name}.glb"
    out.parent.mkdir(parents=True, exist_ok=True)
    scene.export(out)
    return out


def write_stats(name: str, info: dict, sub: str = "characters"):
    out = REVIEW / sub / f"{name}_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=1))


GAME_VIEW = (0.5417, 0.6428, 0.5417)  # the game's camera direction (yaw 45, pitch 40)
