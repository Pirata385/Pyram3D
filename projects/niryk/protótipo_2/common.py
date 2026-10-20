"""Shared pieces for the Niryk prototype 2 characters: proportions, rig, material slots, export.

Conventions (glTF, same as prototype 1): meters, +Y up, characters face +Z, their left is +X,
they stand on y = 0. Bone names follow Unity Humanoid (the list prototype 1 used, so the game's
client logic transfers). Bones have identity rest rotations (local axes = world axes at rest).

Rest (bind) pose: an A-pose, arms 40 deg below the horizontal (palms facing down and back), legs
straight. The arms spend most of the time down, so a lowered rest pose keeps the shoulders from
collapsing under linear-blend skinning; clips are authored as world directions, so they don't
depend on the rest pose.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyram3d").is_dir())))  # the Pyram3D root

import pyram3d as p3  # noqa: E402
from pyram3d import math3d as m3  # noqa: E402

EXPORTS = HERE / "exports"
REVIEW = HERE / "review"

# ----------------------------------------------------------------------------- material slots
# Tintable slots: the game multiplies the material color by the player's choice; vertex colors
# carry painted shading detail as luminance (white = full tint). Base colors are near-neutral
# defaults that read as the starter outfit when the client does not tint.
SLOT_DEFAULTS = {  # GLB base colors: white for the tinted slots (the client multiplies its tint)
    "Skin": "#ffffff",
    "Hair": "#ffffff",
    "Eye": "#ffffff",
    "EyeWhite": "#f4f0e8",  # not tinted
    "Dark": "#241c26",  # not tinted (pupils)
    "Cloth_A": "#ffffff",
    "Cloth_B": "#ffffff",
    "Leather": "#ffffff",
    "Metal": "#ffffff",
}
# Default tints of the starter look (the client's defaults; used for the review renders).
TINTS = {
    "Skin": "#e8b48e",
    "Hair": "#5e3a24",
    "Eye": "#3f8a5c",
    "Cloth_A": "#4d7d3c",
    "Cloth_B": "#6e5a48",
    "Leather": "#7c5032",
    "Metal": "#b9c0ca",
}
_SLOT_CACHE: dict[str, p3.Material] = {}


def slot(name: str) -> p3.Material:
    """The single shared Material for a slot (so every part using it merges into one primitive)."""
    if name not in _SLOT_CACHE:
        metal = name == "Metal"
        _SLOT_CACHE[name] = p3.Material(name, SLOT_DEFAULTS[name], metallic=0.6 if metal else 0.0,
                                        roughness=0.4 if metal else 0.85)
    return _SLOT_CACHE[name]


# ----------------------------------------------------------------------------- proportions
# Human male, heroic stylized: ~6.7 heads (crown 1.785 m, chin 1.52 m), broad shoulders, tapered
# waist, big hands. Hair adds ~4 cm on top (total ~1.82 m).
HEIGHT = 1.785
ARM_DOWN = 40.0  # rest pose: degrees below the horizontal

SHOULDER = np.array([0.185, 1.448, -0.03])  # upper-arm joint (left)
UPPER_ARM = 0.295
FOREARM = 0.262
HAND = 0.19
HIP = np.array([0.098, 0.95, 0.0])
KNEE_Y, ANKLE_Y = 0.515, 0.088


def arm_dir(side: int = 1) -> np.ndarray:
    """Unit direction of the rest-pose arm (side=+1 left, -1 right)."""
    a = math.radians(ARM_DOWN)
    return np.array([side * math.cos(a), -math.sin(a), 0.0])


def arm_rot(side: int = 1) -> np.ndarray:
    """Quaternion taking the canonical T-pose left-arm frame (+X along the arm) to the rest arm."""
    return m3.quat_from_axis_angle((0, 0, 1), -side * ARM_DOWN)


def arm_point(dist: float, side: int = 1, up: float = 0.0, fwd: float = 0.0) -> np.ndarray:
    """A point `dist` along the rest arm from the shoulder joint, offset in the arm's frame."""
    q = arm_rot(side)
    local = np.array([side * dist, up, fwd])
    return mirror_side(SHOULDER, side) + m3.quat_rotate(q, local)


def mirror_side(p, side: int):
    p = np.array(p, float).copy()
    p[0] *= side
    return p


def joints() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Rest joint positions {bone: (head, tail)} for the spine and the LEFT limbs."""
    J = {
        "Hips": ((0, 0.975, 0.0), (0, 1.075, 0.0)),
        "Spine": ((0, 1.075, 0.0), (0, 1.22, 0.0)),
        "Chest": ((0, 1.22, 0.0), (0, 1.45, -0.01)),
        "Neck": ((0, 1.47, -0.018), (0, 1.575, 0.0)),
        "Head": ((0, 1.575, 0.0), (0, HEIGHT, 0.0)),
        "LeftShoulder": ((0.035, 1.43, -0.02), tuple(SHOULDER)),
        "LeftUpperArm": (tuple(SHOULDER), tuple(arm_point(UPPER_ARM))),
        "LeftLowerArm": (tuple(arm_point(UPPER_ARM)), tuple(arm_point(UPPER_ARM + FOREARM))),
        "LeftHand": (tuple(arm_point(UPPER_ARM + FOREARM)), tuple(arm_point(UPPER_ARM + FOREARM + HAND))),
        "LeftUpperLeg": (tuple(HIP), (HIP[0], KNEE_Y, 0.012)),
        "LeftLowerLeg": ((HIP[0], KNEE_Y, 0.012), (HIP[0], ANKLE_Y, -0.022)),
        "LeftFoot": ((HIP[0], ANKLE_Y, -0.022), (HIP[0], 0.028, 0.118)),
        "LeftToes": ((HIP[0], 0.028, 0.118), (HIP[0], 0.022, 0.19)),
    }
    return {k: (np.array(a, float), np.array(b, float)) for k, (a, b) in J.items()}


def mirror(p):
    p = np.array(p, float).copy()
    p[0] = -p[0]
    return p


PARENT = {
    "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "Shoulder": "Chest", "UpperArm": "Shoulder", "LowerArm": "UpperArm", "Hand": "LowerArm",
    "UpperLeg": "Hips", "LowerLeg": "UpperLeg", "Foot": "LowerLeg", "Toes": "Foot",
}
SIDED = ["Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes"]

# Sword grip, in the right hand's rest frame: the fist closes around this axis. Socket_Weapon_R
# sits on it; its local +Y is the blade direction (the sword model: grip at the origin, blade
# along +Y, edges along +-X, flats facing +-Z), so the edges line up with the knuckles.
GRIP_OFFSET = (0.075, -0.034, 0.0)  # along the hand / up / forward, canonical left-hand frame
GRIP_RADIUS = 0.015


def grip_frame(side: int = -1):
    """(position, rotation quaternion) of the weapon grip in rest space for a hand."""
    q = arm_rot(side)
    local = np.array([side * GRIP_OFFSET[0], GRIP_OFFSET[1], GRIP_OFFSET[2]])
    pos = arm_point(UPPER_ARM + FOREARM, side) + m3.quat_rotate(q, local)
    # canonical frame: blade forward (+Z), edges along the hand (X) -> model +Y to +Z: rot X +90
    rot = m3.quat_mul(q, m3.quat_from_axis_angle((1, 0, 0), 90))
    return pos, rot


def build_skeleton() -> p3.Skeleton:
    J = joints()
    sk = p3.Skeleton("Rig")
    for name in ("Hips", "Spine", "Chest", "Neck", "Head"):
        a, b = J[name]
        sk.add_bone(name, a, b, parent=PARENT.get(name))
    for part in SIDED:
        a, b = J["Left" + part]
        for side, (pa, pb) in (("Left", (a, b)), ("Right", (mirror(a), mirror(b)))):
            par = PARENT[part]
            par = par if par in ("Hips", "Chest") else side + par
            sk.add_bone(side + part, pa, pb, parent=par)
    pos, rot = grip_frame(-1)
    sk.add_socket("Socket_Weapon_R", "RightHand", pos, rotation=rot)
    sk.add_socket("Socket_Back", "Chest", (0, 1.32, -0.2), rotation=(0, 0, 0))
    sk.add_socket("Socket_Head", "Head", (0, HEIGHT, 0))
    return sk


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


def review_lights(scene: p3.Scene):
    """Game-like lighting for review renders: warm dramatic key with shadows, cool fill, rim."""
    scene.add_light("directional", position=(2.5, 4.0, 3.0), target=(0, 1, 0), color="#fff1dc", intensity=3.2,
                    name="Key")
    scene.add_light("directional", position=(-3, 1.5, 1.0), target=(0, 1, 0), color="#9fb4ff", intensity=0.9,
                    name="Fill")
    scene.add_light("directional", position=(-1, 2.5, -4), target=(0, 1, 0), color="#ffffff", intensity=1.2,
                    name="Rim")
    for n in scene.walk():
        if n.light is not None:
            n.light.cast_shadows = n.name == "Key"


def tint_for_review(scene: p3.Scene):
    """Review renders: give the slot materials their default tint (the GLB keeps white)."""
    seen = set()
    for n in scene.walk():
        if n.mesh is None:
            continue
        for m in n.mesh.materials:
            if m is not None and m.name in TINTS and id(m) not in seen:
                seen.add(id(m))
                m.base_color = TINTS[m.name]


def untint(scene: p3.Scene):
    for n in scene.walk():
        if n.mesh is None:
            continue
        for m in n.mesh.materials:
            if m is not None and m.name in SLOT_DEFAULTS:
                m.base_color = SLOT_DEFAULTS[m.name]
