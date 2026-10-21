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
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(next(p for p in HERE.parents if (p / "pyram3d").is_dir())))  # the Pyram3D root

import pyram3d as p3  # noqa: E402
from pyram3d import math3d as m3  # noqa: E402

# NIRYK_P2_BODY selects the body being built (one per process: the modeling modules read the
# proportions below at import time); build_characters.py sets it.
BODY = os.environ.get("NIRYK_P2_BODY", "human_m")
FEMALE = BODY.endswith("_f")
HARPIN = BODY.startswith("harpin")

# NIRYK_P2_OUT=<dir> sends exports and review sheets there instead (trial builds that must not
# replace the GLBs the game imports)
_OUT = os.environ.get("NIRYK_P2_OUT")
EXPORTS = Path(_OUT) / "exports" if _OUT else HERE / "exports"
REVIEW = Path(_OUT) / "review" if _OUT else HERE / "review"

# ----------------------------------------------------------------------------- material slots
# Tintable slots: the game multiplies the material color by the player's choice; vertex colors
# carry painted shading detail as luminance (white = full tint). Base colors are near-neutral
# defaults that read as the starter outfit when the client does not tint.
SLOT_DEFAULTS = {  # GLB base colors: white for the tinted slots (the client multiplies its tint)
    "Skin": "#ffffff",
    "Hair": "#ffffff",
    "Eye": "#ffffff",
    "EyeWhite": "#f7f4ef",  # not tinted
    "EyeShine": "#ffffff",  # not tinted, emissive: the eye highlights stay white in any light
    "Dark": "#2a1f26",  # not tinted (pupils, lash lines, mouth line)
    "Cloth_A": "#ffffff",
    "Cloth_B": "#ffffff",
    "Leather": "#ffffff",
    "Metal": "#ffffff",
    "Plume": "#ffffff",  # Harpins: wings, tail, legs and temple tufts (the client's plume color)
    "Talon": "#ffffff",  # Harpins: claws (the client's default dark tint)
}
# Default tints of the starter look (the client's defaults; used for the review renders).
# (the game's defaults: src/shared/appearance.ts defaultAppearance and CharacterView SLOT_DEFAULTS)
TINTS = {
    "Skin": "#e8b994",
    "Hair": "#ecebe6" if HARPIN else "#5b3a26",  # Harpins: white hair, black plumes, ochre tunic
    "Eye": "#3f8a5c",
    "Cloth_A": "#b08a3a" if HARPIN else ("#7a2f3a" if FEMALE else "#7a5236"),
    "Cloth_B": "#5e4c40",
    "Leather": "#7a4a2c",
    "Metal": "#bcc4ce",
    "Plume": "#46414f",  # black (#2b2830) as the client lifts it (liftDark)
    "Talon": "#3a3036",
}
_SLOT_CACHE: dict[str, p3.Material] = {}


def slot(name: str) -> p3.Material:
    """The single shared Material for a slot (so every part using it merges into one primitive)."""
    if name not in _SLOT_CACHE:
        metal = name == "Metal"
        _SLOT_CACHE[name] = p3.Material(name, SLOT_DEFAULTS[name], metallic=0.6 if metal else 0.0,
                                        roughness=0.4 if metal else (1.0 if name == "Hair" else 0.85),
                                        emissive="#ffffff" if name == "EyeShine" else "#000000")
    return _SLOT_CACHE[name]


# ----------------------------------------------------------------------------- proportions
# Human male, heroic stylized: ~6.7 heads (crown 1.785 m, chin 1.52 m), broad shoulders, tapered
# waist, big hands. Hair adds ~4 cm on top (total ~1.82 m).
HEIGHT = 1.785
ARM_DOWN = 40.0  # rest pose: degrees below the horizontal

if FEMALE:
    SHOULDER = np.array([0.163 if HARPIN else 0.166, 1.44, -0.03])  # upper-arm joint (left)
else:
    SHOULDER = np.array([0.176 if HARPIN else 0.183, 1.448, -0.03])
UPPER_ARM = 0.295
FOREARM = 0.262
HAND = 0.19
HIP = np.array([0.094 if HARPIN else 0.098, 0.95, 0.0])
KNEE_Y, ANKLE_Y = 0.515, 0.088

# Harpin anatomy (design space, left side). Digitigrade eagle legs: hip -> knee (forward, under
# the tunic hem) -> hock (the backward "knee", the ankle) -> ball (toe roots, on the ground) ->
# toe tip. Bones: UpperLeg = femur, LowerLeg = shin, Foot = tarsus, Toes = the front toes.
H_KNEE = np.array([HIP[0], 0.7, 0.085])
H_HOCK = np.array([HIP[0], 0.335, -0.1])
H_BALL = np.array([HIP[0], 0.04, 0.045])
H_TOE = np.array([HIP[0], 0.016, 0.17])
# Folded wing (left), laid out on a gently curved surface behind the back: wing coordinates
# (u outward, v up, d off the surface, outward = away from the body) -> design space. The surface
# leans back going down (the tips clear the legs) and wraps forward around the body going out.
# Root in the upper back (shoulder blade) -> elbow -> wrist (beside the head, at the crown's
# height) -> tip of the longest primary (near the ankle): bones Wing1..3, along the arm (d = WING_D).
WING_O = np.array([0.085, 1.375, -0.11])
_WN = np.array([-0.15, 0.13, -1.0]) / np.linalg.norm([-0.15, 0.13, -1.0])
_WV = np.array([0.0, 1.0, 0.0]) - _WN * _WN[1]
_WV /= np.linalg.norm(_WV)
_WU = np.cross(_WV, _WN)
_WU = _WU if _WU[0] > 0 else -_WU
WING_WRAP = 0.35


def _wing_h(u, v):
    u0 = WING_O[0]
    bow = 0.075 * np.sin(np.pi * np.clip((0.15 - v) / 1.45, 0, 1)) ** 1.5  # the lower wing bows back
    return -WING_WRAP * ((u + u0) ** 2 - u0 ** 2) + bow


def wing_point(u, v, d=0.0):
    """Design-space point of wing coordinates (arrays broadcast)."""
    u, v, d = (np.asarray(a, float) for a in (u, v, d))
    e = 1e-4
    S = lambda uu, vv: (WING_O + _WU * np.expand_dims(uu, -1) + _WV * np.expand_dims(vv, -1)  # noqa: E731
                        + _WN * np.expand_dims(_wing_h(uu, vv), -1))
    du = S(u + e, v) - S(u - e, v)
    dv = S(u, v + e) - S(u, v - e)
    N = np.cross(du, dv)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    N = np.where(np.sum(N * _WN, axis=-1, keepdims=True) < 0, -N, N)
    return S(u, v) + N * np.expand_dims(d, -1)


WING_D = 0.03
WING_2D = {"root": (0.0, 0.0), "elbow": (0.07, 0.23), "wrist": (0.25, 0.385), "tip": (0.06, -1.235)}
WING_ROOT, WING_ELBOW, WING_WRIST, WING_TIP = (wing_point(*WING_2D[k], WING_D) for k in ("root", "elbow", "wrist", "tip"))
# Tail (Tail1..6): from the coccyx (under the tunic) back and down between the wings, then out
# behind the legs, the brush curling up and a little to the left.
TAIL_POINTS = np.array([(0, 0.95, -0.075), (0, 0.86, -0.17), (0, 0.745, -0.265), (0, 0.63, -0.37),
                        (0.012, 0.555, -0.5), (0.04, 0.545, -0.64), (0.085, 0.6, -0.78)])


# Final proportions. Every body is modeled in one design space (the male's size); at export the
# legs are lengthened (anime long legs: the stretch is spread between the ankles and the hips, the
# feet and everything above the hips only move) and the whole female is scaled down. Meshes,
# skeleton and sockets all go through xf_points; rig weights are computed in design space.
LEG_EXTRA = 0.05 if FEMALE else 0.04  # meters added to the legs (design space)
BODY_SCALE = 0.93 if FEMALE else 1.0
_WY = np.linspace(-1.0, 3.0, 4001)
_G = np.interp(_WY, [0.1, 0.2, 0.8, 0.92], [0.0, 1.0, 1.0, 0.0])
_S = np.concatenate([[0.0], np.cumsum(0.5 * (_G[1:] + _G[:-1]) * np.diff(_WY))])
_S /= _S[-1]


def warp_y(y):
    return np.asarray(y, float) + LEG_EXTRA * np.interp(y, _WY, _S)


def xf_points(P) -> np.ndarray:
    """Design space -> final rest space (leg stretch, then the body scale)."""
    P = np.array(P, float)
    one = P.ndim == 1
    P = np.atleast_2d(P).copy()
    P[:, 1] = warp_y(P[:, 1])
    P *= BODY_SCALE
    return P[0] if one else P


def xf_normals(P, N) -> np.ndarray:
    """Normals at design-space points P after xf_points (inverse transpose of the y stretch)."""
    N = np.array(N, float)
    dy = 1.0 + LEG_EXTRA * np.interp(np.asarray(P, float)[:, 1], _WY, _G / _S_TOTAL)
    N[:, 1] /= dy
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)


_S_TOTAL = float(np.sum(0.5 * (_G[1:] + _G[:-1]) * np.diff(_WY)))


def xf_mesh(m: p3.Mesh) -> p3.Mesh:
    """Moves a design-space mesh (and its frozen normals) to the final rest space."""
    if m.normals is not None:
        m.normals = xf_normals(m.vertices, m.normals)
    m.vertices = xf_points(m.vertices)
    return m


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
    if HARPIN:
        J.update({
            "LeftUpperLeg": (tuple(HIP), tuple(H_KNEE)),
            "LeftLowerLeg": (tuple(H_KNEE), tuple(H_HOCK)),
            "LeftFoot": (tuple(H_HOCK), tuple(H_BALL)),
            "LeftToes": (tuple(H_BALL), tuple(H_TOE)),
            "LeftWing1": (tuple(WING_ROOT), tuple(WING_ELBOW)),
            "LeftWing2": (tuple(WING_ELBOW), tuple(WING_WRIST)),
            "LeftWing3": (tuple(WING_WRIST), tuple(WING_TIP)),
        })
    return {k: (np.array(a, float), np.array(b, float)) for k, (a, b) in J.items()}


def mirror(p):
    p = np.array(p, float).copy()
    p[0] = -p[0]
    return p


PARENT = {
    "Spine": "Hips", "Chest": "Spine", "Neck": "Chest", "Head": "Neck",
    "Shoulder": "Chest", "UpperArm": "Shoulder", "LowerArm": "UpperArm", "Hand": "LowerArm",
    "UpperLeg": "Hips", "LowerLeg": "UpperLeg", "Foot": "LowerLeg", "Toes": "Foot",
    "Wing1": "Chest", "Wing2": "Wing1", "Wing3": "Wing2",
}
SIDED = ["Shoulder", "UpperArm", "LowerArm", "Hand", "UpperLeg", "LowerLeg", "Foot", "Toes"]
if HARPIN:
    SIDED += ["Wing1", "Wing2", "Wing3"]
TAIL_BONES = [f"Tail{i}" for i in range(1, 7)]

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


def build_skeleton(final: bool = True) -> p3.Skeleton:
    """The rig in final rest space (final=False: in design space, for weights and tests)."""
    J = joints()
    f = xf_points if final else (lambda p: np.asarray(p, float))
    sk = p3.Skeleton("Rig")
    for name in ("Hips", "Spine", "Chest", "Neck", "Head"):
        a, b = J[name]
        sk.add_bone(name, f(a), f(b), parent=PARENT.get(name))
    for part in SIDED:
        a, b = J["Left" + part]
        for side, (pa, pb) in (("Left", (a, b)), ("Right", (mirror(a), mirror(b)))):
            par = PARENT[part]
            par = par if par in ("Hips", "Chest") else side + par
            sk.add_bone(side + part, f(pa), f(pb), parent=par)
    if HARPIN:
        sk.add_chain(TAIL_BONES, [f(p) for p in TAIL_POINTS], parent="Hips")
    pos, rot = grip_frame(-1)
    sk.add_socket("Socket_Weapon_R", "RightHand", f(pos), rotation=rot)
    sk.add_socket("Socket_Back", "Chest", f((0, 1.32, -0.2)), rotation=(0, 0, 0))
    sk.add_socket("Socket_Head", "Head", f((0, HEIGHT, 0)))
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
