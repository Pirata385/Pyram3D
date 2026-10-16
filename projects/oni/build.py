"""Oni — playable character for an isometric 3D MMORPG (Hades-inspired, pixel-art look).

Design (from the user's references): deep-red skin with dark tiger stripes, very long
white hair with iridescent cyan/pink/lavender streaks, red-to-white horns, white eyes
with cross-shaped pupils, pointed ears, a silver snowflake hair ornament on her left
side, and a white draped dress (bare striped shoulders, sleeves draped on the upper
arms, sash, long skirt).

Built to accept armor and gear:
  * Unity-Humanoid bone names, T-pose, 1 unit = 1 m, faces +Z, feet on y = 0
  * modular skinned body parts (Body_Head/Torso/Arms/Hands/Legs/Feet) that share one
    skeleton and have seamless normals -> hide the parts that armor covers
  * garment, hair, horns, eyes and ornament are separate objects
  * empty "Socket_*" transforms on bones for weapons, shields, back items, helmets, belts
  * hair bone chains for spring/dynamic-bone physics in the engine
  * one small palette texture (nearest filtering) for the pixel-art look + palette swaps

Run:  python projects/oni/build.py          (writes projects/oni/exports/)
"""
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyram3d as p3  # noqa: E402
from pyram3d import math3d as m3  # noqa: E402
from pyram3d import sdf  # noqa: E402
from pyram3d.noise import fbm3  # noqa: E402
from pyram3d.palette import palettize  # noqa: E402

OUT = Path(__file__).resolve().parent / "exports"
HEIGHT = 1.70
S = HEIGHT / 1.8          # template proportions are authored for 1.8 m
HEAD_SCALE = 1.1          # slightly larger head reads better from the isometric camera
RES = int(sys.argv[1]) if len(sys.argv) > 1 else 100


def P(x, y, z=0.0):
    return np.array([x, y, z], dtype=np.float64) * S


# ---------------------------------------------------------------------------- palette
PAL = {
    "skin": "#9b1f26", "skin_shadow": "#7a1720", "stripe": "#3d0a12", "lips": "#6a1420",
    "brow": "#e9edf4", "nail": "#4a0d14",
    "hair": "#eef1f6", "hair_shade": "#c9cfdc", "hair_cyan": "#bfeaf2", "hair_pink": "#f3cfe0",
    "hair_lilac": "#d6cdf3",
    "horn_base": "#7b0f2a", "horn_red": "#b3162e", "horn_pink": "#e07a8f", "horn_light": "#f2d9e2",
    "horn_tip": "#fbf6f8",
    "cloth": "#f3f1f4", "cloth_shade": "#d9d6e3", "cloth_deep": "#b9b5c9", "sash": "#b3b8d4",
    "sash_dark": "#7c84a8",
}


def lin(c):
    a = m3.as_color(c)
    a[:3] = m3.srgb_to_linear(a[:3])
    return a


# ---------------------------------------------------------------------------- skeleton
def build_skeleton() -> p3.Skeleton:
    sk = p3.Skeleton("Oni_Rig")
    hips = sk.add_bone("Hips", P(0, 0.96), P(0, 1.06))
    spine = sk.add_bone("Spine", P(0, 1.06), P(0, 1.22), hips)
    chest = sk.add_bone("Chest", P(0, 1.22), P(0, 1.42), spine)
    neck = sk.add_bone("Neck", P(0, 1.47), P(0, 1.56), chest)
    head = sk.add_bone("Head", P(0, 1.56), P(0, 1.80), neck)
    for side, sx in (("Left", 1), ("Right", -1)):
        sh = sk.add_bone(f"{side}Shoulder", P(sx * 0.03, 1.44), P(sx * 0.15, 1.44), chest)
        ua = sk.add_bone(f"{side}UpperArm", P(sx * 0.15, 1.44), P(sx * 0.42, 1.44), sh)
        la = sk.add_bone(f"{side}LowerArm", P(sx * 0.42, 1.44), P(sx * 0.66, 1.44), ua)
        sk.add_bone(f"{side}Hand", P(sx * 0.66, 1.44), P(sx * 0.80, 1.44), la)
        ul = sk.add_bone(f"{side}UpperLeg", P(sx * 0.09, 0.93), P(sx * 0.095, 0.5), hips)
        ll = sk.add_bone(f"{side}LowerLeg", P(sx * 0.095, 0.5), P(sx * 0.1, 0.09), ul)
        ft = sk.add_bone(f"{side}Foot", P(sx * 0.1, 0.09), P(sx * 0.1, 0.025, 0.13), ll)
        sk.add_bone(f"{side}Toes", P(sx * 0.1, 0.025, 0.13), P(sx * 0.1, 0.025, 0.2), ft)
    # hair chains (drive them with spring bones / dynamic bones in the engine)
    back = [P(0, 1.70, -0.11), P(0, 1.50, -0.135), P(0, 1.30, -0.135), P(0, 1.10, -0.14), P(0, 0.90, -0.15),
            P(0, 0.74, -0.15)]
    sk.add_chain([f"Hair_Back_{i + 1}" for i in range(5)], back, head)
    for side, sx in (("L", 1), ("R", -1)):
        front = [P(sx * 0.105, 1.66, 0.03), P(sx * 0.13, 1.48, 0.075), P(sx * 0.14, 1.32, 0.095),
                 P(sx * 0.14, 1.16, 0.10)]
        sk.add_chain([f"Hair_Front_{side}_{i + 1}" for i in range(3)], front, head)
    return sk


def add_sockets(sk: p3.Skeleton):
    """Attachment points for gear. Equip by parenting an object to the socket transform."""
    sk.add_socket("Socket_Weapon_R", "RightHand", P(-0.73, 1.44, 0.0))
    sk.add_socket("Socket_Weapon_L", "LeftHand", P(0.73, 1.44, 0.0))
    sk.add_socket("Socket_Shield_L", "LeftLowerArm", P(0.55, 1.44, 0.05))
    sk.add_socket("Socket_Back", "Chest", P(0, 1.32, -0.17))
    sk.add_socket("Socket_Head", "Head", P(0, 1.70 + 0.0, 0.0))
    sk.add_socket("Socket_Hip_L", "Hips", P(0.17, 0.96, 0.0))
    sk.add_socket("Socket_Hip_R", "Hips", P(-0.17, 0.96, 0.0))


# ---------------------------------------------------------------------------- body
def head_center(sk):
    return sk.head("Head") + P(0, 0.105 * HEAD_SCALE, 0.012)


def build_body(sk: p3.Skeleton) -> p3.Mesh:
    H, T = sk.head, sk.tail
    hs = HEAD_SCALE
    pelvis = sdf.ellipsoid(H("Hips") + P(0, 0.0, -0.005), P(0.158, 0.112, 0.102))
    waist = sdf.round_cone(H("Spine") + P(0, -0.02, 0), T("Spine") - P(0, 0.02, 0), 0.112 * S, 0.104 * S)
    ribs = sdf.ellipsoid(P(0, 1.31, -0.005), P(0.145, 0.145, 0.095))
    bust = sdf.ellipsoid(P(0.058, 1.295, 0.055), P(0.06, 0.055, 0.05)).mirror_x()
    torso = sdf.smooth_union(pelvis, waist, ribs, k=0.06 * S).smooth_union(bust, 0.03 * S)
    traps = sdf.round_cone(P(0.0, 1.43, -0.01), P(0.13, 1.43, -0.01), 0.06 * S, 0.045 * S).mirror_x()
    torso = torso.smooth_union(traps, 0.04 * S)
    neck = sdf.capsule(H("Neck") - P(0, 0.04, 0.0), T("Neck") + P(0, 0.01, 0), 0.04 * S)
    hc = head_center(sk)
    k = S * hs
    cranium = sdf.ellipsoid(hc + np.array([0, 0.01, -0.005]) * k, np.array([0.082, 0.098, 0.092]) * k)
    jaw = sdf.round_cone(hc + np.array([0, -0.02, 0.02]) * k, hc + np.array([0, -0.083, 0.046]) * k,
                         0.066 * k, 0.019 * k)
    nose = sdf.round_cone(hc + np.array([0, -0.004, 0.086]) * k, hc + np.array([0, -0.028, 0.097]) * k,
                          0.006 * k, 0.009 * k)
    ear = sdf.round_cone(hc + np.array([0.074, -0.012, -0.012]) * k, hc + np.array([0.135, 0.04, -0.045]) * k,
                         0.021 * k, 0.003 * k).mirror_x()
    sockets = sdf.sphere(hc + np.array([0.036, 0.0, 0.086]) * k, 0.021 * k).mirror_x()
    head = sdf.smooth_union(cranium, jaw, k=0.035 * k).smooth_union(nose, 0.012 * k).smooth_union(ear, 0.01 * k)
    head = head.smooth_subtract(sockets, 0.012 * k)
    parts = [torso, neck, head]
    for side in ("Left", "Right"):
        sx = 1 if side == "Left" else -1
        ua = sdf.round_cone(H(f"{side}UpperArm") - P(sx * 0.03, 0, 0), T(f"{side}UpperArm"), 0.052 * S, 0.039 * S)
        la = sdf.round_cone(H(f"{side}LowerArm"), T(f"{side}LowerArm"), 0.038 * S, 0.028 * S)
        palm_c = H(f"{side}Hand") + P(sx * 0.055, 0, 0.0)
        palm = sdf.ellipsoid(palm_c, P(0.048, 0.019, 0.037))
        fingers = sdf.ellipsoid(palm_c + P(sx * 0.06, -0.003, 0.0), P(0.042, 0.013, 0.033))
        thumb = sdf.round_cone(palm_c + P(-sx * 0.012, -0.004, 0.028), palm_c + P(sx * 0.02, -0.007, 0.055),
                               0.011 * S, 0.009 * S)
        hand = sdf.smooth_union(palm, fingers, thumb, k=0.012 * S)
        thigh = sdf.round_cone(H(f"{side}UpperLeg") + P(0, 0.02, 0), T(f"{side}UpperLeg"), 0.078 * S, 0.05 * S)
        shin = sdf.round_cone(H(f"{side}LowerLeg"), T(f"{side}LowerLeg") + P(0, 0.025, 0), 0.047 * S, 0.03 * S)
        calf = sdf.ellipsoid(H(f"{side}LowerLeg") + P(0, -0.13, -0.015), P(0.045, 0.1, 0.045))
        heel = H(f"{side}Foot") * np.array([1, 0, 1]) + P(0, 0.035, -0.02)
        foot = sdf.round_cone(heel, P(sx * 0.1, 0.028, 0.11), 0.034 * S, 0.03 * S)
        toes = sdf.ellipsoid(P(sx * 0.1, 0.022, 0.135), P(0.04, 0.022, 0.04))
        foot = sdf.smooth_union(foot, toes, k=0.02 * S).intersect(sdf.plane((0, 0.002, 0), (0, -1, 0)))
        arm = sdf.smooth_union(ua, la, k=0.025 * S).smooth_union(hand, 0.012 * S)
        leg = sdf.smooth_union(thigh, calf, shin, k=0.03 * S).smooth_union(foot, 0.025 * S)
        parts += [arm, leg]
    body = parts[0]
    for prt in parts[1:]:
        body = body.smooth_union(prt, 0.03 * S)
    mesh = body.to_mesh(RES, name="Body")
    # extra density where skin shows up close (face, neck, shoulders, upper chest) so stripes stay crisp
    head_faces = mesh.faces_where(lambda c, n: ((c[:, 1] > 1.47 * S) & (np.abs(c[:, 0]) < 0.16 * S))
                                  | ((c[:, 1] > 1.34 * S) & (np.abs(c[:, 0]) < 0.26 * S)))
    mesh.subdivide_region(head_faces, 1)
    body.project(mesh, iterations=3)
    mesh.smooth_angle = 180.0
    return mesh


def paint_body(mesh: p3.Mesh, sk: p3.Skeleton):
    """Skin, tiger stripes, lips and brows as vertex colors (later baked to the palette)."""
    V = mesh.vertices
    n = len(V)
    col = np.tile(lin(PAL["skin"]), (n, 1))
    dom = mesh.joints[np.arange(n), np.argmax(mesh.weights, axis=1)]
    heads = np.array([sk.head(b) for b in sk.bones])
    tails = np.array([sk.tail(b) for b in sk.bones])
    axis = m3.normalize(tails - heads)
    u = np.einsum("ij,ij->i", V - heads[dom], axis[dom])
    names = [b.name for b in sk.bones]
    nz = fbm3(V * 9.0, 3, seed=4)
    nz2 = fbm3(V * 23.0 + 5.0, 2, seed=9)
    hc = head_center(sk)
    # stripe frequency (stripes per meter) along each bone
    freq = np.full(len(names), 0.0)
    for i, nm in enumerate(names):
        if nm in ("Neck",):
            freq[i] = 32
        elif "Shoulder" in nm or nm in ("Chest", "Spine"):
            freq[i] = 26
        elif "UpperArm" in nm or "LowerArm" in nm:
            freq[i] = 34
        elif "UpperLeg" in nm or "LowerLeg" in nm:
            freq[i] = 22
        elif nm == "Head":
            freq[i] = 46
        elif nm == "Hips":
            freq[i] = 18
    f = freq[dom]
    phase = u * f + 0.45 * nz
    # head: stripes slant up and away from the face centre like tiger cheek marks
    on_head = names.index("Head") == dom
    rel = V - hc
    phase = np.where(on_head, (rel[:, 1] + 0.55 * np.abs(rel[:, 0])) * f + 0.35 * nz, phase)
    wave = np.sin(2 * np.pi * phase)
    # stripes taper toward the inside/front of limbs: thinner threshold where the surface faces forward/inward
    vn = mesh.vertex_normals()
    inner = np.clip(-vn[:, 0] * np.sign(V[:, 0] + 1e-9), 0, 1) * 0.35 + np.clip(vn[:, 2], 0, 1) * 0.15
    thr = 0.68 + inner + 0.2 * nz2
    torso = np.isin(dom, [names.index(nm) for nm in ("Chest", "Spine", "LeftShoulder", "RightShoulder", "Neck")])
    thr = np.where(torso, 0.55 + 0.12 * nz2, thr)  # bolder, cleaner bands on neck and shoulders
    face = on_head & (rel[:, 2] > 0.0) & (rel[:, 1] < 0.06 * S)
    thr = np.where(face, 2.0, thr)  # the face gets hand-placed strokes instead (below)
    stripe = (wave > thr) & (f > 0)
    # keep the centre of the face, palms, soles and bust clean
    face_front = on_head & (np.abs(rel[:, 0]) < 0.024 * S * HEAD_SCALE) & (rel[:, 2] > 0.02 * S)
    eyes_zone = on_head & (np.abs(rel[:, 1] - 0.012 * S) < 0.022 * S) & (rel[:, 2] > 0.04 * S) & \
        (np.abs(rel[:, 0]) < 0.065 * S)
    hands = np.isin(dom, [names.index("LeftHand"), names.index("RightHand")])
    feet = np.isin(dom, [names.index(f"{s}{p}") for s in ("Left", "Right") for p in ("Foot", "Toes")])
    bust = (V[:, 2] > 0.06 * S) & (np.abs(V[:, 1] - 1.29 * S) < 0.07 * S)
    stripe &= ~(face_front | eyes_zone | hands | feet | bust)
    col[stripe] = lin(PAL["stripe"])
    # crisp tiger marks on the cheeks and temples (tapered strokes, like the references)
    strokes = [((0.082, -0.012), (0.047, -0.028), 0.0075), ((0.08, -0.045), (0.05, -0.052), 0.0065),
               ((0.07, -0.072), (0.046, -0.07), 0.005), ((0.078, 0.03), (0.058, 0.042), 0.006)]
    kk = S * HEAD_SCALE
    front = on_head & (rel[:, 2] > 0.01 * kk)
    for (x0, y0), (x1, y1), w0 in strokes:
        for sx in (1, -1):
            a = np.array([sx * x0, y0]) * kk
            b = np.array([sx * x1, y1]) * kk
            q = rel[:, :2]
            ab = b - a
            t = np.clip(((q - a) @ ab) / (ab @ ab), 0, 1)
            dist = np.linalg.norm(q - (a + t[:, None] * ab), axis=1)
            col[front & (dist < w0 * kk * (1 - 0.85 * t))] = lin(PAL["stripe"])
    # lips and brows
    lips = on_head & (np.abs(rel[:, 0]) < 0.014 * S * HEAD_SCALE) & \
        (np.abs(rel[:, 1] + 0.058 * S * HEAD_SCALE + 0.12 * np.abs(rel[:, 0])) < 0.0045 * S) & (rel[:, 2] > 0.05 * S)
    col[lips] = lin(PAL["lips"])
    brow = on_head & (np.abs(np.abs(rel[:, 0]) - 0.04 * S * HEAD_SCALE) < 0.022 * S) & \
        (np.abs(rel[:, 1] - 0.045 * S * HEAD_SCALE + 0.004 * np.abs(rel[:, 0]) / S) < 0.005 * S) & \
        (rel[:, 2] > 0.045 * S)
    col[brow] = lin(PAL["brow"])
    # soft painted shadow under the jaw and on the inner limbs (hand-painted look)
    jaw_shadow = (names.index("Neck") == dom) & (V[:, 1] > 1.5 * S)
    col[jaw_shadow & ~stripe] = lin(PAL["skin_shadow"])
    nails = hands & (np.abs(V[:, 0]) > 0.78 * S)
    col[nails] = lin(PAL["nail"])
    mesh.colors = col


# ---------------------------------------------------------------------------- hair
SKIRT_TOP, SKIRT_BOTTOM, SKIRT_R0, SKIRT_R1 = 1.04, 0.2, 0.172, 0.29
SLIT_TOP = 0.72


def skirt_radius(y):
    """Skirt radius (template units) at height y (template units)."""
    t = np.clip((SKIRT_TOP - y) / (SKIRT_TOP - SKIRT_BOTTOM), 0, 1)
    return SKIRT_R0 + (SKIRT_R1 - SKIRT_R0) * t


def build_hair(sk: p3.Skeleton) -> p3.Mesh:
    hs = HEAD_SCALE
    hc = head_center(sk)
    k = S * hs

    def forehead(x, y, lift=0.006):
        """Point on the cranium surface (template units relative to the head centre)."""
        z = 0.092 * np.sqrt(max(1 - (x / 0.082) ** 2 - ((y - 0.01) / 0.098) ** 2, 0.0)) - 0.005
        return hc + np.array([x, y, z + lift]) * k

    cap = sdf.ellipsoid(hc + np.array([0, 0.016, -0.01]) * k, np.array([0.094, 0.112, 0.104]) * k)
    face_hole = sdf.ellipsoid(hc + np.array([0, -0.05, 0.08]) * k, np.array([0.07, 0.085, 0.085]) * k)
    scalp = cap.smooth_subtract(face_hole, 0.01 * k).paint(bone="Head")
    # back curtain: stacked ellipsoids from the crown down to the thighs, flaring slightly
    curtain = None
    levels = np.linspace(1.66, 0.74, 14)
    for i, y in enumerate(levels):
        t = i / (len(levels) - 1)
        rx = (0.105 + 0.07 * min(t * 3, 1) + 0.02 * t) * S
        z = -(0.085 + 0.06 * min(t * 2.5, 1)) * S
        if y < SKIRT_TOP:  # drape over the flared skirt instead of cutting through it
            z = min(z, -(skirt_radius(y) + 0.03) * S)
        e = sdf.ellipsoid(P(0, y, 0) + np.array([0, 0, z]), np.array([rx, 0.1 * S, (0.05 - 0.018 * t) * S]))
        curtain = e if curtain is None else curtain.smooth_union(e, 0.05 * S)
    # pointed, uneven tips at the bottom: cut away everything under a wavy line
    tips_cut = sdf.custom(lambda p: p[:, 1] - (0.80 * S - 0.07 * S * (0.5 + 0.5 * np.cos(p[:, 0] / S * 45))),
                          (-1, -1, -1), (1, 1, 1))
    curtain = curtain.smooth_subtract(tips_cut, 0.01 * S)
    # side locks from the temples, falling in front of the shoulders
    locks = []
    for sx in (1, -1):
        pts = [hc + np.array([sx * 0.078, 0.005, 0.035]) * k, P(sx * 0.115, 1.5, 0.07), P(sx * 0.13, 1.33, 0.095),
               P(sx * 0.135, 1.17, 0.1)]
        locks.append(sdf.chain(pts, [0.026 * S, 0.025 * S, 0.02 * S, 0.006 * S], k=0.02 * S))
    # fringe: thin strands lying on the forehead; one longer lock reaches her right eye
    strands = []
    for x0, x1, y1, r in ((-0.055, -0.068, 0.025, 0.011), (-0.03, -0.042, 0.02, 0.012), (0.0, -0.022, -0.012, 0.011),
                          (0.026, 0.03, 0.03, 0.012), (0.052, 0.066, 0.028, 0.011)):
        a = forehead(x0, 0.088, 0.012)
        b = forehead(x1, y1, 0.016)
        strands.append(sdf.round_cone(a, b, r * k, 0.003 * k))
    fringe = sdf.smooth_union(*strands, k=0.008 * k)
    hair = scalp.smooth_union(curtain, 0.04 * S)
    for lk in locks:
        hair = hair.smooth_union(lk, 0.02 * S)
    # strand grooves (vertical ridges) for the stylized strand look
    grooves = lambda p: 0.004 * S * np.sin(p[:, 0] / S * 120 + p[:, 2] / S * 50)  # noqa: E731
    mesh = hair.displace(1.0, func=grooves).to_mesh(int(RES * 0.72), name="Hair")
    mesh.smooth_angle = 180.0
    # the fringe is thin: mesh it on its own, finer grid (it overlaps the scalp slightly)
    fringe_mesh = fringe.to_mesh(48, name="Fringe")
    fringe_mesh.smooth_angle = 180.0
    return mesh, fringe_mesh


def paint_hair(mesh: p3.Mesh):
    V = mesh.vertices
    n = len(V)
    col = np.tile(lin(PAL["hair"]), (n, 1))
    vn = mesh.vertex_normals()
    band = np.sin(V[:, 0] / S * 38 + fbm3(V * np.array([4.0, 0.6, 4.0]), 2, seed=2) * 3.0)
    col[band > 0.82] = lin(PAL["hair_cyan"])
    col[band < -0.88] = lin(PAL["hair_pink"])
    lil = (np.abs(band) < 0.04) & (V[:, 1] < 1.45 * S)
    col[lil] = lin(PAL["hair_lilac"])
    # painted shade on surfaces facing down / into the body
    shade = (vn[:, 1] < -0.45) | ((vn[:, 2] > 0.3) & (V[:, 2] < -0.05 * S) & (V[:, 1] < 1.5 * S))
    col[shade & (band <= 0.82)] = lin(PAL["hair_shade"])
    mesh.colors = col


# ---------------------------------------------------------------------------- horns, eyes, ornament
def build_horns(sk) -> p3.Mesh:
    hc = head_center(sk)
    horns = []
    for sx in (1, -1):
        base = hc + P(sx * 0.042, 0.075, 0.03) * HEAD_SCALE
        path = m3.catmull_rom([base, base + P(sx * 0.03, 0.08, -0.01), base + P(sx * 0.035, 0.17, -0.035),
                               base + P(sx * 0.02, 0.25, -0.05)], 6)
        h = p3.sweep(path, radius=lambda t: (0.027 * (1 - t) ** 0.85 + 0.002) * S, segments=12, name="Horn")
        horns.append(h)
    m = p3.Mesh.join(horns, name="Horns")
    y = m.vertices[:, 1]
    t = (y - y.min()) / np.ptp(y)
    stops = [(0.0, "horn_base"), (0.18, "horn_red"), (0.55, "horn_pink"), (0.72, "horn_light"), (0.88, "horn_tip")]
    col = np.zeros((len(y), 4))
    for (t0, c0), (t1, c1) in zip(stops[:-1], stops[1:]):
        sel = (t >= t0) & (t <= t1)
        k = ((t[sel] - t0) / (t1 - t0))[:, None]
        col[sel] = lin(PAL[c0]) * (1 - k) + lin(PAL[c1]) * k
    col[t > stops[-1][0]] = lin(PAL["horn_tip"])
    m.colors = col
    m.smooth_angle = 180.0
    return m


def eye_texture() -> p3.Texture:
    """16x16 pixel-art eye: pale iris, dark rim and a cross-shaped pupil."""
    img = np.zeros((16, 16, 4))
    yy, xx = np.mgrid[0:16, 0:16]
    r = np.hypot(xx - 7.5, yy - 7.5)
    img[..., :3] = m3.as_color("#f6f3f8")[:3]
    img[..., 3] = 1
    img[(r > 6.6)] = m3.as_color("#2a0f18")          # dark eyeliner rim
    img[(r > 5.4) & (r <= 6.6)] = m3.as_color("#cfc9da")
    img[yy <= 2] = m3.as_color("#2a0f18")            # heavier upper lid line
    cross = ((np.abs(xx - 7.5) < 1.0) & (np.abs(yy - 7.5) < 3.6)) | ((np.abs(yy - 7.5) < 1.0) & (np.abs(xx - 7.5) < 3.6))
    img[cross] = m3.as_color("#140a10")
    return p3.Texture(img, name="OniEye", filter="nearest", wrap="clamp")


def build_eye(sk, sx) -> tuple[p3.Mesh, np.ndarray]:
    hc = head_center(sk)
    r = 0.019 * S * HEAD_SCALE
    sxy = (1.15, 0.72)
    e = p3.uv_sphere(r, 16, 10, name="Eye_" + ("L" if sx > 0 else "R")).scale((sxy[0], sxy[1], 0.45))
    # front-facing planar UVs so the 16x16 texture covers the visible disc
    v = e.vertices
    e.uvs = np.column_stack([0.5 + v[:, 0] / (2 * r * sxy[0]), 0.5 + v[:, 1] / (2 * r * sxy[1])])
    e.face_uvs = [tuple(f) for f in e.faces]
    e.rotate((0, sx * 22, 0))  # follow the curve of the face
    pos = hc + P(sx * 0.035, 0.0, 0.074) * HEAD_SCALE
    return e, pos


def build_ornament(sk) -> tuple[p3.Mesh, np.ndarray]:
    """Silver snowflake hair pin: six arms with side branches, a small centre jewel."""
    k = S * HEAD_SCALE
    parts = [p3.extrude_shape(p3.shapes.polygon(6, 0.009 * k), 0.006 * k)]
    for a in range(6):
        arm = p3.box((0.034 * k, 0.0045 * k, 0.004 * k)).translate((0.019 * k, 0, 0))
        for d, ang in ((0.02, 45), (0.02, -45), (0.03, 40), (0.03, -40)):
            br = p3.box((0.01 * k, 0.0035 * k, 0.0035 * k)).translate((0.005 * k, 0, 0)).rotate((0, 0, ang))
            arm.merge(br.translate((d * k, 0, 0)))
        parts.append(arm.rotate((0, 0, 60 * a + 90)))
    flake = p3.Mesh.join(parts, name="HairOrnament")
    flake.smooth_angle = 0.0
    flake.set_material(p3.Material("Ornament", "#dfe3ee", metallic=1.0, roughness=0.28))
    flake.rotate((0, 72, 0))  # face outward from her left temple
    hc = head_center(sk)
    return flake, hc + P(0.088, 0.045, -0.02) * HEAD_SCALE


# ---------------------------------------------------------------------------- outfit
def build_outfit(sk: p3.Skeleton) -> p3.Mesh:
    H = sk.head
    top = sdf.round_cone(P(0, 1.02, -0.005), P(0, 1.34, -0.005), 0.124 * S, 0.158 * S)
    chest = sdf.ellipsoid(P(0, 1.31, 0.0), P(0.158, 0.13, 0.115))
    bust = sdf.ellipsoid(P(0.058, 1.295, 0.064), P(0.066, 0.06, 0.056)).mirror_x()
    bodice = sdf.smooth_union(top, chest, k=0.04 * S).smooth_union(bust, 0.03 * S)
    # off-shoulder neckline (shoulders bare) with a V at the front
    bodice = bodice.intersect(sdf.plane(P(0, 1.385, 0), (0, 1, -0.25)))
    vcut = sdf.box(P(0, 1.42, 0.12), P(0.1, 0.1, 0.12)).rotate((0, 0, 45), pivot=P(0, 1.42, 0.12))
    bodice = bodice.smooth_subtract(vcut, 0.01 * S)
    # wrap overlap line (kimono-like) and sash
    sash = sdf.cylinder(P(0, 1.0, 0), P(0, 1.1, 0), 0.128 * S, rounding=0.012 * S, color=PAL["sash"])
    sash_knot = sdf.ellipsoid(P(0, 1.05, -0.13), P(0.06, 0.045, 0.03), color=PAL["sash_dark"])
    sash_tails = sdf.round_cone(P(0.02, 1.03, -0.14), P(0.05, 0.78, -0.17), 0.028 * S, 0.012 * S,
                                color=PAL["sash"]).union(
        sdf.round_cone(P(-0.02, 1.03, -0.14), P(-0.04, 0.82, -0.165), 0.025 * S, 0.01 * S, color=PAL["sash"]))
    # long A-line skirt to just above the ankles
    skirt = sdf.round_cone(P(0, SKIRT_TOP, -0.005), P(0, SKIRT_BOTTOM, -0.01), SKIRT_R0 * S, SKIRT_R1 * S)
    skirt = skirt.intersect(sdf.plane(P(0, 0.17, 0), (0, -1, 0)))
    # front and back slits up to mid-thigh: the skirt becomes two panels that follow the legs
    for zs in (1, -1):
        slit = sdf.box(P(0, (0.1 + SLIT_TOP) / 2, zs * 0.3), P(0.045, SLIT_TOP - 0.1, 0.36), radius=0.012 * S)
        skirt = skirt.smooth_subtract(slit, 0.012 * S)
    # sleeves draped around the upper arms (bare shoulders above)
    sleeves = []
    for sx in (1, -1):
        sl = sdf.round_cone(P(sx * 0.2, 1.425, 0.0), P(sx * 0.33, 1.385, 0.0), 0.054 * S, 0.088 * S)
        sl = sl.intersect(sdf.plane(P(sx * 0.33, 1.44, 0), (sx, -0.45, 0)))
        sleeves.append(sl)
    dress = sdf.smooth_union(bodice, skirt, k=0.04 * S).paint(color=PAL["cloth"])
    for sl in sleeves:
        dress = dress.union(sl.paint(color=PAL["cloth"]))
    # fabric folds on the skirt, deeper toward the hem
    dress = dress.displace(1.0, func=lambda p: np.where(
        p[:, 1] < 1.0 * S, 0.008 * S * np.clip((1.0 * S - p[:, 1]) / (0.8 * S), 0, 1)
        * np.sin(np.arctan2(p[:, 0], p[:, 2]) * 14), 0.0))
    outfit = dress.union(sash).union(sash_knot).union(sash_tails)
    mesh = outfit.to_mesh(int(RES * 0.68), name="Outfit")
    mesh.smooth_angle = 180.0
    return mesh


def paint_outfit(mesh: p3.Mesh):
    V = mesh.vertices
    vn = mesh.vertex_normals()
    col = mesh.colors.copy() if mesh.colors is not None else np.tile(lin(PAL["cloth"]), (len(V), 1))
    cloth = np.linalg.norm(col[:, :3] - lin(PAL["cloth"])[:3], axis=1) < 0.05
    ang = np.arctan2(V[:, 0], V[:, 2])
    fold = np.sin(ang * 14)
    skirt = V[:, 1] < 0.98 * S
    col[cloth & skirt & (fold < -0.55)] = lin(PAL["cloth_shade"])
    col[cloth & (vn[:, 1] < -0.5)] = lin(PAL["cloth_deep"])
    under_bust = cloth & (np.abs(V[:, 1] - 1.22 * S) < 0.02 * S) & (V[:, 2] > 0)
    col[under_bust] = lin(PAL["cloth_shade"])
    # crossing collar line of the wrap
    collar = cloth & (V[:, 2] > 0.05 * S) & (V[:, 1] > 1.1 * S) & \
        (np.abs((V[:, 1] - 1.12 * S) - 1.1 * (V[:, 0] + 0.02 * S)) < 0.008 * S)
    col[collar] = lin(PAL["cloth_deep"])
    mesh.colors = col


# ---------------------------------------------------------------------------- animation
def add_animations(scene, sk):
    ch = p3.character
    walk = ch.walk_cycle(scene.animation("Walk"), sk, duration=1.0, stride=0.85)
    run = ch.run_cycle(scene.animation("Run"), sk, duration=0.66)
    idle = ch.idle(scene.animation("Idle"), sk, duration=3.0)
    # secondary motion for the hair chains (engines can replace this with spring bones)
    for anim, dur, amp, lag in ((walk, 1.0, 3.0, 4.0), (run, 0.66, 5.0, 8.0), (idle, 3.0, 1.5, 0.0)):
        for k in range(9):
            t = dur * k / 8
            w = math.sin(2 * math.pi * k / 8 * (2 if anim is not idle else 1))
            for i in range(1, 6):
                anim.key(sk.bone(f"Hair_Back_{i}"), "rotation", t, (lag / 5 + amp * w * i / 5, 0, amp * 0.3 * w),
                         "SMOOTH")
            for side in ("L", "R"):
                for i in range(1, 4):
                    anim.key(sk.bone(f"Hair_Front_{side}_{i}"), "rotation", t, (lag * 0.5 + amp * 0.6 * w, 0, 0),
                             "SMOOTH")
    return walk, run, idle


# ---------------------------------------------------------------------------- assemble
def build():
    sk = build_skeleton()
    body = build_body(sk)
    body_bones = [b.name for b in sk.bones if not b.name.startswith("Hair")]
    keep = set(body_bones)
    sk.bind(body, smooth=6, falloff=5.0, exclude=[b.name for b in sk.bones if b.name not in keep])
    paint_body(body, sk)

    hair, fringe = build_hair(sk)
    hair_bones = {b.name for b in sk.bones if b.name.startswith("Hair")} | {"Head"}

    def hair_filter(Pp, i):
        nm = sk.bones[i].name
        if nm not in hair_bones:
            return np.zeros(len(Pp), dtype=bool)
        if nm == "Head":
            return Pp[:, 1] > 1.52 * S
        return np.ones(len(Pp), dtype=bool)
    sk.bind(hair, smooth=8, falloff=3.0, bone_filter=hair_filter)
    sk.bind(fringe, rigid=True, bone_filter=lambda Pp, i: np.full(len(Pp), sk.bones[i].name == "Head"))
    hair.merge(fringe)
    paint_hair(hair)

    outfit = build_outfit(sk)

    def outfit_filter(Pp, i):
        nm = sk.bones[i].name
        return np.full(len(Pp), nm in ("Hips", "Spine", "Chest", "LeftShoulder", "RightShoulder", "LeftUpperArm",
                                       "RightUpperArm", "LeftUpperLeg", "RightUpperLeg"))
    sk.bind(outfit, smooth=6, falloff=5.0, bone_filter=outfit_filter)
    skirt_mask = outfit.vertices[:, 1] < 1.0 * S
    sk.bind_skirt(outfit, skirt_mask, bottom=0.35 * S, influence=0.62, side_blend=0.02 * S)
    paint_outfit(outfit)

    horns = build_horns(sk)

    # one shared palette texture for skin, hair, horns and clothes
    meshes = [body, hair, outfit, horns]
    palette_mat, used = palettize(meshes, palette=list(PAL.values()), swatch=4, name="Oni_Palette", roughness=0.75)

    # modular body: seamless split by dominant bone (hide parts under armor in the engine)
    body.freeze_normals()
    groups = {
        "Body_Torso": ["Hips", "Spine", "Chest", "LeftShoulder", "RightShoulder"],
        "Body_Head": ["Neck", "Head"],
        "Body_Arms": ["LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm"],
        "Body_Hands": ["LeftHand", "RightHand"],
        "Body_Legs": ["LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg"],
        "Body_Feet": ["LeftFoot", "RightFoot", "LeftToes", "RightToes"],
    }
    parts = sk.split_mesh(body, groups)

    scene = p3.Scene("Oni")
    scene.background = "#3a2a33"
    scene.add(sk)
    for nm, m in parts.items():
        node = scene.add(p3.skinned_node(m, sk, nm))
        node.extras["outline_group"] = "body"   # preview renders: no ink lines at modular seams
    scene.add(p3.skinned_node(hair, sk, "Hair"))
    scene.add(p3.skinned_node(outfit, sk, "Outfit"))
    # rigid pieces parented to the head bone
    head = sk.bone("Head")
    head.add(horns, name="Horns", position=-sk.head("Head"))
    eye_mat = p3.Material("Oni_Eyes", "#ffffff", 0.0, 0.2, base_color_texture=eye_texture())
    for sx in (1, -1):
        e, pos = build_eye(sk, sx)
        e.set_material(eye_mat)
        head.add(e, name=e.name, position=pos - sk.head("Head"))
    orn, pos = build_ornament(sk)
    head.add(orn, name="HairOrnament", position=pos - sk.head("Head"))
    add_sockets(sk)
    add_animations(scene, sk)
    scene.add_light("directional", position=(3, 6, 4), intensity=3.0, color="#fff1e6", name="Key")
    scene.add_light("directional", position=(-4, 2, -5), intensity=1.6, color="#ff9db8", cast_shadows=False,
                    name="Rim")
    scene.add_light("directional", position=(-3, 3, 4), intensity=0.6, color="#b8c8ff", cast_shadows=False,
                    name="Fill")
    return scene, sk, {"body": body, "hair": hair, "outfit": outfit, "horns": horns, "parts": parts,
                       "palette": used, "palette_material": palette_mat}


def render_previews(scene, sk, out: Path):
    """Turnaround, portrait, in-game isometric pixel preview and animation strips."""
    from pyram3d.image import grid, write_png
    hc = head_center(sk)
    toon = dict(style="toon", exposure=0.8, bloom=0.12, rim=0.4)
    views = [p3.render(scene, None, 360, 520, view=v, **toon) for v in ("front", "three_quarter", "right", "back")]
    write_png(out / "oni_turnaround.png", grid(views, 4))
    p3.render(scene, out / "oni_portrait.png", 600, 600, view=(0.35, 0.05, 1.0), target=hc - P(0, 0.05, 0),
              distance=0.95, **toon)
    game = dict(style="toon", exposure=0.85, bloom=0.1, rim=0.45, yfov=12, background="#5d7a3a", pixel=4,
                target=(0, 0.85, 0), distance=9.0)
    shots = []
    for anim, t, v in (("Idle", 1.0, (0.8, 1.4, 1.0)), ("Walk", 0.0, (0.8, 1.4, 1.0)), ("Walk", 0.5, (-1.0, 1.4, 0.6)),
                       ("Run", 0.2, (1.0, 1.4, -0.4)), ("Idle", 2.0, (0.0, 1.2, -1.0))):
        shots.append(p3.render(scene, None, 384, 384, view=v, time=t, animation=anim, **game))
    write_png(out / "oni_ingame_pixel.png", grid(shots, 5))
    for name in ("Walk", "Run", "Idle"):
        p3.render_frames(scene, out / f"oni_{name}.png", name, frames=6, size=260, view="three_quarter", **toon)


if __name__ == "__main__":
    import time
    t0 = time.time()
    scene, sk, info = build()
    print(f"built in {time.time() - t0:.1f}s")
    total = 0
    for nd in scene.mesh_nodes():
        total += nd.mesh.num_triangles
        print(f"  {nd.name:14s} {nd.mesh.num_triangles:6d} tris  problems={nd.mesh.validate()}")
    print("total triangles:", total, "| bones:", len(sk.bones), "| palette colors:", len(info["palette"]))
    OUT.mkdir(parents=True, exist_ok=True)
    scene.export(OUT / "oni.fbx")
    scene.export(OUT / "oni.glb")
    from pyram3d.image import write_png
    write_png(OUT / "oni_palette.png", info["palette_material"].base_color_texture.image)
    render_previews(scene, sk, OUT)
    p3.write_viewer(OUT / "oni.glb", OUT / "oni_viewer.html", title="Oni Character",
                    files=["oni.fbx", "oni.glb", "oni_palette.png"])
    print(f"done in {time.time() - t0:.1f}s ->", OUT)
