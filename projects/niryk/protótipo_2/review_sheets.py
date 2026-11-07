"""Review sheets for the prototype 2 characters (build_characters.py calls review()).

  <body>_turnaround.png  front, 3/4, side, back (toon, game lights, default look)
  <body>_face.png        face close-ups (front, 3/4, profile) lit from the left, from the front and from
                         behind-left, Head_Skin shaded with the face shadow map as the client draws it
  <body>_face_ref.png    the style references (docs/art/references/rostos) next to the face, in close-up
                         and at game distance
  <body>_hair.png        the four hair styles: 3/4 front, 3/4 back (over the shoulder), side
  <body>_palette.png     three looks from the game's color lists (the tint slots at work)
  <body>_hands.png       sword fist and relaxed hand, at rest, at the hit of Attack1, first person
  <body>_wings.png       Harpins: the wings in motion (Idle, Run, Attack1 hit, Hit flare, Death end) and
                         the legs and feet up close
  <body>_<Clip>.png      filmstrips (Walk, Run, Attack1; --all-clips: all ten)
"""
from __future__ import annotations

import sys

import numpy as np

import anim as A
import body as B
import hair as HR
from common import BODY, FEMALE, HARPIN, REVIEW, TINTS, review_lights, tint_for_review, untint, xf_points

import pyram3d as p3
from pyram3d import math3d as m3

# the client's toon bands on skin (src/client/render/look.ts): warm light, rosy shadow (sRGB)
LOOK_LIGHT = np.array([1.0, 0.945, 0.84])
LOOK_SHADOW_WARM = np.array([0.69, 0.525, 0.565])
FACE_DECALS = ("Head_Eyes", "Head_Iris", "Head_EyeShine", "Head_Lashes", "Head_Brows", "Head_Mouth")


def bake_face(nodes, sk, light_world, skin_hex):
    """Review only: draw Head_Skin as the client's face map shading does for one light direction
    (an unlit material with the lit texture baked from the face threshold map), and the face decals
    flat (unlit). Returns a function that restores the materials."""
    import face_paint as FP
    rig = A.Rig(sk)
    pose = A.sample(rig, A.Feet(rig), "Idle", 0.0)
    q = rig.world_rot(pose, "Head")
    lh = A.rot(m3.quat_conj(q), np.asarray(light_world, float))
    hs = nodes["Head_Skin"].mesh
    skin = hs.materials[0]
    thr = skin.extras["face_threshold"]
    base = skin.base_color_texture.image[..., :3].astype(float) / 255
    k = base.shape[0] // thr.shape[0]
    thr_up = np.repeat(np.repeat(thr, k, 0), k, 1)
    tint = m3.as_color(skin_hex)[:3]
    lit = FP.lit_skin(base * tint, thr_up, lh, LOOK_LIGHT, LOOK_SHADOW_WARM)
    saved = {"Head_Skin": hs.materials}
    hs.materials = [p3.Material("SkinLit", "#ffffff", unlit=True, base_color_texture=p3.Texture(lit, wrap="repeat"))]
    for n in FACE_DECALS:
        if n in nodes:
            mesh = nodes[n].mesh
            saved[n] = mesh.materials
            mesh.materials = [m.copy(unlit=True) for m in mesh.materials]

    def restore():
        for n, mats in saved.items():
            nodes[n].mesh.materials = mats
    return restore


def key_dir(scene):
    """Toward the review key light (world)."""
    for n in scene.walk():
        if n.light is not None and n.name == "Key":
            return m3.normalize(n.world_matrix()[:3, 2])  # look_at: -Z points at the target
    return m3.normalize(np.array([2.5, 3.0, 3.0]))


def set_key(scene, direction):
    """Point the review key light along `direction` (toward the light); returns the old matrix."""
    for n in scene.walk():
        if n.light is not None and n.name == "Key":
            old = (n.translation.copy(), n.rotation.copy())
            n.translation = np.asarray(direction, float) * 5.0
            n.look_at((0, 0, 0))
            return n, old
    return None, None


def hand_world(rig, pose, side):
    """World position of a hand's center in a pose (side: "Left" / "Right")."""
    bone = side + "Hand"
    off = xf_points(B.hand_center(1 if side == "Left" else -1)) - rig.head[bone]
    return rig.world_head(pose, bone) + A.rot(rig.world_rot(pose, bone), off)


def show_hair(nodes, style):
    for n, node in nodes.items():
        if n.startswith("Hair_"):
            node.visible = n == f"Hair_{style}"


def review_hands(scene, sk, out):
    """Close-ups of both hands (toon): the sword hand from the front, the side, below and from the
    eyes (first person), at rest and at the hit of Attack1; the relaxed hand from three sides."""
    from pyram3d.image import grid, write_png
    rig = A.Rig(sk)
    feet = A.Feet(rig)
    idle = A.sample(rig, feet, "Idle", 0.0)
    hit = A.sample(rig, feet, "Attack1", A.HIT_TIME["Attack1"])
    r, l = hand_world(rig, idle, "Right"), hand_world(rig, idle, "Left")
    eye = rig.world_head(hit, "Head") + A.rot(rig.world_rot(hit, "Head"), [0.0, 0.085, 0.17])
    rh = hand_world(rig, hit, "Right")
    shots = [
        dict(view=(-0.5, 0.25, 1.0), target=r, distance=0.42, time=0.0, animation="Idle"),
        dict(view=(-1.0, 0.35, 0.1), target=r, distance=0.42, time=0.0, animation="Idle"),
        dict(view=(-0.3, -0.6, 0.8), target=r, distance=0.42, time=0.0, animation="Idle"),
        dict(view=tuple(eye - rh), target=rh, distance=float(np.linalg.norm(eye - rh)), yfov=60,
             time=A.HIT_TIME["Attack1"], animation="Attack1"),
        dict(view=(0.2, 0.5, 1.0), target=rh, distance=0.5, time=A.HIT_TIME["Attack1"], animation="Attack1"),
        dict(view=(1.0, 0.3, 0.4), target=l, distance=0.42, time=0.0, animation="Idle"),
        dict(view=(0.4, 0.2, 1.0), target=l, distance=0.42, time=0.0, animation="Idle"),
        dict(view=(-0.2, 0.3, -1.0), target=l, distance=0.42, time=0.0, animation="Idle"),
    ]
    imgs = []
    for sh in shots:
        kw = dict(yfov=30)
        kw.update(sh)
        kw["target"] = tuple(float(v) for v in kw["target"])
        imgs.append(p3.render(scene, None, 420, 420, style="toon", **kw))
    write_png(out / f"{BODY}_hands.png", grid(imgs, 4))


# looks from the game's own lists (src/shared/appearance.ts): skin, hair, eyes, tunic (Cloth_A)
PALETTES = [
    ("rabo", {"Skin": "#f3d2b8", "Hair": "#e0c07a", "Eye": "#3f6fa8", "Cloth_A": "#34507a"}),
    ("coque", {"Skin": "#a96f47", "Hair": "#1f1714", "Eye": "#6a4a2a", "Cloth_A": "#7a2f3a"}),
    ("longo", {"Skin": "#d9a57c", "Hair": "#a8432a", "Eye": "#7a4ab0", "Cloth_A": "#5c8a44"}),
]
if HARPIN:  # Harpin hair and plumes: white, black, gray, green, blue, red, yellow only
    PALETTES = [
        ("crista", {"Skin": "#f3d2b8", "Hair": "#c43a32", "Plume": "#c43a32", "Eye": "#3f6fa8", "Cloth_A": "#34507a"}),
        ("plumado", {"Skin": "#a96f47", "Hair": "#3f72c4", "Plume": "#ecebe6", "Eye": "#6a4a2a",
                     "Cloth_A": "#7a2f3a"}),
        ("espetado", {"Skin": "#d9a57c", "Hair": "#e8c040", "Plume": "#4f9a52", "Eye": "#7a4ab0",
                      "Cloth_A": "#6b6870"}),
    ]


def review_palette(scene, nodes, sk, out):
    """The tint slots at work: three looks with different hair styles (3/4 front, full body)."""
    from pyram3d.image import grid, write_png
    from common import TINTS
    mats = {}
    for node in scene.walk():
        if node.mesh is not None:
            for m in node.mesh.materials:
                if m is not None:
                    mats.setdefault(m.name, []).append(m)
    imgs = []
    for style, tints in PALETTES:
        show_hair(nodes, style)
        for k, c in tints.items():
            for m in mats.get(k, []):
                m.base_color = c
            if k == "Hair":
                for m in mats.get("Brow", []):
                    m.base_color = c
        restore = bake_face(nodes, sk, key_dir(scene), tints["Skin"])
        imgs.append(p3.render(scene, None, 420, 640, view=(0.55, 0.12, 1), style="toon", target=(0, 0.95, 0),
                              distance=3.3, yfov=30, time=0.0, animation="Idle"))
        restore()
    for k, c in TINTS.items():
        for m in mats.get(k, []):
            m.base_color = c
    write_png(out / f"{BODY}_palette.png", grid(imgs, 3))


def review_wings(scene, sk, out):
    """Harpins: the wings and tail in motion from behind (3/4 back), and the legs up close."""
    from pyram3d.image import grid, write_png
    shots = [("Idle", 0.0), ("Run", 0.13), ("Attack1", A.HIT_TIME["Attack1"]), ("Hit", 0.1),
             ("Death", A.DUR["Death"])]
    imgs = []
    for clip, t in shots:
        far = clip == "Death"
        imgs.append(p3.render(scene, None, 440, 520, view=(0.6, 0.35, -1.0) if not far else (0.7, 0.9, -0.6),
                              style="toon", target=(0, 0.95 if not far else 0.3, 0), distance=4.4 if not far else 4.0,
                              yfov=30, time=t, animation=clip))
    for v in [(0.5, 0.15, 1.0), (1.0, 0.1, 0.0), (-0.4, 0.3, -1.0)]:
        imgs.append(p3.render(scene, None, 440, 520, view=v, style="toon", target=(0.0, 0.35, 0.0), distance=1.7,
                              yfov=30, time=0.0, animation="Idle"))
    write_png(out / f"{BODY}_wings.png", grid(imgs, 4))


def review_face(scene, sk, nodes, hc, out):
    """Face close-ups lit from the left, the front and behind-left (front, 3/4, profile), and the
    style references next to the face (close-up and game distance)."""
    from pyram3d.image import grid, read_image, write_png
    lights = [(1.0, 0.45, 0.25), (0.0, 0.5, 1.0), (0.85, 0.45, -0.6)]
    views = [(0.0, 0.03, 1), (0.7, 0.12, 1), (1, 0.02, 0.05)]
    node, old = set_key(scene, (0, 0.5, 1))
    imgs = []
    for L in lights:
        L = m3.normalize(np.array(L))
        set_key(scene, L)
        restore = bake_face(nodes, sk, L, TINTS["Skin"])
        for v in views:
            imgs.append(p3.render(scene, None, 440, 440, view=v, style="toon", target=hc, distance=0.55, yfov=30,
                                  time=0.0, animation="Idle", shadows=False))
        restore()
    write_png(out / f"{BODY}_face.png", grid(imgs, 3))
    # references next to the face
    L = m3.normalize(np.array([0.8, 0.5, 0.6]))
    set_key(scene, L)
    restore = bake_face(nodes, sk, L, TINTS["Skin"])
    ours = [p3.render(scene, None, 440, 440, view=v, style="toon", target=hc, distance=0.55, yfov=30, time=0.0,
                      animation="Idle", shadows=False) for v in [(0.0, 0.03, 1), (0.55, 0.1, 1)]]
    far = p3.render(scene, None, 440, 440, view=(0.35, 0.15, 1), style="toon", target=(hc[0], hc[1] - 0.25, hc[2]),
                    distance=2.5, yfov=30, time=0.0, animation="Idle", shadows=False)
    restore()
    if old is not None:
        node.translation, node.rotation = old
    refs = []
    from pathlib import Path
    ref_dir = next((p / "docs/art/references/rostos" for p in Path(__file__).resolve().parents
                    if (p / "docs/art/references/rostos").is_dir()), None)
    if ref_dir is None:
        alt = Path(__file__).resolve().parents[3].parent / "Niryk-Tales-of-Rondoria/docs/art/references/rostos"
        ref_dir = alt if alt.is_dir() else None
    if ref_dir is not None:
        crops = {"01_rosto_anime_loira.jpg": (110, 0, 260, 150), "03_rosto_anime_azul.jpg": (325, 40, 465, 180)}
        for name, (x0, y0, x1, y1) in crops.items():
            if (ref_dir / name).exists():
                im = read_image(ref_dir / name)[y0:y1, x0:x1, :3]
                h, w = im.shape[:2]
                yi = (np.arange(440) * h / 440).astype(int)
                xi = (np.arange(440) * w / 440).astype(int)
                refs.append(im[yi][:, xi])
    write_png(out / f"{BODY}_face_ref.png", grid(refs + ours + [far], len(refs) + 3))


def review(scene, sk, nodes, sword, clips=None):
    from pyram3d.image import grid, write_png
    out = REVIEW / "characters"
    out.mkdir(parents=True, exist_ok=True)
    review_lights(scene)
    sock = next(c for c in sk.bone("RightHand").children if c.name == "Socket_Weapon_R")
    sock.add(p3.Node("SwordPreview", sword))
    for n in ("Head_Skin",) + FACE_DECALS:
        nodes[n].extras["outline_group"] = "face"  # the game draws no ink between the face decals
    tint_for_review(scene)
    restore = bake_face(nodes, sk, key_dir(scene), TINTS["Skin"])
    default = ("plumado" if FEMALE else "espetado") if HARPIN else ("longo" if FEMALE else "curto")
    show_hair(nodes, default)
    p3.render_views(scene, out / f"{BODY}_turnaround.png", views=("front", "three_quarter", "right", "back"),
                    size=560, cols=4, style="toon", time=0.0, animation="Idle")
    rig = A.Rig(sk)
    idle = A.sample(rig, A.Feet(rig), "Idle", 0.0)
    hc = rig.world_head(idle, "Head") + A.rot(rig.world_rot(idle, "Head"), [0, 0.07, 0.02])
    hc = tuple(float(v) for v in hc)
    restore()
    review_face(scene, sk, nodes, hc, out)
    restore = bake_face(nodes, sk, key_dir(scene), TINTS["Skin"])
    # the four hair styles: 3/4 front, 3/4 back (over-the-shoulder view), side
    imgs = []
    for n in ("Wings", "Tail"):  # they hide the head from the side and the back
        if n in nodes:
            nodes[n].visible = False
    for style in HR.STYLES:
        show_hair(nodes, style)
        for v in [(0.6, 0.15, 1), (0.55, 0.35, -1), (-1, 0.1, -0.1)]:
            far = style in ("rabo", "longo", "plumado") and v[2] < 0.5
            imgs.append(p3.render(scene, None, 400, 400, view=v, style="toon",
                                  target=(hc[0], hc[1] - (0.12 if far else 0.0), hc[2]), distance=1.15 if far else 0.85,
                                  yfov=30, time=0.0, animation="Idle", shadows=False))
    write_png(out / f"{BODY}_hair.png", grid(imgs, 6 if len(HR.STYLES) == 4 else 3))
    for n in ("Wings", "Tail"):
        if n in nodes:
            nodes[n].visible = True
    show_hair(nodes, default)
    restore()
    review_palette(scene, nodes, sk, out)
    review_hands(scene, sk, out)
    if HARPIN:
        review_wings(scene, sk, out)
    if clips is None:
        clips = () if "--no-clips" in sys.argv else (
            ("Walk", "StrafeLeft", "Run", "Attack1", "Attack2", "WalkBack", "StrafeRight", "Idle", "Hit", "Death")
            if "--all-clips" in sys.argv else ("Walk", "Run", "Attack1"))
    for clip in clips:
        view = (1.0, 0.25, 0.25) if clip in ("Walk", "Run", "WalkBack") else (0.55, 0.3, 1.0)
        p3.render_frames(scene, out / f"{BODY}_{clip}.png", clip, frames=6, size=320, style="toon", view=view)
    untint(scene)
    for node in nodes.values():
        node.visible = True


