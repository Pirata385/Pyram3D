"""Builds the prototype 2 Human male: smooth body, starter outfit, Hair_curto, rig, clips, review.

    python projects/niryk/protótipo_2/build_human.py [--fast] [--no-review]

Writes exports/characters/human_m.glb, exports/equipment/sword_iron.glb, review/characters/*.png
and review/characters/human_m_stats.json. --fast meshes coarser (quick iteration); exports should
use the default (fine) settings.

GLB contents (node names are what the client looks for):
  Rig            armature (Unity Humanoid bone names) + Socket_Weapon_R / Socket_Back / Socket_Head
  Body           skin below the neck (arms, hands)                          [Skin]
  Head_Skin      head and neck skin (hidden in first person)               [Skin]
  Head_Eyes      eyeballs, irises, pupils (rigid on Head)                   [EyeWhite, Eye, Dark]
  Head_Brows     eyebrows (rigid on Head)                                  [Hair]
  Hair_curto     short hair (rigid on Head)                                [Hair]
  Outfit         tunic, trousers, belt, bracers, boots                     [Cloth_A, Cloth_B, Leather, Metal]
  clips          Idle Walk Run WalkBack StrafeLeft StrafeRight Attack1 Attack2 Hit Death
"""
from __future__ import annotations

import hashlib
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import anim as A
import body as B
import clothes as C
import face as FC
import hair as HR
import rig as RG
from common import (FOREARM, HEIGHT, KNEE_Y, REVIEW, SHOULDER, UPPER_ARM, arm_dir, build_skeleton, export_glb, mirror_side,
                    review_lights, tint_for_review, untint, write_stats)
from meshing import covered_by, mesh_sdf, sizing

import pyram3d as p3

FAST = "--fast" in sys.argv
CELL = 0.0065 if FAST else 0.0052  # body (mostly under the clothes) and garments
FINE = 0.0036 if FAST else 0.0028  # head and hands: seen up close

# Regions meshed finely on their own (their open cuts lie under the tunic and the bracers).
HEAD_BOX = (np.array([-0.15, 1.415, -0.17]), np.array([0.15, 1.86, 0.17]))
HAND_CUT = UPPER_ARM + FOREARM * 0.7  # hands start here along the arm (under the bracer)
BUDGET = {"Body": 2600, "Head": 5200, "Hand": 1900, "Hair_curto": 3000, "Tunic": 3200, "Trousers": 1500,
          "Boots": 1700, "Belt": 700, "Bracers": 600}


def _along(P, side):
    return (P - mirror_side(SHOULDER, side)) @ arm_dir(side)


def in_head_box(P, shrink=0.0):
    lo, hi = HEAD_BOX
    return np.all((P > lo + shrink) & (P < hi - shrink), axis=1)


def in_hand(P, side=None, cut=HAND_CUT):
    out = np.zeros(len(P), bool)
    for s in ((1, -1) if side is None else (side,)):
        out |= (np.sign(P[:, 0]) == s) & (_along(P, s) > cut)
    return out


def head_sizing():
    face = lambda P: (P[:, 2] > 0.02) & (P[:, 1] > 1.53) & (P[:, 1] < 1.74) & (np.abs(P[:, 0]) < 0.075)  # noqa: E731
    ears = lambda P: (np.abs(P[:, 0]) > 0.07) & (P[:, 1] > 1.6) & (P[:, 1] < 1.7)  # noqa: E731
    return sizing([(face, 0.0085), (ears, 0.011), (lambda P: P[:, 1] < 1.53, 0.022)], 0.016)


def garment_sizing():
    knee = lambda P: np.abs(P[:, 1] - KNEE_Y) < 0.09  # noqa: E731
    shoulder = lambda P: (np.abs(P[:, 0]) > 0.12) & (P[:, 1] > 1.3)  # noqa: E731
    return sizing([(knee, 0.026), (shoulder, 0.03)], 0.05)


def body_sizing():
    return sizing([(lambda P: np.abs(P[:, 0]) > 0.3, 0.022)], 0.05)  # arms bend at the elbows


def build_body(garments, hair):
    """Skin: a coarse body (arms, and what the clothes hide) plus the head and both hands meshed
    finely on their own. Skin hidden under the garments or the hair is dropped."""
    shape = B.body_sdf()
    gl = [g for _, g in garments.values()]
    hid = covered_by(gl)
    coarse_out = lambda P: in_head_box(P, 0.012) | in_hand(P, cut=HAND_CUT + 0.012)  # noqa: E731
    body = mesh_sdf(shape, CELL, BUDGET["Body"], max_edge=body_sizing(), name="Body", outside=coarse_out,
                    hidden=hid)
    head = mesh_sdf(shape, FINE, BUDGET["Head"], max_edge=head_sizing(), name="Head", bounds=HEAD_BOX,
                    hidden=covered_by(gl + [hair], margin=0.0035))
    parts = [body, head]
    for side in (1, -1):
        c = B.hand_center(side)
        lo, hi = c - 0.14, c + 0.14
        hand = mesh_sdf(shape, FINE, BUDGET["Hand"], max_edge=0.013, name="Hand_L" if side == 1 else "Hand_R",
                        bounds=(lo, hi), outside=lambda P, s=side: ~in_hand(P, s), hidden=hid)
        parts.append(hand)
    return p3.Mesh.join(parts, name="Body"), shape


def build_outfit(body_shape):
    """Meshes the garments; faces hidden under another garment or inside the body are dropped.
    Returns {name: (mesh, sdf)}."""
    g = {name: make() for name, make in C.GARMENTS}
    inside = covered_by([body_shape], margin=0.004)
    over = {  # garment -> garments over it (with an optional region where that one hides it)
        "Tunic": [(g["Belt"], None)],
        "Trousers": [(g["Boots"], None), (g["Belt"], None), (g["Tunic"], lambda P: P[:, 1] > 0.94)],
        "Belt": [(g["Tunic"], None)],
        "Boots": [], "Bracers": [],
    }
    out = {}
    for name, _ in C.GARMENTS:
        tests = [inside] + [covered_by([s], where=w) for s, w in over[name]]
        hidden = lambda P, t=tests: np.any([f(P) for f in t], axis=0)  # noqa: E731
        m = mesh_sdf(g[name], CELL, BUDGET[name], max_edge=garment_sizing(), name=name, hidden=hidden)
        out[name] = (m, g[name])
    return out


def paint_garments(outfit):
    """Vertex-color luminance: darker hem rolls, boot soles and cuffs, bracer straps."""
    tun = outfit["Tunic"][0]
    V = tun.vertices
    lum = np.ones(len(V))
    lum *= 1 - 0.18 * (1 - RG.ss(C.TUNIC_HEM, C.TUNIC_HEM + 0.035, V[:, 1]))  # hem band
    for s in (1, -1):  # sleeve hems
        on = (np.sign(V[:, 0]) == s) & (np.abs(V[:, 0]) > 0.15)
        lum = np.where(on, lum * (1 - 0.16 * RG.ss(C.SLEEVE - 0.03, C.SLEEVE - 0.022, _along(V, s))), lum)
    lum *= 1 - 0.1 * RG.ss(0.0, 1.0, (1.0 - V[:, 1]) / 0.25)  # skirt slightly darker lower down
    tun.colors[:, :3] = lum[:, None]
    boots = outfit["Boots"][0]
    V = boots.vertices
    lum = np.where(V[:, 1] < 0.024, 0.45, 1.0)  # sole
    lum = lum * (1 - 0.12 * (V[:, 1] > C.BOOT_TOP - 0.05))  # cuff
    boots.colors[:, :3] = lum[:, None]
    br = outfit["Bracers"][0]
    V = br.vertices
    lum = np.ones(len(V))
    for s in (1, -1):
        d = (V - mirror_side(SHOULDER, s)) @ arm_dir(s)
        on = np.sign(V[:, 0]) == s
        for k in (0.55, 0.85):
            c = UPPER_ARM + FOREARM * k
            lum = np.where(on & (np.abs(d - c) < 0.011), 0.62, lum)
    br.colors[:, :3] = lum[:, None]


def paint_face(body):
    """Lips darker and warmer (the upper one a bit more), a little blush on the cheeks (vertex
    colors multiply the skin tint)."""
    V = body.vertices
    d = (V - B.head_xf([0, B.MOUTH_Y, B.MOUTH_Z])) / B.HEAD_SCALE
    hw = B.MOUTH_W * 1.02
    wx = np.clip(1 - (np.abs(d[:, 0]) / hw) ** 4, 0, 1)
    up = RG.ss(0.0105, 0.0065, d[:, 1]) * RG.ss(-0.002, 0.0005, d[:, 1])
    dn = RG.ss(-0.0125, -0.0085, d[:, 1]) * RG.ss(0.0005, -0.0015, d[:, 1])
    lips = wx * np.maximum(up, 0.75 * dn) * (d[:, 2] > -0.016)
    col = body.colors[:, :3]
    col *= 1 - lips[:, None] * (1 - np.array([0.84, 0.66, 0.64]))
    for s in (1, -1):
        c = B.head_xf([s * 0.052, 1.618, 0.072])
        r = np.linalg.norm(V - c, axis=1)
        k = np.clip(1 - r / 0.024, 0, 1)[:, None] * 0.1
        col *= 1 - k * (1 - np.array([1.0, 0.86, 0.84]))
    body.colors[:, :3] = col


CACHE = Path(__file__).resolve().parent / ".cache"
SOURCES = ("body.py", "clothes.py", "hair.py", "face.py", "shapes.py", "meshing.py", "common.py", "build_human.py")


def build_meshes():
    """Meshing is the slow part: results are cached by the hash of the modeling sources."""
    h = hashlib.sha1(f"{CELL} {FINE}".encode())
    for f in SOURCES:
        h.update((Path(__file__).resolve().parent / f).read_bytes())
    for f in ("sdf.py", "remesh.py", "mesh.py"):
        h.update((Path(p3.__file__).resolve().parent / f).read_bytes())
    path = CACHE / f"meshes_{h.hexdigest()[:16]}.pkl"
    if path.exists() and "--no-cache" not in sys.argv:
        print("meshes from cache", path.name)
        return pickle.loads(path.read_bytes())
    body_shape = B.body_sdf()
    outfit = build_outfit(body_shape)
    hair_shape = HR.hair_sdf()
    hair = mesh_sdf(hair_shape, FINE * 1.15, BUDGET["Hair_curto"], max_edge=0.015, name="Hair_curto",
                    hidden=covered_by([B.head_scaled()], margin=0.003))
    body, _ = build_body(outfit, hair_shape)
    paint_garments(outfit)
    paint_face(body)
    eyes = FC.eyes()
    brows = FC.brows(B.head_scaled())
    res = (body, {k: v[0] for k, v in outfit.items()}, hair, eyes, brows)
    CACHE.mkdir(exist_ok=True)
    for old in CACHE.glob("meshes_*.pkl"):
        old.unlink()
    path.write_bytes(pickle.dumps(res))
    return res


def build():
    t0 = time.time()
    body, outfit, hair, eyes, brows = build_meshes()

    sk = build_skeleton()
    head_sdf = B.head_scaled()
    RG.skin_mesh(body, sk, head_sdf)
    for name, m in outfit.items():
        if name == "Tunic":
            RG.skin_mesh(m, sk, head_sdf, skirt_mask=m.vertices[:, 1] < 1.05)
        elif name == "Trousers":  # under the skirt the trousers move with it (no poking through)
            RG.skin_mesh(m, sk, head_sdf, skirt_mask=m.vertices[:, 1] > 0.85, skirt_blend=lambda y: RG.ss(0.86, 0.91, y) * RG.ss(1.045, 0.975, y))
        else:
            RG.skin_mesh(m, sk, head_sdf)
    for m in (hair, eyes, brows):
        RG.skin_mesh(m, sk, rigid_bone="Head")

    # first person hides the head: split the skin where Head + Neck dominate
    names = [b.name for b in sk.bones]
    hn = [names.index("Head"), names.index("Neck")]
    hw = np.zeros(body.num_vertices)
    for k in range(4):
        hw += np.where(np.isin(body.joints[:, k], hn), body.weights[:, k], 0)
    fh = np.array([np.mean(hw[list(f)]) for f in body.faces]) > 0.5
    head_skin = body.separate(np.flatnonzero(fh), name="Head_Skin")
    body.delete_faces(np.flatnonzero(fh))
    body.name = "Body"
    outfit_mesh = p3.Mesh.join(list(outfit.values()), name="Outfit")

    parts = {"Body": body, "Head_Skin": head_skin, "Head_Eyes": eyes, "Head_Brows": brows, "Hair_curto": hair,
             "Outfit": outfit_mesh}
    for n, m in parts.items():
        probs = m.validate()
        if probs:
            print("  validate", n, probs)
    scene = p3.Scene("Niryk_human_m")
    scene.add(sk.root)
    nodes = {}
    for n, m in parts.items():
        nodes[n] = scene.add(p3.skinned_node(m, sk, n))
    rig = A.Rig(sk)
    A.bake(scene, rig)
    out = export_glb(scene, "human_m")
    stats = {n: m.num_triangles for n, m in parts.items()}
    stats.update({f"g_{k}": v.num_triangles for k, v in outfit.items()})
    stats["total"] = sum(m.num_triangles for m in parts.values())
    stats["height_m"] = round(float(max(m.bounds()[1][1] for m in parts.values())), 3)
    stats["bones"] = len(sk.bones)
    stats["time_s"] = round(time.time() - t0, 1)
    write_stats("human_m", stats)
    print("human_m", stats, "->", out)
    return scene, sk, nodes


def build_sword():
    import equipment as EQ
    s = p3.Scene("Sword_Iron")
    m = EQ.sword_iron()
    s.add(m)
    out = export_glb(s, "sword_iron", sub="equipment")
    print("sword_iron", m.num_triangles, "tris ->", out)
    return m


def hand_world(rig, pose, side):
    """World position of a hand's center in a pose (side: "Left" / "Right")."""
    bone = side + "Hand"
    off = B.hand_center(1 if side == "Left" else -1) - rig.head[bone]
    return rig.world_head(pose, bone) + A.rot(rig.world_rot(pose, bone), off)


def review_hands(scene, sk, out):
    """Close-ups of both hands (toon): the sword hand from the front, the side, below and from the
    eyes (first person), at rest and at the hit of Attack1; the relaxed hand from two sides."""
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
        dict(view=(0.2, 0.5, 1.0), target=hand_world(rig, hit, "Right"), distance=0.5, time=A.HIT_TIME["Attack1"],
             animation="Attack1"),
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
    write_png(out / "human_m_hands.png", grid(imgs, 4))


def review(scene, sk, nodes, sword, clips=None):
    out = REVIEW / "characters"
    out.mkdir(parents=True, exist_ok=True)
    review_lights(scene)
    sock = next(c for c in sk.bone("RightHand").children if c.name == "Socket_Weapon_R")
    sock.add(p3.Node("SwordPreview", sword))
    tint_for_review(scene)
    from pyram3d.image import grid, write_png
    p3.render_views(scene, out / "human_m_turnaround.png", views=("front", "three_quarter", "right", "back"),
                    size=560, cols=4, style="toon", time=0.0, animation="Idle")
    hy = 1.6
    imgs = [p3.render(scene, None, 520, 520, view=v, style="toon", target=(0, hy, 0.0), distance=1.05, yfov=30,
                      time=0.0, animation="Idle") for v in [(0.0, 0.05, 1), (0.75, 0.15, 1), (1, 0.05, -0.1)]]
    imgs.append(p3.render(scene, None, 520, 520, view=(0.45, 0.35, -1), style="toon", target=(0, 1.45, 0),
                          distance=1.5, yfov=30, time=0.0, animation="Idle"))
    write_png(out / "human_m_closeup.png", grid(imgs, 4))
    imgs = [p3.render(scene, None, 520, 520, view=v, style="clay", target=(0, hy, 0.0), distance=1.05, yfov=30,
                      time=0.0, animation="Idle") for v in [(0.0, 0.05, 1), (0.75, 0.15, 1)]]
    write_png(out / "human_m_closeup_clay.png", grid(imgs, 2))
    review_hands(scene, sk, out)
    if clips is None:
        clips = () if "--no-clips" in sys.argv else ("Walk", "StrafeLeft", "Run", "Attack1", "Attack2", "WalkBack",
                                                     "StrafeRight", "Idle", "Hit", "Death")
    for clip in clips:
        view = (1.0, 0.25, 0.25) if clip in ("Walk", "Run", "WalkBack") else (0.55, 0.3, 1.0)
        p3.render_frames(scene, out / f"human_m_{clip}.png", clip, frames=6, size=320, style="toon", view=view)
    untint(scene)


if __name__ == "__main__":
    sword = build_sword()
    scene, sk, nodes = build()
    if "--no-review" not in sys.argv:
        review(scene, sk, nodes, sword)
    print("height", HEIGHT)
