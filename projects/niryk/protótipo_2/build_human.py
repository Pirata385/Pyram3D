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
from common import (FOREARM, HEIGHT, REVIEW, SHOULDER, UPPER_ARM, arm_dir, build_skeleton, export_glb, mirror_side,
                    review_lights, tint_for_review, untint, write_stats)
from meshing import mesh_sdf, sizing

import pyram3d as p3

FAST = "--fast" in sys.argv
CELL = 0.0065 if FAST else 0.0045


def body_sizing():
    hands = [B.hand_center(s) for s in (1, -1)]
    return sizing([
        (lambda P: (P[:, 1] > 1.5) & (np.abs(P[:, 0]) < 0.13), 0.011),  # head & neck
        (lambda P: np.min([np.linalg.norm(P - h, axis=1) for h in hands], axis=0) < 0.12, 0.0095),  # hands
        (lambda P: np.abs(P[:, 0]) > 0.3, 0.022),  # arms (bend at the elbows)
    ], 0.05)  # torso and legs: under the clothes


def build_outfit(body):
    """Meshes the garments and deletes the body faces they cover. Returns {name: (mesh, sdf)}."""
    out = {}
    covered = np.zeros(body.num_vertices, bool)
    for name, make, cs, target in C.GARMENTS:
        g = make()
        m = mesh_sdf(g, CELL * cs, target, max_edge=0.03, name=name)
        out[name] = (m, g)
        covered |= g.dist(body.vertices) < -0.003
    faces = [i for i, f in enumerate(body.faces) if all(covered[v] for v in f)]
    body.delete_faces(faces)
    return out


def paint_garments(outfit):
    """Vertex-color luminance: darker hem rolls, boot soles and cuffs, bracer straps."""
    tun = outfit["Tunic"][0]
    V = tun.vertices
    lum = np.ones(len(V))
    lum *= 1 - 0.18 * (1 - RG.ss(C.TUNIC_HEM, C.TUNIC_HEM + 0.035, V[:, 1]))  # hem band
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
    """Lips slightly darker and warmer, a little blush on the cheeks (vertex colors multiply the
    skin tint)."""
    V = body.vertices
    mouth = B.head_xf([0, B.MOUTH_Y, 0.1])
    d = V - mouth
    hw = 0.024 * B.HEAD_SCALE
    lips = (np.abs(d[:, 0]) < hw) & (np.abs(d[:, 1] + 0.003) < 0.009) & (d[:, 2] > -0.012)
    w = np.clip(1 - np.abs(d[:, 0]) / hw, 0, 1) ** 0.5
    col = body.colors[:, :3]
    col[lips] *= (1 - w[lips, None] * (1 - np.array([0.86, 0.7, 0.68])))
    for s in (1, -1):
        c = B.head_xf([s * 0.05, 1.615, 0.075])
        r = np.linalg.norm(V - c, axis=1)
        k = np.clip(1 - r / 0.022, 0, 1)[:, None] * 0.1
        col *= 1 - k * (1 - np.array([1.0, 0.86, 0.84]))
    body.colors[:, :3] = col


CACHE = Path(__file__).resolve().parent / ".cache"
SOURCES = ("body.py", "clothes.py", "hair.py", "face.py", "shapes.py", "meshing.py", "common.py", "build_human.py")


def build_meshes():
    """Meshing is the slow part: results are cached by the hash of the modeling sources."""
    h = hashlib.sha1(str(CELL).encode())
    for f in SOURCES:
        h.update((Path(__file__).resolve().parent / f).read_bytes())
    for f in ("sdf.py", "remesh.py", "mesh.py"):
        h.update((Path(p3.__file__).resolve().parent / f).read_bytes())
    path = CACHE / f"meshes_{h.hexdigest()[:16]}.pkl"
    if path.exists() and "--no-cache" not in sys.argv:
        print("meshes from cache", path.name)
        return pickle.loads(path.read_bytes())
    shape = B.body_sdf()
    body = mesh_sdf(shape, CELL, 16000 if not FAST else 12000, max_edge=body_sizing(), name="Body")
    outfit = build_outfit(body)
    paint_garments(outfit)
    paint_face(body)
    hair = mesh_sdf(HR.hair_sdf(), CELL * 0.85, 3200, max_edge=0.015, name="Hair_curto")
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
        mask = (m.vertices[:, 1] < 1.0) if name == "Tunic" else None
        RG.skin_mesh(m, sk, head_sdf, skirt_mask=mask, hem=C.TUNIC_HEM)
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
    for clip in clips or ("Walk", "StrafeLeft", "Run", "Attack1", "Attack2", "WalkBack", "StrafeRight", "Idle",
                          "Hit", "Death"):
        view = (1.0, 0.25, 0.25) if clip in ("Walk", "Run", "WalkBack") else (0.55, 0.3, 1.0)
        p3.render_frames(scene, out / f"human_m_{clip}.png", clip, frames=6, size=320, style="toon", view=view)
    untint(scene)


if __name__ == "__main__":
    sword = build_sword()
    scene, sk, nodes = build()
    if "--no-review" not in sys.argv:
        review(scene, sk, nodes, sword)
    print("height", HEIGHT)
