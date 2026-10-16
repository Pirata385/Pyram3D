"""Builds the 4 playable bodies of Niryk: human_m, human_f, harpin_m, harpin_f.

    python projects/niryk/build_characters.py [name ...] [--fast]

Each GLB (exports/characters/<name>.glb) holds the rig (+ sockets), the body, the starter outfit,
every hair style as a separate node (Hair_<style>; the game shows one), eyes, Harpin wings, and
the clips Idle/Walk/Run/Attack1/Attack2/Hit/Death. Material names are the color slots the game
tints. Review sheets go to review/characters/.
"""
from __future__ import annotations

import sys
import time

import numpy as np

import anim as A
import body as B
import clothes as C
import equipment as EQ
import hair as HR
import harpin as H
from common import REVIEW, SPECS, build_skeleton, export_glb, side_filter, slot, write_stats

import pyram3d as p3

FAST = "--fast" in sys.argv
ARMOR_ONLY = "--armor-only" in sys.argv
CELL = 0.024 if FAST else 0.0145


# ----------------------------------------------------------------------------- painting

def paint_plumes(mesh: p3.Mesh, spec, rows=30.0, low=0.74):
    """Overlapping feather rows (luminance) on every Plume face of an SDF mesh. `rows` = rows per
    meter (the wing mass uses fewer, bigger rows so they still read at game scale)."""
    plume_idx = [i for i, m in enumerate(mesh.materials) if m is not None and m.name == "Plume"]
    if not plume_idx or mesh.colors is None:
        return
    fm = np.asarray(mesh.face_materials)
    verts = np.unique(np.concatenate([np.asarray(mesh.faces[i]) for i in np.flatnonzero(np.isin(fm, plume_idx))]))
    P = mesh.vertices[verts]
    ang = np.arctan2(P[:, 0] - np.sign(P[:, 0]) * spec.hip, P[:, 2])
    row = (P[:, 1] * rows - P[:, 2] * rows * 0.4 + 0.35 * np.sin(ang * 5)) % 1.0
    lum = low + (1 - low) * np.clip((row - 0.05) / 0.55, 0, 1)
    mesh.colors[verts, :3] *= lum[:, None]


def paint_garment_hem(mesh: p3.Mesh, hem_y: float):
    if mesh.colors is None:
        return
    y = mesh.vertices[:, 1]
    band = (y > hem_y) & (y < hem_y + 0.035)
    mesh.colors[band, :3] *= 0.72


def paint_face(mesh: p3.Mesh, spec):
    """Lips slightly darker, a hint of blush (vertex colors multiply the skin tint)."""
    s = spec.s
    P = mesh.vertices
    lips = (np.abs(P[:, 0]) < 0.022) & (np.abs(P[:, 1] - 1.565 * s) < 0.008) & (P[:, 2] > 0.07)
    mesh.colors[lips, :3] *= np.array([0.86, 0.68, 0.66]) if spec.gender == "m" else np.array([0.76, 0.52, 0.54])
    cheek = (np.abs(np.abs(P[:, 0]) - 0.05) < 0.016) & (np.abs(P[:, 1] - 1.6 * s) < 0.014) & (P[:, 2] > 0.05)
    mesh.colors[cheek, :3] *= np.array([1.0, 0.9, 0.88])


# ----------------------------------------------------------------------------- parts

def eyes(spec):
    s = spec.s
    parts = []
    big = 1.1 if spec.gender == "f" else 1.0  # slightly larger eyes on the female faces
    for sgn in (1, -1):
        c = np.array([0.036 * sgn, 1.637 * s, 0.079])
        white = p3.uv_sphere(0.0175 * big, 10, 6).translate(c).set_material(slot("EyeWhite"))
        iris = p3.uv_sphere(0.0105 * big, 8, 4).scale((1, 1, 0.45)).translate(c + np.array([0, 0, 0.0145 * big])).set_material(slot("Eye"))
        pupil = p3.uv_sphere(0.0052 * big, 6, 3).scale((1, 1, 0.4)).translate(c + np.array([0, 0, 0.0182 * big])).set_material(slot("Dark"))
        lid = p3.uv_sphere(0.0185, 10, 5).scale((1.05, 0.5, 1.0)).translate(c + np.array([0, 0.011, 0.001])).set_material(slot("Skin"))
        parts += [white, iris, pupil, lid]
    m = p3.Mesh.join(parts, name="Eyes")
    m.colors = None
    return m


def rigid(mesh: p3.Mesh, sk: p3.Skeleton, bone: str):
    idx = [b.name for b in sk.bones].index(bone)
    mesh.joints = np.zeros((mesh.num_vertices, 4), np.int64)
    mesh.weights = np.zeros((mesh.num_vertices, 4))
    mesh.joints[:, 0] = idx
    mesh.weights[:, 0] = 1.0
    return mesh


def body_filter(sk, spec):
    """Body weights: no wing bones; tail bones only for the tail (harpin)."""
    names = [b.name for b in sk.bones]
    side = side_filter(sk)
    s = spec.s

    def f(P, i):
        n = names[i]
        ok = side(P, i)
        # the tail (thick, with a wide brush) hangs behind the legs: everything well behind the heels
        is_tail = (P[:, 2] < -0.16) & (P[:, 1] < 0.98 * s) & (np.abs(P[:, 0]) < 0.25)
        if "Wing" in n:
            return np.zeros(len(P), bool)
        if n.startswith("Tail"):
            return ok & is_tail
        if spec.harpin and n not in ("Hips",):
            return ok & ~is_tail
        return ok
    return f


def build_armor(spec, name, sk, source, hem):
    """Armor set fitted to this body (separate GLB, same bone names / bind pose)."""
    armor_meshes = []
    for aname, a_sdf, cell, skirt_like in EQ.armor_parts(spec):
        am = a_sdf.to_mesh(cell=cell * (1.5 if FAST else 1.0), name=aname)
        if am.num_vertices == 0:
            continue
        am.colors = None
        sk.transfer_weights(am, source)
        if skirt_like:
            sk.bind_skirt(am, mask=am.vertices[:, 1] < 0.95 * spec.s, top=0.98 * spec.s, bottom=hem, influence=0.45)
        armor_meshes.append(am)
    armor = p3.Mesh.join(armor_meshes, name="Armor")
    arm_scene = p3.Scene(f"Armor_{name}")
    sk_a = build_skeleton(spec)
    arm_scene.add(sk_a.root)
    arm_scene.add(p3.skinned_node(armor, sk_a, "Armor"))
    print(f"armor_guard_{name}", armor.num_triangles, "->", export_glb(arm_scene, f"armor_guard_{name}", sub="equipment"))
    return armor


def build(name: str):
    t0 = time.time()
    spec = SPECS[name]
    sk = build_skeleton(spec)
    rig = A.Rig(spec, sk)
    scene = p3.Scene(f"Niryk_{name}")
    scene.add(sk.root)

    # body
    body_sdf = B.body_sdf(spec)
    body = body_sdf.to_mesh(cell=CELL, name="Body")
    if body.colors is None:
        body.colors = np.ones((body.num_vertices, 4))
    paint_face(body, spec)
    paint_plumes(body, spec)
    sk.bind(body, smooth=6, falloff=5.0, bone_filter=body_filter(sk, spec))
    source = body.copy()
    if ARMOR_ONLY:  # quick re-export of the armor only (weights still come from this body)
        build_armor(spec, name, sk, source, C.outfit_parts(spec)[1])
        return None

    # outfit
    garments, hem = C.outfit_parts(spec)
    covered = np.zeros(body.num_vertices, bool)
    stats_g = {}
    outfit_meshes = []
    for gname, g, cell in garments:
        gm = g.to_mesh(cell=cell * (1.6 if FAST else 1.0), name=gname)
        if gm.num_vertices == 0:
            continue
        if gm.colors is None:
            gm.colors = np.ones((gm.num_vertices, 4))
        sk.transfer_weights(gm, source)
        if gname == "Tunic":
            paint_garment_hem(gm, hem)
            skirt = gm.vertices[:, 1] < 0.95 * spec.s
            sk.bind_skirt(gm, mask=skirt, top=0.98 * spec.s, bottom=hem, influence=0.5)
        outfit_meshes.append(gm)
        stats_g[gname] = gm.num_triangles
        covered |= g.dist(body.vertices) < -0.004
    faces_cov = [i for i, f in enumerate(body.faces) if all(covered[v] for v in f)]
    body.delete_faces(faces_cov).remove_unused_vertices()
    outfit = p3.Mesh.join(outfit_meshes, name="Outfit")

    nodes = [p3.skinned_node(body, sk, "Body"), p3.skinned_node(outfit, sk, "Outfit")]
    stats = {"body": body.num_triangles, "outfit": outfit.num_triangles, **{f"g_{k}": v for k, v in stats_g.items()}}

    # eyes (rigid to the head)
    ey = rigid(eyes(spec), sk, "Head")
    nodes.append(p3.skinned_node(ey, sk, "Eyes"))

    # hair styles
    names = [b.name for b in sk.bones]
    head_bones = {names.index(n) for n in ("Head", "Neck", "Chest")}
    for style in HR.styles_for(spec):
        hm = HR.style_sdf(spec, style).to_mesh(cell=0.009 if not FAST else 0.016, name=f"Hair_{style}")
        if spec.harpin:  # temple feathers (cards) belong to every harpin style
            tf = HR.temple_feathers(spec, {"espetado": 1.0, "crista": 0.85, "plumado": 1.05}[style])
            if hm.colors is None:
                hm.colors = np.ones((hm.num_vertices, 4))
            hm = p3.Mesh.join([hm, tf], name=f"Hair_{style}")
        sk.bind(hm, smooth=4, falloff=6.0, bone_filter=lambda P, i: np.full(len(P), i in head_bones))
        low = hm.vertices[:, 1] > 1.55 * spec.s  # everything above the jaw follows the head exactly
        hm.joints[low] = 0
        hm.weights[low] = 0
        hm.joints[low, 0] = names.index("Head")
        hm.weights[low, 0] = 1.0
        nodes.append(p3.skinned_node(hm, sk, f"Hair_{style}"))
        stats[f"hair_{style}"] = hm.num_triangles

    # wings
    if spec.harpin:
        fe, arm = H.wing_left(spec, sk)
        paint_plumes(arm, spec, rows=12.0, low=0.66)
        wings = p3.Mesh.join([fe, arm, H.mirror_skinned(fe, sk, "fr"), H.mirror_skinned(arm, sk, "ar")], name="Wings")
        nodes.append(p3.skinned_node(wings, sk, "Wings"))
        stats["wings"] = wings.num_triangles

    armor = build_armor(spec, name, sk, source, hem)
    stats["armor"] = armor.num_triangles

    for n in nodes:
        scene.add(n)
    A.bake(scene, rig)
    out = export_glb(scene, name)
    stats["time_s"] = round(time.time() - t0, 1)
    stats["total_default"] = stats["body"] + stats["outfit"] + ey.num_triangles + stats.get("wings", 0) + \
        stats[f"hair_{HR.styles_for(spec)[0]}"]
    write_stats(name, stats)
    print(name, stats, "->", out)
    nodes.append(p3.skinned_node(armor, sk, "ArmorPreview"))  # review only (not in the body GLB)
    return scene, sk, nodes


def detach(node):
    if node.parent is not None and node in node.parent.children:
        node.parent.children.remove(node)
    node.parent = None


def review(name, scene, sk, nodes):
    """Turnaround with the default hair, plus a head sheet with every hair style."""
    spec = SPECS[name]
    out = REVIEW / "characters"
    out.mkdir(parents=True, exist_ok=True)
    styles = HR.styles_for(spec)
    hair_nodes = {n.name: n for n in nodes if n.name.startswith("Hair_")}
    armor_node = next(n for n in nodes if n.name == "ArmorPreview")
    for n in hair_nodes.values():  # the renderer draws every node: keep one style at a time
        detach(n)
    first = hair_nodes[f"Hair_{styles[0]}"]
    scene.add(first)
    sword = EQ.sword_iron()
    sock = [c for c in sk.bone("RightHand").children if c.name == "Socket_Weapon_R"][0]
    sword_node = sock.add(p3.Node("SwordPreview", sword))
    p3.render_views(scene, out / f"{name}_turnaround.png", views=("front", "three_quarter", "right", "back"),
                    size=520, cols=4, style="toon", time=0.0, animation="Idle")
    for clip in ("Idle", "Walk", "Run", "Attack1", "Attack2", "Hit", "Death"):
        p3.render_frames(scene, out / f"{name}_{clip}.png", clip, frames=6, size=300, style="toon",
                         view=(0.54, 0.64, 0.54))
    # wearing the armor (game view + front/back)
    scene.add(armor_node)
    p3.render_views(scene, out / f"{name}_armor.png", views=("front", "three_quarter", "back", (0.54, 0.64, 0.54)),
                    size=420, cols=4, style="toon", time=0.0, animation="Idle")
    p3.render_frames(scene, out / f"{name}_armor_Attack1.png", "Attack1", frames=5, size=300, style="toon",
                     view=(0.54, 0.64, 0.54))
    detach(armor_node)
    detach(first)
    detach(sword_node)
    heads = []
    for st in styles:
        hn = hair_nodes[f"Hair_{st}"]
        scene.add(hn)
        heads.append(p3.render(scene, None, 360, 360, view=(0.6, 0.25, 1.0), style="toon", target=(0, 1.62 * spec.s, -0.02),
                               distance=1.1, yfov=35, time=0.0, animation="Idle"))
        heads.append(p3.render(scene, None, 360, 360, view=(-0.4, 0.2, -1.0), style="toon", target=(0, 1.62 * spec.s, -0.02),
                               distance=1.1, yfov=35, time=0.0, animation="Idle"))
        detach(hn)
    from pyram3d.image import grid, write_png
    write_png(out / f"{name}_hair.png", grid(heads, cols=4))
    for n in hair_nodes.values():
        scene.add(n)


def build_sword():
    s = p3.Scene("Sword_Iron")
    s.add(EQ.sword_iron())
    out = export_glb(s, "sword_iron", sub="equipment")
    p3.render_views(s, REVIEW / "characters" / "sword_iron.png", views=("front", "right", "three_quarter"), size=360,
                    cols=3, style="toon")
    print("sword ->", out)


if __name__ == "__main__":
    if "--sword" in sys.argv:
        build_sword()
        sys.exit()
    which = [a for a in sys.argv[1:] if not a.startswith("--")] or list(SPECS)
    for nm in which:
        res = build(nm)
        if res is None:
            continue
        sc, skel, nds = res
        if "--no-review" not in sys.argv:
            review(nm, sc, skel, nds)
