"""Builds the prototype 2 anime-RPG characters (humans and Harpins): body, outfit, face, hair styles,
rig, clips, review.

    python projects/niryk/protótipo_2/build_characters.py [human_m] [human_f] [harpin_m] [harpin_f]
           [--fast] [--jobs 2] [--no-review] [--all-clips] [--no-clips] [--no-cache]

With no body named, builds all four. Each body is built in its own process (the modeling modules read
the body's proportions at import time, from NIRYK_P2_BODY). Writes exports/characters/<body>.glb,
exports/equipment/sword_iron.glb, review/characters/<body>_*.png and <body>_stats.json. --fast
meshes coarser (quick iteration); exports should use the default (fine) settings.

GLB contents (node names are what the client looks for):
  Rig            armature (Unity Humanoid bone names) + Socket_Weapon_R / Socket_Back / Socket_Head
  Body           skin below the neck (hands)                              [Skin]
  Head_Skin      head and neck skin (face normals bent toward an ellipsoid) [Skin]
  Head_Eyes      eye whites, irises, pupils, highlights, lash lines        [EyeWhite, Eye, Dark, EyeShine]
  Head_Brows     brows                                                     [Hair]
  Head_Mouth     mouth line                                                [Dark]
  Hair_curto / Hair_rabo / Hair_longo / Hair_coque   (the client shows one) [Hair (+ Leather tie)]
  Outfit         tunic, collar, trousers, boots, belt + pouch, bracers     [Cloth_A, Cloth_B, Leather, Metal]
  clips          Idle Walk Run WalkBack StrafeLeft StrafeRight Attack1 Attack2 Hit Death
Harpins (harpin_m, harpin_f) differ:
  Body           skin, the plumed digitigrade legs, scaled toes and claws  [Skin, Plume, Talon]
  Head_Plumes    feather tufts at the temples (pointed ears are in Head_Skin) [Plume]
  Hair_espetado / Hair_crista / Hair_plumado   anime locks with feathers   [Hair]
  Wings          two folded wings: plumed arms and feather solids          [Plume]  bones Left/RightWing1..3
  Tail           long plumed tail with a brush                             [Plume]  bones Tail1..6
  Outfit         no trousers or boots; wing slits, shorter at the back with a notch over the tail
"""
from __future__ import annotations

import os
import subprocess
import sys

BODIES = ("human_m", "human_f", "harpin_m", "harpin_f")


def _dispatch():
    """Runs one process per requested body (unless this process is already that body's)."""
    wanted = [a for a in sys.argv[1:] if a in BODIES] or list(BODIES)
    flags = [a for a in sys.argv[1:] if a not in BODIES]
    jobs = 1
    if "--jobs" in flags:
        i = flags.index("--jobs")
        jobs = int(flags[i + 1])
        del flags[i:i + 2]
    if len(wanted) == 1 and os.environ.get("NIRYK_P2_BODY") == wanted[0]:
        return  # build it here
    procs, code = [], 0
    for b in wanted:
        env = dict(os.environ, NIRYK_P2_BODY=b)
        procs.append(subprocess.Popen([sys.executable, __file__, b] + flags, env=env))
        if len(procs) >= jobs:
            code |= procs.pop(0).wait()
    for p in procs:
        code |= p.wait()
    sys.exit(code)


if __name__ == "__main__":
    _dispatch()

import hashlib  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

import anim as A  # noqa: E402
import body as B  # noqa: E402
import clothes as C  # noqa: E402
import face as FC  # noqa: E402
import hair as HR  # noqa: E402
import harpin as HP  # noqa: E402
import head as H  # noqa: E402
import rig as RG  # noqa: E402
from common import (BODY, FEMALE, FOREARM, HARPIN, KNEE_Y, SHOULDER, UPPER_ARM, arm_dir, build_skeleton,  # noqa: E402
                    export_glb, mirror_side, slot, write_stats, xf_mesh)
from meshing import covered_by, mesh_sdf, sizing  # noqa: E402

import pyram3d as p3  # noqa: E402

FAST = "--fast" in sys.argv
CELL = 0.0065 if FAST else 0.0052  # body (mostly under the clothes) and garments
FINE = 0.0036 if FAST else 0.0029  # head and hands: seen up close
HAIR_CELL = 0.0042 if FAST else 0.0034

# Regions meshed finely on their own (their open cuts lie under the collar and the bracers).
HEAD_BOX = (np.array([-0.15, 1.42, -0.17]), np.array([0.15, 1.86, 0.17]))
HAND_CUT = UPPER_ARM + FOREARM * 0.72  # hands start here along the arm (under the bracer)
BUDGET = {"Body": 1400, "Head": 5000, "Hand": 1500, "Tunic": 3600, "Collar": 1700, "Trousers": 1000,
          "Boots": 1700, "Belt": 1100, "Bracers": 700, "Underskirt": 600,
          "Hair_curto": 2600, "Hair_rabo": 3000, "Hair_longo": 3400, "Hair_coque": 2800,
          # Harpins
          "Legs": 2400, "Tail": 1600, "WingArm": 500, "Hair_espetado": 2400, "Hair_crista": 2200, "Hair_plumado": 2600}
if HARPIN:
    BUDGET.update({"Body": 1200, "Head": 4400, "Hand": 1100, "Tunic": 2700, "Collar": 1300})
LEG_CUT = 0.8  # Harpins: the legs below this height are meshed finely on their own (under the skirt)
TAIL_CELL = 0.0048 if FAST else 0.0042


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
    face = lambda P: (P[:, 2] > 0.02) & (P[:, 1] > 1.53) & (P[:, 1] < 1.72) & (np.abs(P[:, 0]) < 0.08)  # noqa: E731
    ears = lambda P: (np.abs(P[:, 0]) > 0.07) & (P[:, 1] > 1.58) & (P[:, 1] < 1.69)  # noqa: E731
    return sizing([(face, 0.009), (ears, 0.011), (lambda P: P[:, 1] < 1.53, 0.022)], 0.02)


def garment_sizing():
    knee = lambda P: np.abs(P[:, 1] - KNEE_Y) < 0.09  # noqa: E731
    shoulder = lambda P: (np.abs(P[:, 0]) > 0.12) & (P[:, 1] > 1.3)  # noqa: E731
    elbow = lambda P: (np.abs(P[:, 0]) > 0.25) & (P[:, 1] > 1.05)  # noqa: E731
    return sizing([(knee, 0.026), (shoulder, 0.028), (elbow, 0.026)], 0.05)


def body_sizing():
    return sizing([(lambda P: np.abs(P[:, 0]) > 0.3, 0.022)], 0.05)


# ----------------------------------------------------------------------------- meshing

def build_outfit(body_shape):
    """Meshes the garments; faces hidden under another garment or inside the body are dropped.
    Returns {name: (mesh, sdf)}."""
    g = {name: make() for name, make in C.GARMENTS}
    inside = covered_by([body_shape], margin=0.004)
    over = {  # garment -> garments over it (with an optional region where that one hides it)
        "Tunic": [(g["Belt"], None), (g["Collar"], None)],
        "Collar": [],
        "Trousers": [(g["Boots"], None), (g["Belt"], None), (g["Tunic"], lambda P: P[:, 1] > 0.94)] if not HARPIN else [],
        "Belt": [(g["Tunic"], None)],
        "Boots": [], "Bracers": [],
        "Underskirt": [(g["Tunic"], None)] if FEMALE else [],
    }
    out = {}
    for name, _ in C.GARMENTS:
        tests = [inside] + [covered_by([s], where=w) for s, w in over[name]]
        hidden = lambda P, t=tests: np.any([f(P) for f in t], axis=0)  # noqa: E731
        me = garment_sizing() if name != "Trousers" else 0.04
        m = mesh_sdf(g[name], CELL, BUDGET[name], max_edge=me, name=name, hidden=hidden)
        out[name] = (m, g[name])
    return out


def build_hair(style, head_shape):
    shape = HR.hair_sdf(style)
    m = mesh_sdf(shape, HAIR_CELL, BUDGET["Hair_" + style], max_edge=HR.hair_sizing(style), name="Hair_" + style,
                 colors=True, hidden=covered_by([head_shape], margin=0.003), before_decimate=HR.mark_ring)
    HR.finish_ring(m)
    HR.hair_normals(m)
    HR.paint_hair(m, style)
    if HARPIN:  # feathers mixed into the locks
        m.merge(HR.harpin_hair_feathers(style))
    return m, shape


def under_all(shapes, margin=0.0035):
    """hidden() test: points inside every one of `shapes` (skin under every hair style)."""
    def f(P):
        h = np.ones(len(P), bool)
        for s in shapes:
            h &= s.dist(P) < -margin
        return h
    return f


def build_body(garments, hairs):
    """Skin: a coarse body plus the head and both hands meshed finely on their own. Skin hidden
    under the garments (or under every hair style) is dropped."""
    shape = B.body_sdf()
    gl = [g for _, g in garments.values()]
    hid = covered_by(gl)
    coarse_out = lambda P: (in_head_box(P, 0.012) | in_hand(P, cut=HAND_CUT + 0.012)  # noqa: E731
                            | ((P[:, 1] < LEG_CUT - 0.012) if HARPIN else False))
    body = mesh_sdf(shape, CELL, BUDGET["Body"], max_edge=body_sizing(), name="Body", outside=coarse_out,
                    hidden=hid)
    under_hair = under_all(hairs)
    head_hidden = lambda P: covered_by(gl, margin=0.0035)(P) | under_hair(P)  # noqa: E731
    head = mesh_sdf(shape, FINE, BUDGET["Head"], max_edge=head_sizing(), name="Head", bounds=HEAD_BOX,
                    hidden=head_hidden)
    head.normals = H.face_normals(head.vertices, head.normals)
    parts = [body, head]
    if HARPIN:  # the plumed legs (fine plume locks) on their own, cut under the skirt; toes and claws
        lo, hi = np.array([-0.32, -0.02, -0.32]), np.array([0.32, LEG_CUT + 0.03, 0.36])
        legs = mesh_sdf(shape, HAIR_CELL, BUDGET["Legs"], max_edge=0.03, name="Legs", bounds=(lo, hi),
                        outside=lambda P: P[:, 1] > LEG_CUT, hidden=hid)
        parts += [legs, HP.feet_meshes(1), HP.feet_meshes(-1)]
    for side in (1, -1):
        c = B.hand_center(side)
        lo, hi = c - 0.14, c + 0.14
        hand = mesh_sdf(shape, FINE, BUDGET["Hand"], max_edge=0.013, name="Hand_L" if side == 1 else "Hand_R",
                        bounds=(lo, hi), outside=lambda P, s=side: ~in_hand(P, s), hidden=hid)
        parts.append(hand)
    return p3.Mesh.join(parts, name="Body"), shape


def build_wings(hidden):
    """Left wing (plumed arm + feather solids), skinned, then mirrored: the Wings mesh."""
    sk = build_skeleton(final=False)
    names = [b.name for b in sk.bones]
    arm = mesh_sdf(HP.wing_arm_sdf(), HAIR_CELL, BUDGET["WingArm"], max_edge=0.03, name="WingArm", hidden=hidden)
    arm.colors = np.ones((arm.num_vertices, 4))
    arm.colors[:, :3] = 0.97
    RG.skin_with(arm, HP.wing_bone_weights(arm.vertices, names))
    parts = [arm]
    for m, w in HP.wing_feathers():
        m.set_material(slot("Plume"))
        HP.set_weights(m, {"Left" + b: v for b, v in w.items()}, names)
        parts.append(m)
    left = p3.Mesh.join(parts, name="Wings")
    right = HP.mirror_wing(left, names)
    return p3.Mesh.join([left, right], name="Wings")


def build_tail(hidden):
    sk = build_skeleton(final=False)
    m = mesh_sdf(HP.tail_sdf(), TAIL_CELL, BUDGET["Tail"], max_edge=0.03, name="Tail", hidden=hidden)
    m.colors = np.ones((m.num_vertices, 4))
    V = m.vertices
    # a lighter brush tip
    s = RG._polyline_param(V, list(HP.TAIL_POINTS))
    m.colors[:, :3] = (0.9 + 0.1 * RG.ss(4.0, 6.3, s))[:, None]
    RG.skin_with(m, RG.tail_weights(V, [b.name for b in sk.bones]))
    return m


# ----------------------------------------------------------------------------- paint

def _lum(m, faces_mat=None):
    if m.colors is None:
        m.colors = np.ones((m.num_vertices, 4))
    return m.colors


def paint_garments(outfit):
    """Vertex-color luminance: darker skirt bottom, boot soles, the boot fold, bracer straps,
    belt edges and (female) the cincher's lacing."""
    tun = outfit["Tunic"]
    V = tun.vertices
    lum = 1 - 0.1 * RG.ss(0.0, 1.0, (1.0 - V[:, 1]) / 0.3)  # skirt slightly darker lower down
    _lum(tun)[:, :3] = lum[:, None]
    if "Boots" in outfit:
        boots = outfit["Boots"]
        V = boots.vertices
        lum = np.where(V[:, 1] < 0.022, 0.4, 0.86)  # sole
        lum = np.where(V[:, 1] > C.BOOT_TOP - 0.052, 1.0, lum)  # the folded cuff shows the lighter side
        lum = np.where(np.abs(V[:, 1] - (C.BOOT_TOP - 0.052)) < 0.005, 0.55, lum)  # shadow under the fold
        _lum(boots)[:, :3] = lum[:, None]
    br = outfit["Bracers"]
    V = br.vertices
    lum = np.ones(len(V))
    for s in (1, -1):
        d = (V - mirror_side(SHOULDER, s)) @ arm_dir(s)
        on = np.sign(V[:, 0]) == s
        for k in (0.56, 0.86):
            c = UPPER_ARM + FOREARM * k
            lum = np.where(on & (np.abs(d - c) < 0.012), 0.6, lum)
    _lum(br)[:, :3] = lum[:, None]
    belt = outfit["Belt"]
    V = belt.vertices
    lum = np.ones(len(V))
    for y in C.BELT_Y:
        lum = np.where(np.abs(V[:, 1] - y) < 0.005, 0.72, lum)  # stitched edges
    if FEMALE:  # lacing: crossing dark lines on both sides of the buckle
        front = (V[:, 2] > 0.04) & (np.abs(V[:, 0]) > 0.04) & (np.abs(V[:, 0]) < 0.075)
        u = (V[:, 1] - C.BELT_Y[0]) / (C.BELT_Y[1] - C.BELT_Y[0])
        lace = np.abs(((u * 4 + (np.abs(V[:, 0]) - 0.04) / 0.035) % 1.0) - 0.5) < 0.12
        lum = np.where(front & lace, 0.55, lum)
    _lum(belt)[:, :3] = lum[:, None]
    for name in ("Collar", "Trousers", "Underskirt"):
        if name in outfit:
            _lum(outfit[name])[:, :3] = 1.0


def paint_face(body):
    """A soft shade under the nose tip and (female) a little blush (vertex colors multiply the
    skin tint)."""
    V = body.vertices
    col = body.colors[:, :3]
    ty, tz = H.P["nose_tip"]
    c = H.head_xf((0.0, ty - 0.006, tz - 0.004))
    r = np.linalg.norm((V - c) / np.array([1.3, 0.8, 1.0]), axis=1)
    k = np.clip(1 - r / 0.008, 0, 1)[:, None] * 0.22
    col *= 1 - k * (1 - np.array([0.9, 0.72, 0.7]))
    blush = 0.16 if FEMALE else 0.05
    for s in (1, -1):
        c = H.head_xf((s * (H.EYE_X + 0.006), H.EYE_Y - 0.026, H.face_front_z(H.EYE_Y - 0.026) - 0.006))
        r = np.linalg.norm((V - c) / np.array([1.0, 0.6, 1.0]), axis=1)
        k = np.clip(1 - r / 0.02, 0, 1)[:, None] ** 1.5 * blush
        col *= 1 - k * (1 - np.array([1.0, 0.62, 0.62]))
    body.colors[:, :3] = col


# ----------------------------------------------------------------------------- cache

CACHE = Path(__file__).resolve().parent / ".cache"
SOURCES = ("body.py", "head.py", "clothes.py", "hair.py", "face.py", "shapes.py", "meshing.py", "common.py",
           "harpin.py", "rig.py")


MESHING = ("in_head_box", "in_hand", "head_sizing", "garment_sizing", "body_sizing", "build_outfit", "build_hair",
           "under_all", "build_body", "build_wings", "build_tail")


def _mesh_key():
    """Hash of everything the meshes depend on: the modeling modules, the meshing functions of
    this file (not the painting and assembly below: changing those keeps the cache), budgets and
    cell sizes."""
    import inspect
    h = hashlib.sha1(f"{BODY} {CELL} {FINE} {HAIR_CELL} {TAIL_CELL} {LEG_CUT} {sorted(BUDGET.items())}".encode())
    for f in SOURCES:
        h.update((Path(__file__).resolve().parent / f).read_bytes())
    for fn in MESHING:
        h.update(inspect.getsource(globals()[fn]).encode())
    for f in ("sdf.py", "remesh.py", "mesh.py"):
        h.update((Path(p3.__file__).resolve().parent / f).read_bytes())
    return h.hexdigest()[:16]


def build_meshes():
    """Meshing is the slow part: each stage (outfit, hair, body, Harpin extras) is cached by the
    hash of what it depends on, saved as soon as it is done (a crash later resumes there).
    Painting happens after loading (it is cheap and not cached)."""
    key = _mesh_key()
    path = CACHE / f"meshes_{BODY}_{key}.pkl"
    stages = {}
    if path.exists() and "--no-cache" not in sys.argv:
        stages = pickle.loads(path.read_bytes())
        print("meshes from cache", path.name, sorted(stages))
    CACHE.mkdir(exist_ok=True)

    def save():
        for old in CACHE.glob(f"meshes_{BODY}_*.pkl"):
            if old != path:
                old.unlink()
        path.write_bytes(pickle.dumps(stages))

    body_shape = B.body_sdf()
    garments = {name: make() for name, make in C.GARMENTS}
    if "outfit" not in stages:
        stages["outfit"] = {k: v[0] for k, v in build_outfit(body_shape).items()}
        save()
    head_shape = B.head_scaled()
    hair_shapes = [HR.hair_sdf(st) for st in HR.STYLES]
    if "hairs" not in stages:
        stages["hairs"] = {st: build_hair(st, head_shape)[0] for st in HR.STYLES}
        save()
    if "body" not in stages:
        stages["body"] = build_body({k: (None, g) for k, g in garments.items()}, hair_shapes)[0]
        save()
    if HARPIN and "extra" not in stages:
        hid = covered_by([body_shape] + list(garments.values()), margin=0.003)
        stages["extra"] = {"Wings": build_wings(hid), "Tail": build_tail(hid)}
        save()
    outfit, hairs, body = stages["outfit"], stages["hairs"], stages["body"]
    extra = stages.get("extra", {})
    paint_garments(outfit)
    paint_face(body)
    face = {m.name: m for m in FC.face_meshes()}
    if HARPIN:
        plumes = HP.temple_feathers()
        plumes.set_material(slot("Plume"))
        face["Head_Plumes"] = plumes
    return body, outfit, hairs, face, extra


# ----------------------------------------------------------------------------- assembly

def skin_hair(m, sk, style):
    """Hair on the head follows the Head bone; a ponytail or long hair hanging behind the back
    shifts to the Neck and then the Chest going down (so it stays behind the back)."""
    RG.skin_mesh(m, sk, rigid_bone="Head")
    if style not in ("rabo", "longo", "plumado"):
        return m
    names = [b.name for b in sk.bones]
    y = m.vertices[:, 1]
    head = RG.ss(1.47, 1.58, y)
    chest = RG.ss(1.5, 1.38, y)
    neck = np.clip(1 - head - chest, 0, 1)
    W = np.stack([head, neck, chest], axis=1)
    W /= W.sum(axis=1, keepdims=True)
    m.joints = np.zeros((m.num_vertices, 4), np.int64)
    m.weights = np.zeros((m.num_vertices, 4))
    for k, b in enumerate(("Head", "Neck", "Chest")):
        m.joints[:, k] = names.index(b)
        m.weights[:, k] = W[:, k]
    return m


def build():
    t0 = time.time()
    body, outfit, hairs, face, extra = build_meshes()

    sk = build_skeleton(final=True)
    head_sdf = B.head_scaled()
    # weights from design-space positions (rig.py's functions), then everything moves to the
    # final proportions (longer legs, the female's scale)
    RG.skin_mesh(body, sk, head_sdf)
    for name, m in outfit.items():
        if name in ("Tunic", "Underskirt"):
            RG.skin_mesh(m, sk, head_sdf, skirt_mask=m.vertices[:, 1] < 1.05)
        elif name == "Trousers":  # under the skirt the trousers move with it (no poking through)
            RG.skin_mesh(m, sk, head_sdf, skirt_mask=m.vertices[:, 1] > 0.85,
                         skirt_blend=lambda y: RG.ss(0.86, 0.91, y) * RG.ss(1.045, 0.975, y))
        else:
            RG.skin_mesh(m, sk, head_sdf)
    for style, m in hairs.items():
        skin_hair(m, sk, style)
    for m in face.values():
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

    parts = {"Body": body, "Head_Skin": head_skin, **face}
    parts.update({f"Hair_{s}": m for s, m in hairs.items()})
    parts["Outfit"] = outfit_mesh
    parts.update(extra)  # Harpins: Wings, Tail (already skinned)
    for n, m in parts.items():
        m.name = n
        xf_mesh(m)
        probs = m.validate()
        if probs:
            print("  validate", n, probs)
    scene = p3.Scene(f"Niryk_{BODY}")
    scene.add(sk.root)
    nodes = {}
    for n, m in parts.items():
        nodes[n] = scene.add(p3.skinned_node(m, sk, n))
    rig = A.Rig(sk)
    A.bake(scene, rig)
    out = export_glb(scene, BODY)
    stats = {n: m.num_triangles for n, m in parts.items()}
    stats.update({f"g_{k}": v.num_triangles for k, v in outfit.items()})
    hair_tris = [m.num_triangles for s, m in hairs.items()]
    stats["total_glb"] = sum(m.num_triangles for m in parts.values())
    stats["shown_with_one_hair"] = stats["total_glb"] - sum(hair_tris) + max(hair_tris)
    lo = np.min([m.bounds()[0] for m in parts.values()], axis=0)
    hi = np.max([m.bounds()[1] for m in parts.values()], axis=0)
    stats["height_m"] = round(float(hi[1]), 3)
    stats["height_no_hair_m"] = round(float(max(m.bounds()[1][1] for n, m in parts.items()
                                                 if not n.startswith("Hair_"))), 3)
    stats["width_m"] = round(float(hi[0] - lo[0]), 3)
    stats["bones"] = len(sk.bones)
    stats["time_s"] = round(time.time() - t0, 1)
    write_stats(BODY, stats)
    print(BODY, stats, "->", out)
    return scene, sk, nodes


def build_sword():
    import equipment as EQ
    s = p3.Scene("Sword_Iron")
    m = EQ.sword_iron()
    s.add(m)
    out = export_glb(s, "sword_iron", sub="equipment")
    print("sword_iron", m.num_triangles, "tris ->", out)
    return m


if __name__ == "__main__":
    sword = build_sword()
    scene, sk, nodes = build()
    if "--no-review" not in sys.argv:
        import review_sheets
        review_sheets.review(scene, sk, nodes, sword)
