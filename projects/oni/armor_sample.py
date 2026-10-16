"""Armor fitting test for the Oni rig: proves gear can be built on (and skinned to) her skeleton.

Pieces: breastplate (skinned to Spine/Chest), pauldrons (rigid on the upper arms), bracers
(rigid on the forearms) and a katana parented to Socket_Weapon_R.

Outputs (projects/oni/exports/armor_sample/):
  oni_armor_sample.fbx   skeleton (same bone names) + armor meshes -> equip by bone-name remap
  oni_with_armor.png     the character wearing the set (Body_Torso hidden underneath)
Run: python projects/oni/armor_sample.py
"""
import runpy
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import pyram3d as p3  # noqa: E402
from pyram3d import sdf  # noqa: E402
from pyram3d.palette import palettize  # noqa: E402

g = runpy.run_path(str(HERE / "build.py"), run_name="lib")
S, P = g["S"], g["P"]
OUT = HERE / "exports" / "armor_sample"
ARMOR = {"steel": "#3c4356", "steel_light": "#7d8aa8", "gold": "#c9a24b", "leather": "#4a2f22",
         "cloth_red": "#8f1d2c", "blade": "#d9e0ea", "blade_edge": "#ffffff", "wrap": "#1d1a24"}


def lin(c):
    return g["lin"](c)


def breastplate():
    shell = sdf.ellipsoid(P(0, 1.29, 0.0), P(0.19, 0.17, 0.142))
    shell = shell.smooth_union(sdf.ellipsoid(P(0, 1.13, 0.0), P(0.165, 0.1, 0.128)), 0.04 * S)
    shell = shell.intersect(sdf.plane(P(0, 1.41, 0), (0, 1, -0.2))).intersect(sdf.plane(P(0, 1.05, 0), (0, -1, 0)))
    collar = sdf.torus(P(0, 1.405, -0.005), 0.115 * S, 0.018 * S)
    plate = shell.smooth_union(collar, 0.01 * S)
    m = plate.to_mesh(cell=0.015, name="Armor_Breastplate")
    V = m.vertices
    col = np.tile(lin(ARMOR["steel"]), (len(V), 1))
    col[np.abs(V[:, 1] - 1.405 * S) < 0.03 * S] = lin(ARMOR["gold"])          # collar trim
    col[np.abs(V[:, 1] - 1.07 * S) < 0.018 * S] = lin(ARMOR["gold"])          # lower rim
    col[(np.abs(V[:, 0]) < 0.012 * S) & (V[:, 2] > 0)] = lin(ARMOR["steel_light"])  # centre ridge
    m.colors = col
    return m


def pauldron(sx):
    c = P(sx * 0.2, 1.47, 0.0)
    plates = None
    for i in range(3):
        e = sdf.ellipsoid(c + P(sx * 0.02 * i, -0.035 * i, 0), P(0.095 - 0.01 * i, 0.06, 0.095 - 0.008 * i))
        e = e.intersect(sdf.plane(c + P(0, -0.035 * i - 0.01, 0), (0, -1, 0)))
        plates = e if plates is None else plates.union(e)
    m = plates.to_mesh(cell=0.01, name=f"Armor_Pauldron_{'L' if sx > 0 else 'R'}")
    V = m.vertices
    col = np.tile(lin(ARMOR["steel"]), (len(V), 1))
    edge = m.vertex_normals()[:, 1] < -0.2
    col[edge] = lin(ARMOR["gold"])
    m.colors = col
    return m


def bracer(sx):
    a, b = P(sx * 0.47, 1.44, 0), P(sx * 0.63, 1.44, 0)
    cyl = sdf.round_cone(a, b, 0.048 * S, 0.041 * S)  # solid sleeve: the forearm inside is hidden
    m = cyl.to_mesh(cell=0.01, name=f"Armor_Bracer_{'L' if sx > 0 else 'R'}")
    V = m.vertices
    col = np.tile(lin(ARMOR["leather"]), (len(V), 1))
    col[np.abs(np.abs(V[:, 0]) - 0.55 * S) < 0.012 * S] = lin(ARMOR["gold"])
    m.colors = col
    return m


def katana():
    """Blade along +Z from the grip (socket origin = middle of the grip)."""
    blade_path = [(0, 0, 0.12), (0, 0.01, 0.5), (0, 0.04, 0.82)]
    blade = p3.sweep(p3.bezier(blade_path, 16), shape=[(-1, 0), (0, 0.25), (1, 0), (0, -0.25)],
                     radius=lambda t: 0.017 * (1 - 0.6 * t ** 4), up=(1, 0, 0), name="Blade")
    blade.colors = np.tile(lin(ARMOR["blade"]), (len(blade.vertices), 1))
    guard = p3.extrude_shape(p3.shapes.rounded_rect(0.075, 0.065, 0.02), 0.012).translate((0, 0, 0.115))
    guard.colors = np.tile(lin(ARMOR["gold"]), (len(guard.vertices), 1))
    grip = p3.cylinder(0.014, 0.23, 10).rotate((90, 0, 0))
    grip.colors = np.tile(lin(ARMOR["wrap"]), (len(grip.vertices), 1))
    kat = p3.Mesh.join([blade, guard, grip], name="Katana")
    kat.smooth_angle = 35.0
    return kat


if __name__ == "__main__":
    scene, sk, info = g["build"]()
    body_bones = lambda names: (lambda Pp, i: np.full(len(Pp), sk.bones[i].name in names))  # noqa: E731
    pieces = []
    bp = breastplate()
    sk.bind(bp, smooth=4, bone_filter=body_bones({"Spine", "Chest", "Hips"}))
    pieces.append(bp)
    for sx, side in ((1, "Left"), (-1, "Right")):
        pd = pauldron(sx)
        sk.bind(pd, rigid=True, bone_filter=body_bones({f"{side}UpperArm"}))
        br = bracer(sx)
        sk.bind(br, rigid=True, bone_filter=body_bones({f"{side}LowerArm"}))
        pieces += [pd, br]
    kat = katana()
    palettize(pieces + [kat], palette=list(ARMOR.values()), name="Armor_Palette", roughness=0.4)
    for m in pieces + [kat]:
        m.materials[0].metallic = 0.6
    nodes = [scene.add(p3.skinned_node(m, sk, m.name)) for m in pieces]
    socket = sk.root.find("Socket_Weapon_R")
    kat_node = socket.add(kat, name="Katana")
    # hide the body part covered by the breastplate (what a game does when armor is equipped)
    scene.find("Body_Torso").visible = False

    OUT.mkdir(parents=True, exist_ok=True)
    toon = dict(style="toon", exposure=0.8, bloom=0.12, rim=0.4)
    from pyram3d.image import grid, write_png
    shots = [p3.render(scene, None, 380, 520, view=v, **toon) for v in ("front", "three_quarter")]
    shots.append(p3.render(scene, None, 380, 520, view="three_quarter", time=0.25, animation="Run", **toon))
    shots.append(p3.render(scene, None, 380, 520, view=(0.8, 1.4, 1.0), time=0.5, animation="Walk", pixel=4,
                           yfov=14, background="#5d7a3a", target=(0, 0.85, 0), distance=7.5,
                           **{k: v for k, v in toon.items()}))
    write_png(OUT / "oni_with_armor.png", grid(shots, 4))

    # armor-only file: same skeleton (bone names), armor meshes, katana under its socket
    armor_scene = p3.Scene("OniArmorSample")
    armor_scene.add(sk)
    for nd in nodes:
        armor_scene.add(nd)
    for nd in list(armor_scene.walk()):
        if nd.mesh is not None and nd.skin is None and nd.name != "Katana":
            nd.visible = False
            nd.parent.remove(nd)   # drop the character's rigid pieces (horns, eyes, ornament)
    armor_scene.export(OUT / "oni_armor_sample.fbx")
    armor_scene.export(OUT / "oni_armor_sample.glb")
    print("armor:", {m.name: m.num_triangles for m in pieces + [kat]}, "->", OUT)
