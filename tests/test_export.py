import json
import struct

import numpy as np

import pyram3d as p3
from fbx_reader import child, find, read_fbx


def build_scene():
    s = p3.Scene("T")
    wood = p3.presets.wood()
    b = p3.box(1).set_material(wood)
    n = s.add(b, position=(0, 0.5, 0))
    s.add(p3.uv_sphere(0.3).set_material(p3.presets.gold()), position=(0, 1, 0), parent=n)
    s.add_default_lights()
    s.add_camera()
    p3.spin(s.animation("Spin"), n)
    return s


def test_glb_structure_and_roundtrip(tmp_path):
    s = build_scene()
    path = s.export(tmp_path / "t.glb")
    data = path.read_bytes()
    magic, ver, length = struct.unpack("<4sII", data[:12])
    assert magic == b"glTF" and ver == 2 and length == len(data)
    js_len = struct.unpack("<I", data[12:16])[0]
    doc = json.loads(data[20:20 + js_len])
    assert doc["asset"]["version"] == "2.0"
    assert len(doc["animations"]) == 1
    assert "KHR_lights_punctual" in doc["extensionsUsed"]
    back = p3.load(path)
    meshes = back.mesh_nodes()
    assert len(meshes) == 2
    assert meshes[0].mesh.materials[0].base_color_texture is not None
    assert len(back.animations) == 1


def test_gltf_embedded_json(tmp_path):
    path = build_scene().export(tmp_path / "t.gltf")
    doc = json.loads(path.read_text())
    assert doc["buffers"][0]["uri"].startswith("data:")
    assert p3.load(path).find("Box") is not None


def test_fbx_binary_structure(tmp_path):
    s = build_scene()
    path = s.export(tmp_path / "t.fbx")
    version, nodes, data = read_fbx(path)
    assert version == 7400
    names = [n["name"] for n in nodes]
    for required in ("FBXHeaderExtension", "GlobalSettings", "Definitions", "Objects", "Connections"):
        assert required in names
    objects = find(nodes, "Objects")[0]["children"]
    geoms = [o for o in objects if o["name"] == "Geometry"]
    models = [o for o in objects if o["name"] == "Model"]
    assert len(geoms) == 2
    assert {m["props"][2] for m in models} >= {b"Mesh", b"Light", b"Camera"}
    verts = child(geoms[0], "Vertices")["props"][0]
    pvi = child(geoms[0], "PolygonVertexIndex")["props"][0]
    assert len(verts) % 3 == 0 and (pvi < 0).sum() == 6  # cube: 6 quads
    conns = find(nodes, "Connections")[0]["children"]
    ids = {o["props"][0] for o in objects} | {0}
    for c in conns:  # every connection references existing objects
        assert c["props"][1] in ids and c["props"][2] in ids
    assert data[-16:] == b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
    stacks = [o for o in objects if o["name"] == "AnimationStack"]
    assert len(stacks) == 1


def test_fbx_skinned_character(tmp_path):
    s = p3.Scene("C")
    hero = p3.character.humanoid(resolution=48, eyes=False)
    hero.add_to(s)
    p3.character.walk_cycle(s.animation("Walk"), hero.skeleton)
    path = s.export(tmp_path / "c.fbx")
    _, nodes, _ = read_fbx(path)
    objects = find(nodes, "Objects")[0]["children"]
    kinds = [o["props"][2] for o in objects if o["name"] == "Deformer"]
    assert kinds.count(b"Skin") == 1 and kinds.count(b"Cluster") == len(hero.skeleton.bones)
    limbs = [o for o in objects if o["name"] == "Model" and o["props"][2] == b"LimbNode"]
    assert len(limbs) == len(hero.skeleton.bones)
    assert any(o["name"] == "Pose" for o in objects)


def test_obj_export(tmp_path):
    path = build_scene().export(tmp_path / "t.obj")
    text = path.read_text()
    assert text.count("\nv ") > 8 and "usemtl" in text
    assert (tmp_path / "t.mtl").exists()
