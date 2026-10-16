"""Tests for game-character tooling: local subdivision, palettes, skirts, modular split, sockets."""
import numpy as np

import pyram3d as p3
from pyram3d import sdf
from pyram3d.palette import palettize


def test_subdivide_region_is_crack_free_and_projectable():
    shape = sdf.sphere((0, 0, 0), 0.5)
    m = shape.to_mesh(24)
    top = m.faces_where(lambda c, n: c[:, 1] > 0.2)
    n0 = m.num_faces
    m.subdivide_region(top)
    shape.project(m)
    assert m.num_faces > n0 + 3 * len(top) - 1
    st = m.stats()
    assert st["watertight"] and st["non_manifold_edges"] == 0
    assert np.abs(np.linalg.norm(m.vertices, axis=1) - 0.5).max() < 1e-6


def test_sdf_cell_size_controls_density():
    small = sdf.sphere((0, 0, 0), 0.1)
    a = small.to_mesh(cell=0.02)
    b = small.to_mesh(cell=0.01)
    assert b.num_faces > 3 * a.num_faces


def test_palettize_flat_face_colors_and_pow2_texture():
    m = p3.uv_sphere(0.5, 16, 8)
    m.set_color("#ff0000")
    m.set_color("#0000ff", verts=np.flatnonzero(m.vertices[:, 1] > 0))
    mat, pal = palettize([m], palette=["#ff0000", "#0000ff", "#00ff00"])
    img = mat.base_color_texture.image
    assert img.shape[0] == img.shape[1] and (img.shape[0] & (img.shape[0] - 1)) == 0
    assert m.colors is None and m.uvs is not None
    # every face uses a single swatch (all corners share one uv)
    assert all(len(set(f)) == 1 for f in m.face_uvs)
    samples = mat.base_color_texture.sample(m.uvs[[f[0] for f in m.face_uvs]])
    top = m.face_centers()[:, 1] > 0.1
    assert samples[top, 2].mean() > 0.9 and samples[~top & (m.face_centers()[:, 1] < -0.1), 0].mean() > 0.9


def test_skirt_weights_follow_both_legs():
    sk = p3.character.humanoid_skeleton(1.8)
    skirt = p3.cone(0.3, 0.8, 24).translate((0, 0.5, 0))
    sk.bind_skirt(skirt)
    w = np.zeros((len(skirt.vertices), len(sk.bones)))
    for k in range(4):
        np.add.at(w, (np.arange(len(w)), skirt.joints[:, k]), skirt.weights[:, k])
    left, right = sk.index("LeftUpperLeg"), sk.index("RightUpperLeg")
    hem_left = (skirt.vertices[:, 1] < 0.2) & (skirt.vertices[:, 0] > 0.2)
    hem_right = (skirt.vertices[:, 1] < 0.2) & (skirt.vertices[:, 0] < -0.2)
    assert w[hem_left, left].min() > 0.5 and w[hem_right, right].min() > 0.5
    assert np.allclose(w.sum(axis=1), 1)


def test_split_mesh_and_sockets():
    hero = p3.character.humanoid(resolution=56, eyes=False)
    hero.mesh.freeze_normals()
    parts = hero.skeleton.split_mesh(hero.mesh, {
        "Torso": ["Hips", "Spine", "Chest", "LeftShoulder", "RightShoulder", "Neck", "Head"],
        "Arms": ["LeftUpperArm", "RightUpperArm", "LeftLowerArm", "RightLowerArm", "LeftHand", "RightHand"],
        "Legs": ["LeftUpperLeg", "RightUpperLeg", "LeftLowerLeg", "RightLowerLeg", "LeftFoot", "RightFoot",
                 "LeftToes", "RightToes"]})
    assert set(parts) == {"Torso", "Arms", "Legs"}
    assert sum(p.num_faces for p in parts.values()) == hero.mesh.num_faces
    for p in parts.values():
        assert p.joints is not None and p.normals is not None and len(p.normals) == len(p.vertices)
    sock = hero.skeleton.add_socket("Socket_Weapon_R", "RightHand", (-0.78, 1.45, 0))
    hero.skeleton.bone("RightUpperArm").set_rotation((0, 0, -60))  # right arm: -Z raises it
    assert sock.world_position()[1] > 1.6  # the socket follows the raised arm


def test_transfer_weights_follows_source():
    sk = p3.Skeleton("Rig")
    sk.add_bone("Root", (0, 0, 0), (0, 0.5, 0))
    sk.add_bone("Top", (0, 0.5, 0), (0, 1, 0), parent="Root")
    body = p3.cylinder(0.1, 1.0, rings=10).translate((0, 0.5, 0))
    sk.bind(body, rigid=True)
    sleeve = p3.cylinder(0.13, 0.9, rings=8).translate((0, 0.5, 0))
    sk.transfer_weights(sleeve, body)
    top = sleeve.vertices[:, 1] > 0.8
    bottom = sleeve.vertices[:, 1] < 0.2
    dom = sleeve.joints[np.arange(len(sleeve.joints)), np.argmax(sleeve.weights, axis=1)]
    assert (dom[top] == sk.index("Top")).all()
    assert (dom[bottom] == sk.index("Root")).all()
    assert np.allclose(sleeve.weights.sum(axis=1), 1.0)
