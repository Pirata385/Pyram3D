import numpy as np

import pyram3d as p3
from pyram3d import math3d as m3


def test_euler_quaternion_roundtrip():
    for e in [(10, 20, 30), (-45, 80, 5), (90, 0, 0), (0, 0, 179)]:
        q = m3.quat_from_euler(e)
        back = m3.quat_from_euler(m3.quat_to_euler(q))
        assert abs(abs(np.dot(q, back)) - 1) < 1e-9


def test_look_at_points_minus_z():
    s = p3.Scene()
    n = s.add_camera(position=(0, 0, 5), target=(0, 0, 0))
    fwd = -n.world_matrix()[:3, 2]
    assert np.allclose(fwd, (0, 0, -1), atol=1e-9)


def test_hierarchy_world_matrix():
    s = p3.Scene()
    a = s.add(None, "A", position=(1, 0, 0), rotation=(0, 90, 0))
    b = a.add(None, "B", position=(0, 0, 1))
    assert np.allclose(b.world_position(), (2, 0, 0), atol=1e-9)


def test_animation_sampling_and_smooth():
    s = p3.Scene()
    n = s.add(p3.box(1))
    a = s.animation("Move")
    a.keys(n, "position", {0: (0, 0, 0), 1: (2, 0, 0)})
    assert np.allclose(a.channels[0].sample(0.5), (1, 0, 0))
    a.keys(n, "rotation", {0: (0, 0, 0), 1: (0, 180, 0), 2: (0, 360, 0)}, "SMOOTH")
    t, v = a.channel(n, "rotation").baked(30)
    assert len(t) > 30 and np.allclose(np.linalg.norm(v, axis=1), 1)


def test_skinning_rest_pose_is_identity():
    hero = p3.character.humanoid(resolution=64, eyes=False)
    posed = hero.skeleton.pose_vertices(hero.mesh)
    assert np.abs(posed - hero.mesh.vertices).max() < 1e-9
    assert np.allclose(hero.mesh.weights.sum(axis=1), 1)
    hero.skeleton.bone("LeftUpperArm").set_rotation((0, 0, -60))
    moved = np.linalg.norm(hero.skeleton.pose_vertices(hero.mesh) - hero.mesh.vertices, axis=1)
    left_hand = hero.mesh.vertices[:, 0] > 0.6
    assert moved[left_hand].min() > 0.2           # the hand follows the arm
    assert moved[hero.mesh.vertices[:, 0] < -0.5].max() < 1e-6  # right arm unaffected


def test_render_produces_image(tmp_path):
    s = p3.Scene()
    s.add(p3.uv_sphere(0.5).set_material(p3.presets.plastic()))
    img = s.render(tmp_path / "a.png", 64, 48)
    assert img.shape == (48, 64, 3)
    assert (tmp_path / "a.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    center = img[24, 32].astype(int)
    corner = img[0, 0].astype(int)
    assert abs(center - corner).sum() > 30  # the sphere is visible


def test_png_roundtrip():
    from pyram3d.image import decode_png, encode_png
    img = (np.random.default_rng(0).random((17, 23, 4)) * 255).astype(np.uint8)
    assert np.array_equal(decode_png(encode_png(img)), img)


def test_save_helper(tmp_path):
    s = p3.Scene("Thing")
    s.add(p3.box(1))
    out = p3.save(s, tmp_path / "thing", size=300, views=False)
    for k in ("glb", "gltf", "fbx", "preview", "viewer"):
        assert out[k].exists()
