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


def test_render_alpha_background(tmp_path):
    s = p3.Scene()
    s.add(p3.uv_sphere(0.5).set_material(p3.presets.plastic()))
    for pixel in (1, 4):
        img = p3.render(s, tmp_path / f"a{pixel}.png", 64, 48, style="toon", alpha=True, pixel=pixel)
        assert img.shape == (48, 64, 4)
        assert img[0, 0, 3] == 0  # background is transparent
        assert img[24, 32, 3] == 255  # the sphere is opaque
        assert img[24, 32, :3].sum() > 30
        from pyram3d.image import decode_png
        assert decode_png((tmp_path / f"a{pixel}.png").read_bytes()).shape == (48, 64, 4)


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


def test_rigid_bodies_settle_on_ground():
    s = p3.Scene()
    w = p3.physics.RigidWorld()
    n = s.add(p3.box(0.4), position=(0, 2, 0), rotation=(30, 20, 10))
    w.add(n, "box")
    a = s.animation("Fall")
    w.simulate(a, 2.0, fps=30, substeps=8)
    final = a.channel(n, "position").values[-1]
    assert 0.15 < final[1] < 0.35  # resting on the ground (half size 0.2, possibly tilted)


def test_cloth_drape_rests_on_collider():
    from pyram3d import sdf
    cloth = p3.plane(1.0, 12).translate((0, 1, 0))
    out = p3.physics.cloth_drape(cloth, [sdf.box((0, 0.4, 0), (0.6, 0.2, 0.6))], steps=120)
    center = np.argmin(np.linalg.norm(out.vertices[:, [0, 2]], axis=1))
    assert 0.49 < out.vertices[center, 1] < 0.56
    assert out.vertices[:, 1].min() < 0.45  # edges hang down


def test_particles_loop(tmp_path):
    s = p3.Scene()
    a = s.animation("FX")
    g = p3.physics.particles(s, a, "sparks", count=5, duration=1.0)
    assert len(g.children) == 5
    for c in a.channels:
        assert np.allclose(c.values[0], c.values[-1], atol=1e-6)  # seamless loop
    s.export(tmp_path / "fx.glb")


def test_sdf_bone_tags_give_exact_skinning():
    from pyram3d import sdf
    body = sdf.box((0, 1, 0), (0.4, 0.6, 0.3), bone="Body")
    arm = sdf.capsule((0.25, 1.2, 0), (0.6, 1.2, 0), 0.05, bone="Arm")
    mesh = body.smooth_union(arm, 0.02).to_mesh(64)
    sk = p3.Skeleton()
    b = sk.add_bone("Body", (0, 0.7, 0), (0, 1.3, 0))
    sk.add_bone("Arm", (0.25, 1.2, 0), (0.6, 1.2, 0), b)
    sk.bind(mesh, rigid=True)
    sk.bone("Arm").set_rotation((0, 0, 45))
    moved = np.linalg.norm(sk.pose_vertices(mesh) - mesh.vertices, axis=1)
    torso = (np.abs(mesh.vertices[:, 0]) < 0.15)
    assert moved[torso].max() < 1e-9          # body untouched
    assert moved[mesh.vertices[:, 0] > 0.5].min() > 0.1  # arm tip follows the bone
