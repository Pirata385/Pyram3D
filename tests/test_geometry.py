import numpy as np
import pytest

import pyram3d as p3
from pyram3d import sdf


def outward_fraction(m):
    c = m.face_centers() - m.center
    return float((np.einsum("ij,ij->i", c, m.face_normals()) > -1e-9).mean())


@pytest.mark.parametrize("make", [
    lambda: p3.box((1, 2, 3)), lambda: p3.rounded_box(1, 0.2), lambda: p3.uv_sphere(),
    lambda: p3.ico_sphere(), lambda: p3.cylinder(), lambda: p3.cone(), lambda: p3.capsule(),
    lambda: p3.extrude_shape(p3.shapes.star(), 0.2),
])
def test_closed_primitives_are_watertight_and_outward(make):
    m = make()
    st = m.stats()
    assert st["watertight"], st
    assert m.validate() == []
    assert outward_fraction(m) == 1.0


def test_box_size_and_uvs():
    m = p3.box((2, 1, 0.5))
    assert np.allclose(m.size, (2, 1, 0.5))
    assert m.uvs is not None and len(m.face_uvs) == len(m.faces)


def test_subdivide_catmull_clark():
    m = p3.box(1).subdivide(1)
    assert len(m.faces) == 24
    assert m.stats()["watertight"]
    r = np.linalg.norm(m.vertices, axis=1)
    assert r.max() < 0.87  # corners are pulled in by smoothing
    m.subdivide(2)
    assert len(m.faces) == 24 * 16


def test_extrude_and_inset_keep_mesh_closed():
    m = p3.box(1, segments=2)
    top = m.faces_facing((0, 1, 0))
    assert len(top) == 4
    m.extrude(top, 0.5)
    assert np.isclose(m.bounds()[1][1], 1.0)
    assert m.stats()["watertight"]
    m.inset(m.faces_facing((0, 0, 1)), 0.05)
    assert m.stats()["watertight"]
    assert m.validate() == []


def test_mirror_welds_seam():
    half = p3.box(1).translate((0.5, 0, 0))
    half.delete_faces(half.faces_facing((-1, 0, 0)))
    half.mirror("x")
    assert half.stats()["watertight"]
    assert np.allclose(half.size, (2, 1, 1))


def test_triangulate_ngons():
    m = p3.extrude_shape(p3.shapes.star(5), 0.1).triangulate()
    assert all(len(f) == 3 for f in m.faces)
    assert m.stats()["watertight"]
    assert outward_fraction(m) == 1.0


def test_deformers_and_shape_keys():
    m = p3.cylinder(0.2, 2, rings=10)
    m.add_shape_key("Fat", lambda p: p * (1.5, 1, 1.5))
    m.bend(90)
    assert "Fat" in m.shape_keys and m.shape_keys["Fat"].shape == m.vertices.shape
    m.twist(45).taper(0.5)
    assert np.isfinite(m.vertices).all()


def test_sdf_sphere_accuracy_and_booleans():
    m = sdf.sphere((0, 0, 0), 0.5).to_mesh(48)
    assert np.abs(np.linalg.norm(m.vertices, axis=1) - 0.5).max() < 1e-3
    assert m.stats()["watertight"]
    assert outward_fraction(m) == 1.0
    cut = (sdf.box(size=(1, 1, 1)) - sdf.sphere(radius=0.6)).to_mesh(48)
    assert cut.num_faces > 0 and cut.validate() == []


def test_sdf_colors_and_materials():
    red = p3.Material("Red", "#ff0000")
    blue = p3.Material("Blue", "#0000ff")
    shape = sdf.sphere((-0.5, 0, 0), 0.4, color="#ff0000", material=red).smooth_union(
        sdf.sphere((0.5, 0, 0), 0.4, color="#0000ff", material=blue), 0.1)
    m = shape.to_mesh(48)
    assert m.colors is not None and len(m.materials) == 2
    left = m.vertices[:, 0] < -0.3
    assert m.colors[left, 0].mean() > 0.9


def test_sculpt_brushes_move_vertices():
    m = p3.uv_sphere(0.5, 48, 24)
    before = m.vertices.copy()
    p3.sculpt.draw(m, (0, 0, 0.5), 0.2, 0.05)
    p3.sculpt.grab(m, (0.4, 0, 0.3), 0.2, (0.1, 0, 0), symmetry="x")
    moved = np.linalg.norm(m.vertices - before, axis=1)
    assert moved.max() > 0.04
    assert moved[(before[:, 2] < -0.3)].max() < 1e-12  # far side untouched


def test_terrain_and_scatter():
    t = p3.terrain(20, 32, 4, seed=1)
    assert t.colors is not None and t.size[1] > 0.5
    pts = p3.scatter(t, 10, seed=2)
    assert len(pts) == 10
