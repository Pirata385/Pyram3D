"""Face parts that are separate meshes: eyeballs (white, iris, pupil) and eyebrows.

Eyes: a white ball in the socket under the upper lid, with the iris and pupil as thin spherical
caps on its front (material slots Eye for the iris tint, EyeWhite and Dark untinted). Brows: a
flattened tapering sweep resting on the brow ridge (Hair slot, so they follow the hair color).
All of them are rigid on the Head bone and hidden in first person (node names start with Head_).
"""
from __future__ import annotations

import math

import numpy as np

import body as B
from common import slot

import pyram3d as p3
from pyram3d import math3d as m3


def sphere_cap(center, radius, axis, half_angle, rings=4, segments=20, name="Cap") -> p3.Mesh:
    """Spherical cap around `axis` (outward), from the pole to `half_angle` degrees."""
    axis = m3.normalize(m3.vec3(axis))
    t = np.cross(axis, [0, 1, 0])
    if np.linalg.norm(t) < 1e-6:
        t = np.cross(axis, [1, 0, 0])
    t = m3.normalize(t)
    b = np.cross(axis, t)
    verts = [m3.vec3(center) + axis * radius]
    for i in range(1, rings + 1):
        a = math.radians(half_angle) * i / rings
        for j in range(segments):
            ph = 2 * math.pi * j / segments
            d = axis * math.cos(a) + (t * math.cos(ph) + b * math.sin(ph)) * math.sin(a)
            verts.append(m3.vec3(center) + d * radius)
    faces = []
    for j in range(segments):
        faces.append((0, 1 + (j + 1) % segments, 1 + j))
    for i in range(rings - 1):
        r0, r1 = 1 + i * segments, 1 + (i + 1) * segments
        for j in range(segments):
            j1 = (j + 1) % segments
            faces.append((r0 + j, r0 + j1, r1 + j1, r1 + j))
    m = p3.Mesh(np.array(verts), faces, name=name)
    # consistent outward winding
    if np.dot(m.face_normals()[0], axis) < 0:
        m.flip()
    m.normals = m3.normalize(m.vertices - m3.vec3(center))
    return m


def eyes() -> p3.Mesh:
    parts = []
    for sx in (1, -1):
        c = B.head_xf(B.EYE_C * np.array([sx, 1, 1]))
        R = B.EYE_R * B.HEAD_SCALE
        gaze = m3.normalize(np.array([0.0, -0.05, 1.0]))
        white = p3.uv_sphere(R, 20, 12).translate(c).set_material(slot("EyeWhite"))
        white.normals = m3.normalize(white.vertices - c)
        iris = sphere_cap(c, R * 1.012, gaze, 38, rings=4, segments=20, name="Iris")
        iris.set_material(slot("Eye"))
        # painted iris: dark limbal ring, lighter toward the bottom
        rel = iris.vertices - c
        ang = np.degrees(np.arccos(np.clip(m3.normalize(rel) @ gaze, -1, 1)))
        lum = 1.0 - 0.45 * np.clip((ang - 26) / 12, 0, 1)
        lum *= 1.0 + 0.15 * np.clip(-(rel[:, 1]) / (R * 0.5), 0, 1)
        iris.colors = np.ones((iris.num_vertices, 4))
        iris.colors[:, :3] = np.clip(lum, 0, 1.1)[:, None]
        pupil = sphere_cap(c, R * 1.02, gaze, 16, rings=2, segments=16, name="Pupil")
        pupil.set_material(slot("Dark"))
        for m in (white, pupil):
            m.colors = np.ones((m.num_vertices, 4))
        parts += [white, iris, pupil]
    out = p3.Mesh.join(parts, name="Head_Eyes")
    return out


def brows(head_sdf) -> p3.Mesh:
    """Thick stylized brows on the brow ridge: a flattened sweep from the glabella outward,
    rising a little and tapering, resting 1.5 mm off the skin."""
    meshes = []
    for sx in (1, -1):
        pts = []
        for u in np.linspace(0, 1, 7):
            x = 0.011 + 0.056 * u
            y = 1.6885 + 0.006 * math.sin(u * math.pi * 0.8) - 0.002 * u
            pts.append(B.head_xf([sx * x, y, 0.12]))
        P = np.array(pts)
        # drop each point onto the skin along -Z, then lift it off the surface a little
        for _ in range(30):
            d = head_sdf.dist(P)
            P[:, 2] -= np.clip(d, -0.004, 0.004)
        n = head_sdf.gradient(P)
        P = P + n * 0.0015
        path = m3.catmull_rom(P, 3)
        shape = [(-1, -0.35), (-0.5, -0.5), (0.5, -0.5), (1, -0.35), (1, 0.35), (0.5, 0.5), (-0.5, 0.5), (-1, 0.35)]
        m = p3.sweep(path, shape=[(a * 0.0034, b * 0.0072) for a, b in shape],
                     radius=lambda t: 1.0 - 0.45 * t ** 1.5, segments=8, up=(0, 0, 1))
        m.uvs = m.face_uvs = None
        m.subdivide(1)
        m.smooth_angle = 180.0
        meshes.append(m)
    out = p3.Mesh.join(meshes, name="Head_Brows").set_material(slot("Hair"))
    out.colors = np.ones((out.num_vertices, 4))
    return out
