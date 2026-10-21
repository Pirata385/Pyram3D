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


def _lid_edge(c, R, up, n, azimuths):
    """Points where the lid plane (through c + up * Y, normal n) cuts the sphere (c, R), at the
    given azimuths (degrees around Y from +Z; positive toward the outer corner of the left eye)."""
    n = m3.normalize(np.asarray(n, float))
    out = []
    for a in np.radians(azimuths):
        lo, hi = -1.2, 1.2
        for _ in range(50):  # bisection on the elevation
            e = 0.5 * (lo + hi)
            d = np.array([math.sin(a) * math.cos(e), math.sin(e), math.cos(a) * math.cos(e)])
            f = np.dot(n, R * d - np.array([0.0, up, 0.0]))
            if f > 0:
                lo = e
            else:
                hi = e
        out.append(c + R * d)
    return np.array(out)


def eyes() -> p3.Mesh:
    """Eyeballs (front cap of a ball: the back is hidden in the socket), big painted irises with
    a dark limbal ring, pupils, a catchlight, and a dark lash line along each upper lid."""
    parts = []
    for sx in (1, -1):
        c = B.head_xf(B.EYE_C * np.array([sx, 1, 1]))
        R = B.EYE_R * B.HEAD_SCALE
        gaze = m3.normalize(B.EYE_GAZE)
        white = sphere_cap(c, R, (sx * 0.35, 0, 1), 108, rings=5, segments=16, name="White")
        white.set_material(slot("EyeWhite"))
        # a little shade under the upper lid
        rel = white.vertices - c
        white.colors = np.ones((white.num_vertices, 4))
        white.colors[:, :3] = (1 - 0.18 * np.clip((rel[:, 1] / R - 0.05) / 0.4, 0, 1))[:, None]
        iris = sphere_cap(c, R * 1.012, gaze, 41, rings=3, segments=16, name="Iris")
        iris.set_material(slot("Eye"))
        rel = iris.vertices - c
        ang = np.degrees(np.arccos(np.clip(m3.normalize(rel) @ gaze, -1, 1)))
        lum = 1.0 - 0.6 * np.clip((ang - 30) / 9, 0, 1)  # limbal ring
        lum *= 1.0 + 0.25 * np.clip(-(rel[:, 1]) / (R * 0.5), 0, 1)  # lighter low down
        lum *= 1.0 - 0.3 * np.clip(rel[:, 1] / (R * 0.45), 0, 1)  # lid shadow on top
        iris.colors = np.ones((iris.num_vertices, 4))
        iris.colors[:, :3] = np.clip(lum, 0, 1.25)[:, None]
        pupil = sphere_cap(c, R * 1.02, gaze, 17, rings=2, segments=12, name="Pupil")
        pupil.set_material(slot("Dark"))
        glint = sphere_cap(c, R * 1.026, m3.normalize(gaze + np.array([sx * 0.22, 0.26, 0])), 6.5, rings=1,
                           segments=8, name="Glint")
        glint.set_material(slot("EyeWhite"))
        for m in (pupil, glint):
            m.colors = np.ones((m.num_vertices, 4))
        parts += [white, iris, pupil, glint, lash_line(sx)]
    return p3.Mesh.join(parts, name="Head_Eyes")


def lash_line(sx) -> p3.Mesh:
    """Dark upper lash line just in front of the lid edge, thicker toward the outer corner, with
    a short flick at the end (Dark slot)."""
    Rl = B.EYE_R + 0.0021
    pt, n = B.lid_plane(True)
    edge = _lid_edge(B.EYE_C, Rl + 0.0006, pt[1] - B.EYE_C[1], n, np.linspace(-58, 66, 9))
    flick = edge[-1] + np.array([0.0045, -0.0012, -0.0028])
    P = np.vstack([edge, flick[None]])
    P[:, 0] *= sx
    P = np.array([B.head_xf(p) for p in P])
    path = m3.catmull_rom(P, 2)
    m = p3.sweep(path, radius=lambda t: 0.0007 + 0.0011 * math.sin(math.pi * min(t * 1.2, 1.0)) ** 0.7,
                 segments=4)
    m.uvs = m.face_uvs = None
    m.smooth_angle = 180.0
    m.set_material(slot("Dark"))
    m.colors = np.ones((m.num_vertices, 4))
    return m


def brows(head_sdf) -> p3.Mesh:
    """Thick angular brows on the brow ridge: a flattened sweep from the glabella outward, low
    and heavy at the inner end, a gentle peak two thirds out, tapering down at the temple,
    resting 1.5 mm off the skin."""
    meshes = []
    for sx in (1, -1):
        pts = []
        for u in np.linspace(0, 1, 8):
            x = 0.01 + 0.058 * u
            y = 1.6875 + 0.0065 * math.sin(min(u / 0.68, 1.0) * math.pi * 0.5) - 0.007 * max(0.0, (u - 0.68) / 0.32) ** 1.4
            pts.append(B.head_xf([sx * x, y, 0.12]))
        P = np.array(pts)
        # drop each point onto the skin along -Z, then lift it off the surface a little
        for _ in range(30):
            d = head_sdf.dist(P)
            P[:, 2] -= np.clip(d, -0.004, 0.004)
        n = head_sdf.gradient(P)
        P = P + n * 0.0015
        path = m3.catmull_rom(P, 2)
        shape = [(-1, -0.4), (-0.55, -0.8), (0.55, -0.8), (1, -0.4), (1, 0.4), (0.55, 0.8), (-0.55, 0.8), (-1, 0.4)]
        m = p3.sweep(path, shape=[(a * 0.0034, b * 0.0072) for a, b in shape],
                     radius=lambda t: (1.08 - 0.5 * t ** 1.6) * (0.75 + 0.25 * min(t / 0.08, 1.0)), segments=8,
                     up=(0, 0, 1))
        m.uvs = m.face_uvs = None
        m.smooth_angle = 180.0
        meshes.append(m)
    out = p3.Mesh.join(meshes, name="Head_Brows").set_material(slot("Hair"))
    out.colors = np.ones((out.num_vertices, 4))
    return out
