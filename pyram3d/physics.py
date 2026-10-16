"""Simple physics simulations baked into keyframes / shape keys, so they play in any engine.

  RigidWorld   falling, bouncing, tumbling spheres and boxes (ground + sphere contacts)
  cloth_drape  drop a cloth onto colliders (static result: tablecloths, flags at rest, blankets)
  cloth_simulate  the same, baked as an animation (shape keys + weight keys)
  particles    looping effects: fire, smoke, sparks, fountain, snow, rain, magic

All units are meters / seconds. Results are regular nodes, keyframes and shape keys,
which export to glTF and FBX.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import math3d as m3
from .material import Material
from .mesh import Mesh
from .primitives import ico_sphere, plane
from .scene import Node


# ---------------------------------------------------------------------------
# rigid bodies
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class Body:
    node: Node
    shape: str
    half: np.ndarray            # box half extents (or radius in [0])
    mass: float
    vel: np.ndarray
    ang: np.ndarray             # angular velocity (rad/s, world)
    fixed: bool = False
    inv_inertia: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self.half)) if self.shape == "box" else float(self.half[0])


class RigidWorld:
    """Minimal rigid-body solver (impulses + position correction).

        world = p3.physics.RigidWorld()
        for i in range(10):
            n = scene.add(p3.box(0.3), position=(0, 1 + i * 0.4, 0), rotation=(i * 20, i * 7, 0))
            world.add(n, "box")
        world.simulate(scene.animation("Fall"), duration=3)
    """

    def __init__(self, gravity: float = -9.81, ground: float | None = 0.0, restitution: float = 0.35,
                 friction: float = 0.5):
        self.g = np.array([0.0, gravity, 0.0])
        self.ground = ground
        self.e = restitution
        self.mu = friction
        self.bodies: list[Body] = []

    def add(self, node: Node, shape: str = "auto", mass: float = 1.0, velocity=(0, 0, 0),
            angular_velocity=(0, 0, 0), fixed: bool = False, size=None) -> Body:
        """shape: "sphere" | "box" | "auto" (from the node's mesh bounds)."""
        mesh = node.mesh
        if size is not None:
            half = m3.vec3(size) / 2
        elif mesh is not None and len(mesh.vertices):
            half = mesh.size * node.scale / 2
        else:
            half = np.full(3, 0.5)
        if shape == "auto":
            shape = "sphere" if np.ptp(half) < 0.05 * half.max() and mesh is not None and \
                mesh.smooth_angle >= 179 else "box"
        if shape == "sphere":
            r = float(half.max())
            half = np.array([r, r, r])
            I = 0.4 * mass * r * r * np.ones(3)
        else:
            w, h, d = half * 2
            I = mass / 12.0 * np.array([h * h + d * d, w * w + d * d, w * w + h * h])
        b = Body(node, shape, half, mass, m3.vec3(velocity), np.radians(m3.vec3(angular_velocity)), fixed,
                 np.diag(1.0 / np.maximum(I, 1e-9)))
        self.bodies.append(b)
        return b

    def _contact(self, b: Body, p: np.ndarray, n: np.ndarray, depth: float):
        """Resolve a contact between body b at world point p with normal n (pointing out of the obstacle)."""
        x = b.node.translation
        r = p - x
        R = m3.quat_to_matrix3(b.node.rotation)
        Iinv = R @ b.inv_inertia @ R.T
        v = b.vel + np.cross(b.ang, r)
        vn = float(v @ n)
        if vn < 0:
            rn = np.cross(r, n)
            k = 1.0 / b.mass + float(n @ np.cross(Iinv @ rn, r))
            j = -(1 + self.e) * vn / k
            b.vel = b.vel + n * j / b.mass
            b.ang = b.ang + Iinv @ np.cross(r, n * j)
            # friction
            v = b.vel + np.cross(b.ang, r)
            vt = v - n * (v @ n)
            vtl = np.linalg.norm(vt)
            if vtl > 1e-6:
                t = vt / vtl
                rt = np.cross(r, t)
                kt = 1.0 / b.mass + float(t @ np.cross(Iinv @ rt, r))
                jt = min(vtl / kt, self.mu * j)
                b.vel = b.vel - t * jt / b.mass
                b.ang = b.ang - Iinv @ np.cross(r, t * jt)
        b.node.translation = b.node.translation + n * max(depth - 1e-4, 0) * 0.8

    def step(self, dt: float):
        for b in self.bodies:
            if b.fixed:
                continue
            b.vel = b.vel + self.g * dt
            b.node.translation = b.node.translation + b.vel * dt
            w = b.ang
            wl = np.linalg.norm(w)
            if wl > 1e-9:
                dq = m3.quat_from_axis_angle(w / wl, math.degrees(wl * dt))
                b.node.rotation = m3.quat_normalize(m3.quat_mul(dq, b.node.rotation))
            b.ang = b.ang * (1 - 0.02 * dt)  # tiny damping
        # ground
        if self.ground is not None:
            n = np.array([0.0, 1.0, 0.0])
            for b in self.bodies:
                if b.fixed:
                    continue
                if b.shape == "sphere":
                    depth = self.ground - (b.node.translation[1] - b.half[0])
                    if depth > 0:
                        self._contact(b, b.node.translation - n * b.half[0], n, depth)
                        # rolling: tie angular velocity to linear velocity on contact
                        b.ang = b.ang * 0.9 + 0.1 * np.cross(n, b.vel) / b.half[0]
                else:
                    corners = _box_corners(b)
                    pen = self.ground - corners[:, 1]
                    hit = pen > 0
                    if hit.any():
                        for c in corners[hit]:
                            self._contact(b, c, n, float(self.ground - c[1]))
        # body-body (bounding spheres)
        nb = len(self.bodies)
        for i in range(nb):
            a = self.bodies[i]
            for j in range(i + 1, nb):
                c = self.bodies[j]
                d = c.node.translation - a.node.translation
                dist = float(np.linalg.norm(d))
                ra = a.half[0] if a.shape == "sphere" else float(a.half.min()) * 1.1
                rc = c.half[0] if c.shape == "sphere" else float(c.half.min()) * 1.1
                if dist < ra + rc and dist > 1e-9:
                    n = d / dist
                    depth = ra + rc - dist
                    rv = c.vel - a.vel
                    vn = float(rv @ n)
                    inv_a = 0 if a.fixed else 1 / a.mass
                    inv_c = 0 if c.fixed else 1 / c.mass
                    if inv_a + inv_c == 0:
                        continue
                    if vn < 0:
                        jimp = -(1 + self.e) * vn / (inv_a + inv_c)
                        a.vel = a.vel - n * jimp * inv_a
                        c.vel = c.vel + n * jimp * inv_c
                    corr = n * depth / (inv_a + inv_c) * 0.8
                    a.node.translation = a.node.translation - corr * inv_a
                    c.node.translation = c.node.translation + corr * inv_c

    def simulate(self, anim, duration: float = 3.0, fps: int = 30, substeps: int = 10) -> "RigidWorld":
        """Run the simulation and bake position/rotation keys into `anim` (start pose at t=0)."""
        dt = 1.0 / (fps * substeps)
        frames = int(round(duration * fps))
        for f in range(frames + 1):
            t = f / fps
            for b in self.bodies:
                if not b.fixed:
                    anim.key(b.node, "position", t, b.node.translation.copy())
                    anim.key(b.node, "rotation", t, b.node.rotation.copy())
            if f < frames:
                for _ in range(substeps):
                    self.step(dt)
        anim.loop = False
        # restore start pose on the nodes (the animation holds the motion)
        for c in anim.channels:
            if c.times:
                setattr(c.node, c.path, c.values[0].copy())
        return self


def _box_corners(b: Body) -> np.ndarray:
    h = b.half
    local = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
    R = m3.quat_to_matrix3(b.node.rotation)
    return local @ R.T + b.node.translation


# ---------------------------------------------------------------------------
# cloth
# ---------------------------------------------------------------------------

def _cloth_constraints(mesh: Mesh):
    edges = mesh.edges()
    pairs = [edges]
    # shear: opposite corners of quads
    quads = [f for f in mesh.faces if len(f) == 4]
    if quads:
        q = np.array(quads)
        pairs.append(q[:, [0, 2]])
        pairs.append(q[:, [1, 3]])
    c = np.unique(np.sort(np.concatenate(pairs), axis=1), axis=0)
    rest = np.linalg.norm(mesh.vertices[c[:, 0]] - mesh.vertices[c[:, 1]], axis=1)
    return c, rest


def _cloth_run(mesh: Mesh, colliders, pinned, steps, dt, gravity, iterations, thickness, ground, damping,
               record_every=None, substeps: int = 4):
    P = mesh.vertices.copy()
    prev = P.copy()
    cons, rest = _cloth_constraints(mesh)
    pin = np.zeros(len(P), dtype=bool)
    if pinned is not None:
        pin[np.asarray(pinned)] = True
    pin_pos = P[pin].copy()
    g = np.array([0.0, gravity, 0.0])
    frames = []
    deg = np.bincount(cons.reshape(-1), minlength=len(P)).astype(np.float64)
    dt = dt / substeps
    damping = 1 - (1 - damping) ** (1.0 / substeps)
    for s in range(steps * substeps):
        vel = (P - prev) * (1 - damping)
        prev = P.copy()
        P = P + vel + g * dt * dt
        for _ in range(iterations):
            d = P[cons[:, 1]] - P[cons[:, 0]]
            ln = np.linalg.norm(d, axis=1)
            corr = d * ((ln - rest) / np.maximum(ln, 1e-9) * 0.5)[:, None]
            acc = np.zeros_like(P)
            np.add.at(acc, cons[:, 0], corr)
            np.add.at(acc, cons[:, 1], -corr)
            P = P + acc / np.maximum(deg, 1)[:, None] * 1.6
            P[pin] = pin_pos
        # collisions
        for col in colliders:
            dist = col(P)
            inside = dist < thickness
            if inside.any():
                e = 1e-4
                Q = P[inside]
                gr = np.stack([col(Q + [e, 0, 0]) - col(Q - [e, 0, 0]), col(Q + [0, e, 0]) - col(Q - [0, e, 0]),
                               col(Q + [0, 0, e]) - col(Q - [0, 0, e])], axis=1) / (2 * e)
                gr = m3.normalize(gr)
                P[inside] = Q + gr * (thickness - dist[inside])[:, None]
                # friction: kill tangential motion on contact
                prev[inside] = prev[inside] + (P[inside] - prev[inside]) * 0.3
        if ground is not None:
            low = P[:, 1] < ground + thickness
            P[low, 1] = ground + thickness
            prev[low] = prev[low] + (P[low] - prev[low]) * 0.5
        P[pin] = pin_pos
        if record_every and s % (record_every * substeps) == 0:
            frames.append(P.copy())
    return P, frames


def cloth_drape(mesh: Mesh, colliders=(), pinned=None, steps: int = 300, dt: float = 1 / 60, gravity: float = -9.81,
                iterations: int = 8, thickness: float = 0.01, ground: float | None = 0.0,
                damping: float = 0.02) -> Mesh:
    """Simulate a cloth (a subdivided plane works best) falling onto colliders; returns the draped mesh.

    colliders: SDF objects (pyram3d.sdf shapes) the cloth must not penetrate.
    pinned   : vertex indices that stay fixed (flag/curtain attachment points).
    """
    P, _ = _cloth_run(mesh, list(colliders), pinned, steps, dt, gravity, iterations, thickness, ground, damping)
    out = mesh.copy()
    out.vertices = P
    out.smooth_angle = 180.0
    return out


def cloth_simulate(mesh: Mesh, anim, node: Node, colliders=(), pinned=None, duration: float = 2.0,
                   keyframes: int = 12, steps_per_second: int = 120, **kw) -> Mesh:
    """Bake a cloth simulation as shape keys + morph-weight keyframes on `node` (exports to glTF and FBX)."""
    steps = int(duration * steps_per_second)
    every = max(steps // keyframes, 1)
    P, frames = _cloth_run(mesh, list(colliders), pinned, steps, 1.0 / steps_per_second, kw.get("gravity", -9.81),
                           kw.get("iterations", 8), kw.get("thickness", 0.01), kw.get("ground", 0.0),
                           kw.get("damping", 0.02), record_every=every)
    frames = frames[1:] + [P]
    for i, F in enumerate(frames):
        mesh.add_shape_key(f"Frame{i:02d}", positions=F)
    n = len(frames)
    for i in range(n + 1):
        w = np.zeros(n)
        if i > 0:
            w[i - 1] = 1.0
        anim.key(node, "weights", duration * i / n, w)
    node.morph_weights = [0.0] * n
    mesh.smooth_angle = 180.0
    anim.loop = False
    return mesh


# ---------------------------------------------------------------------------
# particles
# ---------------------------------------------------------------------------

_PARTICLE_PRESETS = {
    "fire": dict(count=60, life=0.9, speed=(0.6, 1.4), spread=0.15, gravity=1.2, size=(0.12, 0.02),
                 color="#ff7a1a", emissive=6.0, radius=0.15, alpha=1.0),
    "smoke": dict(count=40, life=2.5, speed=(0.4, 0.8), spread=0.25, gravity=0.3, size=(0.06, 0.22),
                  color="#8f9296", emissive=0.0, radius=0.12, alpha=0.35),
    "sparks": dict(count=60, life=0.7, speed=(2.0, 4.0), spread=0.9, gravity=-9.0, size=(0.03, 0.01),
                   color="#ffd27a", emissive=8.0, radius=0.05, alpha=1.0),
    "fountain": dict(count=80, life=1.4, speed=(3.5, 4.5), spread=0.18, gravity=-9.81, size=(0.05, 0.04),
                     color="#7fc8ff", emissive=0.0, radius=0.1, alpha=0.8),
    "snow": dict(count=120, life=6.0, speed=(0.0, 0.1), spread=0.0, gravity=-0.6, size=(0.03, 0.03),
                 color="#ffffff", emissive=0.3, radius=5.0, alpha=1.0, start_height=6.0),
    "rain": dict(count=150, life=1.0, speed=(0.0, 0.0), spread=0.0, gravity=-12.0, size=(0.015, 0.015),
                 color="#a8c8ff", emissive=0.0, radius=5.0, alpha=0.7, start_height=6.0, stretch=8.0),
    "magic": dict(count=50, life=1.6, speed=(0.3, 0.8), spread=1.0, gravity=0.4, size=(0.05, 0.0),
                  color="#b77bff", emissive=6.0, radius=0.4, alpha=1.0, swirl=3.0),
}


def particles(scene, anim, kind: str = "fire", position=(0, 0, 0), duration: float = 2.0, seed: int = 0,
              fps: int = 15, parent: Node | None = None, **overrides) -> Node:
    """Create a looping particle effect as animated nodes (one shared mesh, instanced).

    kind: fire | smoke | sparks | fountain | snow | rain | magic. Any preset key can be
    overridden (count, life, speed, spread, gravity, size, color, emissive, radius, alpha).
    """
    cfg = dict(_PARTICLE_PRESETS[kind])
    cfg.update(overrides)
    rng = np.random.default_rng(seed)
    group = (parent or scene.root).add(None, name=f"Particles_{kind}", position=position)
    mat = Material(f"{kind.title()}Particle", cfg["color"], 0.0, 0.6, emissive=cfg["color"] if cfg["emissive"] else
                   "#000000", emissive_strength=max(cfg["emissive"], 1.0), alpha=cfg["alpha"],
                   unlit=cfg["emissive"] > 2)
    base = ico_sphere(1.0, 1)
    base.smooth_angle = 180.0
    base.set_material(mat)
    stretch = cfg.get("stretch", 1.0)
    times = np.linspace(0, duration, int(duration * fps) + 1)
    for i in range(int(cfg["count"])):
        birth = rng.uniform(0, duration)
        life = cfg["life"] * rng.uniform(0.7, 1.0)
        if cfg["radius"] > 1.0:  # area emitters (weather)
            origin = np.array([rng.uniform(-1, 1) * cfg["radius"], cfg.get("start_height", 5.0),
                               rng.uniform(-1, 1) * cfg["radius"]])
        else:
            a = rng.uniform(0, 2 * math.pi)
            rr = cfg["radius"] * math.sqrt(rng.random())
            origin = np.array([math.cos(a) * rr, 0.0, math.sin(a) * rr])
        sp = rng.uniform(*cfg["speed"])
        dirv = m3.normalize(np.array([rng.normal() * cfg["spread"], 1.0, rng.normal() * cfg["spread"]]))
        v0 = dirv * sp
        swirl = cfg.get("swirl", 0.0)
        node = group.add(base, name=f"P{i:03d}")
        for t in times:
            age = (t - birth) % duration
            alive = age < life
            if alive:
                u = age / life
                pos = origin + v0 * age + np.array([0, 0.5 * cfg["gravity"] * age * age, 0])
                if swirl:
                    ang = swirl * age
                    pos = pos + np.array([math.cos(ang + i), 0, math.sin(ang + i)]) * 0.2 * u
                s0, s1 = cfg["size"]
                s = s0 + (s1 - s0) * u
                if kind == "fire":
                    s *= math.sin(math.pi * min(u * 1.3, 1.0)) + 0.2
                scale = np.array([s, s * stretch, s])
            else:
                pos = origin
                scale = np.zeros(3)
            anim.key(node, "position", float(t), pos)
            anim.key(node, "scale", float(t), np.maximum(scale, 0.0))
        node.translation = anim.channel(node, "position").values[0].copy()
        node.scale = anim.channel(node, "scale").values[0].copy()
    return group
