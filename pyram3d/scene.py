"""Scene graph: nodes with transforms, meshes, lights, cameras, skeletons."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

from . import math3d as m3
from .mesh import Mesh


@dataclass(eq=False)
class Light:
    """Punctual light (glTF KHR_lights_punctual). Lights shine along their node's local -Z.

    intensity: directional/sun in lux (3-10 typical); point/spot in candela (10-1000).
    """
    type: str = "directional"  # directional | point | spot
    color: object = "#ffffff"
    intensity: float = 3.0
    range: float | None = None
    inner_cone: float = 20.0  # degrees (spot)
    outer_cone: float = 35.0  # degrees (spot)
    cast_shadows: bool = True
    name: str = "Light"

    @property
    def color_linear(self) -> np.ndarray:
        return m3.srgb_to_linear(m3.as_color(self.color)[:3])


@dataclass(eq=False)
class Camera:
    """Camera looking along its node's local -Z."""
    type: str = "perspective"  # perspective | orthographic
    yfov: float = 45.0  # degrees
    aspect: float | None = None
    znear: float = 0.05
    zfar: float = 1000.0
    ortho_height: float = 2.0  # half-height for orthographic
    name: str = "Camera"


class Node:
    def __init__(self, name: str = "Node", mesh: Mesh | None = None, translation=(0, 0, 0), rotation=None,
                 scale=(1, 1, 1), light: Light | None = None, camera: Camera | None = None):
        self.name = name
        self.mesh = mesh
        self.light = light
        self.camera = camera
        self.translation = m3.vec3(translation)
        self.rotation = m3.as_quat(rotation)
        self.scale = m3.vec3(scale)
        self.children: list[Node] = []
        self.parent: Node | None = None
        self.skin = None  # Skeleton, set for skinned meshes
        self.morph_weights: list[float] | None = None
        self.extras: dict = {}
        self.visible = True

    def __repr__(self):
        what = []
        if self.mesh is not None:
            what.append(f"mesh={self.mesh.name}")
        if self.light is not None:
            what.append(f"light={self.light.type}")
        if self.camera is not None:
            what.append("camera")
        return f"<Node {self.name!r} {' '.join(what)}>"

    # ---------------------------------------------------------- hierarchy
    def add(self, obj=None, name: str | None = None, position=None, rotation=None, scale=None, **kw) -> "Node":
        """Add a child. obj may be a Node, Mesh, Light, Camera, Skeleton or None (empty)."""
        node = _as_node(obj, name)
        if position is not None:
            node.translation = m3.vec3(position)
        if rotation is not None:
            node.rotation = m3.as_quat(rotation)
        if scale is not None:
            node.scale = m3.vec3(scale)
        for k, v in kw.items():
            setattr(node, k, v)
        if node.parent is not None:
            node.parent.children.remove(node)
        node.parent = self
        self.children.append(node)
        return node

    def remove(self, child: "Node"):
        self.children.remove(child)
        child.parent = None

    def walk(self) -> Iterator["Node"]:
        yield self
        for c in self.children:
            yield from c.walk()

    def find(self, name: str) -> "Node | None":
        for n in self.walk():
            if n.name == name:
                return n
        return None

    # ---------------------------------------------------------- transforms
    @property
    def position(self) -> np.ndarray:
        return self.translation

    @position.setter
    def position(self, v):
        self.translation = m3.vec3(v)

    def set_rotation(self, rotation) -> "Node":
        """Euler degrees (x, y, z) or quaternion (x, y, z, w)."""
        self.rotation = m3.as_quat(rotation)
        return self

    def rotate(self, rotation) -> "Node":
        """Apply an additional rotation (in parent space) on top of the current one."""
        self.rotation = m3.quat_normalize(m3.quat_mul(m3.as_quat(rotation), self.rotation))
        return self

    def look_at(self, target, up=(0, 1, 0)) -> "Node":
        """Point the local -Z axis at `target` (world space). Used for cameras and lights."""
        eye = self.world_matrix()[:3, 3]
        q = m3.quat_look_rotation(m3.vec3(target) - eye, up)
        if self.parent is not None:
            pq = m3.matrix3_to_quat(m3.normalize(self.parent.world_matrix()[:3, :3], axis=0))
            q = m3.quat_mul(m3.quat_conj(pq), q)
        self.rotation = q
        return self

    def local_matrix(self) -> np.ndarray:
        return m3.trs_matrix(self.translation, self.rotation, self.scale)

    def world_matrix(self) -> np.ndarray:
        m = self.local_matrix()
        p = self.parent
        while p is not None:
            m = p.local_matrix() @ m
            p = p.parent
        return m

    def world_position(self) -> np.ndarray:
        return self.world_matrix()[:3, 3].copy()


def _as_node(obj, name: str | None) -> Node:
    from .armature import Skeleton
    if isinstance(obj, Node):
        node = obj
    elif isinstance(obj, Mesh):
        node = Node(name or obj.name, mesh=obj)
    elif isinstance(obj, Light):
        node = Node(name or obj.name, light=obj)
    elif isinstance(obj, Camera):
        node = Node(name or obj.name, camera=obj)
    elif isinstance(obj, Skeleton):
        node = obj.root
    elif obj is None:
        node = Node(name or "Empty")
    else:
        raise TypeError(f"cannot add {type(obj).__name__} to a scene")
    if name:
        node.name = name
    return node


class Scene:
    """A complete 3D scene. Add meshes, lights, cameras and animations, then export."""

    def __init__(self, name: str = "Scene", fps: int = 30):
        self.name = name
        self.root = Node("__root__")
        self.animations: list = []
        self.fps = fps
        self.background = "#d9dee5"  # preview renders
        self.ambient = 0.35          # preview renders

    # ------------------------------------------------------------ building
    def add(self, obj=None, name: str | None = None, position=None, rotation=None, scale=None,
            parent: Node | None = None, **kw) -> Node:
        """Add a Mesh/Node/Light/Camera/Skeleton (optionally under `parent`). Returns the Node."""
        return (parent or self.root).add(obj, name, position, rotation, scale, **kw)

    def add_light(self, type: str = "directional", position=(4, 8, 6), target=(0, 0, 0), color="#ffffff",
                  intensity: float | None = None, name: str | None = None, **kw) -> Node:
        if intensity is None:
            intensity = {"directional": 3.0, "point": 100.0, "spot": 200.0}[type]
        light = Light(type=type, color=color, intensity=intensity, name=name or f"{type.title()}Light", **kw)
        node = self.add(light, position=position)
        node.look_at(target)
        return node

    def add_camera(self, position=(3, 2, 5), target=(0, 0.5, 0), yfov: float = 45.0, name: str = "Camera",
                   **kw) -> Node:
        node = self.add(Camera(yfov=yfov, name=name, **kw), position=position)
        node.look_at(target)
        return node

    def add_default_lights(self, strength: float = 1.0) -> list[Node]:
        """Three-point lighting rig (key, fill, rim)."""
        return [
            self.add_light("directional", (4, 8, 6), intensity=3.0 * strength, name="KeyLight"),
            self.add_light("directional", (-6, 4, 3), intensity=1.0 * strength, name="FillLight",
                           color="#cfe0ff", cast_shadows=False),
            self.add_light("directional", (0, 5, -8), intensity=1.5 * strength, name="RimLight", cast_shadows=False),
        ]

    def animation(self, name: str = "Action"):
        """Create (or get) an animation clip by name."""
        from .animation import Animation
        for a in self.animations:
            if a.name == name:
                return a
        a = Animation(name, fps=self.fps)
        self.animations.append(a)
        return a

    # ------------------------------------------------------------ queries
    @property
    def nodes(self) -> list[Node]:
        return self.root.children

    def walk(self) -> Iterator[Node]:
        for c in self.root.children:
            yield from c.walk()

    def find(self, name: str) -> Node | None:
        for n in self.walk():
            if n.name == name:
                return n
        return None

    def mesh_nodes(self) -> list[Node]:
        return [n for n in self.walk() if n.mesh is not None]

    def bounds(self, time: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        lo = np.full(3, np.inf)
        hi = np.full(3, -np.inf)
        for n in self.mesh_nodes():
            if not n.visible or len(n.mesh.vertices) == 0:
                continue
            if n.skin is not None:
                pts = n.mesh.vertices
            else:
                pts = m3.transform_points(n.world_matrix(), n.mesh.vertices)
            lo = np.minimum(lo, pts.min(axis=0))
            hi = np.maximum(hi, pts.max(axis=0))
        if not np.isfinite(lo).all():
            return np.zeros(3), np.zeros(3)
        return lo, hi

    def materials(self) -> list:
        out = []
        for n in self.mesh_nodes():
            for m in n.mesh.materials:
                if m is not None and all(m is not x for x in out):
                    out.append(m)
        return out

    def describe(self) -> str:
        """Human/AI-readable summary of the scene tree, sizes and animations."""
        lines = [f"Scene {self.name!r}"]

        def rec(node: Node, depth: int):
            pad = "  " * depth
            t = np.round(node.translation, 3).tolist()
            e = np.round(m3.quat_to_euler(node.rotation), 1).tolist()
            s = np.round(node.scale, 3).tolist()
            desc = f"{pad}- {node.name} pos={t}"
            if any(abs(x) > 1e-6 for x in e):
                desc += f" rot={e}"
            if any(abs(x - 1) > 1e-6 for x in s):
                desc += f" scale={s}"
            if node.mesh is not None:
                st = node.mesh.stats()
                desc += (f" | mesh {node.mesh.name}: {st['vertices']}v {st['triangles']}tris size={st['size']}"
                         f" mats={st['materials']}")
                if node.skin is not None:
                    desc += f" skinned({len(node.skin.bones)} bones)"
                if st["shape_keys"]:
                    desc += f" shape_keys={st['shape_keys']}"
            if node.light is not None:
                desc += f" | light {node.light.type} {node.light.intensity}"
            if node.camera is not None:
                desc += f" | camera fov={node.camera.yfov}"
            lines.append(desc)
            for c in node.children:
                rec(c, depth + 1)

        for c in self.root.children:
            rec(c, 1)
        lo, hi = self.bounds()
        lines.append(f"bounds min={np.round(lo, 3).tolist()} max={np.round(hi, 3).tolist()} "
                     f"size={np.round(hi - lo, 3).tolist()}")
        tris = sum(n.mesh.num_triangles for n in self.mesh_nodes())
        lines.append(f"total triangles: {tris}")
        for a in self.animations:
            lines.append(f"animation {a.name!r}: {a.duration:.2f}s, {len(a.channels)} channels")
        return "\n".join(lines)

    # ------------------------------------------------------------ output
    def export(self, path, **kw) -> Path:
        """Export by file extension: .glb, .gltf, .fbx, .obj"""
        path = Path(path)
        ext = path.suffix.lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        if ext in (".glb", ".gltf"):
            from .export.gltf import export_gltf
            return export_gltf(self, path, **kw)
        if ext == ".fbx":
            from .export.fbx import export_fbx
            return export_fbx(self, path, **kw)
        if ext == ".obj":
            from .export.obj import export_obj
            return export_obj(self, path, **kw)
        raise ValueError(f"unsupported export format: {ext}")

    def export_all(self, directory, basename: str | None = None, formats=("glb", "gltf", "fbx")) -> list[Path]:
        """Export the scene in several formats at once."""
        directory = Path(directory)
        basename = basename or self.name
        return [self.export(directory / f"{basename}.{fmt}") for fmt in formats]

    def render(self, path=None, *args, **kw):
        """Render a preview image (PNG). See pyram3d.render.render for options."""
        from .render import render
        return render(self, path, *args, **kw)

    def render_views(self, path=None, **kw):
        """Contact sheet: front, right, back, 3/4, top, left."""
        from .render import render_views
        return render_views(self, path, **kw)
