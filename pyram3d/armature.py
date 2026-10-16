"""Skeletons (armatures/rigs), automatic skin weights and linear-blend skinning.

Bones are scene Nodes, so they are animated with the regular Animation API.
Bones are created in the REST pose with identity rotation: a bone's local axes
are aligned with the world axes when the rig is at rest. That makes rotations
easy to reason about:

    rotate about X -> swing forward/back (pitch)
    rotate about Y -> twist / turn (yaw)
    rotate about Z -> raise/lower sideways (roll)

Characters face +Z. Their LEFT side is +X, RIGHT side is -X.

    sk = Skeleton("Rig")
    hips = sk.add_bone("Hips", head=(0, 1.0, 0), tail=(0, 1.15, 0))
    leg = sk.add_bone("LeftUpperLeg", (0.1, 0.95, 0), (0.1, 0.5, 0), parent=hips)
    sk.bind(mesh)                    # automatic weights
    scene.add(sk); scene.add(mesh_node_with_skin)
"""
from __future__ import annotations

import numpy as np

from . import math3d as m3
from .mesh import Mesh
from .scene import Node


class Skeleton:
    def __init__(self, name: str = "Armature", position=(0, 0, 0)):
        self.name = name
        self.root = Node(name, translation=position)
        self.root.extras["is_armature"] = True
        self.bones: list[Node] = []
        self._head: dict[int, np.ndarray] = {}
        self._tail: dict[int, np.ndarray] = {}

    def __repr__(self):
        return f"<Skeleton {self.name!r} {len(self.bones)} bones>"

    # ------------------------------------------------------------- building
    def add_bone(self, name: str, head, tail=None, parent: Node | str | None = None) -> Node:
        """Add a bone from `head` (joint position) to `tail`, in armature (rest) space."""
        head = m3.vec3(head)
        tail = head + np.array([0, 0.1, 0]) if tail is None else m3.vec3(tail)
        if isinstance(parent, str):
            parent = self.bone(parent)
        if parent is None:
            node = self.root.add(Node(name, translation=head))
        else:
            node = parent.add(Node(name, translation=head - self.head(parent)))
        node.extras["is_bone"] = True
        node.extras["rest_translation"] = node.translation.copy()
        self.bones.append(node)
        self._head[id(node)] = head
        self._tail[id(node)] = tail
        return node

    def add_chain(self, names, points, parent: Node | str | None = None) -> list[Node]:
        """Add consecutive bones through points (len(points) == len(names) + 1)."""
        out = []
        for i, nm in enumerate(names):
            parent = self.add_bone(nm, points[i], points[i + 1], parent)
            out.append(parent)
        return out

    def bone(self, name: str) -> Node:
        for b in self.bones:
            if b.name == name:
                return b
        raise KeyError(f"no bone named {name!r}; bones: {[b.name for b in self.bones]}")

    def __getitem__(self, name: str) -> Node:
        return self.bone(name)

    def head(self, bone: Node | str) -> np.ndarray:
        b = self.bone(bone) if isinstance(bone, str) else bone
        return self._head[id(b)].copy()

    def tail(self, bone: Node | str) -> np.ndarray:
        b = self.bone(bone) if isinstance(bone, str) else bone
        return self._tail[id(b)].copy()

    def index(self, bone: Node | str) -> int:
        b = self.bone(bone) if isinstance(bone, str) else bone
        return self.bones.index(b)

    # ---------------------------------------------------------------- poses
    def reset_pose(self):
        """Return every bone to the rest pose."""
        for b in self.bones:
            b.translation = b.extras["rest_translation"].copy()
            b.rotation = b.extras.get("rest_rotation", m3.quat_identity()).copy()
            b.scale = b.extras.get("rest_scale", np.ones(3)).copy()

    def rest_world_matrices(self) -> np.ndarray:
        root_w = self.root.world_matrix()
        return np.array([root_w @ m3.translation_matrix(self._head[id(b)]) for b in self.bones])

    def inverse_bind_matrices(self) -> np.ndarray:
        if getattr(self, "_ibm", None) is not None:  # imported rigs keep their own bind pose
            return self._ibm
        return np.array([np.linalg.inv(m) for m in self.rest_world_matrices()])

    def joint_matrices(self) -> np.ndarray:
        """Current skinning matrices (world-space pose @ inverse bind)."""
        ibm = self.inverse_bind_matrices()
        return np.array([b.world_matrix() @ ibm[i] for i, b in enumerate(self.bones)])

    def pose_vertices(self, mesh: Mesh) -> np.ndarray:
        """Linear blend skinning of mesh vertices for the current pose."""
        J = self.joint_matrices()
        v = np.concatenate([mesh.vertices, np.ones((len(mesh.vertices), 1))], axis=1)
        out = np.zeros((len(v), 3))
        for k in range(mesh.joints.shape[1]):
            M = J[mesh.joints[:, k]]  # (N,4,4)
            out += mesh.weights[:, k, None] * np.einsum("nij,nj->ni", M, v)[:, :3]
        return out

    # ---------------------------------------------------------------- binding
    def bind(self, mesh: Mesh, rigid: bool = False, smooth: int = 4, falloff: float = 4.0,
             max_influences: int = 4, exclude: list[str] | None = None,
             bone_filter=None) -> Mesh:
        """Compute automatic skin weights for `mesh` (vertices in world/rest space).

        rigid=True      : each vertex follows exactly one bone (robots, mechanical parts)
        smooth          : weight smoothing iterations across the mesh surface
        falloff         : higher = sharper transitions between bones
        exclude         : bone names that should never deform the mesh (e.g. "Root")
        bone_filter     : optional fn(vertex_positions, bone_index) -> bool mask restricting a bone
        """
        P = mesh.vertices
        heads = np.array([self._head[id(b)] for b in self.bones])
        tails = np.array([self._tail[id(b)] for b in self.bones])
        root_w = self.root.world_matrix()
        heads = m3.transform_points(root_w, heads)
        tails = m3.transform_points(root_w, tails)
        nb = len(self.bones)
        D = np.empty((len(P), nb))
        for i in range(nb):
            D[:, i] = _point_segment_distance(P, heads[i], tails[i])
        excluded = set(exclude or [])
        for i, b in enumerate(self.bones):
            if b.name in excluded:
                D[:, i] = np.inf
            if bone_filter is not None:
                mask = np.asarray(bone_filter(P, i), dtype=bool)
                D[~mask, i] = np.inf
        nearest = np.argmin(D, axis=1)
        if rigid:
            W = np.zeros((len(P), nb))
            W[np.arange(len(P)), nearest] = 1.0
        else:
            # per-bone thickness: median distance of the vertices it owns
            thick = np.ones(nb)
            for i in range(nb):
                own = D[nearest == i, i]
                thick[i] = np.median(own) if len(own) else np.median(D[:, i][np.isfinite(D[:, i])] if
                                                                    np.isfinite(D[:, i]).any() else [1.0])
            thick = np.maximum(thick, 1e-3)
            rel = D / thick[None, :]
            relmin = rel.min(axis=1, keepdims=True)
            W = np.exp(-falloff * (rel - relmin))
            W[~np.isfinite(D)] = 0.0
            if smooth > 0 and mesh.faces:
                edges = mesh.edges()
                deg = np.bincount(edges.reshape(-1), minlength=len(P)).astype(np.float64)
                for _ in range(smooth):
                    acc = np.zeros_like(W)
                    np.add.at(acc, edges[:, 0], W[edges[:, 1]])
                    np.add.at(acc, edges[:, 1], W[edges[:, 0]])
                    avg = acc / np.maximum(deg, 1)[:, None]
                    W = 0.5 * W + 0.5 * np.where(deg[:, None] > 0, avg, W)
                W[~np.isfinite(D)] = 0.0
        joints, weights = _top_k(W, max_influences)
        mesh.joints = joints
        mesh.weights = weights
        return mesh

    def attach(self, node: Node, bone: Node | str) -> Node:
        """Rigidly attach an object (hat, sword, glasses) to a bone, keeping its current world placement."""
        b = self.bone(bone) if isinstance(bone, str) else bone
        world = node.world_matrix()
        inv = np.linalg.inv(b.world_matrix())
        local = inv @ world
        t, r, s = m3.decompose(local)
        b.add(node)
        node.translation, node.rotation, node.scale = t, r, s
        return node

    def describe(self) -> str:
        lines = [f"Skeleton {self.name!r} ({len(self.bones)} bones)"]

        def rec(n: Node, depth):
            for c in n.children:
                if c.extras.get("is_bone"):
                    lines.append(f"{'  ' * depth}- {c.name} head={np.round(self.head(c), 3).tolist()} "
                                 f"tail={np.round(self.tail(c), 3).tolist()}")
                    rec(c, depth + 1)
        rec(self.root, 1)
        return "\n".join(lines)


def _point_segment_distance(P: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip(((P - a) @ ab) / max(float(ab @ ab), 1e-12), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1)


def _top_k(W: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    k = min(k, W.shape[1])
    idx = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, idx, axis=1)
    w[w < 0.01] = 0.0
    s = w.sum(axis=1, keepdims=True)
    w = np.where(s > 0, w / np.maximum(s, 1e-12), np.eye(1, k)[0])
    joints = np.zeros((len(W), 4), dtype=np.int64)
    weights = np.zeros((len(W), 4))
    joints[:, :k] = idx
    weights[:, :k] = w
    # unused slots point at joint 0 with weight 0 (valid for glTF)
    joints[weights == 0] = 0
    return joints, weights


def skinned_node(mesh: Mesh, skeleton: Skeleton, name: str | None = None) -> Node:
    """Create the scene node for a skinned mesh (keep it at identity transform)."""
    if mesh.joints is None:
        skeleton.bind(mesh)
    node = Node(name or mesh.name, mesh=mesh)
    node.skin = skeleton
    return node
