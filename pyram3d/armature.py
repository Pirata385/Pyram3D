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
        self._bind_root: np.ndarray | None = None  # root world matrix captured at bind time

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

    def bind_root_matrix(self) -> np.ndarray:
        """Armature root transform the mesh was bound with (moving the root later moves the character)."""
        return self._bind_root if self._bind_root is not None else self.root.world_matrix()

    def rest_world_matrices(self) -> np.ndarray:
        root_w = self.bind_root_matrix()
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

        If the mesh came from SDF parts tagged with `bone="BoneName"`, those
        parts follow their bone exactly (blended only at the seams); untagged
        parts fall back to distance-based weights.

        rigid=True      : each vertex follows exactly one bone (robots, mechanical parts)
        smooth          : weight smoothing iterations across the mesh surface
        falloff         : higher = sharper transitions between bones
        exclude         : bone names that should never deform the mesh (e.g. "Root")
        bone_filter     : optional fn(vertex_positions, bone_index) -> bool mask restricting a bone
        """
        P = mesh.vertices
        heads = np.array([self._head[id(b)] for b in self.bones])
        tails = np.array([self._tail[id(b)] for b in self.bones])
        self._bind_root = self.root.world_matrix()
        root_w = self._bind_root
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
        parts = mesh.extras.get("bone_parts")
        if parts is not None and parts[1].shape[0] == len(P):
            names, PW = parts
            name_to_idx = {b.name: i for i, b in enumerate(self.bones)}
            missing = [n for n in names if n not in name_to_idx]
            if missing:
                raise KeyError(f"SDF parts reference unknown bones {missing}")
            WP = np.zeros_like(W)
            for j, n in enumerate(names):
                WP[:, name_to_idx[n]] = PW[:, j]
            owned = WP.sum(axis=1) > 0
            if rigid:
                WP[owned] = (WP[owned] == WP[owned].max(axis=1, keepdims=True)).astype(float)
            W = np.where(owned[:, None], WP, W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12))
        joints, weights = _top_k(W, max_influences)
        mesh.joints = joints
        mesh.weights = weights
        return mesh

    def transfer_weights(self, mesh: Mesh, source: Mesh, k: int = 3, verts=None) -> Mesh:
        """Copy skin weights from `source` (e.g. the body) to `mesh` (clothes, armor, hair).

        Each vertex blends the weights of its `k` nearest source vertices (inverse distance), so
        a garment modeled over the body deforms exactly like the skin under it. `verts` limits
        the transfer to a vertex selection (others keep their weights; call bind() first then).
        """
        if source.joints is None:
            raise ValueError("source mesh has no skin weights")
        P = mesh.vertices if verts is None else mesh.vertices[verts]
        S = source.vertices
        nb = len(self.bones)
        W = np.zeros((len(P), nb))
        src_full = np.zeros((len(S), nb))
        np.add.at(src_full, (np.arange(len(S))[:, None].repeat(4, 1), source.joints), source.weights)
        step = max(1, 4_000_000 // max(len(S), 1))
        for i in range(0, len(P), step):
            d2 = ((P[i:i + step, None, :] - S[None, :, :]) ** 2).sum(-1)
            idx = np.argpartition(d2, min(k, len(S) - 1), axis=1)[:, :k]
            dk = np.take_along_axis(d2, idx, axis=1)
            w = 1.0 / (np.sqrt(dk) + 1e-4)
            w /= w.sum(axis=1, keepdims=True)
            W[i:i + step] = np.einsum("nk,nkb->nb", w, src_full[idx])
        joints, weights = _top_k(W, 4)
        if verts is None or mesh.joints is None:
            if verts is not None:
                raise ValueError("bind() the mesh before a partial transfer")
            mesh.joints, mesh.weights = joints, weights
        else:
            mesh.joints[verts], mesh.weights[verts] = joints, weights
        return mesh

    def bind_skirt(self, mesh: Mesh, mask=None, hips: str = "Hips", left: str = "LeftUpperLeg",
                   right: str = "RightUpperLeg", top: float | None = None, bottom: float | None = None,
                   influence: float = 0.8, side_blend: float | None = None) -> Mesh:
        """Weights for skirts, robes and long coats that must not tear between the legs.

        Vertices in `mask` (default: everything below the hip joint) are shared between the
        hips and both thighs: the lower the vertex, the more it follows the thighs (up to
        `influence`), split left/right with a smooth blend across x = 0. Call after bind()
        so the rest of the garment keeps its normal weights.
        """
        P = mesh.vertices
        root_w = self.bind_root_matrix()
        hy = m3.transform_points(root_w, self.head(hips)[None])[0][1] if top is None else top
        ky = m3.transform_points(root_w, self.tail(left)[None])[0][1] if bottom is None else bottom
        if mask is None:
            mask = P[:, 1] < hy
        mask = np.asarray(mask)
        if mask.dtype != bool:
            m = np.zeros(len(P), dtype=bool)
            m[mask] = True
            mask = m
        cx = m3.transform_points(root_w, self.head(hips)[None])[0][0]
        width = side_blend or abs(self.head(left)[0] - self.head(right)[0]) * 0.6
        down = np.clip((hy - P[:, 1]) / max(hy - ky, 1e-6), 0, 1) * influence
        side = m3.smoothstep(-width, width, P[:, 0] - cx)
        W = np.zeros((len(P), len(self.bones)))
        W[:, self.index(hips)] = 1 - down
        W[:, self.index(left)] = down * side
        W[:, self.index(right)] = down * (1 - side)
        j, w = _top_k(W, 4)
        if mesh.joints is None:
            mesh.joints, mesh.weights = j, w
        else:
            mesh.joints[mask], mesh.weights[mask] = j[mask], w[mask]
        return mesh

    def split_mesh(self, mesh: Mesh, groups: dict, name_prefix: str = "") -> dict:
        """Split a skinned mesh into parts by dominant bone, e.g. for modular characters:

            parts = sk.split_mesh(body, {"Head": ["Head", "Neck"], "Arms": ["LeftUpperArm", ...], ...})

        Bones not listed go to the first group. Frozen normals keep the seams invisible
        (call mesh.freeze_normals() first). Returns {group_name: Mesh}.
        """
        bone_group = {}
        names = list(groups)
        for g, bones in groups.items():
            for b in bones:
                bone_group[self.index(b)] = names.index(g)
        dom = mesh.joints[np.arange(len(mesh.joints)), np.argmax(mesh.weights, axis=1)]
        vgroup = np.array([bone_group.get(int(b), 0) for b in dom])
        # a face belongs to the group most of its corners belong to
        face_group = np.array([np.bincount(vgroup[list(f)], minlength=len(names)).argmax() for f in mesh.faces])
        out = {}
        for gi, g in enumerate(names):
            sel = np.flatnonzero(face_group == gi)
            if len(sel):
                out[g] = mesh.separate(sel, name=name_prefix + g)
        return out

    def add_socket(self, name: str, bone: Node | str, position, rotation=None) -> Node:
        """Empty attachment point (weapons, shields, helmets, backpacks) parented to a bone.

        `position` is in armature/rest space; the socket follows the bone when animated.
        """
        b = self.bone(bone) if isinstance(bone, str) else bone
        node = b.add(None, name=name, position=m3.vec3(position) - self.head(b), rotation=rotation)
        node.extras["is_socket"] = True
        return node

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
