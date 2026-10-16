"""Polygon mesh with modeling operations.

A Mesh stores:
  vertices       (N, 3) float64 positions
  faces          list of tuples of vertex indices (tris, quads or n-gons, CCW = front)
  uvs/face_uvs   optional face-varying UVs (like OBJ: face_uvs[i] indexes into uvs)
  face_materials per-face index into `materials`
  colors         optional (N, 4) per-vertex colors, LINEAR RGBA
  joints/weights optional (N, 4) skinning data (filled by Skeleton.bind)
  shape_keys     dict name -> (N, 3) absolute positions (morph targets / blend shapes)
  smooth_angle   shading: 0 = flat, 180 = fully smooth, in between = auto-smooth

Every modeling method mutates the mesh in place and returns `self` so calls can
be chained:  box().subdivide(2).scale(0.5).translate((0, 1, 0))
Methods that create new geometry (extrude, inset) return the new face indices
when `return_faces=True`.
"""
from __future__ import annotations

import math
from itertools import chain
from typing import Callable, Iterable, Sequence

import numpy as np

from . import math3d as m3

FaceSel = Iterable[int] | np.ndarray | None


class Mesh:
    def __init__(self, vertices=None, faces=None, uvs=None, face_uvs=None, name: str = "Mesh",
                 materials=None, face_materials=None, colors=None, smooth_angle: float = 40.0):
        self.name = name
        self.vertices = np.zeros((0, 3)) if vertices is None else np.array(vertices, dtype=np.float64).reshape(-1, 3)
        self.faces: list[tuple[int, ...]] = [] if faces is None else [tuple(int(i) for i in f) for f in faces]
        self.uvs = None if uvs is None else np.array(uvs, dtype=np.float64).reshape(-1, 2)
        self.face_uvs = None if face_uvs is None else [tuple(int(i) for i in f) for f in face_uvs]
        self.materials: list = list(materials) if materials else []
        if face_materials is None:
            self.face_materials = np.zeros(len(self.faces), dtype=np.int64)
        else:
            self.face_materials = np.array(face_materials, dtype=np.int64).reshape(-1)
        self.colors = None if colors is None else np.array(colors, dtype=np.float64).reshape(-1, 4)
        self.joints: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.shape_keys: dict[str, np.ndarray] = {}
        self.smooth_angle = float(smooth_angle)
        if (self.uvs is None) != (self.face_uvs is None):
            raise ValueError("uvs and face_uvs must be given together")

    # ------------------------------------------------------------------ basics
    def __repr__(self):
        return f"<Mesh {self.name!r}: {len(self.vertices)} verts, {len(self.faces)} faces>"

    @property
    def num_vertices(self) -> int:
        return len(self.vertices)

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    @property
    def num_triangles(self) -> int:
        return int(sum(len(f) - 2 for f in self.faces))

    def copy(self, name: str | None = None) -> "Mesh":
        m = Mesh(name=name or self.name)
        m.vertices = self.vertices.copy()
        m.faces = list(self.faces)
        m.uvs = None if self.uvs is None else self.uvs.copy()
        m.face_uvs = None if self.face_uvs is None else list(self.face_uvs)
        m.materials = list(self.materials)
        m.face_materials = self.face_materials.copy()
        m.colors = None if self.colors is None else self.colors.copy()
        m.joints = None if self.joints is None else self.joints.copy()
        m.weights = None if self.weights is None else self.weights.copy()
        m.shape_keys = {k: v.copy() for k, v in self.shape_keys.items()}
        m.smooth_angle = self.smooth_angle
        return m

    # ----------------------------------------------------------- corner data
    def _corners(self):
        """Flattened face corners: (corner_vertex, corner_face, face_start, face_size)."""
        sizes = np.fromiter((len(f) for f in self.faces), dtype=np.int64, count=len(self.faces))
        cv = np.fromiter(chain.from_iterable(self.faces), dtype=np.int64, count=int(sizes.sum()))
        cf = np.repeat(np.arange(len(self.faces)), sizes)
        start = np.zeros(len(self.faces), dtype=np.int64)
        if len(sizes):
            start[1:] = np.cumsum(sizes)[:-1]
        return cv, cf, start, sizes

    def _corner_next(self, start, sizes):
        c = int(sizes.sum())
        nxt = np.arange(c) + 1
        ends = start + sizes - 1
        nxt[ends] = start
        prv = np.arange(c) - 1
        prv[start] = ends
        return nxt, prv

    def _corner_uvs(self):
        if self.face_uvs is None:
            return None
        return np.fromiter(chain.from_iterable(self.face_uvs), dtype=np.int64)

    # ------------------------------------------------------------- geometry
    def face_normals(self) -> np.ndarray:
        """Unit normals via Newell's method (robust for n-gons)."""
        if not self.faces:
            return np.zeros((0, 3))
        cv, cf, start, sizes = self._corners()
        nxt, _ = self._corner_next(start, sizes)
        a = self.vertices[cv]
        b = self.vertices[cv[nxt]]
        cr = np.cross(a, b)
        n = np.zeros((len(self.faces), 3))
        np.add.at(n, cf, cr)
        return m3.normalize(n)

    def face_areas(self) -> np.ndarray:
        if not self.faces:
            return np.zeros(0)
        cv, cf, start, sizes = self._corners()
        nxt, _ = self._corner_next(start, sizes)
        cr = np.cross(self.vertices[cv], self.vertices[cv[nxt]])
        n = np.zeros((len(self.faces), 3))
        np.add.at(n, cf, cr)
        return 0.5 * np.linalg.norm(n, axis=1)

    def face_centers(self) -> np.ndarray:
        if not self.faces:
            return np.zeros((0, 3))
        cv, cf, _, sizes = self._corners()
        c = np.zeros((len(self.faces), 3))
        np.add.at(c, cf, self.vertices[cv])
        return c / sizes[:, None]

    def vertex_normals(self) -> np.ndarray:
        """Area-weighted smooth vertex normals."""
        n = np.zeros_like(self.vertices)
        if not self.faces:
            return n
        cv, cf, start, sizes = self._corners()
        nxt, _ = self._corner_next(start, sizes)
        cr = np.cross(self.vertices[cv], self.vertices[cv[nxt]])
        fn = np.zeros((len(self.faces), 3))
        np.add.at(fn, cf, cr)  # length = 2*area -> area weighting
        np.add.at(n, cv, fn[cf])
        return m3.normalize(n)

    def corner_normals(self, angle: float | None = None) -> np.ndarray:
        """Per-corner normals honoring the auto-smooth angle (degrees)."""
        angle = self.smooth_angle if angle is None else angle
        cv, cf, start, sizes = self._corners()
        fn_unit = self.face_normals()
        if angle <= 0.0:
            return fn_unit[cf]
        nxt, _ = self._corner_next(start, sizes)
        cr = np.cross(self.vertices[cv], self.vertices[cv[nxt]])
        fn_w = np.zeros((len(self.faces), 3))
        np.add.at(fn_w, cf, cr)
        if angle >= 180.0:
            vn = np.zeros_like(self.vertices)
            np.add.at(vn, cv, fn_w[cf])
            return m3.normalize(vn[cv])
        # pair every corner with every other corner sharing its vertex
        order = np.argsort(cv, kind="stable")
        sv = cv[order]
        bounds = np.flatnonzero(np.diff(sv)) + 1
        gstart = np.concatenate([[0], bounds])
        gsize = np.diff(np.concatenate([gstart, [len(sv)]]))
        grp_of_sorted = np.repeat(np.arange(len(gstart)), gsize)
        # for sorted corner i, partners are sorted[gstart[g] : gstart[g]+gsize[g]]
        reps = gsize[grp_of_sorted]
        a_sorted = np.repeat(np.arange(len(sv)), reps)
        offs = np.arange(len(a_sorted)) - np.repeat(np.cumsum(reps) - reps, reps)
        b_sorted = gstart[grp_of_sorted[a_sorted]] + offs
        a = order[a_sorted]
        b = order[b_sorted]
        fa, fb = cf[a], cf[b]
        ok = np.einsum("ij,ij->i", fn_unit[fa], fn_unit[fb]) >= math.cos(math.radians(angle)) - 1e-9
        acc = np.zeros((len(cv), 3))
        np.add.at(acc, a[ok], fn_w[fb[ok]])
        out = m3.normalize(acc)
        bad = np.linalg.norm(acc, axis=1) < 1e-12
        out[bad] = fn_unit[cf[bad]]
        return out

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        if len(self.vertices) == 0:
            return np.zeros(3), np.zeros(3)
        return self.vertices.min(axis=0), self.vertices.max(axis=0)

    @property
    def size(self) -> np.ndarray:
        lo, hi = self.bounds()
        return hi - lo

    @property
    def center(self) -> np.ndarray:
        lo, hi = self.bounds()
        return (lo + hi) / 2

    def edges(self) -> np.ndarray:
        """Unique undirected edges (E, 2)."""
        cv, cf, start, sizes = self._corners()
        nxt, _ = self._corner_next(start, sizes)
        e = np.sort(np.stack([cv, cv[nxt]], axis=1), axis=1)
        return np.unique(e, axis=0)

    def _edge_table(self):
        """Per-corner edge ids (edge from corner to next corner) + edge array + face counts."""
        cv, cf, start, sizes = self._corners()
        nxt, prv = self._corner_next(start, sizes)
        e = np.sort(np.stack([cv, cv[nxt]], axis=1), axis=1)
        edges, inv = np.unique(e, axis=0, return_inverse=True)
        inv = inv.reshape(-1)
        counts = np.bincount(inv, minlength=len(edges))
        return cv, cf, start, sizes, nxt, prv, edges, inv, counts

    def boundary_edges(self) -> np.ndarray:
        *_, edges, inv, counts = self._edge_table()
        return edges[counts == 1]

    # ------------------------------------------------------------ transforms
    def transform(self, matrix) -> "Mesh":
        mtx = np.asarray(matrix, dtype=np.float64)
        self.vertices = m3.transform_points(mtx, self.vertices)
        for k in self.shape_keys:
            self.shape_keys[k] = m3.transform_points(mtx, self.shape_keys[k])
        if np.linalg.det(mtx[:3, :3]) < 0:
            self._flip_winding()
        return self

    def translate(self, offset) -> "Mesh":
        return self.transform(m3.translation_matrix(offset))

    def rotate(self, rotation, pivot=None) -> "Mesh":
        """rotation: Euler degrees (x, y, z) or quaternion (x, y, z, w)."""
        mtx = m3.rotation_matrix(rotation)
        if pivot is not None:
            p = m3.vec3(pivot)
            mtx = m3.translation_matrix(p) @ mtx @ m3.translation_matrix(-p)
        return self.transform(mtx)

    def scale(self, factor, pivot=None) -> "Mesh":
        mtx = m3.scale_matrix(factor)
        if pivot is not None:
            p = m3.vec3(pivot)
            mtx = m3.translation_matrix(p) @ mtx @ m3.translation_matrix(-p)
        return self.transform(mtx)

    def resize(self, size, uniform: bool = True) -> "Mesh":
        """Scale so the bounding box has the given size (scalar = largest dimension)."""
        cur = self.size
        target = np.asarray(size, dtype=np.float64)
        if target.size == 1:
            f = float(target) / max(float(cur.max()), 1e-12)
            return self.scale(f, pivot=self.center)
        target = m3.vec3(target)
        f = np.where(cur > 1e-12, target / np.maximum(cur, 1e-12), 1.0)
        if uniform:
            f = np.full(3, f[cur > 1e-12].min())
        return self.scale(f, pivot=self.center)

    def center_origin(self, axes: str = "xyz") -> "Mesh":
        c = self.center
        off = np.array([-c[i] if "xyz"[i] in axes else 0.0 for i in range(3)])
        return self.translate(off)

    def ground(self, y: float = 0.0, center_xz: bool = False) -> "Mesh":
        """Move so the lowest point sits at height y (optionally centered on X/Z)."""
        lo, _ = self.bounds()
        if center_xz:
            self.center_origin("xz")
        return self.translate((0, y - lo[1], 0))

    def _flip_winding(self):
        self.faces = [tuple(reversed(f)) for f in self.faces]
        if self.face_uvs is not None:
            self.face_uvs = [tuple(reversed(f)) for f in self.face_uvs]

    def flip(self, faces: FaceSel = None) -> "Mesh":
        """Reverse face winding (flips normals)."""
        if faces is None:
            self._flip_winding()
            return self
        for i in _sel(faces, len(self.faces)):
            self.faces[i] = tuple(reversed(self.faces[i]))
            if self.face_uvs is not None:
                self.face_uvs[i] = tuple(reversed(self.face_uvs[i]))
        return self

    # --------------------------------------------------------------- merging
    def merge(self, *others: "Mesh") -> "Mesh":
        """Append other meshes into this one (materials and attributes are combined)."""
        for o in others:
            off = len(self.vertices)
            n_self = len(self.vertices)
            # vertex attributes
            if self.colors is not None or o.colors is not None:
                a = self.colors if self.colors is not None else np.ones((n_self, 4))
                b = o.colors if o.colors is not None else np.ones((len(o.vertices), 4))
                self.colors = np.concatenate([a, b])
            if self.joints is not None or o.joints is not None:
                ja = self.joints if self.joints is not None else np.zeros((n_self, 4), dtype=np.int64)
                jb = o.joints if o.joints is not None else np.zeros((len(o.vertices), 4), dtype=np.int64)
                wa = self.weights if self.weights is not None else _default_weights(n_self)
                wb = o.weights if o.weights is not None else _default_weights(len(o.vertices))
                self.joints = np.concatenate([ja, jb])
                self.weights = np.concatenate([wa, wb])
            keys = set(self.shape_keys) | set(o.shape_keys)
            for k in keys:
                a = self.shape_keys.get(k, self.vertices)
                b = o.shape_keys.get(k, o.vertices)
                self.shape_keys[k] = np.concatenate([a, b])
            self.vertices = np.concatenate([self.vertices, o.vertices])
            # materials
            mat_map = []
            for mat in (o.materials or [None]):
                if mat is None:
                    if not self.materials:
                        self.materials.append(None)
                    mat_map.append(self.materials.index(None) if None in self.materials else 0)
                    continue
                for j, existing in enumerate(self.materials):
                    if existing is mat:
                        mat_map.append(j)
                        break
                else:
                    self.materials.append(mat)
                    mat_map.append(len(self.materials) - 1)
            if not self.materials and o.materials:
                pass
            self.face_materials = np.concatenate([
                self.face_materials, np.array(mat_map, dtype=np.int64)[o.face_materials] if len(o.faces) else
                np.zeros(0, dtype=np.int64)])
            # uvs
            if self.uvs is not None or o.uvs is not None:
                if self.uvs is None:
                    self.uvs = np.zeros((1, 2))
                    self.face_uvs = [tuple(0 for _ in f) for f in self.faces]
                uv_off = len(self.uvs)
                if o.uvs is None:
                    self.uvs = np.concatenate([self.uvs, np.zeros((1, 2))])
                    self.face_uvs += [tuple(uv_off for _ in f) for f in o.faces]
                else:
                    self.uvs = np.concatenate([self.uvs, o.uvs])
                    self.face_uvs += [tuple(i + uv_off for i in f) for f in o.face_uvs]
            self.faces += [tuple(i + off for i in f) for f in o.faces]
        self._normalize_materials()
        return self

    def _normalize_materials(self):
        if self.materials and all(m is None for m in self.materials):
            self.materials = []
            self.face_materials[:] = 0

    @staticmethod
    def join(meshes: Sequence["Mesh"], name: str = "Mesh") -> "Mesh":
        out = meshes[0].copy(name=name)
        if len(meshes) > 1:
            out.merge(*meshes[1:])
        return out

    # ------------------------------------------------------- vertex utilities
    def _append_vertices(self, pos: np.ndarray, parents: np.ndarray, parent_w: np.ndarray | None = None) -> np.ndarray:
        """Add vertices whose attributes are interpolated from `parents` (K, m) indices."""
        pos = np.asarray(pos, dtype=np.float64).reshape(-1, 3)
        parents = np.asarray(parents, dtype=np.int64).reshape(len(pos), -1)
        if parent_w is None:
            parent_w = np.full(parents.shape, 1.0 / parents.shape[1])
        start = len(self.vertices)
        if self.colors is not None:
            c = (self.colors[parents] * parent_w[..., None]).sum(axis=1)
            self.colors = np.concatenate([self.colors, c])
        if self.joints is not None:
            dom = parents[np.arange(len(parents)), np.argmax(parent_w, axis=1)]
            self.joints = np.concatenate([self.joints, self.joints[dom]])
            self.weights = np.concatenate([self.weights, self.weights[dom]])
        for k, sk in self.shape_keys.items():
            delta = ((sk[parents] - self.vertices[parents]) * parent_w[..., None]).sum(axis=1)
            self.shape_keys[k] = np.concatenate([sk, pos + delta])
        self.vertices = np.concatenate([self.vertices, pos])
        return np.arange(start, start + len(pos))

    def remove_unused_vertices(self) -> "Mesh":
        used = np.zeros(len(self.vertices), dtype=bool)
        if self.faces:
            cv, *_ = self._corners()
            used[cv] = True
        if used.all():
            return self
        remap = np.cumsum(used) - 1
        self.vertices = self.vertices[used]
        if self.colors is not None:
            self.colors = self.colors[used]
        if self.joints is not None:
            self.joints = self.joints[used]
            self.weights = self.weights[used]
        self.shape_keys = {k: v[used] for k, v in self.shape_keys.items()}
        self.faces = [tuple(int(remap[i]) for i in f) for f in self.faces]
        return self

    def weld(self, threshold: float = 1e-5) -> "Mesh":
        """Merge vertices closer than `threshold` (merge by distance)."""
        if len(self.vertices) == 0:
            return self
        q = np.round(self.vertices / max(threshold, 1e-12)).astype(np.int64)
        _, first, inv = np.unique(q, axis=0, return_index=True, return_inverse=True)
        inv = inv.reshape(-1)
        remap = first[inv]
        new_faces, new_fuv, new_mat = [], [], []
        for fi, f in enumerate(self.faces):
            nf, nuv = [], []
            fuv = self.face_uvs[fi] if self.face_uvs is not None else None
            for k, v in enumerate(f):
                rv = int(remap[v])
                if nf and nf[-1] == rv:
                    continue
                nf.append(rv)
                if fuv is not None:
                    nuv.append(fuv[k])
            if len(nf) > 1 and nf[0] == nf[-1]:
                nf.pop()
                if fuv is not None:
                    nuv.pop()
            if len(set(nf)) >= 3:
                new_faces.append(tuple(nf))
                new_mat.append(self.face_materials[fi])
                if fuv is not None:
                    new_fuv.append(tuple(nuv))
        self.faces = new_faces
        self.face_materials = np.array(new_mat, dtype=np.int64)
        if self.face_uvs is not None:
            self.face_uvs = new_fuv
        return self.remove_unused_vertices()

    def delete_faces(self, faces: FaceSel) -> "Mesh":
        kill = set(int(i) for i in _sel(faces, len(self.faces)))
        keep = [i for i in range(len(self.faces)) if i not in kill]
        self.faces = [self.faces[i] for i in keep]
        if self.face_uvs is not None:
            self.face_uvs = [self.face_uvs[i] for i in keep]
        self.face_materials = self.face_materials[keep] if keep else np.zeros(0, dtype=np.int64)
        return self.remove_unused_vertices()

    def separate(self, faces: FaceSel, name: str | None = None) -> "Mesh":
        """Return a new mesh containing only the selected faces (this mesh is unchanged)."""
        sel = set(int(i) for i in _sel(faces, len(self.faces)))
        out = self.copy(name=name or self.name + "_part")
        out.delete_faces([i for i in range(len(self.faces)) if i not in sel])
        return out

    # ------------------------------------------------------------ selection
    def faces_facing(self, direction, angle: float = 30.0) -> np.ndarray:
        """Indices of faces whose normal is within `angle` degrees of `direction`."""
        d = m3.normalize(m3.vec3(direction))
        return np.flatnonzero(self.face_normals() @ d >= math.cos(math.radians(angle)) - 1e-9)

    def faces_where(self, predicate: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """predicate(centers (F,3), normals (F,3)) -> bool mask. Returns face indices."""
        return np.flatnonzero(np.asarray(predicate(self.face_centers(), self.face_normals()), dtype=bool))

    def faces_in_box(self, lo, hi) -> np.ndarray:
        c = self.face_centers()
        return np.flatnonzero(np.all((c >= m3.vec3(lo)) & (c <= m3.vec3(hi)), axis=1))

    def verts_where(self, predicate: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        return np.flatnonzero(np.asarray(predicate(self.vertices), dtype=bool))

    # -------------------------------------------------------------- modeling
    def triangulate(self) -> "Mesh":
        tris, tri_face = triangulate_corners(self)
        cv, *_ = self._corners()
        cuv = self._corner_uvs()
        self.faces = [tuple(int(cv[c]) for c in t) for t in tris]
        if cuv is not None:
            self.face_uvs = [tuple(int(cuv[c]) for c in t) for t in tris]
        self.face_materials = self.face_materials[tri_face]
        return self

    def subdivide(self, levels: int = 1, smooth: bool = True) -> "Mesh":
        """Catmull-Clark subdivision (smooth=True) or simple linear subdivision."""
        for _ in range(int(levels)):
            self._subdivide_once(smooth)
        return self

    def _subdivide_once(self, smooth: bool):
        if not self.faces:
            return
        cv, cf, start, sizes, nxt, prv, edges, cedge, ecount = self._edge_table()
        nv, ne, nf = len(self.vertices), len(edges), len(self.faces)

        def cc_positions(P):
            fp = np.zeros((nf, 3))
            np.add.at(fp, cf, P[cv])
            fp /= sizes[:, None]
            mid = 0.5 * (P[edges[:, 0]] + P[edges[:, 1]])
            if not smooth:
                return P.copy(), mid, fp
            fsum = np.zeros((ne, 3))
            np.add.at(fsum, cedge, fp[cf])
            ep = mid.copy()
            interior = ecount == 2
            ep[interior] = (P[edges[interior, 0]] + P[edges[interior, 1]] + fsum[interior]) / 4.0
            # vertices
            boundary_e = ecount == 1
            vb = np.zeros(nv, dtype=bool)
            vb[edges[boundary_e].reshape(-1)] = True
            nonman = ecount > 2
            vnm = np.zeros(nv, dtype=bool)
            vnm[edges[nonman].reshape(-1)] = True
            # interior rule
            valence_f = np.bincount(cv, minlength=nv).astype(np.float64)
            F = np.zeros((nv, 3))
            np.add.at(F, cv, fp[cf])
            R = np.zeros((nv, 3))
            val_e = np.zeros(nv)
            np.add.at(R, edges[:, 0], mid)
            np.add.at(R, edges[:, 1], mid)
            np.add.at(val_e, edges[:, 0], 1)
            np.add.at(val_e, edges[:, 1], 1)
            n = np.maximum(val_e, 1)
            newP = P.copy()
            ok = (~vb) & (~vnm) & (valence_f > 0)
            newP[ok] = (F[ok] / valence_f[ok, None] + 2 * R[ok] / n[ok, None] + (n[ok, None] - 3) * P[ok]) / n[ok, None]
            # boundary rule
            bsum = np.zeros((nv, 3))
            bcnt = np.zeros(nv)
            be = edges[boundary_e]
            np.add.at(bsum, be[:, 0], P[be[:, 1]])
            np.add.at(bsum, be[:, 1], P[be[:, 0]])
            np.add.at(bcnt, be[:, 0], 1)
            np.add.at(bcnt, be[:, 1], 1)
            bok = vb & (bcnt == 2) & (~vnm)
            newP[bok] = (6 * P[bok] + bsum[bok]) / 8.0
            return newP, ep, fp

        vP, eP, fP = cc_positions(self.vertices)
        new_vertices = np.concatenate([vP, eP, fP])
        new_keys = {}
        for k, sk in self.shape_keys.items():
            a, b, c = cc_positions(sk)
            new_keys[k] = np.concatenate([a, b, c])

        # faces: one quad per corner
        q0 = cv
        q1 = nv + cedge
        q2 = nv + ne + cf
        q3 = nv + cedge[prv]
        quads = np.stack([q0, q1, q2, q3], axis=1)

        # vertex attributes for new points
        if self.colors is not None:
            fc = np.zeros((nf, 4))
            np.add.at(fc, cf, self.colors[cv])
            fc /= sizes[:, None]
            ec = 0.5 * (self.colors[edges[:, 0]] + self.colors[edges[:, 1]])
            self.colors = np.concatenate([self.colors, ec, fc])
        if self.joints is not None:
            first_corner_v = cv[start]
            self.joints = np.concatenate([self.joints, self.joints[edges[:, 0]], self.joints[first_corner_v]])
            self.weights = np.concatenate([self.weights, self.weights[edges[:, 0]], self.weights[first_corner_v]])

        # uvs (linear)
        if self.face_uvs is not None:
            cuv = self._corner_uvs()
            uv_e = np.sort(np.stack([cuv, cuv[nxt]], axis=1), axis=1)
            uedges, uinv = np.unique(uv_e, axis=0, return_inverse=True)
            uinv = uinv.reshape(-1)
            nu = len(self.uvs)
            umid = 0.5 * (self.uvs[uedges[:, 0]] + self.uvs[uedges[:, 1]])
            ufp = np.zeros((nf, 2))
            np.add.at(ufp, cf, self.uvs[cuv])
            ufp /= sizes[:, None]
            self.uvs = np.concatenate([self.uvs, umid, ufp])
            uq = np.stack([cuv, nu + uinv, nu + len(uedges) + cf, nu + uinv[prv]], axis=1)
            self.face_uvs = [tuple(r) for r in uq.tolist()]

        self.vertices = new_vertices
        self.shape_keys = new_keys
        self.faces = [tuple(r) for r in quads.tolist()]
        self.face_materials = self.face_materials[cf]

    def extrude(self, faces: FaceSel = None, distance: float = 0.1, offset=None,
                individual: bool = False, return_faces: bool = False):
        """Extrude a region of faces along their normals (or by an `offset` vector).

        Side walls are created on the region boundary. Returns self, or the
        indices of the moved (cap) faces when return_faces=True.
        """
        sel = list(_sel(faces, len(self.faces)))
        if individual:
            caps = []
            for fi in sel:
                caps += list(self.extrude([fi], distance, offset, False, True))
            return np.array(caps) if return_faces else self
        if not sel:
            return np.zeros(0, dtype=np.int64) if return_faces else self
        fn = self.face_normals()
        region = set(sel)
        # directed edges in region
        edge_count: dict[tuple[int, int], int] = {}
        for fi in sel:
            f = self.faces[fi]
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                key = (min(a, b), max(a, b))
                edge_count[key] = edge_count.get(key, 0) + 1
        rverts = sorted(set(chain.from_iterable(self.faces[i] for i in sel)))
        # displacement per vertex
        if offset is not None:
            disp = np.tile(m3.vec3(offset), (len(rverts), 1))
        else:
            acc = {v: np.zeros(3) for v in rverts}
            for fi in sel:
                for v in self.faces[fi]:
                    acc[v] += fn[fi]
            disp = np.array([m3.normalize(acc[v]) * distance for v in rverts])
            if len(sel) > 1:  # keep wall thickness even on curved regions
                for idx, v in enumerate(rverts):
                    nrm = m3.normalize(acc[v])
                    cosang = [float(np.dot(nrm, fn[fi])) for fi in sel if v in self.faces[fi]]
                    c = max(min(cosang), 0.3)
                    disp[idx] /= c
        new_ids = self._append_vertices(self.vertices[rverts] + disp, np.array(rverts)[:, None])
        vmap = dict(zip(rverts, new_ids.tolist()))
        side_faces, side_mats, side_uv = [], [], []
        for fi in sel:
            f = self.faces[fi]
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                if edge_count[(min(a, b), max(a, b))] == 1:
                    side_faces.append((a, b, vmap[b], vmap[a]))
                    side_mats.append(self.face_materials[fi])
            self.faces[fi] = tuple(vmap[v] for v in f)
        if self.face_uvs is not None:
            base = len(self.uvs)
            self.uvs = np.concatenate([self.uvs, [[0, 0], [1, 0], [1, 1], [0, 1]]])
            self.face_uvs += [(base, base + 1, base + 2, base + 3)] * len(side_faces)
        self.faces += side_faces
        self.face_materials = np.concatenate([self.face_materials, np.array(side_mats, dtype=np.int64)])
        return np.array(sel) if return_faces else self

    def inset(self, faces: FaceSel = None, thickness: float = 0.05, depth: float = 0.0,
              return_faces: bool = False):
        """Inset each selected face individually (creates a border ring of quads).

        depth > 0 pushes the inner face outward along its normal, < 0 inward.
        """
        sel = list(_sel(faces, len(self.faces)))
        fn = self.face_normals()
        new_side, new_mats = [], []
        for fi in sel:
            f = list(self.faces[fi])
            P = self.vertices[f]
            n = len(f)
            # move each vertex along the bisector of its two edges
            inner = []
            for k in range(n):
                p, pp, pn = P[k], P[k - 1], P[(k + 1) % n]
                e1 = m3.normalize(pp - p)
                e2 = m3.normalize(pn - p)
                bis = e1 + e2
                if np.linalg.norm(bis) < 1e-9:
                    bis = np.cross(fn[fi], e2)
                bis = m3.normalize(bis)
                s = math.sin(math.acos(np.clip(np.dot(e1, e2), -1, 1)) / 2)
                inner.append(p + bis * thickness / max(s, 0.2) + fn[fi] * depth)
            ids = self._append_vertices(np.array(inner), np.array(f)[:, None]).tolist()
            for k in range(n):
                a, b = f[k], f[(k + 1) % n]
                new_side.append((a, b, ids[(k + 1) % n], ids[k]))
                new_mats.append(self.face_materials[fi])
            self.faces[fi] = tuple(ids)
        if self.face_uvs is not None and new_side:
            base = len(self.uvs)
            self.uvs = np.concatenate([self.uvs, [[0, 0], [1, 0], [1, 1], [0, 1]]])
            self.face_uvs += [(base, base + 1, base + 2, base + 3)] * len(new_side)
        self.faces += new_side
        self.face_materials = np.concatenate([self.face_materials, np.array(new_mats, dtype=np.int64)])
        return np.array(sel) if return_faces else self

    def mirror(self, axis: str = "x", weld: bool = True, threshold: float = 1e-4) -> "Mesh":
        """Mirror across the plane axis=0 and append (symmetric modeling)."""
        i = "xyz".index(axis.lower())
        other = self.copy()
        s = np.ones(3)
        s[i] = -1
        other.transform(m3.scale_matrix(s))
        n_orig = len(self.vertices)
        on_plane = np.abs(self.vertices[:, i]) < threshold
        self.merge(other)
        if weld:
            remap = np.arange(len(self.vertices))
            remap[n_orig:][on_plane] = np.flatnonzero(on_plane)
            self.faces = [tuple(int(remap[v]) for v in f) for f in self.faces]
            self.remove_unused_vertices()
        return self

    def solidify(self, thickness: float = 0.02) -> "Mesh":
        """Give an open surface (e.g. a leaf or cloth) thickness: inner shell + rim."""
        vn = self.vertex_normals()
        inner = self.copy()
        inner.vertices = self.vertices - vn * thickness
        inner._flip_winding()
        be = self._boundary_directed_edges()
        n = len(self.vertices)
        self.merge(inner)
        rim = [(b, a, a + n, b + n) for a, b in be]
        if self.face_uvs is not None and rim:
            base = len(self.uvs)
            self.uvs = np.concatenate([self.uvs, [[0, 0], [1, 0], [1, 1], [0, 1]]])
            self.face_uvs += [(base, base + 1, base + 2, base + 3)] * len(rim)
        self.faces += rim
        self.face_materials = np.concatenate([self.face_materials, np.zeros(len(rim), dtype=np.int64)])
        return self

    def _boundary_directed_edges(self) -> list[tuple[int, int]]:
        cnt: dict[tuple[int, int], int] = {}
        directed = []
        for f in self.faces:
            for k in range(len(f)):
                a, b = f[k], f[(k + 1) % len(f)]
                key = (min(a, b), max(a, b))
                cnt[key] = cnt.get(key, 0) + 1
                directed.append((a, b))
        return [(a, b) for a, b in directed if cnt[(min(a, b), max(a, b))] == 1]

    def array(self, count: int, offset=(1, 0, 0), rotation=None, scale=None) -> "Mesh":
        """Repeat the mesh `count` times, each copy offset/rotated/scaled from the previous."""
        base = self.copy()
        step = m3.trs_matrix(offset, rotation, scale if scale is not None else (1, 1, 1))
        mtx = np.eye(4)
        copies = []
        for _ in range(int(count) - 1):
            mtx = step @ mtx
            copies.append(base.copy().transform(mtx))
        return self.merge(*copies) if copies else self

    # ----------------------------------------------------------- deformers
    def deform(self, func: Callable[[np.ndarray], np.ndarray]) -> "Mesh":
        """Apply func(points (N,3)) -> (N,3) to all vertices (and shape keys)."""
        self.vertices = np.asarray(func(self.vertices.copy()), dtype=np.float64)
        for k in self.shape_keys:
            self.shape_keys[k] = np.asarray(func(self.shape_keys[k].copy()), dtype=np.float64)
        return self

    def smooth(self, iterations: int = 1, factor: float = 0.5, verts=None, keep_boundary: bool = True) -> "Mesh":
        """Laplacian smoothing (optionally restricted to a vertex selection)."""
        edges = self.edges()
        nv = len(self.vertices)
        deg = np.bincount(edges.reshape(-1), minlength=nv).astype(np.float64)
        mask = np.ones(nv, dtype=bool) if verts is None else np.isin(np.arange(nv), np.asarray(verts))
        if keep_boundary:
            be = self.boundary_edges()
            mask[be.reshape(-1)] = False
        for _ in range(int(iterations)):
            acc = np.zeros_like(self.vertices)
            np.add.at(acc, edges[:, 0], self.vertices[edges[:, 1]])
            np.add.at(acc, edges[:, 1], self.vertices[edges[:, 0]])
            avg = acc / np.maximum(deg, 1)[:, None]
            upd = mask & (deg > 0)
            self.vertices[upd] += factor * (avg[upd] - self.vertices[upd])
        return self

    def displace(self, amount: float = 0.05, scale: float = 2.0, octaves: int = 4, seed: int = 0,
                 func: Callable[[np.ndarray], np.ndarray] | None = None) -> "Mesh":
        """Push vertices along their normals by fBm noise (or by func(points)->(N,) heights)."""
        from .noise import fbm3
        vn = self.vertex_normals()
        h = func(self.vertices) if func is not None else fbm3(self.vertices * scale, octaves=octaves, seed=seed)
        self.vertices = self.vertices + vn * (np.asarray(h).reshape(-1, 1) * amount)
        return self

    def bend(self, angle: float, axis: str = "y", direction: str = "x", center=None) -> "Mesh":
        """Bend along `axis` (the long axis) toward `direction`; total bend `angle` degrees."""
        ai, di = "xyz".index(axis), "xyz".index(direction)
        lo, hi = self.bounds()
        length = max(hi[ai] - lo[ai], 1e-9)
        c = lo if center is None else m3.vec3(center)
        ang = math.radians(angle)
        if abs(ang) < 1e-9:
            return self
        radius = length / ang

        def f(p):
            t = (p[:, ai] - c[ai]) / length
            th = t * ang
            d = p[:, di] - c[di]
            r = radius - d
            out = p.copy()
            out[:, ai] = c[ai] + r * np.sin(th)
            out[:, di] = c[di] + radius - r * np.cos(th)
            return out
        return self.deform(f)

    def twist(self, angle: float, axis: str = "y") -> "Mesh":
        """Twist around `axis`, rotating `angle` degrees from bottom to top."""
        ai = "xyz".index(axis)
        lo, hi = self.bounds()
        length = max(hi[ai] - lo[ai], 1e-9)
        o = [k for k in range(3) if k != ai]
        c = self.center

        def f(p):
            th = np.radians(angle) * (p[:, ai] - lo[ai]) / length
            a = p[:, o[0]] - c[o[0]]
            b = p[:, o[1]] - c[o[1]]
            out = p.copy()
            sgn = 1 if ai != 1 else -1
            out[:, o[0]] = c[o[0]] + a * np.cos(th) - sgn * b * np.sin(th)
            out[:, o[1]] = c[o[1]] + sgn * a * np.sin(th) + b * np.cos(th)
            return out
        return self.deform(f)

    def taper(self, factor: float, axis: str = "y", curve: float = 1.0) -> "Mesh":
        """Scale cross-sections along `axis`: 1 at the bottom to `factor` at the top."""
        ai = "xyz".index(axis)
        lo, hi = self.bounds()
        length = max(hi[ai] - lo[ai], 1e-9)
        c = self.center

        def f(p):
            t = ((p[:, ai] - lo[ai]) / length) ** curve
            s = 1 + (factor - 1) * t
            out = p.copy()
            for k in range(3):
                if k != ai:
                    out[:, k] = c[k] + (p[:, k] - c[k]) * s
            return out
        return self.deform(f)

    def spherify(self, amount: float = 1.0, center=None, radius: float | None = None) -> "Mesh":
        c = self.center if center is None else m3.vec3(center)
        d = self.vertices - c
        r = np.linalg.norm(d, axis=1)
        rad = r.mean() if radius is None else radius
        target = c + m3.normalize(d) * rad
        self.vertices = self.vertices + (target - self.vertices) * amount
        return self

    # ------------------------------------------------------------------ UVs
    def uv_box(self, scale: float = 1.0) -> "Mesh":
        """Box/tri-planar projection (good default for hard-surface objects)."""
        cv, cf, *_ = self._corners()
        fn = self.face_normals()[cf]
        p = self.vertices[cv] * scale
        ax = np.argmax(np.abs(fn), axis=1)
        sign = np.sign(fn[np.arange(len(fn)), ax])
        sign[sign == 0] = 1
        u = np.where(ax == 0, -sign * p[:, 2], np.where(ax == 1, p[:, 0], sign * p[:, 0]))
        v = np.where(ax == 1, -sign * p[:, 2], p[:, 1])
        self._set_corner_uvs(np.stack([u, v], axis=1))
        return self

    def uv_planar(self, axis: str = "y", fit: bool = True) -> "Mesh":
        cv, *_ = self._corners()
        i = "xyz".index(axis)
        a, b = {0: (2, 1), 1: (0, 2), 2: (0, 1)}[i]
        p = self.vertices[cv]
        uv = np.stack([p[:, a], p[:, b]], axis=1)
        if axis == "y":
            uv[:, 1] = -uv[:, 1]
        if fit:
            lo, hi = uv.min(axis=0), uv.max(axis=0)
            uv = (uv - lo) / np.maximum(hi - lo, 1e-12)
        self._set_corner_uvs(uv)
        return self

    def uv_cylinder(self, axis: str = "y") -> "Mesh":
        cv, cf, start, sizes = self._corners()
        i = "xyz".index(axis)
        o = [k for k in range(3) if k != i]
        c = self.center
        p = self.vertices[cv] - c
        u = (np.arctan2(p[:, o[0]], p[:, o[1]]) / (2 * math.pi)) % 1.0
        lo, hi = self.bounds()
        v = (p[:, i] + c[i] - lo[i]) / max(hi[i] - lo[i], 1e-12)
        u = _fix_seam(u, cf, len(self.faces))
        self._set_corner_uvs(np.stack([u, v], axis=1))
        return self

    def uv_sphere(self) -> "Mesh":
        cv, cf, *_ = self._corners()
        p = m3.normalize(self.vertices[cv] - self.center)
        u = (np.arctan2(p[:, 0], p[:, 2]) / (2 * math.pi)) % 1.0
        v = 0.5 + np.arcsin(np.clip(p[:, 1], -1, 1)) / math.pi
        u = _fix_seam(u, cf, len(self.faces))
        self._set_corner_uvs(np.stack([u, v], axis=1))
        return self

    def uv_transform(self, scale=(1, 1), offset=(0, 0), rotation: float = 0.0) -> "Mesh":
        if self.uvs is None:
            self.uv_box()
        c, s = math.cos(math.radians(rotation)), math.sin(math.radians(rotation))
        uv = self.uvs @ np.array([[c, s], [-s, c]])
        self.uvs = uv * np.asarray(scale, dtype=np.float64) + np.asarray(offset, dtype=np.float64)
        return self

    def _set_corner_uvs(self, uv: np.ndarray):
        self.uvs = np.asarray(uv, dtype=np.float64)
        _, _, start, sizes = self._corners()
        self.face_uvs = [tuple(range(s, s + n)) for s, n in zip(start.tolist(), sizes.tolist())]

    # ------------------------------------------------------- materials/colors
    def set_material(self, material, faces: FaceSel = None) -> "Mesh":
        """Assign a material to all faces or to a selection of faces."""
        if faces is None:
            self.materials = [material]
            self.face_materials = np.zeros(len(self.faces), dtype=np.int64)
            return self
        if not self.materials:
            self.materials = [None]  # None = default material for untouched faces
        for idx, existing in enumerate(self.materials):
            if existing is material:
                break
        else:
            self.materials.append(material)
            idx = len(self.materials) - 1
        sel = np.asarray(list(_sel(faces, len(self.faces))), dtype=np.int64)
        self.face_materials[sel] = idx
        return self

    def set_color(self, color, verts=None, func: Callable[[np.ndarray], np.ndarray] | None = None) -> "Mesh":
        """Paint vertex colors (sRGB input). `func(points)->(N,3|4)` sRGB for gradients."""
        if self.colors is None:
            self.colors = np.ones((len(self.vertices), 4))
        if func is not None:
            c = np.asarray(func(self.vertices), dtype=np.float64)
            if c.shape[1] == 3:
                c = np.concatenate([c, np.ones((len(c), 1))], axis=1)
            c[:, :3] = m3.srgb_to_linear(c[:, :3])
            idx = slice(None) if verts is None else np.asarray(verts)
            self.colors[idx] = c[idx]
            return self
        rgba = m3.as_color(color)
        rgba[:3] = m3.srgb_to_linear(rgba[:3])
        if verts is None:
            self.colors[:] = rgba
        else:
            self.colors[np.asarray(verts)] = rgba
        return self

    def shade_flat(self) -> "Mesh":
        self.smooth_angle = 0.0
        return self

    def shade_smooth(self, angle: float = 180.0) -> "Mesh":
        self.smooth_angle = angle
        return self

    # ------------------------------------------------------------- shape keys
    def add_shape_key(self, name: str, func: Callable[[np.ndarray], np.ndarray] | None = None,
                      positions=None) -> "Mesh":
        """Add a morph target (blend shape) e.g. for facial expressions.

        Either pass `positions` (N,3) or `func(points)->(N,3)` returning the
        deformed positions.
        """
        if positions is None:
            positions = func(self.vertices.copy()) if func is not None else self.vertices.copy()
        positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
        if positions.shape != self.vertices.shape:
            raise ValueError("shape key must have the same vertex count as the mesh")
        self.shape_keys[name] = positions
        return self

    # ------------------------------------------------------------ inspection
    def stats(self) -> dict:
        cv, cf, start, sizes, nxt, prv, edges, inv, counts = self._edge_table() if self.faces else (None,) * 9
        lo, hi = self.bounds()
        info = {
            "name": self.name,
            "vertices": len(self.vertices),
            "faces": len(self.faces),
            "triangles": self.num_triangles,
            "bounds_min": [round(float(x), 4) for x in lo],
            "bounds_max": [round(float(x), 4) for x in hi],
            "size": [round(float(x), 4) for x in hi - lo],
            "materials": [getattr(m, "name", "default") for m in self.materials] or ["default"],
            "has_uvs": self.uvs is not None,
            "has_colors": self.colors is not None,
            "skinned": self.joints is not None,
            "shape_keys": list(self.shape_keys),
        }
        if self.faces:
            info["boundary_edges"] = int((counts == 1).sum())
            info["non_manifold_edges"] = int((counts > 2).sum())
            info["watertight"] = bool((counts == 2).all())
            areas = self.face_areas()
            info["degenerate_faces"] = int((areas < 1e-12).sum())
            info["surface_area"] = round(float(areas.sum()), 5)
        return info

    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty list = OK)."""
        problems = []
        if len(self.vertices) and not np.isfinite(self.vertices).all():
            problems.append("non-finite vertex positions")
        nv = len(self.vertices)
        for i, f in enumerate(self.faces):
            if len(f) < 3:
                problems.append(f"face {i} has fewer than 3 vertices")
                break
            if min(f) < 0 or max(f) >= nv:
                problems.append(f"face {i} references a missing vertex")
                break
        if self.face_uvs is not None and len(self.face_uvs) != len(self.faces):
            problems.append("face_uvs length differs from faces")
        if len(self.face_materials) != len(self.faces):
            problems.append("face_materials length differs from faces")
        if self.joints is not None and (self.weights is None or len(self.joints) != nv):
            problems.append("skin data does not match vertex count")
        st = self.stats() if self.faces and not problems else {}
        if st.get("degenerate_faces"):
            problems.append(f"{st['degenerate_faces']} degenerate (zero-area) faces")
        if st.get("non_manifold_edges"):
            problems.append(f"{st['non_manifold_edges']} non-manifold edges")
        return problems


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _sel(faces: FaceSel, n: int):
    if faces is None:
        return range(n)
    a = np.asarray(faces)
    if a.dtype == bool:
        return np.flatnonzero(a).tolist()
    return [int(i) for i in a.reshape(-1)]


def _default_weights(n: int) -> np.ndarray:
    w = np.zeros((n, 4))
    w[:, 0] = 1.0
    return w


def _fix_seam(u: np.ndarray, cf: np.ndarray, nf: int) -> np.ndarray:
    umax = np.full(nf, -1.0)
    umin = np.full(nf, 2.0)
    np.maximum.at(umax, cf, u)
    np.minimum.at(umin, cf, u)
    wrap = (umax - umin) > 0.5
    fix = wrap[cf] & (u < 0.5)
    u = u.copy()
    u[fix] += 1.0
    return u


def _ear_clip(pts2: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping triangulation of a simple polygon (2D points, any winding)."""
    n = len(pts2)
    idx = list(range(n))
    area = 0.0
    for i in range(n):
        x1, y1 = pts2[i]
        x2, y2 = pts2[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    if area < 0:
        idx.reverse()
    tris = []
    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        m = len(idx)
        found = False
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = pts2[i0], pts2[i1], pts2[i2]
            cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
            if cross <= 1e-14:
                continue
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts2[j]
                d1 = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                d2 = (c[0] - b[0]) * (p[1] - b[1]) - (c[1] - b[1]) * (p[0] - b[0])
                d3 = (a[0] - c[0]) * (p[1] - c[1]) - (a[1] - c[1]) * (p[0] - c[0])
                if d1 >= 0 and d2 >= 0 and d3 >= 0:
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            found = True
            break
        if not found:  # degenerate: fall back to fan
            break
    if len(idx) >= 3:
        for k in range(1, len(idx) - 1):
            tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


def triangulate_corners(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate all faces. Returns (T,3) GLOBAL corner indices and (T,) source face ids."""
    cv, cf, start, sizes = mesh._corners()
    tris = []
    tface = []
    # triangles: fast path
    t_ids = np.flatnonzero(sizes == 3)
    if len(t_ids):
        s = start[t_ids]
        tris.append(np.stack([s, s + 1, s + 2], axis=1))
        tface.append(t_ids)
    q_ids = np.flatnonzero(sizes == 4)
    if len(q_ids):
        s = start[q_ids]
        P = mesh.vertices
        d02 = np.linalg.norm(P[cv[s]] - P[cv[s + 2]], axis=1)
        d13 = np.linalg.norm(P[cv[s + 1]] - P[cv[s + 3]], axis=1)
        use02 = d02 <= d13 + 1e-12
        t1 = np.where(use02[:, None], np.stack([s, s + 1, s + 2], 1), np.stack([s, s + 1, s + 3], 1))
        t2 = np.where(use02[:, None], np.stack([s, s + 2, s + 3], 1), np.stack([s + 1, s + 2, s + 3], 1))
        tris.append(np.concatenate([t1, t2]))
        tface.append(np.concatenate([q_ids, q_ids]))
    n_ids = np.flatnonzero(sizes > 4)
    if len(n_ids):
        fn = mesh.face_normals()
        extra, ef = [], []
        for fi in n_ids:
            s, n = int(start[fi]), int(sizes[fi])
            P = mesh.vertices[cv[s:s + n]]
            nrm = fn[fi]
            ax = np.argmax(np.abs(nrm))
            o = [k for k in range(3) if k != ax]
            for a, b, c in _ear_clip(P[:, o]):
                # keep the winding of the original face
                if np.dot(np.cross(P[b] - P[a], P[c] - P[a]), nrm) < 0:
                    b, c = c, b
                extra.append((s + a, s + b, s + c))
                ef.append(fi)
        tris.append(np.array(extra, dtype=np.int64).reshape(-1, 3))
        tface.append(np.array(ef, dtype=np.int64))
    if not tris:
        return np.zeros((0, 3), dtype=np.int64), np.zeros(0, dtype=np.int64)
    T = np.concatenate(tris).astype(np.int64)
    F = np.concatenate(tface).astype(np.int64)
    order = np.argsort(F, kind="stable")
    return T[order], F[order]
