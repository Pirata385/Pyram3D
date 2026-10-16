"""Wavefront OBJ + MTL exporter (geometry, UVs, normals, basic materials)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import math3d as m3
from ..image import write_png
from .common import mesh_primitives


def export_obj(scene, path) -> Path:
    path = Path(path)
    mtl_path = path.with_suffix(".mtl")
    lines = [f"# Pyram3D export", f"mtllib {mtl_path.name}"]
    mtl_lines = []
    mats_done = {}
    voff = 1
    for node in scene.mesh_nodes():
        mesh = node.mesh
        W = node.world_matrix()
        nm = np.linalg.inv(W[:3, :3]).T
        lines.append(f"o {node.name}")
        for p in mesh_primitives(mesh):
            mat = mesh.materials[p["material"]] if p["material"] >= 0 else None
            mname = getattr(mat, "name", "Default")
            if id(mat) not in mats_done:
                key = f"{mname}_{len(mats_done)}"
                mats_done[id(mat)] = key
                c = mat.base_color if mat is not None else "#cccccc"
                rgba = m3.as_color(c)
                mtl_lines += [f"newmtl {key}", f"Kd {rgba[0]:.4f} {rgba[1]:.4f} {rgba[2]:.4f}",
                              f"d {rgba[3] * (mat.alpha if mat else 1):.4f}", "illum 2"]
                if mat is not None and mat.base_color_texture is not None:
                    tex_name = f"{path.stem}_{key}.png"
                    write_png(path.parent / tex_name, mat.base_color_texture.image)
                    mtl_lines.append(f"map_Kd {tex_name}")
            pos = m3.transform_points(W, p["positions"])
            nrm = m3.normalize(p["normals"] @ nm.T)
            lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in pos]
            lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in nrm]
            has_uv = p["uvs"] is not None
            if has_uv:
                lines += [f"vt {u:.6f} {v:.6f}" for u, v in p["uvs"]]
            lines.append(f"usemtl {mats_done[id(mat)]}")
            tri = p["indices"].reshape(-1, 3) + voff
            if has_uv:
                lines += [f"f {a}/{a}/{a} {b}/{b}/{b} {c}/{c}/{c}" for a, b, c in tri]
            else:
                lines += [f"f {a}//{a} {b}//{b} {c}//{c}" for a, b, c in tri]
            voff += len(pos)
    path.write_text("\n".join(lines) + "\n")
    mtl_path.write_text("\n".join(mtl_lines) + "\n")
    return path
