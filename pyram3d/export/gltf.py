"""glTF 2.0 exporter (.gltf with embedded data, or binary .glb).

Supports: node hierarchy, meshes with multiple materials, normals, UVs,
vertex colors, PBR materials and textures, KHR_materials_unlit,
KHR_materials_emissive_strength, KHR_lights_punctual, cameras, skins
(skeletal rigs), morph targets (shape keys) and animations.
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import numpy as np

from .. import __version__
from ..image import encode_png
from ..material import Material
from .common import compute_tangents, mesh_primitives

FLOAT, UINT, USHORT, UBYTE = 5126, 5125, 5123, 5121
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
_TYPES = {1: "SCALAR", 2: "VEC2", 3: "VEC3", 4: "VEC4", 16: "MAT4"}
_WRAP = {"repeat": 10497, "clamp": 33071, "mirror": 33648}


class _Builder:
    def __init__(self):
        self.blob = bytearray()
        self.views: list[dict] = []
        self.accessors: list[dict] = []

    def view(self, data: bytes, target: int | None = None, stride: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        v = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            v["target"] = target
        if stride:
            v["byteStride"] = stride
        self.blob.extend(data)
        self.views.append(v)
        return len(self.views) - 1

    def accessor(self, arr: np.ndarray, comp: int, target: int | None = None, minmax: bool = False,
                 normalized: bool = False) -> int:
        arr = np.ascontiguousarray(arr)
        ncomp = 1 if arr.ndim == 1 else int(np.prod(arr.shape[1:]))
        flat = arr.reshape(len(arr), ncomp) if arr.ndim > 1 else arr.reshape(-1, 1)
        dtype = {FLOAT: np.float32, UINT: np.uint32, USHORT: np.uint16, UBYTE: np.uint8}[comp]
        data = flat.astype(dtype).tobytes()
        acc = {"bufferView": self.view(data, target), "componentType": comp, "count": int(len(arr)),
               "type": _TYPES[ncomp]}
        if normalized:
            acc["normalized"] = True
        if minmax and len(arr):
            f = flat.astype(np.float64)
            if comp == FLOAT:
                acc["min"] = [float(np.float32(x)) for x in f.min(axis=0)]
                acc["max"] = [float(np.float32(x)) for x in f.max(axis=0)]
            else:
                acc["min"] = [int(x) for x in f.min(axis=0)]
                acc["max"] = [int(x) for x in f.max(axis=0)]
        self.accessors.append(acc)
        return len(self.accessors) - 1


def build_gltf(scene, bake_fps: float | None = None) -> tuple[dict, bytes]:
    b = _Builder()
    doc: dict = {"asset": {"version": "2.0", "generator": f"Pyram3D {__version__}"}}
    ext_used: set[str] = set()

    # ------------------------------------------------------------ nodes
    all_nodes = list(scene.walk())
    node_index = {id(n): i for i, n in enumerate(all_nodes)}

    # ------------------------------------------------------------ materials
    materials: list[Material] = []
    mat_index: dict[int, int] = {}
    default_mat = None

    def material_id(m: Material | None) -> int:
        nonlocal default_mat
        if m is None:
            if default_mat is None:
                default_mat = Material("Default")
            m = default_mat
        if id(m) not in mat_index:
            mat_index[id(m)] = len(materials)
            materials.append(m)
        return mat_index[id(m)]

    # ------------------------------------------------------------ meshes
    gl_meshes = []
    mesh_index: dict[int, int] = {}
    for n in all_nodes:
        mesh = n.mesh
        if mesh is None or id(mesh) in mesh_index or not mesh.faces:
            continue
        prims = []
        for p in mesh_primitives(mesh, flip_v=True):
            attrs = {
                "POSITION": b.accessor(p["positions"], FLOAT, ARRAY_BUFFER, minmax=True),
                "NORMAL": b.accessor(p["normals"], FLOAT, ARRAY_BUFFER),
            }
            if p["uvs"] is not None:
                attrs["TEXCOORD_0"] = b.accessor(p["uvs"], FLOAT, ARRAY_BUFFER)
            if p["colors"] is not None:
                attrs["COLOR_0"] = b.accessor(p["colors"], FLOAT, ARRAY_BUFFER)
            if p["joints"] is not None:
                attrs["JOINTS_0"] = b.accessor(p["joints"], USHORT, ARRAY_BUFFER)
                attrs["WEIGHTS_0"] = b.accessor(p["weights"], FLOAT, ARRAY_BUFFER)
            idx = p["indices"]
            mat_ = mesh.materials[p["material"]] if p["material"] >= 0 else None
            if mat_ is not None and mat_.normal_texture is not None and p["uvs"] is not None:
                attrs["TANGENT"] = b.accessor(compute_tangents(p["positions"], p["normals"], p["uvs"], idx),
                                              FLOAT, ARRAY_BUFFER)
            comp = USHORT if len(p["positions"]) < 65536 else UINT
            prim = {"attributes": attrs, "indices": b.accessor(idx, comp, ELEMENT_ARRAY_BUFFER), "mode": 4}
            mat = mesh.materials[p["material"]] if p["material"] >= 0 else None
            prim["material"] = material_id(mat)
            if p["targets"]:
                prim["targets"] = [{"POSITION": b.accessor(d, FLOAT, ARRAY_BUFFER, minmax=True)}
                                   for d in p["targets"].values()]
            prims.append(prim)
        gm = {"name": mesh.name, "primitives": prims}
        if mesh.shape_keys:
            gm["weights"] = [0.0] * len(mesh.shape_keys)
            gm["extras"] = {"targetNames": list(mesh.shape_keys)}
        mesh_index[id(mesh)] = len(gl_meshes)
        gl_meshes.append(gm)

    # ------------------------------------------------------------ skins
    skins = []
    skin_index: dict[int, int] = {}
    for n in all_nodes:
        sk = n.skin
        if sk is None or id(sk) in skin_index:
            continue
        ibm = sk.inverse_bind_matrices()
        # glTF matrices are column-major
        ibm_cols = np.array([m.T.reshape(-1) for m in ibm], dtype=np.float32)
        skin = {"name": sk.name, "joints": [node_index[id(bn)] for bn in sk.bones],
                "inverseBindMatrices": b.accessor(ibm_cols.reshape(-1, 16), FLOAT)}
        if id(sk.root) in node_index:
            skin["skeleton"] = node_index[id(sk.root)]
        skin_index[id(sk)] = len(skins)
        skins.append(skin)

    # ------------------------------------------------------------ lights / cameras
    lights, cameras = [], []
    gl_nodes = []
    for n in all_nodes:
        gn: dict = {"name": n.name}
        if np.linalg.norm(n.translation) > 0:
            gn["translation"] = [float(x) for x in n.translation]
        if abs(n.rotation[3] - 1.0) > 1e-9:
            gn["rotation"] = [float(x) for x in n.rotation]
        if np.any(np.abs(n.scale - 1) > 1e-12):
            gn["scale"] = [float(x) for x in n.scale]
        if n.children:
            gn["children"] = [node_index[id(c)] for c in n.children]
        if n.mesh is not None and id(n.mesh) in mesh_index:
            gn["mesh"] = mesh_index[id(n.mesh)]
            if n.skin is not None:
                gn["skin"] = skin_index[id(n.skin)]
            if n.morph_weights is not None and n.mesh.shape_keys:
                gn["weights"] = [float(x) for x in n.morph_weights]
        if n.light is not None:
            L = n.light
            ld: dict = {"name": L.name, "type": L.type, "color": [float(x) for x in L.color_linear],
                        "intensity": float(L.intensity)}
            if L.range is not None and L.type != "directional":
                ld["range"] = float(L.range)
            if L.type == "spot":
                ld["spot"] = {"innerConeAngle": math.radians(min(L.inner_cone, L.outer_cone - 0.01)),
                              "outerConeAngle": math.radians(L.outer_cone)}
            gn.setdefault("extensions", {})["KHR_lights_punctual"] = {"light": len(lights)}
            lights.append(ld)
            ext_used.add("KHR_lights_punctual")
        if n.camera is not None:
            C = n.camera
            if C.type == "orthographic":
                asp = C.aspect or 16 / 9
                cd = {"type": "orthographic", "orthographic": {"xmag": C.ortho_height * asp, "ymag": C.ortho_height,
                                                               "znear": C.znear, "zfar": C.zfar}}
            else:
                cd = {"type": "perspective", "perspective": {"yfov": math.radians(C.yfov), "znear": C.znear,
                                                             "zfar": C.zfar}}
                if C.aspect:
                    cd["perspective"]["aspectRatio"] = C.aspect
            cd["name"] = C.name
            gn["camera"] = len(cameras)
            cameras.append(cd)
        if n.extras:
            ex = {k: v for k, v in n.extras.items() if isinstance(v, (str, int, float, bool))}
            if ex:
                gn["extras"] = ex
        gl_nodes.append(gn)

    # ------------------------------------------------------------ materials (after meshes)
    images, textures, samplers = [], [], []
    tex_index: dict[int, int] = {}

    def texture_id(t) -> int:
        if id(t) in tex_index:
            return tex_index[id(t)]
        png = encode_png(t.image)
        images.append({"name": t.name, "mimeType": "image/png", "bufferView": b.view(png)})
        flt = 9728 if t.filter == "nearest" else 9729
        samplers.append({"magFilter": flt, "minFilter": 9728 if t.filter == "nearest" else 9987,
                         "wrapS": _WRAP.get(t.wrap, 10497), "wrapT": _WRAP.get(t.wrap, 10497)})
        textures.append({"source": len(images) - 1, "sampler": len(samplers) - 1})
        tex_index[id(t)] = len(textures) - 1
        return tex_index[id(t)]

    gl_mats = []
    for m in materials:
        pbr = {"baseColorFactor": [float(x) for x in m.base_color_linear], "metallicFactor": float(m.metallic),
               "roughnessFactor": float(m.roughness)}
        md: dict = {"name": m.name, "pbrMetallicRoughness": pbr}
        if m.base_color_texture is not None:
            pbr["baseColorTexture"] = {"index": texture_id(m.base_color_texture)}
        if m.metallic_roughness_texture is not None:
            pbr["metallicRoughnessTexture"] = {"index": texture_id(m.metallic_roughness_texture)}
        if m.normal_texture is not None:
            md["normalTexture"] = {"index": texture_id(m.normal_texture), "scale": float(m.normal_scale)}
        if m.occlusion_texture is not None:
            md["occlusionTexture"] = {"index": texture_id(m.occlusion_texture)}
        em = m.emissive_linear
        if em.max() > 0 or m.emissive_texture is not None:
            md["emissiveFactor"] = [float(x) for x in (em if em.max() > 0 else np.ones(3))]
            if m.emissive_texture is not None:
                md["emissiveTexture"] = {"index": texture_id(m.emissive_texture)}
            if m.emissive_strength > 1.0:
                md.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {
                    "emissiveStrength": float(m.emissive_strength)}
                ext_used.add("KHR_materials_emissive_strength")
        mode = m.resolved_alpha_mode
        if mode != "OPAQUE":
            md["alphaMode"] = mode
            if mode == "MASK":
                md["alphaCutoff"] = float(m.alpha_cutoff)
        if m.double_sided:
            md["doubleSided"] = True
        if m.unlit:
            md.setdefault("extensions", {})["KHR_materials_unlit"] = {}
            ext_used.add("KHR_materials_unlit")
        gl_mats.append(md)

    # ------------------------------------------------------------ animations
    gl_anims = []
    for anim in scene.animations:
        samplers_a, channels_a = [], []
        for ch in anim.channels:
            if not ch.times or id(ch.node) not in node_index:
                continue
            times, values = ch.baked(bake_fps or anim.fps)
            interp = "STEP" if ch.interpolation == "STEP" else "LINEAR"
            vals = np.asarray(values, dtype=np.float32)
            if ch.path == "weights":
                vals = vals.reshape(-1)
            inp = b.accessor(np.asarray(times, dtype=np.float32), FLOAT, minmax=True)
            out = b.accessor(vals, FLOAT)
            samplers_a.append({"input": inp, "output": out, "interpolation": interp})
            channels_a.append({"sampler": len(samplers_a) - 1,
                               "target": {"node": node_index[id(ch.node)], "path": ch.path}})
        if channels_a:
            gl_anims.append({"name": anim.name, "samplers": samplers_a, "channels": channels_a})

    # ------------------------------------------------------------ assemble
    doc["scene"] = 0
    doc["scenes"] = [{"name": scene.name, "nodes": [node_index[id(n)] for n in scene.root.children]}]
    doc["nodes"] = gl_nodes
    if gl_meshes:
        doc["meshes"] = gl_meshes
    if gl_mats:
        doc["materials"] = gl_mats
    if textures:
        doc["textures"] = textures
        doc["images"] = images
        doc["samplers"] = samplers
    if skins:
        doc["skins"] = skins
    if cameras:
        doc["cameras"] = cameras
    if gl_anims:
        doc["animations"] = gl_anims
    if lights:
        doc["extensions"] = {"KHR_lights_punctual": {"lights": lights}}
    if ext_used:
        doc["extensionsUsed"] = sorted(ext_used)
    while len(b.blob) % 4:
        b.blob.append(0)
    doc["accessors"] = b.accessors
    doc["bufferViews"] = b.views
    doc["buffers"] = [{"byteLength": len(b.blob)}]
    return doc, bytes(b.blob)


def export_gltf(scene, path, embed: bool = True, bake_fps: float | None = None) -> Path:
    """Write .glb (binary) or .gltf (JSON; embed=True puts the buffer in a data URI,
    embed=False writes a sibling .bin file)."""
    path = Path(path)
    doc, blob = build_gltf(scene, bake_fps)
    if path.suffix.lower() == ".glb":
        js = json.dumps(doc, separators=(",", ":")).encode("utf-8")
        js += b" " * ((4 - len(js) % 4) % 4)
        bin_ = blob + b"\x00" * ((4 - len(blob) % 4) % 4)
        total = 12 + 8 + len(js) + 8 + len(bin_)
        out = bytearray(struct.pack("<4sII", b"glTF", 2, total))
        out += struct.pack("<I4s", len(js), b"JSON") + js
        out += struct.pack("<I4s", len(bin_), b"BIN\x00") + bin_
        path.write_bytes(bytes(out))
    else:
        if embed:
            doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(blob).decode()
        else:
            bin_path = path.with_suffix(".bin")
            bin_path.write_bytes(blob)
            doc["buffers"][0]["uri"] = bin_path.name
        path.write_text(json.dumps(doc, indent=1))
    return path
