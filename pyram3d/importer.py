"""Import existing models so they can be inspected, edited, rendered and re-exported.

Supported: .glb / .gltf (meshes, materials, textures, node hierarchy, lights,
cameras, node animations) and .obj (geometry + UVs).
"""
from __future__ import annotations

import base64
import json
import math
import struct
from pathlib import Path

import numpy as np

from . import math3d as m3
from .image import decode_image
from .material import Material, Texture
from .mesh import Mesh
from .scene import Camera, Light, Node, Scene

_COMP = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def load(path) -> Scene:
    """Load a model file into a new Scene."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in (".glb", ".gltf"):
        return load_gltf(path)
    if ext == ".obj":
        return load_obj(path)
    raise ValueError(f"unsupported import format {ext} (use .glb, .gltf or .obj)")


def load_gltf(path) -> Scene:
    path = Path(path)
    data = path.read_bytes()
    bin_chunk = None
    if data[:4] == b"glTF":
        _, _, length = struct.unpack("<4sII", data[:12])
        pos = 12
        doc = None
        while pos < length:
            clen, ctype = struct.unpack("<I4s", data[pos:pos + 8])
            chunk = data[pos + 8:pos + 8 + clen]
            if ctype == b"JSON":
                doc = json.loads(chunk.decode("utf-8"))
            elif ctype == b"BIN\x00":
                bin_chunk = chunk
            pos += 8 + clen
    else:
        doc = json.loads(data.decode("utf-8"))

    buffers = []
    for bf in doc.get("buffers", []):
        uri = bf.get("uri")
        if uri is None:
            buffers.append(bin_chunk)
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            buffers.append((path.parent / uri).read_bytes())

    def view_bytes(vi):
        v = doc["bufferViews"][vi]
        buf = buffers[v["buffer"]]
        off = v.get("byteOffset", 0)
        return buf[off:off + v["byteLength"]], v.get("byteStride")

    def accessor(ai):
        a = doc["accessors"][ai]
        dt = np.dtype(_COMP[a["componentType"]])
        nc = _NCOMP[a["type"]]
        count = a["count"]
        if "bufferView" not in a:
            arr = np.zeros((count, nc), dtype=dt)
        else:
            raw, stride = view_bytes(a["bufferView"])
            off = a.get("byteOffset", 0)
            item = dt.itemsize * nc
            if stride and stride != item:
                rows = [np.frombuffer(raw, dtype=dt, count=nc, offset=off + i * stride) for i in range(count)]
                arr = np.array(rows)
            else:
                arr = np.frombuffer(raw, dtype=dt, count=count * nc, offset=off).reshape(count, nc)
        arr = arr.astype(np.float64) if dt.kind == "f" else arr.astype(np.int64)
        if a.get("normalized") and dt.kind != "f":
            arr = arr / float(np.iinfo(dt).max)
        return arr if nc > 1 else arr.reshape(-1)

    # textures
    tex_cache: dict[int, Texture] = {}

    def texture(info, srgb=True):
        if info is None:
            return None
        ti = info["index"]
        if ti in tex_cache:
            return tex_cache[ti]
        t = doc["textures"][ti]
        img = doc["images"][t["source"]]
        if "bufferView" in img:
            raw, _ = view_bytes(img["bufferView"])
            raw = bytes(raw)
        elif img.get("uri", "").startswith("data:"):
            raw = base64.b64decode(img["uri"].split(",", 1)[1])
        else:
            raw = (path.parent / img["uri"]).read_bytes()
        try:
            arr = decode_image(raw, img.get("name", "image"))
        except ValueError:
            return None
        tex = Texture(arr, name=img.get("name", f"Image{ti}"), srgb=srgb)
        tex_cache[ti] = tex
        return tex

    mats = []
    for md in doc.get("materials", []):
        pbr = md.get("pbrMetallicRoughness", {})
        bc = np.array(pbr.get("baseColorFactor", [1, 1, 1, 1]), dtype=np.float64)
        srgb = m3.linear_to_srgb(bc[:3])
        em = m3.linear_to_srgb(np.array(md.get("emissiveFactor", [0, 0, 0]), dtype=np.float64))
        ext = md.get("extensions", {})
        mats.append(Material(
            name=md.get("name", "Material"), base_color=tuple(srgb), alpha=float(bc[3]),
            metallic=float(pbr.get("metallicFactor", 1.0)), roughness=float(pbr.get("roughnessFactor", 1.0)),
            emissive=tuple(em),
            emissive_strength=float(ext.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0)),
            alpha_mode=md.get("alphaMode"), alpha_cutoff=float(md.get("alphaCutoff", 0.5)),
            double_sided=bool(md.get("doubleSided", False)), unlit="KHR_materials_unlit" in ext,
            base_color_texture=texture(pbr.get("baseColorTexture")),
            metallic_roughness_texture=texture(pbr.get("metallicRoughnessTexture"), False),
            normal_texture=texture(md.get("normalTexture"), False),
            occlusion_texture=texture(md.get("occlusionTexture"), False),
            emissive_texture=texture(md.get("emissiveTexture")),
        ))

    meshes = []
    for gm in doc.get("meshes", []):
        parts = []
        for prim in gm["primitives"]:
            if prim.get("mode", 4) != 4:
                continue
            at = prim["attributes"]
            P = accessor(at["POSITION"])
            idx = accessor(prim["indices"]) if "indices" in prim else np.arange(len(P))
            faces = idx.reshape(-1, 3)
            uv = accessor(at["TEXCOORD_0"]) if "TEXCOORD_0" in at else None
            mesh = Mesh(P, faces.tolist(), name=gm.get("name", "Mesh"))
            if uv is not None:
                uv = uv.copy()
                uv[:, 1] = 1.0 - uv[:, 1]
                mesh.uvs = uv
                mesh.face_uvs = [tuple(f) for f in faces.tolist()]
            if "COLOR_0" in at:
                c = accessor(at["COLOR_0"])
                if c.shape[1] == 3:
                    c = np.concatenate([c, np.ones((len(c), 1))], axis=1)
                mesh.colors = c
            if "JOINTS_0" in at and "WEIGHTS_0" in at:
                mesh.joints = accessor(at["JOINTS_0"]).astype(np.int64)
                mesh.weights = accessor(at["WEIGHTS_0"])
            for ti, tgt in enumerate(prim.get("targets", [])):
                names = gm.get("extras", {}).get("targetNames", [])
                nm = names[ti] if ti < len(names) else f"Key{ti}"
                if "POSITION" in tgt:
                    mesh.shape_keys[nm] = P + accessor(tgt["POSITION"])
            if "material" in prim:
                mesh.materials = [mats[prim["material"]]]
            # detect flat vs smooth shading from the stored normals
            if "NORMAL" in at:
                Nn = accessor(at["NORMAL"])
                vn = mesh.vertex_normals()
                agree = np.einsum("ij,ij->i", m3.normalize(Nn), vn)
                mesh.smooth_angle = 180.0 if np.median(agree) > 0.999 else 45.0
            parts.append(mesh)
        if not parts:
            meshes.append(None)
            continue
        m = Mesh.join(parts, name=gm.get("name", "Mesh"))
        m.smooth_angle = parts[0].smooth_angle
        if m.joints is None and m.shape_keys == {}:
            m.weld(1e-6)
        meshes.append(m)

    lights_def = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    scene = Scene(path.stem)
    nodes = []
    for nd in doc.get("nodes", []):
        n = Node(nd.get("name", "Node"))
        if "matrix" in nd:
            mtx = np.array(nd["matrix"], dtype=np.float64).reshape(4, 4).T
            n.translation, n.rotation, n.scale = m3.decompose(mtx)
        else:
            n.translation = m3.vec3(nd.get("translation", [0, 0, 0]))
            n.rotation = m3.quat_normalize(np.array(nd.get("rotation", [0, 0, 0, 1]), dtype=np.float64))
            n.scale = m3.vec3(nd.get("scale", [1, 1, 1]))
        if "mesh" in nd and meshes[nd["mesh"]] is not None:
            n.mesh = meshes[nd["mesh"]]
        if "weights" in nd:
            n.morph_weights = list(nd["weights"])
        lp = nd.get("extensions", {}).get("KHR_lights_punctual")
        if lp is not None and lp["light"] < len(lights_def):
            ld = lights_def[lp["light"]]
            col = m3.linear_to_srgb(np.array(ld.get("color", [1, 1, 1]), dtype=np.float64))
            L = Light(type=ld.get("type", "point"), color=tuple(col), intensity=float(ld.get("intensity", 1.0)),
                      range=ld.get("range"), name=ld.get("name", "Light"))
            if L.type == "spot":
                sp = ld.get("spot", {})
                L.inner_cone = math.degrees(sp.get("innerConeAngle", 0.0))
                L.outer_cone = math.degrees(sp.get("outerConeAngle", math.pi / 4))
            n.light = L
        if "camera" in nd:
            cd = doc["cameras"][nd["camera"]]
            if cd["type"] == "perspective":
                pp = cd["perspective"]
                n.camera = Camera(yfov=math.degrees(pp["yfov"]), znear=pp.get("znear", 0.05),
                                  zfar=pp.get("zfar", 1000.0), aspect=pp.get("aspectRatio"))
            else:
                oo = cd["orthographic"]
                n.camera = Camera(type="orthographic", ortho_height=oo["ymag"], znear=oo["znear"], zfar=oo["zfar"])
        nodes.append(n)
    for i, nd in enumerate(doc.get("nodes", [])):
        for c in nd.get("children", []):
            nodes[i].add(nodes[c])
    sc = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {"nodes": [
        i for i, n in enumerate(nodes) if n.parent is None]}
    for i in sc.get("nodes", []):
        scene.add(nodes[i])

    # skins: rebuild a Skeleton-like binding (joints keep their imported rest transforms)
    from .armature import Skeleton
    for i, nd in enumerate(doc.get("nodes", [])):
        if "skin" in nd and nodes[i].mesh is not None:
            sd = doc["skins"][nd["skin"]]
            sk = Skeleton(sd.get("name", "Armature"))
            sk.root = nodes[sd["skeleton"]] if "skeleton" in sd else nodes[sd["joints"][0]].parent or nodes[
                sd["joints"][0]]
            sk.bones = [nodes[j] for j in sd["joints"]]
            ibm = accessor(sd["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1) \
                if "inverseBindMatrices" in sd else np.array([np.eye(4)] * len(sk.bones))
            for b, inv in zip(sk.bones, ibm):
                rest = np.linalg.inv(inv)
                sk._head[id(b)] = rest[:3, 3]
                sk._tail[id(b)] = rest[:3, 3] + rest[:3, 1] * 0.1
                b.extras["is_bone"] = True
                b.extras["rest_translation"] = b.translation.copy()
                b.extras["rest_rotation"] = b.rotation.copy()
                b.extras["rest_scale"] = b.scale.copy()
            sk._ibm = ibm
            nodes[i].skin = sk

    for ad in doc.get("animations", []):
        anim = scene.animation(ad.get("name", "Action"))
        for ch in ad["channels"]:
            tgt = ch["target"]
            if "node" not in tgt:
                continue
            smp = ad["samplers"][ch["sampler"]]
            times = accessor(smp["input"])
            vals = accessor(smp["output"])
            interp = smp.get("interpolation", "LINEAR")
            if interp == "CUBICSPLINE":
                vals = vals.reshape(len(times), 3, -1)[:, 1]
                interp = "LINEAR"
            path_ = tgt["path"]
            if path_ == "weights":
                vals = vals.reshape(len(times), -1)
            node = nodes[tgt["node"]]
            c = anim.channel(node, path_, interp)
            for t, v in zip(times, vals):
                c.set_key(float(t), np.asarray(v, dtype=np.float64))
    return scene


def load_obj(path) -> Scene:
    path = Path(path)
    V, VT, faces, fuvs = [], [], [], []
    for line in path.read_text(errors="ignore").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            V.append([float(x) for x in parts[1:4]])
        elif parts[0] == "vt":
            VT.append([float(x) for x in parts[1:3]])
        elif parts[0] == "f":
            f, fu = [], []
            for tok in parts[1:]:
                s = tok.split("/")
                vi = int(s[0])
                f.append(vi - 1 if vi > 0 else len(V) + vi)
                if len(s) > 1 and s[1]:
                    ti = int(s[1])
                    fu.append(ti - 1 if ti > 0 else len(VT) + ti)
            faces.append(f)
            fuvs.append(fu)
    has_uv = VT and all(len(fu) == len(f) for f, fu in zip(faces, fuvs))
    mesh = Mesh(V, faces, uvs=VT if has_uv else None, face_uvs=fuvs if has_uv else None, name=path.stem)
    scene = Scene(path.stem)
    scene.add(mesh)
    return scene
