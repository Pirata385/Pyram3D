"""FBX 7.4 binary exporter (no Autodesk SDK required).

Writes: node hierarchy, meshes (polygons kept as quads/n-gons), normals, UVs,
vertex colors, multiple materials per mesh, embedded textures, lights,
cameras, skeletons (LimbNodes) with skin clusters and bind pose, blend shapes
(shape keys) and baked animations (transform + blend shape weights).

Scene units: meters, Y up, -Z forward (same as glTF). UnitScaleFactor=100
tells FBX consumers (Unity, Blender, Maya...) that 1 unit = 1 meter.
"""
from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path

import numpy as np

from .. import __version__
from .. import math3d as m3
from ..image import encode_png

FBX_VERSION = 7400
KTIME = 46186158000  # FBX time units per second
_HEAD_MAGIC = b"Kaydara FBX Binary  \x00\x1a\x00"
_FILE_ID = b"\x28\xb3\x2a\xeb\xb6\x24\xcc\xc2\xbf\xc8\xb0\x2a\xa9\x2b\xfc\xf1"
_TIME_ID = b"1970-01-01 10:00:00:000"
_FOOT_ID = b"\xfa\xbc\xab\x09\xd0\xc8\xd4\x66\xb1\x76\xfb\x83\x1c\xf7\x26\x7e"
_FOOT_MAGIC = b"\xf8\x5a\x8c\x6a\xde\xf5\xd9\x7e\xec\xe9\x0c\xe3\x75\x8f\x29\x0b"
_SENTINEL = b"\x00" * 13


# ---------------------------------------------------------------------------
# low-level binary node encoding
# ---------------------------------------------------------------------------

class Elem:
    __slots__ = ("name", "props", "children")

    def __init__(self, name: str, *props, children=None):
        self.name = name
        self.props = list(props)
        self.children: list[Elem] = list(children or [])

    def add(self, name: str, *props) -> "Elem":
        e = Elem(name, *props)
        self.children.append(e)
        return e


class I32(int):
    pass


class I64(int):
    pass


class I16(int):
    pass


class F32(float):
    pass


class Raw(bytes):
    pass


class Arr:
    def __init__(self, code: str, data):
        self.code = code  # f d l i b
        self.data = np.asarray(data)


def _prop_bytes(p) -> bytes:
    if isinstance(p, bool):
        return b"C" + struct.pack("<?", p)
    if isinstance(p, I16):
        return b"Y" + struct.pack("<h", p)
    if isinstance(p, I64):
        return b"L" + struct.pack("<q", p)
    if isinstance(p, I32):
        return b"I" + struct.pack("<i", p)
    if isinstance(p, int):
        return b"I" + struct.pack("<i", p)
    if isinstance(p, F32):
        return b"F" + struct.pack("<f", p)
    if isinstance(p, float):
        return b"D" + struct.pack("<d", p)
    if isinstance(p, Raw):
        return b"R" + struct.pack("<I", len(p)) + bytes(p)
    if isinstance(p, bytes):
        return b"S" + struct.pack("<I", len(p)) + p
    if isinstance(p, str):
        b = p.encode("utf-8")
        return b"S" + struct.pack("<I", len(b)) + b
    if isinstance(p, Arr):
        dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "?"}[p.code]
        raw = np.ascontiguousarray(p.data.reshape(-1), dtype=dt).tobytes()
        n = int(p.data.size)
        if len(raw) > 128:
            comp = zlib.compress(raw, 6)
            return p.code.encode() + struct.pack("<III", n, 1, len(comp)) + comp
        return p.code.encode() + struct.pack("<III", n, 0, len(raw)) + raw
    raise TypeError(f"unsupported FBX property {type(p)}")


def _elem_bytes(e: Elem, offset: int) -> bytes:
    name = e.name.encode("ascii")
    props = b"".join(_prop_bytes(p) for p in e.props)
    head_len = 13 + len(name) + len(props)
    body = bytearray()
    pos = offset + head_len
    for c in e.children:
        cb = _elem_bytes(c, pos)
        body += cb
        pos += len(cb)
    if e.children or not e.props:
        body += _SENTINEL
    end = offset + head_len + len(body)
    return struct.pack("<IIIB", end, len(e.props), len(props), len(name)) + name + props + bytes(body)


def write_fbx_binary(path: Path, root_elems: list[Elem]):
    out = bytearray(_HEAD_MAGIC + struct.pack("<I", FBX_VERSION))
    for e in root_elems:
        out += _elem_bytes(e, len(out))
    out += _SENTINEL
    out += _FOOT_ID
    out += b"\x00" * 4
    pad = ((len(out) + 15) & ~15) - len(out)
    if pad == 0:
        pad = 16
    out += b"\x00" * pad
    out += struct.pack("<I", FBX_VERSION)
    out += b"\x00" * 120
    out += _FOOT_MAGIC
    path.write_bytes(bytes(out))


# ---------------------------------------------------------------------------
# property helpers
# ---------------------------------------------------------------------------

def _p(parent: Elem, name, ptype, label, flags, *values):
    parent.add("P", name, ptype, label, flags, *values)


def _p_vec(parent, name, ptype, label, flags, v):
    _p(parent, name, ptype, label, flags, float(v[0]), float(v[1]), float(v[2]))


def _name(name: str, cls: str) -> bytes:
    return name.encode("utf-8").replace(b"\x00", b"_") + b"\x00\x01" + cls.encode()


def _mat_array(m) -> Arr:
    """4x4 (column-vector convention) -> FBX 16 doubles (translation at 12..14)."""
    return Arr("d", np.asarray(m, dtype=np.float64).T.reshape(-1))


def _srgb(c):
    return [float(x) for x in m3.as_color(c)[:3]]


def _linear(c):
    """FBX material colors are stored linear (what Blender/Maya write and read)."""
    return [float(x) for x in m3.srgb_to_linear(m3.as_color(c)[:3])]


# ---------------------------------------------------------------------------
# exporter
# ---------------------------------------------------------------------------

class _Ids:
    def __init__(self):
        self.n = 1_000_000_000

    def __call__(self) -> I64:
        self.n += 1
        return I64(self.n)


def export_fbx(scene, path, embed_textures: bool = True, bake_fps: float | None = None) -> Path:
    path = Path(path)
    uid = _Ids()
    objects = Elem("Objects")
    conns = Elem("Connections")
    counts: dict[str, int] = {}

    def count(kind):
        counts[kind] = counts.get(kind, 0) + 1

    def oo(child, parent):
        conns.add("C", "OO", I64(child), I64(parent))

    def op(child, parent, prop):
        conns.add("C", "OP", I64(child), I64(parent), prop)

    fps = bake_fps or scene.fps
    all_nodes = list(scene.walk())
    model_uid: dict[int, I64] = {}
    bone_nodes = set()
    skeletons = []
    for n in all_nodes:
        if n.skin is not None and all(n.skin is not s for s in skeletons):
            skeletons.append(n.skin)
    for sk in skeletons:
        for b in sk.bones:
            bone_nodes.add(id(b))

    # ------------------------------------------------------------ materials & textures
    mat_uid: dict[int, I64] = {}
    tex_uid: dict[int, I64] = {}
    default_mat = None

    def texture_uid(t) -> I64:
        if id(t) in tex_uid:
            return tex_uid[id(t)]
        png = encode_png(t.image)
        fname = f"{path.stem}_{t.name}_{len(tex_uid)}.png"
        (path.parent / fname).write_bytes(png)
        vid, tid = uid(), uid()
        video = objects.add("Video", vid, _name(t.name, "Video"), "Clip")
        video.add("Type", "Clip")
        vp = video.add("Properties70")
        _p(vp, "Path", "KString", "XRefUrl", "", fname)
        _p(vp, "RelPath", "KString", "XRefUrl", "", fname)
        video.add("UseMipMap", 0)
        video.add("Filename", fname)
        video.add("RelativeFilename", fname)
        if embed_textures:
            video.add("Content", Raw(png))
        tex = objects.add("Texture", tid, _name(t.name, "Texture"), "")
        tex.add("Type", "TextureVideoClip")
        tex.add("Version", 202)
        tex.add("TextureName", _name(t.name, "Texture"))
        tp = tex.add("Properties70")
        _p(tp, "UseMaterial", "bool", "", "", 1)
        if t.wrap == "clamp":
            _p(tp, "WrapModeU", "enum", "", "", 1)
            _p(tp, "WrapModeV", "enum", "", "", 1)
        tex.add("Media", _name(t.name, "Video"))
        tex.add("FileName", fname)
        tex.add("RelativeFilename", fname)
        tex.add("ModelUVTranslation", 0.0, 0.0)
        tex.add("ModelUVScaling", 1.0, 1.0)
        tex.add("Texture_Alpha_Source", "None")
        tex.add("Cropping", 0, 0, 0, 0)
        oo(vid, tid)
        count("Video")
        count("Texture")
        tex_uid[id(t)] = tid
        return tid

    def material_uid(m) -> I64:
        nonlocal default_mat
        if m is None:
            if default_mat is None:
                from ..material import Material
                default_mat = Material("Default")
            m = default_mat
        if id(m) in mat_uid:
            return mat_uid[id(m)]
        mid = uid()
        me = objects.add("Material", mid, _name(m.name, "Material"), "")
        me.add("Version", 102)
        me.add("ShadingModel", "Phong")
        me.add("MultiLayer", 0)
        pp = me.add("Properties70")
        base = _linear(m.base_color)
        emis = _linear(m.emissive)
        alpha = float(m3.as_color(m.base_color)[3] * m.alpha)
        shin = float(max(2.0, (1.0 - m.roughness) ** 2 * 100.0))
        spec = 0.5 * (1 - m.roughness) if m.metallic < 0.5 else 0.8
        _p(pp, "ShadingModel", "KString", "", "", "Phong")
        _p_vec(pp, "EmissiveColor", "Color", "", "A", emis)
        _p(pp, "EmissiveFactor", "Number", "", "A", float(m.emissive_strength if max(emis) > 0 else 1.0))
        _p_vec(pp, "AmbientColor", "Color", "", "A", (0.0, 0.0, 0.0))
        _p_vec(pp, "DiffuseColor", "Color", "", "A", base)
        _p(pp, "DiffuseFactor", "Number", "", "A", 1.0)
        _p_vec(pp, "TransparentColor", "Color", "", "A", (1.0, 1.0, 1.0))
        _p(pp, "TransparencyFactor", "Number", "", "A", 1.0 - alpha)
        _p(pp, "Opacity", "Number", "", "A", alpha)
        _p_vec(pp, "SpecularColor", "Color", "", "A", base if m.metallic >= 0.5 else (1.0, 1.0, 1.0))
        _p(pp, "SpecularFactor", "Number", "", "A", float(spec))
        _p(pp, "Shininess", "Number", "", "A", shin)
        _p(pp, "ShininessExponent", "Number", "", "A", shin)
        _p(pp, "ReflectionFactor", "Number", "", "A", float(m.metallic))
        # Common PBR hints read by several importers (e.g. Maya/Unity/Blender via custom props)
        _p(pp, "Roughness", "Number", "", "A+U", float(m.roughness))
        _p(pp, "Metallic", "Number", "", "A+U", float(m.metallic))
        if m.base_color_texture is not None:
            op(texture_uid(m.base_color_texture), mid, "DiffuseColor")
        if m.normal_texture is not None:
            op(texture_uid(m.normal_texture), mid, "NormalMap")
        if m.emissive_texture is not None:
            op(texture_uid(m.emissive_texture), mid, "EmissiveColor")
        if m.alpha < 1.0 and m.base_color_texture is not None and m.resolved_alpha_mode != "OPAQUE":
            op(texture_uid(m.base_color_texture), mid, "TransparentColor")
        count("Material")
        mat_uid[id(m)] = mid
        return mid

    # ------------------------------------------------------------ geometry
    geom_uid: dict[int, I64] = {}
    blendshape_channels: dict[int, list] = {}

    def geometry_uid(mesh) -> I64:
        if id(mesh) in geom_uid:
            return geom_uid[id(mesh)]
        gid = uid()
        g = objects.add("Geometry", gid, _name(mesh.name, "Geometry"), "Mesh")
        g.add("Properties70")
        g.add("GeometryVersion", 124)
        g.add("Vertices", Arr("d", mesh.vertices))
        cv, cf, start, sizes = mesh._corners()
        pvi = cv.copy()
        ends = start + sizes - 1
        pvi[ends] = -pvi[ends] - 1
        g.add("PolygonVertexIndex", Arr("i", pvi))
        layer_elems = []
        # normals
        ln = g.add("LayerElementNormal", 0)
        ln.add("Version", 101)
        ln.add("Name", "")
        ln.add("MappingInformationType", "ByPolygonVertex")
        ln.add("ReferenceInformationType", "Direct")
        ln.add("Normals", Arr("d", mesh.corner_normals()))
        layer_elems.append("LayerElementNormal")
        # vertex colors (FBX consumers expect sRGB-ish values)
        if mesh.colors is not None:
            lc = g.add("LayerElementColor", 0)
            lc.add("Version", 101)
            lc.add("Name", "Col")
            lc.add("MappingInformationType", "ByPolygonVertex")
            lc.add("ReferenceInformationType", "IndexToDirect")
            col = mesh.colors.copy()
            col[:, :3] = m3.linear_to_srgb(np.clip(col[:, :3], 0, 1))
            lc.add("Colors", Arr("d", col))
            lc.add("ColorIndex", Arr("i", cv))
            layer_elems.append("LayerElementColor")
        # uvs
        if mesh.uvs is not None:
            lu = g.add("LayerElementUV", 0)
            lu.add("Version", 101)
            lu.add("Name", "UVMap")
            lu.add("MappingInformationType", "ByPolygonVertex")
            lu.add("ReferenceInformationType", "IndexToDirect")
            lu.add("UV", Arr("d", mesh.uvs))
            lu.add("UVIndex", Arr("i", mesh._corner_uvs()))
            layer_elems.append("LayerElementUV")
        # materials
        lm = g.add("LayerElementMaterial", 0)
        lm.add("Version", 101)
        lm.add("Name", "")
        nmat = max(len(mesh.materials), 1)
        fm = np.clip(mesh.face_materials, 0, nmat - 1) if len(mesh.face_materials) else np.zeros(0, int)
        if nmat == 1:
            lm.add("MappingInformationType", "AllSame")
            lm.add("ReferenceInformationType", "IndexToDirect")
            lm.add("Materials", Arr("i", np.zeros(1, dtype=np.int32)))
        else:
            lm.add("MappingInformationType", "ByPolygon")
            lm.add("ReferenceInformationType", "IndexToDirect")
            lm.add("Materials", Arr("i", fm))
        layer_elems.append("LayerElementMaterial")
        layer = g.add("Layer", 0)
        layer.add("Version", 100)
        for le in layer_elems:
            e = layer.add("LayerElement")
            e.add("Type", le)
            e.add("TypedIndex", 0)
        count("Geometry")
        geom_uid[id(mesh)] = gid
        # blend shapes
        if mesh.shape_keys:
            bs = uid()
            bse = objects.add("Deformer", bs, _name(mesh.name + "_Shapes", "Deformer"), "BlendShape")
            bse.add("Version", 100)
            count("Deformer")
            oo(bs, gid)
            chans = []
            for kname, pos in mesh.shape_keys.items():
                delta = pos - mesh.vertices
                idx = np.flatnonzero(np.linalg.norm(delta, axis=1) > 1e-7)
                if len(idx) == 0:
                    idx = np.array([0])
                sid, cid = uid(), uid()
                sh = objects.add("Geometry", sid, _name(kname, "Geometry"), "Shape")
                sh.add("Version", 100)
                sh.add("Indexes", Arr("i", idx))
                sh.add("Vertices", Arr("d", delta[idx]))
                sh.add("Normals", Arr("d", np.zeros((len(idx), 3))))
                ch = objects.add("Deformer", cid, _name(kname, "SubDeformer"), "BlendShapeChannel")
                ch.add("Version", 100)
                ch.add("DeformPercent", 0.0)
                ch.add("FullWeights", Arr("d", [100.0]))
                count("Geometry")
                count("Deformer")
                oo(cid, bs)
                oo(sid, cid)
                chans.append(cid)
            blendshape_channels[id(mesh)] = chans
        return gid

    # ------------------------------------------------------------ models
    cam_fix = m3.quat_from_axis_angle((0, 1, 0), 90)   # FBX cameras look down +X
    light_fix = m3.quat_from_axis_angle((1, 0, 0), 90)  # FBX lights point down -Y

    def local_rotation(n, q=None):
        q = n.rotation if q is None else q
        if not n.children:
            if n.camera is not None:
                return m3.quat_mul(q, cam_fix)
            if n.light is not None:
                return m3.quat_mul(q, light_fix)
        return q

    def write_model(n, mtype: str, rot_q=None, name=None, translation=None, scale=None) -> I64:
        mid = uid()
        me = objects.add("Model", mid, _name(name or n.name, "Model"), mtype)
        me.add("Version", 232)
        pp = me.add("Properties70")
        t = n.translation if translation is None else translation
        q = rot_q if rot_q is not None else n.rotation
        s = n.scale if scale is None else scale
        _p(pp, "RotationActive", "bool", "", "", 1)
        _p(pp, "InheritType", "enum", "", "", 1)
        _p_vec(pp, "ScalingMax", "Vector3D", "Vector", "", (0.0, 0.0, 0.0))
        _p(pp, "DefaultAttributeIndex", "int", "Integer", "", 0)
        _p_vec(pp, "Lcl Translation", "Lcl Translation", "", "A", t)
        _p_vec(pp, "Lcl Rotation", "Lcl Rotation", "", "A", m3.quat_to_euler(q))
        _p_vec(pp, "Lcl Scaling", "Lcl Scaling", "", "A", s)
        if not n.visible:
            _p(pp, "Visibility", "Visibility", "", "A", 0.0)
        me.add("MultiLayer", 0)
        me.add("MultiTake", 0)
        me.add("Shading", True)
        me.add("Culling", "CullingOff")
        count("Model")
        return mid

    attr_targets: dict[int, I64] = {}  # node id -> model uid carrying its light/camera attribute

    def write_node(n, parent_uid: I64):
        if id(n) in bone_nodes:
            mtype = "LimbNode"
        elif n.mesh is not None:
            mtype = "Mesh"
        elif n.light is not None and not n.children:
            mtype = "Light"
        elif n.camera is not None and not n.children:
            mtype = "Camera"
        else:
            mtype = "Null"
        mid = write_model(n, mtype, local_rotation(n))
        model_uid[id(n)] = mid
        oo(mid, parent_uid)
        if mtype == "LimbNode":
            aid = uid()
            ae = objects.add("NodeAttribute", aid, _name(n.name, "NodeAttribute"), "LimbNode")
            ap = ae.add("Properties70")
            _p(ap, "Size", "double", "Number", "", 10.0)
            ae.add("TypeFlags", "Skeleton")
            count("NodeAttribute")
            oo(aid, mid)
        if n.mesh is not None and n.mesh.faces:
            oo(geometry_uid(n.mesh), mid)
            for mt in (n.mesh.materials or [None]):
                oo(material_uid(mt), mid)
        # lights / cameras (on this model, or on a child carrying the axis fix)
        for kind in ("light", "camera"):
            obj = getattr(n, kind)
            if obj is None:
                continue
            holder = mid
            if n.children or mtype in ("Mesh", "LimbNode"):
                fix = cam_fix if kind == "camera" else light_fix
                from ..scene import Node as _Node
                tmp = _Node(n.name + ("_Camera" if kind == "camera" else "_Light"))
                holder = write_model(tmp, "Camera" if kind == "camera" else "Light", fix)
                oo(holder, mid)
            attr_targets[id(n)] = holder
            aid = uid()
            if kind == "light":
                L = obj
                ae = objects.add("NodeAttribute", aid, _name(L.name, "NodeAttribute"), "Light")
                ap = ae.add("Properties70")
                _p(ap, "LightType", "enum", "", "", {"point": 0, "directional": 1, "spot": 2}[L.type])
                _p_vec(ap, "Color", "Color", "", "A", _linear(L.color))
                inten = L.intensity * (100.0 / 3.0 if L.type == "directional" else 100.0 / 100.0)
                _p(ap, "Intensity", "Number", "", "A", float(inten))
                _p(ap, "CastShadows", "bool", "", "", 1 if L.cast_shadows else 0)
                if L.type == "spot":
                    _p(ap, "InnerAngle", "Number", "", "A", float(L.inner_cone * 2))
                    _p(ap, "OuterAngle", "Number", "", "A", float(L.outer_cone * 2))
                if L.range:
                    _p(ap, "DecayType", "enum", "", "", 2)
                    _p(ap, "FarAttenuationEnd", "Number", "", "A", float(L.range))
                ae.add("TypeFlags", "Light")
                ae.add("GeometryVersion", 124)
            else:
                C = obj
                ae = objects.add("NodeAttribute", aid, _name(C.name, "NodeAttribute"), "Camera")
                ap = ae.add("Properties70")
                aspect = C.aspect or 16 / 9
                _p(ap, "ApertureMode", "enum", "", "", 2)  # vertical FOV
                _p(ap, "FieldOfView", "FieldOfView", "", "A", float(C.yfov))
                _p(ap, "FieldOfViewY", "FieldOfViewY", "", "A", float(C.yfov))
                hfov = math.degrees(2 * math.atan(math.tan(math.radians(C.yfov) / 2) * aspect))
                _p(ap, "FieldOfViewX", "FieldOfViewX", "", "A", float(hfov))
                _p(ap, "AspectWidth", "double", "Number", "", float(aspect * 1080))
                _p(ap, "AspectHeight", "double", "Number", "", 1080.0)
                _p(ap, "NearPlane", "double", "Number", "", float(C.znear))
                _p(ap, "FarPlane", "double", "Number", "", float(C.zfar))
                _p(ap, "CameraProjectionType", "enum", "", "", 1 if C.type == "orthographic" else 0)
                if C.type == "orthographic":
                    _p(ap, "OrthoZoom", "double", "Number", "", float(C.ortho_height))
                ae.add("TypeFlags", "Camera")
                ae.add("GeometryVersion", 124)
                ae.add("Position", 0.0, 0.0, 0.0)
                ae.add("Up", 0.0, 1.0, 0.0)
                ae.add("LookAt", 1.0, 0.0, 0.0)
                ae.add("ShowInfoOnMoving", 1)
                ae.add("ShowAudio", 0)
                ae.add("AudioColor", 0.0, 1.0, 0.0)
                ae.add("CameraOrthoZoom", 1.0)
            count("NodeAttribute")
            oo(aid, holder)
        for c in n.children:
            write_node(c, mid)

    for n in scene.root.children:
        write_node(n, I64(0))

    # ------------------------------------------------------------ skins + bind poses
    for n in all_nodes:
        sk = n.skin
        if sk is None or n.mesh is None or n.mesh.joints is None:
            continue
        mesh = n.mesh
        gid = geometry_uid(mesh)
        skin_id = uid()
        se = objects.add("Deformer", skin_id, _name(mesh.name + "_Skin", "Deformer"), "Skin")
        se.add("Version", 101)
        se.add("Link_DeformAcuracy", 50.0)
        se.add("SkinningType", "Linear")
        count("Deformer")
        oo(skin_id, gid)
        mesh_world = n.world_matrix()
        rest = sk.rest_world_matrices()
        pose_id = uid()
        pose = objects.add("Pose", pose_id, _name(mesh.name + "_BindPose", "Pose"), "BindPose")
        pose.add("Type", "BindPose")
        pose.add("Version", 100)
        pose_nodes = [(model_uid[id(n)], mesh_world)]
        if id(sk.root) in model_uid:
            pose_nodes.append((model_uid[id(sk.root)], sk.bind_root_matrix()))
        for bi, b in enumerate(sk.bones):
            w_idx = np.zeros(len(mesh.vertices), dtype=bool)
            ws = np.zeros(len(mesh.vertices))
            for k in range(mesh.joints.shape[1]):
                sel = (mesh.joints[:, k] == bi) & (mesh.weights[:, k] > 0)
                w_idx |= sel
                ws[sel] += mesh.weights[sel, k]
            vids = np.flatnonzero(w_idx)
            cid = uid()
            ce = objects.add("Deformer", cid, _name(b.name, "SubDeformer"), "Cluster")
            ce.add("Version", 100)
            ce.add("UserData", "", "")
            ce.add("Indexes", Arr("i", vids))
            ce.add("Weights", Arr("d", ws[vids]))
            ce.add("Transform", _mat_array(np.linalg.inv(rest[bi]) @ mesh_world))
            ce.add("TransformLink", _mat_array(rest[bi]))
            if id(sk.root) in model_uid:
                ce.add("TransformAssociateModel", _mat_array(sk.bind_root_matrix()))
            count("Deformer")
            oo(cid, skin_id)
            oo(model_uid[id(b)], cid)
            pose_nodes.append((model_uid[id(b)], rest[bi]))
        pose.add("NbPoseNodes", len(pose_nodes))
        for muid, mtx in pose_nodes:
            pn = pose.add("PoseNode")
            pn.add("Node", I64(muid))
            pn.add("Matrix", _mat_array(mtx))
        count("Pose")

    # ------------------------------------------------------------ animations
    takes = Elem("Takes")
    takes.add("Current", scene.animations[0].name if scene.animations else "")
    for anim in scene.animations:
        chans = [c for c in anim.channels if c.times and id(c.node) in model_uid]
        if not chans:
            continue
        t0, t1 = anim.start, anim.duration
        nframes = max(int(round((t1 - t0) * fps)), 1)
        times = np.linspace(t0, t1, nframes + 1)
        ktimes = np.round(times * KTIME).astype(np.int64)
        stack_id, layer_id = uid(), uid()
        st = objects.add("AnimationStack", stack_id, _name(anim.name, "AnimStack"), "")
        sp = st.add("Properties70")
        _p(sp, "LocalStart", "KTime", "Time", "", I64(int(ktimes[0])))
        _p(sp, "LocalStop", "KTime", "Time", "", I64(int(ktimes[-1])))
        _p(sp, "ReferenceStart", "KTime", "Time", "", I64(int(ktimes[0])))
        _p(sp, "ReferenceStop", "KTime", "Time", "", I64(int(ktimes[-1])))
        objects.add("AnimationLayer", layer_id, _name("BaseLayer", "AnimLayer"), "")
        count("AnimationStack")
        count("AnimationLayer")
        oo(layer_id, stack_id)
        by_node: dict[int, dict] = {}
        for c in chans:
            by_node.setdefault(id(c.node), {"node": c.node})[c.path] = c
        for entry in by_node.values():
            node = entry["node"]
            muid = model_uid[id(node)]
            for path_, fbx_prop, short in (("translation", "Lcl Translation", "T"),
                                           ("rotation", "Lcl Rotation", "R"), ("scale", "Lcl Scaling", "S")):
                ch = entry.get(path_)
                if ch is None:
                    continue
                vals = np.array([ch.sample(t) for t in times])
                if path_ == "rotation":
                    vals = np.array([m3.quat_to_euler(local_rotation(node, q)) for q in vals])
                    vals = _unwrap_degrees(vals)
                cn_id = uid()
                cn = objects.add("AnimationCurveNode", cn_id, _name(short, "AnimCurveNode"), "")
                cp = cn.add("Properties70")
                for ax in range(3):
                    _p(cp, f"d|{'XYZ'[ax]}", "Number", "", "A", float(vals[0, ax]))
                count("AnimationCurveNode")
                oo(cn_id, layer_id)
                op(cn_id, muid, fbx_prop)
                for ax in range(3):
                    cid = uid()
                    _anim_curve(objects, cid, ktimes, vals[:, ax])
                    count("AnimationCurve")
                    op(cid, cn_id, f"d|{'XYZ'[ax]}")
            wch = entry.get("weights")
            if wch is not None and node.mesh is not None and id(node.mesh) in blendshape_channels:
                vals = np.array([wch.sample(t) for t in times]) * 100.0
                for k, chan_uid in enumerate(blendshape_channels[id(node.mesh)]):
                    if k >= vals.shape[1]:
                        break
                    cn_id = uid()
                    cn = objects.add("AnimationCurveNode", cn_id, _name("DeformPercent", "AnimCurveNode"), "")
                    cp = cn.add("Properties70")
                    _p(cp, "d|DeformPercent", "Number", "", "A", float(vals[0, k]))
                    count("AnimationCurveNode")
                    oo(cn_id, layer_id)
                    op(cn_id, chan_uid, "DeformPercent")
                    cid = uid()
                    _anim_curve(objects, cid, ktimes, vals[:, k])
                    count("AnimationCurve")
                    op(cid, cn_id, "d|DeformPercent")
        tk = takes.add("Take", anim.name)
        tk.add("FileName", f"{anim.name}.tak")
        tk.add("LocalTime", I64(int(ktimes[0])), I64(int(ktimes[-1])))
        tk.add("ReferenceTime", I64(int(ktimes[0])), I64(int(ktimes[-1])))

    # ------------------------------------------------------------ header / settings / definitions
    root = [_header(path), Elem("FileId", Raw(_FILE_ID)), Elem("CreationTime", _TIME_ID.decode()),
            Elem("Creator", f"Pyram3D {__version__}"), _global_settings(scene, fps)]
    docs = Elem("Documents")
    docs.add("Count", 1)
    d = docs.add("Document", uid(), "", "Scene")
    dp = d.add("Properties70")
    _p(dp, "SourceObject", "object", "", "")
    _p(dp, "ActiveAnimStackName", "KString", "", "", scene.animations[0].name if scene.animations else "")
    d.add("RootNode", I64(0))
    root.append(docs)
    root.append(Elem("References"))
    defs = Elem("Definitions")
    defs.add("Version", 100)
    defs.add("Count", sum(counts.values()) + 1)
    ot = defs.add("ObjectType", "GlobalSettings")
    ot.add("Count", 1)
    for kind, c in counts.items():
        ot = defs.add("ObjectType", kind)
        ot.add("Count", c)
    root.append(defs)
    root.append(objects)
    root.append(conns)
    root.append(takes)
    write_fbx_binary(path, root)
    return path


def _unwrap_degrees(vals: np.ndarray) -> np.ndarray:
    out = vals.copy()
    for i in range(1, len(out)):
        d = out[i] - out[i - 1]
        out[i] -= 360.0 * np.round(d / 360.0)
    return out


def _anim_curve(objects: Elem, cid, ktimes, values):
    ce = objects.add("AnimationCurve", cid, _name("", "AnimCurve"), "")
    ce.add("Default", float(values[0]))
    ce.add("KeyVer", 4008)
    ce.add("KeyTime", Arr("l", ktimes))
    ce.add("KeyValueFloat", Arr("f", np.asarray(values, dtype=np.float32)))
    ce.add("KeyAttrFlags", Arr("i", [24836]))
    ce.add("KeyAttrDataFloat", Arr("f", np.array([0.0, 0.0, 9.419963346924634e-30, 0.0], dtype=np.float32)))
    ce.add("KeyAttrRefCount", Arr("i", [len(ktimes)]))


def _header(path: Path) -> Elem:
    h = Elem("FBXHeaderExtension")
    h.add("FBXHeaderVersion", 1003)
    h.add("FBXVersion", FBX_VERSION)
    h.add("EncryptionType", 0)
    ts = h.add("CreationTimeStamp")
    for k, v in (("Version", 1000), ("Year", 1970), ("Month", 1), ("Day", 1), ("Hour", 10), ("Minute", 0),
                 ("Second", 0), ("Millisecond", 0)):
        ts.add(k, v)
    h.add("Creator", f"Pyram3D {__version__}")
    si = h.add("SceneInfo", _name("GlobalInfo", "SceneInfo"), "UserData")
    si.add("Type", "UserData")
    si.add("Version", 100)
    md = si.add("MetaData")
    md.add("Version", 100)
    for k in ("Title", "Subject", "Author", "Keywords", "Revision", "Comment"):
        md.add(k, "")
    pp = si.add("Properties70")
    _p(pp, "DocumentUrl", "KString", "Url", "", str(path.name))
    _p(pp, "SrcDocumentUrl", "KString", "Url", "", str(path.name))
    _p(pp, "Original", "Compound", "", "")
    _p(pp, "Original|ApplicationVendor", "KString", "", "", "Pyram3D")
    _p(pp, "Original|ApplicationName", "KString", "", "", "Pyram3D")
    _p(pp, "Original|ApplicationVersion", "KString", "", "", __version__)
    _p(pp, "Original|DateTime_GMT", "DateTime", "", "", "01/01/1970 10:00:00.000")
    _p(pp, "Original|FileName", "KString", "", "", str(path.name))
    _p(pp, "LastSaved", "Compound", "", "")
    _p(pp, "LastSaved|ApplicationVendor", "KString", "", "", "Pyram3D")
    _p(pp, "LastSaved|ApplicationName", "KString", "", "", "Pyram3D")
    _p(pp, "LastSaved|ApplicationVersion", "KString", "", "", __version__)
    _p(pp, "LastSaved|DateTime_GMT", "DateTime", "", "", "01/01/1970 10:00:00.000")
    return h


def _global_settings(scene, fps) -> Elem:
    g = Elem("GlobalSettings")
    g.add("Version", 1000)
    pp = g.add("Properties70")
    _p(pp, "UpAxis", "int", "Integer", "", 1)
    _p(pp, "UpAxisSign", "int", "Integer", "", 1)
    _p(pp, "FrontAxis", "int", "Integer", "", 2)
    _p(pp, "FrontAxisSign", "int", "Integer", "", 1)
    _p(pp, "CoordAxis", "int", "Integer", "", 0)
    _p(pp, "CoordAxisSign", "int", "Integer", "", 1)
    _p(pp, "OriginalUpAxis", "int", "Integer", "", 1)
    _p(pp, "OriginalUpAxisSign", "int", "Integer", "", 1)
    _p(pp, "UnitScaleFactor", "double", "Number", "", 100.0)
    _p(pp, "OriginalUnitScaleFactor", "double", "Number", "", 100.0)
    _p_vec(pp, "AmbientColor", "ColorRGB", "Color", "", (0.0, 0.0, 0.0))
    _p(pp, "DefaultCamera", "KString", "", "", "Producer Perspective")
    mode = {24: 11, 30: 6, 60: 3, 50: 4, 25: 10, 48: 5, 120: 1, 100: 2}.get(int(fps), 14)
    _p(pp, "TimeMode", "enum", "", "", mode)
    _p(pp, "TimeProtocol", "enum", "", "", 2)
    _p(pp, "SnapOnFrameMode", "enum", "", "", 0)
    stop = max((a.duration for a in scene.animations), default=1.0)
    _p(pp, "TimeSpanStart", "KTime", "Time", "", I64(0))
    _p(pp, "TimeSpanStop", "KTime", "Time", "", I64(int(round(stop * KTIME))))
    _p(pp, "CustomFrameRate", "double", "Number", "", float(fps))
    _p(pp, "CurrentTimeMarker", "int", "Integer", "", -1)
    return g
