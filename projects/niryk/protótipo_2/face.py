"""Anime face as painted decals (Genshin Impact line), the Head_Skin UVs and the face materials.

Every decal is a copy of the Head_Skin triangles in its region, pushed 0.4 to 1.0 mm out along the
skin's normal, so it follows the real surface exactly; its shading normals are the skin's. The
opaque ones (eye whites, irises) are clipped to their outline (convex polygons, face_paint.py);
the alpha-tested ones (MASK, cutoff 0.5) carry their shape in the texture's alpha. UVs map the
design frames of face_paint.py onto the textures (the right eye mirrors the left one).

Nodes and materials (the client contract, see README.md):
  Head_Skin      Skin      tinted by the skin color; baseColorTexture near white (blush, nose hint);
                           occlusionTexture = face shadow threshold map (linear, R), same UVs
  Head_Eyes      EyeWhite  untinted eye whites (shaded under the lid)
  Head_Iris      Eye       grayscale iris and pupil, tinted by the eye color
  Head_EyeShine  EyeShine  untinted white highlights, MASK
  Head_Lashes    Lash      untinted lash line, flare, lashes, crease, lower lash hint, MASK
  Head_Brows     Brow      grayscale, tinted by the hair color, MASK
  Head_Mouth     Mouth     untinted mouth line, MASK
All Head_* decals are rigid on the Head bone (hidden in first person with the head).
"""
from __future__ import annotations

import numpy as np

import face_paint as FP
import head as H
from common import slot

import pyram3d as p3

LIFT = {"White": 0.00045, "Iris": 0.0006, "Brow": 0.00055, "Mouth": 0.00055, "Shine": 0.0008, "Lash": 0.00095}
MASK = ("EyeShine", "Lash", "Brow", "Mouth")
_MATS = None


def face_materials():
    """The face materials with their textures (one set per process; slots are shared)."""
    global _MATS
    if _MATS is not None:
        return _MATS
    tex = lambda img, name, srgb=True, wrap="clamp": p3.Texture(img, name=name, srgb=srgb, wrap=wrap)  # noqa: E731
    # Head_Skin's own "Skin" (Body keeps the shared plain slot): same name, so the client tints both
    skin = slot("Skin").copy(extras={})
    skin.base_color_texture = tex(FP.skin_base_color(), "SkinFace", wrap="repeat")
    thr = FP.face_threshold()
    skin.occlusion_texture = tex(np.repeat(thr[..., None], 3, axis=-1), "FaceShadowMap", srgb=False, wrap="repeat")
    skin.extras["face_threshold"] = thr  # review renders bake the face map shading from it
    mats = {"Skin": skin}
    for name, img in (("EyeWhite", FP.paint_eye_white()), ("Eye", FP.paint_iris()),
                      ("EyeShine", FP.paint_eye_shine()), ("Lash", FP.paint_lashes()),
                      ("Brow", FP.paint_brow()), ("Mouth", FP.paint_mouth())):
        m = slot(name)
        m.base_color_texture = tex(img, f"Face{name}")
        if name in MASK:
            m.alpha_mode = "MASK"
            m.alpha_cutoff = 0.5
        mats[name] = m
    _MATS = mats
    return mats


# ----------------------------------------------------------------------------- Head_Skin UVs

def apply_head_uv(m: p3.Mesh) -> p3.Mesh:
    """Head_Skin UVs (head.py's mapping); faces across the seam down the back of the neck get
    u + 1 on their low side (the texture repeats)."""
    uv = H.head_uv(m.vertices)
    F = np.asarray(m.faces, dtype=np.int64).reshape(-1, 3)
    fu = uv[F]  # (F, 3, 2)
    span = fu[:, :, 0].max(axis=1) - fu[:, :, 0].min(axis=1)
    seam = span > 0.5
    fu[seam, :, 0] = np.where(fu[seam, :, 0] < 0.5, fu[seam, :, 0] + 1.0, fu[seam, :, 0])
    m.uvs = fu.reshape(-1, 2)
    m.face_uvs = [tuple(range(3 * i, 3 * i + 3)) for i in range(len(F))]
    return m


# ----------------------------------------------------------------------------- decals

def _coords(Pd, kind, sx=1):
    """Design-space points -> 2D design frame coordinates (meters)."""
    q = H.to_head(Pd)
    if kind == "eye":
        return np.column_stack([sx * q[:, 0] - H.EYE_X, q[:, 1] - H.EYE_Y])
    return np.column_stack([q[:, 0], q[:, 1] - FP.MOUTH["y"]])


def _uv(c2, frame):
    u0, u1, v0, v1 = frame
    return np.column_stack([(c2[:, 0] - u0) / (u1 - u0), (c2[:, 1] - v0) / (v1 - v0)])


def _weld(P, N, UV, faces, tol=2e-7):
    key = np.round(P / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    F = inv[np.asarray(faces, dtype=np.int64)]
    ok = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 2] != F[:, 0])
    F = F[ok]
    Pw = P[first]
    a = np.linalg.norm(np.cross(Pw[F[:, 1]] - Pw[F[:, 0]], Pw[F[:, 2]] - Pw[F[:, 0]]), axis=1) * 0.5
    F = F[a > 2e-12]
    return Pw, N[first], UV[first], F


def _decal(skin, push_n, sel_faces, coords, frame, lift, material, name, clip=None):
    """Copy the selected Head_Skin triangles (clipped to `clip`, a convex polygon in the design
    frame), push them `lift` m out along the skin normal; UVs from the design frame."""
    V, Ns = skin.vertices, skin.normals
    F = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 3)[sel_faces]
    P_out, N_out, C_out, faces = [], [], [], []
    base = 0
    for tri in F:
        poly = np.column_stack([coords[tri], np.eye(3)])  # (3, 5): u, v, barycentrics
        if clip is not None:
            poly = FP.clip_convex(poly, clip)
            if len(poly) < 3:
                continue
        b = poly[:, 2:5]
        p = b @ V[tri]
        n_push = b @ push_n[tri]
        n_push /= np.linalg.norm(n_push, axis=1, keepdims=True)
        n_sh = b @ Ns[tri]
        n_sh /= np.linalg.norm(n_sh, axis=1, keepdims=True)
        P_out.append(p + n_push * lift)
        N_out.append(n_sh)
        C_out.append(poly[:, :2])
        for k in range(1, len(poly) - 1):
            faces.append((base, base + k, base + k + 1))
        base += len(poly)
    if not faces:
        return None
    P, N, C, Fw = _weld(np.vstack(P_out), np.vstack(N_out), np.vstack(C_out), faces)
    m = p3.Mesh(P, [tuple(f) for f in Fw], name=name)
    m.uvs = _uv(C, frame)
    m.face_uvs = [tuple(f) for f in Fw]
    m.normals = N
    m.smooth_angle = 180.0
    m.set_material(material)
    m.colors = np.ones((m.num_vertices, 4))
    return m


def _alpha_keep(img, uvs_faces, grow_px=24):
    """Faces whose UV triangle touches the (dilated) alpha of a MASK texture."""
    a = img[..., 3] > 0.02
    for _ in range(grow_px // 4):  # cheap dilation by 4 px per pass
        g = a.copy()
        for s in (-4, 4):
            g |= np.roll(a, s, axis=0) | np.roll(a, s, axis=1)
        a = g
    h, w = a.shape
    bs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1 / 3, 1 / 3, 1 / 3], [.5, .5, 0], [0, .5, .5], [.5, 0, .5],
                   [.6, .2, .2], [.2, .6, .2], [.2, .2, .6]])
    pts = np.einsum("sk,fkd->fsd", bs, uvs_faces)
    x = np.clip((pts[..., 0] * w).astype(int), 0, w - 1)
    y = np.clip(((1 - pts[..., 1]) * h).astype(int), 0, h - 1)
    inside = (pts[..., 0] >= 0) & (pts[..., 0] <= 1) & (pts[..., 1] >= 0) & (pts[..., 1] <= 1)
    return (a[y, x] & inside).any(axis=1)


def face_decals(skin: p3.Mesh) -> dict[str, p3.Mesh]:
    """All face decals from the Head_Skin mesh (design space, with its shading normals)."""
    mats = face_materials()
    push_n = skin.vertex_normals()
    F = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 3)
    q = H.to_head(skin.vertices)
    front_v = (q[:, 2] > 0.02) & (push_n[:, 2] > 0.1)
    front_f = front_v[F].all(axis=1)
    eye_clip, iris_clip = FP.eye_outline(), FP.iris_outline()
    out = {}

    def in_box(c2, box, f_ok):
        u0, u1, v0, v1 = box
        cf = c2[F]  # (F, 3, 2)
        lo, hi = cf.min(axis=1), cf.max(axis=1)
        return f_ok & (hi[:, 0] > u0) & (lo[:, 0] < u1) & (hi[:, 1] > v0) & (lo[:, 1] < v1)

    def build(kind, box, clip, lift, mat, frame, tex_img=None, sides=(1, -1)):
        parts = []
        for sx in sides:
            c2 = _coords(skin.vertices, kind, sx)
            side_f = front_f & ((np.sign(q[F, 0]).sum(axis=1) * sx) > 0 if kind == "eye" else front_f)
            sel = np.flatnonzero(in_box(c2, box, side_f))
            if tex_img is not None and len(sel):
                keep = _alpha_keep(tex_img, _uv(c2, frame)[F[sel]])
                sel = sel[keep]
            m = _decal(skin, push_n, sel, c2, frame, lift, mat, "d", clip=clip)
            if m is not None:  # (winding: the 3D triangles keep the skin's)
                parts.append(m)
        return p3.Mesh.join(parts) if parts else None

    def bbox(poly, pad=0.0015):
        lo, hi = poly.min(axis=0) - pad, poly.max(axis=0) + pad
        return (lo[0], hi[0], lo[1], hi[1])

    out["Head_Eyes"] = build("eye", bbox(eye_clip), eye_clip, LIFT["White"], mats["EyeWhite"], FP.EYE_FRAME)
    out["Head_Iris"] = build("eye", bbox(iris_clip), iris_clip, LIFT["Iris"], mats["Eye"], FP.EYE_FRAME)
    img = mats["EyeShine"].base_color_texture.image.astype(float) / 255
    out["Head_EyeShine"] = build("eye", bbox(iris_clip), None, LIFT["Shine"], mats["EyeShine"], FP.EYE_FRAME, img)
    img = mats["Lash"].base_color_texture.image.astype(float) / 255
    out["Head_Lashes"] = build("eye", FP.EYE_FRAME, None, LIFT["Lash"], mats["Lash"], FP.EYE_FRAME, img)
    img = mats["Brow"].base_color_texture.image.astype(float) / 255
    out["Head_Brows"] = build("eye", FP.BROW_FRAME, None, LIFT["Brow"], mats["Brow"], FP.BROW_FRAME, img)
    img = mats["Mouth"].base_color_texture.image.astype(float) / 255
    out["Head_Mouth"] = build("mouth", FP.MOUTH_FRAME, None, LIFT["Mouth"], mats["Mouth"], FP.MOUTH_FRAME, img,
                              sides=(1,))
    for k, m in out.items():
        if m is None:
            raise RuntimeError(f"empty face decal {k}")
        m.name = k
    return out
