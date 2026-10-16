"""Software renderer (numpy) for previews — no GPU, no external dependencies.

It lets an AI agent *look* at what it built: render PNGs, open them with an
image-capable tool, and iterate. Styles:

  "realistic" : PBR-ish shading, shadows, sky light, ground contact shadow
  "toon"      : cel shading with outlines (stylized)
  "clay"      : neutral grey material, great for judging shapes
  "normals"   : normal map visualization (debug)
  "wire"      : clay + polygon edges overlay (topology check)

    p3.render(scene, "shot.png")                       # auto camera 3/4 view
    p3.render_views(scene, "sheet.png")                # front/side/back/top/3-4 contact sheet
    p3.render_frames(scene, "walk.png", "Walk", 6)     # animation filmstrip
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from . import math3d as m3
from .image import downsample, grid, write_png, to_uint8
from .mesh import triangulate_corners

VIEW_DIRS = {
    "front": (0, 0, 1), "back": (0, 0, -1), "left": (-1, 0, 0), "right": (1, 0, 0),
    "top": (0, 1, 0.0001), "bottom": (0, -1, 0.0001),
    "three_quarter": (0.75, 0.45, 1.0), "persp": (0.75, 0.45, 1.0), "back_three_quarter": (-0.75, 0.45, -1.0),
}


# ---------------------------------------------------------------------------
# scene flattening
# ---------------------------------------------------------------------------

class _Batch:
    """All triangles of the scene in world space with per-vertex attributes."""

    def __init__(self):
        self.P = []   # (T,3,3) positions
        self.N = []   # (T,3,3) normals
        self.UV = []  # (T,3,2)
        self.C = []   # (T,3,4) vertex colors (linear)
        self.M = []   # (T,) material id
        self.E = []   # (T,3) bool: is polygon edge (for wireframe)
        self.mats = []

    def finish(self):
        if not self.P:
            z = np.zeros((0, 3, 3))
            return z, z, np.zeros((0, 3, 2)), np.zeros((0, 3, 4)), np.zeros(0, int), np.zeros((0, 3), bool)
        return (np.concatenate(self.P), np.concatenate(self.N), np.concatenate(self.UV), np.concatenate(self.C),
                np.concatenate(self.M), np.concatenate(self.E))


def _collect(scene, time: float | None, animation=None):
    snap = None
    anim = None
    if scene.animations and time is not None:
        anim = animation if animation is not None else scene.animations[0]
        if isinstance(anim, str):
            anim = next(a for a in scene.animations if a.name == anim)
        snap = anim.snapshot()
        anim.apply(time)
    try:
        b = _Batch()
        for node in scene.mesh_nodes():
            mesh = node.mesh
            if not node.visible or not mesh.faces:
                continue
            tris, tface = triangulate_corners(mesh)
            cv, cf, _, _ = mesh._corners()
            cn = mesh.corner_normals()
            verts = mesh.vertices
            if mesh.shape_keys and node.morph_weights:
                verts = verts.copy()
                for w, sk in zip(node.morph_weights, mesh.shape_keys.values()):
                    verts = verts + (sk - mesh.vertices) * w
            if node.skin is not None and mesh.joints is not None:
                from .mesh import Mesh as _M
                tmp = _M(verts, mesh.faces)
                tmp.joints, tmp.weights = mesh.joints, mesh.weights
                wp = node.skin.pose_vertices(tmp)
                # recompute normals on the posed mesh
                posed = mesh.copy()
                posed.vertices = wp
                posed.shape_keys = {}
                cn = posed.corner_normals()
                P = wp[cv[tris]]
                N = cn[tris]
            else:
                W = node.world_matrix()
                if abs(np.linalg.det(W[:3, :3])) < 1e-15:
                    continue  # zero scale = hidden (e.g. dead particles)
                wv = m3.transform_points(W, verts)
                nm = np.linalg.inv(W[:3, :3]).T
                P = wv[cv[tris]]
                N = m3.normalize(cn[tris] @ nm.T)
            cuv = mesh._corner_uvs()
            UV = mesh.uvs[cuv[tris]] if cuv is not None else np.zeros(tris.shape + (2,))
            C = mesh.colors[cv[tris]] if mesh.colors is not None else np.ones(tris.shape + (4,))
            # material ids
            base = len(b.mats)
            mats = mesh.materials or [None]
            b.mats.extend(mats)
            fm = mesh.face_materials[tface] if len(mesh.face_materials) else np.zeros(len(tface), int)
            M = base + np.clip(fm, 0, len(mats) - 1)
            # polygon-edge flags: a triangle edge is a real edge if its corners are consecutive in the face
            _, _, start, sizes = mesh._corners()
            fs = start[tface][:, None]
            fsz = sizes[tface][:, None]
            loc = tris - fs
            nxt = np.roll(loc, -1, axis=1)
            E = ((nxt - loc) % fsz == 1) | ((loc - nxt) % fsz == 1)
            b.P.append(P)
            b.N.append(N)
            b.UV.append(UV)
            b.C.append(C)
            b.M.append(M)
            b.E.append(E)
        return b
    finally:
        if snap is not None:
            anim.restore(snap)


# ---------------------------------------------------------------------------
# rasterizer
# ---------------------------------------------------------------------------

def _rasterize(clip: np.ndarray, w: int, h: int, cull_backfaces: bool = False, chunk: int = 4_000_000):
    """clip: (T,3,4) clip-space vertices. Returns (tri_id, b0, b1, b2, depth) per pixel (-1 = empty)."""
    T = len(clip)
    tri_id = np.full(h * w, -1, dtype=np.int64)
    depth = np.full(h * w, np.inf)
    bary = np.zeros((h * w, 3))
    if T == 0:
        return tri_id.reshape(h, w), bary.reshape(h, w, 3), depth.reshape(h, w)
    wc = clip[:, :, 3]
    valid = (wc > 1e-6).all(axis=1)
    ndc = clip[:, :, :3] / np.where(np.abs(wc) < 1e-9, 1e-9, wc)[:, :, None]
    sx = (ndc[:, :, 0] * 0.5 + 0.5) * w
    sy = (1 - (ndc[:, :, 1] * 0.5 + 0.5)) * h
    z = ndc[:, :, 2]
    area = (sx[:, 1] - sx[:, 0]) * (sy[:, 2] - sy[:, 0]) - (sx[:, 2] - sx[:, 0]) * (sy[:, 1] - sy[:, 0])
    valid &= np.abs(area) > 1e-12
    if cull_backfaces:
        valid &= area < 0  # CCW in world -> CW in screen (y down)
    valid &= (z > -1).any(axis=1) & (z < 1).any(axis=1)
    x0 = np.clip(np.floor(sx.min(axis=1)), 0, w - 1).astype(np.int64)
    x1 = np.clip(np.ceil(sx.max(axis=1)), 0, w - 1).astype(np.int64)
    y0 = np.clip(np.floor(sy.min(axis=1)), 0, h - 1).astype(np.int64)
    y1 = np.clip(np.ceil(sy.max(axis=1)), 0, h - 1).astype(np.int64)
    valid &= (sx.max(axis=1) >= 0) & (sx.min(axis=1) <= w) & (sy.max(axis=1) >= 0) & (sy.min(axis=1) <= h)
    ids = np.flatnonzero(valid)
    bw = (x1 - x0 + 1)
    bh = (y1 - y0 + 1)
    npx = bw * bh
    order = ids[np.argsort(npx[ids])]
    # process in chunks of total candidate pixels
    cum = np.cumsum(npx[order])
    start = 0
    while start < len(order):
        base = cum[start - 1] if start > 0 else 0
        end = int(np.searchsorted(cum, base + chunk, side="right"))
        end = max(end, start + 1)
        tids = order[start:end]
        start = end
        cnt = npx[tids]
        tri = np.repeat(tids, cnt)
        offs = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        px = x0[tri] + offs % bw[tri]
        py = y0[tri] + offs // bw[tri]
        cx = px + 0.5
        cy = py + 0.5
        ax, ay = sx[tri, 0], sy[tri, 0]
        bx, by = sx[tri, 1], sy[tri, 1]
        qx, qy = sx[tri, 2], sy[tri, 2]
        ar = area[tri]
        w0 = ((bx - cx) * (qy - cy) - (qx - cx) * (by - cy)) / ar
        w1 = ((qx - cx) * (ay - cy) - (ax - cx) * (qy - cy)) / ar
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        tri, px, py, w0, w1, w2 = tri[inside], px[inside], py[inside], w0[inside], w1[inside], w2[inside]
        zz = w0 * z[tri, 0] + w1 * z[tri, 1] + w2 * z[tri, 2]
        ok = (zz >= -1) & (zz <= 1)
        tri, px, py, w0, w1, w2, zz = tri[ok], px[ok], py[ok], w0[ok], w1[ok], w2[ok], zz[ok]
        pix = py * w + px
        # keep nearest per pixel within chunk
        o = np.lexsort((zz, pix))
        pix_s = pix[o]
        firsts = np.concatenate([[True], pix_s[1:] != pix_s[:-1]])
        sel = o[firsts]
        pix_u = pix[sel]
        closer = zz[sel] < depth[pix_u]
        sel = sel[closer]
        pix_u = pix_u[closer]
        depth[pix_u] = zz[sel]
        tri_id[pix_u] = tri[sel]
        # perspective-correct barycentrics
        iw = 1.0 / wc[tri[sel]]
        pb = np.stack([w0[sel] * iw[:, 0], w1[sel] * iw[:, 1], w2[sel] * iw[:, 2]], axis=1)
        bary[pix_u] = pb / pb.sum(axis=1, keepdims=True)
    return tri_id.reshape(h, w), bary.reshape(h, w, 3), depth.reshape(h, w)


def _camera_for(scene_lo, scene_hi, view, aspect, yfov=35.0, margin=1.08, points=None):
    """Auto-frame: choose a camera on direction `view` that fits all points in the frame."""
    center = (scene_lo + scene_hi) / 2
    radius = max(np.linalg.norm(scene_hi - scene_lo) / 2, 1e-3)
    d = m3.normalize(m3.vec3(VIEW_DIRS[view] if isinstance(view, str) else view))
    up_hint = (0, 0, -1) if abs(d[1]) > 0.99 else (0, 1, 0)
    if points is None or len(points) == 0:
        lo, hi = scene_lo, scene_hi
        points = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    elif len(points) > 20000:
        points = points[np.linspace(0, len(points) - 1, 20000).astype(int)]
    fwd = -d
    right = m3.normalize(np.cross(fwd, up_hint))
    up = np.cross(right, fwd)
    rel = points - center
    px, py, pz = rel @ right, rel @ up, rel @ d  # pz: toward the camera
    # recentre on the projected extents
    cx = (px.max() + px.min()) / 2
    cy = (py.max() + py.min()) / 2
    center = center + right * cx + up * cy
    px, py = px - cx, py - cy
    ty = math.tan(math.radians(yfov) / 2) / margin
    tx = ty * aspect
    dist = float(max(np.max(pz + np.abs(px) / tx), np.max(pz + np.abs(py) / ty), radius * 0.2))
    eye = center + d * dist
    return m3.look_at_matrix(eye, center, up_hint), yfov, dist, radius


def render(scene, path=None, width: int = 800, height: int = 600, view="three_quarter", camera=None,
           style: str = "realistic", time: float | None = None, animation=None, background=None,
           shadows: bool = True, ground: bool = True, supersample: int = 2, yfov: float = 35.0,
           target=None, distance: float | None = None, bloom: float = 0.6, fog: float = 0.0,
           vignette: float = 0.25, exposure: float = 1.0) -> np.ndarray:
    """Render the scene and optionally save a PNG. Returns the (H, W, 3) uint8 image.

    view   : "front" | "back" | "left" | "right" | "top" | "three_quarter" | direction vector
    camera : a camera Node from the scene (overrides view)
    target/distance: override auto-framing (look at point, distance from it)
    time   : animation time in seconds (poses rigs / animated nodes)
    bloom  : glow strength around bright/emissive pixels (0 = off)
    fog    : atmospheric fog density per meter (e.g. 0.02 for landscapes)
    vignette: darkening of the image corners (0 = off)
    exposure: brightness multiplier before tone mapping
    """
    ss = max(int(supersample), 1)
    W, H = width * ss, height * ss
    batch = _collect(scene, time, animation)
    P, N, UV, C, M, E = batch.finish()
    mats = batch.mats
    aspect = W / H
    # camera
    if camera is not None:
        cam_w = camera.world_matrix()
        cam_w = m3.trs_matrix(cam_w[:3, 3], m3.matrix3_to_quat(m3.normalize(cam_w[:3, :3], axis=0)))
        fov = camera.camera.yfov if camera.camera is not None else yfov
        lo, hi = (P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)) if len(P) else (np.zeros(3), np.ones(3))
        radius = max(np.linalg.norm(hi - lo) / 2, 1e-3)
        dist = np.linalg.norm(cam_w[:3, 3] - (lo + hi) / 2)
    else:
        if len(P):
            lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
        else:
            lo, hi = -np.ones(3), np.ones(3)
        cam_w, fov, dist, radius = _camera_for(lo, hi, view, aspect, yfov, points=P.reshape(-1, 3))
        if target is not None or distance is not None:
            c = (lo + hi) / 2 if target is None else m3.vec3(target)
            d = m3.normalize(m3.vec3(VIEW_DIRS[view] if isinstance(view, str) else view))
            radius = max(radius, float(np.linalg.norm(hi - lo) / 2))
            dist = distance if distance is not None else dist
            up = (0, 0, -1) if abs(d[1]) > 0.99 else (0, 1, 0)
            cam_w = m3.look_at_matrix(c + d * dist, c, up)
    view_m = np.linalg.inv(cam_w)
    near = max(dist - radius * 3, dist * 0.01, 0.01)
    far = dist + radius * 4 + 10
    proj = m3.perspective_matrix(fov, aspect, near, far)
    eye = cam_w[:3, 3]

    # ground plane (shadow catcher)
    ground_y = min(float(P[:, :, 1].min()), float(scene.bounds()[0][1])) if len(P) else 0.0
    use_ground = ground and style in ("realistic", "clay", "toon") and len(P)

    # rasterize
    VP = proj @ view_m
    clip = np.concatenate([P, np.ones(P.shape[:2] + (1,))], axis=2) @ VP.T
    tri_id, bary, depth = _rasterize(clip, W, H)

    bg = _background(H, W, background if background is not None else scene.background, style)
    img = bg.copy()
    mask = tri_id >= 0
    hdr = np.zeros((H, W, 3)) if bloom > 0 and style != "normals" else None

    # lights
    lights = _scene_lights(scene)
    key = next((l for l in lights if l["type"] == "directional" and l["shadow"]), None)
    shadow_fn = None
    if shadows and key is not None and len(P) and style in ("realistic", "clay", "toon"):
        shadow_fn = _shadow_map(P, key["dir"], 1024 if ss <= 2 else 1536)

    if mask.any():
        t = tri_id[mask]
        bw = bary[mask]
        pos = np.einsum("nk,nkj->nj", bw, P[t])
        nrm = m3.normalize(np.einsum("nk,nkj->nj", bw, N[t]))
        uv = np.einsum("nk,nkj->nj", bw, UV[t])
        vcol = np.einsum("nk,nkj->nj", bw, C[t])
        view_dir = m3.normalize(eye - pos)
        # double-sided: flip normals facing away
        back = np.einsum("ij,ij->i", nrm, view_dir) < 0
        nrm[back] = -nrm[back]
        mid = M[t]
        base, rough, metal, emis, unlit, alpha = _material_inputs(mats, mid, uv, vcol, style)
        if style == "normals":
            col = nrm * 0.5 + 0.5
            col = m3.srgb_to_linear(col)
        else:
            sh = shadow_fn(pos, nrm) if shadow_fn is not None else np.ones(len(pos))
            col = _shade(pos, nrm, view_dir, base, rough, metal, lights, sh, style, scene.ambient)
            col = col + emis
            col = np.where(unlit[:, None], base + emis, col)
        col = col * exposure
        if hdr is not None:
            hdr[mask] = col
        if style == "wire":
            e = E[t]
            # distance to real polygon edges in barycentric space
            dmin = np.full(len(bw), 9.0)
            for k in range(3):
                # edge k connects corner k and k+1 -> opposite barycentric is (k+2)%3
                dk = np.where(e[:, k], bw[:, (k + 2) % 3], 9.0)
                dmin = np.minimum(dmin, dk)
            px_scale = _bary_pixel_scale(tri_id, bary)[mask]
            line = dmin * px_scale < 0.9 * ss
            col = np.where(line[:, None], np.array([0.02, 0.02, 0.02]), col)
        col = _tonemap(col)
        if fog > 0:
            f = 1 - np.exp(-fog * np.linalg.norm(pos - eye, axis=1))
            col = col * (1 - f[:, None]) + img[mask] * f[:, None]
        a = np.clip(alpha, 0, 1)[:, None]
        img[mask] = col * a + img[mask] * (1 - a)

    if use_ground:
        img = _ground_shadow(img, mask, cam_w, proj, W, H, ground_y, shadow_fn, lo, hi, style)

    if style == "toon" and mask.any():
        img = _outline(img, depth, tri_id, ss)

    if hdr is not None:
        lum = hdr.max(axis=2)
        bright = hdr * (np.clip(lum - 1.0, 0, None) / np.maximum(lum, 1e-6))[:, :, None]
        if bright.max() > 0:
            r = max(int(W * 0.012), 2)
            glow = _blur(bright, r) + 0.5 * _blur(bright, r * 3)
            lin = m3.srgb_to_linear(np.clip(img, 0, 1)) + glow * bloom
            img = m3.linear_to_srgb(np.clip(lin, 0, 1))
    if vignette > 0:
        yy, xx = np.mgrid[0:H, 0:W]
        d = np.hypot((xx / W - 0.5) * 2, (yy / H - 0.5) * 2) / math.sqrt(2)
        img = img * (1 - vignette * np.clip(d - 0.35, 0, 1) ** 1.5 / 0.65 ** 1.5)[:, :, None]

    out = downsample(img, ss) if ss > 1 else img
    out8 = to_uint8(np.clip(out, 0, 1))
    if path is not None:
        write_png(path, out8)
    return out8


def _blur(img, r: int):
    """Approximate gaussian blur: three passes of a separable box blur (cumsum)."""
    out = img
    for _ in range(3):
        for ax in (0, 1):
            pad = [(0, 0)] * out.ndim
            pad[ax] = (r + 1, r)
            c = np.cumsum(np.pad(out, pad, mode="edge"), axis=ax)
            n = out.shape[ax]
            hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=ax)
            lo = np.take(c, np.arange(0, n), axis=ax)
            out = (hi - lo) / (2 * r + 1)
    return out


def _bary_pixel_scale(tri_id, bary):
    # approximate pixels per unit barycentric using screen-space derivative
    gy, gx = np.gradient(bary, axis=(0, 1))
    d = np.sqrt((gx ** 2).sum(axis=2) + (gy ** 2).sum(axis=2))
    same = (tri_id == np.roll(tri_id, 1, axis=1)) & (tri_id >= 0)
    d = np.where(same | (d > 0), d, 0.05)
    return 1.0 / np.maximum(d, 1e-6)


def _background(H, W, color, style):
    c = m3.srgb_to_linear(m3.as_color(color)[:3])
    t = np.linspace(0, 1, H)[:, None, None]
    top = c * 1.08
    bot = c * 0.78
    img = top * (1 - t) + bot * t
    img = np.broadcast_to(img, (H, W, 3)).copy()
    return _tonemap_bg(img)


def _tonemap_bg(img):
    return m3.linear_to_srgb(np.clip(img, 0, 1))


def _tonemap(col):
    # ACES-ish filmic curve then sRGB
    a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    x = np.clip(col, 0, None) * 0.9
    mapped = np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0, 1)
    return m3.linear_to_srgb(mapped)


def _scene_lights(scene):
    out = []
    for n in scene.walk():
        if n.light is None or not n.visible:
            continue
        W = n.world_matrix()
        fwd = m3.normalize(-W[:3, 2])
        out.append({"type": n.light.type, "dir": fwd, "pos": W[:3, 3].copy(), "color": n.light.color_linear,
                    "intensity": n.light.intensity, "shadow": n.light.cast_shadows,
                    "outer": math.radians(n.light.outer_cone), "inner": math.radians(n.light.inner_cone),
                    "range": n.light.range})
    if not out:  # default studio rig
        for d, inten, col, sh in [((-0.45, -0.8, -0.5), 3.0, (1, 0.97, 0.92), True),
                                  ((0.7, -0.4, -0.3), 1.0, (0.8, 0.88, 1.0), False),
                                  ((0.0, -0.4, 1.0), 1.4, (1, 1, 1), False)]:
            out.append({"type": "directional", "dir": m3.normalize(m3.vec3(d)), "pos": np.zeros(3),
                        "color": np.array(col), "intensity": inten, "shadow": sh})
    return out


def _material_inputs(mats, mid, uv, vcol, style):
    n = len(mid)
    base = np.ones((n, 3)) * 0.8
    rough = np.full(n, 0.5)
    metal = np.zeros(n)
    emis = np.zeros((n, 3))
    unlit = np.zeros(n, dtype=bool)
    alpha = np.ones(n)
    for k in np.unique(mid):
        sel = mid == k
        mt = mats[k]
        if style == "clay" or style == "wire":
            base[sel] = 0.55
            rough[sel] = 0.7
            continue
        if mt is None:
            base[sel] = 0.72 * vcol[sel, :3]
            continue
        bc = mt.base_color_linear
        col = np.tile(bc[:3], (sel.sum(), 1))
        a = np.full(sel.sum(), bc[3])
        if mt.base_color_texture is not None:
            tx = mt.base_color_texture.sample(uv[sel])
            col = col * m3.srgb_to_linear(tx[:, :3])
            a = a * tx[:, 3]
        col = col * vcol[sel, :3]
        base[sel] = col
        alpha[sel] = a if mt.resolved_alpha_mode != "OPAQUE" else 1.0
        if mt.resolved_alpha_mode == "MASK":
            alpha[sel] = (a >= mt.alpha_cutoff).astype(float)
        rough[sel] = mt.roughness
        metal[sel] = mt.metallic
        if mt.metallic_roughness_texture is not None:
            mr = mt.metallic_roughness_texture.sample(uv[sel])
            rough[sel] = rough[sel] * mr[:, 1]
            metal[sel] = metal[sel] * mr[:, 2]
        e = mt.emissive_linear * mt.emissive_strength
        if mt.emissive_texture is not None:
            e = e * m3.srgb_to_linear(mt.emissive_texture.sample(uv[sel])[:, :3])
        emis[sel] = e
        unlit[sel] = mt.unlit
    return base, rough, metal, emis, unlit, alpha


def _shade(pos, nrm, view_dir, base, rough, metal, lights, shadow, style, ambient):
    n = len(pos)
    col = np.zeros((n, 3))
    diffuse_c = base * (1 - metal[:, None])
    f0 = 0.04 * (1 - metal[:, None]) + base * metal[:, None]
    a2 = np.maximum(rough, 0.05)[:, None] ** 4
    for i, L in enumerate(lights):
        if L["type"] == "directional":
            ldir = -L["dir"][None, :]
            atten = np.ones(n) * L["intensity"] / 3.0
        else:
            d = L["pos"][None, :] - pos
            dist = np.linalg.norm(d, axis=1)
            ldir = d / np.maximum(dist, 1e-6)[:, None]
            atten = L["intensity"] / np.maximum(dist * dist, 1e-4) / 40.0
            if L.get("range"):
                atten *= np.clip(1 - (dist / L["range"]) ** 4, 0, 1)
            if L["type"] == "spot":
                cosang = np.einsum("ij,j->i", -ldir, L["dir"])
                atten *= m3.smoothstep(math.cos(L["outer"]), math.cos(L["inner"]), cosang)
        ndl = np.clip(np.einsum("ij,ij->i", nrm, ldir), 0, 1)
        if style == "toon":
            ndl = np.where(ndl > 0.5, 1.0, np.where(ndl > 0.08, 0.55, 0.0))
        s = shadow if (L.get("shadow") and L["type"] == "directional") else 1.0
        h = m3.normalize(ldir + view_dir)
        ndh = np.clip(np.einsum("ij,ij->i", nrm, h), 0, 1)[:, None]
        vdh = np.clip(np.einsum("ij,ij->i", view_dir, h), 0, 1)[:, None]
        ndv = np.clip(np.einsum("ij,ij->i", nrm, view_dir), 1e-3, 1)[:, None]
        D = a2 / (math.pi * (ndh * ndh * (a2 - 1) + 1) ** 2)
        F = f0 + (1 - f0) * (1 - vdh) ** 5
        k = a2 ** 0.5 / 2
        G = 1 / ((ndv * (1 - k) + k) * (ndl[:, None] * (1 - k) + k) + 1e-6)
        spec = D * F * G / 4.0
        if style == "toon":
            spec = (spec.mean(axis=1, keepdims=True) > 0.6) * 0.25
        lc = L["color"][None, :] * (atten * ndl * s)[:, None]
        col += (diffuse_c / math.pi * math.pi + spec) * lc
    # hemisphere ambient + simple environment reflection for metals
    sky = np.array([0.75, 0.82, 1.0])
    gnd = np.array([0.35, 0.3, 0.27])
    t = (nrm[:, 1] * 0.5 + 0.5)[:, None]
    amb = (gnd * (1 - t) + sky * t) * ambient
    if style == "toon":
        amb = np.full_like(amb, ambient * 0.9)
    col += diffuse_c * amb
    refl = 2 * np.einsum("ij,ij->i", nrm, view_dir)[:, None] * nrm - view_dir
    tr = (refl[:, 1] * 0.5 + 0.5)[:, None]
    env = (gnd * (1 - tr) + sky * tr) * (1 - np.clip(rough, 0, 1)[:, None] * 0.7)
    ndv = np.clip(np.einsum("ij,ij->i", nrm, view_dir), 0, 1)[:, None]
    fres = f0 + (1 - f0) * (1 - ndv) ** 5 * (1 - rough[:, None])
    col += env * fres * 0.6
    return col


def _shadow_map(P, light_dir, size):
    """Orthographic depth map from a directional light. Returns fn(points, normals) -> visibility."""
    pts = P.reshape(-1, 3)
    up = (0, 0, 1) if abs(light_dir[1]) > 0.99 else (0, 1, 0)
    center = (pts.min(0) + pts.max(0)) / 2
    radius = max(np.linalg.norm(pts.max(0) - pts.min(0)) / 2, 1e-3) * 1.6
    lw = m3.look_at_matrix(center - light_dir * radius * 3, center, up)
    lv = np.linalg.inv(lw)
    proj = m3.orthographic_matrix(radius, radius, 0.01, radius * 8)
    LP = proj @ lv
    clip = np.concatenate([P, np.ones(P.shape[:2] + (1,))], axis=2) @ LP.T
    _, _, depth = _rasterize(clip, size, size)
    texel = 2 * radius / size

    def vis(points, normals):
        p = points + normals * texel * 2.0
        c = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ LP.T
        x = (c[:, 0] * 0.5 + 0.5) * size
        y = (1 - (c[:, 1] * 0.5 + 0.5)) * size
        z = c[:, 2]
        acc = np.zeros(len(p))
        offs = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
        for dx, dy in offs:
            xi = np.clip((x + dx).astype(int), 0, size - 1)
            yi = np.clip((y + dy).astype(int), 0, size - 1)
            acc += (z <= depth[yi, xi] + 0.004).astype(float)
        out = acc / len(offs)
        inside = (x >= 0) & (x < size) & (y >= 0) & (y < size)
        return np.where(inside, out, 1.0)
    return vis


def _ground_shadow(img, mask, cam_w, proj, W, H, ground_y, shadow_fn, lo, hi, style):
    """Darken background pixels where the ground plane receives shadow/contact occlusion."""
    ys, xs = np.nonzero(~mask)
    if len(ys) == 0:
        return img
    ndc_x = (xs + 0.5) / W * 2 - 1
    ndc_y = 1 - (ys + 0.5) / H * 2
    inv_p = np.linalg.inv(proj)
    pts = np.stack([ndc_x, ndc_y, np.ones_like(ndc_x), np.ones_like(ndc_x)], axis=1) @ inv_p.T
    dirs = pts[:, :3] / pts[:, 3:4]
    dirs = m3.normalize(dirs @ cam_w[:3, :3].T)
    eye = cam_w[:3, 3]
    if eye[1] <= ground_y + 1e-6:
        return img
    t = (ground_y - eye[1]) / np.where(np.abs(dirs[:, 1]) < 1e-9, -1e-9, dirs[:, 1])
    hit = t > 0
    gp = eye + dirs * t[:, None]
    size = max(hi[0] - lo[0], hi[2] - lo[2], 1e-3)
    center = (lo + hi) / 2
    r = np.hypot(gp[:, 0] - center[0], gp[:, 2] - center[2]) / size
    fade = np.clip(1 - (r - 0.6) / 2.5, 0, 1) * hit
    vis = np.ones(len(gp))
    if shadow_fn is not None:
        vis = shadow_fn(gp, np.tile([0, 1, 0], (len(gp), 1)))
    # soft contact occlusion under the object footprint
    dx = np.maximum(np.maximum(lo[0] - gp[:, 0], gp[:, 0] - hi[0]), 0)
    dz = np.maximum(np.maximum(lo[2] - gp[:, 2], gp[:, 2] - hi[2]), 0)
    contact = 1 - 0.35 * np.exp(-np.hypot(dx, dz) / (size * 0.08))
    dark = (1 - (1 - vis * contact) * 0.55 * fade)
    if style == "toon":
        dark = np.where(dark < 0.85, 0.75, 1.0)
    img[ys, xs] = img[ys, xs] * dark[:, None]
    return img


def _outline(img, depth, tri_id, ss):
    d = np.where(np.isfinite(depth), depth, 1.0)
    edge = np.zeros(d.shape, dtype=bool)
    for ax in (0, 1):
        dd = np.abs(np.diff(d, axis=ax))
        e = dd > 0.002
        if ax == 0:
            edge[:-1] |= e
        else:
            edge[:, :-1] |= e
    obj = tri_id >= 0
    sil = obj != np.roll(obj, 1, axis=0)
    sil |= obj != np.roll(obj, 1, axis=1)
    edge |= sil
    k = max(ss, 1)
    grow = edge.copy()
    for s in range(1, k):
        grow |= np.roll(edge, s, axis=0) | np.roll(edge, s, axis=1)
    img = img.copy()
    img[grow] = img[grow] * 0.1
    return img


# ---------------------------------------------------------------------------
# convenience renders
# ---------------------------------------------------------------------------

def render_views(scene, path=None, views=("front", "right", "back", "three_quarter", "top", "left"),
                 size: int = 400, cols: int = 3, **kw) -> np.ndarray:
    """Contact sheet of several views (order: left->right, top->bottom as listed)."""
    imgs = [render(scene, None, size, size, view=v, **kw) for v in views]
    sheet = grid(imgs, cols)
    if path is not None:
        write_png(path, sheet)
    return sheet


def render_frames(scene, path=None, animation=None, frames: int = 6, size: int = 320, cols: int | None = None,
                  view="three_quarter", **kw) -> np.ndarray:
    """Filmstrip of an animation sampled at evenly spaced times."""
    anim = animation
    if anim is None:
        anim = scene.animations[0]
    elif isinstance(anim, str):
        anim = next(a for a in scene.animations if a.name == anim)
    dur = anim.duration
    times = np.linspace(anim.start, dur, frames, endpoint=not anim.loop) if frames > 1 else [0.0]
    # fixed camera framing over the whole animation (union of the bounds of every frame)
    if "target" not in kw and "camera" not in kw:
        los, his = [], []
        for t in times:
            b = _collect(scene, float(t), anim).finish()[0].reshape(-1, 3)
            if len(b):
                los.append(b.min(0))
                his.append(b.max(0))
        if los:
            lo, hi = np.min(los, axis=0), np.max(his, axis=0)
            corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            cam, fov, dist, _ = _camera_for(lo, hi, view, 1.0, kw.get("yfov", 35.0), points=corners)
            d = m3.normalize(m3.vec3(VIEW_DIRS[view] if isinstance(view, str) else view))
            kw["target"] = cam[:3, 3] - d * dist
            kw["distance"] = dist
    imgs = [render(scene, None, size, size, view=view, time=float(t), animation=anim, **kw) for t in times]
    sheet = grid(imgs, cols or frames)
    if path is not None:
        write_png(path, sheet)
    return sheet
