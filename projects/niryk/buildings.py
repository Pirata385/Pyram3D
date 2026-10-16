"""Village structures of Rondoria (3D, front faces +Z, standing on y = 0).

    python projects/niryk/buildings.py

Footprints must match src/shared/world/worldgen.ts (STRUCTURES) in the game repo.
Textures are small painted tiles at 32 px per meter with nearest filtering, so the buildings
read like the painted ground and sprites (G1 review).
"""
from __future__ import annotations

import math

import numpy as np

from common import REVIEW, export_glb

import pyram3d as p3
from pyram3d.noise import fbm2

PX_PER_M = 32
RNG = np.random.default_rng(7)


# ----------------------------------------------------------------------------- painted textures

def _hex(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def _bands(v, cuts, cols):
    idx = np.digitize(v, cuts)
    return np.stack([_hex(c) for c in cols])[idx]


def _tex(img, name, meters):
    """img (H, W, 3) floats -> nearest, repeating Texture; `meters` = world size of the tile."""
    rgba = np.concatenate([np.clip(img, 0, 1), np.ones(img.shape[:2] + (1,))], -1)
    t = p3.Texture((rgba * 255).astype(np.uint8), name=name, filter="nearest", wrap="repeat")
    t.extras = {"meters": meters}
    return t


def _noise(n, scale, seed):
    ys, xs = np.mgrid[0:n, 0:n].astype(float)
    return fbm2(xs / scale, ys / scale, octaves=3, seed=seed) * 0.5 + 0.5


def tex_plaster(color="#d8c6a2", n=128):
    v = _noise(n, 18, 1)
    img = _bands(v, [0.42, 0.62], [_shade(color, 0.92), color, _shade(color, 1.05)]).copy()
    for _ in range(12):  # tiny cracks / speckles
        x, y = RNG.integers(0, n, 2)
        img[y % n, x % n] = _hex(_shade(color, 0.78))
        img[(y + 1) % n, (x + 1) % n] = _hex(_shade(color, 0.78))
    return _tex(img, "Plaster", n / PX_PER_M)


def tex_stone(n=128):
    """Irregular stone courses with dark mortar, light top-left bevel."""
    img = np.zeros((n, n, 3))
    row_h = 9
    y = 0
    base = _hex("#9a9488")
    while y < n:
        h = int(RNG.integers(row_h - 2, row_h + 3))
        x = int(RNG.integers(-12, 0))
        while x < n:
            w = int(RNG.integers(12, 22))
            tone = base * RNG.uniform(0.82, 1.08)
            for yy in range(y, min(n, y + h)):
                for xx in range(max(0, x), min(n, x + w)):
                    c = tone
                    if yy == y or xx == x:
                        c = np.minimum(1, tone * 1.18)
                    if yy == y + h - 1 or xx == x + w - 1:
                        c = tone * 0.72
                    img[yy, xx] = c
            for yy in range(y, min(n, y + h)):  # mortar
                if 0 <= x + w - 1 < n:
                    img[yy, x + w - 1] = _hex("#4b4038")
            x += w
        img[min(n - 1, y + h - 1), :] = _hex("#4b4038")
        y += h
    return _tex(img, "Stone", n / PX_PER_M)


def tex_planks(color="#8a5a3a", n=64, vertical=True):
    img = np.zeros((n, n, 3))
    w = 8
    for x0 in range(0, n, w):
        tone = _hex(color) * RNG.uniform(0.85, 1.08)
        img[:, x0:x0 + w] = tone
        img[:, x0] = tone * 0.65  # gap
        img[:, x0 + 1] = np.minimum(1, tone * 1.12)
        for _ in range(3):  # grain
            gy = int(RNG.integers(0, n))
            gl = int(RNG.integers(5, 14))
            gx = x0 + int(RNG.integers(2, w - 1))
            img[gy:gy + gl, gx] = tone * 0.8
    if not vertical:
        img = img.transpose(1, 0, 2)
    return _tex(img, "Planks", n / PX_PER_M)


def tex_roof_tiles(color="#b0543a", n=64):
    """Rows of rounded clay tiles: each tile lit on top, shadow line at the bottom."""
    img = np.zeros((n, n, 3))
    rh, tw = 8, 8
    for r, y0 in enumerate(range(0, n, rh)):
        off = (tw // 2) * (r % 2)
        for x0 in range(-off, n, tw):
            tone = _hex(color) * RNG.uniform(0.88, 1.06)
            for yy in range(y0, min(n, y0 + rh)):
                for xx in range(max(0, x0), min(n, x0 + tw)):
                    u = (xx - x0 + 0.5) / tw
                    rr = (yy - y0) / rh
                    c = tone * (1.12 - 0.3 * abs(u - 0.4) * 2) * (1.05 - rr * 0.25)
                    if yy == y0 + rh - 1:
                        c = tone * 0.55
                    img[yy, xx] = np.clip(c, 0, 1)
            if 0 <= x0 < n:
                img[y0:y0 + rh, x0] = tone * 0.6
    return _tex(img, "RoofTiles", n / PX_PER_M)


def tex_slate(n=64):
    return tex_roof_tiles("#5f6b80", n)


def tex_thatch(n=64):
    img = np.zeros((n, n, 3))
    base = _hex("#c9a24a")
    for y0 in range(0, n, 10):
        img[y0:y0 + 10] = base * RNG.uniform(0.92, 1.04)
        img[min(n - 1, y0 + 9)] = base * 0.55
        for _ in range(40):  # straw strokes
            x = int(RNG.integers(0, n))
            y = y0 + int(RNG.integers(0, 8))
            ln = int(RNG.integers(3, 7))
            tone = base * RNG.choice([0.75, 1.15, 0.9])
            img[y:min(n, y + ln), x] = np.clip(tone, 0, 1)
    return _tex(img, "Thatch", n / PX_PER_M)


def _shade(c, k):
    v = np.clip(_hex(c) * k, 0, 1)
    return "#%02x%02x%02x" % tuple(int(x * 255) for x in v)


def textured(mesh: p3.Mesh, tex: p3.Texture, name: str, color="#ffffff") -> p3.Mesh:
    mesh.uv_box(scale=1.0 / tex.extras["meters"])
    return mesh.set_material(p3.Material(name, color, roughness=0.9, base_color_texture=tex))


def flat(name, color, **kw):
    return p3.Material(name, color, roughness=kw.pop("roughness", 0.85), **kw)


TEX = {}


def T(name):
    if name not in TEX:
        TEX[name] = {
            "plaster": lambda: tex_plaster(),
            "plaster_b": lambda: tex_plaster("#cdb994"),
            "stone": tex_stone,
            "planks": lambda: tex_planks(),
            "planks_dark": lambda: tex_planks("#6a4630"),
            "tiles": lambda: tex_roof_tiles(),
            "slate": tex_slate,
            "thatch": tex_thatch,
        }[name]()
    return TEX[name]


BEAM = flat("Beam", "#4e3424")
DOOR = flat("Door", "#6a4630")
WINDOW = flat("Window", "#2b3348", emissive="#3a4a66", emissive_strength=0.15)
SHUTTER = flat("Shutter", "#3f6a5a")
IRON = flat("Iron", "#3a3a42", metallic=0.6, roughness=0.5)
FLOWERS = flat("Flowers", "#d9505a")
LEAVES = flat("Leaves", "#4b8535")
GLOW = p3.Material("LanternGlow", "#ffd27a", emissive="#ffb84a", emissive_strength=2.0, unlit=True)


# ----------------------------------------------------------------------------- house builder

def gable_roof(w, d, h, overhang=0.35, thick=0.16, tex="tiles"):
    """Two sloped slabs meeting at the ridge (along X), with overhang."""
    W = w + overhang * 2
    half = d / 2 + overhang
    slope = math.hypot(half, h)
    parts = []
    ang = math.degrees(math.atan2(h, half))
    for back in (False, True):
        # slab from the ridge (origin) toward +Z, tilted down to the eave; the back one is turned 180
        slab = p3.box((W, thick, slope)).translate((0, thick / 2, slope / 2))
        slab.rotate((ang, 0, 0))
        if back:
            slab.rotate((0, 180, 0))
        slab.translate((0, h, 0))
        parts.append(slab)
    roof = p3.Mesh.join(parts, name="Roof")
    return textured(roof, T(tex), "Roof_" + tex)


def gable_wall(w, h, depth):
    """Triangular gable end (front/back) as a thin prism."""
    shape = [(-w / 2, 0), (w / 2, 0), (0, h)]
    m = p3.extrude_shape(shape, depth)
    return m


def window(w=0.7, h=0.8):
    parts = [p3.box((w, h, 0.08)).set_material(WINDOW)]
    frame = [p3.box((w + 0.12, 0.08, 0.14)).translate((0, h / 2 + 0.04, 0)), p3.box((w + 0.12, 0.08, 0.14)).translate((0, -h / 2 - 0.04, 0)),
             p3.box((0.06, h, 0.12)), p3.box((w, 0.05, 0.12))]
    for f in frame:
        parts.append(f.set_material(BEAM))
    for sx in (-1, 1):  # open shutters
        parts.append(p3.box((w * 0.5, h, 0.05)).translate((sx * (w * 0.75 + 0.06), 0, 0.03)).set_material(SHUTTER))
    return p3.Mesh.join(parts, name="Window")


def door(w=1.0, h=1.9):
    parts = [textured(p3.box((w, h, 0.1)).translate((0, h / 2, 0)), T("planks_dark"), "Door")]
    parts.append(p3.box((w + 0.2, 0.14, 0.18)).translate((0, h + 0.07, 0)).set_material(BEAM))
    for sx in (-1, 1):
        parts.append(p3.box((0.12, h, 0.16)).translate((sx * (w / 2 + 0.06), h / 2, 0)).set_material(BEAM))
    parts.append(p3.uv_sphere(0.04, 6, 4).translate((w * 0.3, h * 0.5, 0.07)).set_material(IRON))
    return p3.Mesh.join(parts, name="Door")


def house(w, d, wall_h, roof_h, wall_tex="plaster", roof_tex="tiles", plinth=0.55, beams=True, chimney=True,
          upper=None, name="House"):
    parts = []
    # stone plinth
    parts.append(textured(p3.box((w, plinth, d)).translate((0, plinth / 2, 0)), T("stone"), "Stone"))
    # walls
    wh = wall_h - plinth
    walls = textured(p3.box((w - 0.12, wh, d - 0.12)).translate((0, plinth + wh / 2, 0)), T(wall_tex), "Wall_" + wall_tex)
    parts.append(walls)
    top = wall_h
    if upper:  # jettied upper floor (overhangs a bit)
        uh = upper
        parts.append(p3.box((w + 0.25, 0.18, d + 0.25)).translate((0, top + 0.09, 0)).set_material(BEAM))
        parts.append(textured(p3.box((w + 0.1, uh, d + 0.1)).translate((0, top + 0.18 + uh / 2, 0)), T("plaster_b"), "Wall_plaster_b"))
        top = top + 0.18 + uh
    # gable ends on the left/right sides (the ridge runs along X)
    for x in (w / 2 - 0.06, -w / 2 + 0.06):
        g = gable_wall(d - 0.1, roof_h * 0.95, 0.12).translate((0, 0, -0.06)).rotate((0, 90, 0)).translate((x, top, 0))
        parts.append(textured(g, T(wall_tex), "Wall_" + wall_tex))
    if beams:  # timber framing
        for x in (-w / 2 + 0.07, w / 2 - 0.07):
            for z in (-d / 2 + 0.07, d / 2 - 0.07):
                parts.append(p3.box((0.16, top - plinth, 0.16)).translate((x, plinth + (top - plinth) / 2, z)).set_material(BEAM))
        for z in (d / 2 - 0.01, -d / 2 + 0.01):
            parts.append(p3.box((w, 0.14, 0.1)).translate((0, wall_h - 0.07, z)).set_material(BEAM))
            for sx in (-1, 1):  # diagonal braces
                br = p3.box((0.1, (wh * 0.8), 0.08)).rotate((0, 0, sx * 35)).translate((sx * w * 0.32, plinth + wh * 0.5, z))
                parts.append(br.set_material(BEAM))
        for x in (w / 2 - 0.01, -w / 2 + 0.01):
            parts.append(p3.box((0.1, 0.14, d)).translate((x, wall_h - 0.07, 0)).set_material(BEAM))
    # roof
    roof = gable_roof(w, d, roof_h, tex=roof_tex).translate((0, top, 0))
    parts.append(roof)
    parts.append(p3.box((w + 0.8, 0.16, 0.2)).translate((0, top + roof_h + 0.05, 0)).set_material(BEAM))  # ridge
    if chimney:
        parts.append(textured(p3.box((0.6, roof_h + 1.0, 0.6)).translate((w * 0.28, top + (roof_h + 1.0) / 2, -d * 0.2)), T("stone"), "Stone"))
        parts.append(p3.box((0.72, 0.12, 0.72)).translate((w * 0.28, top + roof_h + 1.02, -d * 0.2)).set_material(IRON))
    # door and windows on the front (+Z), windows on the sides
    parts.append(door().translate((-w * 0.18, plinth - 0.02, d / 2 + 0.03)))
    parts.append(window().translate((w * 0.24, plinth + wh * 0.55, d / 2 + 0.03)))
    # flower box under the front window
    parts.append(textured(p3.box((0.9, 0.2, 0.22)).translate((w * 0.24, plinth + wh * 0.55 - 0.55, d / 2 + 0.12)), T("planks"), "Wood"))
    for k in range(5):
        parts.append(p3.uv_sphere(0.08, 6, 4).translate((w * 0.24 - 0.32 + k * 0.16, plinth + wh * 0.55 - 0.4, d / 2 + 0.13)).set_material(FLOWERS if k % 2 else LEAVES))
    for sx in (-1, 1):
        wnd = window(0.6, 0.7).rotate((0, 90 * sx, 0)).translate((sx * (w / 2 + 0.03), plinth + wh * 0.55, 0))
        parts.append(wnd)
    # door step
    parts.append(textured(p3.box((1.3, 0.16, 0.6)).translate((-w * 0.18, 0.08, d / 2 + 0.3)), T("stone"), "Stone"))
    return p3.Mesh.join(parts, name=name)


def thatched_cottage(w=5.2, d=4.6):
    parts = []
    parts.append(textured(p3.box((w, 0.45, d)).translate((0, 0.225, 0)), T("stone"), "Stone"))
    parts.append(textured(p3.box((w - 0.1, 2.2, d - 0.1)).translate((0, 0.45 + 1.1, 0)), T("planks"), "Wall_planks"))
    for x in (w / 2 - 0.06, -w / 2 + 0.06):
        g = gable_wall(d - 0.1, 1.95, 0.12).translate((0, 0, -0.06)).rotate((0, 90, 0)).translate((x, 2.65, 0))
        parts.append(textured(g, T("planks"), "Wall_planks"))
    roof = gable_roof(w, d, 2.0, overhang=0.5, thick=0.38, tex="thatch").translate((0, 2.62, 0))
    parts.append(roof)
    parts.append(door(0.95, 1.8).translate((w * 0.15, 0.43, d / 2 + 0.03)))
    parts.append(window(0.6, 0.6).translate((-w * 0.24, 1.6, d / 2 + 0.03)))
    for sx in (-1, 1):
        parts.append(window(0.55, 0.6).rotate((0, 90 * sx, 0)).translate((sx * (w / 2 + 0.03), 1.6, 0)))
    for x in (-w / 2 + 0.07, w / 2 - 0.07):
        for z in (-d / 2 + 0.07, d / 2 - 0.07):
            parts.append(p3.box((0.16, 2.2, 0.16)).translate((x, 1.55, z)).set_material(BEAM))
    parts.append(textured(p3.box((1.2, 0.14, 0.55)).translate((w * 0.15, 0.07, d / 2 + 0.28)), T("stone"), "Stone"))
    # woodpile by the side
    for k in range(6):
        log = p3.cylinder(0.11, 0.9, 7).rotate((90, 0, 0)).translate((w / 2 + 0.35, 0.12 + (k // 3) * 0.2, -0.8 + (k % 3) * 0.24))
        parts.append(textured(log, T("planks"), "Wood"))
    return p3.Mesh.join(parts, name="House_C")


# ----------------------------------------------------------------------------- props

def well():
    parts = []
    ring = p3.lathe([(0.0, 0), (1.15, 0), (1.15, 0.85), (0.92, 0.85), (0.92, 0.1), (0.0, 0.1)], 16)
    parts.append(textured(ring, T("stone"), "Stone"))
    parts.append(p3.cylinder(0.9, 0.05, 16).translate((0, 0.55, 0)).set_material(flat("Water", "#2f5f86", emissive="#14304a", emissive_strength=0.3)))
    for sx in (-1, 1):
        parts.append(textured(p3.box((0.16, 1.9, 0.16)).translate((sx * 0.95, 1.6, 0)), T("planks_dark"), "Wood"))
    parts.append(textured(p3.cylinder(0.08, 2.1, 8).rotate((0, 0, 90)).translate((0, 2.1, 0)), T("planks_dark"), "Wood"))
    roof = gable_roof(2.4, 1.6, 0.7, overhang=0.15, thick=0.1, tex="tiles").translate((0, 2.5, 0))
    parts.append(roof)
    parts.append(p3.box((0.03, 0.9, 0.03)).translate((0, 1.6, 0)).set_material(IRON))  # rope
    bucket = p3.lathe([(0, 0), (0.16, 0), (0.19, 0.24), (0.17, 0.24), (0.14, 0.03), (0, 0.03)], 10).translate((0, 1.05, 0))
    parts.append(textured(bucket, T("planks"), "Wood"))
    return p3.Mesh.join(parts, name="Well")


def stall():
    parts = []
    w, d = 3.0, 1.8
    parts.append(textured(p3.box((w, 0.9, 0.7)).translate((0, 0.45, d / 2 - 0.35)), T("planks"), "Wood"))  # counter
    for sx in (-1, 1):
        for z in (-d / 2 + 0.1, d / 2 - 0.1):
            parts.append(textured(p3.box((0.12, 2.3, 0.12)).translate((sx * (w / 2 - 0.1), 1.15, z)), T("planks_dark"), "Wood"))
    # striped canopy sloping to the front
    stripes = []
    n = 8
    for k in range(n):
        s = p3.box((w / n, 0.06, d + 0.5)).translate((-w / 2 + (k + 0.5) * w / n, 0, 0))
        s.set_material(flat("Canopy_Red", "#b8443a") if k % 2 == 0 else flat("Canopy_Cream", "#ecdcb6"))
        stripes.append(s)
    canopy = p3.Mesh.join(stripes, name="Canopy").rotate((14, 0, 0)).translate((0, 2.35, 0.1))
    parts.append(canopy)
    # produce: apples and cabbages in crates
    for k, (col, x) in enumerate((("#c8323a", -0.9), ("#7ab04a", 0.0), ("#e0a040", 0.9))):
        parts.append(textured(p3.box((0.7, 0.18, 0.5)).translate((x, 0.99, d / 2 - 0.35)), T("planks"), "Wood"))
        for j in range(6):
            fruit = p3.uv_sphere(0.075, 6, 4).translate((x - 0.22 + (j % 3) * 0.22, 1.12, d / 2 - 0.45 + (j // 3) * 0.2))
            parts.append(fruit.set_material(flat(f"Produce_{k}", col)))
    return p3.Mesh.join(parts, name="Stall")


def notice_board():
    parts = []
    for sx in (-1, 1):
        parts.append(textured(p3.box((0.14, 2.1, 0.14)).translate((sx * 0.7, 1.05, 0)), T("planks_dark"), "Wood"))
    parts.append(textured(p3.box((1.5, 1.0, 0.08)).translate((0, 1.4, 0)), T("planks"), "Wood"))
    for k in range(4):  # pinned papers
        parts.append(p3.box((0.3, 0.38, 0.02)).rotate((0, 0, RNG.uniform(-8, 8))).translate((-0.5 + k * 0.33, 1.45 + RNG.uniform(-0.15, 0.1), 0.05)).set_material(flat("Paper", "#efe6cc")))
    roof = gable_roof(1.7, 0.5, 0.3, overhang=0.1, thick=0.08, tex="tiles").translate((0, 2.05, 0))
    parts.append(roof)
    return p3.Mesh.join(parts, name="Notice")


def lamp():
    parts = [textured(p3.box((0.16, 2.6, 0.16)).translate((0, 1.3, 0)), T("planks_dark"), "Wood")]
    parts.append(p3.box((0.6, 0.08, 0.08)).translate((0.22, 2.5, 0)).set_material(IRON))
    parts.append(p3.box((0.22, 0.3, 0.22)).translate((0.45, 2.28, 0)).set_material(GLOW))
    parts.append(p3.cone(0.2, 0.16, 4).rotate((0, 45, 0)).translate((0.45, 2.5, 0)).set_material(IRON))
    parts.append(p3.box((0.26, 0.04, 0.26)).translate((0.45, 2.12, 0)).set_material(IRON))
    return p3.Mesh.join(parts, name="Lamp")


def crate(s=0.7):
    parts = [textured(p3.box((s, s, s)), T("planks"), "Wood")]
    for y in (-1, 1):
        for z in (-1, 1):
            parts.append(p3.box((s + 0.02, 0.07, 0.07)).translate((0, y * (s / 2 - 0.035), z * (s / 2 - 0.035))).set_material(BEAM))
    return p3.Mesh.join(parts, name="Crate")


def crates():
    a = crate().translate((-0.38, 0.35, 0))
    b = crate().rotate((0, 12, 0)).translate((0.4, 0.35, 0.05))
    c = crate(0.6).rotate((0, -8, 0)).translate((0.0, 1.0, 0.0))
    return p3.Mesh.join([a, b, c], name="Crates")


def barrel():
    prof = [(0, 0), (0.3, 0), (0.36, 0.25), (0.37, 0.4), (0.36, 0.55), (0.3, 0.8), (0, 0.8)]
    b = textured(p3.lathe(prof, 12), T("planks"), "Wood")
    hoops = [p3.torus(0.35, 0.02, 12, 4).translate((0, y, 0)).set_material(IRON) for y in (0.14, 0.66)]
    return p3.Mesh.join([b] + hoops, name="Barrel")


def barrels():
    return p3.Mesh.join([barrel().translate((-0.3, 0, 0)), barrel().translate((0.38, 0, 0.1))], name="Barrels")


def fence(length=3.0):
    parts = []
    for x in (-length / 2 + 0.1, 0.0, length / 2 - 0.1):
        parts.append(textured(p3.box((0.12, 1.05, 0.12)).translate((x, 0.52, 0)), T("planks_dark"), "Wood"))
    for y in (0.45, 0.85):
        parts.append(textured(p3.box((length, 0.1, 0.06)).translate((0, y, 0.05)), T("planks"), "Wood"))
    return p3.Mesh.join(parts, name="Fence")


def bench():
    parts = [textured(p3.box((1.6, 0.08, 0.42)).translate((0, 0.45, 0)), T("planks"), "Wood")]
    for sx in (-1, 1):
        parts.append(textured(p3.box((0.1, 0.45, 0.36)).translate((sx * 0.65, 0.225, 0)), T("planks_dark"), "Wood"))
    parts.append(textured(p3.box((1.6, 0.3, 0.06)).rotate((-10, 0, 0)).translate((0, 0.72, -0.2)), T("planks"), "Wood"))
    return p3.Mesh.join(parts, name="Bench")


BUILDERS = {
    "house_a": lambda: house(6.2, 5.2, 3.3, 2.3, name="House_A"),
    "house_b": lambda: house(7.6, 6.2, 3.0, 2.8, wall_tex="plaster", roof_tex="slate", upper=2.1, name="House_B"),
    "house_c": thatched_cottage,
    "well": well,
    "stall": stall,
    "notice": notice_board,
    "lamp": lamp,
    "crates": crates,
    "barrels": barrels,
    "fence": fence,
    "bench": bench,
}


def main():
    rv = REVIEW / "buildings"
    rv.mkdir(parents=True, exist_ok=True)
    stats = {}
    sheet = p3.Scene("Village")
    for i, (name, fn) in enumerate(BUILDERS.items()):
        m = fn()
        s = p3.Scene(name)
        s.add(m)
        export_glb(s, name, sub="buildings")
        stats[name] = m.num_triangles
        sheet.add(m.copy(), position=((i % 4) * 9 - 13, 0, (i // 4) * 9 - 9))
    p3.render(sheet, rv / "village_sheet.png", 1100, 760, view=(0.54, 0.64, 0.54), style="toon")
    print(stats)


if __name__ == "__main__":
    main()
