"""PBR materials (glTF metallic-roughness model) and procedural textures.

Colors given by the user are sRGB ('#ff8800', (1, 0.5, 0) or 0-255 tuples);
they are converted to linear internally, as glTF requires.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import math3d as m3
from . import noise as nz
from .image import read_image, to_uint8


class Texture:
    """An RGBA image (H, W, 4 uint8) used by materials.

    `srgb` marks color data (base color, emissive). Data maps (normal,
    roughness/metal, occlusion) must be linear (srgb=False).
    """

    def __init__(self, image, name: str = "Texture", srgb: bool = True, wrap: str = "repeat",
                 filter: str = "linear"):
        img = to_uint8(image)
        if img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        if img.shape[2] == 3:
            img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=2)
        self.image = img
        self.name = name
        self.srgb = srgb
        self.wrap = wrap  # 'repeat' | 'clamp' | 'mirror'
        self.filter = filter  # 'linear' | 'nearest' (pixel art)

    def __repr__(self):
        return f"<Texture {self.name!r} {self.image.shape[1]}x{self.image.shape[0]}>"

    @classmethod
    def load(cls, path, srgb: bool = True, **kw) -> "Texture":
        from pathlib import Path
        return cls(read_image(path), name=Path(path).stem, srgb=srgb, **kw)

    @classmethod
    def from_function(cls, func: Callable[[np.ndarray, np.ndarray], np.ndarray], size: int = 512,
                      name: str = "Procedural", srgb: bool = True, **kw) -> "Texture":
        """func(u, v) with u, v arrays in [0, 1) -> (H, W, 3|4) or (H, W) values in [0, 1]."""
        t = (np.arange(size) + 0.5) / size
        u, v = np.meshgrid(t, t[::-1])  # row 0 = top of image = v near 1
        out = np.asarray(func(u, v), dtype=np.float64)
        if out.ndim == 2:
            out = np.repeat(out[:, :, None], 3, axis=2)
        return cls(np.clip(out, 0, 1), name=name, srgb=srgb, **kw)

    def sample(self, uv: np.ndarray) -> np.ndarray:
        """Bilinear sample (N, 2) uvs -> (N, 4) floats in [0, 1] (raw texel values)."""
        img = self.image.astype(np.float64) / 255.0
        h, w = img.shape[:2]
        u = uv[:, 0]
        v = 1.0 - uv[:, 1]
        if self.wrap == "clamp":
            u = np.clip(u, 0, 1)
            v = np.clip(v, 0, 1)
        x = u * w - 0.5
        y = v * h - 0.5
        if self.filter == "nearest":
            xi = np.floor(x + 0.5).astype(np.int64) % w
            yi = np.floor(y + 0.5).astype(np.int64) % h
            return img[yi, xi]
        x0 = np.floor(x).astype(np.int64)
        y0 = np.floor(y).astype(np.int64)
        fx = (x - x0)[:, None]
        fy = (y - y0)[:, None]
        x0m, x1m = x0 % w, (x0 + 1) % w
        y0m, y1m = y0 % h, (y0 + 1) % h
        a = img[y0m, x0m] * (1 - fx) + img[y0m, x1m] * fx
        b = img[y1m, x0m] * (1 - fx) + img[y1m, x1m] * fx
        return a * (1 - fy) + b * fy


@dataclass(eq=False)
class Material:
    """glTF PBR metallic-roughness material.

    base_color:  sRGB color (hex or tuple), multiplied with base_color_texture
    metallic:    0 = dielectric (plastic, wood, skin), 1 = metal
    roughness:   0 = mirror-like, 1 = fully diffuse
    emissive:    sRGB glow color; emissive_strength > 1 for HDR glow
    alpha:       < 1 makes the material transparent (alpha_mode 'BLEND')
    unlit:       flat, lighting-independent shading (stylized / toon)
    """
    name: str = "Material"
    base_color: object = "#cccccc"
    metallic: float = 0.0
    roughness: float = 0.5
    emissive: object = "#000000"
    emissive_strength: float = 1.0
    alpha: float = 1.0
    alpha_mode: str | None = None  # OPAQUE | BLEND | MASK (auto from alpha)
    alpha_cutoff: float = 0.5
    double_sided: bool = False
    unlit: bool = False
    base_color_texture: Texture | None = None
    normal_texture: Texture | None = None
    normal_scale: float = 1.0
    metallic_roughness_texture: Texture | None = None  # G = roughness, B = metallic
    occlusion_texture: Texture | None = None
    emissive_texture: Texture | None = None
    extras: dict = field(default_factory=dict)

    @property
    def base_color_linear(self) -> np.ndarray:
        c = m3.as_color(self.base_color)
        c[:3] = m3.srgb_to_linear(c[:3])
        c[3] = c[3] * self.alpha
        return c

    @property
    def emissive_linear(self) -> np.ndarray:
        c = m3.as_color(self.emissive)[:3]
        return m3.srgb_to_linear(c)

    @property
    def resolved_alpha_mode(self) -> str:
        if self.alpha_mode:
            return self.alpha_mode.upper()
        return "BLEND" if self.base_color_linear[3] < 0.999 else "OPAQUE"

    def textures(self) -> list[tuple[str, Texture]]:
        out = []
        for slot in ("base_color_texture", "normal_texture", "metallic_roughness_texture",
                     "occlusion_texture", "emissive_texture"):
            t = getattr(self, slot)
            if t is not None:
                out.append((slot, t))
        return out

    def copy(self, **changes) -> "Material":
        import dataclasses
        return dataclasses.replace(self, **changes)


# ---------------------------------------------------------------------------
# procedural texture generators (return Texture objects)
# ---------------------------------------------------------------------------

def _col(c) -> np.ndarray:
    return m3.as_color(c)[:3]


def _mix(a, b, t):
    t = np.asarray(t)[..., None]
    return _col(a) * (1 - t) + _col(b) * t


_TEX_CACHE: dict = {}


def _cached(fn):
    """Memoize texture generators (procedural textures are deterministic and immutable)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        try:
            key = (fn.__name__, args, tuple(sorted(kw.items())))
            hash(key)
        except TypeError:
            return fn(*args, **kw)
        if key not in _TEX_CACHE:
            _TEX_CACHE[key] = fn(*args, **kw)
        return _TEX_CACHE[key]
    return staticmethod(wrapper)


class textures:
    """Procedural texture library. All generators tile seamlessly where noted."""

    @_cached
    def checker(color1="#ffffff", color2="#222222", squares: int = 8, size: int = 512) -> Texture:
        def f(u, v):
            m = ((np.floor(u * squares) + np.floor(v * squares)) % 2)
            return _mix(color1, color2, m)
        return Texture.from_function(f, size, "Checker", filter="nearest")

    @_cached
    def noise(color1="#000000", color2="#ffffff", scale: float = 8.0, octaves: int = 5, seed: int = 0,
              size: int = 512, contrast: float = 1.0) -> Texture:
        """Tileable fBm noise."""
        def f(u, v):
            n = _tileable_fbm(u, v, scale, octaves, seed)
            t = np.clip(0.5 + 0.5 * n * contrast, 0, 1)
            return _mix(color1, color2, t)
        return Texture.from_function(f, size, "Noise")

    @_cached
    def wood(light="#b8875a", dark="#8c5d36", rings: float = 7.0, distortion: float = 0.8,
             seed: int = 0, size: int = 512) -> Texture:
        """Wood planks/grain running along V."""
        def f(u, v):
            n = _tileable_fbm(u, v, 3, 4, seed)
            grain = _tileable_fbm(u, v, 24, 2, seed + 3)
            x = u * rings + n * distortion
            t = 0.5 + 0.5 * np.sin(x * 2 * math.pi)
            t = np.clip(0.15 + 0.6 * t ** 2.5 + grain * 0.25, 0, 1)
            return _mix(light, dark, t)
        return Texture.from_function(f, size, "Wood")

    @_cached
    def marble(base="#f2f0eb", vein="#5a5a66", scale: float = 3.0, sharpness: float = 4.0,
               seed: int = 0, size: int = 512) -> Texture:
        def f(u, v):
            n = _tileable_fbm(u, v, scale, 6, seed)
            t = np.abs(np.sin((u + v + n * 1.5) * math.pi * 3))
            t = 1 - np.clip(t, 0, 1) ** (1.0 / sharpness)
            return _mix(base, vein, np.clip(t * 1.4, 0, 1))
        return Texture.from_function(f, size, "Marble")

    @_cached
    def bricks(brick="#9c4a35", mortar="#cfc6b8", rows: int = 8, cols: int = 4, mortar_width: float = 0.06,
               variation: float = 0.15, seed: int = 0, size: int = 512) -> Texture:
        def f(u, v):
            r = v * rows
            row = np.floor(r)
            c = u * cols + (row % 2) * 0.5
            col = np.floor(c)
            fu, fv = c - col, r - row
            mw = mortar_width
            is_m = (fu < mw / 2) | (fu > 1 - mw / 2) | (fv < mw) | (fv > 1 - mw)
            rnd = np.sin(row * 12.9898 + (col % cols) * 78.233) * 43758.5453
            rnd = rnd - np.floor(rnd)
            n = _tileable_fbm(u, v, 24.0, 3, seed)
            base = _col(brick) * (1 - variation + 2 * variation * rnd[..., None]) * (1 + 0.1 * n[..., None])
            return np.where(is_m[..., None], _col(mortar) * (1 + 0.05 * n[..., None]), base)
        return Texture.from_function(f, size, "Bricks")

    @_cached
    def tiles(color="#e8e8e8", grout="#8a8a8a", count: int = 4, grout_width: float = 0.04,
              size: int = 512) -> Texture:
        def f(u, v):
            fu = (u * count) % 1
            fv = (v * count) % 1
            g = (fu < grout_width) | (fu > 1 - grout_width) | (fv < grout_width) | (fv > 1 - grout_width)
            return _mix(color, grout, g.astype(float))
        return Texture.from_function(f, size, "Tiles")

    @_cached
    def gradient(color1="#000000", color2="#ffffff", direction: str = "v", size: int = 256) -> Texture:
        def f(u, v):
            return _mix(color1, color2, v if direction == "v" else u)
        return Texture.from_function(f, size, "Gradient")

    @_cached
    def cells(color1="#2a5d2a", color2="#9fd17a", scale: float = 8.0, seed: int = 0, size: int = 512) -> Texture:
        """Voronoi cells (scales, stones, leather, cracked ground)."""
        def f(u, v):
            p = np.stack([u * scale, v * scale, np.zeros_like(u)], axis=-1)
            f1, f2 = _tileable_worley(p, scale, seed)
            t = np.clip((f2 - f1) * 2.0, 0, 1)
            return _mix(color1, color2, t)
        return Texture.from_function(f, size, "Cells")

    @_cached
    def stripes(color1="#ffffff", color2="#cc2222", count: int = 8, direction: str = "u", softness: float = 0.02,
                size: int = 256) -> Texture:
        def f(u, v):
            x = (u if direction == "u" else v) * count
            t = m3.smoothstep(0.5 - softness, 0.5 + softness, x % 1.0)
            return _mix(color1, color2, t)
        return Texture.from_function(f, size, "Stripes")

    @_cached
    def normal_from_height(height: np.ndarray | Texture, strength: float = 2.0) -> Texture:
        """Convert a height map (or the luminance of a texture) to a tangent-space normal map."""
        if isinstance(height, Texture):
            key = ("nfh", id(height), strength)
            if key in _TEX_CACHE and _TEX_CACHE[key][0] is height:
                return _TEX_CACHE[key][1]
            h = height.image[:, :, :3].astype(np.float64).mean(axis=2) / 255.0
            n = textures._normal(h, strength)
            _TEX_CACHE[key] = (height, n)
            return n
        return textures._normal(np.asarray(height, dtype=np.float64), strength)

    @staticmethod
    def _normal(h: np.ndarray, strength: float) -> Texture:
        dx = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) * strength
        dy = (np.roll(h, 1, axis=0) - np.roll(h, -1, axis=0)) * strength
        n = np.stack([-dx, -dy, np.ones_like(h)], axis=-1)
        n /= np.linalg.norm(n, axis=-1, keepdims=True)
        return Texture(n * 0.5 + 0.5, name="Normal", srgb=False)

    @_cached
    def roughness_metal(roughness: np.ndarray | float, metallic: np.ndarray | float = 0.0, size: int = 256) -> Texture:
        r = np.broadcast_to(np.asarray(roughness, dtype=np.float64), (size, size)) if np.ndim(roughness) == 0 \
            else np.asarray(roughness, dtype=np.float64)
        mt = np.broadcast_to(np.asarray(metallic, dtype=np.float64), r.shape)
        img = np.stack([np.ones_like(r), r, mt], axis=-1)
        return Texture(img, name="MetalRough", srgb=False)


def _tileable_fbm(u, v, scale, octaves, seed):
    """Seamlessly tiling fBm (periodic Perlin lattice)."""
    return nz.tileable_fbm2(u, v, scale, octaves, seed=seed)


def _tileable_worley(p, period, seed):
    best1 = np.full(p.shape[:-1], 1e9)
    best2 = np.full(p.shape[:-1], 1e9)
    cell = np.floor(p[..., :2]).astype(np.int64)
    per = int(round(period))
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            cx = cell[..., 0] + dx
            cy = cell[..., 1] + dy
            wx, wy = cx % per, cy % per
            h = (wx * 73856093) ^ (wy * 19349663) ^ (seed * 83492791)
            h = (h ^ (h >> 13)) * 1274126177
            h = h ^ (h >> 16)
            fx = cx + 0.5 + ((h & 1023) / 1023.0 - 0.5) * 0.9
            fy = cy + 0.5 + (((h >> 10) & 1023) / 1023.0 - 0.5) * 0.9
            d = np.hypot(p[..., 0] - fx, p[..., 1] - fy)
            best2 = np.where(d < best1, best1, np.minimum(best2, d))
            best1 = np.minimum(best1, d)
    return best1, best2


# ---------------------------------------------------------------------------
# preset materials
# ---------------------------------------------------------------------------

class presets:
    """Ready-made materials. Every function accepts overrides as keyword args."""

    @staticmethod
    def plastic(color="#d03030", **kw):
        return Material(kw.pop("name", "Plastic"), color, 0.0, kw.pop("roughness", 0.35), **kw)

    @staticmethod
    def rubber(color="#1c1c1c", **kw):
        return Material(kw.pop("name", "Rubber"), color, 0.0, kw.pop("roughness", 0.9), **kw)

    @staticmethod
    def metal(color="#c0c0c8", roughness: float = 0.3, **kw):
        return Material(kw.pop("name", "Metal"), color, 1.0, roughness, **kw)

    @staticmethod
    def gold(**kw):
        return presets.metal("#ffcc55", 0.25, name=kw.pop("name", "Gold"), **kw)

    @staticmethod
    def copper(**kw):
        return presets.metal("#e0905a", 0.3, name=kw.pop("name", "Copper"), **kw)

    @staticmethod
    def chrome(**kw):
        return presets.metal("#f0f0f5", 0.05, name=kw.pop("name", "Chrome"), **kw)

    @staticmethod
    def glass(color="#cfe8ff", alpha: float = 0.25, **kw):
        return Material(kw.pop("name", "Glass"), color, 0.0, 0.05, alpha=alpha, double_sided=True, **kw)

    @staticmethod
    def emissive(color="#ffdd88", strength: float = 3.0, **kw):
        return Material(kw.pop("name", "Emissive"), color, 0.0, 0.5, emissive=color, emissive_strength=strength, **kw)

    @staticmethod
    def wood(**kw):
        tex = textures.wood(seed=kw.pop("seed", 0))
        return Material(kw.pop("name", "Wood"), "#ffffff", 0.0, kw.pop("roughness", 0.6), base_color_texture=tex, **kw)

    @staticmethod
    def marble(**kw):
        return Material(kw.pop("name", "Marble"), "#ffffff", 0.0, 0.2, base_color_texture=textures.marble(), **kw)

    @staticmethod
    def bricks(**kw):
        tex = textures.bricks()
        return Material(kw.pop("name", "Bricks"), "#ffffff", 0.0, 0.85, base_color_texture=tex,
                        normal_texture=textures.normal_from_height(tex, 3.0), **kw)

    @staticmethod
    def skin(color="#e8b896", **kw):
        return Material(kw.pop("name", "Skin"), color, 0.0, kw.pop("roughness", 0.55), **kw)

    @staticmethod
    def fabric(color="#3a5fa0", **kw):
        return Material(kw.pop("name", "Fabric"), color, 0.0, 0.95, **kw)

    @staticmethod
    def foliage(color="#4f8a35", **kw):
        return Material(kw.pop("name", "Foliage"), color, 0.0, 0.7, double_sided=True, **kw)

    @staticmethod
    def bark(**kw):
        tex = textures.wood("#6b4a32", "#3b281a", rings=24, distortion=2.0, seed=kw.pop("seed", 1))
        return Material(kw.pop("name", "Bark"), "#ffffff", 0.0, 0.9, base_color_texture=tex,
                        normal_texture=textures.normal_from_height(tex, 4.0), **kw)

    @staticmethod
    def stone(color="#8a8680", **kw):
        tex = textures.noise("#5e5a55", color, scale=6, seed=kw.pop("seed", 2))
        return Material(kw.pop("name", "Stone"), "#ffffff", 0.0, 0.9, base_color_texture=tex,
                        normal_texture=textures.normal_from_height(tex, 2.0), **kw)

    @staticmethod
    def water(**kw):
        return Material(kw.pop("name", "Water"), "#2a6f9a", 0.0, 0.05, alpha=0.7, **kw)

    @staticmethod
    def toon(color="#ff8844", **kw):
        return Material(kw.pop("name", "Toon"), color, 0.0, 1.0, unlit=True, **kw)

    @staticmethod
    def vertex_color(**kw):
        """White material so vertex colors show through unchanged."""
        return Material(kw.pop("name", "VertexColor"), "#ffffff", 0.0, kw.pop("roughness", 0.6), **kw)
