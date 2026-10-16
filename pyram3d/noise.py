"""Vectorized procedural noise (Perlin gradient noise, fBm, ridged, Worley)."""
from __future__ import annotations

import numpy as np

_GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
    [1, 1, 0], [-1, 1, 0], [0, -1, 1], [0, -1, -1],
], dtype=np.float64)

_PERM_CACHE: dict[int, np.ndarray] = {}


def _perm(seed: int) -> np.ndarray:
    p = _PERM_CACHE.get(seed)
    if p is None:
        rng = np.random.default_rng(seed)
        base = rng.permutation(256)
        p = np.concatenate([base, base]).astype(np.int64)
        _PERM_CACHE[seed] = p
    return p


def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def perlin3(p, seed: int = 0, period=None) -> np.ndarray:
    """Classic Perlin noise for points p (..., 3). Output roughly in [-1, 1].

    period: optional (px, py, pz) integer lattice periods -> seamlessly tiling noise.
    """
    p = np.asarray(p, dtype=np.float64)
    shape = p.shape[:-1]
    p = p.reshape(-1, 3)
    perm = _perm(seed)
    pi = np.floor(p).astype(np.int64)
    pf = p - pi
    if period is not None:
        per = np.maximum(np.asarray(period, dtype=np.int64), 1)
        x0, y0, z0 = pi[:, 0] % per[0], pi[:, 1] % per[1], pi[:, 2] % per[2]
        x1, y1, z1 = (x0 + 1) % per[0], (y0 + 1) % per[1], (z0 + 1) % per[2]
        x0, y0, z0, x1, y1, z1 = (a & 255 for a in (x0, y0, z0, x1, y1, z1))
    else:
        pi &= 255
        x0, y0, z0 = pi[:, 0], pi[:, 1], pi[:, 2]
        x1, y1, z1 = (x0 + 1) & 255, (y0 + 1) & 255, (z0 + 1) & 255
    u = _fade(pf)

    def g(ix, iy, iz, fx, fy, fz):
        h = perm[perm[perm[ix] + iy] + iz] & 15
        gr = _GRAD3[h]
        return gr[:, 0] * fx + gr[:, 1] * fy + gr[:, 2] * fz

    fx, fy, fz = pf[:, 0], pf[:, 1], pf[:, 2]
    n000 = g(x0, y0, z0, fx, fy, fz)
    n100 = g(x1, y0, z0, fx - 1, fy, fz)
    n010 = g(x0, y1, z0, fx, fy - 1, fz)
    n110 = g(x1, y1, z0, fx - 1, fy - 1, fz)
    n001 = g(x0, y0, z1, fx, fy, fz - 1)
    n101 = g(x1, y0, z1, fx - 1, fy, fz - 1)
    n011 = g(x0, y1, z1, fx, fy - 1, fz - 1)
    n111 = g(x1, y1, z1, fx - 1, fy - 1, fz - 1)
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    nx00 = n000 + ux * (n100 - n000)
    nx10 = n010 + ux * (n110 - n010)
    nx01 = n001 + ux * (n101 - n001)
    nx11 = n011 + ux * (n111 - n011)
    nxy0 = nx00 + uy * (nx10 - nx00)
    nxy1 = nx01 + uy * (nx11 - nx01)
    return (nxy0 + uz * (nxy1 - nxy0)).reshape(shape)


def perlin2(x, y, seed: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    p = np.stack([x, y, np.full_like(x, 0.5)], axis=-1)
    return perlin3(p, seed)


def fbm3(p, octaves: int = 5, lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """Fractal Brownian motion: sum of octaves of Perlin noise. Output roughly in [-1, 1]."""
    p = np.asarray(p, dtype=np.float64)
    total = np.zeros(p.shape[:-1])
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        total += amp * perlin3(p * freq + o * 17.31, seed + o)
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm * 1.6


def tileable_fbm2(u, v, scale: int = 8, octaves: int = 5, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """fBm over the unit square that tiles seamlessly (u, v in [0, 1))."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = max(int(round(scale)), 1)
    total = np.zeros(np.broadcast(u, v).shape)
    amp, norm = 1.0, 0.0
    for o in range(octaves):
        f = s * (2 ** o)
        p = np.stack(np.broadcast_arrays(u * f, v * f, np.full_like(u, 0.5 + o * 3.7)), axis=-1)
        total += amp * perlin3(p, seed + o, period=(f, f, 1 << 20))
        norm += amp
        amp *= gain
    return total / norm * 1.6


def ridged3(p, octaves: int = 5, lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    """Ridged multifractal noise in [0, 1]: sharp crests (mountain ridges, veins)."""
    p = np.asarray(p, dtype=np.float64)
    total = np.zeros(p.shape[:-1])
    amp, freq, norm = 1.0, 1.0, 0.0
    for o in range(octaves):
        n = 1.0 - np.abs(perlin3(p * freq + o * 7.13, seed + o) * 1.4)
        total += amp * np.clip(n, 0, 1) ** 2
        norm += amp
        amp *= gain
        freq *= lacunarity
    return total / norm


def fbm2(x, y, octaves: int = 5, lacunarity: float = 2.0, gain: float = 0.5, seed: int = 0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    return fbm3(np.stack([x, y, np.full_like(x, 0.37)], axis=-1), octaves, lacunarity, gain, seed)


def _hash3(ix, iy, iz, seed):
    h = (ix * 73856093) ^ (iy * 19349663) ^ (iz * 83492791) ^ (seed * 2654435761)
    h = (h ^ (h >> 13)) * 1274126177
    h = h ^ (h >> 16)
    return h


def worley3(p, seed: int = 0, jitter: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cellular (Worley) noise. Returns (F1 distance, F2 distance) for points (..., 3)."""
    p = np.asarray(p, dtype=np.float64)
    shape = p.shape[:-1]
    p = p.reshape(-1, 3)
    cell = np.floor(p).astype(np.int64)
    f1 = np.full(len(p), 1e9)
    f2 = np.full(len(p), 1e9)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                c = cell + np.array([dx, dy, dz])
                h = _hash3(c[:, 0], c[:, 1], c[:, 2], seed)
                fp = np.stack([(h & 1023) / 1023.0, ((h >> 10) & 1023) / 1023.0,
                               ((h >> 20) & 1023) / 1023.0], axis=1)
                feature = c + 0.5 + (fp - 0.5) * jitter
                d = np.linalg.norm(p - feature, axis=1)
                f2 = np.where(d < f1, f1, np.minimum(f2, d))
                f1 = np.minimum(f1, d)
    return f1.reshape(shape), f2.reshape(shape)


def random_unit_vectors(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)
