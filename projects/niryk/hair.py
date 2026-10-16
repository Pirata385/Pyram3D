"""Hair styles (Hair slot) and Harpin head plumes (Plume slot), built on each body's head.

Human: curto, rabo (ponytail), longo, coque (top knot) — unisex.
Harpin: espetado (spiky + temple feather tufts, ref. 06), crista (crest of long feathers),
        plumado (long hair with feather strands).
"""
from __future__ import annotations

import numpy as np

from common import BodySpec, slot

from pyram3d import sdf
from pyram3d.noise import fbm3

HUMAN_STYLES = ["curto", "rabo", "longo", "coque"]
HARPIN_STYLES = ["espetado", "crista", "plumado"]


def hair(shape, shade=1.0):
    g = float(shade)
    return shape.paint(color=(g, g, g), material=slot("Hair"))


def plume(shape, shade=1.0):
    g = float(shade)
    return shape.paint(color=(g, g, g), material=slot("Plume"))


def _p(spec, x, y, z):
    return np.array([x, y * spec.s, z])


def _strands(scale=60.0, seed=0):
    """Groove displacement: strand clumps running front-to-back."""
    def f(p):
        n = fbm3(p * np.array([scale, scale * 0.35, scale * 0.35]), octaves=2, seed=seed)
        return -np.clip(n * 1.4 + 0.2, -0.5, 1.0)
    return dict(amount=0.006, func=f)


def cap(spec: BodySpec, thick=0.016, front=1.672, nape=1.55):
    """Hair cap over the cranium, cut by a tilted hairline plane (high at the forehead,
    low at the nape) and kept clear of the face and ears."""
    s = spec.s
    shell = hair(sdf.ellipsoid(_p(spec, 0, 1.655, -0.014), (0.087 * s + thick, 0.1 * s + thick, 0.099 * s + thick)))
    p0 = _p(spec, 0, front, 0.08)
    p1 = _p(spec, 0, nape, -0.11)
    d = p1 - p0
    n_keep = np.array([0.0, -d[2], d[1]])  # perpendicular to the hairline in the YZ plane
    n_keep = n_keep / np.linalg.norm(n_keep)
    if n_keep[1] < 0:
        n_keep = -n_keep
    shell = shell.intersect(hair(sdf.plane(p0, -n_keep), 0.9))
    # sideburn notch: keep the ears free
    ears = hair(sdf.ellipsoid(_p(spec, 0.088, 1.615, -0.006), (0.03, 0.045, 0.04)), 0.9).mirror_x()
    return shell.subtract(ears)


def bangs(spec: BodySpec, n=4, length=0.06, spread=0.07, tilt=0.05):
    parts = []
    for k in range(n):
        x = -spread + 2 * spread * k / max(1, n - 1)
        a = _p(spec, x, 1.725, 0.05)
        b = _p(spec, x * 1.15 + 0.01, 1.725 - length / spec.s, 0.098 + tilt * 0.2)
        parts.append(hair(sdf.round_cone(a, b, 0.028, 0.009), 0.95 if k % 2 else 1.0))
    return parts


def brows(spec: BodySpec):
    a = _p(spec, 0.018, 1.662, 0.094)
    b = _p(spec, 0.052, 1.664, 0.086)
    return [hair(sdf.capsule(a, b, 0.0055), 0.8).mirror_x()]


def temple_tufts(spec: BodySpec, big=1.0):
    """Feather tufts at the temples sweeping back like small wings (ref. 06)."""
    parts = []
    for k, (dy, ln, w) in enumerate(((0.0, 0.15, 0.024), (0.026, 0.17, 0.022), (0.052, 0.13, 0.018))):
        a = _p(spec, 0.08, 1.665 + dy, 0.01)
        b = a + np.array([0.075, 0.035 + dy * 0.9, -0.13]) * (ln / 0.15) * big
        parts.append(plume(sdf.round_cone(a, b, w, 0.003), 1.0 - 0.06 * k).mirror_x())
    return parts


def style_sdf(spec: BodySpec, style: str):
    s = spec.s
    parts = []
    if style == "curto":
        parts = [cap(spec, 0.018)] + bangs(spec, 4, 0.055)
        crown = hair(sdf.ellipsoid(_p(spec, 0, 1.745, -0.02), (0.075, 0.03, 0.08)))
        parts.append(crown)
    elif style == "rabo":
        parts = [cap(spec, 0.012, front=1.69)]
        pts = [_p(spec, 0, 1.7, -0.1), _p(spec, 0, 1.64, -0.15), _p(spec, 0, 1.5, -0.17), _p(spec, 0, 1.36, -0.15)]
        parts.append(hair(sdf.chain([tuple(p) for p in pts], [0.034, 0.04, 0.032, 0.012], k=0.02)))
        parts.append(sdf.torus(tuple(_p(spec, 0, 1.69, -0.115)), 0.026, 0.009).rotate((70, 0, 0), tuple(_p(spec, 0, 1.69, -0.115))).paint(color=(1, 1, 1), material=slot("Leather")))
    elif style == "longo":
        parts = [cap(spec, 0.016)] + bangs(spec, 3, 0.05, 0.06)
        parts.append(hair(sdf.round_cone(_p(spec, 0, 1.63, -0.07), _p(spec, 0, 1.33, -0.115), 0.1, 0.075)))
        for sgn in (1, -1):
            parts.append(hair(sdf.round_cone(_p(spec, 0.085 * sgn, 1.64, 0.01), _p(spec, 0.11 * sgn, 1.4, 0.035), 0.028, 0.016), 0.95))
    elif style == "coque":
        parts = [cap(spec, 0.011, front=1.69)]
        parts.append(hair(sdf.sphere(tuple(_p(spec, 0, 1.775, -0.06)), 0.05)))
        parts.append(hair(sdf.torus(tuple(_p(spec, 0, 1.745, -0.05)), 0.03, 0.012), 0.85))
    elif style == "espetado":
        parts = [cap(spec, 0.016)]
        rng = np.random.default_rng(3)
        for k in range(13):  # spikes sweeping back and up
            a = rng.uniform(-1.0, 1.0)
            base = _p(spec, 0.07 * a, 1.73 - 0.04 * abs(a), 0.02 - 0.09 * rng.uniform(0, 1))
            tip = base + np.array([0.05 * a, rng.uniform(0.03, 0.07), -rng.uniform(0.07, 0.13)])
            parts.append(hair(sdf.round_cone(tuple(base), tuple(tip), 0.026, 0.004), 0.94 + 0.06 * (k % 2)))
        for k in range(4):  # fringe spikes over the forehead
            x = -0.05 + 0.033 * k
            parts.append(hair(sdf.round_cone(tuple(_p(spec, x, 1.73, 0.06)), tuple(_p(spec, x * 1.3, 1.67, 0.105)), 0.02, 0.004)))
        parts += temple_tufts(spec)
    elif style == "crista":
        parts = [cap(spec, 0.009, front=1.7)]
        for k in range(6):  # crest feathers from the forehead to the nape, sweeping back
            t = k / 5
            base = _p(spec, 0, 1.74 + 0.015 * np.sin(np.pi * t), 0.06 - 0.16 * t)
            tip = base + np.array([0, 0.09 - 0.06 * t, -0.16 - 0.04 * t])
            parts.append(plume(sdf.round_cone(tuple(base), tuple(tip), 0.03 - 0.004 * t, 0.004), 1.0 - 0.05 * (k % 2)))
        parts += temple_tufts(spec, 0.8)
    elif style == "plumado":
        parts = [cap(spec, 0.016)] + bangs(spec, 3, 0.06, 0.06)
        parts.append(hair(sdf.round_cone(_p(spec, 0, 1.63, -0.07), _p(spec, 0, 1.35, -0.11), 0.095, 0.06)))
        for sgn in (1, -1):
            a = _p(spec, 0.06 * sgn, 1.62, -0.1)
            parts.append(plume(sdf.round_cone(tuple(a), tuple(a + np.array([0.03 * sgn, -0.3, -0.04])), 0.022, 0.004), 0.95))
        parts += temple_tufts(spec, 1.1)
    else:
        raise ValueError(style)
    parts += brows(spec)
    return sdf.smooth_union(*parts, k=0.012).displace(**_strands(60, hash(style) % 97))


def styles_for(spec: BodySpec):
    return HARPIN_STYLES if spec.harpin else HUMAN_STYLES
