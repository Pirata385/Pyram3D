"""Hair styles (Hair slot) and Harpin head plumes (Plume slot), built on each body's head.

Human: curto, rabo (ponytail), longo, coque (top knot) — unisex.
Harpin: espetado (spiky + temple feather tufts, ref. 06), crista (crest of long feathers),
        plumado (long hair with feather strands).
"""
from __future__ import annotations

import numpy as np

from common import BodySpec, slot

import pyram3d as p3
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


def bangs(spec: BodySpec, n=4, length=0.05, spread=0.07, tilt=0.05):
    """Fringe of pointed locks ending above the brows."""
    parts = []
    for k in range(n):
        x = -spread + 2 * spread * k / max(1, n - 1)
        a = _p(spec, x, 1.73, 0.05)
        b = _p(spec, x * 1.2 + 0.012 * (1 if k % 2 else -1), 1.73 - min(length, 0.05) / spec.s, 0.1 + tilt * 0.2)
        parts.append(hair(sdf.round_cone(a, b, 0.028, 0.004), 0.95 if k % 2 else 1.0))
    return parts


def brows(spec: BodySpec):
    """Brows sit on the brow ridge, in front of any fringe (they must show on every style)."""
    a = _p(spec, 0.016, 1.664, 0.1)
    b = _p(spec, 0.054, 1.668, 0.09)
    return [hair(sdf.capsule(a, b, 0.0075), 0.75).mirror_x()]


def temple_tufts(spec: BodySpec, big=1.0):
    """Root of the temple feathers (the feathers themselves are cards: temple_feathers)."""
    return [plume(sdf.ellipsoid(_p(spec, 0.085, 1.68, -0.005), (0.026, 0.04, 0.04)), 0.95).mirror_x()]


def temple_feathers(spec: BodySpec, big=1.0):
    """3-4 lanceolate feathers per temple (0.18-0.25 m, ~4.5 cm wide, dark rachis), layered and
    sweeping back and up like small wings (ref. 06): they give the head its silhouette."""
    from harpin import feather
    cards = []
    for sgn in (1, -1):
        for k, (dy, ln, w, up) in enumerate(((0.0, 0.24, 0.05, 0.35), (0.028, 0.22, 0.046, 0.55), (-0.02, 0.19, 0.044, 0.15),
                                             (0.05, 0.18, 0.04, 0.75))):
            root = _p(spec, 0.088 * sgn, 1.675 + dy, 0.0)
            d = np.array([0.42 * sgn, up, -0.85])
            d /= np.linalg.norm(d)
            tip = root + d * ln * big
            nrm = np.array([sgn * 0.9, 0.3, 0.3])
            cards.append(feather(root, tip, w * big, nrm, cup=0.15, curl=0.06, asym=0.0, segs=5,
                                 tone=1.0 - 0.07 * (k % 2), tip_band=0.85, rachis_dark=True))
    m = p3.Mesh.join(cards, name="TempleFeathers")
    m.set_material(p3.Material("Plume_Feather", "#ecebe6", roughness=0.85, double_sided=True))
    return m


def style_sdf(spec: BodySpec, style: str):
    s = spec.s
    parts = []
    if style == "curto":
        # a thin cap with 6 big pointed locks over it (creases between them draw ink lines)
        parts = [cap(spec, 0.012)] + bangs(spec, 3, 0.05, 0.055)
        for k, (bx, bz, d) in enumerate(((0.0, 0.0, (0.0, 0.03, 0.1)), (0.05, -0.02, (0.06, 0.0, 0.07)), (-0.05, -0.02, (-0.06, 0.0, 0.07)),
                                         (0.03, -0.08, (0.05, -0.02, -0.06)), (-0.03, -0.08, (-0.05, -0.02, -0.06)), (0.0, -0.06, (0.0, 0.02, -0.08)))):
            base = _p(spec, bx, 1.74, bz)
            tip = base + np.array(d)
            parts.append(hair(sdf.round_cone(tuple(base), tuple(tip), 0.04, 0.008), 0.92 + 0.08 * (k % 2)))
    elif style == "rabo":
        parts = [cap(spec, 0.012, front=1.69)]
        pts = [_p(spec, 0, 1.7, -0.1), _p(spec, 0, 1.64, -0.15), _p(spec, 0, 1.5, -0.17), _p(spec, 0, 1.36, -0.15)]
        parts.append(hair(sdf.chain([tuple(p) for p in pts], [0.034, 0.04, 0.032, 0.012], k=0.02)))
        parts.append(sdf.torus(tuple(_p(spec, 0, 1.69, -0.115)), 0.026, 0.009).rotate((70, 0, 0), tuple(_p(spec, 0, 1.69, -0.115))).paint(color=(1, 1, 1), material=slot("Leather")))
    elif style == "longo":
        parts = [cap(spec, 0.016)] + bangs(spec, 3, 0.05, 0.06)
        # back: three wide locks ending in points at the shoulder blades
        for k, x in enumerate((-0.055, 0.0, 0.055)):
            a = _p(spec, x * 0.8, 1.64, -0.075)
            b = _p(spec, x * 1.5, 1.34 + 0.02 * (k % 2), -0.12)
            parts.append(hair(sdf.round_cone(tuple(a), tuple(b), 0.06, 0.012), 0.92 + 0.08 * (k % 2)))
        # two locks in front of the shoulders
        for sgn in (1, -1):
            parts.append(hair(sdf.round_cone(_p(spec, 0.085 * sgn, 1.64, 0.01), _p(spec, 0.115 * sgn, 1.38, 0.04), 0.032, 0.006), 0.95))
    elif style == "coque":
        parts = [cap(spec, 0.011, front=1.69)]
        parts.append(hair(sdf.sphere(tuple(_p(spec, 0, 1.775, -0.06)), 0.05)))
        parts.append(hair(sdf.torus(tuple(_p(spec, 0, 1.745, -0.05)), 0.03, 0.012), 0.85))
    elif style == "espetado":
        # big messy spikes (silhouette ~1.3x the head width, ref. 06)
        parts = [cap(spec, 0.026)]
        rng = np.random.default_rng(3)
        for k in range(14):  # spikes sweeping back, up and out
            a = rng.uniform(-1.0, 1.0)
            base = _p(spec, 0.08 * a, 1.74 - 0.04 * abs(a), 0.02 - 0.1 * rng.uniform(0, 1))
            tip = base + np.array([0.075 * a, rng.uniform(0.05, 0.09), -rng.uniform(0.08, 0.14)])
            parts.append(hair(sdf.round_cone(tuple(base), tuple(tip), 0.036, 0.005), 0.94 + 0.06 * (k % 2)))
        for k in range(4):  # fringe spikes over the forehead
            x = -0.05 + 0.033 * k
            parts.append(hair(sdf.round_cone(tuple(_p(spec, x, 1.74, 0.06)), tuple(_p(spec, x * 1.35, 1.69, 0.112)), 0.024, 0.004)))
        parts += temple_tufts(spec)
    elif style == "crista":
        parts = [cap(spec, 0.014, front=1.7)]
        for k in range(6):  # tall crest of feathers from the forehead to the nape, sweeping back
            t = k / 5
            base = _p(spec, 0, 1.745 + 0.015 * np.sin(np.pi * t), 0.06 - 0.16 * t)
            tip = base + np.array([0, 0.13 - 0.07 * t, -0.2 - 0.05 * t])
            parts.append(plume(sdf.round_cone(tuple(base), tuple(tip), 0.04 - 0.005 * t, 0.005), 1.0 - 0.05 * (k % 2)))
        parts += temple_tufts(spec, 0.8)
    elif style == "plumado":
        parts = [cap(spec, 0.024)] + bangs(spec, 3, 0.05, 0.06)
        parts.append(hair(sdf.round_cone(_p(spec, 0, 1.64, -0.075), _p(spec, 0, 1.33, -0.115), 0.112, 0.07)))
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
