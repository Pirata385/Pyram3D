"""Body SDFs for the Niryk races (T-pose). Units: meters, face +Z, left = +X.

Every leaf carries a material slot (see common.slot) and a vertex color used as painted
luminance detail on top of the tint the player chooses.
"""
from __future__ import annotations

import numpy as np

from common import BodySpec, joints, slot

from pyram3d import sdf

SKIN = "Skin"


def _p(spec: BodySpec, x, y, z):
    """Position helper: y is given for the 1.75 m reference and scaled by the spec."""
    return (x, y * spec.s, z)


def skin(shape, shade=1.0):
    g = float(np.clip(shade, 0, 1))
    return shape.paint(color=(g, g, g), material=slot(SKIN))


def torso(spec: BodySpec, detail=True):
    """Torso SDF parts. detail=False leaves out pecs/abs: garments built on it hang as smooth
    cloth instead of hugging the sculpted muscles."""
    f = spec.gender == "f"
    s = spec.s
    hipr = 0.134 if not f else 0.172
    chest = 0.186 if not f else 0.138
    parts = [
        skin(sdf.ellipsoid(_p(spec, 0, 0.965, -0.008), (hipr * (0.95 if spec.harpin else 1), 0.1 * s, 0.098))),
        skin(sdf.ellipsoid(_p(spec, 0, 1.115, 0.004), (0.124 * spec.waist, 0.12 * s, 0.088 * (0.95 if f else 1)))),
        skin(sdf.ellipsoid(_p(spec, 0, 1.295, -0.004), (chest, 0.152 * s, 0.114 if not f else 0.096))),
        skin(sdf.ellipsoid(_p(spec, 0, 1.34, -0.04), (chest * 0.98, 0.11 * s, 0.075))),  # upper back
    ]
    if f:
        b = spec.bust
        parts.append(skin(sdf.sphere(_p(spec, 0.064, 1.3, 0.06), 0.058 * (0.7 + 0.3 * b))).mirror_x())
    elif detail:
        parts.append(skin(sdf.ellipsoid(_p(spec, 0.078, 1.335, 0.062), (0.08, 0.056, 0.048)), 0.97).mirror_x())
        parts.append(skin(sdf.ellipsoid(_p(spec, 0.0, 1.125, 0.07), (0.07, 0.09 * s, 0.03)), 0.98))  # abs
    # glutes
    parts.append(skin(sdf.ellipsoid(_p(spec, 0.068, 0.93, -0.06), (0.066 if not f else 0.082, 0.085 * s, 0.062))).mirror_x())
    # neck + trapezius
    # neck: 55-60% of the head width (a neck as wide as the head reads as a pillar)
    parts.append(skin(sdf.round_cone(_p(spec, 0, 1.41, -0.015), _p(spec, 0, 1.575, -0.006), 0.06 if not f else 0.046, 0.05 if not f else 0.042)))
    trap_r = 0.054 if not f else 0.038
    parts.append(skin(sdf.capsule(_p(spec, 0.035, 1.455, -0.025), _p(spec, spec.shoulder - 0.03, 1.425, -0.025), trap_r)).mirror_x())
    return parts


def head(spec: BodySpec):
    """Cranium, a jaw plane ending in a chin (it meets the neck at an angle, so the outline draws a
    jaw line), wedge nose, brow ridge and ears."""
    f = spec.gender == "f"
    hs = spec.s
    cy = 1.652
    parts = [
        skin(sdf.ellipsoid(_p(spec, 0, cy, -0.012), (0.087 * hs, 0.1 * hs, 0.099 * hs))),  # cranium
        skin(sdf.round_cone(_p(spec, 0, 1.618, 0.018), _p(spec, 0, 1.552 if f else 1.548, 0.058 if f else 0.06),
                            (0.074 if not f else 0.063) * hs, (0.03 if not f else 0.02) * hs)),  # jaw (f: ~85%)
        skin(sdf.sphere(_p(spec, 0, 1.55 if f else 1.546, 0.066 if f else 0.07), (0.026 if not f else 0.017) * hs)),  # chin
        skin(sdf.ellipsoid(_p(spec, 0.046, 1.598, 0.056), (0.03, 0.024, 0.026))).mirror_x(),  # cheekbones
        skin(sdf.round_cone(_p(spec, 0, 1.646 if f else 1.648, 0.09), _p(spec, 0, 1.605 if f else 1.598, 0.108 if f else 0.114),
                            0.009 if f else 0.011, 0.011 if f else 0.017)),  # nose (f: ~30% smaller)
        skin(sdf.ellipsoid(_p(spec, 0, 1.604 if f else 1.6, 0.1 if f else 0.104), (0.015, 0.009, 0.01) if f else (0.02, 0.011, 0.012))),  # nostrils
        skin(sdf.capsule(_p(spec, -0.044, 1.66, 0.083), _p(spec, 0.044, 1.66, 0.083), 0.014 if not f else 0.011)),  # brow
    ]
    if spec.harpin:  # long pointed ears
        parts.append(skin(sdf.round_cone(_p(spec, 0.08, 1.625, -0.01), _p(spec, 0.15, 1.69, -0.05), 0.022, 0.004)).mirror_x())
    else:
        parts.append(skin(sdf.ellipsoid(_p(spec, 0.086, 1.625, -0.006), (0.013, 0.03, 0.02))).mirror_x())
    return parts


def eye_sockets(spec: BodySpec):
    return skin(sdf.sphere(_p(spec, 0.036, 1.637, 0.093), 0.019)).mirror_x()  # no dark "mask" around the eyes


def arms(spec: BodySpec):
    J = joints(spec)
    f = spec.gender == "f"
    L = spec.limb
    ua0, ua1 = J["LeftUpperArm"]
    la0, la1 = J["LeftLowerArm"]
    h0, h1 = J["LeftHand"]
    hand = spec.hand
    parts = [
        skin(sdf.sphere(ua0 + np.array([0.008, 0.0, 0.0]), (0.066 if not f else 0.05) * L)),  # deltoid
        skin(sdf.round_cone(ua0, ua1, 0.054 * L, 0.041 * L)),
        skin(sdf.round_cone(la0, la1, 0.043 * L, 0.03 * L)),
        skin(sdf.ellipsoid(la0 + np.array([0.06, 0.004, 0.006]), (0.07 * L, 0.04 * L, 0.04 * L)), 0.99),
        # hand: palm, finger block slightly curled, thumb
        skin(sdf.ellipsoid(h0 + np.array([0.048, -0.004, 0.004]) * hand, (0.052 * hand, 0.02 * hand, 0.04 * hand))),
        skin(sdf.ellipsoid(h0 + np.array([0.108, -0.012, 0.004]) * hand, (0.05 * hand, 0.017 * hand, 0.037 * hand)), 0.97),
        skin(sdf.capsule(h0 + np.array([0.02, -0.008, 0.036]) * hand, h0 + np.array([0.068, -0.018, 0.07]) * hand, 0.015 * hand)),  # thumb
    ]
    if not f:
        parts.append(skin(sdf.ellipsoid(ua0 + np.array([0.13, 0.008, 0.012]), (0.09 * L, 0.044 * L, 0.044 * L)), 0.99))
    return [p.mirror_x() for p in parts]


def human_legs(spec: BodySpec):
    J = joints(spec)
    f = spec.gender == "f"
    L = spec.limb * (1.06 if f else 1.0)
    t0, t1 = J["LeftUpperLeg"]
    c0, c1 = J["LeftLowerLeg"]
    foot0, foot1 = J["LeftFoot"]
    toe1 = J["LeftToes"][1]
    parts = [
        skin(sdf.round_cone(t0, t1, 0.09 * L, 0.053 * L)),
        skin(sdf.ellipsoid((t0 + t1) / 2 + np.array([0, 0.02, 0.02]), (0.072 * L, 0.17 * spec.s, 0.07 * L)), 0.99),
        skin(sdf.sphere(t1 + np.array([0, 0, 0.012]), 0.05 * L)),
        skin(sdf.round_cone(c0, c1, 0.052 * L, 0.031 * L)),
        skin(sdf.ellipsoid(c0 + np.array([0, -0.13 * spec.s, -0.03]), (0.048 * L, 0.1 * spec.s, 0.048 * L)), 0.99),
        skin(sdf.round_cone(foot0 + np.array([0, -0.04, -0.012]), toe1 + np.array([0, 0.012, -0.02]), 0.04, 0.034)),
    ]
    return [p.mirror_x() for p in parts]


def body_sdf(spec: BodySpec):
    parts = torso(spec) + arms(spec)
    if spec.harpin:
        import harpin as H
        parts += H.legs(spec) + H.feet(spec) + [H.tail(spec), H.ruff(spec)]
    else:
        parts += human_legs(spec)
    # the head blends with itself softly but meets the neck tightly: the jaw keeps an edge
    shape = sdf.smooth_union(*parts, k=0.032).smooth_union(sdf.smooth_union(*head(spec), k=0.02), 0.01)
    shape = shape.smooth_subtract(eye_sockets(spec), 0.008)
    return shape
