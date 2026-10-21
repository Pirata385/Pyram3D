"""Skin weights for the prototype 2 humans: smooth functions of the rest position.

Every vertex gets its weights from where it is, not from mesh connectivity: soft memberships
(arm, leg, head) and smoothstep blends along each limb at the joints (elbow +-3.5 cm, knee +-4 cm,
wrist, ankle...). The same function weights the body, the garments over it and anything else, so
cloth and skin deform together; there are no rigid parts and no hard seams at the joints.
"""
from __future__ import annotations

import numpy as np

import body as B
from common import (ANKLE_Y, FOREARM, H_BALL, H_HOCK, H_KNEE, H_TOE, HARPIN, HIP, KNEE_Y, SHOULDER, TAIL_BONES,
                    TAIL_POINTS, UPPER_ARM, arm_dir, mirror_side)


def ss(e0, e1, x):
    t = np.clip((np.asarray(x, float) - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _arm_coords(P, side):
    S = mirror_side(SHOULDER, side)
    u = arm_dir(side)
    q = P - S
    d = q @ u
    r = np.linalg.norm(q - d[:, None] * u, axis=1)
    return d, r


def weight_matrix(P: np.ndarray, names: list[str], head_sdf=None, skirt: bool = False,
                  hem: float = 0.765) -> np.ndarray:
    """(N, n_bones) weights for rest positions P. skirt=True weights a tunic skirt: below the
    hips each side follows its thigh and the middle is shared by both thighs, so the legs never
    swing through it and it never tears."""
    idx = {n: i for i, n in enumerate(names)}
    N = len(P)
    W = np.zeros((N, len(names)))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]

    # head / neck / spine chain
    if head_sdf is not None:
        hd = head_sdf.dist(P)
        mh = (1 - ss(0.002, 0.022, hd)) * ss(1.5, 1.56, y)
    else:
        mh = ss(1.56, 1.6, y)
    mn = ss(1.43, 1.5, y) * (1 - mh)
    rest = 1 - mh - mn
    chest = ss(1.17, 1.3, y)
    spine = ss(1.02, 1.14, y) - chest
    hips = 1 - ss(1.02, 1.14, y)
    core = {"Head": mh, "Neck": mn, "Chest": rest * chest, "Spine": rest * spine, "Hips": rest * hips}

    # arms (soft membership: along the arm past the shoulder joint and close to its axis)
    arm_parts = {}
    for side, pre in ((1, "Left"), (-1, "Right")):
        d, r = _arm_coords(P, side)
        on_side = ss(0.0, 0.05, x * side)
        ma = ss(-0.05, 0.05, d) * (1 - ss(0.085, 0.125, r)) * on_side
        msh = ss(0.04, 0.14, x * side) * ss(1.3, 1.41, y) * (1 - ma) * (1 - ss(0.12, 0.2, -d - 0.0))
        lower = ss(UPPER_ARM - 0.035, UPPER_ARM + 0.035, d)
        hand = ss(UPPER_ARM + FOREARM - 0.018, UPPER_ARM + FOREARM + 0.025, d)
        arm_parts[pre] = (ma, msh, {"UpperArm": 1 - lower, "LowerArm": lower - hand, "Hand": hand})

    # legs (below the hips, on their side of the crotch; never the hands hanging beside the thighs)
    arm_any = np.clip(sum(a[0] for a in arm_parts.values()), 0, 1)
    leg_parts = {}
    for side, pre in ((1, "Left"), (-1, "Right")):
        if skirt:
            down = ss(1.03, 0.93, y)  # the skirt flaps follow the thighs (they swing far in a run)
            sidew = ss(-0.07, 0.07, x * side)
            ml = down * sidew
        else:
            ml = ss(0.985, 0.875, y) * ss(-0.012, 0.022, x * side) * (1 - arm_any)
        if HARPIN:
            if skirt:
                chain = {"UpperLeg": np.ones(N)}
            else:
                chain = harpin_leg_chain(P, side)
            leg_parts[pre] = (ml, chain)
            continue
        knee = ss(KNEE_Y + 0.045, KNEE_Y - 0.04, y)
        ankle = ss(ANKLE_Y + 0.035, ANKLE_Y - 0.01, y) if not skirt else np.zeros(N)
        toes = ss(0.09, 0.135, z) * ss(0.075, 0.045, y) if not skirt else np.zeros(N)
        foot = ankle * (1 - toes)
        leg_parts[pre] = (ml, {"UpperLeg": 1 - knee, "LowerLeg": knee - ankle, "Foot": foot, "Toes": ankle * toes})

    arm_total = sum(a[0] + a[1] for a in arm_parts.values())
    leg_total = sum(lp[0] for lp in leg_parts.values())
    free = np.clip(1 - arm_total - leg_total, 0, 1)
    for n, w in core.items():
        W[:, idx[n]] += free * w
    for pre, (ma, msh, chain) in arm_parts.items():
        W[:, idx[pre + "Shoulder"]] += msh
        for n, w in chain.items():
            W[:, idx[pre + n]] += ma * w
    for pre, (ml, chain) in leg_parts.items():
        for n, w in chain.items():
            W[:, idx[pre + n]] += ml * w
    W = np.clip(W, 0, None)
    s = W.sum(axis=1, keepdims=True)
    W = np.where(s > 1e-9, W / np.maximum(s, 1e-9), 0)
    W[s[:, 0] <= 1e-9, idx["Hips"]] = 1.0
    return W


def _polyline_param(P, knots):
    """Per point: i + t of the nearest point on the polyline through knots (segment i, t in
    [0, 1]) (the first and last segments extend a little past their ends)."""
    best_d = np.full(len(P), np.inf)
    best_s = np.zeros(len(P))
    n = len(knots) - 1
    for i in range(n):
        a, b = knots[i], knots[i + 1]
        ab = b - a
        t = (P - a) @ ab / (ab @ ab)
        t = np.clip(t, -0.5 if i == 0 else 0.0, 1.5 if i == n - 1 else 1.0)
        d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        better = d < best_d
        best_d = np.where(better, d, best_d)
        best_s = np.where(better, i + t, best_s)
    return best_s


def harpin_leg_chain(P, side):
    """Digitigrade leg: hip -> knee -> hock -> ball -> toe tip; smooth blends at the knee and the
    hock (a few centimeters), a short one at the ball (the toes bend there; the hallux, behind
    the ball, stays on the tarsus)."""
    sx = np.array([side, 1.0, 1.0])
    knots = [np.array([HIP[0], 1.0, 0.0]) * sx, H_KNEE * sx, H_HOCK * sx, H_BALL * sx, H_TOE * sx]
    s = _polyline_param(P, knots)
    knee = ss(0.86, 1.12, s)
    hock = ss(1.9, 2.1, s)
    ball = ss(2.97, 3.08, s) * ss(-0.005, 0.012, (P[:, 2] - H_BALL[2]))
    return {"UpperLeg": 1 - knee, "LowerLeg": knee - hock, "Foot": hock * (1 - ball), "Toes": hock * ball}


def tail_weights(P, names):
    """Tail1..6 along the tail polyline (linear blends between the bones' middles), Hips at the
    root."""
    s = _polyline_param(P, list(TAIL_POINTS))
    s = np.clip(s, 0, len(TAIL_BONES))
    W = np.zeros((len(P), len(names)))
    idx = {n: i for i, n in enumerate(names)}
    c = np.arange(len(TAIL_BONES)) + 0.5
    hat = np.clip(1 - np.abs(s[:, None] - c[None]), 0, 1)
    hat[:, 0] = np.where(s < 0.5, 1.0, hat[:, 0])
    hat[:, -1] = np.where(s > c[-1], 1.0, hat[:, -1])
    hips = 1 - ss(0.0, 0.45, s)
    for k, b in enumerate(TAIL_BONES):
        W[:, idx[b]] = hat[:, k] * (1 - hips)
    W[:, idx["Hips"]] += hips
    return W


def skin_with(mesh, W):
    mesh.joints, mesh.weights = top4(W)
    return mesh


def top4(W):
    k = 4
    order = np.argsort(-W, axis=1)[:, :k]
    w = np.take_along_axis(W, order, axis=1)
    w[w < 0.004] = 0
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    j = np.where(w > 0, order, 0)
    return j.astype(np.int64), w


def skin_mesh(mesh, sk, head_sdf=None, skirt_mask=None, hem=0.765, rigid_bone: str | None = None,
              skirt_blend=None):
    """Fill mesh.joints/weights. skirt_mask: vertices weighted as a skirt (tunic below the hips).
    skirt_blend(y) -> share of the skirt weights (default: from the waist down to the hips).
    rigid_bone: every vertex follows that bone (eyes, brows, hair)."""
    names = [b.name for b in sk.bones]
    if rigid_bone is not None:
        mesh.joints = np.zeros((mesh.num_vertices, 4), np.int64)
        mesh.weights = np.zeros((mesh.num_vertices, 4))
        mesh.joints[:, 0] = names.index(rigid_bone)
        mesh.weights[:, 0] = 1
        return mesh
    W = weight_matrix(mesh.vertices, names, head_sdf)
    if skirt_mask is not None and np.any(skirt_mask):
        Ws = weight_matrix(mesh.vertices[skirt_mask], names, head_sdf, skirt=True, hem=hem)
        # blend into the skirt weights over the hip band so the waist stays continuous
        y = mesh.vertices[skirt_mask, 1]
        b = (ss(1.045, 0.975, y) if skirt_blend is None else skirt_blend(y))[:, None]
        W[skirt_mask] = W[skirt_mask] * (1 - b) + Ws * b
    mesh.joints, mesh.weights = top4(W)
    return mesh
