"""Keyframe animation of node transforms and morph-target weights.

    anim = scene.animation("Walk")
    anim.key(node, "position", 0.0, (0, 0, 0))
    anim.key(node, "position", 1.0, (0, 0, 2))
    anim.key(node, "rotation", 0.5, (0, 90, 0))   # Euler degrees or quaternion
    anim.keys(node, "scale", {0: 1, 0.5: 1.2, 1: 1}, interpolation="SMOOTH")

Interpolation per channel: "LINEAR", "STEP" or "SMOOTH" (smooth ease through
keys; exported as densely sampled LINEAR keys for maximum compatibility).
Bones of a Skeleton are ordinary nodes, so the same API animates rigs.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from . import math3d as m3

_PATH_ALIASES = {"position": "translation", "location": "translation", "translation": "translation",
                 "rotation": "rotation", "scale": "scale", "weights": "weights"}


class Channel:
    def __init__(self, node, path: str, interpolation: str = "LINEAR"):
        self.node = node
        self.path = path
        self.interpolation = interpolation.upper()
        self.times: list[float] = []
        self.values: list[np.ndarray] = []

    def __repr__(self):
        return f"<Channel {self.node.name}.{self.path} {len(self.times)} keys {self.interpolation}>"

    def set_key(self, time: float, value: np.ndarray):
        time = float(time)
        for i, t in enumerate(self.times):
            if abs(t - time) < 1e-9:
                self.values[i] = value
                return
        self.times.append(time)
        self.values.append(value)
        order = np.argsort(self.times)
        self.times = [self.times[i] for i in order]
        self.values = [self.values[i] for i in order]
        if self.path == "rotation":  # keep quaternions in the same hemisphere
            for i in range(1, len(self.values)):
                if np.dot(self.values[i], self.values[i - 1]) < 0:
                    self.values[i] = -self.values[i]

    @property
    def start(self) -> float:
        return self.times[0] if self.times else 0.0

    @property
    def end(self) -> float:
        return self.times[-1] if self.times else 0.0

    def sample(self, t: float) -> np.ndarray:
        times = self.times
        vals = self.values
        if not times:
            raise ValueError("empty channel")
        if t <= times[0]:
            return vals[0].copy()
        if t >= times[-1]:
            return vals[-1].copy()
        i = int(np.searchsorted(times, t, side="right")) - 1
        t0, t1 = times[i], times[i + 1]
        u = (t - t0) / max(t1 - t0, 1e-12)
        a, b = vals[i], vals[i + 1]
        if self.interpolation == "STEP":
            return a.copy()
        if self.interpolation == "SMOOTH":
            pa = vals[i - 1] if i > 0 else a
            pb = vals[i + 2] if i + 2 < len(vals) else b
            # Catmull-Rom with uniform parametrization
            u2, u3 = u * u, u * u * u
            v = 0.5 * ((2 * a) + (-pa + b) * u + (2 * pa - 5 * a + 4 * b - pb) * u2 + (-pa + 3 * a - 3 * b + pb) * u3)
            if self.path == "rotation":
                return m3.quat_normalize(v)
            return v
        if self.path == "rotation":
            return m3.quat_slerp(a, b, u)
        return a + (b - a) * u

    def baked(self, fps: float) -> tuple[np.ndarray, np.ndarray]:
        """Sample at a fixed rate -> (times, values) suitable for LINEAR playback."""
        if self.interpolation in ("LINEAR", "STEP") or len(self.times) < 3:
            return np.array(self.times), np.array(self.values)
        n = max(int(math.ceil((self.end - self.start) * fps)), 1)
        ts = np.linspace(self.start, self.end, n + 1)
        ts = np.unique(np.concatenate([ts, self.times]))
        vs = np.array([self.sample(t) for t in ts])
        if self.path == "rotation":
            for i in range(1, len(vs)):
                if np.dot(vs[i], vs[i - 1]) < 0:
                    vs[i] = -vs[i]
        return ts, vs


class Animation:
    """An animation clip (glTF animation / FBX AnimStack)."""

    def __init__(self, name: str = "Action", fps: int = 30):
        self.name = name
        self.fps = fps
        self.channels: list[Channel] = []
        self.loop = True

    def __repr__(self):
        return f"<Animation {self.name!r} {self.duration:.2f}s {len(self.channels)} channels>"

    def channel(self, node, path: str, interpolation: str | None = None) -> Channel:
        p = _PATH_ALIASES.get(path)
        if p is None:
            raise ValueError(f"unknown animation path {path!r}; use position/rotation/scale/weights")
        for c in self.channels:
            if c.node is node and c.path == p:
                if interpolation:
                    c.interpolation = interpolation.upper()
                return c
        c = Channel(node, p, interpolation or "LINEAR")
        self.channels.append(c)
        return c

    def key(self, node, path: str, time: float, value=None, interpolation: str | None = None) -> "Animation":
        """Insert one keyframe. value=None keys the node's current value."""
        ch = self.channel(node, path, interpolation)
        if value is None:
            value = {"translation": node.translation, "rotation": node.rotation, "scale": node.scale,
                     "weights": node.morph_weights or []}[ch.path]
        ch.set_key(time, _convert(ch.path, value))
        return self

    def keys(self, node, path: str, keyframes: Mapping[float, object], interpolation: str | None = None) -> "Animation":
        """Insert several keyframes {time: value}."""
        for t, v in keyframes.items():
            self.key(node, path, float(t), v, interpolation)
        return self

    def key_pose(self, time: float, pose: Mapping, interpolation: str | None = None) -> "Animation":
        """Key many nodes at once: pose = {node: {"rotation": (x, y, z), "position": ...}}."""
        for node, props in pose.items():
            for path, v in props.items():
                self.key(node, path, time, v, interpolation)
        return self

    @property
    def duration(self) -> float:
        return max((c.end for c in self.channels), default=0.0)

    @property
    def start(self) -> float:
        return min((c.start for c in self.channels), default=0.0)

    def apply(self, t: float):
        """Pose the scene nodes at time t (modifies node transforms)."""
        for c in self.channels:
            if not c.times:
                continue
            v = c.sample(t)
            if c.path == "translation":
                c.node.translation = v
            elif c.path == "rotation":
                c.node.rotation = m3.quat_normalize(v)
            elif c.path == "scale":
                c.node.scale = v
            elif c.path == "weights":
                c.node.morph_weights = list(v)

    def snapshot(self):
        """Remember current transforms of animated nodes (restore with restore())."""
        return [(c.node, c.node.translation.copy(), c.node.rotation.copy(), c.node.scale.copy(),
                 None if c.node.morph_weights is None else list(c.node.morph_weights)) for c in self.channels]

    @staticmethod
    def restore(snap):
        for node, t, r, s, w in snap:
            node.translation, node.rotation, node.scale, node.morph_weights = t, r, s, w


def _convert(path: str, value) -> np.ndarray:
    if path == "rotation":
        return m3.as_quat(value)
    if path == "weights":
        return np.asarray(value, dtype=np.float64).reshape(-1)
    return m3.vec3(value)


# ---------------------------------------------------------------------------
# convenience animations for objects
# ---------------------------------------------------------------------------

def spin(anim: Animation, node, duration: float = 2.0, axis: str = "y", turns: float = 1.0) -> Animation:
    """Continuous rotation (looping turntable)."""
    i = "xyz".index(axis)
    steps = max(int(4 * abs(turns)), 4)
    for k in range(steps + 1):
        e = [0.0, 0.0, 0.0]
        e[i] = 360.0 * turns * k / steps
        anim.key(node, "rotation", duration * k / steps, m3.quat_mul(m3.quat_from_euler(e), node.rotation),
                 "LINEAR")
    return anim


def bounce(anim: Animation, node, height: float = 0.5, duration: float = 1.0, squash: float = 0.15) -> Animation:
    """Bouncing ball style hop with squash and stretch."""
    p = node.translation.copy()
    s = node.scale.copy()
    anim.keys(node, "position", {0: p, duration / 2: p + (0, height, 0), duration: p}, "SMOOTH")
    anim.keys(node, "scale", {0: s * (1 + squash, 1 - squash, 1 + squash), duration * 0.15: s,
                              duration / 2: s * (1 - squash / 2, 1 + squash / 2, 1 - squash / 2),
                              duration * 0.85: s, duration: s * (1 + squash, 1 - squash, 1 + squash)}, "SMOOTH")
    return anim


def orbit(anim: Animation, node, center=(0, 0, 0), radius: float = 5.0, height: float = 2.0,
          duration: float = 4.0, look_at=True, steps: int = 32) -> Animation:
    """Move a node (e.g. a camera) around a circle, optionally looking at the center."""
    c = m3.vec3(center)
    for k in range(steps + 1):
        a = 2 * math.pi * k / steps
        pos = c + np.array([math.sin(a) * radius, height, math.cos(a) * radius])
        anim.key(node, "position", duration * k / steps, pos)
        if look_at:
            anim.key(node, "rotation", duration * k / steps, m3.quat_look_rotation(c - pos))
    return anim
