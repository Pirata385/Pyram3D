"""Vector, quaternion and matrix helpers.

Conventions (same as glTF):
  * Right-handed, +Y up, +Z towards the viewer ("front"), units in meters.
  * Quaternions are stored as (x, y, z, w).
  * Matrices are 4x4 numpy arrays using the column-vector convention
    (p' = M @ p), translation lives in M[:3, 3].
  * Euler angles are in DEGREES, order "XYZ" = rotate about X first, then Y,
    then Z (extrinsic), i.e. R = Rz @ Ry @ Rx. This matches FBX and Blender.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12


def vec3(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    if a.size == 1:
        return np.repeat(a, 3)
    if a.size != 3:
        raise ValueError(f"expected 3 components, got {a.size}")
    return a.copy()


def normalize(v, axis=-1) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(n, EPS)


# --------------------------------------------------------------------------
# quaternions (x, y, z, w)
# --------------------------------------------------------------------------

def quat_identity() -> np.ndarray:
    return np.array([0.0, 0.0, 0.0, 1.0])


def quat_from_axis_angle(axis, angle_deg: float) -> np.ndarray:
    axis = normalize(vec3(axis))
    h = math.radians(angle_deg) * 0.5
    s = math.sin(h)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, math.cos(h)])


def quat_mul(a, b) -> np.ndarray:
    """Hamilton product a*b (apply b first, then a)."""
    ax, ay, az, aw = np.moveaxis(np.asarray(a, dtype=np.float64), -1, 0)
    bx, by, bz, bw = np.moveaxis(np.asarray(b, dtype=np.float64), -1, 0)
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], axis=-1)


def quat_conj(q) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


def quat_normalize(q) -> np.ndarray:
    return normalize(q)


def quat_from_euler(euler_deg, order: str = "XYZ") -> np.ndarray:
    """Euler angles in degrees -> quaternion. order 'XYZ' means X applied first."""
    e = vec3(euler_deg)
    axes = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}
    q = quat_identity()
    for ch in order.upper():
        idx = "XYZ".index(ch)
        q = quat_mul(quat_from_axis_angle(axes[ch], e[idx]), q)
    return quat_normalize(q)


def quat_to_matrix3(q) -> np.ndarray:
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def matrix3_to_quat(m) -> np.ndarray:
    m = np.asarray(m, dtype=np.float64)[:3, :3]
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = quat_normalize(np.array([x, y, z, w]))
    if q[3] < 0:
        q = -q
    return q


def quat_to_euler(q, order: str = "XYZ") -> np.ndarray:
    """Quaternion -> Euler degrees for order XYZ (R = Rz Ry Rx)."""
    if order.upper() != "XYZ":
        raise NotImplementedError("only XYZ order is supported")
    m = quat_to_matrix3(q)
    # R = Rz Ry Rx -> m[2,0] = -sin(y)
    sy = -m[2, 0]
    sy = max(-1.0, min(1.0, sy))
    y = math.asin(sy)
    if abs(sy) < 0.9999999:
        x = math.atan2(m[2, 1], m[2, 2])
        z = math.atan2(m[1, 0], m[0, 0])
    else:  # gimbal lock
        x = math.atan2(-m[1, 2], m[1, 1])
        z = 0.0
    return np.degrees([x, y, z])


def quat_rotate(q, v) -> np.ndarray:
    """Rotate vector(s) v (...,3) by quaternion q."""
    return np.asarray(v, dtype=np.float64) @ quat_to_matrix3(q).T


def quat_slerp(a, b, t: float) -> np.ndarray:
    a = quat_normalize(a)
    b = quat_normalize(b)
    d = float(np.dot(a, b))
    if d < 0:
        b = -b
        d = -d
    if d > 0.9995:
        return quat_normalize(a + (b - a) * t)
    th = math.acos(d)
    s = math.sin(th)
    return (math.sin((1 - t) * th) / s) * a + (math.sin(t * th) / s) * b


def quat_look_rotation(forward, up=(0, 1, 0)) -> np.ndarray:
    """Rotation that makes the local -Z axis point along `forward` (glTF camera/light convention)."""
    f = normalize(vec3(forward))
    u = vec3(up)
    if abs(np.dot(f, normalize(u))) > 0.999:
        u = np.array([0.0, 0.0, 1.0]) if abs(f[1]) > 0.9 else np.array([0.0, 1.0, 0.0])
    z = -f
    x = normalize(np.cross(u, z))
    y = np.cross(z, x)
    return matrix3_to_quat(np.stack([x, y, z], axis=1))


def quat_between(a, b) -> np.ndarray:
    """Shortest rotation taking direction a to direction b."""
    a = normalize(vec3(a))
    b = normalize(vec3(b))
    d = float(np.dot(a, b))
    if d > 0.999999:
        return quat_identity()
    if d < -0.999999:
        axis = np.cross([1.0, 0.0, 0.0], a)
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross([0.0, 1.0, 0.0], a)
        return quat_from_axis_angle(axis, 180.0)
    c = np.cross(a, b)
    q = np.array([c[0], c[1], c[2], 1.0 + d])
    return quat_normalize(q)


def as_quat(rotation) -> np.ndarray:
    """Accept a quaternion (4 values) or Euler degrees (3 values)."""
    if rotation is None:
        return quat_identity()
    r = np.asarray(rotation, dtype=np.float64).reshape(-1)
    if r.size == 4:
        return quat_normalize(r)
    if r.size == 3:
        return quat_from_euler(r)
    raise ValueError("rotation must be Euler degrees (3) or quaternion xyzw (4)")


# --------------------------------------------------------------------------
# matrices
# --------------------------------------------------------------------------

def translation_matrix(t) -> np.ndarray:
    m = np.eye(4)
    m[:3, 3] = vec3(t)
    return m


def scale_matrix(s) -> np.ndarray:
    m = np.eye(4)
    m[0, 0], m[1, 1], m[2, 2] = vec3(s)
    return m


def rotation_matrix(rotation) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix3(as_quat(rotation))
    return m


def trs_matrix(t=(0, 0, 0), r=None, s=(1, 1, 1)) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_to_matrix3(as_quat(r)) * vec3(s)[None, :]
    m[:3, 3] = vec3(t)
    return m


def decompose(m) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    m = np.asarray(m, dtype=np.float64)
    t = m[:3, 3].copy()
    sx = np.linalg.norm(m[:3, 0])
    sy = np.linalg.norm(m[:3, 1])
    sz = np.linalg.norm(m[:3, 2])
    if np.linalg.det(m[:3, :3]) < 0:
        sx = -sx
    rot = m[:3, :3] / np.array([sx, sy, sz])[None, :]
    return t, matrix3_to_quat(rot), np.array([sx, sy, sz])


def transform_points(m, pts) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ m[:3, :3].T + m[:3, 3]


def transform_dirs(m, dirs) -> np.ndarray:
    return np.asarray(dirs, dtype=np.float64) @ m[:3, :3].T


def look_at_matrix(eye, target, up=(0, 1, 0)) -> np.ndarray:
    """Camera-to-world matrix: camera at `eye` looking at `target` (-Z forward)."""
    eye = vec3(eye)
    q = quat_look_rotation(vec3(target) - eye, up)
    return trs_matrix(eye, q)


def perspective_matrix(yfov_deg: float, aspect: float, znear: float, zfar: float) -> np.ndarray:
    f = 1.0 / math.tan(math.radians(yfov_deg) / 2)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (zfar + znear) / (znear - zfar)
    m[2, 3] = 2 * zfar * znear / (znear - zfar)
    m[3, 2] = -1
    return m


def orthographic_matrix(xmag: float, ymag: float, znear: float, zfar: float) -> np.ndarray:
    m = np.eye(4)
    m[0, 0] = 1.0 / xmag
    m[1, 1] = 1.0 / ymag
    m[2, 2] = 2.0 / (znear - zfar)
    m[2, 3] = (zfar + znear) / (znear - zfar)
    return m


def lerp(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, dtype=np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def bezier(points: Sequence[Sequence[float]], n: int = 32) -> np.ndarray:
    """Sample a Bezier curve of any degree. Returns (n, dim) points."""
    p = np.asarray(points, dtype=np.float64)
    t = np.linspace(0, 1, n)[:, None]
    k = len(p) - 1
    out = np.zeros((n, p.shape[1]))
    for i in range(k + 1):
        coef = math.comb(k, i) * (t ** i) * ((1 - t) ** (k - i))
        out += coef * p[i]
    return out


def catmull_rom(points: Sequence[Sequence[float]], samples_per_segment: int = 8, closed: bool = False) -> np.ndarray:
    """Smooth curve passing through all control points."""
    p = np.asarray(points, dtype=np.float64)
    if len(p) < 2:
        return p.copy()
    if closed:
        pts = np.concatenate([p[-1:], p, p[:2]])
    else:
        pts = np.concatenate([2 * p[:1] - p[1:2], p, 2 * p[-1:] - p[-2:-1]])
    out = []
    nseg = len(p) if closed else len(p) - 1
    for i in range(nseg):
        p0, p1, p2, p3 = pts[i], pts[i + 1], pts[i + 2], pts[i + 3]
        last = (i == nseg - 1) and not closed
        ts = np.linspace(0, 1, samples_per_segment + 1)
        if not last:
            ts = ts[:-1]
        for t in ts:
            t2, t3 = t * t, t * t * t
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
    return np.array(out)


def frames_along(path: np.ndarray, up_hint=(0, 1, 0)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parallel-transport frames (tangent, normal, binormal) along a polyline."""
    path = np.asarray(path, dtype=np.float64)
    n = len(path)
    tang = np.zeros_like(path)
    tang[1:-1] = path[2:] - path[:-2]
    tang[0] = path[1] - path[0]
    tang[-1] = path[-1] - path[-2]
    tang = normalize(tang)
    up = vec3(up_hint)
    if abs(np.dot(normalize(up), tang[0])) > 0.99:
        up = np.array([1.0, 0.0, 0.0]) if abs(tang[0][0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    nor = np.zeros_like(path)
    nor[0] = normalize(np.cross(np.cross(tang[0], up), tang[0]))
    for i in range(1, n):
        b = np.cross(tang[i - 1], tang[i])
        bl = np.linalg.norm(b)
        if bl < 1e-9:
            nor[i] = nor[i - 1]
        else:
            ang = math.degrees(math.atan2(bl, float(np.dot(tang[i - 1], tang[i]))))
            nor[i] = quat_rotate(quat_from_axis_angle(b / bl, ang), nor[i - 1])
        nor[i] = normalize(nor[i] - tang[i] * np.dot(nor[i], tang[i]))
    binor = np.cross(tang, nor)
    return tang, nor, binor


def as_color(c, alpha: float | None = None) -> np.ndarray:
    """Accept '#rrggbb', '#rrggbbaa', (r,g,b) or (r,g,b,a) in 0..1 -> RGBA float array."""
    if isinstance(c, str):
        s = c.lstrip("#")
        if len(s) == 3:
            s = "".join(ch * 2 for ch in s)
        vals = [int(s[i:i + 2], 16) / 255.0 for i in range(0, len(s), 2)]
        if len(vals) == 3:
            vals.append(1.0)
        arr = np.array(vals, dtype=np.float64)
    else:
        arr = np.asarray(c, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            arr = np.array([arr[0], arr[0], arr[0], 1.0])
        elif arr.size == 3:
            arr = np.append(arr, 1.0)
        if arr.max() > 1.0 + 1e-9:
            arr = arr / 255.0
    if alpha is not None:
        arr[3] = alpha
    return arr


def srgb_to_linear(c):
    c = np.asarray(c, dtype=np.float64)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c):
    c = np.clip(np.asarray(c, dtype=np.float64), 0, None)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055)
