"""Minimal, dependency-free PNG encoder/decoder (numpy + zlib)."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np


def _chunk(tag: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def to_uint8(img) -> np.ndarray:
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = np.clip(np.round(np.asarray(a, dtype=np.float64) * 255.0), 0, 255).astype(np.uint8)
    if a.ndim == 2:
        a = a[:, :, None]
    return a


def encode_png(img, compress_level: int = 6) -> bytes:
    """img: (H, W), (H, W, 1|2|3|4) uint8, or float in [0, 1]."""
    a = to_uint8(img)
    h, w, c = a.shape
    color_type = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    raw = np.zeros((h, 1 + w * c), dtype=np.uint8)
    raw[:, 1:] = a.reshape(h, w * c)  # filter type 0 (None) for every row
    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", ihdr)
            + _chunk(b"IDAT", zlib.compress(raw.tobytes(), compress_level)) + _chunk(b"IEND", b""))


def write_png(path, img) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encode_png(img))
    return path


def decode_png(data: bytes) -> np.ndarray:
    """Decode a non-interlaced 8-bit PNG into an (H, W, 4) uint8 RGBA array."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    idat = []
    palette = None
    trns = None
    w = h = bit_depth = color_type = interlace = None
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif tag == b"PLTE":
            palette = np.frombuffer(body, dtype=np.uint8).reshape(-1, 3)
        elif tag == b"tRNS":
            trns = np.frombuffer(body, dtype=np.uint8)
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    if bit_depth != 8 or interlace:
        raise ValueError("only 8-bit non-interlaced PNGs are supported (convert the texture first)")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), dtype=np.uint8)
    stride = w * channels
    raw = raw.reshape(h, stride + 1)
    out = np.zeros((h, stride), dtype=np.uint8)
    bpp = channels
    prev = np.zeros(stride, dtype=np.uint8)
    for y in range(h):
        ft = raw[y, 0]
        line = raw[y, 1:].copy()
        if ft == 0:
            pass
        elif ft == 1:
            for i in range(bpp, stride, bpp):
                line[i:i + bpp] = line[i:i + bpp] + line[i - bpp:i]
        elif ft == 2:
            line = line + prev
        elif ft == 3:
            for i in range(stride):
                left = int(line[i - bpp]) if i >= bpp else 0
                line[i] = (int(line[i]) + ((left + int(prev[i])) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = int(line[i - bpp]) if i >= bpp else 0
                b = int(prev[i])
                c = int(prev[i - bpp]) if i >= bpp else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (int(line[i]) + pr) & 0xFF
        else:
            raise ValueError(f"bad PNG filter {ft}")
        out[y] = line
        prev = line
    img = out.reshape(h, w, channels)
    if color_type == 3:
        rgb = palette[img[:, :, 0]]
        alpha = np.full((h, w, 1), 255, dtype=np.uint8)
        if trns is not None:
            lut = np.full(256, 255, dtype=np.uint8)
            lut[:len(trns)] = trns
            alpha = lut[img[:, :, 0]][:, :, None]
        return np.concatenate([rgb, alpha], axis=2)
    if channels == 1:
        img = np.repeat(img, 3, axis=2)
        channels = 3
    elif channels == 2:
        img = np.concatenate([np.repeat(img[:, :, :1], 3, axis=2), img[:, :, 1:]], axis=2)
        channels = 4
    if channels == 3:
        img = np.concatenate([img, np.full((h, w, 1), 255, dtype=np.uint8)], axis=2)
    return img


def read_image(path) -> np.ndarray:
    """Read a PNG (built-in) or any format Pillow understands (if installed). Returns RGBA uint8."""
    path = Path(path)
    data = path.read_bytes()
    return decode_image(data, path.name)


def decode_image(data: bytes, name: str = "image") -> np.ndarray:
    try:  # Pillow is optional; it is faster and supports JPEG etc.
        from PIL import Image
        import io
        return np.asarray(Image.open(io.BytesIO(data)).convert("RGBA")).copy()
    except ImportError:
        pass
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return decode_png(data)
    raise ValueError(f"cannot decode {name}: only PNG is supported without Pillow")


def resize_nearest(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ys = (np.arange(h) * img.shape[0] / h).astype(int)
    xs = (np.arange(w) * img.shape[1] / w).astype(int)
    return img[ys][:, xs]


def downsample(img: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return img
    h, w = img.shape[:2]
    h2, w2 = h // factor, w // factor
    a = img[:h2 * factor, :w2 * factor].astype(np.float64)
    a = a.reshape(h2, factor, w2, factor, -1).mean(axis=(1, 3))
    return a


def grid(images: list[np.ndarray], cols: int, pad: int = 4, bg=(255, 255, 255)) -> np.ndarray:
    """Tile same-size uint8 RGB(A) images into a contact sheet."""
    imgs = [to_uint8(i)[:, :, :3] for i in images]
    h, w = imgs[0].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.zeros((rows * h + (rows + 1) * pad, cols * w + (cols + 1) * pad, 3), dtype=np.uint8)
    sheet[:] = np.array(bg, dtype=np.uint8)
    for i, im in enumerate(imgs):
        r, c = divmod(i, cols)
        y = pad + r * (h + pad)
        x = pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = im
    return sheet
