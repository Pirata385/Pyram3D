"""Tiny FBX binary reader used by the tests to verify exporter output structure."""
import struct
import zlib

import numpy as np


def _read_prop(data, pos):
    t = chr(data[pos])
    pos += 1
    if t == "Y":
        return struct.unpack_from("<h", data, pos)[0], pos + 2
    if t == "C":
        return bool(data[pos]), pos + 1
    if t == "I":
        return struct.unpack_from("<i", data, pos)[0], pos + 4
    if t == "F":
        return struct.unpack_from("<f", data, pos)[0], pos + 4
    if t == "D":
        return struct.unpack_from("<d", data, pos)[0], pos + 8
    if t == "L":
        return struct.unpack_from("<q", data, pos)[0], pos + 8
    if t in "SR":
        n = struct.unpack_from("<I", data, pos)[0]
        return bytes(data[pos + 4:pos + 4 + n]), pos + 4 + n
    if t in "fdlib":
        n, enc, clen = struct.unpack_from("<III", data, pos)
        raw = bytes(data[pos + 12:pos + 12 + clen])
        if enc == 1:
            raw = zlib.decompress(raw)
        dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "?"}[t]
        return np.frombuffer(raw, dtype=dt, count=n), pos + 12 + clen
    raise ValueError(f"bad property type {t!r} at {pos}")


def _read_node(data, pos):
    end, nprops, plen, nlen = struct.unpack_from("<IIIB", data, pos)
    if end == 0:
        return None, pos + 13
    name = bytes(data[pos + 13:pos + 13 + nlen]).decode()
    p = pos + 13 + nlen
    props = []
    for _ in range(nprops):
        v, p = _read_prop(data, p)
        props.append(v)
    children = []
    while p < end:
        child, p = _read_node(data, p)
        if child is None:
            break
        children.append(child)
    return {"name": name, "props": props, "children": children}, end


def read_fbx(path):
    data = open(path, "rb").read()
    assert data[:21] == b"Kaydara FBX Binary  \x00"
    version = struct.unpack_from("<I", data, 23)[0]
    pos = 27
    nodes = []
    while True:
        node, pos = _read_node(data, pos)
        if node is None:
            break
        nodes.append(node)
    return version, nodes, data


def find(nodes, name):
    return [n for n in nodes if n["name"] == name]


def child(node, name):
    for c in node["children"]:
        if c["name"] == name:
            return c
    return None
