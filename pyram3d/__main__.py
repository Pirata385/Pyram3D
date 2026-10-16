"""Command line interface.

    python -m pyram3d info   model.glb
    python -m pyram3d render model.glb -o shot.png [--views] [--style toon] [--time 0.5]
    python -m pyram3d convert model.glb model.fbx
    python -m pyram3d viewer model.glb -o model.html
    python -m pyram3d check  model.glb
    python -m pyram3d run    build.py            (runs a build script)
"""
from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main(argv=None):
    from . import load
    from .render import render, render_frames, render_views
    from .viewer import write_viewer

    ap = argparse.ArgumentParser(prog="pyram3d", description="Pyram3D - AI-first 3D modeling toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info", help="describe a model file")
    p.add_argument("file")
    p = sub.add_parser("render", help="render a model file to PNG")
    p.add_argument("file")
    p.add_argument("-o", "--out", default=None)
    p.add_argument("--views", action="store_true", help="contact sheet of 6 views")
    p.add_argument("--view", default="three_quarter")
    p.add_argument("--style", default="realistic", choices=["realistic", "toon", "clay", "normals", "wire"])
    p.add_argument("--time", type=float, default=None)
    p.add_argument("--animation", default=None, help="filmstrip of this animation")
    p.add_argument("--size", type=int, default=800)
    p = sub.add_parser("convert", help="convert between .glb/.gltf/.fbx/.obj")
    p.add_argument("src")
    p.add_argument("dst")
    p = sub.add_parser("viewer", help="write a self-contained HTML viewer")
    p.add_argument("file")
    p.add_argument("-o", "--out", default=None)
    p = sub.add_parser("check", help="validate meshes (manifold, degenerate faces...)")
    p.add_argument("file")
    p = sub.add_parser("run", help="run a build script")
    p.add_argument("script")
    p.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)

    if a.cmd == "run":
        sys.argv = [a.script] + a.args
        sys.path.insert(0, str(Path(a.script).resolve().parent))
        runpy.run_path(a.script, run_name="__main__")
        return 0
    if a.cmd == "info":
        print(load(a.file).describe())
        return 0
    if a.cmd == "render":
        sc = load(a.file)
        out = a.out or str(Path(a.file).with_suffix(".png"))
        if a.animation:
            render_frames(sc, out, a.animation, style=a.style)
        elif a.views:
            render_views(sc, out, size=a.size // 2, style=a.style)
        else:
            render(sc, out, a.size, int(a.size * 0.75), view=a.view, style=a.style, time=a.time)
        print(out)
        return 0
    if a.cmd == "convert":
        sc = load(a.src)
        print(sc.export(a.dst))
        return 0
    if a.cmd == "viewer":
        out = a.out or str(Path(a.file).with_suffix(".html"))
        src = a.file
        if not a.file.lower().endswith(".glb"):
            src = load(a.file)
        print(write_viewer(src, out))
        return 0
    if a.cmd == "check":
        sc = load(a.file)
        bad = 0
        for n in sc.mesh_nodes():
            probs = n.mesh.validate()
            st = n.mesh.stats()
            status = "OK" if not probs else "; ".join(probs)
            print(f"{n.name}: {st['triangles']} tris, watertight={st.get('watertight')} -> {status}")
            bad += bool(probs)
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
