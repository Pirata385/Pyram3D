"""Generate docs/API.md from the live package (signatures + docstrings).

Run: python tools/gen_api_docs.py
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pyram3d as p3  # noqa: E402
from pyram3d import (animation, armature, importer, material, math3d, mesh, noise, primitives, project, render,  # noqa: E402
                     scene, sculpt, sdf, terrain, viewer)
from pyram3d.generators import architecture, character, nature, props, vehicles  # noqa: E402


def sig(obj) -> str:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return "(...)"


def doc(obj) -> str:
    d = inspect.getdoc(obj) or ""
    return d.strip()


def section_functions(mod, names=None, title=None) -> list[str]:
    out = [f"## {title or mod.__name__}", ""]
    md = doc(mod)
    if md:
        out += [md, ""]
    items = names or [n for n, o in vars(mod).items()
                      if not n.startswith("_") and (inspect.isfunction(o) or inspect.isclass(o))
                      and getattr(o, "__module__", "") == mod.__name__]
    for n in items:
        o = getattr(mod, n)
        if inspect.isclass(o):
            out += [f"### class `{n}{sig(o)}`", ""]
            if doc(o):
                out += [doc(o), ""]
            for mn, mo in vars(o).items():
                if mn.startswith("_"):
                    continue
                if isinstance(mo, (staticmethod, classmethod)):
                    mo = mo.__func__
                if isinstance(mo, property):
                    out.append(f"- `{n}.{mn}` (property){': ' + doc(mo).splitlines()[0] if doc(mo) else ''}")
                    continue
                if inspect.isfunction(mo):
                    first = doc(mo).splitlines()[0] if doc(mo) else ""
                    out.append(f"- `{mn}{sig(mo)}`" + (f" — {first}" if first else ""))
            out.append("")
        elif inspect.isfunction(o):
            out += [f"### `{n}{sig(o)}`", ""]
            if doc(o):
                out += [doc(o), ""]
    return out


def main():
    lines = ["# Pyram3D API reference", "",
             f"_Generated from pyram3d {p3.__version__} by `tools/gen_api_docs.py` — do not edit by hand._", ""]
    for mod, title in [(mesh, "Mesh (pyram3d.mesh)"), (primitives, "Primitives (pyram3d.primitives)"),
                       (sdf, "SDF modeling (pyram3d.sdf)"), (sculpt, "Sculpting (pyram3d.sculpt)"),
                       (material, "Materials & textures (pyram3d.material)"), (scene, "Scene graph (pyram3d.scene)"),
                       (animation, "Animation (pyram3d.animation)"), (armature, "Rigging (pyram3d.armature)"),
                       (terrain, "Terrain (pyram3d.terrain)"), (render, "Preview rendering (pyram3d.render)"),
                       (project, "Saving (pyram3d.project)"), (viewer, "HTML viewer (pyram3d.viewer)"),
                       (importer, "Import (pyram3d.importer)"), (noise, "Noise (pyram3d.noise)"),
                       (math3d, "Math (pyram3d.math3d)"),
                       (character, "Generators: characters"), (nature, "Generators: nature"),
                       (architecture, "Generators: architecture"), (vehicles, "Generators: vehicles"),
                       (props, "Generators: props")]:
        lines += section_functions(mod, title=title)
    (ROOT / "docs" / "API.md").write_text("\n".join(lines) + "\n")
    print("wrote docs/API.md", len(lines), "lines")


if __name__ == "__main__":
    main()
