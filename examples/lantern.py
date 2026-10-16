"""Hard-surface modeling from primitives and face operations (no generators).

A hanging oil lantern: a revolved base, a glass body, a metal cage made with
inset/extrude, a handle swept along a curve and an emissive flame.
Run: python examples/lantern.py  ->  examples/output/lantern/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyram3d as p3

OUT = Path(__file__).resolve().parent / "output" / "lantern" / "lantern"
metal = p3.presets.metal("#3b3f45", roughness=0.45)
brass = p3.presets.metal("#c9a24b", roughness=0.3)

# base and top: revolved profiles (radius, height)
base = p3.lathe([(0, 0), (0.09, 0), (0.1, 0.015), (0.085, 0.03), (0.075, 0.035), (0, 0.035)], 32).set_material(brass)
top = p3.lathe([(0, 0.25), (0.085, 0.25), (0.06, 0.29), (0.02, 0.31), (0.02, 0.33), (0, 0.33)], 32)
top.set_material(brass)

# glass chimney
glass = p3.cylinder(0.06, 0.2, 32, radius_top=0.055, caps=False).translate((0, 0.135, 0))
glass.set_material(p3.presets.glass())

# cage: a square tube whose side faces are inset and pushed in (window frames)
cage = p3.box((0.15, 0.215, 0.15), segments=(1, 2, 1)).translate((0, 0.1425, 0))
cage.delete_faces(cage.faces_facing((0, 1, 0), 10))
cage.delete_faces(cage.faces_facing((0, -1, 0), 10))
sides = cage.faces_where(lambda c, n: abs(n[:, 1]) < 0.5)
cage.inset(sides, 0.012)
cage.delete_faces(sides)                    # open windows, keep the frame ring
cage.solidify(0.006)
cage.set_material(metal)

# handle: a tube swept along a smooth curve
handle = p3.sweep(p3.catmull_rom([(-0.07, 0.27, 0), (-0.06, 0.38, 0), (0, 0.42, 0), (0.06, 0.38, 0),
                                   (0.07, 0.27, 0)], 8), radius=0.005, segments=8).set_material(metal)

# flame: emissive teardrop via SDF
flame = p3.sdf.round_cone((0, 0.07, 0), (0, 0.12, 0), 0.014, 0.002).to_mesh(40)
flame.set_material(p3.presets.emissive("#ffb347", 6.0))

lantern = p3.Mesh.join([base, top, glass, cage, handle, flame], name="Lantern")

scene = p3.Scene("Lantern")
scene.add(lantern)
scene.add_default_lights()

if __name__ == "__main__":
    print(lantern.stats())
    out = p3.save(scene, OUT, size=800)
    print({k: str(v.name) for k, v in out.items()})
