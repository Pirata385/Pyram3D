# Pyram3D — instructions for AI agents

Pyram3D is a 3D modeling toolkit **built to be operated by an AI agent** (you), not by a
human with a mouse. The user describes what they want (usually in Portuguese, often from a
phone); you model it with Python code, *look* at software-rendered previews, iterate, and
deliver `.glb`, `.gltf` and `.fbx` files (plus an interactive viewer page).

Runtime: Python 3.10+ and **numpy only** (Pillow optional). No Blender, no GPU, nothing
for the user to install. Full API: `docs/API.md` (generated, always accurate).

## Workflow (follow it every time)

1. **Pin down the spec.** Size in meters, style (low-poly / stylized / realistic), where it
   will be used (Unity, Unreal, Godot, web/three.js, Roblox…), polygon budget, needed
   animations, colors/materials. Ask only what you cannot reasonably default.
2. **Create `projects/<name>/build.py`** (one folder per asset or scene). Start from
   `examples/` scripts when one is close.
3. **Model** with primitives + modeling ops, SDF shapes, generators, materials (cookbook below).
4. **Run** `python projects/<name>/build.py`.
5. **Inspect numerically**: `print(scene.describe())`, `mesh.stats()`, `mesh.validate()`.
6. **Inspect visually — always.** Render and then open the PNGs with your image-reading tool:
   - `p3.render(scene, "…/preview.png")` – 3/4 beauty shot
   - `p3.render_views(scene, "…/views.png")` – front, right, back, 3/4, top, left
   - `p3.render(scene, "…/wire.png", style="wire")` – topology check
   - `p3.render_frames(scene, "…/walk.png", "Walk", frames=6)` – animation filmstrip
   Compare against the spec; fix proportions, gaps, floating parts, wrong colors. Iterate.
7. **Deliver** with one call: `p3.save(scene, "projects/<name>/exports/<name>")` → writes
   `.glb .gltf .fbx`, `_preview.png`, `_views.png`, one filmstrip per animation and
   `_viewer.html` (three.js page with the model embedded, works on phones).
8. **Show the user**: publish `_viewer.html` as an artifact when that tool exists (it is a
   complete HTML page; it renders on phones), otherwise send the preview PNG. Then
   commit and push the `projects/<name>/` folder so the user can download the files from
   GitHub. Mention the triangle count and dimensions.

## Conventions (glTF-style, everywhere)

- Units **meters**; **+Y up**; the model's front faces **+Z**; objects stand on **y = 0**.
- A character's **left is +X**, right is −X.
- Rotations are **Euler degrees `(x, y, z)`** (applied X, then Y, then Z) or quaternions `(x, y, z, w)`.
- Colors are **sRGB**: `"#ff8800"`, `(1, 0.5, 0)` or 0–255 tuples. They are converted to linear internally.
- Mesh methods mutate and return `self` → chain them: `p3.box(1).subdivide(2).translate((0, 1, 0))`.
  Use `.copy()` before reusing a mesh with modifications.
- Re-using the same `Mesh` object in many nodes = instancing (small files).

## Cookbook

```python
import pyram3d as p3
from pyram3d import sdf

scene = p3.Scene("Name")              # fps=30 default

# Hard-surface: primitives + face ops
m = p3.box((2, 1, 1), segments=(4, 2, 2))
top = m.faces_facing((0, 1, 0))       # select faces by normal (also faces_where / faces_in_box)
m.inset(top, 0.05).extrude(top, 0.3)  # inset then extrude the region
m.set_material(p3.presets.metal(), top)
p3.rounded_box((1, 0.5, 1), radius=0.05)   # bevelled box
p3.lathe([(0, 0), (0.3, 0), (0.25, 0.6), (0.1, 0.9), (0, 0.9)], 32)   # revolve (vases, bottles, wheels)
p3.sweep(p3.catmull_rom(points), radius=lambda t: 0.1 * (1 - t))       # tubes, branches, cables, horns
p3.loft([section0, section1, ...])          # hulls, car bodies, blades
p3.extrude_shape(p3.shapes.star(5), 0.1)    # 2D outline → solid (logos, gears, signs)
m.mirror("x"); m.subdivide(2); m.bend(45); m.twist(90); m.taper(0.5); m.solidify(0.02)
m.uv_box() / m.uv_cylinder() / m.uv_sphere() / m.uv_planar()   # re-project UVs after heavy edits

# Organic: SDF shapes with smooth blending, booleans, per-part colors/materials
body = sdf.round_cone((0, 0.5, 0), (0, 1.2, 0), 0.25, 0.18, color="#3b6fb6")
head = sdf.sphere((0, 1.45, 0), 0.18, color="#e2b08f")
shape = body.smooth_union(head, k=0.05)
shape = shape - sdf.sphere((0, 1.5, 0.17), 0.04)          # boolean subtract
mesh = shape.to_mesh(resolution=120)                      # 64 draft · 96–128 normal · 160+ detail
# thin parts (< ~2 grid cells) disappear at low resolution: raise resolution or thicken them

# Sculpting on dense meshes
p3.sculpt.draw(mesh, center, radius=0.1, strength=0.02, symmetry="x")
p3.sculpt.grab(mesh, center, radius=0.2, offset=(0, 0.1, 0)); p3.sculpt.smooth(...); p3.sculpt.crease(...)

# Materials (PBR metallic/roughness) and procedural textures
p3.Material("Paint", "#c0392b", metallic=0.6, roughness=0.25)
p3.presets.wood() / metal() / gold() / chrome() / glass() / plastic() / rubber() / stone() / bricks()
p3.presets.marble() / fabric() / skin() / foliage() / bark() / water() / emissive() / toon()
p3.textures.checker() / noise() / wood() / marble() / bricks() / tiles() / cells() / stripes() / gradient()
p3.textures.normal_from_height(tex); p3.Texture.load("file.png"); p3.Texture.from_function(f, 512)
mesh.set_color("#ff0000", verts=idx)   # vertex colors (cheap, engine friendly)

# Scene, lights, cameras
node = scene.add(mesh, position=(0, 0, 0), rotation=(0, 45, 0), scale=1)
child = node.add(other_mesh, position=(0, 1, 0))          # hierarchy
scene.add_default_lights()                                 # key/fill/rim (or add_light(...))
scene.add_camera(position=(4, 2, 6), target=(0, 1, 0))

# Animation (any node, bones included)
a = scene.animation("Open")
a.keys(node, "rotation", {0: (0, 0, 0), 1: (0, 90, 0)}, interpolation="SMOOTH")
p3.spin(a, node); p3.bounce(a, node); p3.orbit(a, camera_node)

# Rigged characters & animals (bone names follow Unity Humanoid)
hero = p3.character.humanoid(height=1.75, shirt="#aa3333", build=1.0)
hero.add_to(scene)
p3.character.walk_cycle(scene.animation("Walk"), hero.skeleton)   # run_cycle, idle, jump, wave
dog = p3.character.quadruped(length=1.0, height=0.55); dog.add_to(scene, position=(2, 0, 0))
p3.character.quadruped_walk(scene.animation("DogWalk"), dog.skeleton)
# custom rig: sk = p3.Skeleton(); sk.add_bone(name, head, tail, parent); sk.bind(mesh); node.skin = sk

# Worlds
t = p3.terrain(size=80, resolution=160, height=10, island=0.8, seed=3)
for s in p3.scatter(t, 40, max_slope=25, min_height=0.5, seed=1):
    scene.add(tree_mesh, position=s["position"], rotation=s["rotation"], scale=s["scale"])
scene.add(p3.water_plane(80, level=0.3))

# Generators (starting points — modify their meshes freely)
p3.nature.tree() / pine() / palm() / bush() / rock() / grass_patch() / flower() / mushroom() / cactus()
p3.architecture.building() / house() / column() / fence() / stairs() / wall_segment() / street_lamp() / road()
p3.vehicles.car(style="sedan|hatchback|suv|pickup") (Node with Wheel_* children) / boat() / airplane()
p3.vehicles.drive(scene.animation("Drive"), car_node, distance=10)
p3.props.crate() / barrel() / table() / chair() / sofa() / lamp() / bookshelf() / sword() / treasure_chest()

# Import existing models to edit/convert
scene = p3.load("model.glb")   # .glb/.gltf/.obj

p3.save(scene, "projects/name/exports/name")
```

CLI: `python -m pyram3d info|render|convert|viewer|check|run …` (e.g. `convert a.glb a.fbx`).

## Quality checklist before delivering

- `mesh.validate()` is empty; closed objects report `watertight=True` in `stats()`.
- Real-world scale (door ≈ 2 m, chair seat ≈ 0.45 m, person ≈ 1.7–1.8 m, car ≈ 4.3 m).
- Nothing floats or intersects by accident; the model sits on y = 0 and faces +Z.
- Triangle budget fits the target (mobile prop < 5k, hero character 10–30k, scenery instanced).
- Every part has a material; colors match the brief; the 6-view sheet looks right from all sides.
- Animations: filmstrip looks right, loops are seamless (first key == last key), feet don't sink.
- Re-run `python -m pytest -q` after changing the library itself.

## Notes and limits

- The preview renderer is a CPU rasterizer (PBR-ish, shadows, toon/clay/wire/normals styles).
  Large scenes take a few seconds; keep preview sizes ≤ 1200 px.
- FBX is written as binary FBX 7.4 without the Autodesk SDK; it is validated by importing into
  Blender. Materials are exported as Phong with PBR hints; for exact PBR prefer glTF/GLB.
- glTF output passes the Khronos glTF-Validator with 0 errors.
- Not implemented yet: fluid/smoke simulation, physics-based cloth animation, path-traced renders,
  UV unwrapping beyond projections, retopology/decimation. Say so when a request needs them.
- If you change the library API, regenerate the docs: `python tools/gen_api_docs.py`.
