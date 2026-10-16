# Pyram3D API reference

_Generated from pyram3d 0.1.0 by `tools/gen_api_docs.py` — do not edit by hand._

## Mesh (pyram3d.mesh)

Polygon mesh with modeling operations.

A Mesh stores:
  vertices       (N, 3) float64 positions
  faces          list of tuples of vertex indices (tris, quads or n-gons, CCW = front)
  uvs/face_uvs   optional face-varying UVs (like OBJ: face_uvs[i] indexes into uvs)
  face_materials per-face index into `materials`
  colors         optional (N, 4) per-vertex colors, LINEAR RGBA
  joints/weights optional (N, 4) skinning data (filled by Skeleton.bind)
  shape_keys     dict name -> (N, 3) absolute positions (morph targets / blend shapes)
  smooth_angle   shading: 0 = flat, 180 = fully smooth, in between = auto-smooth

Every modeling method mutates the mesh in place and returns `self` so calls can
be chained:  box().subdivide(2).scale(0.5).translate((0, 1, 0))
Methods that create new geometry (extrude, inset) return the new face indices
when `return_faces=True`.

### class `Mesh(vertices=None, faces=None, uvs=None, face_uvs=None, name: 'str' = 'Mesh', materials=None, face_materials=None, colors=None, smooth_angle: 'float' = 40.0)`

- `Mesh.num_vertices` (property)
- `Mesh.num_faces` (property)
- `Mesh.num_triangles` (property)
- `copy(self, name: 'str | None' = None) -> "'Mesh'"`
- `face_normals(self) -> 'np.ndarray'` — Unit normals via Newell's method (robust for n-gons).
- `face_areas(self) -> 'np.ndarray'`
- `face_centers(self) -> 'np.ndarray'`
- `vertex_normals(self) -> 'np.ndarray'` — Area-weighted smooth vertex normals.
- `corner_normals(self, angle: 'float | None' = None) -> 'np.ndarray'` — Per-corner normals honoring the auto-smooth angle (degrees).
- `bounds(self) -> 'tuple[np.ndarray, np.ndarray]'`
- `Mesh.size` (property)
- `Mesh.center` (property)
- `edges(self) -> 'np.ndarray'` — Unique undirected edges (E, 2).
- `boundary_edges(self) -> 'np.ndarray'`
- `transform(self, matrix) -> "'Mesh'"`
- `translate(self, offset) -> "'Mesh'"`
- `rotate(self, rotation, pivot=None) -> "'Mesh'"` — rotation: Euler degrees (x, y, z) or quaternion (x, y, z, w).
- `scale(self, factor, pivot=None) -> "'Mesh'"`
- `resize(self, size, uniform: 'bool' = True) -> "'Mesh'"` — Scale so the bounding box has the given size (scalar = largest dimension).
- `center_origin(self, axes: 'str' = 'xyz') -> "'Mesh'"`
- `ground(self, y: 'float' = 0.0, center_xz: 'bool' = False) -> "'Mesh'"` — Move so the lowest point sits at height y (optionally centered on X/Z).
- `flip(self, faces: 'FaceSel' = None) -> "'Mesh'"` — Reverse face winding (flips normals).
- `merge(self, *others: "'Mesh'") -> "'Mesh'"` — Append other meshes into this one (materials and attributes are combined).
- `join(meshes: "Sequence['Mesh']", name: 'str' = 'Mesh') -> "'Mesh'"`
- `remove_unused_vertices(self) -> "'Mesh'"`
- `weld(self, threshold: 'float' = 1e-05) -> "'Mesh'"` — Merge vertices closer than `threshold` (merge by distance).
- `delete_faces(self, faces: 'FaceSel') -> "'Mesh'"`
- `separate(self, faces: 'FaceSel', name: 'str | None' = None) -> "'Mesh'"` — Return a new mesh containing only the selected faces (this mesh is unchanged).
- `faces_facing(self, direction, angle: 'float' = 30.0) -> 'np.ndarray'` — Indices of faces whose normal is within `angle` degrees of `direction`.
- `faces_where(self, predicate: 'Callable[[np.ndarray, np.ndarray], np.ndarray]') -> 'np.ndarray'` — predicate(centers (F,3), normals (F,3)) -> bool mask. Returns face indices.
- `faces_in_box(self, lo, hi) -> 'np.ndarray'`
- `verts_where(self, predicate: 'Callable[[np.ndarray], np.ndarray]') -> 'np.ndarray'`
- `triangulate(self) -> "'Mesh'"`
- `subdivide(self, levels: 'int' = 1, smooth: 'bool' = True) -> "'Mesh'"` — Catmull-Clark subdivision (smooth=True) or simple linear subdivision.
- `extrude(self, faces: 'FaceSel' = None, distance: 'float' = 0.1, offset=None, individual: 'bool' = False, return_faces: 'bool' = False)` — Extrude a region of faces along their normals (or by an `offset` vector).
- `inset(self, faces: 'FaceSel' = None, thickness: 'float' = 0.05, depth: 'float' = 0.0, return_faces: 'bool' = False)` — Inset each selected face individually (creates a border ring of quads).
- `mirror(self, axis: 'str' = 'x', weld: 'bool' = True, threshold: 'float' = 0.0001) -> "'Mesh'"` — Mirror across the plane axis=0 and append (symmetric modeling).
- `solidify(self, thickness: 'float' = 0.02) -> "'Mesh'"` — Give an open surface (e.g. a leaf or cloth) thickness: inner shell + rim.
- `array(self, count: 'int', offset=(1, 0, 0), rotation=None, scale=None) -> "'Mesh'"` — Repeat the mesh `count` times, each copy offset/rotated/scaled from the previous.
- `deform(self, func: 'Callable[[np.ndarray], np.ndarray]') -> "'Mesh'"` — Apply func(points (N,3)) -> (N,3) to all vertices (and shape keys).
- `smooth(self, iterations: 'int' = 1, factor: 'float' = 0.5, verts=None, keep_boundary: 'bool' = True) -> "'Mesh'"` — Laplacian smoothing (optionally restricted to a vertex selection).
- `displace(self, amount: 'float' = 0.05, scale: 'float' = 2.0, octaves: 'int' = 4, seed: 'int' = 0, func: 'Callable[[np.ndarray], np.ndarray] | None' = None) -> "'Mesh'"` — Push vertices along their normals by fBm noise (or by func(points)->(N,) heights).
- `bend(self, angle: 'float', axis: 'str' = 'y', direction: 'str' = 'x', center=None) -> "'Mesh'"` — Bend along `axis` (the long axis) toward `direction`; total bend `angle` degrees.
- `twist(self, angle: 'float', axis: 'str' = 'y') -> "'Mesh'"` — Twist around `axis`, rotating `angle` degrees from bottom to top.
- `taper(self, factor: 'float', axis: 'str' = 'y', curve: 'float' = 1.0) -> "'Mesh'"` — Scale cross-sections along `axis`: 1 at the bottom to `factor` at the top.
- `spherify(self, amount: 'float' = 1.0, center=None, radius: 'float | None' = None) -> "'Mesh'"`
- `uv_box(self, scale: 'float' = 1.0) -> "'Mesh'"` — Box/tri-planar projection (good default for hard-surface objects).
- `uv_planar(self, axis: 'str' = 'y', fit: 'bool' = True) -> "'Mesh'"`
- `uv_cylinder(self, axis: 'str' = 'y') -> "'Mesh'"`
- `uv_sphere(self) -> "'Mesh'"`
- `uv_transform(self, scale=(1, 1), offset=(0, 0), rotation: 'float' = 0.0) -> "'Mesh'"`
- `set_material(self, material, faces: 'FaceSel' = None) -> "'Mesh'"` — Assign a material to all faces or to a selection of faces.
- `set_color(self, color, verts=None, func: 'Callable[[np.ndarray], np.ndarray] | None' = None) -> "'Mesh'"` — Paint vertex colors (sRGB input). `func(points)->(N,3|4)` sRGB for gradients.
- `shade_flat(self) -> "'Mesh'"`
- `shade_smooth(self, angle: 'float' = 180.0) -> "'Mesh'"`
- `add_shape_key(self, name: 'str', func: 'Callable[[np.ndarray], np.ndarray] | None' = None, positions=None) -> "'Mesh'"` — Add a morph target (blend shape) e.g. for facial expressions.
- `stats(self) -> 'dict'`
- `validate(self) -> 'list[str]'` — Return a list of human-readable problems (empty list = OK).

### `triangulate_corners(mesh: 'Mesh') -> 'tuple[np.ndarray, np.ndarray]'`

Triangulate all faces. Returns (T,3) GLOBAL corner indices and (T,) source face ids.

## Primitives (pyram3d.primitives)

Primitive and procedural shape builders.

Every function returns a new Mesh, centered on the origin unless stated
otherwise, with outward-facing normals and UVs.
Sizes are in meters. +Y is up.

### `box(size=1.0, segments: 'int | Sequence[int]' = 1, name: 'str' = 'Box') -> 'Mesh'`

Axis-aligned box. size: scalar or (width_x, height_y, depth_z).

### `cube(size=1.0, segments: 'int | Sequence[int]' = 1, name: 'str' = 'Box') -> 'Mesh'`

Axis-aligned box. size: scalar or (width_x, height_y, depth_z).

### `rounded_box(size=1.0, radius: 'float' = 0.1, segments: 'int' = 4, name: 'str' = 'RoundedBox') -> 'Mesh'`

Box with rounded edges/corners (radius in meters, `segments` per rounded edge).

### `plane(size=1.0, segments: 'int | Sequence[int]' = 1, name: 'str' = 'Plane') -> 'Mesh'`

Flat grid in the XZ plane facing +Y. size: scalar or (width_x, depth_z).

### `grid(size=1.0, segments: 'int | Sequence[int]' = 1, name: 'str' = 'Plane') -> 'Mesh'`

Flat grid in the XZ plane facing +Y. size: scalar or (width_x, depth_z).

### `disc(radius: 'float' = 0.5, segments: 'int' = 32, name: 'str' = 'Disc') -> 'Mesh'`

Flat circle (n-gon) in the XZ plane facing +Y.

### `uv_sphere(radius: 'float' = 0.5, segments: 'int' = 32, rings: 'int' = 16, name: 'str' = 'Sphere') -> 'Mesh'`

### `sphere(radius: 'float' = 0.5, segments: 'int' = 32, rings: 'int' = 16, name: 'str' = 'Sphere') -> 'Mesh'`

### `ico_sphere(radius: 'float' = 0.5, subdivisions: 'int' = 2, name: 'str' = 'IcoSphere') -> 'Mesh'`

### `lathe(profile: 'Sequence[Sequence[float]]', segments: 'int' = 32, angle: 'float' = 360.0, name: 'str' = 'Lathe', smooth_angle: 'float' = 40.0) -> 'Mesh'`

Revolve a 2D profile [(radius, y), ...] around the Y axis (vases, bottles, trunks, wheels).

Points with radius == 0 become poles (closed ends). Order the profile from
bottom to top so normals face outward.

### `cylinder(radius: 'float' = 0.5, height: 'float' = 1.0, segments: 'int' = 32, rings: 'int' = 1, radius_top: 'float | None' = None, caps: 'bool' = True, name: 'str' = 'Cylinder') -> 'Mesh'`

Cylinder along Y centered at the origin. radius_top makes a truncated cone.

### `cone(radius: 'float' = 0.5, height: 'float' = 1.0, segments: 'int' = 32, name: 'str' = 'Cone') -> 'Mesh'`

### `capsule(radius: 'float' = 0.25, height: 'float' = 1.0, segments: 'int' = 24, rings: 'int' = 8, name: 'str' = 'Capsule') -> 'Mesh'`

Capsule along Y. height is the total height including the hemispheres.

### `torus(major_radius: 'float' = 0.5, minor_radius: 'float' = 0.15, major_segments: 'int' = 48, minor_segments: 'int' = 16, name: 'str' = 'Torus') -> 'Mesh'`

Torus lying in the XZ plane (ring around the Y axis).

### `sweep(path: 'Sequence[Sequence[float]]', shape: 'Sequence[Sequence[float]] | None' = None, radius: 'float | Sequence[float] | Callable[[float], float]' = 0.1, segments: 'int' = 12, caps: 'bool' = True, closed: 'bool' = False, twist: 'float' = 0.0, up=(0, 1, 0), name: 'str' = 'Sweep') -> 'Mesh'`

Sweep a cross-section along a 3D path (tubes, branches, cables, horns, rails).

path:   (N, 3) points (use m3.catmull_rom / m3.bezier for smooth curves)
shape:  optional closed 2D cross-section [(x, y), ...] (default: circle)
radius: scalar, per-point list, or function of t in [0, 1] scaling the section
twist:  total twist in degrees along the path

### `tube(path, radius=0.05, segments: 'int' = 12, caps: 'bool' = True, name: 'str' = 'Tube', **kw) -> 'Mesh'`

### `loft(sections: 'Sequence[Sequence[Sequence[float]]]', closed_sections: 'bool' = True, caps: 'bool' = True, name: 'str' = 'Loft', smooth_angle: 'float' = 50.0) -> 'Mesh'`

Skin a surface through cross-sections (each (M,3), same M). Great for hulls, car bodies, blades.

### `extrude_shape(shape: 'Sequence[Sequence[float]]', depth: 'float' = 0.1, caps: 'bool' = True, name: 'str' = 'Extrusion') -> 'Mesh'`

Extrude a closed 2D polygon [(x, y), ...] (in the XY plane) along Z, centered on z=0.

Concave shapes are fine (caps are triangulated by ear clipping on export).
Use shapes.* helpers (star, rounded_rect, circle...) or your own outline.

### class `shapes()`

2D outlines for extrude_shape / sweep (lists of (x, y)).

- `circle(radius: 'float' = 0.5, segments: 'int' = 32)`
- `ellipse(rx: 'float' = 0.5, ry: 'float' = 0.3, segments: 'int' = 32)`
- `rect(w: 'float' = 1.0, h: 'float' = 1.0)`
- `rounded_rect(w: 'float' = 1.0, h: 'float' = 1.0, r: 'float' = 0.1, segments: 'int' = 6)`
- `star(points: 'int' = 5, outer: 'float' = 0.5, inner: 'float' = 0.2)`
- `polygon(sides: 'int' = 6, radius: 'float' = 0.5)`
- `gear(teeth: 'int' = 12, outer: 'float' = 0.5, inner: 'float' = 0.42, tooth_width: 'float' = 0.5)`
- `from_curve(points, samples_per_segment: 'int' = 8)` — Smooth closed outline through control points.

## SDF modeling (pyram3d.sdf)

Signed-distance-field (SDF) modeling: organic shapes, smooth blends and booleans.

Build a shape by combining primitives, then call .to_mesh(). Ideal for
characters, animals, creatures, rocks, fruit, smooth stylized props and
any boolean operation (union / subtract / intersect).

    body = sdf.capsule((0, 0.9, 0), (0, 1.4, 0), 0.18)
    head = sdf.sphere((0, 1.65, 0), 0.13)
    char = body.smooth_union(head, 0.05)
    mesh = char.to_mesh(resolution=96)

Every primitive can carry a `color` (sRGB) and/or `material`; the mesh gets
vertex colors / per-face materials from the closest primitive.

### class `SDF()`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p: 'np.ndarray') -> 'np.ndarray'`
- `bounds(self) -> 'tuple[np.ndarray, np.ndarray]'`
- `leaves(self) -> "list['SDF']"`
- `leaf_dists(self, p: 'np.ndarray') -> 'np.ndarray'` — (N, L) distance of p to every leaf primitive (respecting transforms).
- `union(self, *others: "'SDF'") -> "'SDF'"`
- `smooth_union(self, other: "'SDF'", k: 'float' = 0.05) -> "'SDF'"`
- `subtract(self, other: "'SDF'") -> "'SDF'"`
- `smooth_subtract(self, other: "'SDF'", k: 'float' = 0.05) -> "'SDF'"`
- `intersect(self, other: "'SDF'") -> "'SDF'"`
- `smooth_intersect(self, other: "'SDF'", k: 'float' = 0.05) -> "'SDF'"`
- `translate(self, offset) -> "'SDF'"`
- `rotate(self, rotation, pivot=(0, 0, 0)) -> "'SDF'"`
- `scale(self, factor: 'float') -> "'SDF'"`
- `mirror_x(self) -> "'SDF'"` — Symmetric copy across x=0 (model one side of a character).
- `round(self, radius: 'float') -> "'SDF'"`
- `shell(self, thickness: 'float') -> "'SDF'"` — Hollow shell (onion) of the given thickness.
- `displace(self, amount: 'float' = 0.02, scale: 'float' = 4.0, octaves: 'int' = 4, seed: 'int' = 0, func: 'Callable[[np.ndarray], np.ndarray] | None' = None) -> "'SDF'"` — Add noise (or func(points)->(N,)) to the surface: bumps, rock roughness, bark.
- `paint(self, color=None, material=None, bone=None) -> "'SDF'"` — Set the color/material/bone of every leaf of this shape.
- `to_mesh(self, resolution: 'int' = 96, bounds=None, padding: 'float' = 0.05, smooth: 'bool' = True, name: 'str' = 'SDFMesh', project_iterations: 'int' = 2, colors: 'bool | None' = None, default_color='#cccccc') -> 'Mesh'` — Polygonize with naive surface nets (quad mesh).

### class `Sphere(center=(0, 0, 0), radius: 'float' = 0.5, **kw)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Ellipsoid(center=(0, 0, 0), radii=(0.5, 0.3, 0.4), **kw)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Box(center=(0, 0, 0), size=(1, 1, 1), radius: 'float' = 0.0, **kw)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Capsule(a=(0, 0, 0), b=(0, 1, 0), radius: 'float' = 0.1, **kw)`

Segment a-b with radius r (limbs, fingers, horns, branches).

- `dist(self, p)`
- `bounds(self)`

### class `RoundCone(a=(0, 0, 0), b=(0, 1, 0), r1: 'float' = 0.2, r2: 'float' = 0.1, **kw)`

Tapered capsule from a (radius r1) to b (radius r2): arms, legs, tails, necks, tentacles.

- `dist(self, p)`
- `bounds(self)`

### class `Cylinder(a=(0, 0, 0), b=(0, 1, 0), radius: 'float' = 0.2, rounding: 'float' = 0.0, **kw)`

Capped cylinder between a and b (optionally rounded edges).

- `dist(self, p)`
- `bounds(self)`

### class `Torus(center=(0, 0, 0), major: 'float' = 0.5, minor: 'float' = 0.15, **kw)`

Torus around the Y axis.

- `dist(self, p)`
- `bounds(self)`

### class `Plane(point=(0, 0, 0), normal=(0, 1, 0), extent: 'float' = 10.0, **kw)`

Half-space below the plane through `point` with `normal` (use with intersect to cut).

- `dist(self, p)`
- `bounds(self)`

### class `Custom(func, lo, hi, **kw)`

Wrap any function f(points)->(N,) as an SDF (you provide the bounds).

- `dist(self, p)`
- `bounds(self)`

### class `Union(children)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `SmoothUnion(a, b, k)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Subtract(a, b, k)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `leaf_dists(self, p)` — (N, L) distance of p to every leaf primitive (respecting transforms).
- `bounds(self)`

### class `Intersect(a, b, k)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Transformed(child, matrix)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `leaf_dists(self, p)` — (N, L) distance of p to every leaf primitive (respecting transforms).
- `bounds(self)`

### class `MirrorX(child)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `leaf_dists(self, p)` — (N, L) distance of p to every leaf primitive (respecting transforms).
- `bounds(self)`

### class `Rounded(child, r)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Shell(child, t)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### class `Displaced(child, amount, scale, octaves, seed, func)`

Base class: callable mapping points (N,3) -> signed distances (N,).

- `dist(self, p)`
- `bounds(self)`

### `sphere(center=(0, 0, 0), radius=0.5, **kw) -> 'SDF'`

### `ellipsoid(center=(0, 0, 0), radii=(0.5, 0.3, 0.4), **kw) -> 'SDF'`

### `box(center=(0, 0, 0), size=(1, 1, 1), radius=0.0, **kw) -> 'SDF'`

### `capsule(a=(0, 0, 0), b=(0, 1, 0), radius=0.1, **kw) -> 'SDF'`

### `round_cone(a=(0, 0, 0), b=(0, 1, 0), r1=0.2, r2=0.1, **kw) -> 'SDF'`

### `cylinder(a=(0, 0, 0), b=(0, 1, 0), radius=0.2, rounding=0.0, **kw) -> 'SDF'`

### `torus(center=(0, 0, 0), major=0.5, minor=0.15, **kw) -> 'SDF'`

### `plane(point=(0, 0, 0), normal=(0, 1, 0), **kw) -> 'SDF'`

### `custom(func, lo, hi, **kw) -> 'SDF'`

### `union(*shapes: 'SDF') -> 'SDF'`

### `smooth_union(*shapes: 'SDF', k: 'float' = 0.05) -> 'SDF'`

### `chain(points, radii, k: 'float' = 0.02, **kw) -> 'SDF'`

Smoothly joined round-cones through a list of points (tails, tentacles, spines, snakes).

### `surface_nets(f: 'SDF', lo: 'np.ndarray', h: 'float', n: 'np.ndarray', name: 'str' = 'SDFMesh', project_iterations: 'int' = 2, chunk: 'int' = 2000000) -> 'Mesh'`

## Sculpting (pyram3d.sculpt)

Digital sculpting brushes for dense meshes (SDF meshes or subdivided meshes).

Each brush acts on vertices within `radius` of a center point with a smooth
falloff. Strokes apply a brush repeatedly along a path. `symmetry="x"` mirrors
the stroke across x=0 (characters, faces, vehicles).

    m = p3.sphere(0.5, 64, 32)
    sculpt.draw(m, center=(0, 0.3, 0.4), radius=0.15, strength=0.05)   # bump
    sculpt.grab(m, center=(0, -0.5, 0), radius=0.3, offset=(0, -0.2, 0)) # pull
    sculpt.stroke(m, "crease", path=[(..), (..)], radius=0.05, strength=0.01)

### `draw(mesh: 'Mesh', center, radius: 'float' = 0.1, strength: 'float' = 0.02, falloff: 'str' = 'smooth', symmetry: 'str | None' = None) -> 'Mesh'`

Raise (strength > 0) or carve (< 0) the surface along its normals.

### `inflate(mesh: 'Mesh', center, radius: 'float' = 0.1, strength: 'float' = 0.02, falloff: 'str' = 'smooth', symmetry: 'str | None' = None) -> 'Mesh'`

Push each vertex along its own normal (puffs up / deflates with negative strength).

### `grab(mesh: 'Mesh', center, radius: 'float' = 0.2, offset=(0, 0.1, 0), falloff: 'str' = 'smooth', symmetry: 'str | None' = None) -> 'Mesh'`

Move a region by `offset` (pull a nose, stretch a snout, bend an ear).

### `smooth(mesh: 'Mesh', center, radius: 'float' = 0.1, strength: 'float' = 0.5, iterations: 'int' = 3, symmetry: 'str | None' = None) -> 'Mesh'`

Relax the surface locally.

### `flatten(mesh: 'Mesh', center, radius: 'float' = 0.1, strength: 'float' = 1.0, symmetry: 'str | None' = None) -> 'Mesh'`

Pull vertices toward the average plane of the region.

### `pinch(mesh: 'Mesh', center, radius: 'float' = 0.1, strength: 'float' = 0.5, symmetry: 'str | None' = None) -> 'Mesh'`

Pull vertices toward the brush center (sharpens creases and ridges).

### `crease(mesh: 'Mesh', center, radius: 'float' = 0.05, strength: 'float' = 0.01, symmetry: 'str | None' = None) -> 'Mesh'`

Carve a sharp groove (wrinkles, lips, eyelids, panel lines).

### `noise(mesh: 'Mesh', center, radius: 'float' = 0.2, strength: 'float' = 0.01, scale: 'float' = 20.0, seed: 'int' = 0, symmetry: 'str | None' = None) -> 'Mesh'`

Local noisy displacement (skin pores, rock grit).

### `stroke(mesh: 'Mesh', brush: 'str', path: 'Sequence[Sequence[float]]', radius: 'float' = 0.05, spacing: 'float' = 0.5, **kw) -> 'Mesh'`

Apply a brush repeatedly along a path. spacing is relative to radius.

### class `Brush(kind: 'str' = 'draw', radius: 'float' = 0.1, strength: 'float' = 0.02, symmetry: 'str | None' = None) -> None`

Reusable brush settings: Brush("draw", radius=0.1, strength=0.02, symmetry="x").apply(mesh, point).

- `apply(self, mesh: 'Mesh', center, **kw) -> 'Mesh'`
- `stroke(self, mesh: 'Mesh', path, **kw) -> 'Mesh'`

## Materials & textures (pyram3d.material)

PBR materials (glTF metallic-roughness model) and procedural textures.

Colors given by the user are sRGB ('#ff8800', (1, 0.5, 0) or 0-255 tuples);
they are converted to linear internally, as glTF requires.

### class `Texture(image, name: 'str' = 'Texture', srgb: 'bool' = True, wrap: 'str' = 'repeat', filter: 'str' = 'linear')`

An RGBA image (H, W, 4 uint8) used by materials.

`srgb` marks color data (base color, emissive). Data maps (normal,
roughness/metal, occlusion) must be linear (srgb=False).

- `load(cls, path, srgb: 'bool' = True, **kw) -> "'Texture'"`
- `from_function(cls, func: 'Callable[[np.ndarray, np.ndarray], np.ndarray]', size: 'int' = 512, name: 'str' = 'Procedural', srgb: 'bool' = True, **kw) -> "'Texture'"` — func(u, v) with u, v arrays in [0, 1) -> (H, W, 3|4) or (H, W) values in [0, 1].
- `sample(self, uv: 'np.ndarray') -> 'np.ndarray'` — Bilinear sample (N, 2) uvs -> (N, 4) floats in [0, 1] (raw texel values).

### class `Material(name: 'str' = 'Material', base_color: 'object' = '#cccccc', metallic: 'float' = 0.0, roughness: 'float' = 0.5, emissive: 'object' = '#000000', emissive_strength: 'float' = 1.0, alpha: 'float' = 1.0, alpha_mode: 'str | None' = None, alpha_cutoff: 'float' = 0.5, double_sided: 'bool' = False, unlit: 'bool' = False, base_color_texture: 'Texture | None' = None, normal_texture: 'Texture | None' = None, normal_scale: 'float' = 1.0, metallic_roughness_texture: 'Texture | None' = None, occlusion_texture: 'Texture | None' = None, emissive_texture: 'Texture | None' = None, extras: 'dict' = <factory>) -> None`

glTF PBR metallic-roughness material.

base_color:  sRGB color (hex or tuple), multiplied with base_color_texture
metallic:    0 = dielectric (plastic, wood, skin), 1 = metal
roughness:   0 = mirror-like, 1 = fully diffuse
emissive:    sRGB glow color; emissive_strength > 1 for HDR glow
alpha:       < 1 makes the material transparent (alpha_mode 'BLEND')
unlit:       flat, lighting-independent shading (stylized / toon)

- `Material.base_color_linear` (property)
- `Material.emissive_linear` (property)
- `Material.resolved_alpha_mode` (property)
- `textures(self) -> 'list[tuple[str, Texture]]'`
- `copy(self, **changes) -> "'Material'"`

### class `textures()`

Procedural texture library. All generators tile seamlessly where noted.

- `checker(color1='#ffffff', color2='#222222', squares: 'int' = 8, size: 'int' = 512) -> 'Texture'`
- `noise(color1='#000000', color2='#ffffff', scale: 'float' = 8.0, octaves: 'int' = 5, seed: 'int' = 0, size: 'int' = 512, contrast: 'float' = 1.0) -> 'Texture'` — Tileable fBm noise.
- `wood(light='#b8875a', dark='#8c5d36', rings: 'float' = 7.0, distortion: 'float' = 0.8, seed: 'int' = 0, size: 'int' = 512) -> 'Texture'` — Wood planks/grain running along V.
- `marble(base='#f2f0eb', vein='#5a5a66', scale: 'float' = 3.0, sharpness: 'float' = 4.0, seed: 'int' = 0, size: 'int' = 512) -> 'Texture'`
- `bricks(brick='#9c4a35', mortar='#cfc6b8', rows: 'int' = 8, cols: 'int' = 4, mortar_width: 'float' = 0.06, variation: 'float' = 0.15, seed: 'int' = 0, size: 'int' = 512) -> 'Texture'`
- `tiles(color='#e8e8e8', grout='#8a8a8a', count: 'int' = 4, grout_width: 'float' = 0.04, size: 'int' = 512) -> 'Texture'`
- `gradient(color1='#000000', color2='#ffffff', direction: 'str' = 'v', size: 'int' = 256) -> 'Texture'`
- `cells(color1='#2a5d2a', color2='#9fd17a', scale: 'float' = 8.0, seed: 'int' = 0, size: 'int' = 512) -> 'Texture'` — Voronoi cells (scales, stones, leather, cracked ground).
- `stripes(color1='#ffffff', color2='#cc2222', count: 'int' = 8, direction: 'str' = 'u', softness: 'float' = 0.02, size: 'int' = 256) -> 'Texture'`
- `normal_from_height(height: 'np.ndarray | Texture', strength: 'float' = 2.0) -> 'Texture'` — Convert a height map (or the luminance of a texture) to a tangent-space normal map.
- `roughness_metal(roughness: 'np.ndarray | float', metallic: 'np.ndarray | float' = 0.0, size: 'int' = 256) -> 'Texture'`

### class `presets()`

Ready-made materials. Every function accepts overrides as keyword args.

- `plastic(color='#d03030', **kw)`
- `rubber(color='#1c1c1c', **kw)`
- `metal(color='#c0c0c8', roughness: 'float' = 0.3, **kw)`
- `gold(**kw)`
- `copper(**kw)`
- `chrome(**kw)`
- `glass(color='#cfe8ff', alpha: 'float' = 0.25, **kw)`
- `emissive(color='#ffdd88', strength: 'float' = 3.0, **kw)`
- `wood(**kw)`
- `marble(**kw)`
- `bricks(**kw)`
- `skin(color='#e8b896', **kw)`
- `fabric(color='#3a5fa0', **kw)`
- `foliage(color='#4f8a35', **kw)`
- `bark(**kw)`
- `stone(color='#8a8680', **kw)`
- `water(**kw)`
- `toon(color='#ff8844', **kw)`
- `vertex_color(**kw)` — White material so vertex colors show through unchanged.

## Scene graph (pyram3d.scene)

Scene graph: nodes with transforms, meshes, lights, cameras, skeletons.

### class `Light(type: 'str' = 'directional', color: 'object' = '#ffffff', intensity: 'float' = 3.0, range: 'float | None' = None, inner_cone: 'float' = 20.0, outer_cone: 'float' = 35.0, cast_shadows: 'bool' = True, name: 'str' = 'Light') -> None`

Punctual light (glTF KHR_lights_punctual). Lights shine along their node's local -Z.

intensity: directional/sun in lux (3-10 typical); point/spot in candela (10-1000).

- `Light.color_linear` (property)

### class `Camera(type: 'str' = 'perspective', yfov: 'float' = 45.0, aspect: 'float | None' = None, znear: 'float' = 0.05, zfar: 'float' = 1000.0, ortho_height: 'float' = 2.0, name: 'str' = 'Camera') -> None`

Camera looking along its node's local -Z.


### class `Node(name: 'str' = 'Node', mesh: 'Mesh | None' = None, translation=(0, 0, 0), rotation=None, scale=(1, 1, 1), light: 'Light | None' = None, camera: 'Camera | None' = None)`

- `add(self, obj=None, name: 'str | None' = None, position=None, rotation=None, scale=None, **kw) -> "'Node'"` — Add a child. obj may be a Node, Mesh, Light, Camera, Skeleton or None (empty).
- `remove(self, child: "'Node'")`
- `walk(self) -> "Iterator['Node']"`
- `find(self, name: 'str') -> "'Node | None'"`
- `Node.position` (property)
- `set_rotation(self, rotation) -> "'Node'"` — Euler degrees (x, y, z) or quaternion (x, y, z, w).
- `rotate(self, rotation) -> "'Node'"` — Apply an additional rotation (in parent space) on top of the current one.
- `look_at(self, target, up=(0, 1, 0)) -> "'Node'"` — Point the local -Z axis at `target` (world space). Used for cameras and lights.
- `local_matrix(self) -> 'np.ndarray'`
- `world_matrix(self) -> 'np.ndarray'`
- `world_position(self) -> 'np.ndarray'`

### class `Scene(name: 'str' = 'Scene', fps: 'int' = 30)`

A complete 3D scene. Add meshes, lights, cameras and animations, then export.

- `add(self, obj=None, name: 'str | None' = None, position=None, rotation=None, scale=None, parent: 'Node | None' = None, **kw) -> 'Node'` — Add a Mesh/Node/Light/Camera/Skeleton (optionally under `parent`). Returns the Node.
- `add_light(self, type: 'str' = 'directional', position=(4, 8, 6), target=(0, 0, 0), color='#ffffff', intensity: 'float | None' = None, name: 'str | None' = None, **kw) -> 'Node'`
- `add_camera(self, position=(3, 2, 5), target=(0, 0.5, 0), yfov: 'float' = 45.0, name: 'str' = 'Camera', **kw) -> 'Node'`
- `add_default_lights(self, strength: 'float' = 1.0) -> 'list[Node]'` — Three-point lighting rig (key, fill, rim).
- `animation(self, name: 'str' = 'Action')` — Create (or get) an animation clip by name.
- `Scene.nodes` (property)
- `walk(self) -> 'Iterator[Node]'`
- `find(self, name: 'str') -> 'Node | None'`
- `mesh_nodes(self) -> 'list[Node]'`
- `bounds(self, time: 'float | None' = None) -> 'tuple[np.ndarray, np.ndarray]'`
- `materials(self) -> 'list'`
- `describe(self) -> 'str'` — Human/AI-readable summary of the scene tree, sizes and animations.
- `export(self, path, **kw) -> 'Path'` — Export by file extension: .glb, .gltf, .fbx, .obj
- `export_all(self, directory, basename: 'str | None' = None, formats=('glb', 'gltf', 'fbx')) -> 'list[Path]'` — Export the scene in several formats at once.
- `render(self, path=None, *args, **kw)` — Render a preview image (PNG). See pyram3d.render.render for options.
- `render_views(self, path=None, **kw)` — Contact sheet: front, right, back, 3/4, top, left.

## Animation (pyram3d.animation)

Keyframe animation of node transforms and morph-target weights.

    anim = scene.animation("Walk")
    anim.key(node, "position", 0.0, (0, 0, 0))
    anim.key(node, "position", 1.0, (0, 0, 2))
    anim.key(node, "rotation", 0.5, (0, 90, 0))   # Euler degrees or quaternion
    anim.keys(node, "scale", {0: 1, 0.5: 1.2, 1: 1}, interpolation="SMOOTH")

Interpolation per channel: "LINEAR", "STEP" or "SMOOTH" (smooth ease through
keys; exported as densely sampled LINEAR keys for maximum compatibility).
Bones of a Skeleton are ordinary nodes, so the same API animates rigs.

### class `Channel(node, path: 'str', interpolation: 'str' = 'LINEAR')`

- `set_key(self, time: 'float', value: 'np.ndarray')`
- `Channel.start` (property)
- `Channel.end` (property)
- `sample(self, t: 'float') -> 'np.ndarray'`
- `baked(self, fps: 'float') -> 'tuple[np.ndarray, np.ndarray]'` — Sample at a fixed rate -> (times, values) suitable for LINEAR playback.

### class `Animation(name: 'str' = 'Action', fps: 'int' = 30)`

An animation clip (glTF animation / FBX AnimStack).

- `channel(self, node, path: 'str', interpolation: 'str | None' = None) -> 'Channel'`
- `key(self, node, path: 'str', time: 'float', value=None, interpolation: 'str | None' = None) -> "'Animation'"` — Insert one keyframe. value=None keys the node's current value.
- `keys(self, node, path: 'str', keyframes: 'Mapping[float, object]', interpolation: 'str | None' = None) -> "'Animation'"` — Insert several keyframes {time: value}.
- `key_pose(self, time: 'float', pose: 'Mapping', interpolation: 'str | None' = None) -> "'Animation'"` — Key many nodes at once: pose = {node: {"rotation": (x, y, z), "position": ...}}.
- `Animation.duration` (property)
- `Animation.start` (property)
- `apply(self, t: 'float')` — Pose the scene nodes at time t (modifies node transforms).
- `snapshot(self)` — Remember current transforms of animated nodes (restore with restore()).
- `restore(snap)`

### `spin(anim: 'Animation', node, duration: 'float' = 2.0, axis: 'str' = 'y', turns: 'float' = 1.0) -> 'Animation'`

Continuous rotation (looping turntable).

### `bounce(anim: 'Animation', node, height: 'float' = 0.5, duration: 'float' = 1.0, squash: 'float' = 0.15) -> 'Animation'`

Bouncing ball style hop with squash and stretch.

### `orbit(anim: 'Animation', node, center=(0, 0, 0), radius: 'float' = 5.0, height: 'float' = 2.0, duration: 'float' = 4.0, look_at=True, steps: 'int' = 32) -> 'Animation'`

Move a node (e.g. a camera) around a circle, optionally looking at the center.

## Rigging (pyram3d.armature)

Skeletons (armatures/rigs), automatic skin weights and linear-blend skinning.

Bones are scene Nodes, so they are animated with the regular Animation API.
Bones are created in the REST pose with identity rotation: a bone's local axes
are aligned with the world axes when the rig is at rest. That makes rotations
easy to reason about:

    rotate about X -> swing forward/back (pitch)
    rotate about Y -> twist / turn (yaw)
    rotate about Z -> raise/lower sideways (roll)

Characters face +Z. Their LEFT side is +X, RIGHT side is -X.

    sk = Skeleton("Rig")
    hips = sk.add_bone("Hips", head=(0, 1.0, 0), tail=(0, 1.15, 0))
    leg = sk.add_bone("LeftUpperLeg", (0.1, 0.95, 0), (0.1, 0.5, 0), parent=hips)
    sk.bind(mesh)                    # automatic weights
    scene.add(sk); scene.add(mesh_node_with_skin)

### class `Skeleton(name: 'str' = 'Armature', position=(0, 0, 0))`

- `add_bone(self, name: 'str', head, tail=None, parent: 'Node | str | None' = None) -> 'Node'` — Add a bone from `head` (joint position) to `tail`, in armature (rest) space.
- `add_chain(self, names, points, parent: 'Node | str | None' = None) -> 'list[Node]'` — Add consecutive bones through points (len(points) == len(names) + 1).
- `bone(self, name: 'str') -> 'Node'`
- `head(self, bone: 'Node | str') -> 'np.ndarray'`
- `tail(self, bone: 'Node | str') -> 'np.ndarray'`
- `index(self, bone: 'Node | str') -> 'int'`
- `reset_pose(self)` — Return every bone to the rest pose.
- `bind_root_matrix(self) -> 'np.ndarray'` — Armature root transform the mesh was bound with (moving the root later moves the character).
- `rest_world_matrices(self) -> 'np.ndarray'`
- `inverse_bind_matrices(self) -> 'np.ndarray'`
- `joint_matrices(self) -> 'np.ndarray'` — Current skinning matrices (world-space pose @ inverse bind).
- `pose_vertices(self, mesh: 'Mesh') -> 'np.ndarray'` — Linear blend skinning of mesh vertices for the current pose.
- `bind(self, mesh: 'Mesh', rigid: 'bool' = False, smooth: 'int' = 4, falloff: 'float' = 4.0, max_influences: 'int' = 4, exclude: 'list[str] | None' = None, bone_filter=None) -> 'Mesh'` — Compute automatic skin weights for `mesh` (vertices in world/rest space).
- `attach(self, node: 'Node', bone: 'Node | str') -> 'Node'` — Rigidly attach an object (hat, sword, glasses) to a bone, keeping its current world placement.
- `describe(self) -> 'str'`

### `skinned_node(mesh: 'Mesh', skeleton: 'Skeleton', name: 'str | None' = None) -> 'Node'`

Create the scene node for a skinned mesh (keep it at identity transform).

## Terrain (pyram3d.terrain)

Generate a terrain grid in the XZ plane centered on the origin.

size      : meters (scalar or (x, z))
height    : max elevation in meters
scale     : noise frequency (smaller = broader hills)
ridged    : 0..1 mix of ridged noise (mountain crests)
island    : 0..1 radial falloff to make an island / bowl
terraces  : number of terrace steps (0 = off)
erosion   : smoothing passes that soften valleys (cheap erosion)
heightmap : optional (H, W) array in [0, 1] instead of noise
height_func: optional f(x, z) -> heights (meters), overrides noise
colors    : True = automatic grass/rock/snow/sand vertex colors; dict to customize
            {"sand": "#..", "grass": "#..", "rock": "#..", "snow": "#..", "water_level": 0.0}

## Physics & effects (pyram3d.physics)

Simple physics simulations baked into keyframes / shape keys, so they play in any engine.

  RigidWorld   falling, bouncing, tumbling spheres and boxes (ground + sphere contacts)
  cloth_drape  drop a cloth onto colliders (static result: tablecloths, flags at rest, blankets)
  cloth_simulate  the same, baked as an animation (shape keys + weight keys)
  particles    looping effects: fire, smoke, sparks, fountain, snow, rain, magic

All units are meters / seconds. Results are regular nodes, keyframes and shape keys,
which export to glTF and FBX.

### class `Body(node: 'Node', shape: 'str', half: 'np.ndarray', mass: 'float', vel: 'np.ndarray', ang: 'np.ndarray', fixed: 'bool' = False, inv_inertia: 'np.ndarray' = <factory>) -> None`

Body(node: 'Node', shape: 'str', half: 'np.ndarray', mass: 'float', vel: 'np.ndarray', ang: 'np.ndarray', fixed: 'bool' = False, inv_inertia: 'np.ndarray' = <factory>)

- `Body.radius` (property)

### class `RigidWorld(gravity: 'float' = -9.81, ground: 'float | None' = 0.0, restitution: 'float' = 0.35, friction: 'float' = 0.5)`

Minimal rigid-body solver (impulses + position correction).

world = p3.physics.RigidWorld()
for i in range(10):
    n = scene.add(p3.box(0.3), position=(0, 1 + i * 0.4, 0), rotation=(i * 20, i * 7, 0))
    world.add(n, "box")
world.simulate(scene.animation("Fall"), duration=3)

- `add(self, node: 'Node', shape: 'str' = 'auto', mass: 'float' = 1.0, velocity=(0, 0, 0), angular_velocity=(0, 0, 0), fixed: 'bool' = False, size=None) -> 'Body'` — shape: "sphere" | "box" | "auto" (from the node's mesh bounds).
- `step(self, dt: 'float')`
- `simulate(self, anim, duration: 'float' = 3.0, fps: 'int' = 30, substeps: 'int' = 10) -> "'RigidWorld'"` — Run the simulation and bake position/rotation keys into `anim` (start pose at t=0).

### `cloth_drape(mesh: 'Mesh', colliders=(), pinned=None, steps: 'int' = 300, dt: 'float' = 0.016666666666666666, gravity: 'float' = -9.81, iterations: 'int' = 8, thickness: 'float' = 0.01, ground: 'float | None' = 0.0, damping: 'float' = 0.02) -> 'Mesh'`

Simulate a cloth (a subdivided plane works best) falling onto colliders; returns the draped mesh.

colliders: SDF objects (pyram3d.sdf shapes) the cloth must not penetrate.
pinned   : vertex indices that stay fixed (flag/curtain attachment points).

### `cloth_simulate(mesh: 'Mesh', anim, node: 'Node', colliders=(), pinned=None, duration: 'float' = 2.0, keyframes: 'int' = 12, steps_per_second: 'int' = 120, **kw) -> 'Mesh'`

Bake a cloth simulation as shape keys + morph-weight keyframes on `node` (exports to glTF and FBX).

### `particles(scene, anim, kind: 'str' = 'fire', position=(0, 0, 0), duration: 'float' = 2.0, seed: 'int' = 0, fps: 'int' = 15, parent: 'Node | None' = None, **overrides) -> 'Node'`

Create a looping particle effect as animated nodes (one shared mesh, instanced).

kind: fire | smoke | sparks | fountain | snow | rain | magic. Any preset key can be
overridden (count, life, speed, spread, gravity, size, color, emissive, radius, alpha).

## Preview rendering (pyram3d.render)

Render the scene and optionally save a PNG. Returns the (H, W, 3) uint8 image.

view   : "front" | "back" | "left" | "right" | "top" | "three_quarter" | direction vector
camera : a camera Node from the scene (overrides view)
target/distance: override auto-framing (look at point, distance from it)
time   : animation time in seconds (poses rigs / animated nodes)
bloom  : glow strength around bright/emissive pixels (0 = off)
fog    : atmospheric fog density per meter (e.g. 0.02 for landscapes)
vignette: darkening of the image corners (0 = off)
exposure: brightness multiplier before tone mapping

## Saving (pyram3d.project)

One-call output helper used at the end of build scripts.

### `save(scene, path, formats=('glb', 'gltf', 'fbx'), preview: 'bool' = True, views: 'bool' = True, viewer: 'bool' = True, frames: 'bool' = True, size: 'int' = 900) -> 'dict'`

Export a scene in every format plus review images and an HTML viewer.

save(scene, "projects/chair/exports/chair") writes:
  chair.glb, chair.gltf, chair.fbx      (models)
  chair_preview.png                     (3/4 beauty render)
  chair_views.png                       (front/right/back/3-4/top/left sheet)
  chair_<Animation>.png                 (filmstrip per animation)
  chair_viewer.html                     (interactive three.js page, model embedded)
Returns {kind: Path}.

## HTML viewer (pyram3d.viewer)

Self-contained HTML 3D viewer (three.js) for sharing models, e.g. as a phone-friendly artifact page.

### `write_viewer(source, path, title: 'str | None' = None, files: 'list[str] | None' = None) -> 'Path'`

Write an interactive HTML viewer page for a Scene or an existing .glb file.

The model is embedded (base64), so the page is a single file you can
publish/share. `files` lists exported file names to mention on the page.

## Import (pyram3d.importer)

Import existing models so they can be inspected, edited, rendered and re-exported.

Supported: .glb / .gltf (meshes, materials, textures, node hierarchy, lights,
cameras, node animations) and .obj (geometry + UVs).

### `load(path) -> 'Scene'`

Load a model file into a new Scene.

### `load_gltf(path) -> 'Scene'`

### `load_obj(path) -> 'Scene'`

## Noise (pyram3d.noise)

Vectorized procedural noise (Perlin gradient noise, fBm, ridged, Worley).

### `perlin3(p, seed: 'int' = 0, period=None) -> 'np.ndarray'`

Classic Perlin noise for points p (..., 3). Output roughly in [-1, 1].

period: optional (px, py, pz) integer lattice periods -> seamlessly tiling noise.

### `perlin2(x, y, seed: 'int' = 0) -> 'np.ndarray'`

### `fbm3(p, octaves: 'int' = 5, lacunarity: 'float' = 2.0, gain: 'float' = 0.5, seed: 'int' = 0) -> 'np.ndarray'`

Fractal Brownian motion: sum of octaves of Perlin noise. Output roughly in [-1, 1].

### `tileable_fbm2(u, v, scale: 'int' = 8, octaves: 'int' = 5, gain: 'float' = 0.5, seed: 'int' = 0) -> 'np.ndarray'`

fBm over the unit square that tiles seamlessly (u, v in [0, 1)).

### `ridged3(p, octaves: 'int' = 5, lacunarity: 'float' = 2.0, gain: 'float' = 0.5, seed: 'int' = 0) -> 'np.ndarray'`

Ridged multifractal noise in [0, 1]: sharp crests (mountain ridges, veins).

### `fbm2(x, y, octaves: 'int' = 5, lacunarity: 'float' = 2.0, gain: 'float' = 0.5, seed: 'int' = 0) -> 'np.ndarray'`

### `worley3(p, seed: 'int' = 0, jitter: 'float' = 1.0) -> 'tuple[np.ndarray, np.ndarray]'`

Cellular (Worley) noise. Returns (F1 distance, F2 distance) for points (..., 3).

### `random_unit_vectors(n: 'int', seed: 'int' = 0) -> 'np.ndarray'`

## Math (pyram3d.math3d)

Vector, quaternion and matrix helpers.

Conventions (same as glTF):
  * Right-handed, +Y up, +Z towards the viewer ("front"), units in meters.
  * Quaternions are stored as (x, y, z, w).
  * Matrices are 4x4 numpy arrays using the column-vector convention
    (p' = M @ p), translation lives in M[:3, 3].
  * Euler angles are in DEGREES, order "XYZ" = rotate about X first, then Y,
    then Z (extrinsic), i.e. R = Rz @ Ry @ Rx. This matches FBX and Blender.

### `vec3(v) -> 'np.ndarray'`

### `normalize(v, axis=-1) -> 'np.ndarray'`

### `quat_identity() -> 'np.ndarray'`

### `quat_from_axis_angle(axis, angle_deg: 'float') -> 'np.ndarray'`

### `quat_mul(a, b) -> 'np.ndarray'`

Hamilton product a*b (apply b first, then a).

### `quat_conj(q) -> 'np.ndarray'`

### `quat_normalize(q) -> 'np.ndarray'`

### `quat_from_euler(euler_deg, order: 'str' = 'XYZ') -> 'np.ndarray'`

Euler angles in degrees -> quaternion. order 'XYZ' means X applied first.

### `quat_to_matrix3(q) -> 'np.ndarray'`

### `matrix3_to_quat(m) -> 'np.ndarray'`

### `quat_to_euler(q, order: 'str' = 'XYZ') -> 'np.ndarray'`

Quaternion -> Euler degrees for order XYZ (R = Rz Ry Rx).

### `quat_rotate(q, v) -> 'np.ndarray'`

Rotate vector(s) v (...,3) by quaternion q.

### `quat_slerp(a, b, t: 'float') -> 'np.ndarray'`

### `quat_look_rotation(forward, up=(0, 1, 0)) -> 'np.ndarray'`

Rotation that makes the local -Z axis point along `forward` (glTF camera/light convention).

### `quat_between(a, b) -> 'np.ndarray'`

Shortest rotation taking direction a to direction b.

### `as_quat(rotation) -> 'np.ndarray'`

Accept a quaternion (4 values) or Euler degrees (3 values).

### `translation_matrix(t) -> 'np.ndarray'`

### `scale_matrix(s) -> 'np.ndarray'`

### `rotation_matrix(rotation) -> 'np.ndarray'`

### `trs_matrix(t=(0, 0, 0), r=None, s=(1, 1, 1)) -> 'np.ndarray'`

### `decompose(m) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'`

### `transform_points(m, pts) -> 'np.ndarray'`

### `transform_dirs(m, dirs) -> 'np.ndarray'`

### `look_at_matrix(eye, target, up=(0, 1, 0)) -> 'np.ndarray'`

Camera-to-world matrix: camera at `eye` looking at `target` (-Z forward).

### `perspective_matrix(yfov_deg: 'float', aspect: 'float', znear: 'float', zfar: 'float') -> 'np.ndarray'`

### `orthographic_matrix(xmag: 'float', ymag: 'float', znear: 'float', zfar: 'float') -> 'np.ndarray'`

### `lerp(a, b, t)`

### `smoothstep(e0, e1, x)`

### `bezier(points: 'Sequence[Sequence[float]]', n: 'int' = 32) -> 'np.ndarray'`

Sample a Bezier curve of any degree. Returns (n, dim) points.

### `catmull_rom(points: 'Sequence[Sequence[float]]', samples_per_segment: 'int' = 8, closed: 'bool' = False) -> 'np.ndarray'`

Smooth curve passing through all control points.

### `frames_along(path: 'np.ndarray', up_hint=(0, 1, 0)) -> 'tuple[np.ndarray, np.ndarray, np.ndarray]'`

Parallel-transport frames (tangent, normal, binormal) along a polyline.

### `as_color(c, alpha: 'float | None' = None) -> 'np.ndarray'`

Accept '#rrggbb', '#rrggbbaa', (r,g,b) or (r,g,b,a) in 0..1 -> RGBA float array.

### `srgb_to_linear(c)`

### `linear_to_srgb(c)`

## Generators: characters

Rigged characters: humanoid and quadruped templates + animation cycles.

    from pyram3d.generators import character as ch
    hero = ch.humanoid(height=1.75, shirt="#3b6fb6")
    hero.add_to(scene)
    ch.walk_cycle(scene.animation("Walk"), hero.skeleton)
    ch.run_cycle(scene.animation("Run"), hero.skeleton)

Bone names follow the Unity Humanoid convention (Hips, Spine, Chest, Neck,
Head, LeftUpperArm, LeftLowerArm, LeftHand, LeftUpperLeg, LeftLowerLeg,
LeftFoot, LeftToes, ...), so Unity/Unreal/Godot retargeting maps automatically.

The rest pose is a T-pose (arms along X), the character faces +Z and its
left side is +X.

### class `Character(mesh: 'Mesh', skeleton: 'Skeleton', extras: 'list' = <factory>, name: 'str' = 'Character', node: 'Node | None' = None) -> None`

Character(mesh: 'Mesh', skeleton: 'Skeleton', extras: 'list' = <factory>, name: 'str' = 'Character', node: 'Node | None' = None)

- `add_to(self, scene, position=None) -> 'Node'` — Add skeleton + skinned body (+ attachments like eyes) to the scene.

### `humanoid_skeleton(height: 'float' = 1.8, name: 'str' = 'Rig') -> 'Skeleton'`

### `humanoid(height: 'float' = 1.8, build: 'float' = 1.0, skin='#e2b08f', shirt='#3b6fb6', pants='#2f3542', shoes='#3a2a20', hair='#3b2416', eyes: 'bool' = True, resolution: 'int' = 120, head_scale: 'float' = 1.0, name: 'str' = 'Humanoid', material: 'Material | None' = None) -> 'Character'`

Stylized rigged humanoid in T-pose built from SDF parts.

build      : 0.8 slim .. 1.0 average .. 1.3 heavy
head_scale : >1 for cartoon / chibi proportions
Colors are vertex colors (single material) -> lightweight, engine-friendly.

### `walk_cycle(anim, sk: 'Skeleton', duration: 'float' = 1.0, stride: 'float' = 1.0)`

Natural in-place walk. stride scales the leg/arm swing.

### `run_cycle(anim, sk: 'Skeleton', duration: 'float' = 0.66, stride: 'float' = 1.0)`

### `idle(anim, sk: 'Skeleton', duration: 'float' = 3.0)`

Breathing idle with subtle weight shift.

### `jump(anim, sk: 'Skeleton', duration: 'float' = 1.2, height: 'float' = 0.45)`

Anticipation, take-off, airborne tuck, landing, recovery (not looping).

### `wave(anim, sk: 'Skeleton', duration: 'float' = 2.0, side: 'str' = 'Right')`

Raise one hand and wave.

### `t_pose(anim, sk: 'Skeleton', time: 'float' = 0.0)`

### `quadruped_skeleton(length: 'float' = 1.0, height: 'float' = 0.6, name: 'str' = 'QuadRig') -> 'Skeleton'`

Four-legged rig facing +Z (dog/cat/horse/deer proportions via length/height).

### `quadruped(length: 'float' = 1.0, height: 'float' = 0.6, fur='#b07a45', belly='#e8d2b0', nose='#222222', ears: 'str' = 'pointy', resolution: 'int' = 110, name: 'str' = 'Quadruped', eyes: 'bool' = True) -> 'Character'`

Stylized rigged four-legged animal (dog/fox/cat-like). Tweak length/height/colors.

### `quadruped_walk(anim, sk: 'Skeleton', duration: 'float' = 1.0, stride: 'float' = 1.0)`

Diagonal-pair walk/trot cycle for quadruped_skeleton rigs.

## Generators: nature

Procedural plants, rocks and other natural props.

Every generator is deterministic for a given `seed` and returns a Mesh
(materials already assigned) standing on y = 0 and centered on x = z = 0.

### `tree(height: 'float' = 5.0, trunk_radius: 'float | None' = None, levels: 'int' = 3, branches: 'int' = 5, spread: 'float' = 45.0, leaves: 'str' = 'blobs', leaf_color='#4f8a35', bark=None, foliage=None, leaf_size: 'float' = 1.0, seed: 'int' = 0, segments: 'int' = 8, name: 'str' = 'Tree') -> 'Mesh'`

Deciduous tree with recursive branches.

leaves: "blobs" (stylized low-poly canopy clusters), "cards" (alpha leaf
        quads, more realistic), or "none" (bare winter tree)
spread: branch angle from the parent in degrees

### `leaf_texture(size: 'int' = 256, color='#ffffff') -> 'Texture'`

Leaf silhouette with a central vein, alpha-masked (use with alpha_mode='MASK').

### `pine(height: 'float' = 6.0, layers: 'int' = 5, radius: 'float | None' = None, color='#2f5d34', trunk_color='#5a3d28', segments: 'int' = 10, seed: 'int' = 0, snow: 'bool' = False, name: 'str' = 'Pine') -> 'Mesh'`

Stylized conifer: stacked cones on a trunk.

### `palm(height: 'float' = 6.0, fronds: 'int' = 8, lean: 'float' = 15.0, seed: 'int' = 0, name: 'str' = 'Palm') -> 'Mesh'`

### `bush(size: 'float' = 1.0, color='#4a7d34', seed: 'int' = 0, berries: 'str | None' = None, name: 'str' = 'Bush') -> 'Mesh'`

### `rock(size: 'float' = 1.0, seed: 'int' = 0, roughness: 'float' = 0.25, flatness: 'float' = 0.7, low_poly: 'bool' = True, color='#8a857e', material=None, name: 'str' = 'Rock') -> 'Mesh'`

Boulder. low_poly=True gives a faceted game-art rock; False a smooth, detailed one.

### `grass_patch(size: 'float' = 1.0, blades: 'int' = 120, height: 'float' = 0.25, color='#5f9a3a', seed: 'int' = 0, name: 'str' = 'Grass') -> 'Mesh'`

Clump of grass blades (thin triangles), good for scattering.

### `flower(height: 'float' = 0.4, petals: 'int' = 6, petal_color='#e84a7a', center_color='#f2c230', seed: 'int' = 0, name: 'str' = 'Flower') -> 'Mesh'`

### `mushroom(height: 'float' = 0.25, cap_color='#c8322a', spots: 'bool' = True, seed: 'int' = 0, name: 'str' = 'Mushroom') -> 'Mesh'`

### `cactus(height: 'float' = 1.6, arms: 'int' = 2, color='#4f8f4a', seed: 'int' = 0, name: 'str' = 'Cactus') -> 'Mesh'`

## Generators: architecture

Procedural buildings, houses and architectural props.

Generators return Meshes standing on y = 0, centered on x = z = 0, facing +Z
(the front door is on the +Z side).

### `building(width: 'float' = 12.0, depth: 'float' = 10.0, floors: 'int' = 5, floor_height: 'float' = 3.2, bays: 'int | None' = None, wall='#c9c2b6', frame='#4a4a4a', glass=None, roof: 'str' = 'flat', window_ratio: 'float' = 0.55, door: 'bool' = True, name: 'str' = 'Building') -> 'Mesh'`

Multi-storey building with an inset window in every bay of every floor.

bays: windows per floor along the width (default: one per ~3 m)
roof: "flat" (parapet + rooftop box) | "gable" | "hip"

### `house(width: 'float' = 8.0, depth: 'float' = 7.0, wall='#e8dcc6', roof_color='#8a3a2a', door_color='#5a3b26', chimney: 'bool' = True, porch: 'bool' = True, name: 'str' = 'House') -> 'Mesh'`

Single-storey house with gable roof, door, windows, chimney and porch step.

### `wall_segment(length: 'float' = 4.0, height: 'float' = 2.0, thickness: 'float' = 0.3, material=None, name: 'str' = 'Wall') -> 'Mesh'`

### `fence(length: 'float' = 6.0, height: 'float' = 1.0, posts: 'int | None' = None, color='#c8b28f', name: 'str' = 'Fence') -> 'Mesh'`

### `stairs(steps: 'int' = 8, width: 'float' = 1.2, rise: 'float' = 0.18, run: 'float' = 0.28, material=None, name: 'str' = 'Stairs') -> 'Mesh'`

### `column(height: 'float' = 4.0, radius: 'float' = 0.3, flutes: 'int' = 16, material=None, name: 'str' = 'Column') -> 'Mesh'`

Classical column (base, fluted shaft, capital).

### `street_lamp(height: 'float' = 4.0, name: 'str' = 'StreetLamp') -> 'Mesh'`

### `road(length: 'float' = 20.0, width: 'float' = 7.0, lanes: 'int' = 2, name: 'str' = 'Road') -> 'Mesh'`

## Generators: vehicles

Procedural vehicles. Each returns a Node hierarchy whose wheels are separate
child nodes (named "Wheel_FL", "Wheel_FR", ...) so they can be animated.

Vehicles face +Z and stand on y = 0.

### `wheel(radius: 'float' = 0.34, width: 'float' = 0.24, rim_color='#c8ccd0', spokes: 'int' = 5) -> 'Mesh'`

Tire + rim, axle along X.

### `car(color='#c0392b', length: 'float' = 4.3, width: 'float' = 1.8, height: 'float' = 1.4, style: 'str' = 'sedan', name: 'str' = 'Car') -> 'Node'`

Stylized car. style: "sedan" | "hatchback" | "suv" | "pickup".

### `drive(anim: 'Animation', vehicle: 'Node', distance: 'float' = 10.0, duration: 'float' = 3.0, steps: 'int' = 12)`

Move a vehicle forward along +Z (its facing) and spin its wheels accordingly.

### `boat(length: 'float' = 5.0, color='#e8e2d6', hull_color='#2c4a6e', name: 'str' = 'Boat') -> 'Mesh'`

Small motorboat hull built by lofting cross sections.

### `airplane(span: 'float' = 10.0, color='#e8e8ec', accent='#c0392b', name: 'str' = 'Airplane') -> 'Mesh'`

Stylized propeller airplane facing +Z.

## Generators: props

Everyday props and furniture. Each returns a Mesh standing on y = 0, front = +Z.

### `crate(size: 'float' = 1.0, color='#a8743f', name: 'str' = 'Crate') -> 'Mesh'`

Wooden crate with framed edges.

### `barrel(height: 'float' = 1.0, radius: 'float' = 0.32, name: 'str' = 'Barrel') -> 'Mesh'`

### `table(width: 'float' = 1.4, depth: 'float' = 0.8, height: 'float' = 0.75, material=None, name: 'str' = 'Table') -> 'Mesh'`

### `chair(seat_height: 'float' = 0.45, width: 'float' = 0.45, material=None, cushion=None, name: 'str' = 'Chair') -> 'Mesh'`

### `sofa(width: 'float' = 2.0, color='#5b6d8a', name: 'str' = 'Sofa') -> 'Mesh'`

### `lamp(height: 'float' = 1.5, shade_color='#f2e6c9', name: 'str' = 'FloorLamp') -> 'Mesh'`

### `bookshelf(width: 'float' = 1.0, height: 'float' = 1.8, shelves: 'int' = 5, seed: 'int' = 0, name: 'str' = 'Bookshelf') -> 'Mesh'`

### `sword(length: 'float' = 1.0, blade_color='#d8dde3', grip_color='#4a2c1a', name: 'str' = 'Sword') -> 'Mesh'`

Sword pointing up (+Y), guard at y = grip length.

### `treasure_chest(width: 'float' = 0.8, name: 'str' = 'Chest') -> 'Mesh'`

