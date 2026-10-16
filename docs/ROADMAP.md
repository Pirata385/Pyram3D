# Pyram3D roadmap

## Done (v0.1)
- Mesh modeling: primitives, extrude/inset, Catmull-Clark, mirror, weld, deformers, solidify, lathe, sweep,
  loft, 2D extrusion, UV projections
- SDF organic modeling + booleans (surface nets), sculpt brushes
- PBR materials, procedural seamless textures, normal maps (with tangents in glTF)
- Scene graph, lights, cameras, keyframe animation (linear/step/smooth), morph targets
- Skeletons with automatic weights; humanoid + quadruped rigs with walk/run/idle/jump/wave cycles
- Terrain (noise, island, terraces, flatten), scatter/instancing, water
- Physics baked to keyframes: rigid bodies, cloth drape/simulation, particle effects
- Exporters: glTF/GLB (Khronos-validator clean), FBX 7.4 binary (Blender-verified), OBJ
- Importers: glTF/GLB (incl. skins and animations), OBJ
- CPU preview renderer (realistic/toon/clay/wire/normals, shadows, bloom, fog), contact sheets, filmstrips
- three.js HTML viewer, CLI, generators for nature/architecture/vehicles/props

## Next
1. **Decimation / LOD** (quadric edge collapse) and quad remeshing for SDF meshes
2. **Automatic UV unwrapping** (seams + LSCM) and texture baking (vertex colors / AO → texture)
3. **Bevel** operator with segments on selected edges; loop cuts; knife/boolean on polygon meshes
4. **Facial rig**: shape-key presets (smile, blink, visemes) on generated heads
5. **IK** helpers (two-bone IK, foot planting) for authoring animations; animation retargeting
6. **Better characters**: hands with fingers, clothing layers, hair cards, more body types
7. **Path-traced preview** (progressive, CPU) for realistic stills
8. **Fluids and smoke** (grid-based, baked to flipbook/morph targets or VDB-like volumes)
9. **More formats**: USDZ (AR on iPhone), STL/3MF for 3D printing, PLY
10. **Scene description format** (JSON) so models can be edited declaratively and diffed
