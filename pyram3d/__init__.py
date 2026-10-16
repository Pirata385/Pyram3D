"""Pyram3D - a 3D modeling toolkit designed to be driven by AI agents.

Typical script:

    import pyram3d as p3

    scene = p3.Scene("Chair")
    wood = p3.presets.wood()
    seat = p3.rounded_box((0.5, 0.06, 0.5), 0.01).set_material(wood)
    scene.add(seat, position=(0, 0.45, 0))
    scene.add_default_lights()
    scene.render("out/chair.png")          # look at it, iterate
    scene.export_all("out", "chair")       # .glb .gltf .fbx
"""
__version__ = "0.1.0"

from . import math3d, noise, sdf
from .math3d import bezier, catmull_rom, quat_from_euler, quat_from_axis_angle
from .mesh import Mesh
from .primitives import (box, capsule, cone, cube, cylinder, disc, extrude_shape, grid, ico_sphere, lathe, loft,
                         plane, rounded_box, shapes, sphere, sweep, torus, tube, uv_sphere)
from .material import Material, Texture, presets, textures
from .scene import Camera, Light, Node, Scene
from .animation import Animation, bounce, orbit, spin
from .armature import Skeleton, skinned_node
from . import sculpt, physics
from .sculpt import Brush
from .terrain import flatten, height_at, scatter, terrain, water_plane
from .importer import load
from .render import render, render_views, render_frames
from .viewer import write_viewer
from .project import save
from . import generators
from .generators import architecture, character, nature, props, vehicles

__all__ = [
    "Mesh", "Scene", "Node", "Light", "Camera", "Material", "Texture", "presets", "textures", "Animation",
    "Skeleton", "skinned_node", "sdf", "noise", "math3d", "sculpt", "Brush", "box", "cube", "rounded_box",
    "plane", "grid", "disc", "sphere", "uv_sphere", "ico_sphere", "cylinder", "cone", "capsule", "torus",
    "lathe", "sweep", "tube", "loft", "extrude_shape", "shapes", "terrain", "scatter", "water_plane", "load",
    "flatten", "height_at",
    "render", "render_views", "render_frames", "write_viewer", "spin", "bounce", "orbit", "bezier",
    "catmull_rom", "quat_from_euler", "quat_from_axis_angle", "save", "generators", "architecture", "character",
    "nature", "props", "vehicles", "physics",
]
