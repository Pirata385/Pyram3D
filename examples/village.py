"""Showcase: a small island village with a rigged character, an animated dog,
procedural houses, trees, rocks, a car and a campfire.

Run:  python examples/village.py
Out:  examples/output/village/ (glb, gltf, fbx, previews, viewer.html)
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

import pyram3d as p3

OUT = Path(__file__).resolve().parent / "output" / "village" / "village"

scene = p3.Scene("Island Village")

# --- world -------------------------------------------------------------------
ground = p3.terrain(size=60, resolution=110, height=7, scale=0.05, island=0.95, ridged=0.25, seed=11)
plaza_h = p3.flatten(ground, (0, 0), radius=9, height=1.6)
scene.add(ground)
scene.add(p3.water_plane(90, level=0.55))

# --- buildings ---------------------------------------------------------------
house = p3.architecture.house(width=7, depth=6, wall="#efe2c8", roof_color="#9c3b2a")
scene.add(house, position=(-1.5, plaza_h, -3.5))
shed = p3.architecture.house(width=4.5, depth=4, wall="#c9d6df", roof_color="#3d5a73", chimney=False)
scene.add(shed, position=(6.0, plaza_h, -1.5), rotation=(0, -70, 0))
scene.add(p3.architecture.fence(length=8), position=(-1.5, plaza_h, 2.8))
scene.add(p3.architecture.street_lamp(3.5), position=(3.2, plaza_h, 1.8), rotation=(0, 180, 0))

# --- props -------------------------------------------------------------------
scene.add(p3.props.barrel(0.9), position=(2.4, plaza_h, -1.2))
scene.add(p3.props.crate(0.7), position=(3.1, plaza_h, -0.6), rotation=(0, 25, 0))
car = p3.vehicles.car("#2f7d6d", style="hatchback")
car.translation = np.array([-6.5, plaza_h, 1.0])
car.set_rotation((0, 100, 0))
scene.add(car)

# campfire with looping particles
logs = p3.Mesh.join([p3.cylinder(0.06, 0.7).rotate((90, a, 0)).translate((0, 0.06, 0)) for a in (0, 60, 120)],
                    name="Logs").set_material(p3.presets.bark())
stones = p3.Mesh.join([p3.nature.rock(0.22, seed=i).translate((math.cos(a) * 0.5, 0, math.sin(a) * 0.5))
                       for i, a in enumerate(np.linspace(0, 2 * math.pi, 9)[:-1])], name="FireStones")
fire_pos = np.array([1.5, plaza_h, 4.8])
scene.add(logs, position=fire_pos)
scene.add(stones, position=fire_pos)

# --- vegetation (one mesh per species, instanced many times) --------------------
oak = p3.nature.tree(height=4.5, seed=3)
pine = p3.nature.pine(height=5.5, seed=4)
bush = p3.nature.bush(1.0, berries="#c0392b")
rock = p3.nature.rock(1.0, seed=8)
away_from_plaza = lambda p: np.hypot(p[:, 0], p[:, 2]) > 10.5  # noqa: E731
for i, s in enumerate(p3.scatter(ground, 26, seed=5, max_slope=28, min_height=1.0, min_distance=3.0,
                                 mask=away_from_plaza)):
    scene.add(oak if i % 3 else pine, name=f"Tree{i}", position=s["position"], rotation=s["rotation"],
              scale=s["scale"])
for i, s in enumerate(p3.scatter(ground, 14, seed=6, max_slope=35, min_height=0.8, min_distance=2.0,
                                 mask=away_from_plaza)):
    scene.add(bush if i % 2 else rock, name=f"Detail{i}", position=s["position"], rotation=s["rotation"],
              scale=s["scale"])

# --- characters ------------------------------------------------------------
hero = p3.character.humanoid(height=1.75, shirt="#d35400", pants="#34495e", hair="#2c1b10")
hero.add_to(scene, position=(0.2, plaza_h, 3.2))
hero.skeleton.root.set_rotation((0, 20, 0))  # face the campfire and the camera
p3.character.wave(scene.animation("Wave"), hero.skeleton, duration=2.0)
p3.character.idle(scene.animation("Idle"), hero.skeleton)

dog = p3.character.quadruped(length=0.9, height=0.5, fur="#b5753d")
dog.add_to(scene, position=(fire_pos[0] + 2.2, fire_pos[1], fire_pos[2]))
walk = scene.animation("DogWalk")
p3.character.quadruped_walk(walk, dog.skeleton, duration=0.8)
# make the dog trot around the campfire: loop the 0.8 s gait over a 8 s circle
gait = [(c.node, c.path, list(c.times), list(c.values)) for c in walk.channels]
for k in range(1, 10):
    for node, path, times, values in gait:
        for t, v in zip(times, values):
            walk.key(node, path, t + 0.8 * k, v, "SMOOTH")
root = dog.skeleton.root
for k in range(33):
    t = 8.0 * k / 32
    a = 2 * math.pi * k / 32
    pos = fire_pos + np.array([math.cos(a) * 2.2, 0, math.sin(a) * 2.2])
    walk.key(root, "position", t, pos)
    walk.key(root, "rotation", t, (0, -math.degrees(a), 0))  # face along the circle (tangent)

fx = scene.animation("Campfire")
p3.physics.particles(scene, fx, "fire", position=fire_pos + (0, 0.1, 0), count=40)
p3.physics.particles(scene, fx, "sparks", position=fire_pos + (0, 0.1, 0), count=20)

# --- light & camera ----------------------------------------------------------
scene.add_light("directional", position=(20, 30, 15), intensity=3.2, color="#fff4e0", name="Sun")
scene.add_light("directional", position=(-15, 10, -10), intensity=0.8, color="#bcd4ff", cast_shadows=False,
                name="Sky")
scene.add_camera(position=(7, 5.5, 14), target=(0.5, 2.0, 2.5), name="MainCamera")

if __name__ == "__main__":
    print(scene.describe().splitlines()[-8:])
    out = p3.save(scene, OUT, views=False, frames=False, size=1000, preview=False)
    p3.render(scene, OUT.parent / "village_preview.png", 1000, 750, view=(0.6, 0.55, 1.0), target=(0, 1, 0),
              distance=55)
    p3.render(scene, OUT.parent / "village_closeup.png", 1000, 650, camera=scene.find("MainCamera"), time=1.0)
    for k, v in out.items():
        print(k, v.relative_to(Path.cwd()) if v.is_relative_to(Path.cwd()) else v)
