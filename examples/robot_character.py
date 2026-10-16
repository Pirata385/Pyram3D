"""Custom character from SDF parts + a hand-built skeleton + keyframed animation.

Shows how to rig something that is not covered by the generators: a small
two-legged robot with an antenna, rigid (mechanical) skin weights and a
hop animation.
Run: python examples/robot_character.py  ->  examples/output/robot/
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pyram3d as p3
from pyram3d import sdf

OUT = Path(__file__).resolve().parent / "output" / "robot" / "robot"
WHITE, GREY, ACCENT = "#e9edf0", "#5c636b", "#ff7a1a"

# --- shape (one SDF, colored per part) -------------------------------------------------
# every part names the bone it follows -> exact skinning for mechanical characters
body = sdf.box((0, 0.62, 0), (0.42, 0.36, 0.3), radius=0.08, color=WHITE, bone="Body")
visor = sdf.box((0, 0.68, 0.14), (0.3, 0.12, 0.05), radius=0.03, color="#1d2733", bone="Body")
head = body.smooth_union(visor, 0.01)
antenna = sdf.capsule((0, 0.8, 0), (0, 0.95, 0), 0.012, color=GREY).union(
    sdf.sphere((0, 0.97, 0), 0.03, color=ACCENT)).paint(bone="Antenna")
parts = [head, antenna]
for side, sx in (("Left", 1), ("Right", -1)):
    hip = sdf.sphere((sx * 0.11, 0.42, 0), 0.06, color=GREY, bone="Body")
    leg = sdf.capsule((sx * 0.11, 0.42, 0), (sx * 0.11, 0.1, 0), 0.04, color=GREY, bone=f"{side}Leg")
    foot = sdf.box((sx * 0.11, 0.04, 0.04), (0.12, 0.08, 0.2), radius=0.03, color=ACCENT, bone=f"{side}Leg")
    arm = sdf.capsule((sx * 0.25, 0.66, 0), (sx * 0.3, 0.45, 0.02), 0.035, color=WHITE, bone=f"{side}Arm")
    parts += [hip, leg, foot, arm]
robot = parts[0]
for p in parts[1:]:
    robot = robot.smooth_union(p, 0.01)
mesh = robot.to_mesh(140, name="Robot")
mesh.set_material(p3.Material("RobotPaint", "#ffffff", metallic=0.2, roughness=0.35))

# --- skeleton ------------------------------------------------------------------------
sk = p3.Skeleton("RobotRig")
root = sk.add_bone("Body", (0, 0.45, 0), (0, 0.8, 0))
sk.add_bone("Antenna", (0, 0.8, 0), (0, 0.97, 0), root)
for side, sx in (("Left", 1), ("Right", -1)):
    sk.add_bone(f"{side}Leg", (sx * 0.11, 0.42, 0), (sx * 0.11, 0.1, 0), root)
    sk.add_bone(f"{side}Arm", (sx * 0.25, 0.66, 0), (sx * 0.3, 0.45, 0.02), root)
sk.bind(mesh, rigid=True)     # mechanical parts: each vertex follows one bone

scene = p3.Scene("Robot")
scene.add(sk)
scene.add(p3.skinned_node(mesh, sk, "RobotBody"))
scene.add_default_lights()

# --- animation: a hop with squash and arm swing ----------------------------------------
hop = scene.animation("Hop")
rest = sk.bone("Body").extras["rest_translation"]
hop.keys(sk.bone("Body"), "position", {0: rest, 0.2: rest + (0, -0.06, 0), 0.5: rest + (0, 0.25, 0),
                                       0.8: rest + (0, -0.05, 0), 1.0: rest}, "SMOOTH")
for side, sgn in (("Left", 1), ("Right", -1)):
    hop.keys(sk.bone(f"{side}Arm"), "rotation", {0: (0, 0, 0), 0.5: (0, 0, sgn * 60), 1.0: (0, 0, 0)}, "SMOOTH")
    hop.keys(sk.bone(f"{side}Leg"), "rotation", {0: (0, 0, 0), 0.2: (-15, 0, 0), 0.5: (10, 0, 0), 1.0: (0, 0, 0)},
             "SMOOTH")
hop.keys(sk.bone("Antenna"), "rotation", {0: (0, 0, 0), 0.5: (-20, 0, 0), 0.7: (15, 0, 0), 1.0: (0, 0, 0)}, "SMOOTH")

if __name__ == "__main__":
    print(scene.describe())
    out = p3.save(scene, OUT, size=700)
    print({k: str(v.name) for k, v in out.items()})
